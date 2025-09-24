# ge_listings/base.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .common import (
    HEADERS, REQ_TIMEOUT, UA, PW_TIMEOUT_MS,
    ensure_dir, uniq_keep_order, uniq_keep_order_any, HTTP2_AVAILABLE
)

# PIL optional
try:
    from PIL import Image, ImageOps, ImageFile  # type: ignore
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False


def _convert_webp_to_jpeg_bytes(content: bytes) -> Optional[bytes]:
    if not _PIL_OK:
        return None
    try:
        im = Image.open(BytesIO(content))
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            rgb = bg
        else:
            rgb = im.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=92, optimize=True)
        return buf.getvalue()
    except Exception:
        return None


async def async_download(urls: List[str], dest_dir: Path, client: httpx.AsyncClient) -> List[str]:
    saved: List[str] = []
    for idx, u in enumerate(urls, 1):
        try:
            variants = [u]
            if "_Thumb." in u:
                variants = [u.replace("_Thumb.", "_Large."), u.replace("_Thumb.", "."), u]
            content = None
            final_url = None
            for cand in variants:
                r = await client.get(cand, timeout=REQ_TIMEOUT)
                if r.status_code == 200 and (r.headers.get("content-type", "").startswith("image")):
                    content = r.content
                    final_url = cand
                    break
            if not content:
                continue
            ext = os.path.splitext(urlparse(final_url).path)[1].lower() or ".jpg"
            if ext == ".webp":
                jbytes = _convert_webp_to_jpeg_bytes(content)
                if jbytes:
                    ext = ".jpg"
                    content = jbytes
            fpath = dest_dir / f"{idx:03d}{ext}"
            fpath.write_bytes(content)
            saved.append(fpath.name)
        except Exception:
            continue
    return saved


class BaseExtractor:
    def __init__(self, url: str):
        self.url = url
        self.host = urlparse(url).netloc
        self._last_render_log: Optional[dict] = None  # для детальной диагностики

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.8, min=1, max=6),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def fetch(self) -> str:
        with httpx.Client(
            http2=(HTTP2_AVAILABLE and os.getenv("GE_HTTP2", "1") != "0"),
            headers=HEADERS,
            follow_redirects=True,
            timeout=REQ_TIMEOUT,
        ) as c:
            r = c.get(self.url)
            r.raise_for_status()
            return r.text

    async def fetch_images(self, urls: List[str], dest_dir: Path) -> List[str]:
        if not urls:
            return []
        ensure_dir(dest_dir)
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=REQ_TIMEOUT) as ac:
            return await async_download(urls, dest_dir, ac)

    # ---------- Playwright helpers ----------

    def _render_with_capture(
        self, after_action: Optional[Callable[[Any], Optional[dict]]] = None
    ) -> Tuple[str, List[str], List[str], List[dict]]:
        """
        Рендерит страницу Playwright-ом, перехватывает JSON (в том числе ответ /phone/show).
        Возвращает: (html, image_urls_from_dom, image_urls_from_json, json_blobs)
        Детальный лог — в self._last_render_log (dict).
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except Exception:
            return "", [], [], []

        headful = os.getenv("HEADFUL", "0") == "1"
        html = ""
        dom_imgs: List[str] = []
        json_blobs: List[dict] = []
        phone_statuses: List[int] = []
        api_phone_number: Optional[str] = None
        ls_after_value: Optional[List[str]] = None

        def _walk_collect_str(o, out: List[str]):
            if isinstance(o, str):
                out.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    _walk_collect_str(v, out)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    _walk_collect_str(v, out)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headful)
            context = browser.new_context(
                user_agent=UA,
                locale="ru-RU",
                ignore_https_errors=True,
                service_workers="block",
            )

            noisy_hosts = {
                "googletagmanager.com",
                "google-analytics.com",
                "doubleclick.net",
                "facebook.net",
                "clarity.ms",
                "hotjar.com",
                "yandex.ru",
                "mc.yandex.ru",
            }

            def _should_block(u: str) -> bool:
                """
                Блокируем ТОЛЬКО аналитические/рекламные домены
                (точное совпадение или поддомен).
                """
                try:
                    from urllib.parse import urlparse as _u
                    h = _u(u).netloc.lower()
                    return any(h == x or h.endswith("." + x) for x in noisy_hosts)
                except Exception:
                    return False

            def _block(route):
                req = route.request
                if req.resource_type in ("image", "media", "font", "stylesheet") or _should_block(req.url):
                    return route.abort()
                return route.continue_()

            context.route("**/*", _block)

            def on_response(res):
                try:
                    url_l = (getattr(res, "url", "") or "")
                    ct = (res.headers or {}).get("content-type", "").lower()
                    is_jsonish = "json" in ct
                    is_text_plain_json = "text/plain" in ct and "/v1/statements/phone/show" in url_l
                    if "/v1/statements/phone/show" in url_l:
                        phone_statuses.append(res.status)
                    if not (is_jsonish or is_text_plain_json):
                        return
                    txt = res.text()
                    if not txt or len(txt) > 2_000_000:
                        return
                    s = txt.lstrip()
                    if not (s.startswith("{") or s.startswith("[")):
                        if not is_text_plain_json:
                            return
                    data = json.loads(txt)
                    # вытащим номер прямо отсюда (если он есть)
                    try:
                        d = data.get("data") if isinstance(data, dict) else None
                        if isinstance(d, dict) and isinstance(d.get("data"), dict):
                            d = d["data"]
                        cand = None
                        if isinstance(d, dict):
                            cand = d.get("phone_number") or d.get("phoneNumber") or d.get("phone")
                        if not cand and isinstance(data, dict):
                            cand = data.get("phone_number") or data.get("phoneNumber") or data.get("phone")
                        if cand:
                            api_phone_number = str(cand)
                    except Exception:
                        pass
                    json_blobs.append(data)
                except Exception:
                    pass

            context.on("response", on_response)

            page = context.new_page()
            page.set_default_timeout(PW_TIMEOUT_MS)
            page.goto(self.url, wait_until="domcontentloaded", timeout=PW_TIMEOUT_MS)

            try:
                page.wait_for_selector("script#__NEXT_DATA__, img, [class*=gallery]", timeout=min(PW_TIMEOUT_MS, 30000))
            except PWTimeout:
                pass

            action_log: Optional[dict] = None
            if after_action:
                try:
                    action_log = after_action(page) or {}
                    # подхватим содержимое LS после клика (если сайт туда положил номер)
                    try:
                        ls_val = page.evaluate("() => window.localStorage.getItem('phoneNumbers')")
                        if ls_val:
                            try:
                                arr = json.loads(ls_val)
                                if isinstance(arr, list) and arr:
                                    json_blobs.append({"data": {"data": {"phone_number": arr[0]}}})
                                    ls_after_value = arr
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    action_log = {"exception": True}

            # HTML
            html = page.content()

            # картинки из DOM
            try:
                imgs = page.evaluate(
                    """
                    () => {
                      const xs = new Set(); const push = u => { if (u) xs.add(u); };
                      for (const el of Array.from(document.images)) {
                        const src = el.currentSrc || el.src || "";
                        if ((el.naturalWidth||0) >= 512 || /(large|xlarge|big|1024|1280|1920)/i.test(src)) push(src);
                      }
                      for (const s of Array.from(document.querySelectorAll('source[srcset]'))) {
                        const list = s.getAttribute('srcset') || "";
                        for (const part of list.split(',')) {
                          const u = part.trim().split(' ')[0];
                          if (u) push(u);
                        }
                      }
                      return Array.from(xs);
                    }
                """
                )
                dom_imgs = [u for u in imgs if isinstance(u, str)]
            except Exception:
                pass
            finally:
                context.close()
                browser.close()

        dom_imgs = [
            u
            for u in dom_imgs
            if isinstance(u, str)
            and not u.startswith("data:")
            and not re.search(r"(?:_thumb|_blur|google_map)", u, flags=re.I)
        ]

        json_img_urls: List[str] = []
        for blob in json_blobs:
            strs: List[str] = []
            _walk_collect_str(blob, strs)
            for s in strs:
                if re.search(r"\.(?:jpe?g|png|webp)(?:\?.*)?$", s, flags=re.I) and not re.search(
                    r"(?:_thumb|_blur|google_map)", s, flags=re.I
                ):
                    json_img_urls.append(s)

        # сохраним детальный лог
        self._last_render_log = {
            "headful": headful,
            "timeout_ms": PW_TIMEOUT_MS,
            "json_count": len(json_blobs),
            "phone_show_statuses": phone_statuses,
            "api_phone_number": api_phone_number,
            "ls_after_value": ls_after_value,
            "after_action": action_log or {},
        }

        return html, uniq_keep_order(dom_imgs), uniq_keep_order(json_img_urls), uniq_keep_order_any(json_blobs)

    def via_playwright(self) -> Optional[str]:
        html, _, _, _ = self._render_with_capture()
        return html or None

    def extract(self):
        raise NotImplementedError
