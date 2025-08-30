#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape Georgian real-estate listings (myhome.ge, home.ss.ge).
Saves structured JSON + downloads photos (+ converts WebP→JPEG) and writes RU summary.

- SS.ge: parses HTML/JSON-LD/OG (RU/KA/EN), robust address, floors, attributes.
- myhome.ge: Playwright render + capture JSON/DOM gallery; resolves /_next/image?url=...
- WebP images auto-convert to JPEG(92) with proper alpha handling.

Usage:
  pip install httpx[http2] beautifulsoup4 lxml tenacity pydantic playwright pillow
  python -m playwright install chromium

  python scrape_ge_listings.py "<URL>" [-o OUTDIR] [--no-photos] [--headful]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
from urllib.parse import urlparse, urljoin, parse_qs, unquote, urlunparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

# image conversion
try:
    from PIL import Image, ImageOps, ImageFile  # type: ignore
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    PIL_OK = True
except Exception:
    PIL_OK = False

# ---------- Utils ----------

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# --- ТАЙМАУТЫ и ретраи настраиваемые через ENV ---
# GE_HTTP_TIMEOUT: общий таймаут запроса (сек), GE_HTTP_CONNECT_TIMEOUT: коннект (сек)
# GE_HTTP_RETRIES: число попыток, GE_PW_TIMEOUT_MS: таймауты Playwright (мс)
HTTP_TIMEOUT_S = float(os.getenv("GE_HTTP_TIMEOUT", "45.0"))
HTTP_CONNECT_TIMEOUT_S = float(os.getenv("GE_HTTP_CONNECT_TIMEOUT", "30.0"))
HTTP_RETRIES = int(os.getenv("GE_HTTP_RETRIES", "5"))
PW_TIMEOUT_MS = int(os.getenv("GE_PW_TIMEOUT_MS", "60000"))

REQ_TIMEOUT = httpx.Timeout(HTTP_TIMEOUT_S, connect=HTTP_CONNECT_TIMEOUT_S)
HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "ru-RU,ru;q=0.95,ka;q=0.6,en-US;q=0.5,en;q=0.4",
}
MEASURE_M2_RU_KA_EN = r"(?:м²|m²|кв\.?\s?м|მ²|კვ\.?\s?მ)"
SKIP_IMG_HOST_SUBSTR = ("adocean.pl",)  # баннерная сеть, игнорим

class ScrapeError(Exception):
    pass

def safe_slug(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[\s/\\:;,\|\[\]\(\)\{\}\<\>\?\"'\!@#\$%\^&\*\=]+", "_", (s or "").strip())
    return (s or "item")[:max_len].strip("_") or "item"

def extract_first_number(text: str) -> Optional[int]:
    m = re.search(r"(\d[\d\s,\.]*)", text or "")
    if not m:
        return None
    raw = m.group(1).replace(" ", "").replace(",", "")
    try:
        return int(float(raw))
    except Exception:
        return None

def normalize_price(text: str) -> Tuple[Optional[int], Optional[str]]:
    if not text:
        return None, None
    cur = None
    t = text.upper()
    if "₾" in text or "GEL" in t:
        cur = "GEL"
    elif "$" in text or "USD" in t:
        cur = "USD"
    elif "€" in text or "EUR" in t:
        cur = "EUR"
    amount = extract_first_number(text)
    return amount, cur

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def textify(el) -> str:
    if not el:
        return ""
    # даём чуть больше воздуха между блоками
    return re.sub(r"\s+", " ", el.get_text(separator=" • ", strip=True)).strip()

def meta(soup: BeautifulSoup, key: str) -> Optional[str]:
    el = soup.find("meta", {"property": key}) or soup.find("meta", {"name": key})
    return (el.get("content") or None) if el else None

def pick_json_ld(html: str) -> List[Dict[str, Any]]:
    out = []
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or tag.text)
            if isinstance(data, dict):
                out.append(data)
            elif isinstance(data, list):
                out.extend([x for x in data if isinstance(x, dict)])
        except Exception:
            continue
    return out

def absolutize(src: str, base: str) -> str:
    try:
        return urljoin(base, src)
    except Exception:
        return src

def uniq_keep_order(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def uniq_keep_order_any(seq: Iterable[Any]) -> List[Any]:
    seen = set(); out = []
    for x in seq:
        key = json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else x
        if key not in seen:
            seen.add(key); out.append(x)
    return out

def is_good_image_url(u: str) -> bool:
    if not re.match(r"^https?://", u, re.I):
        return False
    host = urlparse(u).netloc.lower()
    if any(bad in host for bad in SKIP_IMG_HOST_SUBSTR):
        return False
    return bool(re.search(r"\.(?:jpe?g|png|webp)(?:\?.*)?$", u, flags=re.I))

# ---------- Phones ----------
_RE_PHONE = re.compile(
    r"(?:\+?995[\s\-]?)?(?:0[\s\-]?)?(?:\(?5\d{2}\)?[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}|"
    r"\(?3\d\)?[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2,3})"
)

def _normalize_ge_phone(s: str) -> Optional[str]:
    if not s: return None
    digits = re.sub(r"\D+", "", s)
    if not digits:
        return None
    # уже в международном?
    if digits.startswith("995"):
        norm = digits
    else:
        # mobile 5xx... (9 цифр) или стационарный 3x...
        if digits.startswith("5") and len(digits) == 9:
            norm = "995" + digits
        elif digits.startswith("0") and len(digits) >= 10:
            # отрезаем начальный 0 (например 032...)
            norm = "995" + digits[1:]
        elif (digits.startswith("32") or digits.startswith("3")) and len(digits) >= 9:
            norm = "995" + digits
        else:
            # не грузинский формат — игнорируем
            return None
    return "+" + norm

def _phones_from_text(text: str) -> List[str]:
    if not text: return []
    out = []
    for m in _RE_PHONE.findall(text):
        p = _normalize_ge_phone(m)
        if p: out.append(p)
    return uniq_keep_order(out)

def _flatten_strings(o: Any, limit: int = 200) -> List[str]:
    """Собираем короткие строковые значения из произвольного JSON."""
    out: List[str] = []
    def walk(x: Any):
        if isinstance(x, str):
            s = x.strip()
            if 3 <= len(s) <= limit:
                out.append(s)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
    walk(o)
    return out

# ---------- Image conversion ----------

def _convert_webp_to_jpeg_bytes(content: bytes) -> Optional[bytes]:
    if not PIL_OK:
        return None
    try:
        im = Image.open(BytesIO(content))
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            # JPEG без альфы → кладём на белый фон
            bg = Image.new("RGB", im.size, (255, 255, 255))
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            rgb = bg
        else:
            rgb = im.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=92, optimize=True)  # JPEG качество (см. доку)
        return buf.getvalue()
    except Exception:
        return None

async def async_download(urls: List[str], dest_dir: Path, client: httpx.AsyncClient) -> List[str]:
    saved = []
    for idx, u in enumerate(urls, 1):
        try:
            variants = [u]
            if "_Thumb." in u:
                variants = [
                    u.replace("_Thumb.", "_Large."),
                    u.replace("_Thumb.", "."),
                    u,
                ]
            content = None
            final_url = None
            for cand in variants:
                r = await client.get(cand, timeout=REQ_TIMEOUT)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    content = r.content
                    final_url = cand
                    break
            if not content:
                continue

            ext = os.path.splitext(urlparse(final_url).path)[1].lower() or ".jpg"
            # Если .webp — сразу конвертируем в JPEG (лучше на лету)
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

# ---------- Data Models ----------

class Price(BaseModel):
    amount: Optional[int] = None
    currency: Optional[str] = None

class Listing(BaseModel):
    url: str
    source: str
    listing_id: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    address_line: Optional[str] = None
    price: Price = Field(default_factory=Price)
    area_m2: Optional[int] = None
    land_area_m2: Optional[int] = None
    rooms: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floor: Optional[int] = None
    floors_total: Optional[int] = None
    description: Optional[str] = None
    attributes: List[str] = Field(default_factory=list)
    photos: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    summary_ru: Optional[str] = None
    raw_meta: Dict[str, Any] = Field(default_factory=dict)

# ---------- Base Extractor ----------

class BaseExtractor:
    def __init__(self, url: str):
        self.url = url
        self.host = urlparse(url).netloc

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.8, min=1, max=6),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def fetch(self) -> str:
        with httpx.Client(http2=True, headers=HEADERS, follow_redirects=True, timeout=REQ_TIMEOUT) as c:
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

    def _render_with_capture(self) -> Tuple[str, List[str], List[str], List[dict]]:
        """
        Render page via Playwright.
        Returns: (html, image_urls_from_dom, image_urls_from_json, json_blobs_from_network)
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except Exception:
            return "", [], [], []

        headful = os.getenv("HEADFUL", "0") == "1"
        html = ""
        dom_imgs: List[str] = []
        json_blobs: List[dict] = []

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
            # Блокируем сервис-воркеры (они мешают networkidle) и 3rd-party трекеры,
            # чтобы быстрее наступала «тишина».
            context = browser.new_context(
                user_agent=UA,
                locale="ru-RU",
                ignore_https_errors=True,
                service_workers="block",   # <-- ключевая строка
            )

            # отрезаем шумные внешние домены (не изображения и не данные лота)
            noisy_hosts = {
                "googletagmanager.com","google-analytics.com","doubleclick.net",
                "facebook.net","clarity.ms","hotjar.com","yandex.ru","mc.yandex.ru"
            }
            def _should_block(u: str) -> bool:
                try:
                    from urllib.parse import urlparse
                    h = urlparse(u).netloc.lower()
                    return any(h.endswith(x) or x in h for x in noisy_hosts)
                except Exception:
                    return False
            context.route("**/*", lambda route: route.abort() if _should_block(route.request.url) else route.continue_())

            # собираем JSON-ответы
            def on_response(res):
                ct = (res.headers or {}).get("content-type", "")
                if "application/json" in ct.lower():
                    try:
                        txt = res.text()
                        if txt and len(txt) < 2_000_000:
                            data = json.loads(txt)
                            json_blobs.append(data)
                    except Exception:
                        pass
            context.on("response", on_response)

            page = context.new_page()
            page.set_default_timeout(PW_TIMEOUT_MS)  # регулируется через ENV

            # 1) Навигация — ждём обычный "load" (а не только DOMContentLoaded)
            page.goto(self.url, wait_until="load", timeout=PW_TIMEOUT_MS)

            # 2) Ждём «признаки жизни» страницы (любой из селекторов), но без фатала
            try:
                page.wait_for_selector("script#__NEXT_DATA__, img, [class*=gallery]", timeout=min(PW_TIMEOUT_MS, 30000))
            except PWTimeout:
                pass

            # 3) Пытаемся дождаться networkidle, но это *не критично* — не упадём, если не случится
            try:
                page.wait_for_load_state("networkidle", timeout=min(PW_TIMEOUT_MS, 55000))
            except PWTimeout:
                pass  # на SPA с WS/поллингом networkidle может не наступить — это ок

            html = page.content()

            # далее — сбор изображений из DOM (img/srcset)
            try:
                imgs = page.evaluate("""
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
                """)
                dom_imgs = [u for u in imgs if isinstance(u, str)]
            except Exception:
                pass

            finally:
                context.close()
                browser.close()

        # фильтр DOM картинок
        dom_imgs = [u for u in dom_imgs if isinstance(u, str) and not u.startswith("data:") and not re.search(r'(?:_thumb|_blur|google_map)', u)]

        # ссылки из JSON
        json_img_urls: List[str] = []
        for blob in json_blobs:
            strs: List[str] = []
            _walk_collect_str(blob, strs)
            for s in strs:
                if re.search(r'\.(?:jpe?g|png|webp)(?:\?.*)?$', s, flags=re.I) and not re.search(r'(?:_thumb|_blur|google_map)', s, flags=re.I):
                    json_img_urls.append(s)

        return html, uniq_keep_order(dom_imgs), uniq_keep_order(json_img_urls), uniq_keep_order_any(json_blobs)

    def via_playwright(self) -> Optional[str]:
        html, _, _, _ = self._render_with_capture()
        return html or None

    def extract(self) -> 'Listing':
        raise NotImplementedError

# ---------- SS.GE Extractor ----------

def _pick_total_price(text: str) -> Tuple[Optional[int], Optional[str]]:
    pairs = []
    for m in re.finditer(r"(\d[\d\s,\.]{1,9})\s*(₾|\$|€|GEL|USD|EUR)", text, flags=re.I):
        left = text[max(0, m.start()-12):m.start()].lower()
        # отбрасываем 'за м²'
        if any(x in left for x in ("m²", "m2", "м²", "кв. м", "кв.м")):
            continue
        amt = extract_first_number(m.group(1))
        cur = normalize_price(m.group(2))[1]
        if amt and cur:
            pairs.append((amt, cur))
    if not pairs:
        return None, None
    # берём максимальную сумму (почти всегда total выше, чем $/m²)
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[0]


class SSGeExtractor(BaseExtractor):
    """Extracts from home.ss.ge listing pages (RU/KA/EN)."""

    RU_FEATURES = [
        "кондиционер","балкон","подвал","кабельное телевидение","питьевая вода","лифт","холодильник","мебель",
        "гараж","стекло-пакет","цент. отопление","горячая вода","интернет","железная дверь",
        "природный газ","сигнализация","хранилище","телефон","телевизор","стиральная машина","бассейн"
    ]
    KA_FEATURES = [
        "კონდიციონერი","აივანი","სარდაფი","საკაბელო ტელევიზია","სასმელი წყალი","ლიფტი","მაცივარი","ავეჯი",
        "გარაჟი","მინა-პაკეტი","ცენტ. გათბობა","ცხელი წყალი","ინტერნეტი","რკინის კარი","ბუნებრივი აირი",
        "სიგნალიზაცია","სათავსო","ტელეფონი","ტელევიზორი","სარეცხი მანქანა","აუზი"
    ]
    EN_FEATURES = [
        "air conditioner","balcony","basement","cable tv","drinking water","elevator","fridge","furniture",
        "garage","double glazed","central heating","hot water","internet","metal door","natural gas",
        "alarm system","storage","telephone","tv","washing machine","pool"
    ]

    PHONE_BLACKLIST = {"+995322121661"}

    def _jsonld_fill(self, jlds: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        photos = []
        for node in jlds:
            if not isinstance(node, dict):
                continue
            out["title"] = out.get("title") or node.get("name") or node.get("headline")
            offer = node.get("offers") if isinstance(node.get("offers"), dict) else None
            if offer:
                amt = extract_first_number(str(offer.get("price") or ""))
                cur = offer.get("priceCurrency") or None
                if amt and not out.get("price_amount"):
                    out["price_amount"] = amt
                if cur and not out.get("price_currency"):
                    out["price_currency"] = cur
            adr = node.get("address")
            if adr and not out.get("location"):
                if isinstance(adr, dict):
                    parts = [adr.get(x) for x in ("streetAddress", "addressLocality", "addressRegion")]
                    out["location"] = " ".join([p for p in parts if p])
                elif isinstance(adr, str):
                    out["location"] = adr
            imgs = node.get("image")
            if isinstance(imgs, list):
                photos.extend([str(x) for x in imgs])
            elif isinstance(imgs, str):
                photos.append(imgs)
            out.setdefault("rooms", node.get("numberOfRooms"))
            area = node.get("floorSize") or node.get("area")
            if isinstance(area, dict):
                out.setdefault("area_m2", extract_first_number(str(area.get("value") or "")))
        if photos:
            out["photos"] = photos
        return out

    def _extract_address(self, soup: BeautifulSoup, text_all: str) -> Tuple[Optional[str], Optional[str]]:
        # direct street line
        addr = None
        m = re.search(r"(ул\.?|улица|пр(?:осп\.?|оспект|\.?)|пр-т|пер\.?|переулок|ш\.?|шоссе|пр\.)\s*[^\n,]{2,80}?\d+[А-Яа-яЁё0-9\/\-]*", text_all, flags=re.I)
        if m:
            addr = m.group(0).strip()
        # chips / breadcrumbs for city & district
        city = None; district = None
        chips = [textify(a) for a in soup.find_all(["a","span"]) if a and a.get("href") and len(textify(a)) <= 40]
        for t in chips:
            if re.fullmatch(r"(Тбилиси|თბილისი|Tbilisi)", t, flags=re.I):
                city = "Тбилиси"
            if re.search(r"(Сабуртало|საბურთალო|Saburtalo)", t, flags=re.I):
                district = "Сабуртало"
        parts = [p for p in [city, district, addr] if p]
        return (", ".join(parts) if parts else None, addr)

    def _parse_floors(self, text_all: str) -> Tuple[Optional[int], Optional[int]]:
        # этаж/этажность
        m = re.search(r"(?:Этаж(?:ность)?|სართული)[\s•:–-]*?(\d+)\s*/\s*(\d+)", text_all, flags=re.I)
        if m:
            return int(m.group(1)), int(m.group(2))
        f = None; ft = None
        m = re.search(r"(?:Этаж|სართული)[\s•:–-]*?(\d+)", text_all, flags=re.I)
        if m: f = int(m.group(1))
        m = re.search(r"(?:Этажей|Этажность|სართულიანი)[\s•:–-]*?(\d+)", text_all, flags=re.I)
        if m: ft = int(m.group(1))
        return f, ft

    def extract(self) -> Listing:
        html = ""
        dom_imgs_cap: List[str] = []
        json_imgs_cap: List[str] = []
        json_blobs_cap: List[dict] = []
        # Сначала пробуем обычный HTTP, затем рендер для захвата JSON/фото
        try:
            html = self.fetch()
        except Exception:
            pass
        html2, dom_imgs_cap, json_imgs_cap, json_blobs_cap = self._render_with_capture()
        if html2:
            html = html2
        if not html:
            raise ScrapeError("Failed to load page")

        soup = BeautifulSoup(html, "lxml")
        main = soup.find("main") or soup
        text_all = textify(main)

        jlds = pick_json_ld(html)
        jl = self._jsonld_fill(jlds) if jlds else {}

        title = jl.get("title") or textify(soup.find("h1")) or meta(soup, "og:title")
        listing_id = None
        m = re.search(r"\bID\s*[-–]?\s*(\d+)\b", text_all, flags=re.I)
        if m:
            listing_id = m.group(1)

        amount = jl.get("price_amount")
        currency = jl.get("price_currency")
        if not (amount and currency):
            # попробуем вытащить цену из встроенного JSON-блока "price":{...}
            m_price = re.search(r'"price":(\{[^{}]+\})', html)
            price_obj = None
            if m_price:
                try:
                    price_obj = json.loads(m_price.group(1))
                except Exception:
                    price_obj = None
            if price_obj:
                cur_map = {1: "GEL", 2: "USD", 3: "EUR"}
                cur_type = price_obj.get("currencyType")
                currency = cur_map.get(cur_type)
                if currency == "USD":
                    amount = price_obj.get("priceUsd")
                    unit_price = price_obj.get("unitPriceUsd") or 0
                elif currency == "GEL":
                    amount = price_obj.get("priceGeo")
                    unit_price = price_obj.get("unitPriceGeo") or 0
                elif currency == "EUR":
                    amount = price_obj.get("priceEur")
                    unit_price = price_obj.get("unitPriceEur") or 0
                else:
                    unit_price = 0
                if amount and unit_price and not jl.get("area_m2"):
                    try:
                        jl["area_m2"] = int(round(float(amount) / float(unit_price)))
                    except Exception:
                        pass
            if not (amount and currency):
                og_amt = meta(soup, "product:price:amount")
                og_cur = meta(soup, "product:price:currency")
                if og_amt and not amount:
                    amount = extract_first_number(og_amt)
                if og_cur and not currency:
                    currency = og_cur
        if not (amount and currency):
            amount, currency = _pick_total_price(text_all)

        # Description
        desc = None
        for h in soup.find_all(["h2","h3","div","span"]):
            t = textify(h).lower()
            if t in ("описание","აღწერა","description"):
                nxt = h.find_next(lambda x: x and x.name in ("p","div") and len(textify(x)) > 40)
                if nxt:
                    desc = textify(nxt); break
        if not desc:
            od = meta(soup, "og:description")
            if od and len(od) > 50:
                desc = od.strip()

        # Attributes: берём только значения из <li> без зачёркивания/классов "нет"
        attributes = set()
        feat_words = self.RU_FEATURES + self.KA_FEATURES + self.EN_FEATURES
        for li in soup.find_all("li"):
            cls = " ".join(li.get("class", []))
            style = li.get("style", "")
            if re.search(r"no|not|нет|false|absent|unavailable|close", cls, flags=re.I):
                continue
            if "line-through" in style.lower() or li.find(["s", "del"]):
                continue
            txt = textify(li)
            for kw in feat_words:
                if re.fullmatch(rf"\s*{re.escape(kw)}\s*", txt, flags=re.I):
                    attributes.add(kw)
                    break

        # Numbers / areas
        def find_int(patts: List[str]) -> Optional[int]:
            for pat in patts:
                m = re.search(pat, text_all, flags=re.I)
                if m:
                    n = extract_first_number(m.group(1))
                    if n is not None:
                        return n
            return None

        rooms = jl.get("rooms") or find_int([
            r"(?:Комнат[ыа]|Комнаты)[\s•:–-]*?(\d+)", r"(\d+)\s*room\b", r"(\d+)\s*ოთახ"
        ])
        bedrooms = find_int([
            r"Спальн\w*[\s•:–-]*?(\d+)", r"(\d+)\s*bed\b", r"(\d+)\s*საძინებ\w*"
        ])
        bathrooms = find_int([
            r"(?:Санузел\w*|С/У)[\s•:–-]*?(\d+)", r"(\d+)\s*bath"
        ])

        def area_by_labels(labels: List[str]) -> Optional[int]:
            for lb in labels:
                m = re.search(rf"{lb}\s*[:\-–]?\s*(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I)
                if m:
                    return extract_first_number(m.group(1))
            return None

        # Площадь: только по явным меткам или JSON/JSON-LD (без эвристик)
        area_m2 = jl.get("area_m2") or area_by_labels([
            "Площадь дома","Площадь квартиры","Общая площадь","Жилая площадь",
            "House area","Apartment area","Total area",
            "სახლის ფართი","ბინის ფართი","საერთო ფართი"
        ])
        land_area_m2 = area_by_labels([
            "Площадь участка","Площадь земли","Участок","Land area","Lot area","ეზოს ფართი"
        ])

        # Location + address + floors
        location, addr_line = self._extract_address(soup, text_all)
        if not location:
            location = jl.get("location")
        floor, floors_total = self._parse_floors(text_all)

        # Photos (добавим снимки из рендера + ограничим 10 шт.)
        photo_urls: List[str] = []
        if jl.get("photos"):
            photo_urls.extend([absolutize(u, self.url) for u in jl["photos"]])
        if not photo_urls:
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                if src and "ss.ge" in urlparse(absolutize(src, self.url)).netloc:
                    photo_urls.append(absolutize(src, self.url))
        # дом + JSON от рендера
        photo_urls.extend(dom_imgs_cap)
        photo_urls.extend(json_imgs_cap)
        ordered_photos = uniq_keep_order([u for u in photo_urls if is_good_image_url(u)])[:10]

        # Phones: берём только из ссылок и видимого текста в основной части страницы
        phones = _phones_from_text(" ".join(a.get("href", "") for a in main.select('a[href^="tel:"]')))
        phones.extend(_phones_from_text(text_all))
        phones = [p for p in uniq_keep_order(phones) if p not in self.PHONE_BLACKLIST]

        listing = Listing(
            url=self.url,
            source="home.ss.ge",
            listing_id=listing_id,
            title=title,
            location=location,
            address_line=addr_line,
            price=Price(amount=amount, currency=currency),
            area_m2=area_m2,
            land_area_m2=land_area_m2,
            rooms=rooms,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            floor=floor,
            floors_total=floors_total,
            description=desc,
            attributes=sorted(attributes),
            phones=phones,
            raw_meta={"json_ld": jlds},
        )
        listing.photos = ordered_photos
        return listing

# ---------- MyHome.ge Extractor ----------

class MyHomeExtractor(BaseExtractor):
    """Extracts from myhome.ge pages (JS-heavy SPA)."""

    def _resolve_next_image(self, u: str) -> str:
        if "/_next/image" in u:
            try:
                qs = parse_qs(urlparse(u).query).get("url")
                if qs:
                    return unquote(qs[0])
            except Exception:
                pass
        return u

    def _parse_ru_ka_address_from_text(self, text_all: str) -> Optional[str]:
        # GE: «… ქუჩა 37»
        m = re.search(
            r"([ა-ჰ][ა-ჰ'’\-\s]+?)\s+(ქუჩა|გამზირი|პროსპექტი)\s*(\d+[ა-ჰA-Za-z\-]?)\b(?!\s*\d)",
            text_all, flags=re.I
        )
        if m:
            return f"{m.group(1).strip()} {m.group(2)} {m.group(3)}"

        # RU: «ул./улица/пр-т/проспект … 37»
        m = re.search(
            r"(?:ул\.?|улица|просп(?:\.|ект)?|пр-т|пер\.?|переулок|ш(?:\.|оссе)?)\s*[А-ЯЁа-яё\.\- ]+?\s*(\d+[А-Яа-яA-Za-z\-\/]?)\b(?!\s*\д)",
            text_all, flags=re.I
        )
        return m.group(0).strip() if m else None

    def _parse_floors(self, text_all: str) -> Tuple[Optional[int], Optional[int]]:
        # этаж/этажность
        m = re.search(r"(?:Этаж(?:ность)?|სართული)[\s•:–-]*?(\d+)\s*/\s*(\d+)", text_all, flags=re.I)
        if m:
            return int(m.group(1)), int(m.group(2))
        f = None; ft = None
        m = re.search(r"(?:Этаж|სართული)[\s•:–-]*?(\d+)", text_all, flags=re.I)
        if m: f = int(m.group(1))
        m = re.search(r"(?:Этажей|Этажность|სართულიანი)[\s•:–-]*?(\d+)", text_all, flags=re.I)
        if m: ft = int(m.group(1))
        return f, ft

    def extract(self) -> Listing:
        # сначала HTTP, затем всегда делаем полноценный рендер+захват (SPA)
        html = ""
        try:
            html = self.fetch()
        except Exception:
            pass

        captured_dom_imgs: List[str] = []
        captured_json_imgs: List[str] = []
        captured_json_blobs: List[dict] = []
        html2, dom_imgs, json_imgs, json_blobs = self._render_with_capture()
        if html2:
            html = html2
        captured_dom_imgs = dom_imgs
        captured_json_imgs = json_imgs
        captured_json_blobs = json_blobs

        if not html:
            raise ScrapeError("Failed to load page")

        soup = BeautifulSoup(html, "lxml")
        main = soup.find("main") or soup
        text_all = textify(main)

        # JSON-LD
        jlds = pick_json_ld(html)
        title = None
        location = None
        price_amount, price_currency = None, None
        photos: List[str] = []

        for node in jlds:
            title = title or node.get("name") or node.get("headline")
            offer = node.get("offers") if isinstance(node.get("offers"), dict) else None
            if offer:
                price_amount = price_amount or extract_first_number(str(offer.get("price") or ""))
                price_currency = price_currency or (offer.get("priceCurrency") or None)
            if not location:
                adr = node.get("address")
                if isinstance(adr, dict):
                    parts = [adr.get(x) for x in ("streetAddress", "addressLocality", "addressRegion")]
                    location = " ".join([p for p in parts if p])
                elif isinstance(adr, str):
                    location = adr
            imgs = node.get("image")
            if isinstance(imgs, list):
                photos.extend([str(x) for x in imgs])
            elif isinstance(imgs, str):
                photos.append(imgs)

        # __NEXT_DATA__ – часто содержит массив images
        nd_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        address_line = None
        if nd_tag:
            try:
                nd = json.loads(nd_tag.string or nd_tag.text or '{}')
                s = nd
                for k in ["props", "pageProps", "dehydratedState", "queries"]:
                    s = s.get(k, {})
                if isinstance(s, list) and s:
                    st = s[0].get("state", {}).get("data", {}).get("data", {}).get("statement", {})
                    # фото
                    for img in st.get("images", []) or []:
                        u = (img.get("large") or img.get("thumb") or "").strip()
                        if u: photos.append(u)
                    # адрес прямо из JSON
                    for key in ("address", "addressName", "addressText", "streetAddress"):
                        val = st.get(key)
                        if isinstance(val, str) and re.search(r"(ქუჩა|გამზირი|პროსპექტი|ул|улица|просп|пер|шос)", val,
                                                              re.I):
                            address_line = val.strip()
                            break
                    location = location or st.get("locationName") or st.get("districtName") or None
            except Exception:
                pass

        # OG/product meta
        og_amt = meta(soup, "product:price:amount")
        og_cur = meta(soup, "product:price:currency")
        if not title:
            title = meta(soup, "og:title") or textify(soup.find("h1")) or textify(soup.find("h2"))
        if (not price_amount) and og_amt:
            price_amount = extract_first_number(og_amt)
        if (not price_currency) and og_cur:
            price_currency = og_cur
        # --- Предпочитаем USD, затем EUR, затем GEL (MyHome часто отдаёт GEL) ---
        def pick_best_price(text: str) -> Tuple[Optional[int], Optional[str]]:
            pairs = re.findall(r"([\d\s,\.]+)\s*(₾|\$|€|GEL|USD|EUR)", text, flags=re.I)
            seen: Dict[str, int] = {}
            for amt_raw, cur_raw in pairs:
                amt = extract_first_number(amt_raw)
                cur = normalize_price(cur_raw)[1]
                if amt and cur and cur not in seen:
                    seen[cur] = amt
            for pref in ("USD","EUR","GEL"):
                if pref in seen:
                    return seen[pref], pref
            return None, None
        if True:
            cand_amt, cand_cur = pick_best_price(text_all)
            if cand_amt and cand_cur:
                price_amount, price_currency = cand_amt, cand_cur

        # ID from URL (/pr/<id>)
        path = urlparse(self.url).path
        listing_id = None
        mi = re.search(r"/pr/(\d+)", path)
        if mi:
            listing_id = mi.group(1)

        # rooms / bedrooms & areas
        def find_int(patts: List[str]) -> Optional[int]:
            for pat in patts:
                m = re.search(pat, text_all, flags=re.I)
                if m:
                    n = extract_first_number(m.group(1))
                    if n is not None:
                        return n
            return None

        rooms = find_int([
            r"(?:Комнат[ыа]|Комнаты)[\s•:–-]*?(\d+)", r"(\d+)\s*room\b", r"(\d+)\s*ოთახ"
        ])
        bedrooms = find_int([
            r"Спальн\w*[\s•:–-]*?(\d+)", r"(\d+)\s*bed\b", r"(\d+)\s*საძिनებ\w*"
        ])
        bathrooms = find_int([
            r"(?:Санузел\w*|С/У)[\s•:–-]*?(\d+)", r"(\d+)\s*bath"
        ])

        def area_by_labels(labels: List[str]) -> Optional[int]:
            for lb in labels:
                m = re.search(rf"{lb}\s*[:\-–]?\s*(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I)
                if m:
                    return extract_first_number(m.group(1))
            return None

        area_m2 = area_by_labels(["სახლის ფართი", "ბინის ფართი", "საერთო ფართი",
                                  "Площадь дома", "Площадь квартиры", "Общая площадь", "Площадь",
                                  "House area", "Apartment area", "Total area"])
        land_area_m2 = area_by_labels(["ეზოს ფართი", "Площадь участка", "Land area", "Lot area"])

        # Надёжный fallback: берём РОВНО числа перед m², без склейки "37 60"
        if area_m2 is None or land_area_m2 is None:
            m2s = find_m2_values(text_all)
            if m2s:
                if area_m2 is None:
                    area_m2 = m2s[0]
                # если есть второе значение — обычно это участок у домов
                if land_area_m2 is None and len(m2s) > 1:
                    land_area_m2 = m2s[1]

        # address + floors
        addr_line = address_line or self._parse_ru_ka_address_from_text(text_all)
        floor, floors_total = self._parse_floors(text_all)
        try:
            for blob in captured_json_blobs:
                # часто попадаются ключи floor / totalFloors / numberOfFloors
                strs = json.dumps(blob, ensure_ascii=False)
                m = re.search(r'"(?:floor|floorNumber)"\s*:\s*(\d+)', strs)
                if m and not floor:
                    floor = int(m.group(1))
                mt = re.search(r'"(?:totalFloors|numberOfFloors|floorsTotal)"\s*:\s*(\d+)', strs)
                if mt and not floors_total:
                    floors_total = int(mt.group(1))
                if floor and floors_total:
                    break
        except Exception:
            pass

        # Photos: JSON (captured) -> DOM (captured) -> OG -> fallback DOM
        cand = list(photos)
        cand.extend(captured_json_imgs)
        cand.extend(captured_dom_imgs)
        og_img = meta(soup, "og:image")
        if og_img:
            cand.append(og_img)
        if not cand:
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                if src:
                    cand.append(src)

        resolved = [absolutize(self._resolve_next_image(u), self.url) for u in cand]
        photos_ord = uniq_keep_order([u for u in resolved if is_good_image_url(u) and not re.search(r'(?:_thumb|_blur|google_map)', u)])[:10]

        # Phones: tel: + текст + JSON (только из основной части страницы)
        phones = _phones_from_text(" ".join(a.get("href","") for a in main.select('a[href^="tel:"]')))
        phones.extend(_phones_from_text(text_all))
        for blob in captured_json_blobs:
            for s in _flatten_strings(blob):
                if any(ch.isdigit() for ch in s):
                    phones.extend(_phones_from_text(s))
        phones = uniq_keep_order(phones)

        return Listing(
            url=self.url,
            source="myhome.ge",
            listing_id=listing_id,
            title=title,
            location=location,
            address_line=address_line or self._parse_ru_ka_address_from_text(text_all),
            price=Price(amount=price_amount, currency=price_currency),
            area_m2=area_m2,
            land_area_m2=land_area_m2,
            rooms=rooms,
            bedrooms=bedrooms,
            description=meta(soup, "og:description") or None,
            attributes=[],
            floor=floor,
            floors_total=floors_total,
            photos=photos_ord,
            phones=phones,
            raw_meta={"json_ld": jlds},
        )

# ---------- Router / Runner ----------

def get_extractor(url: str) -> BaseExtractor:
    host = urlparse(url).netloc.lower()
    if "ss.ge" in host:
        return SSGeExtractor(url)
    if "myhome.ge" in host:
        url = force_myhome_ru(url)  # <- всегда русская версия
        return MyHomeExtractor(url)
    raise ScrapeError(f"Unsupported host: {host}")

def load_urls(arg: str) -> List[str]:
    if re.match(r"^https?://", arg.strip(), re.I):
        return [arg.strip()]
    p = Path(arg)
    if not p.exists():
        raise ScrapeError(f"No such file: {arg}")
    urls = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not re.match(r"^https?://", line, re.I):
            continue
        urls.append(line)
    if not urls:
        raise ScrapeError("No valid URLs in file")
    return urls

# ---------- Summary ----------

RU_MAP = {
    "საბურთალო": "Сабуртало",
    "ვაკე": "Ваке",
    "Тбилиси": "Тбилиси",
    "თბილისი": "Тбилиси",
    "Saguramo": "Сагурамо",
    "საგურამო": "Сагурамო",
}

def _sym(cur: Optional[str]) -> str:
    return {"USD": "$", "GEL": "₾", "EUR": "€"}.get((cur or "").upper(), "")

def find_m2_values(text: str) -> List[int]:
    """
    Возвращает все значения перед единицей площади (м² / m² / კვ.მ / მ²),
    не позволяя захватывать *несколько* чисел с пробелом (типа '37 60 м²').
    """
    pat = rf"(?<!\d)(\d+(?:[.,]\d+)?)\s*{MEASURE_M2_RU_KA_EN}"
    vals = []
    for s in re.findall(pat, text, flags=re.I):
        try:
            vals.append(int(float(s.replace(",", ".").strip())))
        except Exception:
            pass
    return vals

def force_myhome_ru(url: str) -> str:
    """
    Делает русскую версию ссылки myhome.ge или home.ss.ge.
    - /pr/...        -> /ru/pr/...
    - /ka/pr/...     -> /ru/pr/...
    - /en/pr/...     -> /ru/pr/...
    Остальные части (query/fragment) сохраняем.
    """
    p = urlparse(url)
    host = p.netloc.lower()
    if "myhome.ge" not in host and "home.ss.ge" not in host:
        return url

    path = p.path or "/"
    # уже RU
    if path.startswith("/ru/"):
        new_path = path
    # была KA/EN
    elif path.startswith("/ka/") or path.startswith("/en/"):
        parts = path.split("/", 2)  # ["", "ka", "rest..."]
        rest = parts[2] if len(parts) >= 3 else ""
        new_path = f"/ru/{rest}".rstrip("/") + ("/" if rest and not rest.endswith("/") and not "." in rest.split("/")[-1] else "")
    else:
        # без языка в пути — просто добавляем /ru
        new_path = ("/ru" + path) if not path.startswith("/ru") else path
    # нормализуем двойные слеши
    new_path = re.sub(r"/{2,}", "/", new_path)
    return urlunparse((p.scheme, p.netloc, new_path, p.params, p.query, p.fragment))

def build_ru_summary(lst: Listing) -> str:
    price = f"{_sym(lst.price.currency)}{lst.price.amount}" if lst.price and lst.price.amount else "—"
    # короткая локация
    loc = (lst.location or "").split(">")[-1].strip() if lst.location else None
    loc_ru = RU_MAP.get(loc or "", loc) if loc else None
    # первая строка
    line1 = f"Аренда {price}" + (f" – {loc_ru}" if loc_ru else "")
    # адрес
    line2 = lst.address_line or (lst.location or "")
    # комнаты/спальни
    line3 = ""
    if lst.rooms is not None or lst.bedrooms is not None:
        parts = []
        if lst.rooms is not None:
            parts.append(f"{lst.rooms} Комнаты")
        if lst.bedrooms is not None:
            parts.append(f"{lst.bedrooms} Спальня" if lst.bedrooms == 1 else f"{lst.bedrooms} Спальни")
        line3 = ", ".join(parts)
    # этаж + метры
    et = ""
    if lst.floor is not None and lst.floors_total is not None:
        et = f"{lst.floor}/{lst.floors_total} Этажей"
    elif lst.floor is not None:
        et = f"{lst.floor} Этаж"
    area = f"{lst.area_m2} м²" if lst.area_m2 else ""
    line4 = ", ".join([x for x in [et, area] if x])
    return "\n".join([x for x in [line1, line2, line3, line4] if x]).strip()

def save_listing(outdir: Path, listing: Listing) -> Path:
    # populate summary
    listing.summary_ru = build_ru_summary(listing)
    lid = listing.listing_id or safe_slug(listing.title or "listing")
    base = outdir / safe_slug(urlparse(listing.url).netloc) / (lid or "item")
    ensure_dir(base)
    (base / "listing.json").write_text(json.dumps(listing.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "summary_ru.txt").write_text(listing.summary_ru or "", encoding="utf-8")
    return base

async def maybe_download_photos(base_dir: Path, listing: Listing, download: bool) -> None:
    if not download or not listing.photos:
        return
    dest = base_dir / "photos"
    # на всякий случай ограничим и при скачивании
    listing.photos = listing.photos[:10]
    extractor = get_extractor(listing.url)
    saved = await extractor.fetch_images(listing.photos, dest)
    (base_dir / "saved_photos.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape myhome.ge / ss.ge to JSON + photos (with WebP→JPEG) + RU summary.")
    ap.add_argument("input", help="URL or path to a text file with URLs (one per line)")
    ap.add_argument("-o", "--outdir", default="out", help="Output directory (default: ./out)")
    ap.add_argument("--no-photos", action="store_true", help="Do not download photos")
    ap.add_argument("--headful", action="store_true", help="Run browser in visible mode for debugging")
    args = ap.parse_args()

    if args.headful:
        os.environ["HEADFUL"] = "1"

    try:
        urls = load_urls(args.input)
    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    rc = 0
    for url in urls:
        print(f"[.] Processing: {url}")
        try:
            extractor = get_extractor(url)
            listing = extractor.extract()

            base_dir = save_listing(outdir, listing)
            print(f"[OK] Saved JSON → {base_dir/'listing.json'}")
            if listing.summary_ru:
                print("----- RU Summary -----")
                print(listing.summary_ru)

            if not args.no_photos:
                asyncio.run(maybe_download_photos(base_dir, listing, download=True))
                print(f"[OK] Photos → {base_dir/'photos'} (if available)")
        except Exception as e:
            rc = 1
            print(f"[FAIL] {url} → {e}", file=sys.stderr)
    return rc

if __name__ == "__main__":
    sys.exit(main())
