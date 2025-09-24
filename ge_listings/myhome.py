# ge_listings/myhome.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from .base import BaseExtractor
from .common import (
    ScrapeError, MEASURE_M2_RU_KA_EN,
    textify, meta, pick_json_ld, absolutize, uniq_keep_order,
    is_good_image_url, extract_first_number, normalize_price, find_m2_values,
    to_ru, to_ru_addr, _phones_from_text, _flatten_strings,
    HEADERS, REQ_TIMEOUT, HTTP2_AVAILABLE,
)

try:
    from loguru import logger as log
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("ge_listings.myhome")


class MyHomeExtractor(BaseExtractor):
    """Extracts from myhome.ge pages (SPA)."""

    PHONE_BLACKLIST = {"+995322800015", "+995507796845"}  # служебные

    # --- мини-транслит KA→RU (для адресов) ---
    _GE2RU = {
        "ა": "а", "ბ": "б", "გ": "г", "დ": "д", "ე": "е", "ვ": "в", "ზ": "з", "თ": "т",
        "ი": "и", "კ": "к", "ლ": "л", "მ": "м", "ნ": "н", "ო": "о", "პ": "п", "ჟ": "ж",
        "რ": "р", "ს": "с", "ტ": "т", "უ": "у", "ფ": "ф", "ქ": "к", "ღ": "г", "ყ": "к",
        "შ": "ш", "ჩ": "ч", "ც": "ц", "ძ": "дз", "წ": "ц", "ჭ": "ч", "ხ": "х", "ჯ": "дж", "ჰ": "х",
    }
    _GE_ADDR_TOKENS = [
        (r"\bქუჩა\b", "ул."), (r"(^|[\s,])ქ\.?\s*", r"\1ул. "),
        (r"\bგამზირი\b", "просп."), (r"\bბულვარი\b", "бул."),
        (r"\bჩიხი\b", "пер."), (r"\bხეივანი\b", "аллея"),
        (r"\bმ/რ\b", "мкр."), (r"\bმისამართი\b", ""),
    ]

    def _resolve_next_image(self, u: str) -> str:
        if "/_next/image" in u:
            try:
                qs = parse_qs(urlparse(u).query).get("url")
                if qs:
                    return unquote(qs[0])
            except Exception:
                pass
        return u

    def _ka_to_ru(self, s: str) -> str:
        return "".join(self._GE2RU.get(ch, ch) for ch in s)

    def _addr_to_ru_smart(self, addr: Optional[str]) -> Optional[str]:
        if not isinstance(addr, str) or not addr.strip():
            return None
        s = addr.strip()
        for pat, repl in self._GE_ADDR_TOKENS:
            s = re.sub(pat, repl, s, flags=re.I)
        if re.search(r"[\u10D0-\u10FF]", s):
            s = self._ka_to_ru(s)
        return re.sub(r"\s{2,}", " ", s).strip()

    def _parse_floors(self, text_all: str) -> Tuple[Optional[int], Optional[int]]:
        m = re.search(r"(?:Этаж(?:ность)?|სართული)\s*[:•–-]?\s*(\d+)\s*/\s*(\d+)", text_all, flags=re.I)
        if m:
            return int(m.group(1)), int(m.group(2))
        f = None
        ft = None
        m = re.search(r"(?:Этаж|სართული)\s*[:•–-]?\s*(\d+)", text_all, flags=re.I)
        if m:
            f = int(m.group(1))
        m = re.search(r"(?:Этажей|Этажность|სართულიანი)\s*[:•–-]?\s*(\d+)", text_all, flags=re.I)
        if m:
            ft = int(m.group(1))
        return f, ft

    # --- извлечь телефоны из произвольного JSON/объекта ---
    def _phones_from_obj(self, o) -> List[str]:
        found: List[str] = []

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(k, str) and "phone" in k.lower():
                        found.extend(_phones_from_text(str(v)))
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
            elif isinstance(x, str) and any(ch.isdigit() for ch in x):
                found.extend(_phones_from_text(x))

        walk(o)
        return uniq_keep_order(found)

    # утилита: только цифры
    @staticmethod
    def _digits(s: str) -> str:
        return re.sub(r"\D+", "", s or "")

    def extract(self):
        debug = os.getenv("GE_DEBUG", "0") != "0"
        if debug:
            log.debug(
                "myhome.extract start url={} need_phones={} env.NEED_PHONES={} api_flag={}",
                self.url,
                bool(getattr(self, "_need_phones", False)),
                os.getenv("GE_NEED_PHONES"),
                os.getenv("GE_MYHOME_PHONE_API", "1"),
            )
        phone_dbg: dict = {
            "url": self.url,
            "need_phones_flag": bool(getattr(self, "_need_phones", False)),
            "env_NEED_PHONES": os.getenv("GE_NEED_PHONES"),
            "api_flag": os.getenv("GE_MYHOME_PHONE_API", "1"),
        }

        # 1) HTML (быстрый путь)
        html = ""
        try:
            html = self.fetch()
        except Exception:
            html = ""

        # 2) Если надо телефон — рендерим и кликаем
        NEED_PHONES = bool(getattr(self, "_need_phones", False) or os.getenv("GE_NEED_PHONES", "0") == "1")

        def _reveal_phone(page: Any) -> dict:
            """Клик по кнопке + ожидание ответа /phone/show + попытка прочитать localStorage."""
            act = {
                "clicked": False,
                "tried_selectors": [],
                "wait_error": None,
                "click_error": None,
                "eval_fetch": {"ok": False, "status": None, "text": None},
                "api_resp_status": None,
                "ls_before": False,
                "ls_after": False,
                "ls_after_sample": "",
                "tel_anchors_count": 0,
                "tel_anchors_sample": [],
            }
            sels = [
                r'role=button[name=/Показать\s+(номер|телефон)/i]',
                r'role=button[name=/Show\s*(number|phone)/i]',
                r'role=button[name=/ნომრის\s*ნახვა/i]',
                'button:has-text("Показать номер")',
                'text=/Показать\\s+(?:номер|телефон)/i',
                'text=/ნომრის\\s*ნახვა/i',
                'text=/Show\\s*(?:number|phone)/i',
            ]
            try:
                # перед кликом — что в LS?
                try:
                    v = page.evaluate("() => window.localStorage.getItem('phoneNumbers')")
                    act["ls_before"] = bool(v)
                except Exception:
                    pass

                # клик
                btn = None
                for sel in sels:
                    act["tried_selectors"].append(sel)
                    l = page.locator(sel)
                    if l.count() > 0:
                        btn = l.first
                        break
                if btn:
                    btn.click(timeout=8_000)
                    act["clicked"] = True

                # ждём целевой ответ (совместимо со старыми Playwright)
                try:
                    page.wait_for_response(
                        lambda r: r and "/v1/statements/phone/show" in (getattr(r, "url", "") or ""), timeout=15_000
                    )
                except AttributeError as e:
                    # Fallback: ждём событие через context
                    try:
                        page.context.wait_for_event(
                            "response",
                            lambda r: r and "/v1/statements/phone/show" in (getattr(r, "url", "") or ""),
                            timeout=15_000,
                        )
                    except Exception as ee:
                        act["wait_error"] = f"{type(e).__name__}/{type(ee).__name__}: {e} / {ee}"
                except Exception as e:
                    act["wait_error"] = f"{type(e).__name__}: {e}"

                # небольшой таймаут, чтобы сайт успел положить номер в LS
                page.wait_for_timeout(500)

                # после клика — localStorage
                try:
                    v2 = page.evaluate("() => window.localStorage.getItem('phoneNumbers')")
                    act["ls_after"] = bool(v2)
                    if isinstance(v2, str) and len(v2) < 300:
                        act["ls_after_sample"] = v2
                except Exception:
                    pass

                # якоря tel: на странице (диагностика)
                try:
                    hrefs = page.evaluate(
                        """() => Array.from(document.querySelectorAll('a[href^="tel:"]'))
                                .map(a => a.textContent.trim()).filter(Boolean)"""
                    )
                    if isinstance(hrefs, list):
                        act["tel_anchors_count"] = len(hrefs)
                        act["tel_anchors_sample"] = [str(x) for x in hrefs[:5]]
                except Exception:
                    pass

                # Крайний случай — попробовать вызвать API прямо из контекста страницы (УЖЕ через POST)
                try:
                    st_uuid = page.evaluate(
                        "() => {"
                        "  const nd=document.querySelector('#__NEXT_DATA__');"
                        "  if(!nd) return null;"
                        "  try{"
                        "    const d=JSON.parse(nd.textContent);"
                        "    const qs=(d?.props?.pageProps?.dehydratedState?.queries)||[];"
                        "    for (const q of qs){"
                        "      const st=q?.state?.data;"
                        "      const inner=(st && (st.data||st))||{};"
                        "      const s=inner?.statement;"
                        "      if (s && (s.uuid||s.statement_uuid)) return s.uuid||s.statement_uuid;"
                        "    }"
                        "  }catch(e){return null}"
                        "  return null;"
                        "}"
                    )
                except Exception:
                    st_uuid = None
                if st_uuid:
                    try:
                        js = """
                          (uuid) => fetch('https://api-statements.tnet.ge/v1/statements/phone/show', {
                            method: 'POST',
                            credentials: 'include',
                            headers: {
                              'content-type': 'application/json',
                              'accept': 'application/json, text/plain, */*'
                            },
                            body: JSON.stringify({ statement_uuid: uuid })
                          }).then(r => r.text()
                                   .then(t => ({ok:r.ok,status:r.status,text:t})))
                            .catch(e => ({ok:false,status:null,text:String(e)}))
                        """
                        res = page.evaluate(js, st_uuid)
                        if isinstance(res, dict):
                            act["eval_fetch"] = res
                    except Exception:
                        pass
            except Exception as e:
                act["click_error"] = f"{type(e).__name__}: {e}"
            return act

        # рендер с действием — либо как единственный, либо как дополнительный
        captured_dom_imgs: List[str] = []
        captured_json_imgs: List[str] = []
        captured_json_blobs: List[dict] = []
        if not html:
            html2, d1, j1, b1 = self._render_with_capture(_reveal_phone if NEED_PHONES else None)
            if html2:
                html = html2
            captured_dom_imgs, captured_json_imgs, captured_json_blobs = d1, j1, b1
        elif NEED_PHONES:
            html2, d1, j1, b1 = self._render_with_capture(_reveal_phone)
            if html2:
                html = html2
            captured_dom_imgs, captured_json_imgs, captured_json_blobs = d1, j1, b1

        if not html:
            raise ScrapeError("Failed to load page")

        # доступ к внутреннему логу рендера от BaseExtractor
        render_log = getattr(self, "_last_render_log", None)
        if render_log:
            phone_dbg["render_log"] = render_log

        soup = BeautifulSoup(html, "lxml")
        main = soup.find("main") or soup
        text_all = textify(main)

        # JSON-LD
        jlds = pick_json_ld(html)
        title, location = None, None
        price_amount, price_currency = None, None
        description = None
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
                    city = adr.get("addressLocality")
                    region = adr.get("addressRegion")
                    cand = ", ".join([p for p in [city, region] if p])
                    if cand:
                        location = cand
                elif isinstance(adr, str):
                    location = adr
            imgs = node.get("image")
            if isinstance(imgs, list):
                photos.extend([str(x) for x in imgs])
            elif isinstance(imgs, str):
                photos.append(imgs)

        # __NEXT_DATA__ → statement
        nd_tag = soup.find("script", {"id": "__NEXT_DATA__"})
        raw_addr = None
        st: dict = {}
        statement_uuid: Optional[str] = None
        masked_user_phone: Optional[str] = None
        mask_prefix: Optional[str] = None  # например "595858"
        if nd_tag:
            try:
                nd = json.loads(nd_tag.string or nd_tag.text or "{}")
                queries = nd.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
                if isinstance(queries, list):
                    for q in queries:
                        state = q.get("state", {})
                        data = state.get("data") if isinstance(state.get("data"), dict) else {}
                        inner = data.get("data") if isinstance(data.get("data"), dict) else data
                        cand = inner.get("statement") if isinstance(inner.get("statement"), dict) else {}
                        if cand.get("price") or cand.get("images") or cand.get("id"):
                            st = cand
                            break

                if isinstance(st, dict):
                    statement_uuid = st.get("uuid") or st.get("statement_uuid")
                    masked_user_phone = st.get("user_phone_number") or st.get("userPhoneNumber")
                    if masked_user_phone:
                        # исправлено: берём цифры слева от звёздочек, не вычищая их заранее
                        mp = re.search(r"(\d+)\s*\*+", str(masked_user_phone))
                        if mp:
                            mask_prefix = re.sub(r"\D+", "", mp.group(1))
                        else:
                            left = str(masked_user_phone).split("*", 1)[0]
                            if re.search(r"\d", left or ""):
                                mask_prefix = re.sub(r"\D+", "", left)
                    if debug:
                        log.debug(
                            "myhome: statement_uuid={} masked={} mask_prefix={}",
                            statement_uuid,
                            masked_user_phone,
                            mask_prefix,
                        )
                    phone_dbg["statement_uuid"] = statement_uuid
                    phone_dbg["mask_from_nd"] = masked_user_phone
                    phone_dbg["mask_prefix"] = mask_prefix

                # фото
                for img in (st.get("images") or []):
                    u = (img.get("large") or img.get("thumb") or "").strip()
                    if u:
                        photos.append(u)

                # заголовок
                title = title or st.get("dynamic_title") or None

                # адрес (сырой)
                for key in ("address", "addressName", "addressText", "streetAddress"):
                    val = st.get(key)
                    if isinstance(val, str) and val.strip():
                        raw_addr = val.strip()
                        break

                # локация: только city + urban
                city_name = st.get("city_name") or st.get("cityName")
                urban_name = st.get("urban_name") or st.get("urbanName")
                if city_name:
                    location = city_name if not urban_name else f"{city_name}, {urban_name}"

                # цена USD
                price_map = st.get("price") if isinstance(st.get("price"), dict) else {}
                usd = price_map.get("2") if isinstance(price_map, dict) else None
                if isinstance(usd, dict):
                    amt = usd.get("price_total") or usd.get("priceTotal") or usd.get("total")
                    if isinstance(amt, (int, float)):
                        price_amount, price_currency = int(amt), "USD"

                if price_amount is None:
                    amt = st.get("total_price") or st.get("price_total") or st.get("price")
                    cid = st.get("currency_id") or st.get("currencyId")
                    if isinstance(amt, (int, float)) and cid == 2:
                        price_amount, price_currency = int(amt), "USD"

                # описание
                if not description:
                    descr = st.get("comment") or st.get("description")
                    if isinstance(descr, str):
                        try:
                            desc_txt = BeautifulSoup(descr, "lxml").get_text("\n", strip=True)
                        except Exception:
                            desc_txt = descr.strip()
                        description = to_ru(desc_txt)

            except Exception:
                pass

        # OG/фолбэки
        og_amt = meta(soup, "product:price:amount")
        og_cur = meta(soup, "product:price:currency")
        if not title:
            title = meta(soup, "og:title") or textify(soup.find("h1")) or textify(soup.find("h2"))
        if (not price_amount) and og_amt:
            price_amount = extract_first_number(og_amt)
        if (not price_currency) and og_cur:
            price_currency = og_cur

        # если не нашли цену — берём из текста (USD)
        if price_amount is None or price_currency is None:
            for amt_raw, cur_raw in re.findall(r"([\d\s,\.]+)\s*(₾|\$|€|GEL|USD|EUR)", text_all, flags=re.I):
                amt = extract_first_number(amt_raw)
                cur = normalize_price(cur_raw)[1]
                if amt and cur == "USD":
                    price_amount, price_currency = amt, "USD"
                    break

        # id из URL
        mi = re.search(r"/pr/(\d+)", urlparse(self.url).path)
        listing_id = mi.group(1) if mi else None

        # минимальный парсинг числовых полей
        def find_int2(patts: List[str]) -> Optional[int]:
            for pat in patts:
                m = re.search(pat, text_all, flags=re.I)
                if m:
                    n = extract_first_number(m.group(1))
                    if n is not None:
                        return n
            return None

        rooms = st.get("room_type_id") if isinstance(st.get("room_type_id"), int) else None
        if rooms is None:
            rooms = find_int2([r"(?:Комнат[ыа]|Комнаты)\s*[:•–-]?\s*(\d+)", r"(\d+)\s*room\b", r"(\d+)\s*ოთახ"])

        bedrooms = st.get("bedroom_type_id") if isinstance(st.get("bedroom_type_id"), int) else None
        if bedrooms is None:
            bedrooms = find_int2([r"Спальн\w*\s*[:•–-]?\s*(\d+)", r"(\d+)\s*bed\b", r"(\d+)\s*საძინებ\w*"])

        bathrooms = st.get("bathroom_type_id") if isinstance(st.get("bathroom_type_id"), int) else None
        if bathrooms is None:
            bathrooms = find_int2([r"(?:Санузел\w*|С/У)\s*[:•–-]?\s*(\d+)", r"(\d+)\s*bath"])

        area_m2 = None
        if isinstance(st.get("area"), (int, float)):
            area_m2 = int(st["area"])

        def area_by_labels2(labels: List[str]) -> Optional[int]:
            for lb in labels:
                m = re.search(
                    rf"{lb}\s*[:\-–]?\s*(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I
                )
                if m:
                    return extract_first_number(m.group(1))
            return None

        if area_m2 is None:
            area_m2 = area_by_labels2(
                [
                    "სახლის ფართი",
                    "ბინის ფართი",
                    "საერთო ფართი",
                    "Площадь дома",
                    "Площадь квартиры",
                    "Общая площадь",
                    "Площадь",
                    "House area",
                    "Apartment area",
                    "Total area",
                ]
            )

        land_area_m2 = None
        if isinstance(st.get("yard_area"), (int, float)) and st["yard_area"] > 0:
            land_area_m2 = int(st["yard_area"])
        if land_area_m2 is None:
            land_area_m2 = area_by_labels2(["ეზოს ფართი", "Площадь участка", "Land area", "Lot area"])
        if land_area_m2 == area_m2:
            land_area_m2 = None

        # этажность
        floor = st.get("floor") if isinstance(st.get("floor"), int) else None
        floors_total = st.get("total_floors") if isinstance(st.get("total_floors"), int) else None
        if floor is None or floors_total is None:
            f, ft = self._parse_floors(text_all)
            floor = floor or f
            floors_total = floors_total or ft

        # адрес и локация
        address_line = None
        if raw_addr:
            address_line = self._addr_to_ru_smart(raw_addr) or to_ru_addr(raw_addr)
        if not address_line and location:
            address_line = self._addr_to_ru_smart(location) or to_ru_addr(location)
        location_ru = None
        if location:
            location_ru = self._addr_to_ru_smart(location) or to_ru_addr(location)
        if not location_ru and location:
            location_ru = to_ru(location)

        # --- Фото ---
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

        def _looks_like_service_icon(u: str) -> bool:
            try:
                p = urlparse(u)
                host = (p.netloc or "").lower()
                path = (p.path or "").lower()
                if "livo.ge" in host and "/svg/" in path:
                    return True
                if "nearby-places" in path or "/icons/" in path or "/icon" in path:
                    return True
                return False
            except Exception:
                return False

        photos_ord = uniq_keep_order(
            [
                u
                for u in resolved
                if is_good_image_url(u)
                and not re.search(r"(?:_thumb|_blur|google_map)", u, flags=re.I)
                and not _looks_like_service_icon(u)
            ]
        )[:10]

        # --- Телефоны ---
        phones: List[str] = []
        # 1) JSON-блоб прямо с API или localStorage -> json_blobs
        for blob in captured_json_blobs:
            if isinstance(blob, (dict, list)):
                phs = self._phones_from_obj(blob)
                if phs:
                    phones.extend(phs)
        # 2) <a href="tel:..."> и текст (как крайний фолбэк)
        if not phones:
            phones.extend(_phones_from_text(" ".join(a.get("href", "") for a in soup.select('a[href^="tel:"]'))))
            phones.extend(_phones_from_text(textify(soup)))

        # remove служебные
        phones = [p for p in uniq_keep_order(phones) if p not in self.PHONE_BLACKLIST]

        # --- Ключевой фильтр по маске из __NEXT_DATA__ (ориентир) ---
        phone_dbg["candidates_all"] = phones.copy()
        picked_source = None
        if mask_prefix:
            masked_match = [p for p in phones if self._digits(p).startswith(mask_prefix)]
            phone_dbg["candidates_mask_filtered"] = masked_match.copy()
            if masked_match:
                phones = masked_match
                picked_source = "mask_prefix"

        # Если есть точный ответ из API (/phone/show), используем в приоритете
        if render_log and isinstance(render_log, dict):
            api_phone = render_log.get("api_phone_number")
            if api_phone:
                parsed = _phones_from_text(str(api_phone))
                if parsed:
                    phones = [parsed[0]]
                    picked_source = picked_source or "api_phone"

        # Ещё одна попытка — localStorage.phoneNumbers
        if (not phones) and render_log and isinstance(render_log, dict):
            ls = render_log.get("ls_after_value")
            if isinstance(ls, list) and ls:
                phs = _phones_from_text(" ".join(str(x) for x in ls))
                if phs:
                    phones = phs
                    picked_source = picked_source or "localStorage"

        phone_dbg["picked_source"] = picked_source or "fallback"
        if debug:
            log.debug("myhome: phones final={}", phones)

        return self._as_listing(
            title=title,
            location_ru=location_ru,
            address_line=address_line,
            price_amount=price_amount,
            price_currency=price_currency,
            area_m2=area_m2,
            land_area_m2=land_area_m2,
            rooms=rooms,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            floor=floor,
            floors_total=floors_total,
            description=description,
            photos=photos_ord,
            phones=phones,
            jlds=jlds,
            phone_dbg=phone_dbg,
            listing_id=listing_id,
        )

    # pack to Listing
    def _as_listing(
        self,
        *,
        title,
        location_ru,
        address_line,
        price_amount,
        price_currency,
        area_m2,
        land_area_m2,
        rooms,
        bedrooms,
        bathrooms,
        floor,
        floors_total,
        description,
        photos,
        phones,
        jlds,
        phone_dbg,
        listing_id,
    ):
        from .models import Listing, Price

        return Listing(
            url=self.url,
            source="myhome.ge",
            listing_id=listing_id,
            title=title,
            location=location_ru,
            address_line=address_line,
            price=Price(amount=price_amount, currency=price_currency),
            area_m2=area_m2,
            land_area_m2=land_area_m2,
            rooms=rooms,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            floor=floor,
            floors_total=floors_total,
            description=description,
            attributes=[],
            photos=photos,
            phones=phones,
            raw_meta={"json_ld": jlds, "debug": {"phone": phone_dbg}},
        )
