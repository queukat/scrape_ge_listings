from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

from .base import BaseExtractor
from .common import (
    ScrapeError, MEASURE_M2_RU_KA_EN,
    textify, meta, pick_json_ld, absolutize, uniq_keep_order,
    is_good_image_url, extract_first_number, normalize_price, find_m2_values,
    parse_ru_ka_address_from_text, to_ru, _phones_from_text
)
from .models import Listing, Price


def _pick_total_price(text: str, allowed: Optional[Tuple[str, ...]] = None) -> Tuple[Optional[int], Optional[str]]:
    """
    Фоллбек: берём только пары "число валюта" из явных цен, игнорируя м² и
    стараясь НЕ подцепить «похожие» и «историю». Для надёжности анализируйте
    результат DOM-парсинга .price прежде чем вызывать это.
    """
    pairs: List[Tuple[int, str, int]] = []  # (amt, cur, start_index)
    for m in re.finditer(r"(\d[\d\s,\.]{1,9})\s*(₾|\$|€|GEL|USD|EUR)", text, flags=re.I):
        # игнорируем unit price по m²
        left = text[max(0, m.start() - 16):m.start()].lower()
        if any(x in left for x in ("m²", "m2", "м²", "кв. м", "кв.м", "m² -", "m2 -")):
            continue
        amt = extract_first_number(m.group(1))
        cur = normalize_price(m.group(2))[1]
        if amt and cur and (allowed is None or cur in allowed):
            pairs.append((amt, cur, m.start()))
    if not pairs:
        return None, None
    # Приоритет: ПЕРВОЕ в тексте вхождение (обычно главный блок страницы), а не максимум.
    pairs.sort(key=lambda x: x[2])  # по позиции
    best_amt, best_cur, _ = pairs[0]
    return best_amt, best_cur


class SSGeExtractor(BaseExtractor):
    """Extracts from home.ss.ge listing pages (RU/KA/EN)."""

    RU_FEATURES = [
        "кондиционер", "балкон", "подвал", "кабельное телевидение", "питьевая вода", "лифт", "холодильник", "мебель",
        "гараж", "стекло-пакет", "цент. отопление", "горячая вода", "интернет", "железная дверь",
        "природный газ", "сигнализация", "хранилище", "телефон", "телевизор", "стиральная машина", "бассейн"
    ]
    KA_FEATURES = [
        "კონდიციონერი", "აივანი", "სარდაფი", "საკაბელო ტელევიზია", "სასმელი წყალი", "ლიფტი", "მაცივარი", "ავეჯი",
        "გარაჟი", "მინა-პაკეტი", "ცენტ. გათბობა", "ცხელი წყალი", "ინტერნეტი", "რკინის კარი", "ბუნებრივი აირი",
        "სიგნალიზაცია", "სათავსო", "ტელეფონი", "ტელევიზორი", "სარეცხი მანქანა", "აუზი"
    ]
    EN_FEATURES = [
        "air conditioner", "balcony", "basement", "cable tv", "drinking water", "elevator", "fridge", "furniture",
        "garage", "double glazed", "central heating", "hot water", "internet", "metal door", "natural gas",
        "alarm system", "storage", "telephone", "tv", "washing machine", "pool"
    ]

    PHONE_BLACKLIST = {"+995322121661"}

    # 1) Чёрный список картинок (подстроки в URL)
    AD_IMAGE_PATTERNS = [
        # Рекламная/служебная картинка, которая попадает в галерею
        "098af163-1157-4fa7-ac28-729e596b3a33",
        # при необходимости сюда можно добавлять другие стабильные UUID/пути
    ]

    def _is_ad_image(self, url: str) -> bool:
        """True если картинка — реклама/заглушка, которую надо исключить."""
        if not url:
            return False
        u = url.lower()
        # точечные паттерны
        for token in self.AD_IMAGE_PATTERNS:
            if token in u:
                return True
        # лёгкая подстраховка против иных служебных изображений
        if re.search(r'/icons/|/svg/|/banners?/', u):
            return True
        return False

    # ---------- helpers ----------

    def _jsonld_fill(self, jlds: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        photos: List[str] = []
        for node in jlds or []:
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

    def _extract_address_from_h2(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Адрес часто находится в одном из <h2>. Просматриваем все h2 и
        распознаём больше типов: пл./площадь, бул./бульвар, ал./аллея, квартал и т.п.
        """
        addr_tokens = (
            r"(ул\.?|улица|пр(?:осп\.?|оспект)?|пр-т|пер\.?|переулок|"
            r"ш(?:\.|оссе)?|пл\.?|площадь|бул\.?|бульвар|ал\.?|аллея|"
            r"кв-л|квартал)"
        )
        for h2 in soup.find_all("h2"):
            t = textify(h2)
            if re.search(fr"\b{addr_tokens}\b", t, flags=re.I):
                return t.strip()
        return None

    def _extract_city_district(self, soup: BeautifulSoup, text_all: str, title: Optional[str]) -> Tuple[
        Optional[str], Optional[str]]:
        """
        Выделяем город/район: из h1 ('Темка' в конце), из заголовков блоков,
        а город — «Тбилиси» по умолчанию для известных районов Тбилиси.
        """
        # район по словарю (расширяем по мере надобности)
        DIST_MAP = {
            "темка": "Темка",
            "сабуртало": "Сабуртало",
            "нуцубидзе": "Нуцубидзе",
            "вара": "Вера",
            "вера": "Вера",
            "ваке": "Ваке",
            "мтацминда": "Мтацминда",
            "исани": "Исани",
            "самгори": "Самгори",
            "глдани": "Глдани",
            "надзаладеви": "Надзаладеви",
            "авлабари": "Авлабари",
            "ортапала": "Ортачала",
            "ортачала": "Ортачала",
            "дигоми": "Дигоми",
        }
        # ищем район в h1
        district = None
        h1 = soup.find("h1")
        h1t = (textify(h1) if h1 else "") or (title or "")
        for k, v in DIST_MAP.items():
            if re.search(rf"\b{k}\b", h1t, flags=re.I):
                district = v
                break
        # если не нашли — пробуем по другим текстам (динам. цен и т.п.)
        if not district:
            m = re.search(r"динамик\w* цен\s*[-–—]\s*([A-Za-zА-Яа-яЁё\- ]{2,30})", text_all, flags=re.I)
            if m:
                cand = m.group(1).strip()
                for k, v in DIST_MAP.items():
                    if re.search(rf"\b{k}\b", cand, flags=re.I):
                        district = v
                        break

        # город
        city = None
        if re.search(r"\b(Тбилиси|თბილისი|Tbilisi)\b", text_all, flags=re.I):
            city = "Тбилиси"
        elif re.search(r"\b(Батуми|ბათუმი|Batumi)\b", text_all, flags=re.I):
            city = "Батуми"
        # если район тбилисский, но город не увидели явным текстом — подставим Тбилиси
        if district and not city:
            city = "Тбилиси"
        return city, district

    def _parse_floors(self, text_all: str) -> Tuple[Optional[int], Optional[int]]:
        m = re.search(r"(?:Этаж(?:ность)?|სართული)[\s•:–-]*?(\d+)\s*/\s*(\d+)", text_all, flags=re.I)
        if m:
            return int(m.group(1)), int(m.group(2))
        f = None;
        ft = None
        m = re.search(r"(?:Этаж|სართული)[\s•:–-]*?(\d+)", text_all, flags=re.I)
        if m: f = int(m.group(1))
        m = re.search(r"(?:Этажей|Этажность|სართულიანი)[\s•:–-]*?(\d+)", text_all, flags=re.I)
        if m: ft = int(m.group(1))
        return f, ft

    def _click_show_phone(self, page) -> None:
        """Пробуем раскрыть телефоны (RU/KA/EN). Кликаем все контролы показа номеров."""
        selectors = [
            'button:has-text("Показать номер")',
            'text=/Показать\\s+(?:номер|телефон)/i',
            'text=/ნომრის\\s*ნახვა/i',
            'text=/Show\\s*(?:number|phone)/i',
        ]
        try:
            for sel in selectors:
                loc = page.locator(sel)
                for i in range(loc.count()):
                    try:
                        loc.nth(i).click(timeout=2000)
                    except Exception:
                        pass
            for sel in selectors:
                loc = page.locator(sel)
                for i in range(loc.count()):
                    try:
                        loc.nth(i).click(timeout=1500)
                    except Exception:
                        pass
            try:
                page.wait_for_selector(
                    'a[href^="tel:"], a[href*="wa.me/"], a[href^="viber:"]',
                    timeout=5000
                )
            except Exception:
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass
        except Exception:
            pass

    def _extract_gallery_photos(self, soup: BeautifulSoup) -> List[str]:
        """Достаём картинки из lightGallery (.lg-react-element) + слайдер (#details-page-gallery)."""
        urls: List[str] = []

        # 1) lightGallery — полноразмерные .jpg
        lg = soup.select_one('.lg-react-element')
        if lg:
            for a in lg.select('a[data-src]'):
                u = (a.get('data-src') or '').strip()
                if u:
                    urls.append(u)
            for img in lg.select('img[src]'):
                u = (img.get('src') or '').strip()
                if u:
                    urls.append(u)

        # 2) Слайды с _Thumb — конвертируем в оригинал
        gal = soup.select_one('#details-page-gallery')
        if gal:
            for img in gal.select('img[src], img[data-src]'):
                u = (img.get('src') or img.get('data-src') or '').strip()
                if not u:
                    continue
                u = re.sub(r'_Thumb(\.\w+)$', r'\1', u)
                urls.append(u)

        urls = [absolutize(u, self.url) for u in urls]
        urls = [u for u in urls if is_good_image_url(u)]
        # >>> добавляем:
        urls = [u for u in urls if not self._is_ad_image(u)]
        return uniq_keep_order(urls)[:10]

    def _find_contact_container(self, soup: BeautifulSoup):
        # 0) Сначала отталкиваемся от явных ссылок на телефон/мессенджеры
        anchor = soup.find('a', href=re.compile(r'^(?:tel:|viber:)|wa\.me/'))
        node = anchor
        for _ in range(6):
            if not node: break
            if getattr(node, 'find', None) and (
                    node.find('a', href=re.compile(r'^(?:tel:|viber:)')) or
                    node.find('a', href=re.compile(r'wa\.me/')) or
                    node.find('h6') or
                    node.find('button')
            ):
                return node
            node = node.parent

        # 1) Фоллбек — от ссылки профиля пользователя
        anchor = soup.find('a', href=re.compile(r'userlist\?userId='))
        node = anchor
        for _ in range(6):
            if not node: break
            if getattr(node, 'find', None) and (
                    node.find('a', href=re.compile(r'^(?:tel:|viber:)')) or
                    node.find('a', href=re.compile(r'wa\.me/')) or
                    node.find('h6') or
                    node.find('button')
            ):
                return node
            node = node.parent
        return None

    def _price_from_dom(self, soup: BeautifulSoup) -> Tuple[Optional[int], Optional[str], Optional[int]]:
        """
        Достаём цену из верхнего блока .price (главный прайс): число + валютный тоггл.
        Возвращаем (amount, currency, unit_usd) — unit_usd может быть None.
        """
        box = soup.select_one('.price')
        if not box:
            return None, None, None
        # число (первая жирная цифра)
        amt = None
        for sp in box.find_all("span"):
            t = (sp.get_text(strip=True) or "")
            if re.fullmatch(r"\d[\d\s,\.]*", t):
                amt = extract_first_number(t)
                if amt:
                    break
        # валюта: если в блоке есть символ '$' (видимый), считаем USD
        cur = None
        if "$" in textify(box):
            cur = "USD"
        elif "USD" in textify(box):
            cur = "USD"
        elif "₾" in textify(box) or "GEL" in textify(box):
            cur = "GEL"
        elif "€" in textify(box) or "EUR" in textify(box):
            cur = "EUR"

        # unit price "1 m² - 8 $" — попробуем вытащить
        unit = None
        unit_el = soup.find(string=re.compile(r"1\s*m[²2]\s*-\s*\d"))
        if unit_el:
            m = re.search(r"1\s*m[²2]\s*-\s*(\d[\d\s,\.]*)\s*(?:\$|USD)", unit_el, flags=re.I)
            if m:
                unit = extract_first_number(m.group(1))

        return (amt, cur, unit)

    # ---------- main ----------

    def extract(self) -> Listing:
        html = ""
        dom_imgs_cap: List[str] = []
        json_imgs_cap: List[str] = []
        json_blobs_cap: List[dict] = []
        try:
            html = self.fetch()
        except Exception:
            pass

        # всегда пробуем раскрыть телефон (и перехватить JSON) — дешёво по времени
        html2, dom_imgs_cap, json_imgs_cap, json_blobs_cap = self._render_with_capture(self._click_show_phone)
        if html2:
            html = html2
        if not html:
            raise ScrapeError("Failed to load page")

        soup = BeautifulSoup(html, "lxml")
        main = soup.find("main") or soup
        text_all = textify(main)

        # JSON-LD как слабый источник (на ss.ge часто пустой/неполный)
        jlds = pick_json_ld(html)
        jl = self._jsonld_fill(jlds) if jlds else {}

        # --- Title / ID ---
        title = jl.get("title") or textify(soup.find("h1")) or meta(soup, "og:title")
        listing_id = None
        m = re.search(r"\bID\s*[-–]?\s*(\d+)\b", text_all, flags=re.I)
        if m:
            listing_id = m.group(1)

        # --- PRICE (USD only) ---
        amount = jl.get("price_amount")
        currency = jl.get("price_currency")
        if currency and currency.upper() != "USD":
            amount, currency = None, None

        unit_price_usd = None

        if not (amount and currency):
            # Встроенный JSON: "price": {...}
            m_price = re.search(r'"price"\s*:\s*(\{[^{}]+\})', html)
            price_obj = None
            if m_price:
                try:
                    price_obj = json.loads(m_price.group(1))
                except Exception:
                    price_obj = None

            # Прямые поля (если есть) — предпочитаем USD
            if price_obj:
                usd_amt = price_obj.get("priceUsd")
                usd_unit = price_obj.get("unitPriceUsd")
                if usd_amt:
                    amount = int(round(float(usd_amt)))
                    currency = "USD"
                    unit_price_usd = int(round(float(usd_unit))) if usd_unit else None
                else:
                    # Маппинг на случай отсутствия priceUsd
                    cur_map = {1: "GEL", 2: "USD", 3: "EUR"}
                    cur_type = price_obj.get("currencyType")
                    cur = cur_map.get(cur_type)
                    if cur == "USD":
                        amount = int(round(float(price_obj.get("priceUsd") or 0)))
                        unit_price_usd = int(round(float(price_obj.get("unitPriceUsd") or 0))) or None
                        currency = "USD" if amount else None

            # Если всё ещё пусто — любые следы priceUsd/Unit из HTML/JSON‑блобов
            if not (amount and currency):
                try:
                    s_all = html
                    m_amt = re.search(r'"priceUsd"\s*:\s*([0-9]+(?:\.[0-9]+)?)', s_all)
                    if m_amt:
                        amount = int(round(float(m_amt.group(1))))
                        currency = "USD"
                    m_unit = re.search(r'"unitPriceUsd"\s*:\s*([0-9]+(?:\.[0-9]+)?)', s_all)
                    if m_unit:
                        unit_price_usd = int(round(float(m_unit.group(1))))
                except Exception:
                    pass

        # DOM‑прайс (главный блок .price)
        if not (amount and currency):
            dom_amt, dom_cur, dom_unit = self._price_from_dom(soup)
            if dom_cur == "USD" and dom_amt:
                amount, currency = dom_amt, "USD"
                unit_price_usd = unit_price_usd or dom_unit

        # OG (если настроено и в USD)
        if not (amount and currency):
            og_amt = meta(soup, "product:price:amount")
            og_cur = meta(soup, "product:price:currency")
            if og_amt and og_cur and og_cur.upper() == "USD":
                amount = extract_first_number(og_amt)
                currency = "USD"

        # Текстовый фоллбек — только USD и только первое вхождение
        if not (amount and currency):
            amount, currency = _pick_total_price(text_all, allowed=("USD",))

        # --- Description ---
        desc = None
        # 1) «Описание» -> следующий большой блок
        for h in soup.find_all(["h2", "h3", "div", "span"]):
            t = textify(h).strip().lower()
            if t in ("описание", "აღწერა", "description"):
                nxt = h.find_next(lambda x: x and x.name in ("p", "div") and len(textify(x)) > 40)
                if nxt:
                    desc = textify(nxt).strip()
                    break
        # 2) application_desc (ваш пример)
        if not desc:
            desc_el = soup.select_one('#application_desc')
            if desc_el:
                d = textify(desc_el).strip()
                if d:
                    desc = d

        # --- Attributes (через <h3>, с фильтром disabled) ---
        attributes = set()
        feat_words = [*self.RU_FEATURES, *self.KA_FEATURES, *self.EN_FEATURES]
        for h3 in soup.find_all("h3"):
            txt = textify(h3).strip().lower()
            if not txt:
                continue
            # пропускаем отключённые
            par = h3.parent
            disabled = False
            for up in [par, getattr(par, "parent", None)]:
                if getattr(up, "has_attr", lambda *_: False)("disabled") and up["disabled"] is not None:
                    disabled = True
                    break
                classes = " ".join(getattr(up, "get", lambda *_: [])("class", [])) if up else ""
                if re.search(r"\bcwTuFe\b", classes) and getattr(up, "has_attr", lambda *_: False)("disabled"):
                    disabled = True
                    break
            if disabled:
                continue
            if any(re.fullmatch(rf"{re.escape(kw)}", txt, flags=re.I) for kw in feat_words):
                attributes.add(txt)

        # --- Numbers / areas ---
        def find_int(patts: List[str]) -> Optional[int]:
            for pat in patts:
                m = re.search(pat, text_all, flags=re.I)
                if m:
                    n = extract_first_number(m.group(1))
                    if n is not None:
                        return n
            return None

        # добавили варианты «число перед словом»
        rooms = jl.get("rooms") or find_int([
            r"(?:Комнат[ыа]|Комнаты)[\s•:–-]*?(\d+)",
            r"(\d+)\s*Комнат",
            r"(\d+)\s*room\b",
            r"(\d+)\s*ოთახ"
        ])
        bedrooms = find_int([
            r"Спальн\w*[\s•:–-]*?(\d+)",
            r"(\d+)\s*Спальн",
            r"(\d+)\s*bed\b",
            r"(\d+)\s*საძინებ\w*"
        ])
        bathrooms = find_int([
            r"(?:Санузел\w*|С/У)[\s•:–-]*?(\d+)",
            r"(\d+)\s*Сануз",
            r"(\d+)\s*bath"
        ])

        def area_by_labels(labels: List[str]) -> Optional[int]:
            for lb in labels:
                m = re.search(rf"{lb}\s*[:\-–]?\s*(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I)
                if m:
                    return extract_first_number(m.group(1))
            return None

        area_m2 = jl.get("area_m2") or area_by_labels([
            "Площадь дома", "Площадь квартиры", "Общая площадь", "Жилая площадь",
            "House area", "Apartment area", "Total area",
            "სახლის ფართი", "ბინის ფართი", "საერთო ფართი"
        ])

        # если метка слева от числа (как у вас), сработает общий парсер m²
        if area_m2 is None:
            m2s = find_m2_values(text_all)
            if m2s:
                area_m2 = m2s[0]

        land_area_m2 = area_by_labels([
            "Площадь участка", "Площадь земли", "Участок", "Land area", "Lot area", "ეზოს ფართი"
        ])
        if land_area_m2 == area_m2:
            land_area_m2 = None

        # --- Address / Location / Floors ---
        # адрес — из h2 ("ул. Анапи 16") либо общим парсером
        addr_from_h2 = self._extract_address_from_h2(soup)
        raw_addr = addr_from_h2 or parse_ru_ka_address_from_text(text_all)
        address_line = to_ru(raw_addr) if raw_addr else None
        if address_line:
            address_line = re.sub(r'\b(ул)\.\s*', r'\1 ', address_line, flags=re.I)
            address_line = address_line.replace('\xa0', ' ')
            address_line = re.sub(r'\s*[•‧∙·]\s*', ', ', address_line)
            address_line = re.sub(r'\s+', ' ', address_line).strip(' ,')

        # город/район
        city, district = self._extract_city_district(soup, text_all, title)
        location = None
        if city and district:
            location = f"{city}, {district}"
        elif city:
            location = city
        elif jl.get("location"):
            location = to_ru(str(jl.get("location")))
        else:
            # последний фоллбек — адрес показываем как локацию
            location = address_line

        # чистим возможные хвосты "ID - 123..."
        def _strip_id(s: Optional[str]) -> Optional[str]:
            if not s:
                return s
            s = re.sub(r'(?i)(?:[,•;:\s-]*\bID\b\s*[-–—:]*\s*,?\s*\d+)', '', s)
            return re.sub(r'\s+', ' ', s).strip(' ,;') or None

        address_line = _strip_id(address_line)
        location = _strip_id(location)

        floor, floors_total = self._parse_floors(text_all)
        try:
            for blob in json_blobs_cap:
                strs = json.dumps(blob, ensure_ascii=False)
                m1 = re.search(r'"(?:floor|floorNumber)"\s*:\s*(\d+)', strs)
                m2 = re.search(r'"(?:totalFloors|numberOfFloors|floorsTotal)"\s*:\s*(\d+)', strs)
                if m1 and not floor:
                    floor = int(m1.group(1))
                if m2 and not floors_total:
                    floors_total = int(m2.group(1))
                if floor and floors_total:
                    break
        except Exception:
            pass

        # --- Photos ---
        gallery_photos = self._extract_gallery_photos(soup)
        photo_urls: List[str] = []
        if gallery_photos:
            photo_urls.extend(gallery_photos)
        else:
            if jl.get("photos"):
                photo_urls.extend([absolutize(u, self.url) for u in jl["photos"]])
            photo_urls.extend(dom_imgs_cap)
            photo_urls.extend(json_imgs_cap)
            og_img = meta(soup, "og:image")
            if og_img:
                photo_urls.append(og_img)
            if not photo_urls:
                for img in soup.find_all("img"):
                    src = img.get("src") or img.get("data-src") or ""
                    if src:
                        photo_urls.append(src)

        def _resolve_next_image(u: str) -> str:
            if "/_next/image" in u:
                try:
                    qs = parse_qs(urlparse(u).query).get("url")
                    if qs:
                        return unquote(qs[0])
                except Exception:
                    pass
            return u

        resolved = [absolutize(_resolve_next_image(u), self.url) for u in photo_urls]
        ordered_photos = uniq_keep_order([
            u for u in resolved
            if (
                    is_good_image_url(u)
                    and not re.search(r'(?:_thumb|_blur|google_map)', u, flags=re.I)
                    and not self._is_ad_image(u)  # <<< добавили
            )
        ])[:10]

        # --- Phones ---
        phones: List[str] = []
        contact_container = self._find_contact_container(soup)

        def _grab_phones_from(node):
            if not node:
                return
            # a) явные ссылки
            for a in node.select('a[href^="tel:"], a[href*="wa.me/"], a[href^="viber:"]'):
                href = a.get('href', '')
                if href:
                    phones.extend(_phones_from_text(href))
            # b) h6 и кнопки (как в вашем примере)
            for el in node.find_all(['h6', 'button', 'span']):
                phones.extend(_phones_from_text(textify(el)))
            if len(phones) < 2:
                phones.extend(_phones_from_text(textify(node)))

        if contact_container:
            _grab_phones_from(contact_container)
        if not phones:
            # по всей странице
            for a in soup.select('a[href^="tel:"], a[href*="wa.me/"], a[href^="viber:"]'):
                href = a.get('href', '')
                if href:
                    phones.extend(_phones_from_text(href))
                anc = a
                for _ in range(3):
                    if not anc: break
                    for el in anc.find_all(['h6', 'button', 'span']):
                        phones.extend(_phones_from_text(textify(el)))
                    anc = anc.parent
        if not phones:
            # описание
            desc_el = soup.select_one('#application_desc')
            if desc_el:
                phones.extend(_phones_from_text(textify(desc_el)))
        if not phones:
            # JSON-ответы
            for blob in json_blobs_cap:
                if not isinstance(blob, dict):
                    continue
                data = blob.get('data') if isinstance(blob.get('data'), dict) else None
                candidates: List[Optional[str]] = []
                if isinstance(data, dict):
                    candidates += [data.get('phone_number'), data.get('phoneNumber'), data.get('phone')]
                candidates += [blob.get('phone_number'), blob.get('phoneNumber'), blob.get('phone')]
                for val in candidates:
                    if val:
                        phones.extend(_phones_from_text(str(val)))
                if phones:
                    break

        # нормализация и фильтр
        normed = []
        for p in uniq_keep_order(phones):
            if not p:
                continue
            d = re.sub(r'\D+', '', p)
            if re.fullmatch(r'5\d{8}', d):  # 598xxxxxx → +995598xxxxxx
                normed.append('+995' + d)
            elif d.startswith('995') and len(d) >= 11:
                normed.append('+' + d)
            else:
                # короткие/маскированные отбрасываем
                if len(d) >= 9:
                    normed.append(p)
        phones = [p for p in uniq_keep_order(normed) if p not in self.PHONE_BLACKLIST][:2]

        # --- build Listing ---
        listing = Listing(
            url=self.url,
            source="home.ss.ge",
            listing_id=listing_id,
            title=title,
            location=location,
            address_line=address_line,
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
