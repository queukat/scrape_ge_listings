#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape Georgian real-estate listings (myhome.ge, home.ss.ge).
Saves structured JSON + downloads photos.

- SS.ge: parse from HTML/JSON-LD/OG reliably (RU/KA/EN), robust address & attributes.
- myhome.ge: headless render via Playwright + JSON capture from network responses
  + DOM gallery extraction; filters out non-gallery banners (e.g., statements.tnet.ge).
- Retries, timeouts, polite headers, minimal dependencies.

Usage:
  pip install httpx[http2] beautifulsoup4 lxml tenacity pydantic playwright
  python -m playwright install chromium

  python scrape_ge_listings.py "<URL>" [-o OUTDIR] [--no-photos] [--headful]

Notes:
  --headful  -> set HEADFUL=1 to debug Playwright (visible browser).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, parse_qs, unquote

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

# ---------- Utils ----------

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

REQ_TIMEOUT = httpx.Timeout(20.0, connect=15.0)
HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "ru-RU,ru;q=0.95,ka;q=0.6,en-US;q=0.5,en;q=0.4",
}
MEASURE_M2_RU_KA_EN = r"(?:м²|m²|кв\.?\s?м|მ²|კვ\.?\s?მ)"

SKIP_IMG_HOST_SUBSTR = ("adocean.pl",)


class ScrapeError(Exception):
    pass

def safe_slug(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[\s/\\:;,\|\[\]\(\)\{\}\<\>\?\"'\!@#\$%\^&\*\+=]+", "_", (s or "").strip())
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
    return re.sub(r"\s+", " ", el.get_text(separator=" ", strip=True)).strip()

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

def is_good_image_url(u: str) -> bool:
    if not re.match(r"^https?://", u, re.I):
        return False
    host = urlparse(u).netloc.lower()
    if any(bad in host for bad in SKIP_IMG_HOST_SUBSTR):
        return False
    # only pick real images
    return bool(re.search(r"\.(?:jpe?g|png|webp)", u, flags=re.I))

async def async_download(urls: List[str], dest_dir: Path, client: httpx.AsyncClient) -> List[str]:
    saved = []
    for idx, u in enumerate(urls, 1):
        try:
            variants = [u]
            # known ss.ge pattern
            if "_Thumb." in u:
                variants = [
                    u.replace("_Thumb.", "_Large."),  # large guess
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
            ext = os.path.splitext(urlparse(final_url).path)[1] or ".jpg"
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
    price: Price = Field(default_factory=Price)
    area_m2: Optional[int] = None
    land_area_m2: Optional[int] = None
    rooms: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    description: Optional[str] = None
    attributes: List[str] = Field(default_factory=list)
    photos: List[str] = Field(default_factory=list)
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

    def _render_with_capture(self) -> Tuple[str, List[str], List[dict]]:
        """
        Render page via Playwright.
        Returns: (html, image_urls_from_dom, json_blobs_from_network)
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except Exception:
            return "", [], []

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
            context = browser.new_context(user_agent=UA, locale="ru-RU", ignore_https_errors=True)
            # capture JSON responses
            def on_response(res):
                ct = (res.headers or {}).get("content-type", "")
                if "application/json" in ct.lower():
                    try:
                        txt = res.text()
                        if txt and len(txt) < 2_000_000:  # safety
                            data = json.loads(txt)
                            json_blobs.append(data)
                    except Exception:
                        pass

            context.on("response", on_response)
            page = context.new_page()
            try:
                page.goto(self.url, wait_until="domcontentloaded", timeout=30000)
                # дождаться появления контента (meta/галерея/JSON-LD)
                try:
                    page.wait_for_selector("img, script[type='application/ld+json'], meta[property='og:title']", timeout=8000)
                except PWTimeout:
                    pass
                page.wait_for_load_state("networkidle", timeout=15000)

                # собрать HTML
                html = page.content()

                # вытащить максимум фоток из DOM-галерей (img + picture/source)
                try:
                    # только крупные/осмысленные картинки
                    imgs = page.evaluate("""
                        () => {
                          const xs = new Set();
                          const push = (u) => { if (u) xs.add(u); };

                          // IMG elements
                          for (const el of Array.from(document.images)) {
                            const src = el.currentSrc || el.src || "";
                            if ((el.naturalWidth||0) >= 512 || /(large|xlarge|big|1024|1280|1920)/i.test(src)) push(src);
                          }
                          // <source srcset> in <picture>
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

        # фильтруем
        dom_imgs = [u for u in dom_imgs if isinstance(u, str) and not u.startswith("data:") and not re.search(r'(?:_thumb|_blur|google_map)', u)]

        # из JSON-вызовов достанем любые ссылки на jpg/png/webp
        json_img_urls: List[str] = []
        for blob in json_blobs:
            strs: List[str] = []
            _walk_collect_str(blob, strs)
            for s in strs:
                if re.search(r'\.(?:jpe?g|png|webp)', s, flags=re.I) and not re.search(r'(?:_thumb|_blur|google_map)', s, flags=re.I):
                    json_img_urls.append(s)

        return html, uniq_keep_order(dom_imgs), uniq_keep_order(json_img_urls)

    def via_playwright(self) -> Optional[str]:
        html, _, _ = self._render_with_capture()
        return html or None

    def extract(self) -> 'Listing':
        raise NotImplementedError

# ---------- SS.GE Extractor ----------

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
                out.setdefault("price_amount", amt)
                out.setdefault("price_currency", cur)
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
        # address line like "пр. Маршала Геловани 1" appears as a separate heading/link
        addr = None
        # 1) try direct pattern
        m = re.search(r"(ул\.?|улица|пр(?:осп\.?|оспект|\.?)|пр-т|пер\.?|переулок|ш\.?|шоссе|пр\.)\s*[^\n,]{2,60}?\d+[А-Яа-яё0-9\/\-]*", text_all, flags=re.I)
        if m:
            addr = m.group(0).strip()

        # 2) check chips / breadcrumbs with city & district
        city = None
        district = None
        chips = [textify(a) for a in soup.find_all(["a","span"]) if a and a.get("href") and len(textify(a)) <= 40]
        for t in chips:
            if re.fullmatch(r"(Тбилиси|თბილისი|Tbilisi)", t, flags=re.I):
                city = "Тбилиси"
            if re.search(r"(Сабуртало|საბურთალო|Saburtalo)", t, flags=re.I):
                district = "Сабуртало"

        # 3) if address appears as a standalone heading/link
        if not addr:
            for tag in soup.find_all(["a","h2","h3","div","span"]):
                s = textify(tag)
                if re.search(r"\b\d{1,4}\b", s) and re.search(r"[А-ЯЁа-яё]", s):
                    if re.search(r"(пр|ул|шос|пер|просп)", s, flags=re.I):
                        addr = s
                        break

        # final compose
        parts = [p for p in [city, district, addr] if p]
        return (", ".join(parts) if parts else None, addr)

    def extract(self) -> Listing:
        html = None
        try:
            html = self.fetch()
        except Exception:
            html = self.via_playwright() or ""
        if not html:
            raise ScrapeError("Failed to load page")

        soup = BeautifulSoup(html, "lxml")
        text_all = textify(soup)

        # JSON-LD / OG
        jlds = pick_json_ld(html)
        jl = self._jsonld_fill(jlds) if jlds else {}

        title = jl.get("title") or textify(soup.find("h1")) or meta(soup, "og:title")
        listing_id = None
        m = re.search(r"\bID\s*[-–]?\s*(\d+)\b", text_all, flags=re.I)
        if m:
            listing_id = m.group(1)

        # Price
        amount = jl.get("price_amount")
        currency = jl.get("price_currency")
        if not (amount and currency):
            og_amt = meta(soup, "product:price:amount")
            og_cur = meta(soup, "product:price:currency")
            if og_amt and not amount:
                amount = extract_first_number(og_amt)
            if og_cur and not currency:
                currency = og_cur
        if not amount:
            m = re.search(rf"([\d\s,\.]+)\s*(₾|\$|€|GEL|USD|EUR)", text_all, flags=re.I)
            if m:
                amount, currency = normalize_price(m.group(0))

        # Description
        desc = None
        for h in soup.find_all(["h2","h3","div","span"]):
            t = textify(h).lower()
            if t in ("описание","აღწერა","description"):
                nxt = h.find_next(lambda x: x and x.name in ("p","div") and len(textify(x)) > 40)
                if nxt:
                    desc = textify(nxt)
                    break
        if not desc:
            od = meta(soup, "og:description")
            if od and len(od) > 50:
                desc = od.strip()

        # Attributes
        attributes = set()
        for kw in (self.RU_FEATURES + self.KA_FEATURES + self.EN_FEATURES):
            if re.search(rf"\b{re.escape(kw)}\b", text_all, flags=re.I):
                attributes.add(kw)

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
            r"(?:Комнат[ыа]|Комнаты)\s*[:\-–]?\s*(\d+)",
            r"(\d+)\s*room\b",
            r"(\d+)\s*ოთახ"
        ])
        bedrooms = find_int([
            r"Спальн\w*\s*[:\-–]?\s*(\d+)",
            r"(\d+)\s*bed\b",
            r"(\d+)\s*საძინებ\w*"
        ])
        bathrooms = find_int([
            r"Санузел\w*\s*[:\-–]?\s*(\d+)",
            r"(\d+)\s*bath"
        ])

        def area_by_labels(labels: List[str]) -> Optional[int]:
            for lb in labels:
                m = re.search(rf"{lb}\s*[:\-–]?\s*(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I)
                if m:
                    return extract_first_number(m.group(1))
            return None

        area_m2 = jl.get("area_m2") or area_by_labels([
            "Площадь дома","Площадь квартиры","Общая площадь","Жилая площадь",
            "House area","Apartment area","Total area",
            "სახლის ფართი","ბინის ფართი","საერთო ფართი"
        ])
        land_area_m2 = area_by_labels([
            "Площадь участка","Площадь земли","Участок",
            "Land area","Lot area",
            "ეზოს ფართი"
        ])
        if area_m2 is None or land_area_m2 is None:
            ms = re.findall(rf"(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I)
            nums = []
            for x in ms:
                n = extract_first_number(x)
                if n and n not in nums:
                    nums.append(n)
            if len(nums) >= 2:
                small, big = min(nums), max(nums)
                if area_m2 is None:
                    area_m2 = small
                if land_area_m2 is None and big >= small:
                    land_area_m2 = big
            elif len(nums) == 1 and area_m2 is None:
                area_m2 = nums[0]

        # Location
        location, _addr = self._extract_address(soup, text_all)
        if not location:
            location = jl.get("location")

        # Photos
        photo_urls: List[str] = []
        if jl.get("photos"):
            photo_urls.extend([absolutize(u, self.url) for u in jl["photos"]])
        if not photo_urls:
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                if src and "ss.ge" in urlparse(absolutize(src, self.url)).netloc:
                    photo_urls.append(absolutize(src, self.url))
        ordered_photos = uniq_keep_order([u for u in photo_urls if is_good_image_url(u)])

        listing = Listing(
            url=self.url,
            source="home.ss.ge",
            listing_id=listing_id,
            title=title,
            location=location,
            price=Price(amount=amount, currency=currency),
            area_m2=area_m2,
            land_area_m2=land_area_m2,
            rooms=rooms,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            description=desc,
            attributes=sorted(attributes),
            raw_meta={"json_ld": jlds},
        )
        listing.photos = ordered_photos
        return listing

# ---------- MyHome.ge Extractor ----------

class MyHomeExtractor(BaseExtractor):
    """Extracts from myhome.ge pages (JS-heavy SPA)."""

    def extract(self) -> Listing:
        # 1) сначала попытка HTTP
        html = ""
        try:
            html = self.fetch()
        except Exception:
            pass

        captured_dom_imgs: List[str] = []
        captured_json_imgs: List[str] = []
        html2, dom_imgs, json_imgs = self._render_with_capture()
        if html2:
            html = html2
        captured_dom_imgs = dom_imgs
        captured_json_imgs = json_imgs

        if not html:
            raise ScrapeError("Failed to load page")

        soup = BeautifulSoup(html, "lxml")
        text_all = textify(soup)

        # JSON-LD first
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

        nd_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        if nd_tag:
            try:
                nd = json.loads(nd_tag.string or nd_tag.text or '{}')
                imgs = nd.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [{}])[0].get('state', {}).get('data', {}).get('data', {}).get('statement', {}).get('images', [])
                for img in imgs:
                    u = img.get('large') or img.get('thumb')
                    if u:
                        photos.append(u)
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
        if not price_amount:
            m = re.search(rf"([\d\s,\.]+)\s*(₾|\$|€|GEL|USD|EUR)", text_all, flags=re.I)
            if m:
                price_amount, price_currency = normalize_price(m.group(0))

        # ID from URL (/pr/<id>)
        path = urlparse(self.url).path
        listing_id = None
        mi = re.search(r"/pr/(\d+)", path)
        if mi:
            listing_id = mi.group(1)

        # rooms / bedrooms & areas
        rooms = None
        bedrooms = None
        area_m2 = None
        land_area_m2 = None

        m = re.search(r"(\d+)\s*(room|ოთახ)", text_all, flags=re.I)
        rooms = int(m.group(1)) if m else None
        m = re.search(r"(\d+)\s*(bed|საძინებ\w*)", text_all, flags=re.I)
        bedrooms = int(m.group(1)) if m else None

        def area_by_labels(labels: List[str]) -> Optional[int]:
            for lb in labels:
                m = re.search(rf"{lb}\s*[:\-–]?\s*(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I)
                if m:
                    return extract_first_number(m.group(1))
            return None

        area_m2 = area_by_labels(["სახლის ფართი","ბინის ფართი","საერთო ფართი",
                                  "Площадь дома","Площадь квартиры","Общая площадь",
                                  "House area","Apartment area","Total area"])
        land_area_m2 = area_by_labels(["ეზოს ფართი","Площадь участка","Land area","Lot area"])

        if area_m2 is None or land_area_m2 is None:
            ms = re.findall(rf"(\d[\d\s,\.]*)\s*{MEASURE_M2_RU_KA_EN}", text_all, flags=re.I)
            nums = []
            for x in ms:
                n = extract_first_number(x)
                if n and n not in nums:
                    nums.append(n)
            if nums:
                small, big = min(nums), max(nums)
                if area_m2 is None:
                    area_m2 = small
                if land_area_m2 is None and big >= small:
                    land_area_m2 = big

        # Location: breadcrumbs or title hint
        if not location and title:
            m = re.search(r"(საბურთალო|ვაკე|დიდი დიღომი|ავლაბარი|ბათუმი|Saguramo|საგურამო|Saburtalo|Vake|Tbilisi)", title, flags=re.I)
            if m:
                # нормализуем популярные
                g = m.group(1)
                if re.search(r"საბურთალო|Saburtalo", g, flags=re.I):
                    location = "საბურთალო"
                else:
                    location = g

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

        def _resolve_next_image(u: str) -> str:
            if "/_next/image" in u:
                try:
                    qs = parse_qs(urlparse(u).query).get("url")
                    if qs:
                        return unquote(qs[0])
                except Exception:
                    pass
            return u

        resolved = [absolutize(_resolve_next_image(u), self.url) for u in cand]
        photos_ord = uniq_keep_order([u for u in resolved if is_good_image_url(u) and not re.search(r'(?:_thumb|_blur|google_map)', u)])

        return Listing(
            url=self.url,
            source="myhome.ge",
            listing_id=listing_id,
            title=title,
            location=location,
            price=Price(amount=price_amount, currency=price_currency),
            area_m2=area_m2,
            land_area_m2=land_area_m2,
            rooms=rooms,
            bedrooms=bedrooms,
            description=meta(soup, "og:description") or None,
            attributes=[],
            photos=photos_ord,
            raw_meta={"json_ld": jlds},
        )

# ---------- Router / Runner ----------

def get_extractor(url: str) -> BaseExtractor:
    host = urlparse(url).netloc.lower()
    if "ss.ge" in host:
        return SSGeExtractor(url)
    if "myhome.ge" in host:
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

def save_listing(outdir: Path, listing: Listing) -> Path:
    lid = listing.listing_id or safe_slug(listing.title or "listing")
    base = outdir / safe_slug(urlparse(listing.url).netloc) / (lid or "item")
    ensure_dir(base)
    (base / "listing.json").write_text(json.dumps(listing.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return base

async def maybe_download_photos(base_dir: Path, listing: Listing, download: bool) -> None:
    if not download or not listing.photos:
        return
    dest = base_dir / "photos"
    extractor = get_extractor(listing.url)
    saved = await extractor.fetch_images(listing.photos, dest)
    (base_dir / "saved_photos.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape myhome.ge / ss.ge listing to JSON + photos.")
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

            if not args.no_photos:
                asyncio.run(maybe_download_photos(base_dir, listing, download=True))
                print(f"[OK] Photos → {base_dir/'photos'} (if available)")
        except Exception as e:
            rc = 1
            print(f"[FAIL] {url} → {e}", file=sys.stderr)
    return rc

if __name__ == "__main__":
    sys.exit(main())
