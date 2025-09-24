from __future__ import annotations

import json
import os
import re
from html import escape as html_escape
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

# --------- Config / constants ---------

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

HTTP_TIMEOUT_S = float(os.getenv("GE_HTTP_TIMEOUT", "45.0"))
HTTP_CONNECT_TIMEOUT_S = float(os.getenv("GE_HTTP_CONNECT_TIMEOUT", "30.0"))
PW_TIMEOUT_MS = int(os.getenv("GE_PW_TIMEOUT_MS", "60000"))

REQ_TIMEOUT = httpx.Timeout(HTTP_TIMEOUT_S, connect=HTTP_CONNECT_TIMEOUT_S)
HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "ru-RU,ru;q=0.95,ka;q=0.6,en-US;q=0.5,en;q=0.4",
}
try:
    import h2  # type: ignore
    HTTP2_AVAILABLE = True
except Exception:
    HTTP2_AVAILABLE = False

MEASURE_M2_RU_KA_EN = r"(?:м²|m²|кв\.?\s?м|მ²|კვ\.?\s?მ)"
SKIP_IMG_HOST_SUBSTR = ("adocean.pl",)


class ScrapeError(Exception):
    pass


# --------- Utils ---------

def ensure_dir(p) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_slug(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[\s/\\:;,\|\[\]\(\)\{\}\<\>\?\"'\!@#\$%\^&\*\=]+", "_", (s or "").strip())
    return (s or "item")[:max_len].strip("_") or "item"


def textify(el) -> str:
    if not el:
        return ""
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
    seen = set();
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x);
            out.append(x)
    return out


def uniq_keep_order_any(seq: Iterable[Any]) -> List[Any]:
    seen = set();
    out = []
    for x in seq:
        key = json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else x
        if key not in seen:
            seen.add(key);
            out.append(x)
    return out


def is_good_image_url(u: str) -> bool:
    if not re.match(r"^https?://", u, re.I):
        return False
    host = urlparse(u).netloc.lower()
    if any(bad in host for bad in SKIP_IMG_HOST_SUBSTR):
        return False
    return bool(re.search(r"\.(?:jpe?g|png|webp)(?:\?.*)?$", u, flags=re.I))


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


# --------- Phones ---------

_RE_PHONE = re.compile(
    r"(?:\+?995[\s\-]?)?(?:0[\s\-]?)?(?:\(?5\d{2}\)?[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}|"
    r"\(?3\d\)?[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2,3})"
)


def _normalize_ge_phone(s: str) -> Optional[str]:
    if not s: return None
    digits = re.sub(r"\D+", "", s)
    if not digits:
        return None
    if digits.startswith("995"):
        norm = digits
    else:
        if digits.startswith("5") and len(digits) == 9:
            norm = "995" + digits
        elif digits.startswith("0") and len(digits) >= 10:
            norm = "995" + digits[1:]
        elif (digits.startswith("32") or digits.startswith("3")) and len(digits) >= 9:
            norm = "995" + digits
        else:
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


# --------- Translation ---------

_RE_GEORGIAN = re.compile(r"[ა-ჰ]")
_RE_CYRILLIC = re.compile(r"[А-ЯЁа-яё]")

try:
    from deep_translator import GoogleTranslator  # type: ignore

    _gt = GoogleTranslator(source="auto", target="ru")
    _GT_OK = True
except Exception:
    _GT_OK = False
    _gt = None


def to_ru(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    # Переводим, если строка НЕ на русском (нет кириллицы)
    if _GT_OK and not _RE_CYRILLIC.search(text):
        try:
            return _gt.translate(text)  # type: ignore
        except Exception:
            return text
    return text


# --------- Areas ---------

def find_m2_values(text: str) -> List[int]:
    pat = rf"(?<!\d)(\d+(?:[.,]\d+)?)\s*{MEASURE_M2_RU_KA_EN}"
    vals = []
    for s in re.findall(pat, text, flags=re.I):
        try:
            vals.append(int(float(s.replace(",", ".").strip())))
        except Exception:
            pass
    return vals


def parse_ru_ka_address_from_text(text_all: str) -> Optional[str]:
    # GE: «… ქუჩა 37», «… მოედანი 24» (мოედანი = площадь)
    m = re.search(
        r"([ა-ჰ][ა-ჰ'’\-\s]+?)\s+(ქუჩა|გამზირი|პროსპექტი?|მოედანი)\s*(\d+[ა-ჰA-Za-z\-]?)\b(?!\s*\d)",
        text_all, flags=re.I
    )
    if m:
        return f"{m.group(1).strip()} {m.group(2)} {m.group(3)}"

    # RU: «ул./улица/пр-т/проспект/пл./площадь/бул./бульвар/ал./аллея/кв-л/квартал … 37»
    m = re.search(
        r"(?:ул\.?|улица|просп(?:\.|ект)?|пр-т|пер\.?|переулок|ш(?:\.|оссе)?|"
        r"пл\.?|площадь|бул\.?|бульвар|ал\.?|аллея|кв-л|квартал)"
        r"\s*[А-ЯЁа-яё\.\- ]+?\s*(\d+[А-Яа-яA-Za-z\-\/]?)\b(?!\s*\d)",
        text_all, flags=re.I
    )
    return m.group(0).strip() if m else None


# --------- Summary helpers ---------

RU_MAP = {
    "საბურთალო": "Сабуртало",
    "ვაკე": "Ваке",
    "Тбилиси": "Тбилиси",
    "თბილისი": "Тбилиси",
    "Saguramo": "Сагурамо",
    "საგურამო": "Сагурамо",
}


def _sym(cur: Optional[str]) -> str:
    return {"USD": "$", "GEL": "₾", "EUR": "€"}.get((cur or "").upper(), "")


def esc(s: Optional[str]) -> Optional[str]:
    return html_escape(s, quote=False) if s else None


def build_ru_summary(listing) -> str:
    import re

    sym = {"USD": "$", "EUR": "€", "GEL": "₾"}

    def ru_plural(n: int, forms: tuple[str, str, str]) -> str:
        n = abs(int(n))
        if n % 10 == 1 and n % 100 != 11:
            return forms[0]
        if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            return forms[1]
        return forms[2]

    def norm(s: str | None) -> str | None:
        if not s:
            return None
        s = s.replace("\xa0", " ")
        # единообразные аббревиатуры: «ул.»/«пл.»/«просп.»/«бул.»/«пер.» -> без точки
        s = re.sub(r"\b(ул|пл|просп|бул|пер)\.\s*", r"\1 ", s, flags=re.I)
        s = re.sub(r"\s*[•‧∙·]\s*", " ", s)  # буллеты -> пробел
        s = re.sub(r"\s+", " ", s).strip(" ,")
        return s

    def clean_location(loc: str | None, addr: str | None) -> str | None:
        loc = norm(loc)
        addr_n = norm(addr)
        if not loc:
            return None
        # 1) если адрес содержится в location — удаляем его
        if addr_n:
            try:
                loc = re.sub(re.escape(addr_n), "", loc, flags=re.I).strip(" ,")
            except re.error:
                pass
        # 2) отрезаем «улицу/проспект/переулок…» с номером, если остались
        loc = re.sub(
            r"(?:,?\s*)(?:ул\.?|улица|пр(?:осп\.?|оспект|\.?)|пр-т|пер\.?|переулок|ш\.?|шоссе|пр\.|st(?:reet)?|ave(?:nue)?|str\.)\b.*$",
            "",
            loc,
            flags=re.I,
        ).strip(" ,")
        loc = re.sub(r"\s+", " ", loc)
        return loc or None

    # 1) строка с действием и ценой
    price = listing.price
    price_s = (
        f"{sym.get((price.currency or '').upper(), price.currency or '')}{int(float(price.amount))}"
        if price and price.amount is not None else None
    )
    title = (listing.title or "")
    action = (
        "Аренда" if re.search(r"\b(аренда|сда[её]тся|rent)\b", title, re.I)
        else ("Продажа" if re.search(r"\b(продажа|прода[её]тся|sale)\b", title, re.I) else None)
    )
    line1_raw = " - ".join([x for x in [action, f"Цена {price_s}" if price_s else None] if x]) or (
        f"Цена {price_s}" if price_s else "").strip()
    line1 = f"<b>{esc(line1_raw)}</b>" if line1_raw else ""

    # 2) локация и адрес
    addr = norm(listing.address_line)
    loc = clean_location(listing.location, addr)

    # 3) комнаты / спальни с правильными формами
    parts_rooms = []
    if listing.rooms:
        parts_rooms.append(f"{listing.rooms} " + ru_plural(listing.rooms, ("Комната", "Комнаты", "Комнат")))
    if listing.bedrooms:
        parts_rooms.append(f"{listing.bedrooms} " + ru_plural(listing.bedrooms, ("Спальня", "Спальни", "Спален")))
    rooms_line = ", ".join(parts_rooms) if parts_rooms else None

    # 4) этажность и площадь
    floors = (
        f"{listing.floor}/{listing.floors_total} Этажей" if (listing.floor and listing.floors_total)
        else (f"{listing.floor} Этаж" if listing.floor else (
            f"{listing.floors_total} Этажей" if listing.floors_total else None))
    )
    area = f"{int(listing.area_m2)} м²" if listing.area_m2 else None
    last = ", ".join([x for x in [floors, area] if x]) or None

    return "\n".join([x for x in [line1 or None, loc, addr, rooms_line, last] if x])


def force_myhome_ru(url: str) -> str:
    """
    Приводит ссылку к русской версии для:
      - myhome.ge
      - home.ss.ge
      - ss.ge
    Логика совпадает с ensure_ru_myhome в боте.
    """
    p = urlparse(url)
    host = p.netloc.lower()
    if ("myhome.ge" not in host) and ("home.ss.ge" not in host) and ("ss.ge" not in host):
        return url

    path = p.path or "/"
    if path.startswith("/ru/"):
        new_path = path
    elif path.startswith(("/ka/", "/en/", "/az/", "/am/")):
        # /ka/xxx -> /ru/xxx
        parts = path.split("/", 2)
        rest = parts[2] if len(parts) >= 3 else ""
        new_path = f"/ru/{rest}".rstrip("/")
        if rest and not rest.endswith("/") and "." not in rest.split("/")[-1]:
            new_path += "/"
    else:
        new_path = "/ru" + path if not path.startswith("/ru") else path

    new_path = re.sub(r"/{2,}", "/", new_path)
    return urlunparse((p.scheme, p.netloc, new_path, p.params, p.query, p.fragment))

# --------- Address translation & translit fallback ---------

# Базовые замены типа улиц/сокращений
_KA_ADDR_REPL = (
    (r"(^|[\s,])ქ(?:\.)?\s*", r"\1ул. "),
    (r"\bქუჩა\b", "улица"),
    (r"\bგამზირი\b", "проспект"),
    (r"\bგამზ\.\b", "просп."),
    (r"\bბულვარი\b", "бульвар"),
    (r"\bჩიხი\b", "переулок"),
    (r"\bკვარტალი\b", "квартал"),
)

# Побуквенная транслитерация грузинского -> ру (приближенно, для адресов достаточно)
_KA_TO_RU = {
    "ა": "а", "ბ": "б", "გ": "г", "დ": "д", "ე": "е", "ვ": "в", "ზ": "з", "თ": "т", "ი": "и", "კ": "к",
    "ლ": "л", "მ": "м", "ნ": "н", "ო": "о", "პ": "п", "ჟ": "ж", "რ": "р", "ს": "с", "ტ": "т", "უ": "у", "ფ": "п",
    "ქ": "к", "ღ": "г", "ყ": "к", "შ": "ш", "ჩ": "ч", "ც": "ц", "ძ": "дз", "წ": "ц", "ჭ": "ч", "ხ": "х", "ჯ": "дж",
    "ჰ": "х",
    " ": " ", "-": "-", ".": ".", ",": ",", "’": "’", "'": "'", "/": "/"
}


def _ka_to_ru_translit(s: str) -> str:
    res = "".join(_KA_TO_RU.get(ch, ch) for ch in s)
    # Типичные родовые суффиксы в фамилиях/топонимах
    res = re.sub(r"дзе(?:ს|ის)\b", "дзе", res, flags=re.I)
    res = re.sub(r"швили(?:ს|ის)\b", "швили", res, flags=re.I)
    return res


def to_ru_addr(text: Optional[str]) -> Optional[str]:
    """
    Перевод/нормализация адресной строки:
    1) Пытаемся машинный перевод (если доступен).
    2) Если не вышло — локальные замены + транслитерация грузинского -> ру.
    3) Приводим порядок: "Кавтарадзе ул. 3" -> "ул. Кавтарадзе 3".
    """
    if not text:
        return text
    s = text.strip()
    if not s:
        return s

    # 1) Пытаемся машинный перевод, если нет кириллицы
    if _GT_OK and not _RE_CYRILLIC.search(s):
        try:
            tr = _gt.translate(s)  # type: ignore
            if isinstance(tr, str) and _RE_CYRILLIC.search(tr):
                return re.sub(r"\s+", " ", tr).strip()
        except Exception:
            pass

    # 2) Локальные замены адресных маркеров
    out = s
    for pat, repl in _KA_ADDR_REPL:
        out = re.sub(pat, repl, out, flags=re.I)

    # 3) Транслитерация грузинских букв -> русские аналоги
    if _RE_GEORGIAN.search(out):
        out = _ka_to_ru_translit(out)

    out = re.sub(r"\s+", " ", out).strip()

    # 4) Порядок: "{Название} (ул.|проспект|...)\s*{дом}" -> "{ул.|проспект} {Название} {дом}"
    out = re.sub(
        r"^\s*([^,\d]+?)\s+(ул\.|улица|проспект|просп\.|бульвар|переулок|квартал)\s*(\d+[^\s,]*)\s*$",
        lambda m: f"{m.group(2)} {m.group(1).strip()} {m.group(3)}",
        out, flags=re.I
    )

    # Финальный трим
    return out.strip()
