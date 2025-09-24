#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx  # noqa: E402

from ge_listings.base import BaseExtractor
from ge_listings.common import (
    ScrapeError, ensure_dir, safe_slug, build_ru_summary,
    force_myhome_ru,
)
from ge_listings.models import Listing
from ge_listings.myhome import MyHomeExtractor
from ge_listings.ss_ge import SSGeExtractor
# 👉 очистка водяных знаков (ss.ge / myhome.ge)
#    использует IOPaint HTTP API (если доступен) или локальный cv2.inpaint (Telea)
from ge_listings.wm_clean import should_clean, clean_image_bytes  # noqa: E402


def get_extractor(url: str) -> BaseExtractor:
    # всегда приводим к RU-версии для поддерживаемых хостов
    url = force_myhome_ru(url)
    host = urlparse(url).netloc.lower()
    if "ss.ge" in host:
        return SSGeExtractor(url)
    if "myhome.ge" in host or "home.ss.ge" in host:
        url = force_myhome_ru(url)  # всегда русская версия
        return MyHomeExtractor(url)
    raise ScrapeError(f"Unsupported host: {host}")


def load_urls(arg: str) -> list[str]:
    if re.match(r"^https?://", arg.strip(), re.I):
        return [arg.strip()]
    p = Path(arg)
    if not p.exists():
        raise ScrapeError(f"No such file: {arg}")
    urls: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not re.match(r"^https?://", line, re.I):
            continue
        urls.append(line)
    if not urls:
        raise ScrapeError("No valid URLs in file")
    return urls


def save_listing(outdir: Path, listing: Listing) -> Path:
    listing.summary_ru = build_ru_summary(listing)
    lid = listing.listing_id or safe_slug(listing.title or "listing")
    base = outdir / safe_slug(urlparse(listing.url).netloc) / (lid or "item")
    ensure_dir(base)
    # pydantic v2
    (base / "listing.json").write_text(
        json.dumps(listing.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (base / "summary_ru.txt").write_text(listing.summary_ru or "", encoding="utf-8")
    return base


async def _maybe_clean_downloaded(dest: Path, listing: Listing, saved_names: list[str]) -> None:
    """
    Пост‑обработка скачанных картинок:
      - для ss.ge → нижняя левая зона (полоски/логотип)
      - для myhome.ge → центр (полупрозрачный логотип)
    Попытка через IOPaint, иначе локальный OpenCV Telea.
    """
    if not should_clean(listing.source):
        return
    if not saved_names:
        return
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "ListingScraper/1.0"}) as ac:
        for name in saved_names:
            p = dest / name
            if not p.exists() or p.stat().st_size == 0:
                continue
            try:
                original = p.read_bytes()
                cleaned = await clean_image_bytes(original, listing.source, ac)
                if cleaned:
                    p.write_bytes(cleaned)
            except Exception:
                # не ломаем пайплайн, просто пропускаем конкретный файл
                continue


async def maybe_download_photos(base_dir: Path, listing: Listing, download: bool, *, clean: bool = True) -> None:
    if not download or not listing.photos:
        return
    dest = base_dir / "photos"
    listing.photos = listing.photos[:10]
    extractor = get_extractor(listing.url)
    saved = await extractor.fetch_images(listing.photos, dest)
    (base_dir / "saved_photos.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")

    # пост‑обработка: очистка водяных знаков (если включено)
    if clean:
        await _maybe_clean_downloaded(dest, listing, saved)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scrape myhome.ge / ss.ge to JSON + photos (with WebP→JPEG) + RU summary + optional watermark cleaning."
    )
    ap.add_argument("input", help="URL or path to a text file with URLs (one per line)")
    ap.add_argument("-o", "--outdir", default="out", help="Output directory (default: ./out)")
    ap.add_argument("--no-photos", action="store_true", help="Do not download photos")
    ap.add_argument("--no-clean", action="store_true", help="Do not remove watermarks (ss.ge/myhome)")
    ap.add_argument("--headful", action="store_true", help="Run browser in visible mode for debugging")
    args = ap.parse_args()

    if args.headful:
        os.environ["HEADFUL"] = "1"
        # Включаем Playwright-рендер для SPA (myhome.ge) даже в CLI
        os.environ.setdefault("GE_NEED_PHONES", "1")

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
            try:
                listing = extractor.extract()
                if not listing.phones and os.getenv("GE_NEED_PHONES", "0") != "1":
                    # один повтор «с рендером», чтобы перехватить JSON с телефоном
                    os.environ["GE_NEED_PHONES"] = "1"
                    listing = extractor.extract()
            except Exception as e:
                # Авто-фолбэк: повторить с рендером через Playwright (разрешает динамику/CF)
                if os.getenv("GE_NEED_PHONES") != "1":
                    os.environ["GE_NEED_PHONES"] = "1"
                    listing = extractor.extract()
                else:
                    raise

            base_dir = save_listing(outdir, listing)
            print(f"[OK] Saved JSON → {base_dir / 'listing.json'}")

            # NEW: читаемый phone‑debug при GE_DEBUG=1
            if os.getenv("GE_DEBUG", "0") != "0":
                try:
                    dbg = listing.raw_meta.get("debug", {}).get("phone", {})
                    if dbg:
                        print("----- DEBUG phone -----")
                        import json as _json
                        print(_json.dumps(dbg, ensure_ascii=False, indent=2)[:2000])
                except Exception:
                    pass

            if listing.summary_ru:
                print("----- RU Summary -----")
                print(listing.summary_ru)

            if not args.no_photos:
                asyncio.run(maybe_download_photos(base_dir, listing, download=True, clean=not args.no_clean))
                print(f"[OK] Photos → {base_dir / 'photos'} (clean={'off' if args.no_clean else 'on'})")
        except Exception as e:
            rc = 1
            print(f"[FAIL] {url} → {e}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
