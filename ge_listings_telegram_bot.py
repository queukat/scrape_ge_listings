# ge_listings_telegram_bot.py
# -*- coding: utf-8 -*-
"""
Телеграм-бот для объявлений (myhome.ge, home.ss.ge).

Новое:
- /status: сводка за день/неделю/месяц (по posted_at).
- На повторную ссылку парсинг не запускается: бот сначала
  показывает карточку с кнопками и предлагает удалить из группы.
  Парсинг делаем только если оператор нажал «Отправить» или «Редактировать».

Зависимости:
  pip install aiogram>=3 httpx[http2] pillow
  # скрапер рядом: scrape_ge_listings.py
Переменные окружения:
  BOT_TOKEN, TARGET_GROUP_ID, DB_PATH (опц.)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import httpx
from PIL import Image

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    InlineKeyboardMarkup,
    CallbackQuery,
    Message,
    InputMediaPhoto,
    BufferedInputFile,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.exceptions import TelegramBadRequest

# ваш скрапер
from scrape_ge_listings import get_extractor  # type: ignore


BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TARGET_GROUP_ID = int(os.getenv("TARGET_GROUP_ID", "0"))
DB_PATH = os.getenv("DB_PATH", "bot_data.sqlite")
ALBUM_GET_TIMEOUT = float(os.getenv("BOT_ALBUM_GET_TIMEOUT", "60.0"))


# ============== БД ==============

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with closing(_db()) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS postings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                listing_id TEXT,
                canonical_url TEXT NOT NULL,
                title TEXT,
                summary_ru TEXT,
                created_at INTEGER NOT NULL,
                posted_at INTEGER,
                deleted_at INTEGER,
                chat_id INTEGER,
                message_ids TEXT,
                status TEXT NOT NULL DEFAULT 'draft'
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_posting_key
              ON postings(source, listing_id, canonical_url);
            """
        )
        # мягкая миграция: убеждаемся, что колонки есть (если база старая)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(postings)")}
        for need in ("posted_at", "deleted_at"):
            if need not in cols:
                conn.execute(f"ALTER TABLE postings ADD COLUMN {need} INTEGER")

def db_find_by_keys(source: str, listing_id: Optional[str], canonical_url: str) -> Optional[sqlite3.Row]:
    q = """
      SELECT * FROM postings
      WHERE (source=? AND listing_id IS ? AND canonical_url=?)
         OR (source=? AND listing_id=?)
         OR (canonical_url=?)
      ORDER BY id DESC LIMIT 1
    """
    with closing(_db()) as conn:
        cur = conn.execute(q, (source, listing_id, canonical_url, source, listing_id, canonical_url))
        return cur.fetchone()

def db_insert_draft(info: Dict[str, Any]) -> int:
    with closing(_db()) as conn, conn:
        cur = conn.execute(
            """INSERT INTO postings(source, listing_id, canonical_url, title, summary_ru, created_at, status)
               VALUES(?,?,?,?,?,?, 'draft')""",
            (
                info["source"], info.get("listing_id"), info["canonical_url"],
                info.get("title"), info.get("summary_ru"),
                int(datetime.now(tz=timezone.utc).timestamp()),
            ),
        )
        return int(cur.lastrowid)

def db_update_brief(row_id: int, title: Optional[str], summary_ru: Optional[str]) -> None:
    with closing(_db()) as conn, conn:
        conn.execute(
            "UPDATE postings SET title=COALESCE(?, title), summary_ru=COALESCE(?, summary_ru) WHERE id=?",
            (title, summary_ru, row_id),
        )

def db_mark_posted(row_id: int, chat_id: int, message_ids: List[int]) -> None:
    with closing(_db()) as conn, conn:
        now = int(datetime.now(tz=timezone.utc).timestamp())
        conn.execute(
            "UPDATE postings SET status='posted', posted_at=?, chat_id=?, message_ids=? WHERE id=?",
            (now, chat_id, json.dumps(message_ids, ensure_ascii=False), row_id),
        )

def db_mark_deleted(row_id: int) -> None:
    with closing(_db()) as conn, conn:
        now = int(datetime.now(tz=timezone.utc).timestamp())
        conn.execute("UPDATE postings SET status='deleted', deleted_at=? WHERE id=?", (now, row_id))

def db_recent(limit: int = 10) -> List[sqlite3.Row]:
    with closing(_db()) as conn:
        cur = conn.execute("SELECT * FROM postings ORDER BY id DESC LIMIT ?", (limit,))
        return cur.fetchall()

def db_stats_window(since_ts: int) -> Dict[str, int]:
    """Возвращает counts для статусов в интервале [since, now] по дате posted_at."""
    with closing(_db()) as conn:
        # опубликовано
        posted = conn.execute(
            "SELECT COUNT(*) FROM postings WHERE status='posted' AND posted_at>=?",
            (since_ts,),
        ).fetchone()[0]
        # удалено
        deleted = conn.execute(
            "SELECT COUNT(*) FROM postings WHERE status='deleted' AND deleted_at>=?",
            (since_ts,),
        ).fetchone()[0]
        # черновики (созданы за окно и не опубликованы)
        drafts = conn.execute(
            "SELECT COUNT(*) FROM postings WHERE status='draft' AND created_at>=?",
            (since_ts,),
        ).fetchone()[0]
    return {"posted": posted, "deleted": deleted, "drafts": drafts}


# ============== Вспомогательные ==============

def ensure_ru_myhome(url: str) -> str:
    """Нормализует ссылки myhome.ge и home.ss.ge на русскую версию."""
    u = urlsplit(url)
    host = u.netloc
    if "myhome.ge" in host or "home.ss.ge" in host:
        path = u.path
        if not path.startswith("/ru/"):
            if path.startswith(("/ka/", "/en/", "/az/", "/am/")):
                path = "/ru/" + path.split("/", 2)[2]
            else:
                path = "/ru" + path
        path = re.sub(r"/{2,}", "/", path)
        return urlunsplit((u.scheme, host, path, u.query, u.fragment))
    return url

def pick_first_url(text: str) -> Optional[str]:
    m = re.search(r"https?://[^\s]+", text or "")
    return m.group(0) if m else None

def currency_symbol(cur: Optional[str]) -> str:
    if not cur:
        return ""
    return {"USD": "$", "EUR": "€", "GEL": "₾"}.get(cur.upper(), cur)

def build_summary_ru(data: Dict[str, Any]) -> str:
    rooms = data.get("rooms")
    beds = data.get("bedrooms")
    floor = data.get("floor")
    floors = data.get("floors_total")
    area = data.get("area_m2")
    price = data.get("price", {}) or {}
    loc = data.get("location") or data.get("address_line") or ""
    cur = price.get("currency")
    amt = price.get("amount")
    head = []
    if amt:
        head.append(f"Аренда {currency_symbol(cur)}{amt}")
    if loc:
        head.append(str(loc))
    lines = [" – ".join(head) if head else "Аренда"]
    if data.get("address_line"):
        lines.append(str(data["address_line"]))
    second = []
    if rooms:
        second.append(f"{rooms} Комнаты")
    if beds:
        second.append(f"{beds} Спальня")
    if second:
        lines.append(", ".join(second))
    third = []
    if floor and floors:
        third.append(f"{floor}/{floors} Этажей")
    elif floor:
        third.append(f"{floor} этаж")
    if area:
        third.append(f"{area} м²")
    if third:
        lines.append(", ".join(third))
    return "\n".join(lines)

def quick_source_and_id(url: str) -> Tuple[str, Optional[str]]:
    """Без парсинга страницы пытаемся извлечь (source, listing_id) из URL."""
    u = urlsplit(url)
    host = u.netloc.lower()
    if "myhome.ge" in host:
        m = re.search(r"/pr/(\d+)", u.path)
        return ("myhome.ge", m.group(1) if m else None)
    if "ss.ge" in host:
        m = re.search(r"-(\d{6,})/?$", u.path)
        return ("home.ss.ge", m.group(1) if m else None)
    return (host, None)

async def download_and_prepare_album(urls: List[str], client: httpx.AsyncClient, cap_text: Optional[str]) -> List[InputMediaPhoto]:
    media: List[InputMediaPhoto] = []
    urls = urls[:10]  # телега не принимает >10
    for i, u in enumerate(urls):
        try:
            r = await client.get(u, timeout=ALBUM_GET_TIMEOUT, follow_redirects=True)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            content = r.content
            if ".webp" in u.lower() or "image/webp" in ctype.lower():
                im = Image.open(BytesIO(content)).convert("RGB")
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=90)
                buf.seek(0)
                file = BufferedInputFile(buf.read(), filename=f"img_{i+1:02d}.jpg")
            else:
                file = BufferedInputFile(content, filename=f"img_{i+1:02d}.jpg")
            if i == 0:
                cap = (cap_text or "")[:1024]
                media.append(InputMediaPhoto(media=file, caption=cap, parse_mode=ParseMode.HTML))
            else:
                media.append(InputMediaPhoto(media=file))
        except Exception:
            continue
    return media

@dataclass
class ScrapeResult:
    source: str
    listing_id: Optional[str]
    canonical_url: str
    title: Optional[str]
    summary_ru: str
    photos: List[str]
    phones: List[str]
    raw: Dict[str, Any]

async def scrape_listing(url: str) -> ScrapeResult:
    url = ensure_ru_myhome(url)
    extractor = get_extractor(url)
    listing = await asyncio.to_thread(extractor.extract)
    data = listing.dict()
    canonical_url = ensure_ru_myhome(data.get("url") or url)
    photos = data.get("photos") or []
    summary_ru = data.get("summary_ru") or build_summary_ru(data)
    return ScrapeResult(
        source=data.get("source") or "",
        listing_id=data.get("listing_id"),
        canonical_url=canonical_url,
        title=data.get("title"),
        summary_ru=summary_ru,
        photos=photos,
        phones=data.get("phones") or [],
        raw=data,
    )


# ============== Телеграм-бот ==============

dp = Dispatcher()
bot = Bot(BOT_TOKEN, parse_mode=ParseMode.HTML)

AWAIT_EDIT: Dict[int, str] = {}  # user_id -> canonical_url

def build_keyboard(row_id: int, exists_posted: bool) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📤 Отправить", callback_data=f"send:{row_id}")
    kb.button(text="✏️ Редактировать", callback_data=f"edit:{row_id}")
    if exists_posted:
        kb.button(text="🗑 Удалить из группы", callback_data=f"delete:{row_id}")
    kb.adjust(2, 1)
    return kb.as_markup()

@dp.message(CommandStart())
async def on_start(m: Message):
    await m.answer(
        "Пришлите ссылку на объявление (myhome.ge / home.ss.ge).\n"
        "Бот соберёт фото и краткое описание, либо — если такая ссылка уже публиковалась — предложит удалить пост из группы."
    )

@dp.message(Command("list"))
async def on_list(m: Message):
    rows = db_recent(10)
    if not rows:
        await m.answer("Список пуст.")
        return
    text = "\n".join(f"#{r['id']} [{r['status']}] {r['canonical_url']}" for r in rows)
    await m.answer(text)

@dp.message(Command("status"))
async def on_status(m: Message):
    now = datetime.now(tz=timezone.utc)
    def pack(delta_days: int) -> str:
        since = int((now - timedelta(days=delta_days)).timestamp())
        s = db_stats_window(since)
        return f"Опубликовано: {s['posted']}, Удалено: {s['deleted']}, Черновики: {s['drafts']}"
    msg = (
        "<b>Статистика</b>\n"
        f"За день:   {pack(1)}\n"
        f"За неделю: {pack(7)}\n"
        f"За месяц:  {pack(30)}"
    )
    await m.answer(msg)

@dp.message(F.text.func(lambda t: bool(pick_first_url(t or ""))))
async def on_link(m: Message):
    url_raw = pick_first_url(m.text or "")
    assert url_raw
    url = ensure_ru_myhome(url_raw)
    source, lid_guess = quick_source_and_id(url)

    # 1) Быстрая проверка дублей по URL/ID — без парсинга.
    row = db_find_by_keys(source, lid_guess, url)
    if row and row["status"] == "posted":
        # Уже публиковалось — ничего не парсим, предлагаем удалить/переслать/редактировать
        text = (
            f"⚠️ Такая ссылка уже публиковалась.\n\n"
            f"<b>{row['title'] or 'Объявление'}</b>\n"
            f"{(row['summary_ru'] or '')}"
        )
        await m.answer(text, reply_markup=build_keyboard(int(row["id"]), exists_posted=True))
        return

    await m.answer("Собираю данные…")
    # 2) Если новый лот или черновик — парсим для превью и заводим draft.
    try:
        sr = await scrape_listing(url)
    except Exception as e:
        await m.answer(f"Не удалось собрать данные: {e}")
        return

    # запись (draft) — либо обновим краткие поля, если уже есть черновик
    exists = db_find_by_keys(sr.source, sr.listing_id, sr.canonical_url)
    if exists:
        row_id = int(exists["id"])
        db_update_brief(row_id, sr.title, sr.summary_ru)
        exists_posted = exists["status"] == "posted"
    else:
        row_id = db_insert_draft(
            {
                "source": sr.source,
                "listing_id": sr.listing_id,
                "canonical_url": sr.canonical_url,
                "title": sr.title,
                "summary_ru": sr.summary_ru,
            }
        )
        exists_posted = False

    # превью альбомом оператору
    async with httpx.AsyncClient(headers={"User-Agent": "ListingBot/1.0"}) as client:
        album = await download_and_prepare_album(
            sr.photos,
            client,
            f"<b>{sr.title or ''}</b>\n{sr.summary_ru}"
        )
    if album:
        try:
            await bot.send_media_group(chat_id=m.chat.id, media=album)
        except TelegramBadRequest:
            await m.answer_photo(album[0].media, caption=album[0].caption)

    text = (
        f"<b>{sr.title or 'Объявление'}</b>\n{sr.summary_ru}"
    )
    await m.answer(text, reply_markup=build_keyboard(row_id, exists_posted))

# --- callbacks ---

async def _load_row(row_id: int) -> Optional[sqlite3.Row]:
    with closing(_db()) as conn:
        cur = conn.execute("SELECT * FROM postings WHERE id=?", (row_id,))
        return cur.fetchone()

@dp.callback_query(F.data.startswith("send:"))
async def on_send(cb: CallbackQuery):
    row_id = int(cb.data.split(":")[1])
    row = await _load_row(row_id)
    if not row:
        await cb.answer("Не нашёл запись", show_alert=True); return

    if row["status"] == "posted":
        await cb.message.answer("Это объявление уже опубликовано. Удалить публикацию из группы?",
                                reply_markup=build_keyboard(row_id, exists_posted=True))
        await cb.answer(); return

    # парсим актуальные данные только при реальной отправке
    try:
        sr = await scrape_listing(row["canonical_url"])
    except Exception as e:
        await cb.message.answer(f"Ошибка при сборе: {e}"); await cb.answer(); return

    async with httpx.AsyncClient(headers={"User-Agent": "ListingBot/1.0"}) as client:
        album = await download_and_prepare_album(
            sr.photos,
            client,
            f"<b>{sr.title or ''}</b>\n{sr.summary_ru}"
        )
    if not album:
        await cb.message.answer("Нет фото для отправки."); await cb.answer(); return

    msgs: List[Message] = await bot.send_media_group(chat_id=TARGET_GROUP_ID, media=album)
    message_ids = [m.message_id for m in msgs]
    db_mark_posted(row_id, TARGET_GROUP_ID, message_ids)
    # обновим заголовок/summary (могли измениться)
    db_update_brief(row_id, sr.title, sr.summary_ru)

    await cb.message.answer("✅ Отправлено в группу.")
    await cb.answer()

@dp.callback_query(F.data.startswith("edit:"))
async def on_edit(cb: CallbackQuery):
    row_id = int(cb.data.split(":")[1])
    row = await _load_row(row_id)
    if not row:
        await cb.answer("Не нашёл запись", show_alert=True); return
    # запомним, что ждём текст
    AWAIT_EDIT[cb.from_user.id] = row["canonical_url"]
    await cb.message.answer("Пришлите текст. Его поставлю как подпись к альбому при отправке в группу.")
    await cb.answer()

@dp.message(F.from_user.func(lambda u: u.id in AWAIT_EDIT))
async def on_edit_text(m: Message):
    url = AWAIT_EDIT.pop(m.from_user.id)
    row = db_find_by_keys(*quick_source_and_id(url), ensure_ru_myhome(url))
    if not row:
        await m.answer("Запись не найдена, пришлите ссылку ещё раз."); return
    try:
        sr = await scrape_listing(url)
    except Exception as e:
        await m.answer(f"Ошибка при сборе: {e}"); return
    caption = (m.text or "").strip()[:1024]
    async with httpx.AsyncClient(headers={"User-Agent": "ListingBot/1.0"}) as client:
        album = await download_and_prepare_album(sr.photos, client, caption)
    if not album:
        await m.answer("Нет фото для отправки."); return
    msgs = await bot.send_media_group(chat_id=TARGET_GROUP_ID, media=album)
    message_ids = [mm.message_id for mm in msgs]
    db_mark_posted(int(row["id"]), TARGET_GROUP_ID, message_ids)
    await m.answer("✅ Отправлено в группу с вашим текстом.")

@dp.callback_query(F.data.startswith("delete:"))
async def on_delete(cb: CallbackQuery):
    row_id = int(cb.data.split(":")[1])
    row = await _load_row(row_id)
    if not row or row["status"] != "posted":
        await cb.answer("Публикации нет или уже удалена."); return
    chat_id = int(row["chat_id"])
    ids = json.loads(row["message_ids"] or "[]")
    ok_all = True
    for mid in ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=int(mid))
        except TelegramBadRequest:
            ok_all = False
    db_mark_deleted(row_id)
    await cb.message.answer("🗑 Удалено из группы." if ok_all else "Удалил не всё (проверьте права бота).")
    await cb.answer()

# ============== Main ==============

async def main():
    if not BOT_TOKEN or not TARGET_GROUP_ID:
        raise SystemExit("Укажите BOT_TOKEN и TARGET_GROUP_ID в переменных окружения.")
    init_db()
    print("Bot is running…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
