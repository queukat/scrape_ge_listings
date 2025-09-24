# ge_listings/wm_clean.py
from __future__ import annotations

import base64
import os

import cv2
import httpx
import numpy as np

INPAINT_URL = os.getenv("INPAINT_URL", "http://127.0.0.1:8080/api/v1/inpaint")
SIZE_LIMIT = os.getenv("SIZE_LIMIT", "1536")


# Флаги были инвертированы — исправлено:
# "1" -> включено, "0" -> выключено (по умолчанию выключено)
def _flag_on(env_name: str, default: str = "0") -> bool:
    return os.getenv(env_name, default) != "0"


WM_ENGINE = os.getenv("WM_ENGINE", "iopaint").lower()  # iopaint | opencv
WM_CLEAN = _flag_on("WM_CLEAN", "0")
WM_SSGE = _flag_on("WM_SSGE", "0")
WM_MYHOME = _flag_on("WM_MYHOME", "0")
MIN_COVERAGE = float(os.getenv("WM_MIN_COVERAGE", "0.0005"))


def _to_data_url(png_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(png_bytes).decode('ascii')}"


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(gray)
    lo = int(max(0, (1.0 - sigma) * v));
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lo, hi)


def _mask_ss_bottom(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    band_h = int(max(40, round(h * 0.30)))
    y1 = h - band_h
    roi = img[y1:h, 0:w]

    L = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)

    kx = max(9, min(45, int(w * 0.045)))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
    combo = cv2.max(cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k_h),
                    cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k_h))
    combo = cv2.normalize(combo, None, 0, 255, cv2.NORM_MINMAX)
    _, th_otsu = cv2.threshold(combo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(combo, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 31, -5)
    th = cv2.max(th_otsu, th_adapt)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (max(11, int(w * 0.015)), 3)), 1)

    edges = _auto_canny(L, 0.33)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(18, int(w * 0.01)),
                            minLineLength=max(6, int(w * 0.02)),
                            maxLineGap=max(2, int(w * 0.01)))
    m_lines = np.zeros((band_h, w), np.uint8)
    if lines is not None:
        for x1, y1l, x2, y2 in lines[:, 0]:
            if abs(y2 - y1l) <= 3 and max(x1, x2) <= int(w * 0.35):
                Lg = np.hypot(x2 - x1, y2 - y1l)
                if Lg >= max(6, int(w * 0.02)) and Lg <= int(w * 0.25):
                    cv2.line(m_lines, (x1, y1l), (x2, y2), 255, 8)

    g1 = cv2.GaussianBlur(L, (0, 0), 1.0);
    g2 = cv2.GaussianBlur(L, (0, 0), 3.0)
    dog = cv2.absdiff(g1, g2)
    white = cv2.inRange(L, 210, 255);
    thin = cv2.inRange(dog, 6, 40)
    wthin = cv2.bitwise_and(white, thin);
    wthin[:, int(w * 0.4):] = 0

    roi_mask = cv2.max(th, cv2.max(m_lines, wthin))
    roi_mask = cv2.dilate(roi_mask, np.ones((3, 3), np.uint8), 1)

    mask = np.zeros((h, w), np.uint8)
    mask[y1:h, 0:w] = roi_mask
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 1)
    return mask


def _mask_myhome_center(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = int(w * float(os.getenv("CENTER_X1", "0.15")))
    y1 = int(h * float(os.getenv("CENTER_Y1", "0.35")))
    x2 = int(w * float(os.getenv("CENTER_X2", "0.85")))
    y2 = int(h * float(os.getenv("CENTER_Y2", "0.75")))

    roi = img[y1:y2, x1:x2]
    L = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)

    g1 = cv2.GaussianBlur(L, (0, 0), 1.0);
    g2 = cv2.GaussianBlur(L, (0, 0), 3.0)
    dog = cv2.absdiff(g1, g2)
    bright = cv2.inRange(L, 200, 255);
    thin = cv2.inRange(dog, 4, 40)
    m = cv2.bitwise_and(bright, thin)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(21, int(w * 0.12)), max(9, int(h * 0.04))))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5)), 1)

    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = m
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 1)
    return mask


def should_clean(source: str | None) -> bool:
    if not WM_CLEAN:
        return False
    if not source:
        return True
    s = source.lower()
    if "ss.ge" in s:      return WM_SSGE
    if "myhome.ge" in s:  return WM_MYHOME
    return WM_CLEAN


async def _post_iopaint(client: httpx.AsyncClient, img_bgr: np.ndarray, mask: np.ndarray) -> bytes | None:
    ok1, png = cv2.imencode(".png", img_bgr)
    ok2, msk = cv2.imencode(".png", mask)
    if not (ok1 and ok2):
        return None
    payload = {
        "image": _to_data_url(png.tobytes()),
        "mask": _to_data_url(msk.tobytes()),
        "size_limit": SIZE_LIMIT,
    }
    r = await client.post(INPAINT_URL, json=payload, headers={"Accept": "image/png,application/json"})
    ct = (r.headers.get("content-type") or "").lower()
    if ct.startswith("image/"):
        return r.content
    try:
        data = r.json()
        img_field = data.get("image") or data.get("result") or ""
        if isinstance(img_field, str) and img_field.startswith("data:"):
            _, b64 = img_field.split(",", 1)
            return base64.b64decode(b64)
    except Exception:
        pass
    return None


def _fallback_inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> bytes | None:
    out = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None


async def clean_image_bytes(content: bytes, source: str | None, client: httpx.AsyncClient) -> bytes | None:
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    masks: list[np.ndarray] = []
    s = (source or "").lower()
    if "ss.ge" in s:
        masks.append(_mask_ss_bottom(img))
    if "myhome.ge" in s:
        masks.append(_mask_myhome_center(img))
    if not masks:
        masks = [_mask_ss_bottom(img), _mask_myhome_center(img)]

    mask = max(masks, key=lambda m: cv2.countNonZero(m))
    coverage = cv2.countNonZero(mask) / float(mask.size)
    if coverage < MIN_COVERAGE:
        return None

    if WM_ENGINE == "iopaint" and INPAINT_URL:
        try:
            out = await _post_iopaint(client, img, mask)
            if out:
                return out
        except Exception:
            pass
    return _fallback_inpaint(img, mask)
