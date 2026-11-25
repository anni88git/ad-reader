"""Image loading, quality assessment and light enhancement before OCR / VLM."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

TARGET_MIN_LONG_SIDE = 1600   # upscale smaller images so small print survives OCR
MAX_LONG_SIDE = 4000          # cap to keep OCR time bounded


@dataclass
class ImageQuality:
    width: int
    height: int
    sharpness: float      # variance of Laplacian; < ~80 is usually blurry
    contrast: float       # std-dev of grey levels
    warnings: list[str]


def load_image(src: str | Path | bytes) -> np.ndarray:
    """Load as BGR uint8, honouring EXIF orientation (phone photos)."""
    pil = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def assess(img: np.ndarray) -> ImageQuality:
    h, w = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    contrast = float(grey.std())
    warns = []
    if max(h, w) < 1000:
        warns.append(f"low resolution ({w}x{h}); small print may be unreadable")
    if sharp < 80:
        warns.append(f"image looks blurry (sharpness {sharp:.0f})")
    if contrast < 35:
        warns.append(f"low contrast ({contrast:.0f})")
    return ImageQuality(w, h, round(sharp, 1), round(contrast, 1), warns)


def upscale_factor(img: np.ndarray) -> float:
    long_side = max(img.shape[:2])
    if long_side >= TARGET_MIN_LONG_SIDE:
        return 1.0
    return min(3.0, TARGET_MIN_LONG_SIDE / long_side)


def resize(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return img
    interp = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def cap_size(img: np.ndarray) -> np.ndarray:
    long_side = max(img.shape[:2])
    return resize(img, MAX_LONG_SIDE / long_side) if long_side > MAX_LONG_SIDE else img


def enhance(img: np.ndarray) -> np.ndarray:
    """Mild, OCR-friendly enhancement: local contrast (CLAHE on luminance) + light sharpening.
    Deliberately conservative - aggressive filters create fake strokes that OCR misreads."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around centre, expanding canvas so nothing is cut off (white fill)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def to_jpeg_bytes(img: np.ndarray, max_long_side: int = 2048, quality: int = 92) -> bytes:
    """Encode for sending to a VLM API (bounded payload size)."""
    long_side = max(img.shape[:2])
    if long_side > max_long_side:
        img = resize(img, max_long_side / long_side)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
