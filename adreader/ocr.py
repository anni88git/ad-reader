"""OCR layer: PaddleOCR (PP-OCR) models via RapidOCR / ONNX Runtime.

Pipeline (accuracy-first):
  1. upscale small images, run text DETECTION on the plain and the contrast-enhanced image
     (decorative headlines on coloured backgrounds are often found by only one of the two)
  2. auto-deskew from the detected box angles (tilted phone photos)
  3. RECOGNISE every detected box with each requested script model (Latin + Devanagari by
     default; Tamil / Telugu / Kannada optional) and keep the best reading per box, while
     remembering the alternatives (used later for evidence grounding)
  4. remove duplicate boxes, normalise Indic digits (४४ -> 44), sort into reading order

The output is both a machine structure (boxes, scores) and a numbered text listing that is
fed to the vision-language model as grounding.
"""
from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from . import preprocess as pp

logging.getLogger("RapidOCR").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------- scripts
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}
# zero digit code point for each Indic script; digits are contiguous 0..9
_INDIC_ZEROS = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66]
DIGIT_MAP = {z + i: str(i) for z in _INDIC_ZEROS for i in range(10)}

SUPPORTED_LANGS = ("latin", "devanagari", "tamil", "telugu", "kannada")
DEFAULT_LANGS = ("latin", "devanagari")
AUTO = "auto"                      # langs="auto" / ["auto"]: detect the script(s) per ad (see _probe_scripts)
PROBE_SCRIPTS = ("devanagari", "telugu", "tamil", "kannada")
PROBE_SAMPLES = 10                 # tallest boxes the Latin model could not read, used as the probe set
PROBE_MIN = 0.25                   # a script is "present" when its own-script readings cover >= 25% of the probe
PROBE_MODEL_TIMEOUT_S = 20         # one script model (load + read the probe) - slower than this: skipped
PROBE_BUDGET_S = 30                # whole probe; when exceeded the remaining scripts are not tried
REGIONAL_READ_TIMEOUT_S = 180      # auto mode's final regional read; on timeout that script is dropped


def script_of(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if ch.isascii() and ch.isalpha():
            counts["latin"] = counts.get("latin", 0) + 1
            continue
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.get) if counts else "none"


def _script_count(text: str, script: str) -> int:
    lo, hi = SCRIPT_RANGES[script]
    return sum(1 for ch in text if lo <= ord(ch) <= hi and unicodedata.category(ch).startswith("L"))


def normalize_text(text: str) -> str:
    """Unicode NFC + Indic digits -> ASCII digits + collapse whitespace."""
    t = unicodedata.normalize("NFC", text).translate(DIGIT_MAP)
    return " ".join(t.split())


# ----------------------------------------------------------------------------- engines
@lru_cache(maxsize=None)
def _engine(lang: str):
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    det = {"Det.limit_side_len": 1600, "Det.limit_type": "max", "Det.box_thresh": 0.5}
    if lang == "latin":
        return RapidOCR(params=det)   # default PP-OCR rec handles Latin + digits very well
    lang_enum = {"devanagari": LangRec.DEVANAGARI, "tamil": LangRec.TA,
                 "telugu": LangRec.TE, "kannada": LangRec.KA}[lang]
    last_err = None
    for ver in (OCRVersion.PPOCRV5, OCRVersion.PPOCRV4):
        try:
            eng = RapidOCR(params={**det, "Rec.lang_type": lang_enum, "Rec.ocr_version": ver,
                                   "Rec.model_type": ModelType.MOBILE})
            eng.recognize_txt([np.full((48, 160, 3), 255, np.uint8)])  # force model load/validate
            return eng
        except Exception as e:  # model combo not available -> try older version
            last_err = e
    raise RuntimeError(f"No OCR model available for {lang}: {last_err}")


def _detect(img: np.ndarray) -> list[np.ndarray]:
    res = _engine("latin")(img, use_det=True, use_cls=False, use_rec=False)
    boxes = getattr(res, "boxes", None)
    return [] if boxes is None else [np.asarray(b, dtype=np.float32) for b in boxes]


def _recognize(lang: str, crops: list[np.ndarray], eng=None) -> list[tuple[str, float]]:
    if not crops:
        return []
    eng = eng if eng is not None else _engine(lang)
    try:
        out = eng.recognize_txt(crops)                 # batched (fast path)
        return list(zip(out.txts, map(float, out.scores)))
    except Exception:
        res = []
        for c in crops:                               # public per-crop fallback
            r = eng(c, use_det=False, use_cls=False, use_rec=True)
            txts, scores = getattr(r, "txts", None), getattr(r, "scores", None)
            res.append((txts[0], float(scores[0])) if txts else ("", 0.0))
        return res


def _crop(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Perspective-crop a 4-point text box into a horizontal strip."""
    pts = box.astype(np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 2), max(h, 2)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if h > 1.5 * w:                                   # vertical text strip -> rotate
        crop = np.rot90(crop)
    return crop


def _aabb(box: np.ndarray) -> tuple[float, float, float, float]:
    return float(box[:, 0].min()), float(box[:, 1].min()), float(box[:, 0].max()), float(box[:, 1].max())


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _box_angle(box: np.ndarray) -> float:
    dx, dy = box[1] - box[0]
    return float(np.degrees(np.arctan2(dy, dx)))


# ----------------------------------------------------------------------------- results
@dataclass
class OCRLine:
    text: str                      # best reading, raw
    norm: str                      # normalised (NFC, ASCII digits, single spaces)
    score: float
    lang: str                      # recogniser that produced the best reading
    script: str
    box: list[list[float]]         # 4 points in OCRResult.image coordinates
    alts: list[tuple[str, float, str]] = field(default_factory=list)  # other readings

    @property
    def low_confidence(self) -> bool:
        return self.score < 0.80


@dataclass
class OCRResult:
    lines: list[OCRLine]
    image: np.ndarray              # the (upscaled / deskewed) image the boxes refer to
    scale: float                   # image = original * scale (before deskew)
    deskew_deg: float
    quality: pp.ImageQuality
    langs: tuple[str, ...]
    timings_s: dict[str, float]
    # set only in auto mode: {"mode": "auto", "detected": [...], "scores": {...}, "probed": n, "fallback": bool}
    script_detection: dict | None = None

    def full_text(self) -> str:
        return "\n".join(l.norm for l in self.lines)

    def prompt_listing(self, max_lines: int = 400) -> str:
        """Numbered listing for the VLM prompt. Low-confidence lines are marked with '?'."""
        out = []
        for i, l in enumerate(self.lines[:max_lines], 1):
            mark = "?" if l.low_confidence else " "
            out.append(f"[{i:03d}]{mark} {l.norm}")
        return "\n".join(out)

    def all_readings(self) -> list[str]:
        """Every reading (best + alternatives), normalised - used for evidence grounding."""
        seen, out = set(), []
        for l in self.lines:
            for t in [l.norm] + [normalize_text(a[0]) for a in l.alts]:
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
        return out


# ----------------------------------------------------------------------------- main entry
def is_auto(langs) -> bool:
    return langs == AUTO or (not isinstance(langs, str) and AUTO in set(langs or ()))


def _with_timeout(fn, timeout_s: float):
    """Run fn() in a helper thread; raise TimeoutError if it takes longer (the thread is abandoned, the job
    goes on - a hung ONNX call can never freeze the OCR stage)."""
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn).result(timeout=timeout_s)
    finally:
        ex.shutdown(wait=False)


def _probe_crop(c: np.ndarray) -> np.ndarray:
    """A script needs only a few glyphs to identify: scale to the recogniser's 48 px line height and keep the
    first 480 px, so a full-width line of fine print costs no more than a headline."""
    h, w = c.shape[:2]
    s = 48.0 / max(h, 1)
    return cv2.resize(c, (max(8, int(w * s)), 48), interpolation=cv2.INTER_AREA)[:, :480]


def _probe_scripts(crops: list[np.ndarray]) -> tuple[tuple[str, ...], dict]:
    """Which Indic script(s) are in these crops (boxes the Latin model could not read)?

    RAM-safe for small CPU containers: the probe models are loaded ONE AT A TIME and freed straight away
    (only Devanagari - a default script that stays loaded anyway - uses the shared engine cache), so peak
    memory is the Latin model plus one Indic model. A model that fails to load or exceeds its time limit is
    skipped; past the overall budget the remaining scripts are not tried (-> default scripts)."""
    import gc
    lines = [c for c in crops if c.shape[0] >= 12 and c.shape[1] >= 1.5 * c.shape[0]]   # text-line shaped
    sample = [_probe_crop(c) for c in sorted(lines, key=lambda c: c.shape[0], reverse=True)[:PROBE_SAMPLES]]
    info: dict = {"mode": AUTO, "probed": len(sample), "scores": {}, "failed": [], "timed_out": []}
    if not sample:
        return (), info
    best_per_crop: list[tuple[float, str | None]] = [(0.0, None)] * len(sample)
    t_start = time.perf_counter()
    for lang in PROBE_SCRIPTS:
        if time.perf_counter() - t_start > PROBE_BUDGET_S:
            info["timed_out"].append(lang)
            continue
        holder: dict = {}

        def _load_and_read(lang=lang, holder=holder):
            holder["eng"] = _engine(lang) if lang in DEFAULT_LANGS else _engine.__wrapped__(lang)
            return _recognize(lang, sample, eng=holder["eng"])
        try:
            out = _with_timeout(_load_and_read, PROBE_MODEL_TIMEOUT_S)
        except TimeoutError:
            info["timed_out"].append(lang)
            continue
        except Exception:
            info["failed"].append(lang)
            continue
        finally:
            if lang not in DEFAULT_LANGS:
                holder.clear()
                gc.collect()
        for i, (t, s) in enumerate(out):        # an own-script reading; the most confident model wins the box
            if s >= 0.6 and _script_count(t, lang) >= 2 and script_of(t) == lang and s > best_per_crop[i][0]:
                best_per_crop[i] = (s, lang)
    # share of probe boxes each script wins. Per-box winners separate look-alike scripts (the Telugu model
    # reads Kannada with high confidence too, but the Kannada model reads it better box by box).
    info["probe_s"] = round(time.perf_counter() - t_start, 2)
    for lang in PROBE_SCRIPTS:
        if lang not in info["failed"] and lang not in info["timed_out"]:
            info["scores"][lang] = round(sum(1 for _, w in best_per_crop if w == lang) / len(sample), 3)
    found = tuple(l for l in PROBE_SCRIPTS if info["scores"].get(l, 0.0) >= PROBE_MIN)
    return found, info


def run_ocr(img: np.ndarray, langs: tuple[str, ...] | list[str] = DEFAULT_LANGS,
            min_score: float = 0.45) -> OCRResult:
    auto = is_auto(langs)
    detection = None
    langs = ("latin",) if auto else (tuple(l for l in SUPPORTED_LANGS if l in set(langs)) or DEFAULT_LANGS)
    t0 = time.perf_counter()
    quality = pp.assess(img)
    scale = pp.upscale_factor(img)
    work = pp.cap_size(pp.resize(img, scale))
    scale = work.shape[1] / img.shape[1]

    # -- detection on plain + enhanced, then deskew if needed
    enhanced = pp.enhance(work)
    boxes = _detect(work) + _detect(enhanced)
    angles = [_box_angle(b) for b in boxes if (_aabb(b)[2] - _aabb(b)[0]) > 80]
    deskew = float(np.median(angles)) if len(angles) >= 5 else 0.0
    if 0.7 <= abs(deskew) <= 15:
        work = pp.rotate(work, deskew)
        enhanced = pp.enhance(work)
        boxes = _detect(work) + _detect(enhanced)
    else:
        deskew = 0.0
    boxes = _merge_boxes(boxes)
    t_det = time.perf_counter()

    # -- recognise: Latin model on every box; Indic models only on boxes the Latin model could
    #    not read convincingly (see _needs_indic) - same accuracy, far less CPU time
    crops = [_crop(work, b) for b in boxes]
    readings: dict[str, dict[int, tuple[str, float]]] = {}
    if "latin" in langs:
        readings["latin"] = dict(enumerate(_recognize("latin", crops)))
        todo = [i for i, c in enumerate(crops) if _needs_indic(readings["latin"][i], c)]
    else:
        todo = list(range(len(crops)))
    if auto:            # detect, then read with latin + the detected script(s); nothing found -> the defaults
        found, detection = _probe_scripts([crops[i] for i in todo])
        langs = tuple(l for l in SUPPORTED_LANGS if l == "latin" or l in found)
        detection["fallback"] = len(langs) == 1
        if detection["fallback"]:
            langs = DEFAULT_LANGS
        detection["detected"] = list(langs)
    for lang in langs:
        if lang == "latin":
            continue
        if auto and lang not in DEFAULT_LANGS:  # auto mode: regional model loaded for this ad only, then freed
            import gc
            holder: dict = {}

            def _read(lang=lang, holder=holder):
                holder["eng"] = _engine.__wrapped__(lang)
                return _recognize(lang, [crops[i] for i in todo], eng=holder["eng"])
            try:
                readings[lang] = dict(zip(todo, _with_timeout(_read, REGIONAL_READ_TIMEOUT_S)))
            except TimeoutError:                # too slow: keep what the other scripts read
                detection.setdefault("timed_out", []).append(lang)
                readings[lang] = {}
            except Exception:                   # model failed to load: keep what the other scripts read
                detection.setdefault("failed", []).append(lang)
                readings[lang] = {}
            finally:
                holder.clear()
                gc.collect()
        else:
            readings[lang] = dict(zip(todo, _recognize(lang, [crops[i] for i in todo])))
    t_rec = time.perf_counter()

    candidates: list[OCRLine] = []
    for i, box in enumerate(boxes):
        opts = [(readings[lang][i][0], readings[lang][i][1], lang) for lang in langs if i in readings[lang]]
        best = max(opts, key=lambda o: _rank(o))
        if not best[0].strip() or best[1] < min_score:
            continue
        candidates.append(OCRLine(
            text=best[0], norm=normalize_text(best[0]), score=round(best[1], 4), lang=best[2],
            script=script_of(best[0]), box=box.round(1).tolist(),
            alts=[(t, round(s, 4), lg) for t, s, lg in opts if lg != best[2] and t.strip()]))

    lines = _reading_order(_dedupe(candidates))
    t_end = time.perf_counter()
    return OCRResult(lines=lines, image=work, scale=round(scale, 4), deskew_deg=round(deskew, 2),
                     quality=quality, langs=langs,
                     timings_s={"detect": round(t_det - t0, 2), "recognise": round(t_rec - t_det, 2),
                                "total": round(t_end - t0, 2)},
                     script_detection=detection)


_OK_NON_ASCII = set("₹•·–—“”‘’°×©®™…")


def _needs_indic(latin: tuple[str, float], crop: np.ndarray) -> bool:
    """Should this box also be read by the Indic script models?
    The Latin/Chinese model often 'reads' Devanagari as CJK glyphs or a few junk characters -
    sometimes with high confidence - so score alone is not enough."""
    text, score = latin
    t = text.strip()
    if score < 0.95 or not t:
        return True
    if any(not ch.isascii() and ch not in _OK_NON_ASCII for ch in t):
        return True                                        # CJK / odd glyphs => wrong script
    aspect = crop.shape[1] / max(1, crop.shape[0])
    return len(t) < 0.4 * aspect                           # far too few chars for box width


def _rank(opt: tuple[str, float, str]) -> float:
    """Choose between script models' readings of the same box.
    A script model that actually outputs its own script gets a small bonus; the Latin model
    wins ties on pure ASCII text (Indic models occasionally inject look-alike glyphs)."""
    text, score, lang = opt
    sc = script_of(text)
    if lang != "latin" and sc == lang and _script_count(text, lang) < 2:
        sc = "none"   # a lone glyph (e.g. '₹' misread as 'र') is not evidence of the script
    if lang != "latin" and sc == lang:
        return score + 0.05
    if lang == "latin" and sc in ("latin", "none"):
        return score + 0.01
    if lang != "latin" and sc != lang:
        return score - 0.05
    return score


def _merge_boxes(boxes: list[np.ndarray], thr: float = 0.6) -> list[np.ndarray]:
    """Union of the detection passes BEFORE recognition: near-identical boxes are recognised
    once (keeping the larger box, which is less likely to clip characters)."""
    kept: list[np.ndarray] = []
    kept_ab: list[tuple] = []
    for b in sorted(boxes, key=lambda b: -cv2.contourArea(b.astype(np.float32))):
        ab = _aabb(b)
        if all(_iou(ab, k) <= thr for k in kept_ab):
            kept.append(b)
            kept_ab.append(ab)
    return kept


def _dedupe(lines: list[OCRLine], thr: float = 0.5) -> list[OCRLine]:
    """Two detection passes find many of the same boxes: keep the better-scored reading."""
    kept: list[OCRLine] = []
    for l in sorted(lines, key=lambda x: -x.score):
        a = _aabb(np.asarray(l.box))
        dup = next((k for k in kept if _iou(a, _aabb(np.asarray(k.box))) > thr), None)
        if dup is None:
            kept.append(l)
        elif l.norm != dup.norm:
            dup.alts.append((l.text, l.score, l.lang))
    return kept


def _reading_order(lines: list[OCRLine]) -> list[OCRLine]:
    """Top-to-bottom rows, left-to-right within a row (row tolerance = half median height)."""
    if not lines:
        return lines
    heights = [(_aabb(np.asarray(l.box))[3] - _aabb(np.asarray(l.box))[1]) for l in lines]
    tol = max(4.0, float(np.median(heights)) * 0.5)
    items = sorted(lines, key=lambda l: _aabb(np.asarray(l.box))[1])
    rows: list[list[OCRLine]] = []
    for l in items:
        y = _aabb(np.asarray(l.box))[1]
        if rows and abs(_aabb(np.asarray(rows[-1][0].box))[1] - y) <= tol:
            rows[-1].append(l)
        else:
            rows.append([l])
    return [l for row in rows for l in sorted(row, key=lambda l: _aabb(np.asarray(l.box))[0])]
