"""Run the OCR layer on images and save box visualisations.

  python scripts/ocr_demo.py data/ads/002_horlicks_hi.png [more images...] [--langs latin,devanagari]
Outputs data/outputs/ocr/<name>.txt and <name>_boxes.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adreader import ocr, preprocess as pp  # noqa: E402
from adreader.config import OUTPUTS_DIR  # noqa: E402


def draw(res: ocr.OCRResult) -> np.ndarray:
    vis = res.image.copy()
    for i, l in enumerate(res.lines, 1):
        pts = np.asarray(l.box, dtype=np.int32)
        colour = (0, 170, 0) if not l.low_confidence else (0, 140, 255)
        cv2.polylines(vis, [pts], True, colour, 2)
        cv2.putText(vis, str(i), tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return vis


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--langs", default=",".join(ocr.DEFAULT_LANGS))
    a = ap.parse_args()
    out_dir = OUTPUTS_DIR / "ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in a.images:
        res = ocr.run_ocr(pp.load_image(p), langs=a.langs.split(","))
        name = Path(p).stem
        listing = res.prompt_listing()
        (out_dir / f"{name}.txt").write_text(listing, encoding="utf-8")
        cv2.imwrite(str(out_dir / f"{name}_boxes.jpg"), draw(res))
        q = res.quality
        print(f"\n===== {p}  lines={len(res.lines)}  scale={res.scale}  deskew={res.deskew_deg}  "
              f"time={res.timings_s}  quality={q.width}x{q.height} sharp={q.sharpness} {q.warnings}")
        print(listing)
