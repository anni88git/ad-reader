"""Baseline system 'ocr_rules': OCR + hand-written rules, NO vision-language model.
Answers the assignment's question of what OCR (PaddleOCR) achieves on its own.

  headline    = text of the tallest OCR line (largest font)
  sub_headline= second tallest line
  brand       = longest master-data brand name/alias found in the OCR text
  phones/prices/websites/emails = regex over OCR lines
  everything else (ad type, category, product, features, ...) = not attempted
Writes data/eval/<set>/runs/ocr_rules/raw/<id>.json (then: run_eval.py validate --system ocr_rules)
"""
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import normalize as nz  # noqa: E402
from adreader.matcher import default_matcher  # noqa: E402
from adreader.schema import Ev, RawExtraction, RawPrice  # noqa: E402

PHONE = re.compile(r"(?:\+91[\s-]?)?(?:1800[\s-]?\d{3}[\s-]?\d{4}|0?\d{2,4}[\s-]?\d{6,8}|[6-9]\d{4}[\s-]?\d{5})")
PRICE = re.compile(r"(?:₹|rs\.?|र|रु\.?)\s?\d[\d,\s]*(?:\.\d+)?\s*(?:lakh|lac|cr|crore|/-)?", re.I)
WEB = re.compile(r"(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|in|co\.in|org|net|info)(?:/\S*)?", re.I)
MAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)


def height(l):
    b = np.asarray(l.box)
    return min(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[1] - b[2]))


def extract(ocr) -> RawExtraction:
    lines = [l for l in ocr.lines if l.score >= 0.8 and len(nz.key(l.norm)) >= 3]
    big = sorted(lines, key=height, reverse=True)
    raw = RawExtraction()
    if big:
        raw.headline = Ev(value=big[0].norm, evidence=big[0].norm)
    if len(big) > 1:
        raw.sub_headline = Ev(value=big[1].norm, evidence=big[1].norm)
    corpus = "|".join(nz.key(l.norm) for l in ocr.lines)
    m = default_matcher()
    best = None
    for b in m.tables["brands"]:
        for n in b.names:
            k = nz.key(n)
            if len(k) >= 3 and k in corpus and (best is None or len(k) > len(nz.key(best[1]))):
                best = (b, n)
    if best:
        raw.brand = Ev(value=best[0].name, evidence=best[1])
    seen = set()
    for l in ocr.lines:
        for mm in PHONE.finditer(l.norm):
            p = nz.parse_phone(mm.group(0))
            if p.valid and p.number not in seen:
                seen.add(p.number)
                raw.phones.append(Ev(value=mm.group(0), evidence=l.norm))
        for mm in PRICE.finditer(l.norm):
            raw.prices.append(RawPrice(amount_text=mm.group(0).strip(), label=l.norm))
        for mm in WEB.finditer(l.norm):
            if "@" not in l.norm[max(0, mm.start() - 1):mm.start()]:
                raw.websites.append(Ev(value=mm.group(0), evidence=l.norm))
        for mm in MAIL.finditer(l.norm):
            raw.emails.append(Ev(value=mm.group(0), evidence=l.norm))
    scripts = {l.script for l in lines}
    raw.language = (["English"] if "latin" in scripts else []) + (["Hindi"] if "devanagari" in scripts else [])
    return raw


if __name__ == "__main__":
    set_dir = ROOT / "data/eval" / (sys.argv[1] if len(sys.argv) > 1 else "set50")
    out = set_dir / "runs" / "ocr_rules" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((set_dir / "manifest.csv").open(encoding="utf-8")))
    for r in rows:
        ocr = pickle.loads((set_dir / "ocr" / f"{r['id']}.pkl").read_bytes())
        raw = extract(ocr)
        (out / f"{r['id']}.json").write_text(json.dumps({"id": r["id"], "model": "ocr+rules (no VLM)", "seconds": 0,
                                                         "raw": raw.model_dump()}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} baseline extractions -> {out}")
