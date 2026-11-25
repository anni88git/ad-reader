"""OCR engine comparison on the same images: native PaddleOCR (Kaggle CPU) vs RapidOCR (PaddleOCR
models via ONNX Runtime, laptop CPU - the engine used by the pipeline).

Metric without a character-level OCR ground truth: FACT RECALL = share of labelled facts that occur
verbatim (normalised) in the engine's text: headline wording, phone digits, website, price digits.
  python scripts/compare_ocr.py [--set holdout20] [--paddle data/eval/kaggle_out/adreader-paddle-ocr/paddle_ocr.jsonl]
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import normalize as nz  # noqa: E402
from adreader.config import LABELS_DIR  # noqa: E402


def facts(lab: dict) -> list[tuple[str, str]]:
    def alts(v):
        return v["any_of"] if isinstance(v, dict) and "any_of" in v else [v]

    def items(v):
        if isinstance(v, dict) and "required" in v:
            return v["required"]
        return v if isinstance(v, list) else []
    out = []
    if lab.get("headline") and "headline" not in lab.get("_skip", []):
        first = next((a for a in alts(lab["headline"]) if a), None)
        if first:
            out.append(("headline", nz.key(first)))
    for p in items(lab.get("phones")):
        out.append(("phone", nz.digits(str(p))[-10:]))
    for w in items(lab.get("websites")):
        out.append(("website", nz.key(str(w).lower().replace("www.", "").replace("https://", ""))))
    pr = next((a for a in alts(lab.get("price")) if isinstance(a, int)), None) if "price" in lab else None
    if pr:
        out.append(("price", str(pr)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="holdout20")
    ap.add_argument("--paddle", default="data/eval/kaggle_out/adreader-paddle-ocr/paddle_ocr.jsonl")
    a = ap.parse_args()
    set_dir = ROOT / "data/eval" / a.set
    rows = list(csv.DictReader((set_dir / "manifest.csv").open(encoding="utf-8")))
    paddle = {}
    pf = ROOT / a.paddle
    if pf.exists():
        for line in pf.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            paddle[d["id"]] = d
    engines = {"RapidOCR (ONNX, laptop CPU)": {}, "PaddleOCR (native, Kaggle CPU)": {}}
    stats = {e: {"found": 0, "total": 0, "by_kind": {}, "seconds": [], "lines": [], "errors": 0} for e in engines}
    for r in rows:
        lf = LABELS_DIR / f"{Path(r['file']).stem}.json"
        if not lf.exists():
            continue
        lab = json.loads(lf.read_text(encoding="utf-8"))
        rap = pickle.loads((set_dir / "ocr" / f"{r['id']}.pkl").read_bytes())
        texts = {"RapidOCR (ONNX, laptop CPU)": [l.norm for l in rap.lines] + [a[0] for l in rap.lines for a in l.alts]}
        stats["RapidOCR (ONNX, laptop CPU)"]["seconds"].append(rap.timings_s["total"])
        stats["RapidOCR (ONNX, laptop CPU)"]["lines"].append(len(rap.lines))
        pd = paddle.get(r["id"])
        if pd and pd.get("lines") is not None and not pd.get("error"):
            texts["PaddleOCR (native, Kaggle CPU)"] = [x["text"] for x in pd["lines"]]
            stats["PaddleOCR (native, Kaggle CPU)"]["seconds"].append(pd["seconds"])
            stats["PaddleOCR (native, Kaggle CPU)"]["lines"].append(len(pd["lines"]))
        else:
            stats["PaddleOCR (native, Kaggle CPU)"]["errors"] += 1
        for eng, lines in texts.items():
            key_corpus = "|".join(nz.key(t) for t in lines)
            dig_corpus = "|".join(nz.digits(t) for t in lines)
            for kind, f in facts(lab):
                if not f:
                    continue
                hit = (f in dig_corpus) if kind in ("phone", "price") else (f in key_corpus.replace("|", "") or f in key_corpus)
                st = stats[eng]
                st["total"] += 1
                st["found"] += hit
                k = st["by_kind"].setdefault(kind, [0, 0])
                k[0] += hit
                k[1] += 1
    report = {}
    for eng, st in stats.items():
        if not st["total"]:
            report[eng] = {"available": False, "errors": st["errors"]}
            continue
        report[eng] = {"fact_recall_pct": round(100 * st["found"] / st["total"], 1),
                       "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in st["by_kind"].items()},
                       "avg_seconds": round(sum(st["seconds"]) / max(1, len(st["seconds"])), 1),
                       "avg_lines": round(sum(st["lines"]) / max(1, len(st["lines"])), 1),
                       "ads": len(st["seconds"]), "errors": st["errors"]}
    out = set_dir / "report" / "ocr_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
