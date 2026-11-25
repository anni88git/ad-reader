"""Should native PaddleOCR be a SELECTIVE fallback (difficult images / unsupported scripts)?

Per-ad analysis on the held-out 20, using the same facts as compare_ocr.py (headline, phone,
website, price found verbatim) plus headline word recall, and
image-difficulty signals known BEFORE running a second OCR (resolution, blur, contrast, OCR
confidence, script). Read-only: does not change any scoring or evaluation output.

  python scripts/ocr_fallback_analysis.py [--paddle data/eval/kaggle_out/paddle_ocr_all20.jsonl]
Writes data/eval/holdout20/report/ocr_fallback.json
"""
import argparse
import csv
import json
import pickle
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adreader import normalize as nz  # noqa: E402
from adreader.config import LABELS_DIR  # noqa: E402
from compare_ocr import facts  # noqa: E402

SUPPORTED_PADDLE = {"en", "hi", "mr", "ta", "te", "ka"}   # what the Kaggle run actually had models for


def hits(lines, fs):
    key_corpus = "|".join(nz.key(t) for t in lines)
    dig_corpus = "|".join(nz.digits(t) for t in lines)
    out = {}
    for kind, f in fs:
        if not f:
            continue
        h = (f in dig_corpus) if kind in ("phone", "price") else (f in key_corpus.replace("|", "") or f in key_corpus)
        out[f"{kind}:{f[:30]}"] = bool(h)
    return out


def headline_fuzzy(lines, lab):
    """Headline WORD recall: share of the labelled headline's words read exactly somewhere in the
    engine output (robust to the headline being split over several lines)."""
    h = lab.get("headline")
    h = (h.get("any_of") or [None])[0] if isinstance(h, dict) else h
    if not h:
        return None
    words = [nz.key(w) for w in str(h).split() if nz.key(w)]
    seen = {nz.key(w) for t in lines for w in str(t).split()}
    return round(100 * sum(w in seen for w in words) / len(words)) if words else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paddle", default="data/eval/kaggle_out/paddle_ocr_all20.jsonl")
    a = ap.parse_args()
    set_dir = ROOT / "data/eval/holdout20"
    rows = list(csv.DictReader((set_dir / "manifest.csv").open(encoding="utf-8")))
    paddle = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / a.paddle).read_text(encoding="utf-8").splitlines()}
    per_ad, tot = [], {"rapid": 0, "paddle": 0, "union": 0, "facts": 0}
    for r in rows:
        lab = json.loads((LABELS_DIR / f"{Path(r['file']).stem}.json").read_text(encoding="utf-8"))
        rap = pickle.loads((set_dir / "ocr" / f"{r['id']}.pkl").read_bytes())
        rl = [l.norm for l in rap.lines] + [x[0] for l in rap.lines for x in l.alts]
        pd = paddle[r["id"]]
        pl = [x["text"] for x in pd.get("lines") or []]
        fs = facts(lab)
        hr, hp = hits(rl, fs), hits(pl, fs)
        q = rap.quality
        conf = [l.score for l in rap.lines]
        ad = {"id": r["id"], "language": r["language_label"], "paddle_lang": pd["lang"],
              "paddle_has_script_model": pd["lang"] in SUPPORTED_PADDLE,
              "size": f"{q.width}x{q.height}", "sharpness": round(q.sharpness, 1), "contrast": round(q.contrast, 1),
              "quality_warnings": list(q.warnings),
              "rapid_mean_conf": round(statistics.mean(conf), 3) if conf else None,
              "rapid_low_conf_share": round(sum(c < 0.8 for c in conf) / len(conf), 2) if conf else None,
              "facts": len(hr), "rapid": sum(hr.values()), "paddle": sum(hp.values()),
              "union": sum(hr[k] or hp[k] for k in hr),
              "only_rapid": [k for k in hr if hr[k] and not hp[k]], "only_paddle": [k for k in hr if hp[k] and not hr[k]],
              "headline_fuzzy_rapid": headline_fuzzy(rl, lab), "headline_fuzzy_paddle": headline_fuzzy(pl, lab),
              "seconds_rapid": rap.timings_s["total"], "seconds_paddle": pd.get("seconds"), "paddle_scale": pd.get("scale", 1.0)}
        per_ad.append(ad)
        for k in ("rapid", "paddle", "union", "facts"):
            tot[k] += ad[k]
    # would a difficulty trigger have picked the ads where Paddle helps?
    helps = [x["id"] for x in per_ad if x["only_paddle"]]
    hurts = [x["id"] for x in per_ad if x["only_rapid"]]
    triggers = {
        "low_rapid_confidence (mean < 0.9)": lambda x: (x["rapid_mean_conf"] or 1) < 0.9,
        "many_low_conf_lines (>= 25%)": lambda x: (x["rapid_low_conf_share"] or 0) >= 0.25,
        "quality_warning": lambda x: bool(x["quality_warnings"]),
        "regional_script": lambda x: x["language"].lower() != "english",
    }
    trig = {}
    for name, fn in triggers.items():
        sel = [x["id"] for x in per_ad if fn(x)]
        gain = sum(len(x["only_paddle"]) for x in per_ad if x["id"] in sel)
        loss_if_replaced = sum(len(x["only_rapid"]) for x in per_ad if x["id"] in sel)
        trig[name] = {"ads_selected": len(sel), "ids": sel,
                      "facts_gained_if_union": gain, "facts_lost_if_paddle_replaces": loss_if_replaced,
                      "extra_seconds": round(sum(x["seconds_paddle"] or 0 for x in per_ad if x["id"] in sel), 1)}
    fz = [(x["headline_fuzzy_rapid"], x["headline_fuzzy_paddle"]) for x in per_ad if x["headline_fuzzy_rapid"] is not None]
    out = {"status": "measured on 20 held-out ads (real); facts from machine-drafted labels; Paddle on Kaggle CPU, "
                     "RapidOCR on laptop CPU - speeds are not same-hardware",
           "fact_recall": {k: f"{tot[k]}/{tot['facts']}" for k in ("rapid", "paddle", "union")},
           "headline_word_recall_mean": {"rapid": round(statistics.mean(r for r, _ in fz), 1),
                                   "paddle": round(statistics.mean(p for _, p in fz), 1)},
           "ads_where_paddle_adds_facts": helps, "ads_where_rapid_has_facts_paddle_misses": hurts,
           "triggers": trig, "per_ad": per_ad}
    p = set_dir / "report" / "ocr_fallback.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_ad"}, ensure_ascii=False, indent=1))
    print("\nper ad: id lang | rapid/paddle/union of facts | headline words % R/P | only_paddle | only_rapid | conf")
    for x in per_ad:
        print(f" {x['id']} {x['language'][:7]:7} | {x['rapid']}/{x['paddle']}/{x['union']} of {x['facts']} | "
              f"{x['headline_fuzzy_rapid']}/{x['headline_fuzzy_paddle']} | {x['only_paddle']} | {x['only_rapid']} | "
              f"{x['rapid_mean_conf']} {x['quality_warnings']}")


if __name__ == "__main__":
    main()
