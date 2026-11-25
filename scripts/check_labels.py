"""Automatic consistency check of ground-truth labels against two independent model readings
(primary extractor + image-only verifier). Labels are NOT changed - suspicious ones are listed.

  suspicious : both models agree with each other but NOT with the label  -> likely label error
  ambiguous  : the two models disagree and the label matches one of them -> genuinely debatable
Writes data/eval/<set>/report/label_check.json
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adreader import normalize as nz  # noqa: E402
from adreader.config import LABELS_DIR  # noqa: E402
from evaluate import matches_label, same_value  # noqa: E402

FIELDS = ["brand", "ad_type", "category", "headline", "price", "language"]


def model_value(raw: dict, f: str):
    if f == "price":
        ps = raw.get("prices") or []
        return nz.parse_amount(ps[0]["amount_text"] or "")[0] if ps else None
    if f == "language":
        return raw.get("language") or None
    v = raw.get(f)
    return v.get("value") if isinstance(v, dict) else v


def main(set_name="set50", system="nemotron", verifier="nano"):
    set_dir = ROOT / "data/eval" / set_name
    rows = list(csv.DictReader((set_dir / "manifest.csv").open(encoding="utf-8")))
    out = {"suspicious": [], "ambiguous": [], "checked_fields": 0, "labels": 0}
    for r in rows:
        if r.get("degraded_of"):
            continue
        lf = LABELS_DIR / f"{Path(r['file']).stem}.json"
        rf = set_dir / "runs" / system / "raw" / f"{r['id']}.json"
        vf = set_dir / "verify" / verifier / f"{r['id']}.json"
        if not (lf.exists() and rf.exists() and vf.exists()):
            continue
        lab = json.loads(lf.read_text(encoding="utf-8"))
        a = json.loads(rf.read_text(encoding="utf-8")).get("raw") or {}
        b = json.loads(vf.read_text(encoding="utf-8")).get("raw") or {}
        out["labels"] += 1
        for f in FIELDS:
            if f not in lab or f in lab.get("_skip", []):
                continue
            va, vb = model_value(a, f), model_value(b, f)
            if va in (None, "", []) or vb in (None, "", []):
                continue
            out["checked_fields"] += 1
            ma, mb = matches_label(f, va, lab[f]), matches_label(f, vb, lab[f])
            entry = {"id": r["id"], "field": f, "label": lab[f], "primary": va, "verifier": vb,
                     "label_source": "human" if lab.get("_verified_by") else lab.get("_label_source", "?")}
            if not ma and not mb and same_value(f, va, vb):
                out["suspicious"].append(entry)
            elif ma != mb:
                out["ambiguous"].append(entry)
    # mark (never change) uncertain label fields so evaluation can report a sensitivity analysis
    marks: dict[str, list] = {}
    for kind in ("suspicious", "ambiguous"):
        for e in out[kind]:
            marks.setdefault(e["id"], []).append({"field": e["field"], "why": kind})
    for r in rows:
        lf = LABELS_DIR / f"{Path(r['file']).stem}.json"
        if not lf.exists() or r.get("degraded_of"):
            continue
        lab = json.loads(lf.read_text(encoding="utf-8"))
        if lab.get("_verified_by"):
            continue                       # human-verified labels are not second-guessed
        lab["_uncertain_fields"] = marks.get(r["id"], [])
        lf.write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    rep = set_dir / "report"
    rep.mkdir(exist_ok=True)
    (rep / "label_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"labels {out['labels']}, fields checked {out['checked_fields']}, "
          f"suspicious {len(out['suspicious'])}, ambiguous {len(out['ambiguous'])}")
    for e in out["suspicious"]:
        print("  SUSPICIOUS", e["id"], e["field"], "| label:", json.dumps(e["label"], ensure_ascii=False)[:70],
              "| both models:", json.dumps(e["primary"], ensure_ascii=False)[:50])


if __name__ == "__main__":
    main(*sys.argv[1:])
