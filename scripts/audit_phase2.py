"""Phase 2 audit (read-only): database linking, causes of unnecessary review flags, OCR by script.

  python scripts/audit_phase2.py [--system nemotron+nano] [--sets set50 holdout20]
Writes data/outputs/audit_phase2.json. Uses the unchanged labels and evaluate.matches_label.
"""
import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adreader import normalize as nz  # noqa: E402
from adreader.config import LABELS_DIR, OUTPUTS_DIR  # noqa: E402
from adreader.matcher import default_matcher  # noqa: E402
from evaluate import SCALARS, _alts, matches_label  # noqa: E402

LINKED = {"brand": "brands", "advertiser": "advertisers"}
SCRIPT_OF_LANG = {"english": "latin", "hindi": "devanagari", "marathi": "devanagari", "tamil": "tamil",
                  "telugu": "telugu", "kannada": "kannada", "bengali": "bengali", "gujarati": "gujarati"}


def norm_reason(r: str) -> str:
    r = re.sub(r"'[^']*'|\"[^\"]*\"|\([^)]*\)|#\d+|\d[\d,.]*%?", "…", r)
    r = re.sub(r"suggestion:.*", "suggestion: …", r)
    return r[:80]


def label_text(v):
    a = [x for x in _alts(v) if x not in (None, "", [])]
    return a[0] if a else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="nemotron+nano")
    ap.add_argument("--sets", nargs="*", default=["set50", "holdout20"])
    a = ap.parse_args()
    m = default_matcher()
    link = {f: Counter() for f in ("brand", "product", "advertiser")}
    link_cases = defaultdict(list)
    flag_reasons = defaultdict(Counter)       # field -> reason -> count, over correct-but-flagged values
    flag_single = defaultdict(Counter)        # flagged by exactly one non-informational reason
    over = Counter()
    correct_n = Counter()
    ocr_by_script = defaultdict(lambda: {"ads": 0, "lines": 0, "conf": [], "fields": Counter(), "found": Counter()})
    for s in a.sets:
        sd = ROOT / "data/eval" / s
        for r in csv.DictReader((sd / "manifest.csv").open(encoding="utf-8")):
            if r.get("degraded_of"):
                continue
            lf = LABELS_DIR / f"{Path(r['file']).stem}.json"
            rf = sd / "runs" / a.system / "records" / f"{r['id']}.json"
            if not (lf.exists() and rf.exists()):
                continue
            lab = json.loads(lf.read_text(encoding="utf-8"))
            rec = json.loads(rf.read_text(encoding="utf-8"))
            F = rec["fields"]
            # ---------------- A. linking
            for f in ("brand", "product", "advertiser"):
                if f not in lab or f in lab.get("_skip", []):
                    continue
                r_ = F.get(f) or {}
                lv = label_text(lab[f])
                sv = r_.get("value")
                ok = matches_label(f, sv, lab[f]) if sv not in (None, "") else False
                eid = r_.get("matched_id") or r_.get("suggested_id")
                # does the LABEL's entity exist in the master DB?
                if f == "product":
                    lab_in_db = bool(lv) and any(nz.key(p.name) == nz.key(str(lv)) for p in m.products)
                    ent_name = next((p.name for p in m.products if p.id == eid), None) if eid else None
                else:
                    res = m.match(str(lv), m.tables[LINKED[f]]) if lv else None
                    lab_in_db = bool(res and res.matched)
                    ent_name = next((e.name for e in m.tables[LINKED[f]] if e.id == eid), None) if eid else None
                if eid and lv and ent_name:
                    good = matches_label(f, ent_name, lab[f]) or nz.key(ent_name) in nz.key(str(lv)) or nz.key(str(lv)) in nz.key(ent_name)
                    kind = "linked_correct" if good else "linked_WRONG"
                elif lv is None:
                    kind = "label_empty"
                elif lab_in_db:
                    kind = "missed_link (label entity exists in DB)"
                else:
                    kind = "not_in_db (correctly unlinked)"
                link[f][kind] += 1
                if kind.startswith(("linked_WRONG", "missed")):
                    link_cases[f].append({"id": f"{s}/{r['id']}", "label": lv, "system": sv, "value_ok": ok,
                                          "db_match": r_.get("db_match"), "entity": ent_name,
                                          "candidates": [c["name"] for c in r_.get("candidates", [])][:3],
                                          "status": r_.get("status")})
            # ---------------- B. why correct values are flagged
            for f in SCALARS:
                if f not in lab or f in lab.get("_skip", []) or f not in F:
                    continue
                r_ = F[f]
                if r_.get("value") in (None, "", []) or not matches_label(f, r_["value"], lab[f]):
                    continue
                correct_n[f] += 1
                if r_["status"] == "confirmed":
                    continue
                over[f] += 1
                rs = [norm_reason(x) for x in r_.get("reasons", [])]
                neg = [x for x in rs if not x.startswith(("evidence verified", "second model agrees", "exact match",
                                                          "matched existing", "amount digits match", "brand printed"))]
                for x in set(neg):
                    flag_reasons[f][x] += 1
                if len(set(neg)) == 1:
                    flag_single[f][neg[0]] += 1
            # ---------------- C. OCR by script / field
            of = sd / "ocr" / f"{r['id']}.pkl"
            if of.exists():
                o = pickle.loads(of.read_bytes())
                langs = [x.strip().lower() for x in r["language_label"].split("+")]
                sc = SCRIPT_OF_LANG.get(next((x for x in langs if x != "english"), "english"), "other")
                st = ocr_by_script[sc]
                st["ads"] += 1
                st["lines"] += len(o.lines)
                st["conf"] += [l.score for l in o.lines]
                corpus = "|".join(nz.key(l.norm) for l in o.lines) + "|".join(nz.key(x[0]) for l in o.lines for x in l.alts)
                for f in ("brand", "headline", "advertiser", "product"):
                    lv = label_text(lab.get(f)) if f in lab and f not in lab.get("_skip", []) else None
                    if lv:
                        st["fields"][f] += 1
                        st["found"][f] += nz.key(str(lv)) in corpus.replace("|", "")    # may span lines
    out = {
        "linking": {f: dict(c) for f, c in link.items()},
        "linking_problem_cases": link_cases,
        "over_flag": {f: f"{over[f]}/{correct_n[f]}" for f in sorted(over, key=lambda f: -over[f])},
        "over_flag_reasons": {f: flag_reasons[f].most_common(8) for f in sorted(over, key=lambda f: -over[f])},
        "over_flag_single_cause": {f: flag_single[f].most_common(5) for f in sorted(over, key=lambda f: -over[f])},
        "ocr_by_script": {sc: {"ads": v["ads"], "avg_lines": round(v["lines"] / max(1, v["ads"]), 1),
                               "mean_conf": round(sum(v["conf"]) / max(1, len(v["conf"])), 3),
                               "label_text_found_in_ocr": {f: f"{v['found'][f]}/{v['fields'][f]}" for f in v["fields"]}}
                          for sc, v in sorted(ocr_by_script.items())},
    }
    p = OUTPUTS_DIR / "audit_phase2.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
