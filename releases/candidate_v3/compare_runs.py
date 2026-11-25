"""Compare a candidate run against a baseline run with the UNCHANGED scorer (evaluate.score_record /
evaluate.metrics) and the SAME labels. Writes data/eval/<set>/compare/<cand>__vs__<base>.json.

  python scripts/compare_runs.py --set bench200 --base nemotron+nano --cand cand_v3+nano
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adreader.config import LABELS_DIR  # noqa: E402
from bench200_report import industry, lang_group  # noqa: E402
from evaluate import metrics, score_record  # noqa: E402

VALUED = ("correct", "wrong", "false_positive")


def summ(rows):
    c = defaultdict(int)
    for o, s in rows:
        c[f"{o}|{s}"] += 1
    m = metrics(c)
    valued = [s for o, s in rows if o in VALUED]
    m["review_rate_pct"] = round(100 * sum(1 for s in valued if s != "confirmed") / len(valued), 1) if valued else None
    m["correct_auto_accepted"] = sum(1 for o, s in rows if o == "correct" and s == "confirmed")
    m["correct_sent_to_review"] = sum(1 for o, s in rows if o == "correct" and s != "confirmed")
    return m


def load(set_dir, run, man):
    out = {}
    for r in man:
        f = set_dir / "runs" / run / "records" / f"{r['id']}.json"
        if f.exists():
            out[r["id"]] = json.loads(f.read_text(encoding="utf-8"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    d = ROOT / "data/eval" / a.set
    man = list(csv.DictReader((d / "manifest.csv").open(encoding="utf-8")))
    base, cand = load(d, a.base, man), load(d, a.cand, man)
    rows = {"base": defaultdict(list), "cand": defaultdict(list)}
    groups = {k: {"base": defaultdict(list), "cand": defaultdict(list)} for k in ("language", "industry", "quality")}
    changes = []
    for r in man:
        lp = LABELS_DIR / f"{Path(r['file']).stem}.json"
        if not lp.exists() or r["id"] not in base or r["id"] not in cand:
            continue
        lab = json.loads(lp.read_text(encoding="utf-8"))
        g = {"language": lang_group(lab, r.get("language_label")), "industry": industry(lab),
             "quality": r.get("quality", "?")}
        fb, _, _ = score_record(base[r["id"]], lab)
        fc, _, _ = score_record(cand[r["id"]], lab)
        for tag, fields in (("base", fb), ("cand", fc)):
            for f, o, st, sv, lv in fields:
                rows[tag][f].append((o, st))
                for k in g:
                    groups[k][tag][g[k]].append((o, st))
        cb = {f: (o, st, sv) for f, o, st, sv, lv in fb}
        for f, o, st, sv, lv in fc:
            if cb.get(f, (None, None, None))[:2] != (o, st):
                changes.append({"id": r["id"], "field": f, "base": f"{cb[f][0]}|{cb[f][1]}" if f in cb else None,
                                "cand": f"{o}|{st}", "base_value": cb.get(f, (0, 0, None))[2], "cand_value": sv,
                                "label": lv})
    allb = [x for v in rows["base"].values() for x in v]
    allc = [x for v in rows["cand"].values() for x in v]
    res = {"set": a.set, "base": a.base, "cand": a.cand, "ads": len(set(base) & set(cand)),
           "overall": {"base": summ(allb), "cand": summ(allc)},
           "by_field": {f: {"base": summ(rows["base"][f]), "cand": summ(rows["cand"][f])} for f in sorted(rows["base"])},
           **{f"by_{k}": {gname: {"values": len(v), "base": summ(v), "cand": summ(groups[k]["cand"][gname])}
                          for gname, v in sorted(groups[k]["base"].items(), key=lambda x: -len(x[1]))}
              for k in groups},
           "changes": changes}
    out = d / "compare"
    out.mkdir(exist_ok=True)
    (out / f"{a.cand}__vs__{a.base}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                     encoding="utf-8")
    keys = ["accuracy_pct", "confirmed_values", "silent_errors", "confirmed_precision_pct", "auto_fill_rate_pct",
            "correct_auto_accepted", "correct_sent_to_review", "review_rate_pct", "over_flag_rate_pct", "missed_values"]
    print(f"[{a.set}] ads {res['ads']}  {a.base} -> {a.cand}")
    for k in keys:
        print(f"   {k:24} {res['overall']['base'][k]!s:>7} -> {res['overall']['cand'][k]!s:>7}")
    if not a.quiet:
        print(f"   changed field outcomes: {len(changes)}")
        for c in changes:
            print(f"     {c['id']:5} {c['field']:16} {c['base']:24} -> {c['cand']:24} {json.dumps(c['cand_value'], ensure_ascii=False)[:60]}")


if __name__ == "__main__":
    main()
