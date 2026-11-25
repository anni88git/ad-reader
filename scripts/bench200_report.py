"""bench200 results: overall + breakdowns by language, industry, image quality and source, using the
UNCHANGED scorer (evaluate.score_record / evaluate.metrics). Also writes distribution_report.json.

Groups come from the labels (what is printed), not from dataset metadata:
  language  = first non-English language of the label's first accepted language list (else English)
  industry  = top level of the label's first accepted category ("Unlabelled" if no category label)
Uncertainty: 95% bootstrap interval over ADS (fields of one ad are correlated), 2000 resamples, seed 0.

  python scripts/bench200_report.py [--system nemotron+nano]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adreader.config import LABELS_DIR  # noqa: E402
from evaluate import LISTS, _alts, metrics, score_record  # noqa: E402

B = ROOT / "data/eval/bench200"


def first(v):
    a = [x for x in _alts(v) if x not in (None, "", [])]
    return a[0] if a else None


def lang_group(lab, meta_lang):
    v = lab.get("language")
    langs = first(v) if isinstance(v, dict) else v
    if isinstance(langs, list) and langs:
        non_en = [x for x in langs if str(x).lower() != "english"]
        return non_en[0] if non_en else "English"
    return meta_lang or "?"


def industry(lab):
    c = first(lab.get("category")) if "category" in lab else None
    return str(c).split(">")[0].strip() if c else "Unlabelled"


def counts_of(rows):
    c = defaultdict(int)
    for outcome, st in rows:
        c[f"{outcome}|{st}"] += 1
    return c


def summarize(field_rows):
    m = metrics(counts_of(field_rows))
    m["correct"] = sum(1 for o, _ in field_rows if o in ("correct", "correct_empty"))
    m["wrong_values"] = sum(1 for o, _ in field_rows if o in ("wrong", "false_positive"))
    m["wrong_label_present"] = sum(1 for o, _ in field_rows if o == "wrong")
    m["false_positives"] = sum(1 for o, _ in field_rows if o == "false_positive")   # value given, label says not printed
    m["sent_to_review"] = sum(1 for o, s in field_rows if s != "confirmed" and o in ("correct", "wrong", "false_positive"))
    return m


def bootstrap(per_ad, stat, n=2000, seed=0):
    rng = random.Random(seed)
    ads = list(per_ad)
    vals = []
    for _ in range(n):
        sample = [r for _ in ads for r in per_ad[rng.choice(ads)]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 1), round(vals[int(0.975 * len(vals)) - 1], 1)] if vals else None


def acc(rows):
    return 100 * sum(1 for o, _ in rows if o in ("correct", "correct_empty")) / len(rows) if rows else None


def silent_rate(rows):
    conf = [o for o, s in rows if s == "confirmed" and o in ("correct", "wrong", "false_positive")]
    return 100 * sum(1 for o in conf if o != "correct") / len(conf) if conf else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="nemotron+nano")
    a = ap.parse_args()
    man = list(csv.DictReader((B / "manifest.csv").open(encoding="utf-8")))
    per_ad, groups, field_level, lists_tot = {}, defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(Counter)
    dist = {k: Counter() for k in ("language", "industry", "quality", "source")}
    label_status = Counter()
    missing_records = []
    for r in man:
        lab = json.loads((LABELS_DIR / f"{Path(r['file']).stem}.json").read_text(encoding="utf-8"))
        g = {"language": lang_group(lab, r["language_label"]), "industry": industry(lab),
             "quality": r["quality"], "source": r["source"]}
        for k, v in g.items():
            dist[k][v] += 1
        label_status["human-verified" if lab.get("_verified_by") else "claude-drafted (provisional)"] += 1
        rf = B / "runs" / a.system / "records" / f"{r['id']}.json"
        if not rf.exists():
            missing_records.append(r["id"])
            continue
        rec = json.loads(rf.read_text(encoding="utf-8"))
        fields, lists, _ = score_record(rec, lab)
        rows = [(o, st) for f, o, st, sv, lv in fields]
        per_ad[r["id"]] = rows
        for f, o, st, sv, lv in fields:
            field_level[f].append((o, st))
        for k, v in g.items():
            groups[k][v] += rows
        for f, d in lists.items():
            for k in ("tp", "fp", "fn", "confirmed_fp"):
                lists_tot[f][k] += d[k]
    all_rows = [x for rows in per_ad.values() for x in rows]
    n = len(man)
    pct = lambda c: {k: {"count": v, "pct": round(100 * v / n, 1)} for k, v in c.most_common()}  # noqa: E731
    distribution = {"images": n, "language": pct(dist["language"]), "industry": pct(dist["industry"]),
                    "quality": pct(dist["quality"]), "source": pct(dist["source"]),
                    "note": "language and industry taken from the labels (what is printed), not from dataset metadata"}
    (B / "distribution_report.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=1), encoding="utf-8")
    overall = summarize(all_rows)
    overall["accuracy_ci95_ad_bootstrap"] = bootstrap(per_ad, acc)
    overall["silent_error_rate_ci95_ad_bootstrap"] = bootstrap(per_ad, silent_rate)
    res = {"system": a.system, "ads_evaluated": len(per_ad), "ads_missing_record": missing_records,
           "labels": dict(label_status),
           "status": "PROVISIONAL - labels are Claude-drafted; human audit pending" if not label_status["human-verified"] else
                     f"{label_status['human-verified']} human-verified labels; rest provisional",
           "scoring": "evaluate.score_record (unchanged): single fields + category attributes; a value counts as correct "
                      "if it matches any accepted label alternative; a red (missing) field is 'not filled'. Lists "
                      "(phones/websites/emails) are reported separately.",
           "overall": overall,
           "by_field": {f: summarize(v) for f, v in sorted(field_level.items())},
           "lists": {f: dict(v) for f, v in lists_tot.items()}}
    for k in ("language", "industry", "quality", "source"):
        res[f"by_{k}"] = {g: {**summarize(v), "ads": dist[k][g],
                              "accuracy_ci95": bootstrap({i: rows for i, rows in per_ad.items() if i in
                                                          {r['id'] for r in man if (lang_group(json.loads((LABELS_DIR / f"{Path(r['file']).stem}.json").read_text(encoding='utf-8')), r['language_label']) if k == 'language' else industry(json.loads((LABELS_DIR / f"{Path(r['file']).stem}.json").read_text(encoding='utf-8'))) if k == 'industry' else r[k]) == g}}, acc, n=500)}
                          for g, v in sorted(groups[k].items(), key=lambda x: -len(x[1]))}
    (B / "report").mkdir(exist_ok=True)
    (B / "report" / "bench200_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    o = res["overall"]
    print(f"{res['status']}\nads {res['ads_evaluated']} (missing records: {len(missing_records)}) | values {o['values_scored']} | "
          f"accuracy {o['accuracy_pct']}% CI {o['accuracy_ci95_ad_bootstrap']} | wrong values {o['wrong_values']} | "
          f"silent {o['silent_errors']}/{o['confirmed_values']} | auto-fill {o['auto_fill_rate_pct']}% | "
          f"false flags {o['over_flag_rate_pct']}% | missed {o['missed_values']}")
    for k in ("language", "industry", "quality", "source"):
        print(f"-- by {k}")
        for g, m in res[f"by_{k}"].items():
            print(f"   {g:32} ads {m['ads']:3} values {m['values_scored']:4} acc {m['accuracy_pct']}% {m['accuracy_ci95']} "
                  f"silent {m['silent_errors']}/{m['confirmed_values']} auto {m['auto_fill_rate_pct']}% flags {m['over_flag_rate_pct']}%")


if __name__ == "__main__":
    main()
