"""Score system records against human-verified labels (data/labels/FORMAT.md).

  python scripts/evaluate.py --systems nemotron qwen3vl8b qwen25vl7b [--set set50] [--include-drafts]

Per field value, the outcome is one of
  correct  / wrong / missed (label has value, system empty) / false_positive (label null, system value)
crossed with the system status (confirmed / review / missing). Key metrics:
  * silent error rate   = wrong-or-false-positive CONFIRMED values / all confirmed values   (target ~0)
  * confirmed precision = 1 - silent error rate
  * auto-fill rate      = correct confirmed values / label values present  (manual work saved)
  * error catch rate    = wrong values that were flagged (review/missing) / all wrong values
  * over-flag rate      = correct values that were flagged / all correct values
  * list precision / recall (phones, websites, features, offers ...)
Only labels with "_verified_by" set are used unless --include-drafts.
Writes data/eval/<set>/report/summary.json + per_field.csv + errors.csv
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import normalize as nz  # noqa: E402
from adreader.config import LABELS_DIR  # noqa: E402

EXACT_TEXT = {"headline", "sub_headline"}            # must be verbatim (normalised)
FUZZY_TEXT = {"description": 90, "warranty": 88, "call_to_action": 88}
NAME_FIELDS = {"brand", "product", "model_variant", "advertiser"}
ENUM_FIELDS = {"ad_type", "category"}
SCALARS = ["brand", "product", "model_variant", "advertiser", "ad_type", "category", "headline",
           "sub_headline", "description", "price", "warranty", "call_to_action", "language",
           "qr_code_present", "disclaimer_present"]
LISTS = ["phones", "websites", "emails", "features", "offers", "reference_numbers"]
ID_FIELDS = ["brand", "product", "category", "advertiser"]


# ----------------------------------------------------------------------------- comparison
def _alts(lab):
    if isinstance(lab, dict) and "any_of" in lab:
        return lab["any_of"]
    return [lab]


def _empty(v):
    return v in (None, "", [], {})


def same_value(field, sysv, labv) -> bool:
    if _empty(sysv) and _empty(labv):
        return True
    if _empty(sysv) or _empty(labv):
        return False
    if field == "price":
        try:
            return int(sysv) == int(labv)
        except (TypeError, ValueError):
            return False
    if field in ("qr_code_present", "disclaimer_present"):
        return bool(sysv) == bool(labv)
    if field == "language":
        norm = lambda x: {str(i).strip().lower() for i in (x if isinstance(x, list) else [x])}  # noqa: E731
        return norm(sysv) == norm(labv)
    if field in ENUM_FIELDS:
        return nz.key(str(sysv)) == nz.key(str(labv))
    a, b = nz.key(str(sysv)), nz.key(str(labv))
    if field in EXACT_TEXT:
        return a == b
    if field in FUZZY_TEXT:
        return fuzz.ratio(a, b) >= FUZZY_TEXT[field]
    if field == "brand" and _same_brand_record(str(sysv), str(labv)):
        return True
    if field in NAME_FIELDS:
        if field == "advertiser":    # order-insensitive for 'A / B'
            sa = sorted(nz.key(x) for x in str(sysv).split("/"))
            sb = sorted(nz.key(x) for x in str(labv).split("/"))
            return sa == sb
        return a == b or fuzz.ratio(a, b) >= 92
    return a == b


_BRAND_CACHE = {}


def _same_brand_record(a: str, b: str) -> bool:
    """Two brand spellings are the same if they resolve to the same master record
    ('Hero' and 'Hero MotoCorp' -> record #19)."""
    from adreader.matcher import default_matcher
    m = default_matcher()
    ids = []
    for v in (a, b):
        if v not in _BRAND_CACHE:
            r = m.brand(v)
            _BRAND_CACHE[v] = r.matched.id if r.matched else None
        ids.append(_BRAND_CACHE[v])
    return ids[0] is not None and ids[0] == ids[1]


def matches_label(field, sysv, lab) -> bool:
    return any(same_value(field, sysv, alt) for alt in _alts(lab))


def item_key(field, v):
    if isinstance(v, dict):
        v = v.get("number") or v.get("value") or json.dumps(v, ensure_ascii=False)
    s = str(v)
    if field == "phones":
        return nz.parse_phone(s).number
    if field == "websites":
        return nz.key(s.lower().replace("https://", "").replace("http://", "").replace("www.", "")).rstrip("/")
    return nz.key(s)


def list_match(field, sys_items, lab):
    """Greedy one-to-one matching. Returns (tp, fp, fn, details)."""
    if isinstance(lab, dict) and "required" in lab:
        req, opt = lab.get("required", []), lab.get("optional", [])
    else:
        req, opt = (lab or []), []
    thr = 100 if field in ("phones", "websites", "emails", "reference_numbers") else 85
    lab_items = [(item_key(field, x), True) for x in req] + [(item_key(field, x), False) for x in opt]
    used = set()
    tp = fp = 0
    wrong = []
    for s in sys_items:
        sk = item_key(field, s["value"])
        best, best_i = 0, None
        for i, (lk, _) in enumerate(lab_items):
            if i in used:
                continue
            if sk == lk:
                sc = 100
            elif field == "reference_numbers" and min(len(sk), len(lk)) >= 6 and (lk in sk or sk in lk):
                sc = 100      # same identifier, printed label included or not ('Plan No. 857' vs '857...')
            else:
                sc = fuzz.ratio(sk, lk) if thr < 100 else 0
            if sc > best:
                best, best_i = sc, i
        if best_i is not None and best >= thr:
            used.add(best_i)
            tp += 1
        else:
            fp += 1
            wrong.append((s["value"], s["status"]))
    fn = sum(1 for i, (_, required) in enumerate(lab_items) if required and i not in used)
    return tp, fp, fn, wrong


# ----------------------------------------------------------------------------- per record
EXCLUDE_UNCERTAIN = False


def score_record(rec, lab):
    out = []        # (field, outcome, status, system_value, label)
    skip = set(lab.get("_skip", []))
    if EXCLUDE_UNCERTAIN:
        skip |= {u["field"] for u in lab.get("_uncertain_fields", [])}
    for f in SCALARS:
        if f in skip or f not in lab:
            continue
        r = rec["fields"].get(f) or {"value": None, "status": "missing"}
        sv, st = r.get("value"), r.get("status", "missing")
        filled = not _empty(sv) and st != "missing"          # red values are NOT filled in the form
        labv = lab[f]
        lab_present = not all(_empty(a) for a in _alts(labv))
        null_ok = any(_empty(a) for a in _alts(labv))
        if not filled:
            outcome = "correct_empty" if null_ok else "missed"
        elif matches_label(f, sv, labv):
            outcome = "correct"
        else:
            outcome = "false_positive" if not lab_present else "wrong"
        out.append((f, outcome, st, sv, labv))
    attrs_lab = lab.get("category_attributes", {}) or {}
    for k, labv in attrs_lab.items():
        r = rec.get("category_attributes", {}).get(k) or {"value": None, "status": "missing"}
        sv, st = r.get("value"), r.get("status", "missing")
        filled = not _empty(sv) and st != "missing"
        null_ok = any(_empty(a) for a in _alts(labv))
        if not filled:
            outcome = "correct_empty" if null_ok else "missed"
        else:
            if k == "fuel_type":
                ok = any(sorted(map(str.lower, sv if isinstance(sv, list) else [sv])) ==
                         sorted(map(str.lower, a if isinstance(a, list) else [a])) for a in _alts(labv) if a)
            else:
                ok = any(not _empty(a) and fuzz.ratio(nz.key(str(sv)), nz.key(str(a))) >= 90 for a in _alts(labv))
            outcome = "correct" if ok else ("false_positive" if all(_empty(a) for a in _alts(labv)) else "wrong")
        out.append((f"attr.{k}", outcome, st, sv, labv))
    lists = {}
    for f in LISTS:
        if f in skip or f not in lab:
            continue
        items = [x for x in rec["lists"].get(f, []) if x.get("status") != "missing"]
        tp, fp, fn, wrong = list_match(f, items, lab[f])
        conf_wrong = sum(1 for v, s in wrong if s == "confirmed")
        lists[f] = {"tp": tp, "fp": fp, "fn": fn, "confirmed_fp": conf_wrong, "wrong_items": wrong}
    ids = {}
    for f in ID_FIELDS:
        if f"{f}_id" in lab:
            ids[f] = {"label": lab[f"{f}_id"], "system": (rec["fields"].get(f) or {}).get("matched_id")}
    return out, lists, ids


# ----------------------------------------------------------------------------- aggregate
def evaluate(set_dir, systems, include_drafts):
    manifest = list(csv.DictReader((set_dir / "manifest.csv").open(encoding="utf-8")))
    labels = {}
    for r in manifest:
        stem = Path(r["file"]).stem
        base = next((m for m in manifest if m["id"] == r.get("degraded_of")), None) if r.get("degraded_of") else None
        lf = LABELS_DIR / f"{stem}.json"
        if not lf.exists() and base:
            lf = LABELS_DIR / f"{Path(base['file']).stem}.json"         # degraded copy shares labels
        if lf.exists():
            lab = json.loads(lf.read_text(encoding="utf-8"))
            if lab.get("_verified_by") or lab.get("_label_source") or include_drafts:
                labels[r["id"]] = (lab, r)
    report = {"labelled_ads": len(labels), "systems": {}}
    per_field_rows, error_rows = [], []
    for system in systems:
        agg = defaultdict(int)
        by_field = defaultdict(lambda: defaultdict(int))
        by_group = defaultdict(lambda: defaultdict(int))
        list_tot = defaultdict(lambda: defaultdict(int))
        id_tot = defaultdict(int)
        n_rec, times, no_record = 0, [], 0
        for i, (lab, row) in labels.items():
            rf = set_dir / "runs" / system / "records" / f"{i}.json"
            if not rf.exists():
                no_record += 1
                continue
            rec = json.loads(rf.read_text(encoding="utf-8"))
            n_rec += 1
            times.append(sum((rec.get("timings_s") or {}).values()))
            group = "degraded" if row["source"].startswith("degraded") else (
                "english" if row["language_label"].lower() == "english" else "regional")
            fields, lists, ids = score_record(rec, lab)
            for f, outcome, st, sv, labv in fields:
                key = f"{outcome}|{st}"
                agg[key] += 1
                by_field[f][key] += 1
                by_group[group][key] += 1
                if outcome in ("wrong", "false_positive", "missed"):
                    error_rows.append({"system": system, "id": i, "field": f, "outcome": outcome, "status": st,
                                       "system_value": json.dumps(sv, ensure_ascii=False)[:200],
                                       "label": json.dumps(labv, ensure_ascii=False)[:200]})
            for f, d in lists.items():
                for k in ("tp", "fp", "fn", "confirmed_fp"):
                    list_tot[f][k] += d[k]
                for v, s in d["wrong_items"]:
                    error_rows.append({"system": system, "id": i, "field": f, "outcome": "extra_item", "status": s,
                                       "system_value": json.dumps(v, ensure_ascii=False)[:200], "label": ""})
            for f, d in ids.items():
                if d["system"] is None:
                    id_tot["id_empty"] += 1
                elif d["system"] == d["label"]:
                    id_tot["id_correct"] += 1
                else:
                    id_tot["id_wrong"] += 1
                    error_rows.append({"system": system, "id": i, "field": f"{f}_id", "outcome": "wrong_id",
                                       "status": "confirmed", "system_value": d["system"], "label": d["label"]})
        report["systems"][system] = {"records": n_rec, "labelled_without_record": no_record, **metrics(agg), "lists": lists_metrics(list_tot),
                                     "ids": dict(id_tot), "by_group": {g: metrics(v) for g, v in by_group.items()},
                                     "avg_seconds_per_ad": round(sum(times) / len(times), 1) if times else None}
        for f, v in by_field.items():
            per_field_rows.append({"system": system, "field": f, **metrics(v)})
    return report, per_field_rows, error_rows


def metrics(c):
    g = lambda *keys: sum(v for k, v in c.items() if k.split("|")[0] in keys)  # noqa: E731
    conf = lambda *outs: sum(v for k, v in c.items() if k.split("|")[1] == "confirmed" and k.split("|")[0] in outs)  # noqa: E731
    flagged = lambda *outs: sum(v for k, v in c.items() if k.split("|")[1] != "confirmed" and k.split("|")[0] in outs)  # noqa: E731
    confirmed_total = conf("correct", "wrong", "false_positive")
    silent = conf("wrong", "false_positive")
    wrong_total = g("wrong", "false_positive")
    correct_total = g("correct")
    label_present = g("correct", "wrong", "missed")
    pct = lambda a, b: round(100 * a / b, 1) if b else None  # noqa: E731
    return {
        "values_scored": sum(c.values()),
        "accuracy_pct": pct(g("correct", "correct_empty"), sum(c.values())),
        "confirmed_values": confirmed_total,
        "silent_errors": silent,
        "silent_error_rate_pct": pct(silent, confirmed_total),
        "confirmed_precision_pct": pct(confirmed_total - silent, confirmed_total),
        "auto_fill_rate_pct": pct(conf("correct"), label_present),
        "error_catch_rate_pct": pct(flagged("wrong", "false_positive"), wrong_total),
        "over_flag_rate_pct": pct(flagged("correct"), correct_total),
        "missed_values": g("missed"),
    }


def lists_metrics(t):
    out = {}
    for f, d in t.items():
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else None
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else None
        out[f] = {**d, "precision_pct": round(100 * p, 1) if p is not None else None,
                  "recall_pct": round(100 * r, 1) if r is not None else None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["nemotron"])
    ap.add_argument("--set", default="set50")
    ap.add_argument("--include-drafts", action="store_true")
    ap.add_argument("--exclude-uncertain", action="store_true",
                    help="sensitivity analysis: ignore label fields marked uncertain by check_labels.py")
    a = ap.parse_args()
    EXCLUDE_UNCERTAIN = a.exclude_uncertain
    set_dir = ROOT / "data/eval" / a.set
    report, per_field, errors = evaluate(set_dir, a.systems, a.include_drafts)
    out = set_dir / ("report_certain_only" if a.exclude_uncertain else "report")
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, rows in (("per_field.csv", per_field), ("errors.csv", errors)):
        if rows:
            with (out / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(json.dumps({s: {k: v for k, v in d.items() if k not in ("lists", "by_group")}
                      for s, d in report["systems"].items()}, indent=1), f"\nlabelled ads: {report['labelled_ads']}")
