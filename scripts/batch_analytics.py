"""Batch analytics over processed ads: campaign deduplication + trends + review load.

  python scripts/batch_analytics.py [--sets set50 holdout20] [--system nemotron+nano] [--records DIR ...]
  python scripts/batch_analytics.py --bench          # dedup benchmark on synthetic re-runs (no API calls)

Inputs: saved records (data/eval/<set>/runs/<system>/records/*.json, or any --records folder written
by the pipeline) + the ad images + cached OCR text when available.
Outputs: data/outputs/analytics/{summary.json, campaigns.csv, pairs.csv, report.md}
"""
import argparse
import csv
import json
import pickle
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import dedup  # noqa: E402
from adreader.config import OUTPUTS_DIR  # noqa: E402

OUT = OUTPUTS_DIR / "analytics"
PERIODIC = __import__("re").compile(r"\b(emi|per|a year|a month|/month|/year|monthly|yearly|annual|rent|service|subscription)\b")


def _v(rec, f):
    x = rec.get("fields", {}).get(f)
    return x.get("value") if isinstance(x, dict) else x


def load_set(set_name: str, system: str):
    set_dir = ROOT / "data/eval" / set_name
    rows = list(csv.DictReader((set_dir / "manifest.csv").open(encoding="utf-8")))
    items = []
    for r in rows:
        rf = set_dir / "runs" / system / "records" / f"{r['id']}.json"
        if not rf.exists():
            continue
        rec = json.loads(rf.read_text(encoding="utf-8"))
        of = set_dir / "ocr" / f"{r['id']}.pkl"
        lines = [l.norm for l in pickle.loads(of.read_bytes()).lines] if of.exists() else None
        items.append({"id": f"{set_name}/{r['id']}", "rec": rec, "image": ROOT / r["file"], "ocr": lines,
                      "meta": {"set": set_name, "source": r.get("source", ""), "degraded_of": r.get("degraded_of", "")}})
    return items


def load_records_dir(d: Path):
    items = []
    for rf in sorted(d.glob("*.json")):
        rec = json.loads(rf.read_text(encoding="utf-8"))
        img = Path(rec.get("image", "")) if rec.get("image") else None
        items.append({"id": rf.stem, "rec": rec, "image": img if img and img.exists() else None, "ocr": None, "meta": {}})
    return items


def trends(items) -> dict:
    by_cat, by_brand, by_lang, by_type = Counter(), Counter(), Counter(), Counter()
    prices = defaultdict(list)
    offers_ads, flagged, total_fields, unverified_prices = 0, Counter(), Counter(), 0
    for it in items:
        rec = it["rec"]
        cat = (_v(rec, "category") or "Unknown").split(">")[0].strip()
        by_cat[cat] += 1
        by_brand[_v(rec, "brand") or "Unknown"] += 1
        for lang in _v(rec, "language") or ["Unknown"]:
            by_lang[lang] += 1
        by_type[_v(rec, "ad_type") or "Unknown"] += 1
        p = _v(rec, "price")
        pr = rec.get("fields", {}).get("price") or {}
        if isinstance(p, (int, float)) and p > 0:
            ctx = " ".join(f"{(it.get('value') or {}).get('label', '')} {(it.get('value') or {}).get('item', '')}"
                           for it in rec.get("lists", {}).get("prices", [])
                           if isinstance(it.get("value"), dict) and it["value"].get("amount") == p).lower()
            if PERIODIC.search(ctx):                  # EMI / rent / service cost per year: not a selling price
                unverified_prices += 1
            elif pr.get("status") == "confirmed":     # trends use verified prices only
                prices[cat].append(p)
            else:
                unverified_prices += 1
        if rec.get("lists", {}).get("offers"):
            offers_ads += 1
        for f, r in rec.get("fields", {}).items():
            if isinstance(r, dict) and r.get("value") not in (None, "", []):
                total_fields[f] += 1
                if r.get("status") != "confirmed":
                    flagged[f] += 1
    n = max(1, len(items))
    return {
        "ads": len(items),
        "by_category": by_cat.most_common(), "by_brand_top15": by_brand.most_common(15),
        "by_language": by_lang.most_common(), "by_ad_type": by_type.most_common(),
        "price_by_category": {c: {"ads_with_price": len(v), "median": int(statistics.median(v)),
                                  "min": int(min(v)), "max": int(max(v))} for c, v in prices.items()},
        "ads_with_offers_pct": round(100 * offers_ads / n, 1),
        "prices_excluded_unverified": unverified_prices,
        "review_load_by_field_pct": {f: round(100 * flagged[f] / total_fields[f], 1)
                                     for f in sorted(total_fields, key=lambda f: -flagged[f] / total_fields[f])},
    }


def run(items, tag=""):
    sigs = [dedup.signature(it["id"], it["rec"], it["image"], it["ocr"], it["meta"]) for it in items]
    g = dedup.group(sigs)
    t = trends(items)
    # deduplicated brand counts: campaigns and unique creatives instead of raw insertions
    t["brand_campaigns_top15"] = Counter(c["brand"] or "unknown" for c in g["campaign_list"]).most_common(15)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"ads": g["ads"], "unique_creatives": g["unique_creatives"], "campaigns": g["campaigns"],
               "relations": Counter(p["relation"] for p in g["pairs"]), "trends": t}
    (OUT / f"summary{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    with (OUT / f"pairs{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "relation", "image_distance", "text_similarity", "details"])
        for p in g["pairs"]:
            w.writerow([p["a"], p["b"], p["relation"], p["image_distance"], p["text_similarity"],
                        "; ".join(p.get("changes", []) or p.get("why", []))])
    with (OUT / f"campaigns{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["brand", "product", "insertions", "unique_creatives", "relations", "ads"])
        for c in g["campaign_list"]:
            w.writerow([c["brand"], c["product"], c["insertions"], c["unique_creatives"],
                        "|".join(c["relations"]), " ".join(c["ads"])])
    (OUT / f"report{tag}.md").write_text(report_md(summary, g), encoding="utf-8")
    return summary, g


def inr(n: int) -> str:
    """Indian digit grouping: 2799000 -> 27,99,000."""
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + parts + [tail])


def report_md(s, g) -> str:
    t = s["trends"]
    L = [f"# Batch analytics - {s['ads']} ads", "",
         f"**{s['ads']} insertions -> {s['unique_creatives']} unique creatives -> {s['campaigns']} campaigns.** "
         f"Relations found: {dict(s['relations'])}.", "", "## Repeated campaigns", "",
         "| Brand | Product | Insertions | Creatives | Relation | Ads |", "|---|---|---|---|---|---|"]
    for c in g["campaign_list"]:
        if c["insertions"] > 1:
            L.append(f"| {c['brand']} | {c['product']} | {c['insertions']} | {c['unique_creatives']} | "
                     f"{', '.join(c['relations'])} | {' '.join(c['ads'])} |")
    changed = [p for p in g["pairs"] if p["relation"] == dedup.OFFER_CHANGED]
    if changed:
        L += ["", "## Same creative, changed price / offer", ""]
        L += [f"- {p['a']} vs {p['b']}: {'; '.join(p['changes'])}" for p in changed]
    L += ["", "## Mix", "", "| Category | Ads |", "|---|---|"] + [f"| {k} | {v} |" for k, v in t["by_category"]]
    L += ["", "| Language | Ads |", "|---|---|"] + [f"| {k} | {v} |" for k, v in t["by_language"]]
    L += ["", "| Ad type | Ads |", "|---|---|"] + [f"| {k} | {v} |" for k, v in t["by_ad_type"]]
    L += ["", f"## Prices by category (confirmed main prices only; {t['prices_excluded_unverified']} unverified or periodic (EMI/service/rent) excluded)", "", "| Category | Ads with price | Median | Min | Max |", "|---|---|---|---|---|"]
    L += [f"| {c} | {v['ads_with_price']} | Rs {inr(v['median'])} | Rs {inr(v['min'])} | Rs {inr(v['max'])} |"
          for c, v in sorted(t["price_by_category"].items(), key=lambda x: -x[1]["ads_with_price"])]
    L += ["", f"Ads with at least one offer: {t['ads_with_offers_pct']}%", "",
          "## Review load (share of filled values sent to a human)", "", "| Field | Flagged |", "|---|---|"]
    L += [f"| {f} | {v}% |" for f, v in t["review_load_by_field_pct"].items()]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- synthetic dedup benchmark
def _rescan(img, rng):
    """Simulate the same printed ad in another paper / date: scale, crop, blur, JPEG, brightness."""
    h, w = img.shape[:2]
    s = rng.uniform(0.6, 0.9)
    out = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cy, cx = int(out.shape[0] * rng.uniform(0, 0.03)), int(out.shape[1] * rng.uniform(0, 0.03))
    out = out[cy:out.shape[0] - cy, cx:out.shape[1] - cx]
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.85, 1.1), beta=rng.uniform(-15, 15))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(35, 70))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def bench(items, seed=7):
    """Expected relation is known by construction. Measures: correct, missed, wrong links."""
    rng = random.Random(seed)
    base = [it for it in items if it["image"] and not it["meta"].get("degraded_of")]
    sigs, truth, prov = [], {}, {}
    for it in base:
        img = cv2.imdecode(np.fromfile(str(it["image"]), np.uint8), cv2.IMREAD_COLOR)
        sigs.append(dedup.signature(it["id"], it["rec"], img, it["ocr"]))
    for it in rng.sample(base, min(15, len(base))):            # same creative re-run elsewhere
        img = _rescan(cv2.imdecode(np.fromfile(str(it["image"]), np.uint8), cv2.IMREAD_COLOR), rng)
        sid = it["id"] + "#rerun"
        sigs.append(dedup.signature(sid, it["rec"], img, it["ocr"]))
        truth[frozenset((it["id"], sid))] = dedup.DUPLICATE
        prov[frozenset((it["id"], sid))] = "simulated: re-scan (scale/crop/blur/JPEG/brightness)"
    priced = [it for it in base if isinstance(_v(it["rec"], "price"), (int, float))]
    for it in rng.sample(priced, min(10, len(priced))):         # same creative, price changed
        rec = json.loads(json.dumps(it["rec"]))
        old = rec["fields"]["price"]["value"]
        rec["fields"]["price"]["value"] = int(old * rng.choice([0.9, 0.95, 1.05]))
        for p in rec.get("lists", {}).get("prices", []):
            if isinstance(p.get("value"), dict) and p["value"].get("amount") == old:
                p["value"]["amount"] = rec["fields"]["price"]["value"]
        ocr = [l.replace(str(old), str(rec["fields"]["price"]["value"])) for l in it["ocr"]] if it["ocr"] else None
        img = _rescan(cv2.imdecode(np.fromfile(str(it["image"]), np.uint8), cv2.IMREAD_COLOR), rng)
        sid = it["id"] + "#price"
        sigs.append(dedup.signature(sid, rec, img, ocr))
        truth[frozenset((it["id"], sid))] = dedup.OFFER_CHANGED
        prov[frozenset((it["id"], sid))] = "simulated: price edited in record + re-scan"
    # real degraded copies in set50 are true duplicates of their originals
    for it in items:
        if it["meta"].get("degraded_of"):
            orig = f"{it['meta']['set']}/{it['meta']['degraded_of']}"
            img = cv2.imdecode(np.fromfile(str(it["image"]), np.uint8), cv2.IMREAD_COLOR)
            sigs.append(dedup.signature(it["id"], it["rec"], img, it["ocr"]))
            truth[frozenset((orig, it["id"]))] = dedup.DUPLICATE
            prov[frozenset((orig, it["id"]))] = "simulated: set50 degraded copy (our own blur/JPEG)"
    # real repeat found in the data and confirmed by eye: the same Rupa creative in two newspapers
    truth[frozenset(("set50/027", "set50/034"))] = dedup.DUPLICATE
    prov[frozenset(("set50/027", "set50/034"))] = "real: same creative in two newspapers (found by the system, then confirmed - not independent)"
    g = dedup.group(sigs)
    found = {frozenset((p["a"], p["b"])): p for p in g["pairs"] if p["relation"] in dedup.LINKING}
    res = {"expected_links": len(truth), "correct": 0, "wrong_relation": [], "missed": [], "unexpected_links": []}
    res["by_provenance"] = {}
    for k, rel in truth.items():
        p = found.get(k)
        bp = res["by_provenance"].setdefault(prov[k], {"expected": 0, "correct": 0})
        bp["expected"] += 1
        if p is None:
            res["missed"].append(sorted(k))
        elif p["relation"] == rel:
            res["correct"] += 1
            bp["correct"] += 1
        else:
            res["wrong_relation"].append({"pair": sorted(k), "expected": rel, "got": p["relation"]})
    for k, p in found.items():
        if k in truth:
            continue
        # links between two derived copies of the same original are expected too
        roots = {x.split("#")[0] for x in k}
        if roots == {"set50/027", "set50/034"}:
            continue
        origs = {next((f"{it['meta']['set']}/{it['meta']['degraded_of']}" for it in items if it["id"] == x and
                       it["meta"].get("degraded_of")), x) for x in roots}
        if len(origs) == 1:
            continue
        res["unexpected_links"].append({"pair": sorted(k), "relation": p["relation"],
                                        "image_distance": p["image_distance"], "text": p["text_similarity"],
                                        "why": p.get("why") or p.get("changes")})
    res["ads_in_benchmark"] = len(sigs)
    res["status"] = ("DEVELOPMENT BENCHMARK - thresholds were set while looking at it; repeats are simulated "
                     "except one. Not a production accuracy figure. See docs/DEDUP_EVAL_PLAN.md.")
    res["unexpected_links_scope"] = "real ads only, no ground truth for absent links beyond 70 ads"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dedup_bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["set50", "holdout20"])
    ap.add_argument("--system", default="nemotron+nano")
    ap.add_argument("--records", nargs="*", default=[])
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    items = [it for s in a.sets for it in load_set(s, a.system)]
    for d in a.records:
        items += load_records_dir(Path(d))
    if a.bench:
        r = bench(items)
        print(f"dedup benchmark: {r['ads_in_benchmark']} ads, {r['correct']}/{r['expected_links']} expected links correct, "
              f"{len(r['missed'])} missed, {len(r['wrong_relation'])} wrong relation, "
              f"{len(r['unexpected_links'])} unexpected links")
        print("  ", r["status"])
        for k, v in r["by_provenance"].items():
            print(f"   {v['correct']}/{v['expected']}  {k}")
        for k in ("missed", "wrong_relation", "unexpected_links"):
            for x in r[k][:10]:
                print(" ", k, x)
        return
    s, g = run(items)
    print(f"{s['ads']} ads -> {s['unique_creatives']} unique creatives, {s['campaigns']} campaigns; relations {dict(s['relations'])}")
    for c in g["campaign_list"]:
        if c["insertions"] > 1:
            print(f"  {c['brand'] or '?':20} {c['product'] or '':20} {c['insertions']} ads, "
                  f"{c['unique_creatives']} creatives {c['relations']}: {' '.join(c['ads'])}")
    print("outputs ->", OUT)


if __name__ == "__main__":
    main()
