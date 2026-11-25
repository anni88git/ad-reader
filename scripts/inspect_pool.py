"""Inspect candidate images for the 200+ benchmark: quality stats + duplicate / near-duplicate scan.

Pools: data/eval/dedup_real (400), data/eval/set50, data/eval/holdout20 and optionally
data/eval/bench200/extra (newly fetched). Read-only except the output JSON.
Near-duplicates: perceptual-hash distance (64-bit DCT pHash) - 0..6 identical creative,
7..12 likely the same creative (crop / scan differences), checked by eye before exclusion.

  python scripts/inspect_pool.py
Writes data/eval/bench200/pool_inspection.json
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import dedup  # noqa: E402
from adreader import preprocess as pp  # noqa: E402

OUT = ROOT / "data/eval/bench200"


def pools():
    out = []
    for name, man in (("dedup_real", "data/eval/dedup_real/manifest.csv"), ("set50", "data/eval/set50/manifest.csv"),
                      ("holdout20", "data/eval/holdout20/manifest.csv"), ("extra", "data/eval/bench200/extra/manifest.csv")):
        p = ROOT / man
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8")):
            out.append({"pool": name, "id": f"{name}/{r['id']}", "file": r["file"],
                        "language": r.get("language") or r.get("language_label", ""),
                        "source": r.get("newspaper") or r.get("source", ""), "degraded_of": r.get("degraded_of", "")})
    return out


def quality_bucket(w, h, sharp):
    mp = w * h / 1e6
    if mp < 0.5 or sharp < 60:
        return "low"
    if mp < 1.5 or sharp < 200:
        return "medium"
    return "high"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = pools()
    for it in items:
        img = cv2.imdecode(np.fromfile(str(ROOT / it["file"]), np.uint8), cv2.IMREAD_COLOR)
        q = pp.assess_quality(img) if hasattr(pp, "assess_quality") else None
        h, w = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var()) if q is None else q.sharpness
        it.update({"w": w, "h": h, "sharpness": round(sharp, 1), "contrast": round(float(g.std()), 1),
                   "aspect": round(w / h, 2), "quality": quality_bucket(w, h, sharp), "phash": dedup.phash(img)})
    near = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = dedup.hamming(items[i]["phash"], items[j]["phash"])
            if d <= 12:
                near.append({"a": items[i]["id"], "b": items[j]["id"], "distance": d,
                             "same_pool": items[i]["pool"] == items[j]["pool"]})
    for it in items:
        it.pop("phash")
    par = {it["id"]: it["id"] for it in items}

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for x in near:
        par[root(x["a"])] = root(x["b"])
    (OUT / "clusters.json").write_text(json.dumps({k: root(k) for k in par}), encoding="utf-8")
    from collections import Counter
    summary = {
        "images": Counter(it["pool"] for it in items),
        "language_by_pool": {p: Counter(it["language"] for it in items if it["pool"] == p) for p in {it["pool"] for it in items}},
        "quality_by_pool": {p: Counter(it["quality"] for it in items if it["pool"] == p) for p in {it["pool"] for it in items}},
        "near_duplicate_pairs": len(near), "identical_pairs_d0_6": sum(1 for x in near if x["distance"] <= 6),
    }
    (OUT / "pool_inspection.json").write_text(json.dumps({"summary": summary, "near": near, "items": items},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for x in sorted(near, key=lambda x: x["distance"])[:40]:
        print(f"  d={x['distance']:2} {x['a']} ~ {x['b']}")


if __name__ == "__main__":
    main()
