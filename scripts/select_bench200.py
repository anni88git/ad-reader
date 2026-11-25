"""Build the bench200 manifest: 200 NEW distinct creatives (set50 / holdout20 are kept separate).

Rules
  * one image per creative: pHash clusters (distance <= 12, all sampled pairs verified by eye) plus
    manual visual exclusions found during labelling (data/eval/bench200/manual_exclusions.json);
  * creatives already in set50 or holdout20 are excluded;
  * images selected before the target changed are locked (selection_lock.json) so no label is wasted;
  * the rest is filled from the 400-ad Hindi/Marathi pool, stratified by newspaper and quality (equal
    low/medium/high thirds, so low-quality scans are not under-represented), fixed seed;
  * all extra (non-Hindi/Marathi, English) fetched images are used unless they duplicate something.
Selection never uses system output. Writes manifest.csv and duplicate_audit.json.

  python scripts/select_bench200.py [--target 200]
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data/eval/bench200"
PAPER_WEIGHT = {"Navbharat Times": 0.3, "Amar Ujala": 0.3, "Dainik Bhaskar": 0.15, "Maharashtra Times": 0.25}
SEED = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    a = ap.parse_args()
    insp = json.loads((B / "pool_inspection.json").read_text(encoding="utf-8"))
    info = {it["id"]: it for it in insp["items"]}
    cluster = json.loads((B / "clusters.json").read_text(encoding="utf-8"))
    members = defaultdict(list)
    for k, c in cluster.items():
        members[c].append(k)
    lock = json.loads((B / "selection_lock.json").read_text(encoding="utf-8"))["locked"]
    manual = {m["id"]: m for m in json.loads((B / "manual_exclusions.json").read_text(encoding="utf-8"))}
    eval_clusters = {cluster[k] for k in cluster if k.startswith(("set50/", "holdout20/"))}
    pool_rows = {}
    for r in csv.DictReader((ROOT / "data/eval/dedup_real/manifest.csv").open(encoding="utf-8")):
        pool_rows[f"dedup_real/{r['id']}"] = {"file": r["file"], "language": r["language"], "source": r["newspaper"],
                                              "pool": "dedup_real"}
    for r in csv.DictReader((B / "extra/manifest.csv").open(encoding="utf-8")):
        pool_rows[f"extra/{r['id']}"] = {"file": r["file"], "language": r["language"], "source": r["source"],
                                         "pool": "extra", "category_label": r.get("category_label", ""),
                                         "brand_label": r.get("brand_label", "")}
    audit = {"rules": "pHash distance <= 12 = same creative (sampled pairs verified by eye); manual exclusions by eye",
             "excluded": [], "clusters_with_repeats": []}
    chosen, taken = [], set()

    def exclusion_reason(pid):
        c = cluster.get(pid, pid)
        if pid in manual:
            return "manual exclusion (visual duplicate or out of scope)", manual[pid]["reference"], manual[pid]["reason"]
        if c in eval_clusters:
            ref = next(m for m in members[c] if m.startswith(("set50/", "holdout20/")))
            return "creative already in set50/holdout20", ref, "pHash cluster overlaps an existing evaluation set"
        if c in taken:
            ref = next(x for x in chosen if cluster.get(x, x) == c)
            return "duplicate of a selected image", ref, "same pHash cluster"
        return None

    def take(pid):
        why = exclusion_reason(pid)
        if why:
            audit["excluded"].append({"id": pid, "reason": why[0], "reference": why[1], "detail": why[2],
                                      "cluster": cluster.get(pid, pid)})
            return False
        chosen.append(pid)
        taken.add(cluster.get(pid, pid))
        return True

    for pid in lock:                                          # 1. locked (already labelled)
        take(pid)
    for pid in sorted(k for k in pool_rows if k.startswith("extra/") and k not in lock):   # 2. all new extras
        take(pid)
    need = a.target - len(chosen)                             # 3. fill from the 400 pool, stratified
    rng = random.Random(SEED)
    reps = defaultdict(list)
    for pid, r in pool_rows.items():
        if r["pool"] != "dedup_real" or pid in chosen:
            continue
        c = cluster[pid]
        if c in taken or c in eval_clusters:
            continue
        if pid != sorted(m for m in members[c] if m.startswith("dedup_real/"))[0]:
            continue                                          # one representative per creative
        reps[(r["source"], info[pid]["quality"])].append(pid)
    have = Counter(pool_rows[p]["source"] for p in chosen if pool_rows[p]["pool"] == "dedup_real")
    total_400 = sum(have.values()) + max(0, need)
    for paper, w in PAPER_WEIGHT.items():
        k = max(0, round(total_400 * w) - have[paper])
        for i in range(k):
            q = ("low", "medium", "high")[i % 3]
            cand = reps[(paper, q)] or reps[(paper, "medium")] or reps[(paper, "high")] or reps[(paper, "low")]
            if cand and len(chosen) < a.target:
                pid = cand.pop(rng.randrange(len(cand)))
                take(pid)
    # rounding of the per-paper quotas can leave a gap: top up from the remaining representatives
    rest = sorted(p for v in reps.values() for p in v)
    while len(chosen) < a.target and rest:
        take(rest.pop(rng.randrange(len(rest))))
    # repeats inside the pool that were NOT chosen (one representative kept)
    for c, ms in members.items():
        pool_ms = [m for m in ms if m.startswith(("dedup_real/", "extra/"))]
        if len(pool_ms) > 1:
            rep = next((m for m in pool_ms if m in chosen), None)
            audit["clusters_with_repeats"].append({"cluster": c, "members": sorted(ms), "selected": rep,
                                                   "overlaps_eval_set": c in eval_clusters})
    rows = []
    for pid in chosen:
        r, it = pool_rows[pid], info.get(pid, {})
        rows.append({"id": pid.split("/")[-1], "file": r["file"], "pool": r["pool"], "language_label": r["language"],
                     "source": r["source"], "quality": it.get("quality", ""), "width": it.get("w", ""),
                     "height": it.get("h", ""), "sharpness": it.get("sharpness", ""), "cluster": cluster.get(pid, pid),
                     "category_label": r.get("category_label", ""), "brand_label": r.get("brand_label", ""),
                     "orig_id": pid, "locked": pid in lock, "degraded_of": ""})
    with (B / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    audit["summary"] = {"downloaded_400_pool": sum(1 for p in pool_rows if p.startswith("dedup_real/")),
                        "downloaded_extra": sum(1 for p in pool_rows if p.startswith("extra/")),
                        "distinct_creatives_in_400": len({cluster[p] for p in pool_rows if p.startswith("dedup_real/")}),
                        "selected": len(rows), "excluded_logged": len(audit["excluded"]),
                        "excluded_by_reason": Counter(x["reason"] for x in audit["excluded"])}
    (B / "duplicate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(audit["summary"], indent=1))
    print("by language:", Counter(r["language_label"] for r in rows))
    print("by quality:", Counter(r["quality"] for r in rows))
    print("by source:", Counter(r["source"] for r in rows))


if __name__ == "__main__":
    main()
