"""Prepare the REAL-ad deduplication evaluation (docs/DEDUP_EVAL_PLAN.md). Nothing is scored here.

Data: MAdVerse Epaper2 (public, Zenodo). Files carry no dates, but each newspaper's ads are numbered
in collection order, so a contiguous block from one paper spans many issues - natural repeats of
the same creative on different days occur; the same national campaigns also run across papers.

Steps (all reproducible, no model calls, no cost):
  fetch   download a contiguous block of N ads per newspaper            -> data/eval/dedup_real/images
  sheets  contact sheets per newspaper (numbered thumbnails) for annotators -> .../sheets/*.jpg
  freeze  sha256 of the dedup code + thresholds, BEFORE any extraction    -> .../FROZEN.json
  template annotation file (groups of ads; who annotated; human / machine) -> .../annotations.json

  python scripts/prepare_dedup_eval.py fetch [--per-paper 100]
  python scripts/prepare_dedup_eval.py sheets
  python scripts/prepare_dedup_eval.py freeze
Split (fixed now, before any result): dev = Dainik Bhaskar, test = the other three papers.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data/eval/dedup_real"
PAPERS = [("Hindi", "Navbharat Times"), ("Hindi", "Amar Ujala "), ("Hindi", "Dainik Bhaskar"),
          ("Marathi", "Maharashtra Times")]
DEV_PAPERS = {"Dainik Bhaskar"}
START = 1


def fetch(per_paper: int):
    from fetch_madverse import MAD, SOURCES, key_of, open_zip
    ann = json.loads((MAD / SOURCES["epaper2"][1]).read_text(encoding="utf-8"))
    by_paper = {}
    for rec in ann:
        name = rec["img_path"].replace("\\", "/").split("/")[-1]
        m = re.match(r"(.+)_(\d+)\.jpg$", name)
        if m:
            by_paper.setdefault((rec["language"], m.group(1)), {})[int(m.group(2))] = rec
    wanted = []
    for lang, paper in PAPERS:
        idx = sorted(i for i in by_paper.get((lang, paper), {}) if i >= START)[:per_paper]
        wanted += [(lang, paper, i, by_paper[(lang, paper)][i]) for i in idx]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    with open_zip(SOURCES["epaper2"][0]) as z:
        index = {key_of(n): n for n in z.namelist() if not n.endswith("/")}
        for k, (lang, paper, i, rec) in enumerate(wanted, 1):
            zpath = index.get(key_of(rec["img_path"]))
            if not zpath:
                continue
            ad_id = f"d{k:03d}"
            dest = OUT / "images" / f"{ad_id}_{paper.strip().replace(' ', '_')}_{i}.jpg"
            if not dest.exists():
                for t in range(3):
                    try:
                        dest.write_bytes(z.read(zpath))
                        break
                    except Exception as e:
                        print("  retry", zpath, e, file=sys.stderr)
                        time.sleep(3 * (t + 1))
                else:
                    continue
            rows.append({"id": ad_id, "file": dest.relative_to(ROOT).as_posix(), "newspaper": paper.strip(),
                         "language": lang, "paper_index": i,
                         "split": "dev" if paper.strip() in DEV_PAPERS else "test", "zip_path": zpath})
            if k % 25 == 0:
                print(f"  {k}/{len(wanted)}", flush=True)
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} ads -> {OUT / 'manifest.csv'}")


def sheets(cols: int = 8, thumb: int = 240):
    rows = list(csv.DictReader((OUT / "manifest.csv").open(encoding="utf-8")))
    (OUT / "sheets").mkdir(exist_ok=True)
    by = {}
    for r in rows:
        by.setdefault(r["newspaper"], []).append(r)
    for paper, rs in by.items():
        for page in range(0, len(rs), cols * 5):
            chunk = rs[page:page + cols * 5]
            tiles = []
            for r in chunk:
                im = cv2.imdecode(np.fromfile(str(ROOT / r["file"]), np.uint8), cv2.IMREAD_COLOR)
                h, w = im.shape[:2]
                s = thumb / max(h, w)
                im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                tile = np.full((thumb + 30, thumb, 3), 255, np.uint8)
                tile[:im.shape[0], :im.shape[1]] = im
                cv2.putText(tile, r["id"], (5, thumb + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
                tiles.append(tile)
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            p = OUT / "sheets" / f"{paper.replace(' ', '_')}_{page // (cols * 5) + 1}.jpg"
            cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheets ->", OUT / "sheets")
    tpl = OUT / "annotations.json"
    if not tpl.exists():
        tpl.write_text(json.dumps({
            "_instructions": "Groups of ad ids that are the same creative (duplicate), the same creative with a "
                             "changed price/offer (offer_changed) or a new version of one campaign (new_version). "
                             "Annotate WITHOUT looking at the system's dedup output. Ads in no group are unique.",
            "annotator": "", "annotator_type": "human | machine (state which)", "annotated_at": "",
            "groups": [{"relation": "duplicate", "ads": ["d001", "d057"], "note": ""}]}, indent=1), encoding="utf-8")


def freeze():
    from adreader import dedup
    files = ["adreader/dedup.py", "scripts/batch_analytics.py"]
    info = {"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
            "thresholds": {k: getattr(dedup, k) for k in
                           ("IMG_SAME", "IMG_SIMILAR", "TEXT_SAME", "TEXT_VERSION", "HEADLINE_VERSION", "CREATIVE_VERSION")},
            "split": {"dev": sorted(DEV_PAPERS), "test": [p.strip() for _, p in PAPERS if p.strip() not in DEV_PAPERS]},
            "rule": "test split is scored ONCE with these exact files; any change after that must be re-frozen "
                    "and evaluated on new data"}
    (OUT / "FROZEN.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "sheets", "freeze"])
    ap.add_argument("--per-paper", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"fetch": lambda: fetch(a.per_paper), "sheets": sheets, "freeze": freeze}[a.cmd]()
