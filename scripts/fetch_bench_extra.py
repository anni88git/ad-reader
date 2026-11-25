"""Fetch the extra ads the 200+ benchmark needs for language and industry coverage (MAdVerse, public).

The 400-ad dedup pool is Hindi/Marathi only, so the other scripts come from Epaper2 and English
ads from Advert_Gallery (spread over categories, brands not already in set50 / holdout20).
Images already used in set50 / holdout20 are excluded by file name.

  python scripts/fetch_bench_extra.py
Writes data/eval/bench200/extra/{images/, manifest.csv}
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_madverse import MAD, SOURCES, key_of, open_zip  # noqa: E402

OUT = ROOT / "data/eval/bench200/extra"
EPAPER = {"Tamil": 10, "Telugu": 10, "Kannada": 10, "Bengali": 8, "Gujarati": 8, "Malayalam": 6, "Odia": 6, "Urdu": 4}
ADGAL = {"travel": 2, "food": 2, "financial_institutions": 2, "cosmetics": 1, "home_essentials": 1, "drinks": 1,
         "body_wear": 1, "electronics": 1, "vehicles": 1}
SEED = 2026
# round 2 (200 NEW creatives target): more English industries + scripts not yet covered (Epaper1)
ROUND2_EPAPER1 = {"Punjabi": 5, "Assamese": 5}
ROUND2_EPAPER2 = {"Malayalam": 3, "Odia": 2}
ROUND2_ADGAL = {"travel": 3, "food": 2, "financial_institutions": 3, "cosmetics": 2, "home_essentials": 2,
                "drinks": 2, "baby_products": 1, "sports": 1}


def used_names():
    names, brands = set(), set()
    for man in ("data/eval/set50/manifest.csv", "data/eval/holdout20/manifest.csv"):
        for r in csv.DictReader((ROOT / man).open(encoding="utf-8")):
            names.add(Path(r.get("original") or r["file"]).name.lower())
            if r.get("brand_label"):
                brands.add(r["brand_label"].lower())
    return names, brands


def fetch(zip_name, picks, rows, prefix):
    with open_zip(zip_name) as z:
        index = {key_of(n): n for n in z.namelist() if not n.endswith("/")}
        for meta, rec in picks:
            zpath = index.get(key_of(rec["img_path"]))
            if not zpath:
                continue
            ad_id = f"{prefix}{len(rows) + 1:03d}"
            dest = OUT / "images" / f"{ad_id}_{Path(zpath).name.replace(' ', '_')}"
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
            rows.append({"id": ad_id, "file": dest.relative_to(ROOT).as_posix(), **meta, "zip_path": zpath})
            print("  saved", dest.name, flush=True)


def main_round2():
    """Second fetch: excludes everything already fetched (round 1 manifest) and set50/holdout20 images."""
    rng = random.Random(SEED + 2)
    names, brands = used_names()
    man = OUT / "manifest.csv"
    rows = list(csv.DictReader(man.open(encoding="utf-8")))
    have = {Path(r["zip_path"]).name.lower() for r in rows}
    have_brands = {r["brand_label"].lower() for r in rows if r.get("brand_label")}
    n0 = len(rows)
    for zipkey, quota in (("epaper1", ROUND2_EPAPER1), ("epaper2", ROUND2_EPAPER2)):
        ep = json.loads((MAD / SOURCES[zipkey][1]).read_text(encoding="utf-8"))
        picks = []
        for lang, n in quota.items():
            pool = [r for r in ep if r["language"] == lang and
                    Path(r["img_path"].replace("\\", "/")).name.lower() not in names | have]
            picks += [({"source": f"madverse_{zipkey}", "language": lang, "category_label": "", "brand_label": ""}, r)
                      for r in rng.sample(pool, n)]
        fetch(SOURCES[zipkey][0], picks, rows, "e")
    ag = json.loads((MAD / SOURCES["adgal"][1]).read_text(encoding="utf-8"))
    picks = []
    for cat, n in ROUND2_ADGAL.items():
        by_brand = {}
        for r in ag:
            h = r.get("hier_annot") or ["", "", ""]
            if h[0] == cat and h[-1].lower() not in brands | have_brands:
                by_brand.setdefault(h[-1], []).append(r)
        for b in rng.sample(sorted(by_brand), min(n, len(by_brand))):
            picks.append(({"source": "madverse_adgal", "language": "English", "category_label": cat, "brand_label": b},
                          rng.choice(by_brand[b])))
    fetch(SOURCES["adgal"][0], picks, rows, "e")
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows) - n0, "round-2 ads added ->", man)


def main():
    rng = random.Random(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    names, brands = used_names()
    rows = []
    ep = json.loads((MAD / SOURCES["epaper2"][1]).read_text(encoding="utf-8"))
    picks = []
    for lang, n in EPAPER.items():
        pool = [r for r in ep if r["language"] == lang and Path(r["img_path"].replace("\\", "/")).name.lower() not in names]
        picks += [({"source": "madverse_epaper2", "language": lang, "category_label": "", "brand_label": ""}, r)
                  for r in rng.sample(pool, n)]
    fetch(SOURCES["epaper2"][0], picks, rows, "e")
    ag = json.loads((MAD / SOURCES["adgal"][1]).read_text(encoding="utf-8"))
    picks = []
    for cat, n in ADGAL.items():
        by_brand = {}
        for r in ag:
            h = r.get("hier_annot") or ["", "", ""]
            if h[0] == cat and h[-1].lower() not in brands:
                by_brand.setdefault(h[-1], []).append(r)
        for b in rng.sample(sorted(by_brand), n):
            r = rng.choice(by_brand[b])
            picks.append(({"source": "madverse_adgal", "language": "English", "category_label": cat,
                           "brand_label": b}, r))
    fetch(SOURCES["adgal"][0], picks, rows, "e")
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "extra ads ->", OUT / "manifest.csv")


if __name__ == "__main__":
    main_round2() if "--round2" in sys.argv else main()
