"""Sample ads from the MAdVerse dataset (IIIT Hyderabad, WACV 2024) WITHOUT downloading the
multi-GB zips: files are pulled individually from the Zenodo zips via HTTP range requests.

  python scripts/fetch_madverse.py --source epaper2 --langs Hindi,Marathi --per-group 10
  python scripts/fetch_madverse.py --source adgal --cats vehicles,electronics --per-group 5

Images land in data/madverse/images/<source>/..., plus manifest.csv with the dataset labels
(language for e-paper ads; category / sub-category / brand for Advert_Gallery ads).
Dataset: https://zenodo.org/records/10657763  (cite: Sagar et al., MAdVerse, WACV 2024)
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent.parent
MAD = ROOT / "data" / "madverse"
BASE = "https://zenodo.org/records/10657763/files/{}?download=1"
SOURCES = {
    "epaper1": ("Epaper1.zip", "epaper1_annotation.json"),
    "epaper2": ("Epaper2.zip", "epaper2_annotation.json"),
    "adgal": ("Advert_Gallery.zip", "adgal_annot_j.json"),
}
HEADERS = {"User-Agent": "ad-reader-research"}


def open_zip(name: str, tries: int = 4) -> RemoteZip:
    for i in range(tries):
        try:
            return RemoteZip(BASE.format(name), headers=HEADERS)
        except Exception as e:  # Zenodo intermittently returns 504
            print(f"  open {name} failed ({e}); retry {i + 1}/{tries}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise SystemExit(f"Could not open {name}")


def key_of(path: str) -> str:
    """Annotation img_path -> comparable tail 'Lang/File.jpg' or 'Brand/File.jpg'."""
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]).lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--langs", default="", help="comma list (e-paper sources)")
    ap.add_argument("--cats", default="", help="comma list of top categories (adgal)")
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    zip_name, ann_name = SOURCES[a.source]
    ann = json.loads((MAD / ann_name).read_text(encoding="utf-8"))
    langs = {s.strip().lower() for s in a.langs.split(",") if s.strip()}
    cats = {s.strip().lower() for s in a.cats.split(",") if s.strip()}

    groups: dict[str, list[dict]] = {}
    for rec in ann:
        lang = (rec.get("language") or "").lower()
        cat = (rec.get("hier_annot") or [""])[0].lower()
        if langs and lang not in langs:
            continue
        if cats and cat not in cats:
            continue
        g = cat if a.source == "adgal" else lang
        groups.setdefault(g, []).append(rec)

    rng = random.Random(a.seed)
    wanted = {}
    for g, recs in sorted(groups.items()):
        for rec in rng.sample(recs, min(a.per_group, len(recs))):
            wanted[key_of(rec["img_path"])] = (g, rec)
    print(f"{len(wanted)} images selected from {len(groups)} groups: "
          + ", ".join(f"{g}={min(a.per_group, len(r))}" for g, r in sorted(groups.items())))

    out_dir = MAD / "images" / a.source
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = MAD / "manifest.csv"
    new_file = not manifest.exists()
    with open_zip(zip_name) as z, manifest.open("a", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        if new_file:
            w.writerow(["file", "source", "language", "category", "subcategory", "brand", "ad_type", "zip_path"])
        index = {key_of(n): n for n in z.namelist() if not n.endswith("/")}
        for k, (g, rec) in wanted.items():
            zpath = index.get(k)
            if not zpath:
                print(f"  not in zip: {k}")
                continue
            dest = out_dir / g / Path(zpath).name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                try:
                    dest.write_bytes(z.read(zpath))
                    break
                except Exception as e:
                    print(f"  read failed {zpath} ({e}); retry", file=sys.stderr)
                    time.sleep(3 * (i + 1))
            else:
                continue
            h = rec.get("hier_annot") or ["", "", ""]
            w.writerow([dest.relative_to(ROOT).as_posix(), a.source, rec.get("language"),
                        h[0], h[1] if len(h) > 1 else "", h[-1], rec.get("ad_type"), zpath])
            print(f"  saved {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
