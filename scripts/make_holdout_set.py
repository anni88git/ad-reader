"""Held-out evaluation set: 20 ads NEVER used while developing or tuning the system.
12 English newspaper ads (Advert_Gallery, brands not in set50) + 8 regional e-paper ads not in set50.
Output: data/eval/holdout20/images + manifest.csv
"""
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import preprocess as pp  # noqa: E402

OUT = ROOT / "data/eval/holdout20"
MAD = ROOT / "data/madverse"
SEED = 777


def main():
    rng = random.Random(SEED)
    used = {r["original"] for r in csv.DictReader((ROOT / "data/eval/set50/manifest.csv").open(encoding="utf-8"))}
    used_brands = {r["brand_label"] for r in csv.DictReader((ROOT / "data/eval/set50/manifest.csv").open(encoding="utf-8"))}
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    rows = []

    def add(data, name, **meta):
        i = len(rows) + 1
        ext = ".png" if name.lower().endswith(".png") else ".jpg"
        dest = OUT / "images" / f"h{i:02d}_{''.join(c if c.isalnum() else '_' for c in Path(name).stem)[:36]}{ext}"
        dest.write_bytes(data)
        rows.append({"id": f"h{i:02d}", "file": dest.relative_to(ROOT).as_posix(), "original": name, **meta})

    ann = json.loads((MAD / "adgal_annot_j.json").read_text(encoding="utf-8"))
    z = zipfile.ZipFile(MAD / "Advert_Gallery.zip")
    index = {"/".join(n.replace("\\", "/").split("/")[-2:]).lower(): n for n in z.namelist() if not n.endswith("/")}
    rng.shuffle(ann)
    seen_brands = set()
    for a in ann:
        if len(rows) >= 12:
            break
        brand = a["hier_annot"][-1]
        k = "/".join(a["img_path"].replace("\\", "/").split("/")[-2:]).lower()
        if brand in used_brands or brand in seen_brands or k not in index or index[k] in used:
            continue
        data = z.read(index[k])
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None or min(im.shape[:2]) < 650:
            continue
        seen_brands.add(brand)
        add(data, index[k], source="madverse_adgal", language_label=a.get("language", ""), brand_label=brand)

    regional = []
    for f in (MAD / "images/epaper2").rglob("*.jpg"):
        if f.name not in used and min(pp.load_image(f).shape[:2]) >= 650:
            regional.append(f)
    rng.shuffle(regional)
    regional.sort(key=lambda f: f.parent.name == "hindi")      # prefer language variety
    for f in regional[:8]:
        add(f.read_bytes(), f.name, source="madverse_epaper2", language_label=f.parent.name.title(), brand_label="")

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "file", "source", "language_label", "brand_label", "degraded_of", "original"])
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in w.fieldnames})
    from collections import Counter
    print(len(rows), "held-out ads;", dict(Counter(r["language_label"].lower() for r in rows)))


if __name__ == "__main__":
    main()
