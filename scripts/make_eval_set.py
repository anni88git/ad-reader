"""Build the 50-ad evaluation set (reproducible, seed fixed).

Composition (covers industries, languages and image qualities - per the assignment):
  * 3  reference/test ads we collected (Kia from the brief, 2 Hindi ads from Amar Ujala)
  * 22 English newspaper ads from MAdVerse Advert_Gallery, stratified by industry
  * 22 regional-language e-paper ads from MAdVerse Epaper2 (Hindi-heavy + 6 other languages)
  * 3  synthetically degraded copies (blur+JPEG, tilt, phone-photo perspective) of set members,
       re-using their labels -> measures robustness to image quality
Output: data/eval/set50/images/*.jpg|png + manifest.csv
"""
import csv
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

OUT = ROOT / "data/eval/set50"
IMG = OUT / "images"
MAD = ROOT / "data/madverse"
SEED = 50

ADGAL_QUOTAS = {  # top category -> count (22)
    "vehicles": 5, "electronics": 4, "financial_institutions": 3, "body_wear": 3,
    "food": 2, "travel": 2, "cosmetics": 1, "home_essentials": 1, "drinks": 1,
}
EPAPER_QUOTAS = {"hindi": 10, "marathi": 3, "tamil": 2, "telugu": 2, "kannada": 2, "gujarati": 2, "bengali": 1}
MIN_SIDE = 650   # skip thumbnails: unreadable even for a human labeller


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")[:40]


def main():
    import json
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    IMG.mkdir(parents=True)
    rows = []

    def add(src_bytes_or_path, name, **meta):
        i = len(rows) + 1
        ext = ".png" if str(name).lower().endswith(".png") else ".jpg"
        dest = IMG / f"{i:03d}_{slug(Path(name).stem)}{ext}"
        if isinstance(src_bytes_or_path, (bytes, bytearray)):
            dest.write_bytes(src_bytes_or_path)
        else:
            shutil.copy(src_bytes_or_path, dest)
        rows.append({"id": f"{i:03d}", "file": dest.relative_to(ROOT).as_posix(), "original": str(name), **meta})
        return dest

    # ---- 3 own ads
    for p, lang in [("data/ads/_kia_from_screenshot.jpg", "English"), ("data/ads/002_horlicks_hi.png", "Hindi"),
                    ("data/ads/003_namaste_butter_hi.png", "Hindi")]:
        add(ROOT / p, p, source="own", language_label=lang, category_label="", brand_label="")

    # ---- 22 Advert_Gallery (English newspaper ads), one per brand, stratified
    ann = json.loads((MAD / "adgal_annot_j.json").read_text(encoding="utf-8"))
    z = zipfile.ZipFile(MAD / "Advert_Gallery.zip")
    index = {"/".join(n.replace("\\", "/").split("/")[-2:]).lower(): n for n in z.namelist() if not n.endswith("/")}
    by_cat: dict[str, list] = {}
    for a in ann:
        by_cat.setdefault(a["hier_annot"][0], []).append(a)
    for cat, n in ADGAL_QUOTAS.items():
        pool = by_cat.get(cat, [])[:]
        rng.shuffle(pool)
        brands_used, taken = set(), 0
        for a in pool:
            if taken >= n:
                break
            brand = a["hier_annot"][-1]
            k = "/".join(a["img_path"].replace("\\", "/").split("/")[-2:]).lower()
            if brand in brands_used or k not in index:
                continue
            data = z.read(index[k])
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if im is None or min(im.shape[:2]) < MIN_SIDE:
                continue
            brands_used.add(brand)
            add(data, index[k], source="madverse_adgal", language_label=a.get("language", ""),
                category_label="/".join(a["hier_annot"][:2]), brand_label=brand)
            taken += 1

    # ---- 22 regional e-paper ads (from the downloaded Epaper2 sample)
    leftovers = []
    for lang, n in EPAPER_QUOTAS.items():
        files = sorted((MAD / "images/epaper2" / lang).glob("*.jpg"))
        files = [f for f in files if min(pp.load_image(f).shape[:2]) >= MIN_SIDE]
        rng.shuffle(files)
        for f in files[:n]:
            add(f, f.name, source="madverse_epaper2", language_label=lang.title(), category_label="", brand_label="")
        leftovers += [(lang, f) for f in files[n:]]
    # top up to 22 if some languages had too few readable images (prefer non-Hindi variety)
    target = sum(EPAPER_QUOTAS.values())
    have = sum(r["source"] == "madverse_epaper2" for r in rows)
    leftovers.sort(key=lambda x: x[0] == "hindi")
    for lang, f in leftovers[:max(0, target - have)]:
        add(f, f.name, source="madverse_epaper2", language_label=lang.title(), category_label="", brand_label="")

    # ---- 3 degraded copies
    def blur_jpeg(im):
        im = cv2.GaussianBlur(im, (0, 0), 1.6)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 35])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def tilt(im):
        return pp.rotate(im, 4.0)

    def phone_photo(im):
        h, w = im.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * .04, h * .02], [w * .97, 0], [w, h * .98], [0, h * .95]])
        out = cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(90, 90, 90))
        shade = np.tile(np.linspace(1.0, 0.75, w, dtype=np.float32), (h, 1))[..., None]
        noise = rng.normalvariate(0, 1) * 0 + np.random.default_rng(SEED).normal(0, 6, out.shape)
        return np.clip(out * shade + noise, 0, 255).astype(np.uint8)

    picks = [("blur_jpeg", blur_jpeg, rows[0]), ("tilt4", tilt, next(r for r in rows if r["source"] == "madverse_adgal")),
             ("phone_photo", phone_photo, next(r for r in rows if r["language_label"] == "Hindi" and r["source"] == "madverse_epaper2"))]
    for tag, fn, base in picks:
        im = fn(pp.load_image(ROOT / base["file"]))
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        add(buf.tobytes(), f"{Path(base['file']).stem}_{tag}.jpg", source=f"degraded:{tag}",
            language_label=base["language_label"], category_label=base["category_label"],
            brand_label=base["brand_label"], degraded_of=base["id"])

    cols = ["id", "file", "source", "language_label", "category_label", "brand_label", "degraded_of", "original"]
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    from collections import Counter
    print(f"{len(rows)} images -> {OUT}")
    print("by source:", dict(Counter(r["source"].split(":")[0] for r in rows)))
    print("by language:", dict(Counter(r["language_label"].lower() for r in rows)))


if __name__ == "__main__":
    main()
