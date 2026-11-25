"""Red-team test for stage 4: inject realistic model errors into real extractions and check
that NONE of them comes out 'confirmed'. Exit code 1 if any error slips through.

  python scripts/redteam_validate.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adreader import preprocess as pp  # noqa: E402
from adreader.config import OUTPUTS_DIR  # noqa: E402
from adreader.schema import Ev, FieldStatus, RawExtraction, RawPrice  # noqa: E402
from adreader.validate import validate  # noqa: E402

EX = OUTPUTS_DIR / "extract"
MODEL = "nemotron-3.5-super-vl-120b-a12b"


def load(stem):
    d = json.loads((EX / f"{stem}.{MODEL}.json").read_text(encoding="utf-8"))
    img_path = Path(d["image"])
    ocr = pickle.loads((OUTPUTS_DIR / "ocr_cache" / f"{img_path.stem}.latin-devanagari.pkl").read_bytes())
    return RawExtraction.model_validate(d["raw"]), ocr, pp.load_image(img_path)


def find(rec, pred):
    return [(k, r) for k, r in rec.all_results() if pred(k, r)]


CASES = []


def case(stem, desc):
    def deco(fn):
        CASES.append((stem, desc, fn))
        return fn
    return deco


@case("_kia_from_screenshot", "phone digit misread 1800 108 5505 (real: 5005)")
def _(raw):
    raw.phones[0] = Ev(value="1800 108 5505", evidence="Call 1800 108 5505")
    return lambda rec: find(rec, lambda k, r: k.startswith("phones") and r.value["number"] == "18001085505")


@case("_kia_from_screenshot", "price digit misread ₹ 27 89 000 (real: 27 99 000)")
def _(raw):
    raw.prices = [RawPrice(amount_text="₹ 27 89 000", label="Introductory price starts at")]
    return lambda rec: find(rec, lambda k, r: k == "price")


@case("_kia_from_screenshot", "invented website not printed in the ad")
def _(raw):
    raw.websites.append(Ev(value="www.kiaudaipur.in", evidence="www.kiaudaipur.in"))
    return lambda rec: find(rec, lambda k, r: k.startswith("websites") and "kiaudaipur" in str(r.value))


@case("_kia_from_screenshot", "wrong brand with real evidence (value/evidence mismatch)")
def _(raw):
    raw.brand = Ev(value="Hyundai", evidence="Kia Sorento")
    return lambda rec: find(rec, lambda k, r: k == "brand")


@case("_kia_from_screenshot", "invented feature with invented evidence")
def _(raw):
    raw.features.append(Ev(value="Panoramic Sunroof", evidence="Panoramic Sunroof"))
    return lambda rec: find(rec, lambda k, r: k.startswith("features") and "Sunroof" in str(r.value))


@case("_kia_from_screenshot", "mileage digit misread 18.6 km/l (real: 19.6)")
def _(raw):
    raw.category_attributes["mileage"] = Ev(value="18.6 km/l", evidence="ARAI Certified Hybrid Mileage 18.6 km/l")
    return lambda rec: find(rec, lambda k, r: k == "attr.mileage")


@case("_kia_from_screenshot", "invalid phone number (9 digits)")
def _(raw):
    raw.phones.append(Ev(value="928930569", evidence="Tel: 928930569"))
    return lambda rec: find(rec, lambda k, r: k.startswith("phones") and r.value["number"] == "928930569")


@case("002_horlicks_hi", "price invented from '27 ग्राम' in small print (seen in real run)")
def _(raw):
    raw.prices = [RawPrice(amount_text="रुपए 27", label="price")]
    return lambda rec: find(rec, lambda k, r: k == "price")


@case("002_horlicks_hi", "Hindi headline with a spelling-level change (बढ़ाएं vs printed बढ़ाए)")
def _(raw):
    raw.headline = Ev(value="हॉर्लिक्स दूध की शक्ति बढ़ाएं", evidence="हॉर्लिक्स दूध की शक्ति बढ़ाएं")
    return lambda rec: find(rec, lambda k, r: k == "headline")


@case("_kia_from_screenshot", "English headline with one word changed")
def _(raw):
    raw.headline = Ev(value="Superior tech and elegance, beyond the best.",
                      evidence="Superior tech and elegance, beyond the best.")
    return lambda rec: find(rec, lambda k, r: k == "headline")


@case("002_horlicks_hi", "invented toll-free number")
def _(raw):
    raw.phones = [Ev(value="1800 102 2299", evidence="1800 102 2299")]
    return lambda rec: find(rec, lambda k, r: k.startswith("phones"))


@case("003_namaste_butter_hi", "contact phone digit misread 7988859460 (real: ...406)")
def _(raw):
    raw.phones = [Ev(value="7988859460", evidence="मो0 नं0- 7988859460")]
    raw.dealers = []
    return lambda rec: find(rec, lambda k, r: k.startswith("phones") and r.value["number"] == "7988859460")


@case("003_namaste_butter_hi", "struck-out ₹48 presented as the price")
def _(raw):
    raw.prices = [RawPrice(amount_text="₹48/-", label="Promotional Offer"), RawPrice(amount_text="₹44/-", label="मात्र")]
    return lambda rec: find(rec, lambda k, r: k.startswith("prices") and r.value["amount"] == 48)


@case("003_namaste_butter_hi", "wrong ad type (dealer) without any dealer evidence")
def _(raw):
    raw.ad_type = Ev(value="dealer", evidence=None)
    raw.dealers = []
    return lambda rec: find(rec, lambda k, r: k == "ad_type")


if __name__ == "__main__":
    leaks = 0
    for stem, desc, fn in CASES:
        raw, ocr, img = load(stem)
        raw = raw.model_copy(deep=True)
        selector = fn(raw)
        rec = validate(raw, ocr, image=img, image_name=stem)
        hits = selector(rec)
        if not hits:
            print(f"  ??   {desc}: injected value not present in record")
            leaks += 1
            continue
        for k, r in hits:
            ok = r.status != FieldStatus.confirmed
            leaks += 0 if ok else 1
            print(f"  {'PASS' if ok else 'LEAK'} {r.status.value:9} {r.confidence:.2f}  {desc}\n"
                  f"        -> {k}: {'; '.join(r.reasons)[:150]}")
    print(f"\n{len(CASES)} injected errors, {leaks} leaked as 'confirmed'")
    sys.exit(1 if leaks else 0)
