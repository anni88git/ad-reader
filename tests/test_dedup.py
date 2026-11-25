"""Campaign deduplication: same creative re-run, changed price, new version, unrelated."""
import cv2
import numpy as np

from adreader import dedup


def _img(seed, text="SALE"):
    rng = np.random.default_rng(seed)
    im = np.full((400, 300, 3), 255, np.uint8)
    for _ in range(12):
        x, y = rng.integers(0, 250), rng.integers(0, 350)
        cv2.rectangle(im, (int(x), int(y)), (int(x) + 50, int(y) + 40), [int(c) for c in rng.integers(0, 255, 3)], -1)
    cv2.putText(im, text, (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return im


def _rec(brand, product, headline, price=None, desc="", offers=()):
    f = lambda v: {"value": v, "status": "confirmed"}
    return {"fields": {"brand": f(brand), "product": f(product), "headline": f(headline),
                       "description": f(desc), "price": f(price)},
            "lists": {"offers": [{"value": o} for o in offers], "phones": [], "prices": []}}


OCR = ["Fresh Mango Delight", "Made with real Alphonso mangoes", "Available at all stores", "Call 98100 12345"]


def test_rescanned_copy_is_duplicate():
    img = _img(1)
    small = cv2.resize(img, (210, 280))[5:-5, 5:-5]
    a = dedup.signature("a", _rec("Frooti", "Mango", "Fresh Mango Delight", 20), img, OCR)
    b = dedup.signature("b", _rec("Frooti", "Mango", "Fresh Mango Delight", 20), small, OCR)
    assert dedup.compare(a, b)["relation"] == dedup.DUPLICATE


def test_price_change_detected_with_details():
    img = _img(1)
    a = dedup.signature("a", _rec("Frooti", "Mango", "Fresh Mango Delight", 20), img, OCR)
    b = dedup.signature("b", _rec("Frooti", "Mango", "Fresh Mango Delight", 25), img, OCR)
    r = dedup.compare(a, b)
    assert r["relation"] == dedup.OFFER_CHANGED and "20" in r["changes"][0] and "25" in r["changes"][0]


def test_missing_value_on_blurry_copy_is_not_a_change():
    img = _img(1)
    a = dedup.signature("a", _rec("Frooti", "Mango", "Fresh Mango Delight", 20, offers=["Buy 2 get 1"]), img, OCR)
    b = dedup.signature("b", _rec("Frooti", "Mango", "Fresh Mango Delight", 20), img, OCR)
    assert dedup.compare(a, b)["relation"] == dedup.DUPLICATE


def test_new_version_shares_headline_but_not_layout():
    a = dedup.signature("a", _rec("Frooti", "Mango", "Fresh Mango Delight", desc="Summer pack"), _img(1), OCR)
    b = dedup.signature("b", _rec("Frooti", "Mango", "Fresh Mango Delight", desc="Festive pack"), _img(99),
                        ["Fresh Mango Delight", "Festive combo now in stores", "Gift packs"])
    assert dedup.compare(a, b)["relation"] == dedup.NEW_VERSION


def test_same_product_different_campaign_not_merged():
    a = dedup.signature("a", _rec("Toyota", "Yaris", "This Fabulous February", desc="Offers on all models"), _img(1), OCR)
    b = dedup.signature("b", _rec("Toyota", "Yaris", "Arre waah that's a Toyota", desc="Sedan with 7 airbags"),
                        _img(50), ["Arre waah that's a Toyota", "Sedan with seven airbags"])
    r = dedup.compare(a, b)
    assert r["relation"] == dedup.SAME_PRODUCT
    g = dedup.group([a, b])
    assert g["campaigns"] == 2


def test_different_brands_never_link_even_if_layout_identical():
    img = _img(1)
    a = dedup.signature("a", _rec("Frooti", "Mango", "Fresh Mango Delight"), img, OCR)
    b = dedup.signature("b", _rec("Maaza", "Mango", "Fresh Mango Delight"), img, OCR)
    assert dedup.compare(a, b) is None


def test_group_counts_creatives_and_campaigns():
    img = _img(1)
    s = [dedup.signature("a", _rec("Frooti", "Mango", "Fresh Mango Delight", 20), img, OCR),
         dedup.signature("b", _rec("Frooti", "Mango", "Fresh Mango Delight", 20), img, OCR),
         dedup.signature("c", _rec("Frooti", "Mango", "Fresh Mango Delight", 25), img, OCR),
         dedup.signature("d", _rec("Amul", "Butter", "Utterly Butterly"), _img(7), ["Utterly Butterly Delicious"])]
    g = dedup.group(s)
    assert g["campaigns"] == 2 and g["unique_creatives"] == 3


def test_brand_substring_is_not_same_brand():
    """'Kia' is inside 'Nokia' - must not be treated as the same advertiser."""
    img = _img(1)
    a = dedup.signature("a", _rec("Kia", "Seltos", "Fresh Mango Delight"), img, OCR)
    b = dedup.signature("b", _rec("Nokia", "6.1", "Fresh Mango Delight"), img, OCR)
    assert dedup.compare(a, b) is None


def test_brand_word_prefix_is_same_brand():
    img = _img(1)
    a = dedup.signature("a", _rec("Hero", "Splendor", "Ride Ahead"), img, OCR)
    b = dedup.signature("b", _rec("Hero MotoCorp", "Splendor", "Ride Ahead"), img, OCR)
    assert dedup.compare(a, b)["relation"] == dedup.DUPLICATE
