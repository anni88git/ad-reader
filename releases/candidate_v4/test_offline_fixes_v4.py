"""Regression tests for accuracy pass v4 (adreader/accuracy_rules_v4.py). Synthetic inputs only.
Every switched value must end in REVIEW (never confirmed) and carry the v4 lock."""
import pytest

from adreader import accuracy_rules_v4 as v4
from adreader.matcher import link_records
from adreader.schema import AdRecord, Ev, FieldResult, FieldStatus, RawExtraction
from adreader.validate import Evidence
from tests.test_accuracy_rules import fake_ocr


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("ADREADER_RULES_V4", "1")


def rec_with(**fields):
    r = AdRecord(image="t.jpg")
    for k in ("brand", "product", "headline", "ad_type", "category", "language", "price"):
        r.fields[k] = FieldResult()
    r.fields.update(fields)
    r.lists["prices"] = []
    return r


def f(value, status=FieldStatus.review, reasons=None, conf=0.6):
    return FieldResult(value=value, status=status, confidence=conf, reasons=list(reasons or []))


def ver(**kw):
    base = {"language": []}
    for k, v in kw.items():
        base[k] = v if k == "language" else {"value": v}
    return RawExtraction.model_validate(base)


def ocr(*t):
    return fake_ocr([(x, 40) for x in t])


def in_review(r):
    return r.status == FieldStatus.review and v4.LOCK in r.reasons


# ---------------------------------------------------------------- 1. ad type
def test_manufacturer_vs_service_provider_takes_verifier_in_review():
    r = rec_with(ad_type=f("manufacturer"))
    v4.fix_ad_type(r, ver(ad_type="service_provider"))
    assert r.fields["ad_type"].value == "service_provider" and in_review(r.fields["ad_type"])


@pytest.mark.parametrize("p,v", [("manufacturer", "public_notice"), ("retailer", "dealer"), ("manufacturer", "manufacturer")])
def test_other_ad_type_disagreements_untouched(p, v):
    r = rec_with(ad_type=f(p))
    v4.fix_ad_type(r, ver(ad_type=v))
    assert r.fields["ad_type"].value == p and v4.LOCK not in r.fields["ad_type"].reasons


def test_switched_value_never_confirmed_even_from_confirmed():
    r = rec_with(ad_type=f("manufacturer", FieldStatus.confirmed, conf=0.9))
    v4.fix_ad_type(r, ver(ad_type="service_provider"))
    assert r.fields["ad_type"].status == FieldStatus.review and r.fields["ad_type"].confidence <= 0.6


# ---------------------------------------------------------------- 2. price parsing
@pytest.mark.parametrize("text,amount", [("500 mL ₹28*", 28), ("Rs. 37.6 - 52.67 Lacs*", 3760000),
                                         ("₹1.2 - 1.8 Cr", 12000000), ("10 cap pack 295/-", 295)])
def test_reparse_known_misparses(text, amount):
    assert v4.reparse_amount(text) == amount


@pytest.mark.parametrize("text", ["₹ 85 120/-", "₹2,999", "MRP ₹548/-", "₹ 27 99 000*"])
def test_reparse_leaves_good_amounts_alone(text):
    assert v4.reparse_amount(text) is None


def _p(amount, kind, label, printed):
    return FieldResult(value={"amount": amount, "currency": "INR", "kind": kind, "label": label, "item": None,
                              "as_printed": printed}, status=FieldStatus.review, confidence=0.7)


def test_mrp_only_becomes_price_in_review():
    r = rec_with()
    r.lists["prices"] = [_p(548, "old_price", "MRP", "MRP 548/-")]
    v4.fix_price(r, None)
    assert r.fields["price"].value == 548 and in_review(r.fields["price"])


def test_mrp_ignored_when_a_selling_price_exists():
    r = rec_with()
    r.lists["prices"] = [_p(999, "old_price", "MRP", "MRP ₹999"), _p(699, "offer_price", "Offer", "₹699")]
    v4.fix_price(r, None)
    assert r.fields["price"].value is None          # main-price choice is validate's job; v4 adds nothing


def test_struck_price_without_mrp_word_not_used():
    r = rec_with()
    r.lists["prices"] = [_p(999, "old_price", "was", "₹999")]
    v4.fix_price(r, None)
    assert r.fields["price"].value is None


def test_misparsed_main_price_corrected_in_review():
    r = rec_with(price=f(500))
    r.lists["prices"] = [_p(500, "price", None, "500 mL ₹28*")]
    v4.fix_price(r, None)
    assert r.fields["price"].value == 28 and in_review(r.fields["price"])


def test_unit_check_misfire_with_both_models_agreeing_goes_to_review():
    reasons = ["'54' appears in the ad as a quantity/unit, not as a price", "too uncertain to fill",
               "second model agrees"]
    r = rec_with(price=f(54, FieldStatus.missing, reasons, 0.35))
    r.lists["prices"] = [_p(54, "price", "/ 1 L", "₹54*")]
    v4.fix_price(r, None)
    assert in_review(r.fields["price"])


def test_unit_value_without_verifier_agreement_stays_hidden():
    r = rec_with(price=f(54, FieldStatus.missing, ["'54' appears in the ad as a quantity/unit, not as a price"], 0.35))
    r.lists["prices"] = [_p(54, "price", None, "₹54")]
    v4.fix_price(r, None)
    assert r.fields["price"].status == FieldStatus.missing


# ---------------------------------------------------------------- 3. language
GARBLE = " ".join(["वबे", "घेत्वी", "भडी", "छ्", "लही", "उेभल", "वैभी", "हवमर", "यंणरे", "चर", "तप्ल", "बतपुत",
                   "भडे", "वुभ्म्", "इुत्ली", "घुर", "भत्तभ"])


def test_garbled_devanagari_takes_verifier_unsupported_script_language():
    r = rec_with(language=f(["Hindi", "English"]))
    v4.fix_language(r, ver(language=["pa", "en"]), ocr(GARBLE, "HAIR COLOUR"))
    assert r.fields["language"].value == ["Punjabi", "English"] and in_review(r.fields["language"])


def test_real_hindi_text_keeps_primary_language():
    r = rec_with(language=f(["Hindi", "English"]))
    v4.fix_language(r, ver(language=["Bengali"]),
                    ocr("यह ऑफर सभी दुकानों में उपलब्ध है और आज ही खरीदें के लिए आपके साथ हम भी हैं " * 2))
    assert r.fields["language"].value == ["Hindi", "English"]


def test_verifier_with_ocr_readable_script_not_used():
    r = rec_with(language=f(["Hindi"]))
    v4.fix_language(r, ver(language=["Marathi"]), ocr(GARBLE))
    assert r.fields["language"].value == ["Hindi"]


# ---------------------------------------------------------------- 4. brand
def test_empty_brand_filled_from_verifier_when_printed():
    o = ocr("KALINGA SALES", "Best deals in town")
    r = rec_with()
    v4.fix_brand(r, ver(brand="KALINGA SALES"), Evidence(o))
    assert r.fields["brand"].value == "KALINGA SALES" and in_review(r.fields["brand"])


def test_verifier_brand_not_printed_is_ignored():
    o = ocr("Best deals in town")
    r = rec_with()
    v4.fix_brand(r, ver(brand="Imaginary Co"), Evidence(o))
    assert r.fields["brand"].value is None


def test_existing_brand_never_overwritten():
    o = ocr("ACME", "KALINGA SALES")
    r = rec_with(brand=f("ACME"))
    v4.fix_brand(r, ver(brand="KALINGA SALES"), Evidence(o))
    assert r.fields["brand"].value == "ACME"


def test_matcher_does_not_confirm_a_switched_brand():
    o = ocr("Tata Motors", "New Nexon")
    r = rec_with()
    v4.fix_brand(r, ver(brand="Tata Motors"), Evidence(o))
    link_records(r, "tatamotorsnewnexon")
    assert r.fields["brand"].status == FieldStatus.review and r.fields["brand"].matched_id is None


# ---------------------------------------------------------------- 5. headline
def test_fragment_headline_replaced_by_longer_printed_one():
    o = ocr("नाजुक आँखों को रखें STRAIN FREE")
    r = rec_with(headline=f("STRAIN FREE"))
    v4.fix_headline(r, ver(headline="नाजुक आँखों को रखें STRAIN FREE"), Evidence(o))
    assert r.fields["headline"].value == "नाजुक आँखों को रखें STRAIN FREE" and in_review(r.fields["headline"])


def test_longer_headline_not_printed_is_ignored():
    o = ocr("STRAIN FREE")
    r = rec_with(headline=f("STRAIN FREE"))
    v4.fix_headline(r, ver(headline="Keep your eyes STRAIN FREE every day"), Evidence(o))
    assert r.fields["headline"].value == "STRAIN FREE"


def test_different_headline_not_containing_primary_ignored():
    o = ocr("Big Sale", "Festive Offers")
    r = rec_with(headline=f("Big Sale"))
    v4.fix_headline(r, ver(headline="Festive Offers"), Evidence(o))
    assert r.fields["headline"].value == "Big Sale"


# ---------------------------------------------------------------- switch
def test_v4_off_by_default(monkeypatch):
    monkeypatch.delenv("ADREADER_RULES_V4", raising=False)
    r = rec_with(ad_type=f("manufacturer"))
    v4.apply(r, RawExtraction(), ver(ad_type="service_provider"), None, None)
    assert r.fields["ad_type"].value == "manufacturer"
