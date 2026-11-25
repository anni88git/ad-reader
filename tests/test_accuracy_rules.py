"""Regression tests for the generalised accuracy rules (built from evaluation error analysis)."""
import numpy as np
import pytest

from adreader import normalize as nz
from adreader.ocr import OCRLine, OCRResult, normalize_text, script_of
from adreader.preprocess import ImageQuality
from adreader.schema import Ev, FieldStatus, RawExtraction, RawPrice
from adreader.validate import validate


def fake_ocr(lines):
    """lines: list of (text, height) -> OCRResult with stacked boxes."""
    out, y = [], 10
    for text, h in lines:
        out.append(OCRLine(text=text, norm=normalize_text(text), score=0.97, lang="latin", script=script_of(text),
                           box=[[10, y], [10 + 12 * len(text), y], [10 + 12 * len(text), y + h], [10, y + h]]))
        y += h + 10
    return OCRResult(lines=out, image=np.zeros((y + 10, 2000, 3), np.uint8), scale=1.0, deskew_deg=0.0,
                     quality=ImageQuality(2000, y, 500.0, 60.0, []), langs=("latin",), timings_s={})


# ---------------------------------------------------------------- Indian number formats
@pytest.mark.parametrize("text,amount", [
    ("₹ 27 99 000*", 2799000), ("₹27,99,000", 2799000), ("Rs. 9.16 Lakh*", 916000), ("₹9.16 Lakh'", 916000),
    ("₹ 97 Lacs Onwards", 9700000), ("₹1.5 Cr", 15000000), ("INR 2,499", 2499), ("रु ४४", 44),
    ("₹14.99 లక్ష*", 1499000), ("₹ 2.5 கோடி", 25000000), ("₹ 21.5 ಲಕ್ಷ", 2150000), ("₹৫০", 50),
    ("₹3 लाखात", 300000),
])
def test_indian_amounts(text, amount):
    assert nz.parse_amount(text)[0] == amount, text


def test_unknown_unit_after_decimal_is_flagged():
    amt, problems = nz.parse_amount("₹14.99 xyzअ")
    assert problems and "unit" in problems[0]


@pytest.mark.parametrize("label,kind", [("Maximum Benefits up to", "discount"), ("Big Savings", "discount"),
                                        ("Free shampoo worth", "discount"), ("Exchange bonus", "discount"),
                                        ("Introductory price starts at", "price"), ("Get cashback", "cashback")])
def test_benefits_are_not_prices(label, kind):
    assert nz.price_kind(label, "₹45 000") == kind


# ---------------------------------------------------------------- recall from OCR
def test_missed_phone_and_rera_added_as_review():
    ocr = fake_ocr([("Gold Crest Homes", 60), ("Call 875-0039-171 for site visit", 20),
                    ("MahaRERA No. P51800050191 maharera.mahaonline.gov.in", 14)])
    raw = RawExtraction(brand=Ev(value="Gold Crest", evidence="Gold Crest Homes"))
    rec = validate(raw, ocr, image_name="t")
    phones = rec.lists["phones"]
    ids = rec.lists["reference_numbers"]
    assert any(p.value["number"] == "8750039171" and p.status == FieldStatus.review for p in phones)
    assert any("P51800050191" in str(r.value) and r.status == FieldStatus.review for r in ids)


def test_recall_does_not_duplicate_extracted_phone():
    ocr = fake_ocr([("Call 9289305692", 20)])
    raw = RawExtraction(phones=[Ev(value="9289305692", evidence="Call 9289305692")])
    rec = validate(raw, ocr, image_name="t")
    assert [p.value["number"] for p in rec.lists["phones"]].count("9289305692") == 1


# ---------------------------------------------------------------- confusable characters
def test_lookalike_characters_not_confirmed():
    ocr = fake_ocr([("Hyundai AURA fuelled by 1.2 I Eco TORQ engine", 20)])
    raw = RawExtraction(model_variant=Ev(value="1.2 I Eco TORQ", evidence="1.2 I Eco TORQ"))
    rec = validate(raw, ocr, image_name="t")
    assert rec.fields["model_variant"].status != FieldStatus.confirmed


def test_plain_variant_still_confirmed():
    ocr = fake_ocr([("Available in Hybrid variant", 20)])
    raw = RawExtraction(model_variant=Ev(value="Hybrid", evidence="Hybrid"))
    rec = validate(raw, ocr, image_name="t")
    assert rec.fields["model_variant"].status == FieldStatus.confirmed


# ---------------------------------------------------------------- language vs OCR scripts
def test_language_missing_indic_script_flagged():
    ocr = fake_ocr([("नमक नहीं लो मक्खन", 30), ("शुद्ध दूध से बना", 20), ("संतुलित नमक वाला", 20), ("Customer Care", 15)])
    raw = RawExtraction(language=["English"])
    rec = validate(raw, ocr, image_name="t")
    assert rec.fields["language"].status != FieldStatus.confirmed


# ---------------------------------------------------------------- headline / sub-headline policy
def test_subheadline_never_auto_confirmed_even_if_models_agree():
    ocr = fake_ocr([("A New View", 60), ("Thank you for all the love.", 30)])
    raw = RawExtraction(headline=Ev(value="A New View", evidence="A New View"),
                        sub_headline=Ev(value="Thank you for all the love.", evidence="Thank you for all the love."))
    ver = RawExtraction(headline=Ev(value="A New View"), sub_headline=Ev(value="Thank you for all the love."))
    rec = validate(raw, ocr, image_name="t", verifier=ver)
    assert rec.fields["headline"].status == FieldStatus.confirmed          # agreement allowed for headline
    assert rec.fields["sub_headline"].status != FieldStatus.confirmed      # never for sub-headline


def test_headline_disagreement_flags_with_suggestion():
    ocr = fake_ocr([("n Duty, Always!", 60)])
    raw = RawExtraction(headline=Ev(value="n Duty, Always!", evidence="n Duty, Always!"))
    ver = RawExtraction(headline=Ev(value="On Duty, Always!"))
    rec = validate(raw, ocr, image_name="t", verifier=ver)
    h = rec.fields["headline"]
    assert h.status != FieldStatus.confirmed and any("On Duty, Always!" in x for x in h.reasons)


def test_larger_unused_line_flags_headline():
    ocr = fake_ocr([("BRANDS FOR LESS", 30), ("BUY 2 GET 3 FREE", 120)])
    raw = RawExtraction(headline=Ev(value="BRANDS FOR LESS", evidence="BRANDS FOR LESS"))
    ver = RawExtraction(headline=Ev(value="BRANDS FOR LESS"))     # even if both models agree
    rec = validate(raw, ocr, image_name="t", verifier=ver)
    assert rec.fields["headline"].status != FieldStatus.confirmed


# ---------------------------------------------------------------- Phase 2: verifier agreement normalisation
def test_language_iso_codes_count_as_agreement():
    ocr = fake_ocr([("नमक नहीं लो मक्खन", 30), ("शुद्ध दूध से बना", 20), ("संतुलित नमक वाला", 20), ("Customer Care", 15)])
    raw = RawExtraction(language=["Hindi", "English"])
    rec = validate(raw, ocr, image_name="t", verifier=RawExtraction(language=["hi", "en"]))
    assert not any("disagrees" in r for r in rec.fields["language"].reasons)


def test_language_real_difference_still_flagged():
    ocr = fake_ocr([("नमक नहीं लो मक्खन", 30), ("शुद्ध दूध से बना", 20), ("संतुलित नमक वाला", 20), ("Customer Care", 15)])
    rec = validate(RawExtraction(language=["Hindi", "English"]), ocr, image_name="t",
                   verifier=RawExtraction(language=["en"]))
    assert any("disagrees" in r for r in rec.fields["language"].reasons)


def test_category_parent_vs_child_of_same_parent_agrees():
    ocr = fake_ocr([("Samsung Washing Machine", 40)])
    rec = validate(RawExtraction(category=Ev(value="Retail & Consumer Electronics")), ocr, image_name="t",
                   verifier=RawExtraction(category=Ev(value="Retail & Consumer Electronics > Home Appliances")))
    assert not any("disagrees" in r for r in rec.fields["category"].reasons)


def test_category_child_vs_parent_or_other_branch_still_flagged():
    ocr = fake_ocr([("Samsung Washing Machine", 40)])
    a = validate(RawExtraction(category=Ev(value="Retail & Consumer Electronics > Home Appliances")), ocr, image_name="t",
                 verifier=RawExtraction(category=Ev(value="Retail & Consumer Electronics")))
    b = validate(RawExtraction(category=Ev(value="FMCG")), ocr, image_name="t",
                 verifier=RawExtraction(category=Ev(value="Healthcare > Pharmacy")))
    assert any("disagrees" in r for r in a.fields["category"].reasons)
    assert any("disagrees" in r for r in b.fields["category"].reasons)


def test_qr_detector_needs_strong_evidence():
    from adreader.validate import detect_qr
    blank = np.full((400, 400, 3), 255, np.uint8)
    assert detect_qr(blank) == (False, None)


# ---------------------------------------------------------------- Phase 2.5: language removal needs two signals
def test_language_removed_when_ocr_and_verifier_both_lack_it():
    ocr = fake_ocr([("Hero Splendor", 40), ("Ride the legend", 20), ("heromotocorp.com", 15)])
    rec = validate(RawExtraction(language=["English", "Hindi"]), ocr, image_name="t",
                   verifier=RawExtraction(language=["English"]))
    lg = rec.fields["language"]
    assert lg.value == ["English"] and lg.status != FieldStatus.confirmed


def test_language_kept_when_verifier_lists_it():
    ocr = fake_ocr([("Hero Splendor", 40), ("Ride the legend", 20)])
    rec = validate(RawExtraction(language=["English", "Hindi"]), ocr, image_name="t",
                   verifier=RawExtraction(language=["en", "hi"]))
    assert rec.fields["language"].value == ["English", "Hindi"]


def test_language_kept_without_verifier():
    ocr = fake_ocr([("Hero Splendor", 40), ("Ride the legend", 20)])
    rec = validate(RawExtraction(language=["English", "Hindi"]), ocr, image_name="t")
    assert rec.fields["language"].value == ["English", "Hindi"]
