"""Regression tests for accuracy pass v3 (adreader/accuracy_rules.py): field-specific evidence gates
built from the bench200 wrong-confirmation audit. Synthetic inputs only - no benchmark ad is replayed."""
import pytest

from adreader import accuracy_rules as ar
from adreader.schema import AdRecord, FieldResult, FieldStatus
from tests.test_accuracy_rules import fake_ocr


def conf(value, reasons=None, **kw):
    return FieldResult(value=value, confidence=0.88, status=FieldStatus.confirmed,
                       reasons=list(reasons or ["second model agrees"]), **kw)


def rec_with(**fields):
    r = AdRecord(image="t.jpg")
    for k in ("brand", "product", "model_variant", "advertiser", "headline", "sub_headline", "description",
              "call_to_action", "ad_type", "category", "language", "price", "body_copy", "ad_message"):
        r.fields[k] = FieldResult()
    r.fields.update(fields)
    return r


def ocr(*texts):
    return fake_ocr([(t, 40) for t in texts])


# ---------------------------------------------------------------- language: Devanagari is not "Hindi"
def test_marathi_words_replace_hindi_and_go_to_review():
    r = rec_with(language=conf(["Hindi", "English"]))
    ar.language_rules(r, ocr("सर्व प्रमुख दुकानांमध्ये उपलब्ध आहे", "ऑफर आजपासून सुरू आणि मर्यादित", "SALE"))
    f = r.fields["language"]
    assert f.value == ["Marathi", "English"] and f.status == FieldStatus.review


def test_hindi_text_with_hindi_words_untouched():
    r = rec_with(language=conf(["Hindi", "English"]))
    ar.language_rules(r, ocr("यह ऑफर सभी दुकानों में उपलब्ध है", "आज ही खरीदें और बचत करें", "SALE"))
    assert r.fields["language"].status == FieldStatus.confirmed and r.fields["language"].value == ["Hindi", "English"]


def test_bilingual_hindi_marathi_adds_marathi_keeps_hindi():
    r = rec_with(language=conf(["Hindi"]))
    ar.language_rules(r, ocr("उपलब्ध आहे आणि मध्ये करा", "यह है"))
    assert set(r.fields["language"].value) == {"Hindi", "Marathi"}


def test_language_without_ocr_script_never_confirmed_even_if_verifier_agrees():
    r = rec_with(language=conf(["Hindi", "English"], ["Hindi: OCR found no devanagari text", "second model agrees"]))
    ar.language_rules(r, ocr("EXCELLENCE IN EDUCATION", "REGISTER NOW"))
    assert r.fields["language"].status == FieldStatus.review


def test_garbled_devanagari_without_hindi_words_is_not_confirmed_as_hindi():
    junk = " ".join(["वबे", "घेत्वी", "भडी", "छ्", "लही", "उेभल", "वैभी", "हवमर", "यंणरे", "चर", "तप्ल", "बतपुत",
                     "भडे", "वुभ्म्", "इुत्ली", "घुर", "भत्तभ"])
    r = rec_with(language=conf(["Hindi", "English"]))
    ar.language_rules(r, ocr(junk, "HAIR COLOUR"))
    assert r.fields["language"].status == FieldStatus.review


# ---------------------------------------------------------------- price: role of the number
def _price(amount, label, item=None, status=FieldStatus.confirmed):
    return FieldResult(value={"amount": amount, "currency": "INR", "kind": "price", "label": label, "item": item,
                              "as_printed": f"₹{amount}"}, confidence=0.92, status=status, reasons=["digits match"])


def test_hindi_benefit_amount_is_not_the_price():
    p = _price(50000, "तक के फायदे*")
    r = rec_with(price=FieldResult(value=50000, confidence=0.92, status=FieldStatus.confirmed))
    r.lists["prices"], r.lists["offers"] = [p], []
    ar.price_rules(r, ocr("50000", "तक के फायदे*"))
    assert r.fields["price"].value is None and r.lists["prices"] == [] and r.lists["offers"][0].value["kind"] == "discount"


def test_english_item_benefits_is_not_the_price():
    p = _price(45000, None, item="maximum benefits")
    r = rec_with(price=FieldResult(value=45000, confidence=0.92, status=FieldStatus.confirmed))
    r.lists["prices"], r.lists["offers"] = [p], []
    ar.price_rules(r, None)
    assert r.fields["price"].value is None


def test_real_price_kept_when_benefit_is_separate():
    real, ben = _price(799000, "Price starting at"), _price(45000, "Benefits up to")
    r = rec_with(price=FieldResult(value=45000, confidence=0.92, status=FieldStatus.confirmed))
    r.lists["prices"], r.lists["offers"] = [real, ben], []
    ar.price_rules(r, None)
    assert r.fields["price"].value == 799000


def test_confirmed_price_next_to_benefit_words_goes_to_review():
    r = rec_with(price=FieldResult(value=25000, confidence=0.92, status=FieldStatus.confirmed))
    r.lists["prices"] = []
    ar.price_rules(r, ocr("Get", "₹25,000", "benefits on exchange"))
    assert r.fields["price"].status == FieldStatus.review


def test_plain_price_stays_confirmed():
    r = rec_with(price=FieldResult(value=1299, confidence=0.92, status=FieldStatus.confirmed))
    r.lists["prices"] = []
    ar.price_rules(r, ocr("Sneakers", "Now ₹1,299", "Free delivery"))
    assert r.fields["price"].status == FieldStatus.confirmed


# ---------------------------------------------------------------- ad type
def test_admission_ad_not_confirmed_as_recruitment():
    r = rec_with(ad_type=conf("recruitment"), category=conf("Education > Universities & Colleges"))
    ar.ad_type_rules(r, ocr("ADMISSION OPEN 2024-25", "MBA BBA programmes"))
    assert r.fields["ad_type"].status == FieldStatus.review


def test_hindi_admission_ad_not_confirmed_as_recruitment():
    r = rec_with(ad_type=conf("recruitment"), category=conf("Recruitment"))
    ar.ad_type_rules(r, ocr("एमबीबीएस 2023 के पहले बैच का दाखिला शुरू"))
    assert r.fields["ad_type"].status == FieldStatus.review


def test_real_job_ad_stays_recruitment():
    r = rec_with(ad_type=conf("recruitment"), category=conf("Recruitment"))
    ar.ad_type_rules(r, ocr("WALK-IN INTERVIEW", "Vacancies for Sales Executive", "Salary 25,000"))
    assert r.fields["ad_type"].status == FieldStatus.confirmed


def test_manufacturer_of_a_service_goes_to_review():
    r = rec_with(ad_type=conf("manufacturer"), category=conf("Travel & Hospitality"))
    ar.ad_type_rules(r, ocr("FARE SALE", "Domestic fares from 999"))
    assert r.fields["ad_type"].status == FieldStatus.review


def test_manufacturer_of_goods_stays():
    r = rec_with(ad_type=conf("manufacturer"), category=conf("FMCG > Food & Beverages"))
    ar.ad_type_rules(r, ocr("Butter Cookies"))
    assert r.fields["ad_type"].status == FieldStatus.confirmed


# ---------------------------------------------------------------- sub-category needs words from the ad
def test_subcategory_without_supporting_words_goes_to_review():
    r = rec_with(category=conf("FMCG > Personal & Home Care"), brand=FieldResult(value="ZZQX"),
                 product=FieldResult(value="Cement", english="cement"))
    ar.subcategory_rules(r, ocr("STRONG FOUNDATIONS FOR STRONG DREAMS", "CONCRETE HD"))
    assert r.fields["category"].status == FieldStatus.review


def test_subcategory_with_supporting_words_stays():
    r = rec_with(category=conf("FMCG > Personal & Home Care"), brand=FieldResult(value="ZZQX"),
                 product=FieldResult(value="Herbal Shampoo"))
    ar.subcategory_rules(r, ocr("Strong hair in 14 days"))
    assert r.fields["category"].status == FieldStatus.confirmed


def test_top_level_category_not_gated():
    r = rec_with(category=conf("Fashion & Apparel"), product=FieldResult(value="Sandals"))
    ar.subcategory_rules(r, ocr("Sandals"))
    assert r.fields["category"].status == FieldStatus.confirmed


# ---------------------------------------------------------------- mixed scripts inside a word
def test_mixed_script_word_goes_to_review():
    r = rec_with(headline=conf("तમારા પરિવાર માટે"))
    ar.mixed_script_rules(r)
    assert r.fields["headline"].status == FieldStatus.review


@pytest.mark.parametrize("text", ["તમારા પરિવાર માટે", "Amul बटर", "नया कायम चूर्ण"])
def test_clean_script_words_stay(text):
    r = rec_with(headline=conf(text))
    ar.mixed_script_rules(r)
    assert r.fields["headline"].status == FieldStatus.confirmed


# ---------------------------------------------------------------- switch
def test_switch_off_restores_frozen_behaviour(monkeypatch):
    monkeypatch.setenv("ADREADER_RULES_V3", "0")
    r = rec_with(language=conf(["Hindi"]), ad_type=conf("recruitment"), category=conf("Education"))
    ar.apply(r, ocr("उपलब्ध आहे आणि", "ADMISSION OPEN"))
    assert r.fields["language"].value == ["Hindi"] and r.fields["ad_type"].status == FieldStatus.confirmed
