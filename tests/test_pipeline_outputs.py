"""Regression tests on the saved test ads (cached OCR + saved Nemotron extractions; no API calls).
Covers the pre-chunk-6 audit checks."""
import json
import pickle
import os
import subprocess
import sys
from pathlib import Path

import pytest

from adreader import preprocess as pp
from adreader.config import LABELS_DIR, OUTPUTS_DIR, ROOT
from adreader.pipeline import process_ad
from adreader.schema import FieldStatus, RawExtraction

MODEL = "nemotron-3.5-super-vl-120b-a12b"
ADS = {
    "kia": ROOT / "data/ads/_kia_from_screenshot.jpg",
    "horlicks": ROOT / "data/ads/002_horlicks_hi.png",
    "butter": ROOT / "data/ads/003_namaste_butter_hi.png",
    "lic": ROOT / "data/madverse/images/adgal/financial_institutions/LIC_433.jpg",
    "apple": ROOT / "data/madverse/images/adgal/electronics/Apple_mobiles_430.jpg",
}


def _record(path: Path):
    ocr_f = OUTPUTS_DIR / "ocr_cache" / f"{path.stem}.latin-devanagari.pkl"
    ex_f = OUTPUTS_DIR / "extract" / f"{path.stem}.{MODEL}.json"
    if not (path.exists() and ocr_f.exists() and ex_f.exists()):
        pytest.skip(f"cached outputs missing for {path.name}")
    raw = RawExtraction.model_validate(json.loads(ex_f.read_text(encoding="utf-8"))["raw"])
    return process_ad(pp.load_image(path), path.name, ocr=pickle.loads(ocr_f.read_bytes()), raw=raw,
                      verifier_model="").record   # offline: no second-opinion API call in tests


@pytest.fixture(scope="module")
def recs():
    return {k: _record(p) for k, p in ADS.items()}


# ---- check 2: red team
def test_redteam_no_leaks():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/redteam_validate.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert "14 injected errors, 0 leaked" in r.stdout, r.stdout[-2000:]


# ---- check 3: prices
def test_kia_price(recs):
    p = recs["kia"].fields["price"]
    assert p.value == 2799000 and p.status == FieldStatus.confirmed


def test_butter_price_is_offer_not_struck_out(recs):
    p = recs["butter"].fields["price"]
    assert p.value == 44                          # not the struck-out 48
    assert p.status != FieldStatus.confirmed      # two prices -> human confirms roles


# ---- check 4: no IDs saved on unconfirmed fields
def test_no_ids_on_unconfirmed_fields(recs):
    for rec in recs.values():
        for k, r in rec.all_results():
            if r.status != FieldStatus.confirmed:
                assert r.matched_id is None, (rec.image, k)
        for d in rec.lists.get("dealers", []):
            if d.status != FieldStatus.confirmed:
                assert d.value.get("advertiser_id") is None and d.value.get("city_id") is None


def test_kia_advertiser_held_as_suggestion(recs):
    a = recs["kia"].fields["advertiser"]
    assert a.status == FieldStatus.review and a.matched_id is None and a.suggested_id is not None


def test_confirmed_links_keep_ids(recs):
    assert recs["kia"].fields["brand"].matched_id is not None
    assert recs["horlicks"].fields["brand"].status == FieldStatus.confirmed   # via Devanagari alias


# ---- fix A
def test_ev_type(recs):
    attrs = recs["kia"].category_attributes
    assert attrs["fuel_type"].value == ["Hybrid", "Diesel"]
    assert attrs["ev_type"].value == "Non-EV"


# ---- fix F
def test_body_copy_has_status(recs):
    bc = recs["kia"].fields["body_copy"]
    assert bc.value and bc.status == FieldStatus.review and bc.reasons


# ---- fix E
def test_ocr_line_review_categories(recs):
    lines = recs["kia"].ocr_lines_review
    cats = {x["category"] for x in lines}
    assert cats <= {"covered", "ignored", "small_print", "decorative"}
    assert sum(x["category"] == "covered" for x in lines) > len(lines) / 3


# ---- check 5: label file
def test_kia_label_has_missing_features():
    lab = json.loads((LABELS_DIR / "001_kia_from_screenshot.json").read_text(encoding="utf-8"))
    if lab.get("_verified_by"):
        # a human-verified label takes precedence over the draft; check its core facts only
        assert lab["price"] == 2799000 and lab["brand"] == "Kia"
        return
    req = lab["features"]["required"]
    assert "Intelligence by Perplexity" in req and "Voice AI Powered by SoundHound AI" in req
    assert lab["category_attributes"]["ev_type"] == "Non-EV"


# ---- check 6: every field result has a status and (if flagged) a reason
def test_schema_consistency(recs):
    for rec in recs.values():
        for k, r in rec.all_results():
            assert r.status in FieldStatus
            if r.status != FieldStatus.confirmed and r.value not in (None, "", []):
                assert r.reasons, (rec.image, k)
