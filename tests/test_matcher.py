import pytest

from adreader.matcher import Matcher


@pytest.fixture(scope="module")
def m():
    return Matcher()


def name(r):
    return r.matched.name if r.matched else None


@pytest.mark.parametrize("value,expected", [
    ("Kia", "Kia"), ("Kia India", "Kia"), ("किआ", "Kia"), ("हॉर्लिक्स", "Horlicks"),
    ("नमस्ते इंडिया", "Namaste India"), ("Life Insurance Corporation of India", "LIC"),
    ("LG", "LG Electronics"), ("HDFC Bank Ltd", "HDFC Bank"), ("Maruti", "Maruti Suzuki"),
])
def test_brand_exact_and_aliases(m, value, expected):
    assert name(m.brand(value)) == expected


@pytest.mark.parametrize("value", ["Hyundia", "Unknown Brand Pvt Ltd", "Honda", "Godrej"])
def test_brand_never_guessed(m, value):
    # typos, unknowns and ambiguous names must NOT be auto-linked
    assert m.brand(value).matched is None


def test_category_tie_break(m):
    assert name(m.brand("Honda", "Automobile > Cars")) == "Honda Cars"
    assert name(m.brand("Honda", "Automobile > Two Wheelers")) == "Honda Motorcycle & Scooter India"


def test_products_scoped_and_strict(m):
    kia, apple = m.brand("Kia").matched, m.brand("Apple").matched
    assert name(m.product("Kia Sorento", kia)) == "Sorento"
    assert name(m.product("iPhone X", apple)) == "iPhone X"
    assert m.product("iPhone 11", apple).matched is None         # containment is not a match
    assert m.product("Sorento Hybrid", kia).matched is None      # suggested only
    assert m.product("Seltos", None).matched is None             # no brand -> no product link


def test_cities(m):
    assert name(m.city("Bombay")) == "Mumbai"
    assert name(m.city("उदयपुर")) == "Udaipur"


# ---- OCR look-alike suggestions (Phase 2): suggestion only, never an ID, field stays in review
def test_lookalike_misreading_is_suggested_not_linked():
    from adreader.matcher import default_matcher, link_records
    from adreader.schema import AdRecord, FieldResult, FieldStatus
    rec = AdRecord(image="x", fields={"brand": FieldResult(value="Kla", evidence="Kla", confidence=0.95,
                                                           status=FieldStatus.confirmed)})
    link_records(rec, matcher=default_matcher())
    b = rec.fields["brand"]
    assert b.matched_id is None and b.status != FieldStatus.confirmed
    kia = default_matcher().brand("Kia").matched
    assert b.suggested_id == kia.id and any("suggestion: Kia" in r for r in b.reasons)


def test_lookalike_needs_unique_match_and_min_length():
    from adreader.matcher import default_matcher
    m = default_matcher()
    assert m.lookalike("Zorblax", m.tables["brands"]) is None
    assert m.lookalike("L6", m.tables["brands"]) is None          # too short to be safe
