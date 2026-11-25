"""Workstream B: safe database matching + end-to-end form population.

Guarantees tested: IDs only from existing records, never from the model; exact / normalised / alias
matching; ambiguous -> review with candidates; missing -> unresolved; DB failure -> no invented ID;
form payload verifies every ID against the database."""
import sqlite3

import pytest

from adreader import db, db_link, review
from adreader.matcher import Matcher
from adreader.schema import AdRecord, FieldResult, FieldStatus, RawExtraction
from adreader.validate import validate
from tests.test_accuracy_rules import fake_ocr


@pytest.fixture()
def conn(tmp_path):
    """Real seeded master database in a temp file + two look-alike brands for the ambiguity case."""
    p = tmp_path / "m.db"
    db.build(p)
    c = db.connect(p)
    cat = c.execute("SELECT id FROM categories WHERE name = 'Furniture & Home'").fetchone()[0]
    with c:
        c.execute("INSERT INTO brands(name, category_id) VALUES ('Zenith Paints', ?)", (cat,))
        c.execute("INSERT INTO brands(name, category_id) VALUES ('Zenith Pipes', ?)", (cat,))
    yield c
    c.close()


def _id(conn, table, name):
    return conn.execute(f"SELECT id FROM {table} WHERE name = ?", (name,)).fetchone()[0]


def rec_of(**vals):
    return AdRecord(image="ad.jpg", fields={k: FieldResult(value=v, evidence=str(v), confidence=0.95,
                                                           status=FieldStatus.confirmed) for k, v in vals.items()})


def linked(conn, **vals):
    return db_link.safe_link_records(rec_of(**vals), "", matcher=Matcher(conn))


# ---------------------------------------------------------------- 1. exact match
def test_exact_match_gives_the_real_db_id(conn):
    r = linked(conn, brand="Kia").fields["brand"]
    assert r.matched_id == _id(conn, "brands", "Kia")
    assert db_link.match_report(rec_of())[0]["outcome"] == "empty"


# ---------------------------------------------------------------- 2. normalised / alias match
@pytest.mark.parametrize("value", ["KIA", "  kia  ", "Kia.", "kia-", "Kia India", "Kia Motors", "किआ"])
def test_normalised_and_alias_matches(conn, value):
    assert linked(conn, brand=value).fields["brand"].matched_id == _id(conn, "brands", "Kia")


def test_category_closed_list_exact(conn):
    r = linked(conn, category="Automobile > Cars").fields["category"]
    assert r.matched_id == _id(conn, "categories", "Automobile > Cars")


# ---------------------------------------------------------------- 3. ambiguous -> review with candidates
def test_ambiguous_match_goes_to_review_with_candidates(conn):
    rec = linked(conn, brand="Zenith")
    r = rec.fields["brand"]
    assert r.matched_id is None and r.status == FieldStatus.review
    names = {c.name for c in r.candidates}
    assert {"Zenith Paints", "Zenith Pipes"} <= names
    row = next(x for x in db_link.match_report(rec) if x["field"] == "brand")
    assert row["outcome"] in ("ambiguous", "near_miss", "suggested") and len(row["candidates"]) >= 2


def test_near_miss_is_suggestion_not_id(conn):
    r = linked(conn, brand="Kiaa Motorz").fields["brand"]
    assert r.matched_id is None


# ---------------------------------------------------------------- 4. missing -> unresolved
def test_missing_record_stays_unresolved(conn):
    rec = linked(conn, brand="Qwertyuiop Foods", advertiser="Nobody Traders Pvt Ltd")
    for k in ("brand", "advertiser"):
        assert rec.fields[k].matched_id is None and rec.fields[k].suggested_id is None
        assert rec.fields[k].status == FieldStatus.review
    rows = {x["field"]: x for x in db_link.match_report(rec)}
    assert rows["brand"]["outcome"] == "unresolved" and rows["brand"]["note"] == db_link.UNMATCHED


def test_product_of_unknown_brand_not_linked(conn):
    rec = linked(conn, brand="Qwertyuiop Foods", product="Sonet")      # 'Sonet' is a real Kia product
    assert rec.fields["product"].matched_id is None


# ---------------------------------------------------------------- 5. DB failure -> no invented ID
class _Broken:
    def __getattr__(self, name):
        raise sqlite3.OperationalError("no such table: brands")


def test_database_error_leaves_everything_unresolved():
    rec = db_link.safe_link_records(rec_of(brand="Kia", category="Automobile > Cars"), "", matcher=_Broken())
    for k in ("brand", "category"):
        r = rec.fields[k]
        assert r.matched_id is None and r.suggested_id is None and r.candidates == []
        assert r.status == FieldStatus.review and r.db_match == "db_unavailable"
    assert any("database unavailable" in f for f in rec.record_flags)


def test_missing_database_file_is_not_created(monkeypatch, tmp_path):
    import adreader.config as cfg
    missing = tmp_path / "nope" / "master.db"
    monkeypatch.setattr(cfg, "DB_PATH", missing)
    rec = db_link.safe_link_records(rec_of(brand="Kia"), "")
    assert rec.fields["brand"].matched_id is None and not missing.exists()


def test_empty_tables_match_nothing(tmp_path):
    p = tmp_path / "empty.db"
    c = db.connect(p)
    c.executescript(db.SCHEMA)
    rec = db_link.safe_link_records(rec_of(brand="Kia", category="Automobile > Cars"), "", matcher=Matcher(c))
    assert rec.fields["brand"].matched_id is None and rec.fields["category"].matched_id is None
    c.close()


def test_model_supplied_ids_are_ignored(conn):
    """Even if a model reply contains id fields, extraction has no place for them."""
    raw = RawExtraction.model_validate({"brand": {"value": "Kia", "evidence": "Kia", "id": 9999, "matched_id": 9999},
                                        "language": ["English"]})
    rec = validate(raw, fake_ocr([("Kia", 60)]))
    assert rec.fields["brand"].matched_id is None
    db_link.safe_link_records(rec, "kia", matcher=Matcher(conn))
    assert rec.fields["brand"].matched_id in (None, _id(conn, "brands", "Kia"))
    assert 9999 not in {rec.fields["brand"].matched_id, rec.fields["brand"].suggested_id}


# ---------------------------------------------------------------- form payload: IDs verified
def _final(**fields):
    return {"image": "ad.jpg", "reviewed_by": "t", "reviewed_at": "now",
            "fields": {k: {"value": v[0], "id": v[1]} for k, v in fields.items()},
            "lists": {"phones": [{"number": "9876543210"}], "websites": ["kia.com"], "emails": [],
                      "prices": [{"amount": 799000}], "offers": ["Benefits up to ₹50,000"], "dealers": []},
            "category_attributes": {}}


def test_form_keeps_real_ids_and_drops_fabricated_ones(conn):
    kia = _id(conn, "brands", "Kia")
    form = db_link.build_form(_final(brand=("Kia", kia), advertiser=("Ghost Dealer", 987654),
                                     headline=("Drive the future", None)), conn)
    assert form["fields"]["brand"]["db_id"] == kia and form["fields"]["brand"]["db_resolution"] == "matched"
    assert form["fields"]["advertiser"]["db_id"] is None
    assert form["fields"]["advertiser"]["db_resolution"] == db_link.UNMATCHED
    assert any("987654" in f for f in form["flags"])
    assert form["contacts"]["phones"] and form["prices"] and form["offers"]


def test_form_with_database_down_has_no_ids(conn):
    kia = _id(conn, "brands", "Kia")
    form = db_link.build_form(_final(brand=("Kia", kia)), None)
    assert form["fields"]["brand"]["db_id"] is None and form["flags"]


def test_form_id_from_wrong_table_rejected(conn):
    cat_id = _id(conn, "categories", "Automobile > Cars")
    max_brand = conn.execute("SELECT MAX(id) FROM brands").fetchone()[0]
    form = db_link.build_form(_final(brand=("Kia", max_brand + 1000), category=("Automobile > Cars", cat_id)), conn)
    assert form["fields"]["brand"]["db_id"] is None and form["fields"]["category"]["db_id"] == cat_id


# ---------------------------------------------------------------- end to end: extract -> link -> review -> form
def test_end_to_end_form_population(conn):
    raw = RawExtraction.model_validate({
        "brand": {"value": "Zenith", "evidence": "ZENITH"},
        "product": {"value": "Weather Shield", "evidence": "Weather Shield"},
        "advertiser": {"value": "Qwertyuiop Traders", "evidence": "Qwertyuiop Traders"},
        "headline": {"value": "Colours that last", "evidence": "Colours that last"},
        "category": {"value": "Furniture & Home"}, "ad_type": {"value": "manufacturer"},
        "language": ["English"],
        "phones": [{"value": "98765 43210", "evidence": "98765 43210"}],
        "prices": [{"amount_text": "₹1,299", "label": "Price"}]})
    ocr = fake_ocr([("ZENITH", 90), ("Weather Shield", 50), ("Colours that last", 70),
                    ("Qwertyuiop Traders", 30), ("Call 98765 43210", 30), ("₹1,299", 40)])
    rec = validate(raw, ocr)
    db_link.safe_link_records(rec, "zenithweathershieldcoloursthatlast", matcher=Matcher(conn))

    # ambiguous brand -> review, no ID; unknown advertiser -> unresolved
    assert rec.fields["brand"].matched_id is None and rec.fields["brand"].status == FieldStatus.review
    assert rec.fields["advertiser"].matched_id is None

    edits = review.initial_edits(rec)
    draft = db_link.preview_form(rec, edits, conn)
    assert draft["pending_review"] and draft["status"].startswith("draft")

    # reviewer picks the right record for the ambiguous brand and checks everything flagged
    paints = _id(conn, "brands", "Zenith Paints")
    edits["fields"]["brand"].update(value="Zenith Paints", id=paints)
    for grp in ("fields", "attributes"):
        for x in edits[grp].values():
            x["checked"] = True
    for rows in edits["lists"].values():
        for x in rows:
            x["checked"] = True
    final = review.build_final(rec, edits, "reviewer")
    form = db_link.build_form(final, conn)

    assert form["status"] == "final" and not form["flags"]
    assert form["fields"]["brand"] == {"label": "Brand", "value": "Zenith Paints", "db_id": paints,
                                       "db_resolution": "matched"}
    assert form["fields"]["advertiser"]["db_id"] is None
    assert form["fields"]["advertiser"]["db_resolution"] == db_link.UNMATCHED
    assert form["fields"]["category"]["db_id"] == _id(conn, "categories", "Furniture & Home")
    assert form["fields"]["headline"]["value"] == "Colours that last"
    assert form["contacts"]["phones"] and form["fields"]["price"]["value"] == 1299
    import json
    json.dumps(form, ensure_ascii=False)           # exportable payload
