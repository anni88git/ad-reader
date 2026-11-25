"""Adding new master records from review: guarded, audited, never automatic."""
import pytest

from adreader import db, master_data as md, review
from adreader.matcher import Matcher, link_records
from adreader.schema import AdRecord, FieldResult, FieldStatus


@pytest.fixture()
def conn(tmp_path):
    p = tmp_path / "m.db"
    db.build(p)
    c = db.connect(p)
    yield c
    c.close()


def _rec(**vals):
    return AdRecord(image="x.jpg", fields={k: FieldResult(value=v, evidence=str(v), confidence=0.95,
                                                          status=FieldStatus.confirmed) for k, v in vals.items()})


# ---------------------------------------------------------------- lookup outcome classification
def test_match_kinds(conn):
    m = Matcher(conn)
    assert m.brand("Kia").kind == "matched"
    assert m.brand("Zorblax Paints").kind == "unknown"          # nothing similar: likely new
    assert m.brand("Kiaa Motors").kind in ("near_miss", "matched")
    rec = link_records(_rec(brand="Zorblax Paints", product="Ultra Gloss"), matcher=m)
    assert rec.fields["brand"].db_match == "unknown"
    assert rec.fields["product"].db_match == "blocked"          # brand not linked
    assert rec.fields["brand"].matched_id is None               # not-found rule unchanged


def test_unknown_brand_still_goes_to_review(conn):
    rec = link_records(_rec(brand="Zorblax Paints"), matcher=Matcher(conn))
    assert rec.fields["brand"].status != FieldStatus.confirmed


# ---------------------------------------------------------------- adding
def test_add_new_brand_then_it_matches(conn):
    bid = md.add_entity(conn, "brand", "Zorblax Paints", added_by="anni", source_image="ad1.jpg")
    m = Matcher(conn)
    r = m.brand("Zorblax Paints")
    assert r.matched is not None and r.matched.id == bid
    a = md.list_additions(conn)
    assert a[0]["name"] == "Zorblax Paints" and a[0]["added_by"] == "anni"


def test_exact_duplicate_refused(conn):
    with pytest.raises(md.MasterDataError, match="already exists"):
        md.add_entity(conn, "brand", "KIA", added_by="anni")


def test_lookalike_needs_acknowledgement(conn):
    chk = md.check_new(conn, "brand", "Horlick")
    assert chk["similar"], "Horlicks should be flagged as a look-alike"
    with pytest.raises(md.MasterDataError, match="similar records"):
        md.add_entity(conn, "brand", "Horlick", added_by="anni")
    ids = [s["id"] for s in chk["similar"]]
    assert md.add_entity(conn, "brand", "Horlick", added_by="anni", acknowledged_ids=ids) > 0


def test_product_needs_brand_and_reviewer(conn):
    with pytest.raises(md.MasterDataError, match="brand"):
        md.add_entity(conn, "product", "Ultra Gloss", added_by="anni")
    with pytest.raises(md.MasterDataError, match="reviewer"):
        md.add_entity(conn, "brand", "Zorblax Paints", added_by=" ")


def test_undo_blocked_when_used_by_submission(conn):
    bid = md.add_entity(conn, "brand", "Zorblax Paints", added_by="anni")
    aid = md.list_additions(conn)[0]["id"]
    import json
    with conn:
        conn.execute("INSERT INTO submissions(image, record_json, reviewed_by) VALUES (?,?,?)",
                     ("x", json.dumps({"final": {"fields": {"brand": {"value": "Zorblax Paints", "id": bid}}}}), "anni"))
    with pytest.raises(md.MasterDataError, match="used by a saved submission"):
        md.undo_addition(conn, aid)


def test_undo_removes_record(conn):
    md.add_entity(conn, "brand", "Zorblax Paints", added_by="anni")
    md.undo_addition(conn, md.list_additions(conn)[0]["id"])
    assert Matcher(conn).brand("Zorblax Paints").matched is None
    assert md.list_additions(conn) == []


# ---------------------------------------------------------------- review flow
def _final_for(rec, edits):
    return review.build_final(rec, edits, "anni")


def test_review_adds_brand_and_product_together(conn):
    rec = link_records(_rec(brand="Zorblax Paints", product="Ultra Gloss"), matcher=Matcher(conn))
    edits = review.initial_edits(rec)
    for f in edits["fields"].values():
        f["checked"] = True
    edits["fields"]["brand"]["new_record"] = {"add": True}
    edits["fields"]["product"]["new_record"] = {"add": True}
    final = _final_for(rec, edits)
    created = review.apply_new_records(conn, final, edits, rec)
    assert [c["field"] for c in created] == ["brand", "product"]
    assert final["fields"]["product"]["id"] and final["fields"]["brand"]["id"]
    row = conn.execute("SELECT brand_id FROM products WHERE id = ?", (final["fields"]["product"]["id"],)).fetchone()
    assert row["brand_id"] == final["fields"]["brand"]["id"]


def test_review_refusal_creates_nothing(conn):
    rec = link_records(_rec(brand="Zorblax Paints", advertiser="Kia India"), matcher=Matcher(conn))
    edits = review.initial_edits(rec)
    for f in edits["fields"].values():
        f["checked"] = True
    edits["fields"]["brand"]["new_record"] = {"add": True}
    edits["fields"]["advertiser"]["new_record"] = {"add": True}     # exists already -> whole request refused
    edits["fields"]["advertiser"]["id"] = None
    final = _final_for(rec, edits)
    final["fields"]["advertiser"]["id"] = None
    with pytest.raises(md.MasterDataError):
        review.apply_new_records(conn, final, edits, rec)
    assert Matcher(conn).brand("Zorblax Paints").matched is None   # brand NOT half-created
