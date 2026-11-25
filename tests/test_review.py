"""Save rules: review gate, ID policy, corrections log, label export."""
import json

import pytest

from adreader import db, review
from adreader.schema import FieldStatus
from tests.test_pipeline_outputs import ADS, _record


@pytest.fixture(scope="module")
def kia():
    return _record(ADS["kia"])


def test_red_fields_not_prefilled_and_ids_only_when_confirmed(kia):
    e = review.initial_edits(kia)
    for name, f in e["fields"].items():
        r = kia.fields.get(name)
        if r is not None and r.status == FieldStatus.missing:
            assert f["value"] is None
        if r is not None and r.status != FieldStatus.confirmed:
            assert f["id"] is None
    assert e["fields"]["brand"]["id"] is not None        # confirmed match pre-selected


def test_gate_blocks_unchecked(kia):
    e = review.initial_edits(kia)
    assert review.pending_checks(e)                      # Kia has flagged fields
    with pytest.raises(ValueError):
        review.build_final(kia, e, "t")


def test_suggestion_never_saved_as_id(kia):
    e = review.initial_edits(kia)
    for f in e["fields"].values():
        f["checked"] = True
    for rows in e["lists"].values():
        for r in rows:
            r["checked"] = True
    for a in e["attributes"].values():
        a["checked"] = True
    final = review.build_final(kia, e, "t")
    assert final["fields"]["advertiser"]["id"] is None                 # suggestion not saved
    assert final["fields"]["brand"]["id"] == kia.fields["brand"].matched_id


def test_reviewer_correction_logged_and_label_format(kia):
    e = review.initial_edits(kia)
    for grp in (e["fields"].values(), e["attributes"].values()):
        for f in grp:
            f["checked"] = True
    for rows in e["lists"].values():
        for r in rows:
            r["checked"] = True
    e["fields"]["advertiser"]["value"] = "Allied Kia / West Coast Kia"
    e["fields"]["advertiser"]["id"] = 2                                # reviewer explicitly picks
    final = review.build_final(kia, e, "t")
    assert final["fields"]["advertiser"]["id"] == 2
    assert any(c["field"] == "advertiser" for c in final["corrections"])
    lab = review.to_label(final)
    assert lab["advertiser"] == "Allied Kia / West Coast Kia" and lab["advertiser_id"] == 2
    assert lab["phones"] == ["18001085005", "9289305692", "9351229009"]
    # saving + cleanup
    with db.connect() as conn:
        sid = review.save_submission(conn, final, kia)
        row = conn.execute("SELECT record_json FROM submissions WHERE id=?", (sid,)).fetchone()
        assert json.loads(row["record_json"])["final"]["fields"]["advertiser"]["id"] == 2
        conn.execute("DELETE FROM submissions WHERE id=?", (sid,))
        conn.commit()
