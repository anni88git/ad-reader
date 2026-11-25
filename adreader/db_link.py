"""Safe database linking + the final data-entry form (Workstream B).

Guarantees (enforced here, independent of the UI):
  1. A database ID in the form comes ONLY from a row that exists in the master database at the
     moment the form is built - every ID is re-checked with a SELECT; anything else is dropped.
  2. The language model never supplies IDs: extraction has no ID fields, and IDs are set only by the
     string matcher (adreader/matcher.py) or by the reviewer picking an existing record.
  3. Ambiguous / near matches stay in review with their candidates; no ID until a human picks one.
  4. No match -> explicitly "Unmatched - new record" (id None), never forced onto a wrong record.
  5. Database missing / unreadable / empty -> every ID is None and the fields go to review; the
     pipeline keeps running (no crash, no invented IDs).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Any

from .schema import AdRecord, FieldStatus

LINKED_TABLES = {"brand": "brands", "product": "products", "advertiser": "advertisers", "category": "categories"}
DB_DOWN = "database unavailable - no record could be looked up; nothing was linked"
UNMATCHED = "Unmatched - new record"

# the data-entry form: label -> source field(s)
FORM_FIELDS = [("brand", "Brand"), ("product", "Product"), ("advertiser", "Client / advertiser"),
               ("headline", "Headline"), ("sub_headline", "Sub-headline"), ("description", "Description"),
               ("price", "Price (₹)"), ("language", "Language(s)"), ("category", "Category"),
               ("ad_type", "Ad type"), ("call_to_action", "Call to action"), ("warranty", "Warranty")]
CONTACT_LISTS = ["phones", "websites", "emails", "addresses", "social_handles"]


# ============================================================================ linking
def safe_link_records(rec: AdRecord, ocr_corpus_key: str = "", matcher=None) -> AdRecord:
    """matcher.link_records, but a database failure leaves every link unresolved instead of crashing."""
    from .config import DB_PATH
    from .matcher import default_matcher, link_records
    if matcher is None and not DB_PATH.exists():          # never let sqlite create an empty database
        unresolve_all(rec, f"{DB_DOWN} (file not found: {DB_PATH.name})")
        return rec
    try:
        m = matcher or default_matcher()
        link_records(rec, ocr_corpus_key, matcher=m)
    except (sqlite3.Error, OSError, KeyError) as e:
        unresolve_all(rec, f"{DB_DOWN} ({type(e).__name__}: {str(e)[:80]})")
    return rec


def unresolve_all(rec: AdRecord, reason: str = DB_DOWN) -> None:
    for name in LINKED_TABLES:
        r = rec.fields.get(name)
        if r is None:
            continue
        r.matched_id = r.suggested_id = None
        r.candidates = []
        r.db_match = "db_unavailable" if r.value not in (None, "", []) else None
        if r.value not in (None, "", []):
            r.reasons.append(reason)
            if r.status == FieldStatus.confirmed:
                r.status = FieldStatus.review
                r.confidence = min(r.confidence, 0.7)
    for d in rec.lists.get("dealers", []):
        if isinstance(d.value, dict):
            for k in ("advertiser_id", "city_id", "suggested_advertiser_id", "suggested_city_id"):
                d.value.pop(k, None)
            d.value["advertiser_id"] = d.value["city_id"] = None
            d.value["advertiser_candidates"] = []
    if reason not in rec.record_flags:
        rec.record_flags.append(reason)


# ============================================================================ match report
def match_report(rec: AdRecord) -> list[dict]:
    """One row per linkable field: what the database lookup found, for the reviewer."""
    rows = []
    for name in LINKED_TABLES:
        r = rec.fields.get(name)
        if r is None or r.value in (None, "", []):
            rows.append({"field": name, "value": None, "outcome": "empty", "db_id": None, "candidates": [],
                         "note": "nothing extracted"})
            continue
        cands = [{"id": c.id, "name": c.name, "score": c.score} for c in r.candidates]
        if r.db_match == "db_unavailable":
            outcome, note = "db_unavailable", DB_DOWN
        elif r.matched_id is not None:
            name_ = next((c["name"] for c in cands if c["id"] == r.matched_id), None)
            outcome, note = "matched", f"existing record #{r.matched_id}" + (f" '{name_}'" if name_ else "")
        elif r.suggested_id is not None:
            outcome, note = "suggested", f"record #{r.suggested_id} suggested - confirm in review"
        elif r.db_match == "ambiguous" or len(cands) > 1:
            outcome, note = "ambiguous", "several records fit - pick one in review"
        elif r.db_match == "near_miss" or cands:
            outcome, note = "near_miss", "similar record(s) exist - pick one or add new"
        elif r.db_match == "blocked":
            outcome, note = "blocked", "product not linked because the brand is not linked"
        else:
            outcome, note = "unresolved", UNMATCHED
        rows.append({"field": name, "value": r.value, "outcome": outcome, "db_id": r.matched_id,
                     "suggested_id": r.suggested_id, "candidates": cands, "status": r.status.value, "note": note})
    return rows


# ============================================================================ the form
def verified_ids(conn: sqlite3.Connection | None, wanted: dict[str, set[int]]) -> tuple[dict[str, set[int]], bool]:
    """{table: ids that EXIST in that table}. (empty, False) if the database cannot be read."""
    out: dict[str, set[int]] = {}
    if conn is None:
        return {}, False
    try:
        for table, ids in wanted.items():
            ids = {int(i) for i in ids if isinstance(i, int) or (isinstance(i, str) and i.isdigit())}
            if not ids:
                out[table] = set()
                continue
            q = f"SELECT id FROM {table} WHERE id IN ({','.join('?' * len(ids))})"   # table from a fixed map
            out[table] = {r[0] for r in conn.execute(q, sorted(ids))}
        return out, True
    except sqlite3.Error:
        return {}, False


def build_form(final: dict[str, Any], conn: sqlite3.Connection | None, pending: list[str] | None = None) -> dict:
    """Reviewed record (review.build_final) -> structured data-entry form with VERIFIED database IDs."""
    f, lists = final.get("fields", {}), final.get("lists", {})
    wanted: dict[str, set[int]] = {t: set() for t in ("brands", "products", "advertisers", "categories", "cities")}
    for k, t in LINKED_TABLES.items():
        if (f.get(k) or {}).get("id") is not None:
            wanted[t].add(f[k]["id"])
    for d in lists.get("dealers", []):
        if isinstance(d, dict):
            if d.get("advertiser_id") is not None:
                wanted["advertisers"].add(d["advertiser_id"])
            if d.get("city_id") is not None:
                wanted["cities"].add(d["city_id"])
    ok_ids, db_ok = verified_ids(conn, wanted)
    flags = [] if db_ok else [DB_DOWN + " - all database IDs left empty"]

    def resolve(k, table, the_id, value):
        if value in (None, "", []):
            return None, "empty"
        if the_id is None:
            return None, UNMATCHED
        if the_id in ok_ids.get(table, set()):
            return the_id, "new record (added by reviewer)" if (f.get(k) or {}).get("new_record") else "matched"
        flags.append(f"{k}: id {the_id} is not an existing {table} record - dropped")
        return None, UNMATCHED

    form_fields = {}
    for k, label in FORM_FIELDS:
        v = (f.get(k) or {}).get("value")
        entry = {"label": label, "value": v}
        if k in LINKED_TABLES:
            entry["db_id"], entry["db_resolution"] = resolve(k, LINKED_TABLES[k], (f.get(k) or {}).get("id"), v)
        form_fields[k] = entry
    dealers = []
    for d in lists.get("dealers", []):
        d = dict(d) if isinstance(d, dict) else {"name": str(d)}
        aid, cid = d.get("advertiser_id"), d.get("city_id")
        dealers.append({"name": d.get("name"), "city": d.get("city"), "phones": d.get("phones"),
                        "advertiser_id": aid if aid in ok_ids.get("advertisers", set()) else None,
                        "city_id": cid if cid in ok_ids.get("cities", set()) else None})
    return {
        "form_version": 1,
        "status": "final" if not pending else "draft - items still need review",
        "pending_review": pending or [],
        "image": final.get("image"), "reviewed_by": final.get("reviewed_by"), "reviewed_at": final.get("reviewed_at"),
        "fields": form_fields,
        "prices": lists.get("prices", []), "offers": lists.get("offers", []),
        "contacts": {k: lists.get(k, []) for k in CONTACT_LISTS},
        "dealers": dealers,
        "category_attributes": final.get("category_attributes", {}),
        "id_policy": "every db_id was verified to exist in the master database when this form was built; "
                     "IDs are never produced by the AI model",
        "flags": flags,
    }


def preview_form(rec: AdRecord, edits: dict, conn: sqlite3.Connection | None) -> dict:
    """Live form from the reviewer's current edits (before saving). Pending items are listed, not hidden."""
    import copy

    from . import review
    pend = review.pending_checks(edits)
    e = copy.deepcopy(edits)
    for grp in ("fields", "attributes"):
        for x in e[grp].values():
            x["checked"] = True
    for rows in e["lists"].values():
        for x in rows:
            x["checked"] = True
    final = review.build_final(rec, e, reviewer="(preview)")
    return build_form(final, conn, pending=pend)


def open_db() -> sqlite3.Connection | None:
    """Connection to the master database, or None if it is missing / unreadable."""
    from . import db
    from .config import DB_PATH
    if not DB_PATH.exists():
        return None
    try:
        conn = db.connect()
        conn.execute("SELECT 1 FROM brands LIMIT 1")
        return conn
    except sqlite3.Error:
        return None


def with_db(fn):
    """Call fn(conn) with an open (or None) master-database connection, closing it afterwards."""
    conn = open_db()
    if conn is None:
        return fn(None)
    with closing(conn):
        return fn(conn)
