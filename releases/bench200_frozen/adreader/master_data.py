"""Adding NEW master records (brands, products, advertisers) from reviewed ads.

The not-found rule is unchanged: the system never creates a record or an ID by itself. A record is
created only when a human reviewer, after checking the value, explicitly asks for it. Guards:
  * the value must not already exist (exact name or alias) -> the reviewer must link it instead;
  * similar existing records (possible misreading) must be explicitly acknowledged by ID
    ("I checked these, it is a different one"), otherwise the add is refused;
  * a product needs an existing (or just-created) brand;
  * every addition is written to master_additions (who, when, source ad, evidence, acknowledged
    look-alikes) and can be undone while no saved submission uses it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from rapidfuzz import fuzz

from . import normalize as nz

TABLES = {"brand": "brands", "product": "products", "advertiser": "advertisers"}
NEAR = 85          # name similarity at which an existing record must be acknowledged

_AUDIT = """CREATE TABLE IF NOT EXISTS master_additions (
    id INTEGER PRIMARY KEY, entity TEXT NOT NULL, entity_id INTEGER NOT NULL, name TEXT NOT NULL,
    added_by TEXT NOT NULL, source_image TEXT, evidence TEXT, acknowledged_ids TEXT,
    created_at TEXT NOT NULL, reverted_at TEXT)"""


class MasterDataError(ValueError):
    def __init__(self, msg: str, existing: list[dict] | None = None):
        super().__init__(msg)
        self.existing = existing or []


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_AUDIT)


def _rows(conn, table: str, brand_id: int | None):
    if table == "products":
        q = "SELECT id, name FROM products" + (" WHERE brand_id = ?" if brand_id else "")
        rows = [dict(r) for r in conn.execute(q, (brand_id,) if brand_id else ())]
        for r in rows:
            r["aliases"] = []
        return rows
    rows = [dict(r) for r in conn.execute(f"SELECT id, name FROM {table}")]   # table from TABLES only
    al = {}
    for r in conn.execute("SELECT entity_id, alias FROM aliases WHERE entity = ?", (table,)):
        al.setdefault(r["entity_id"], []).append(r["alias"])
    for r in rows:
        r["aliases"] = al.get(r["id"], [])
    return rows


def check_new(conn: sqlite3.Connection, kind: str, name: str, brand_id: int | None = None) -> dict:
    """What adding `name` would collide with: exact duplicates (blocking) and look-alikes (need ack)."""
    table = TABLES[kind]
    k = nz.key(name)
    exact, similar = [], []
    for r in _rows(conn, table, brand_id):
        names = [r["name"]] + r["aliases"]
        if any(nz.key(n) == k for n in names):
            exact.append({"id": r["id"], "name": r["name"]})
            continue
        score = max(fuzz.ratio(k, nz.key(n)) for n in names)
        if score >= NEAR:
            similar.append({"id": r["id"], "name": r["name"], "score": round(score)})
    return {"exact": exact, "similar": sorted(similar, key=lambda x: -x["score"])}


def add_entity(conn: sqlite3.Connection, kind: str, name: str, *, added_by: str,
               source_image: str = "", evidence: str = "", brand_id: int | None = None,
               category_id: int | None = None, aliases: list[str] | tuple = (),
               acknowledged_ids: list[int] | tuple = (), advertiser_type: str | None = None) -> int:
    if kind not in TABLES:
        raise MasterDataError(f"cannot add {kind!r}: only {', '.join(TABLES)}")
    name = " ".join(str(name or "").split())
    if len(nz.key(name)) < 2:
        raise MasterDataError("name is empty or too short")
    if not added_by or not added_by.strip():
        raise MasterDataError("reviewer name is required to add a master record")
    if kind == "product":
        if not brand_id or not conn.execute("SELECT 1 FROM brands WHERE id = ?", (brand_id,)).fetchone():
            raise MasterDataError("a product needs a linked brand - link or add the brand first")
    chk = check_new(conn, kind, name, brand_id if kind == "product" else None)
    if chk["exact"]:
        raise MasterDataError(f"{name!r} already exists - link the existing record instead", chk["exact"])
    unack = [s for s in chk["similar"] if s["id"] not in set(acknowledged_ids)]
    if unack:
        raise MasterDataError("similar records exist - confirm it is not one of them (possible misreading): "
                              + ", ".join(f"{s['name']} (#{s['id']}, {s['score']}%)" for s in unack), unack)
    ensure_schema(conn)
    table = TABLES[kind]
    with conn:
        if kind == "brand":
            new_id = conn.execute("INSERT INTO brands(name, category_id) VALUES (?,?)", (name, category_id)).lastrowid
        elif kind == "product":
            new_id = conn.execute("INSERT INTO products(brand_id, name) VALUES (?,?)", (brand_id, name)).lastrowid
        else:
            new_id = conn.execute("INSERT INTO advertisers(name, type, brand_id) VALUES (?,?,?)",
                                  (name, advertiser_type, brand_id)).lastrowid
        if kind != "product":
            for a in aliases:
                if a and nz.key(a) != nz.key(name):
                    conn.execute("INSERT OR IGNORE INTO aliases VALUES (?,?,?)", (table, new_id, a))
        conn.execute("INSERT INTO master_additions(entity, entity_id, name, added_by, source_image, evidence, "
                     "acknowledged_ids, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     (table, new_id, name, added_by.strip(), source_image, evidence,
                      json.dumps(sorted(set(acknowledged_ids))), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    _refresh_matcher()
    return new_id


def list_additions(conn: sqlite3.Connection, include_reverted: bool = False) -> list[dict]:
    ensure_schema(conn)
    q = "SELECT * FROM master_additions" + ("" if include_reverted else " WHERE reverted_at IS NULL")
    return [dict(r) for r in conn.execute(q + " ORDER BY id")]


def undo_addition(conn: sqlite3.Connection, addition_id: int) -> None:
    """Remove a reviewer-added record, only if no saved submission references it."""
    ensure_schema(conn)
    row = conn.execute("SELECT * FROM master_additions WHERE id = ? AND reverted_at IS NULL",
                       (addition_id,)).fetchone()
    if not row:
        raise MasterDataError("no such active addition")
    table, eid = row["entity"], row["entity_id"]
    field = {"brands": "brand", "products": "product", "advertisers": "advertiser"}[table]
    for (rj,) in conn.execute("SELECT record_json FROM submissions"):
        f = (json.loads(rj).get("final") or {}).get("fields", {}).get(field) or {}
        if f.get("id") == eid:
            raise MasterDataError(f"record #{eid} is used by a saved submission - cannot undo")
    if table == "brands" and conn.execute("SELECT 1 FROM products WHERE brand_id = ?", (eid,)).fetchone():
        raise MasterDataError("brand has products - undo the products first")
    with conn:
        conn.execute("DELETE FROM aliases WHERE entity = ? AND entity_id = ?", (table, eid))
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (eid,))
        conn.execute("UPDATE master_additions SET reverted_at = ? WHERE id = ?",
                     (datetime.now(timezone.utc).isoformat(timespec="seconds"), addition_id))
    _refresh_matcher()


def _refresh_matcher():
    from .matcher import default_matcher     # local import: matcher imports db, not this module
    default_matcher.cache_clear()
