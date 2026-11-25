"""Master-data database (SQLite). Builds the mock 'existing records' and reads them back.

Tables: categories, brands, products, advertisers, cities, aliases(entity, entity_id, alias).
IDs are created HERE by the database - the AI never produces or guesses an ID.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from .config import DB_PATH
from . import seed_data as S

SCHEMA = """
CREATE TABLE categories (id INTEGER PRIMARY KEY, slug TEXT UNIQUE, name TEXT NOT NULL,
                         parent_id INTEGER REFERENCES categories(id));
CREATE TABLE brands (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                     category_id INTEGER REFERENCES categories(id));
CREATE TABLE products (id INTEGER PRIMARY KEY, brand_id INTEGER NOT NULL REFERENCES brands(id),
                       name TEXT NOT NULL, UNIQUE(brand_id, name));
CREATE TABLE cities (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, state TEXT);
CREATE TABLE advertisers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT,
                          brand_id INTEGER REFERENCES brands(id),
                          city_id INTEGER REFERENCES cities(id), phone TEXT);
CREATE TABLE aliases (entity TEXT NOT NULL, entity_id INTEGER NOT NULL, alias TEXT NOT NULL,
                      UNIQUE(entity, entity_id, alias));
CREATE INDEX idx_alias ON aliases(entity, alias);
CREATE TABLE submissions (id INTEGER PRIMARY KEY, image TEXT, record_json TEXT,
                          reviewed_by TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with closing(connect(path)) as c, c:
        c.executescript(SCHEMA)

        cat_id: dict[str, int] = {}
        for slug, name, parent in S.CATEGORIES:
            cur = c.execute("INSERT INTO categories(slug, name, parent_id) VALUES (?,?,?)",
                            (slug, name, cat_id.get(parent) if parent else None))
            cat_id[slug] = cur.lastrowid
            # leaf name ("Cars") and slug are useful aliases
            leaf = name.split(">")[-1].strip()
            for a in {leaf, slug.replace("_", " ")}:
                c.execute("INSERT OR IGNORE INTO aliases VALUES ('categories',?,?)", (cat_id[slug], a))

        brand_id: dict[str, int] = {}
        for bname, (cslug, aliases, products) in S.BRANDS.items():
            bid = c.execute("INSERT INTO brands(name, category_id) VALUES (?,?)",
                            (bname, cat_id[cslug])).lastrowid
            brand_id[bname] = bid
            for a in aliases:
                c.execute("INSERT OR IGNORE INTO aliases VALUES ('brands',?,?)", (bid, a))
            for p in products:
                c.execute("INSERT INTO products(brand_id, name) VALUES (?,?)", (bid, p))

        city_id: dict[str, int] = {}
        for cname, state, aliases in S.CITIES:
            cid = c.execute("INSERT INTO cities(name, state) VALUES (?,?)", (cname, state)).lastrowid
            city_id[cname] = cid
            for a in aliases:
                c.execute("INSERT OR IGNORE INTO aliases VALUES ('cities',?,?)", (cid, a))

        for name, typ, brand, city, phone in S.ADVERTISERS:
            c.execute("INSERT INTO advertisers(name, type, brand_id, city_id, phone) VALUES (?,?,?,?,?)",
                      (name, typ, brand_id.get(brand) if brand else None,
                       city_id.get(city) if city else None, phone))


def entities(conn: sqlite3.Connection, table: str) -> list[dict]:
    """Return records of a master table with their aliases: [{id, name, aliases, ...}]."""
    rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]  # table is internal, not user input
    alias_rows = conn.execute("SELECT entity_id, alias FROM aliases WHERE entity = ?", (table,))
    by_id: dict[int, list[str]] = {}
    for r in alias_rows:
        by_id.setdefault(r["entity_id"], []).append(r["alias"])
    for r in rows:
        r["aliases"] = by_id.get(r["id"], [])
    return rows


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("categories", "brands", "products", "cities", "advertisers", "aliases")}
