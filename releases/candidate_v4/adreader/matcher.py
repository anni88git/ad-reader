"""Stage 5: link extracted values to EXISTING master records (brands, products, categories,
advertisers, cities).

Rules (the system must never guess database IDs):
  * an ID is filled only when exactly one record matches clearly
    (score >= MATCH_ACCEPT and ahead of the runner-up by >= MATCH_MARGIN);
  * near matches / ties -> top candidates shown, field set to REVIEW;
  * no match -> REVIEW with "no existing record" (a human decides; records are never auto-created);
  * matching is pure string logic over names + aliases (incl. Devanagari spellings); the
    language model never sees or produces IDs.
Also provides OCR corroboration: if a matched brand's name or any alias is printed in the ad
(found in OCR text), that independently confirms the brand even when the logo is stylised.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz

from . import db
from . import normalize as nz
from .config import CONFIRM_THRESHOLD, MATCH_ACCEPT, MATCH_MARGIN
from .schema import AdRecord, Candidate, FieldResult, FieldStatus

_GENERIC = {"india", "ltd", "limited", "pvt", "private", "the", "co", "company", "group", "inc",
            "motors", "motor", "corporation", "corp", "and"}


@dataclass
class Entity:
    id: int
    name: str
    aliases: list[str]
    extra: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [self.name] + self.aliases


@dataclass
class MatchResult:
    matched: Entity | None
    candidates: list[Candidate]
    reason: str
    kind: str = ""

    def __post_init__(self):
        # lookup outcome for the reviewer; derived here so every return path is covered
        if not self.kind:
            r = self.reason
            if self.matched is not None:
                self.kind = "matched"
            elif r.startswith("no value"):
                self.kind = "none"
            elif r.startswith(("no existing record", "no products on record")):
                self.kind = "unknown"
            elif r.startswith("ambiguous"):
                self.kind = "ambiguous"
            elif r.startswith(("partial match", "no exact record")):
                self.kind = "near_miss"
            elif r.startswith("product not linked"):
                self.kind = "blocked"
            else:
                self.kind = "unknown" if not self.candidates else "near_miss"


def _core(s: str) -> str:
    """Key without generic corporate words: 'Kia India' -> 'kia', 'HDFC Bank Ltd' -> 'hdfcbank'."""
    words = [w for w in re.split(r"[\s\-_/&.,()]+", s.lower()) if w and w not in _GENERIC]
    return nz.key(" ".join(words)) or nz.key(s)


_LOOKALIKE_SEQ = (("rn", "m"), ("cl", "d"), ("vv", "w"))
_LOOKALIKE_CHR = str.maketrans({"1": "l", "i": "l", "|": "l", "0": "o", "5": "s", "8": "b"})


def _canon(s: str) -> str:
    """Name key with OCR look-alike characters folded together (for suggestions only)."""
    k = nz.key(s)
    for a, b in _LOOKALIKE_SEQ:
        k = k.replace(a, b)
    return k.translate(_LOOKALIKE_CHR)


def _score(value: str, name: str) -> float:
    a, b = nz.key(value), nz.key(name)
    if not a or not b:
        return 0.0
    if a == b or _core(value) == _core(name):
        return 100.0
    s = fuzz.ratio(a, b)
    # token-level containment ('Horlicks Chocolate Delight' vs 'Horlicks'), but only for names
    # of reasonable length - prevents 'LG' matching 'LIC' or 'Mi' matching anything with 'mi'
    if min(len(a), len(b)) >= 4:
        s = max(s, fuzz.token_set_ratio(value.lower(), name.lower()) - 4)
    return float(s)


class Matcher:
    def __init__(self, conn: sqlite3.Connection | None = None):
        self._own = conn is None
        self.conn = conn or db.connect()
        self.tables: dict[str, list[Entity]] = {}
        for t in ("brands", "categories", "cities", "advertisers"):
            self.tables[t] = [Entity(r["id"], r["name"], r["aliases"],
                                     {k: v for k, v in r.items() if k not in ("id", "name", "aliases")})
                              for r in db.entities(self.conn, t)]
        self.products = [Entity(r["id"], r["name"], [], {"brand_id": r["brand_id"]})
                         for r in self.conn.execute("SELECT id, name, brand_id FROM products")]

    def close(self):
        if self._own:
            self.conn.close()

    # ------------------------------------------------------------------ core
    def match(self, value: str | None, entities: list[Entity],
              prefer: list | None = None) -> MatchResult:
        """prefer: optional predicates(Entity)->bool, tried in order, used ONLY to break ties
        between equally good candidates (e.g. the ad's category); never promotes a weak match."""
        if not value or not str(value).strip():
            return MatchResult(None, [], "no value")
        scored = []
        for e in entities:
            best = max(_score(str(value), n) for n in e.names)
            exact = any(nz.key(str(value)) == nz.key(n) or _core(str(value)) == _core(n) for n in e.names)
            scored.append((best, exact, e))
        scored.sort(key=lambda x: (-x[0], not x[1]))
        top = [(s, ex, e) for s, ex, e in scored[:3] if s >= 60]
        cands = [Candidate(id=e.id, name=e.name, score=round(s, 1)) for s, _, e in top]
        if not top:
            return MatchResult(None, [], "no existing record matches")
        best_s, best_exact, best_e = top[0]
        second = top[1][0] if len(top) > 1 else 0.0
        exact_ids = [e.id for s, ex, e in top if ex]
        if best_exact and len(exact_ids) == 1:
            return MatchResult(best_e, cands, f"exact match to existing record '{best_e.name}' (#{best_e.id})")
        # Non-exact auto-link only for near-identical SPELLING (character ratio, e.g. a missing
        # hyphen or OCR-level typo) - never for mere containment ('iPhone 11' vs 'iPhone').
        spell = max(fuzz.ratio(nz.key(str(value)), nz.key(n)) for n in best_e.names)
        if spell >= MATCH_ACCEPT and best_s - second >= MATCH_MARGIN:
            return MatchResult(best_e, cands, f"matched existing record '{best_e.name}' (#{best_e.id}, spelling {spell:.0f}%)")
        if best_s >= MATCH_ACCEPT:
            tied = [e for s, _, e in top if best_s - s < MATCH_MARGIN]
            if len(tied) > 1:
                # several equally good records (e.g. 'Honda' -> Honda Cars / Honda 2-Wheelers):
                # pick one only if exactly one fits the ad's category
                for pred in (prefer or []):
                    fit = [e for e in tied if pred(e)]
                    if len(fit) == 1:
                        return MatchResult(fit[0], cands, f"matched '{fit[0].name}' (#{fit[0].id}): "
                                                          f"tie between {len(tied)} records resolved by ad category")
                return MatchResult(None, cands, f"ambiguous: several records match ({', '.join(c.name for c in cands)})")
            return MatchResult(None, cands, f"partial match only - confirm: {', '.join(c.name for c in cands)}")
        return MatchResult(None, cands, f"no exact record; closest: {', '.join(c.name for c in cands)}")

    def lookalike(self, value: str | None, entities: list[Entity]) -> Entity | None:
        """Exactly one record whose name equals `value` after correcting OCR look-alike characters
        ('Kla' -> Kia, 'Kia lndia' -> Kia India). Used ONLY to offer a suggestion on a field that
        stays in review - never to set an ID."""
        c = _canon(str(value or ""))
        if len(c) < 3:
            return None
        hits = {e.id: e for e in entities for n in e.names if _canon(n) == c}
        return next(iter(hits.values())) if len(hits) == 1 else None

    def brand(self, value, category: str | None = None):
        prefer = []
        if category:
            exact_ids = {e.id for e in self.tables["categories"] if nz.key(e.name) == nz.key(category)}
            branch_ids = {e.id for e in self.tables["categories"] if _same_branch(e.name, category)}
            prefer = [lambda e: e.extra.get("category_id") in exact_ids,     # exact category first
                      lambda e: e.extra.get("category_id") in branch_ids]    # then same top level
        return self.match(value, self.tables["brands"], prefer=prefer)

    def category(self, value):
        # closed list: exact equality only
        for e in self.tables["categories"]:
            if value and nz.key(value) == nz.key(e.name):
                return MatchResult(e, [Candidate(id=e.id, name=e.name, score=100)], f"category #{e.id}")
        return self.match(value, self.tables["categories"])

    def product(self, value, brand: Entity | None):
        if brand is None:
            return MatchResult(None, [], "product not linked: brand not matched")
        pool = [p for p in self.products if p.extra["brand_id"] == brand.id]
        if not pool:
            return MatchResult(None, [], f"no products on record for {brand.name}")
        v = str(value or "")
        for n in brand.names:                         # strip the brand from 'Kia Sorento'
            v = re.sub(re.escape(n), " ", v, flags=re.I)
        return self.match(v.strip() or value, pool)

    def advertiser(self, value):
        return self.match(value, self.tables["advertisers"])

    def city(self, value):
        return self.match(value, self.tables["cities"])

    def brand_printed(self, brand: Entity, ocr_corpus_key: str) -> str | None:
        """Return the brand name/alias that is printed in the ad (OCR), if any."""
        for n in sorted(brand.names, key=len, reverse=True):
            k = nz.key(n)
            if len(k) >= 3 and k in ocr_corpus_key:
                return n
        return None


@lru_cache(maxsize=1)
def default_matcher() -> Matcher:
    return Matcher(db.connect())


# ============================================================================ apply to a record
def _set(r: FieldResult, m: MatchResult) -> None:
    r.candidates = m.candidates
    r.db_match = m.kind if m.kind != "none" else None
    if m.matched:
        r.matched_id = m.matched.id
    r.reasons.append(m.reason)


def _require_review(r: FieldResult, cap: float, reason: str | None = None) -> None:
    if reason:
        r.reasons.append(reason)
    r.confidence = min(r.confidence, cap)
    if r.status == FieldStatus.confirmed:
        r.status = FieldStatus.review


def link_records(rec: AdRecord, ocr_corpus_key: str = "", matcher: Matcher | None = None) -> AdRecord:
    m = matcher or default_matcher()
    f = rec.fields

    # ---- brand
    brand_entity = None
    b = f.get("brand")
    if b is not None and b.value:
        cat_val = f.get("category").value if f.get("category") else None
        res = m.brand(str(b.value), category=str(cat_val) if cat_val else None)
        _set(b, res)
        brand_entity = res.matched
        if brand_entity:
            printed = m.brand_printed(brand_entity, ocr_corpus_key)
            # a value switched by accuracy pass v4 must stay in human review
            from .accuracy_rules_v4 import locked as _v4_locked
            if printed and b.status != FieldStatus.confirmed and not _v4_locked(b):
                # independent confirmation: the brand's name/alias is printed in the ad
                b.confidence = max(b.confidence, 0.9)
                b.status = FieldStatus.confirmed if b.confidence >= CONFIRM_THRESHOLD else b.status
                b.reasons.append(f"brand name '{printed}' found printed in the ad (OCR)")
        else:
            _require_review(b, 0.7)
            _suggest_lookalike(b, m.lookalike(b.value, m.tables["brands"]))

    # ---- product (only within the matched brand)
    p = f.get("product")
    if p is not None and p.value:
        res = m.product(str(p.value), brand_entity)
        _set(p, res)
        if res.matched:
            printed = nz.key(res.matched.name) in ocr_corpus_key
            if printed and p.status != FieldStatus.confirmed:
                p.confidence = max(p.confidence, 0.88)
                p.status = FieldStatus.confirmed
                p.reasons.append(f"product name '{res.matched.name}' printed in the ad")
        else:
            _require_review(p, 0.75)

    # ---- category (closed list)
    c = f.get("category")
    if c is not None and c.value:
        res = m.category(str(c.value))
        _set(c, res)
        if not res.matched:
            _require_review(c, 0.5)
        elif brand_entity and brand_entity.extra.get("category_id") not in (None, res.matched.id):
            cat_of_brand = next((e for e in m.tables["categories"]
                                 if e.id == brand_entity.extra["category_id"]), None)
            if cat_of_brand and not _same_branch(cat_of_brand.name, res.matched.name):
                _require_review(c, 0.7, f"brand '{brand_entity.name}' is filed under "
                                        f"'{cat_of_brand.name}' in the master data")

    # ---- advertiser
    a = f.get("advertiser")
    if a is not None and a.value:
        names = [x.strip() for x in str(a.value).split("/") if x.strip()]
        results = [m.advertiser(n) for n in names]
        if len(results) == 1:
            _set(a, results[0])
            if not results[0].matched:
                _require_review(a, 0.75)
                _suggest_lookalike(a, m.lookalike(names[0], m.tables["advertisers"]))
        else:  # several dealers -> each must match
            a.candidates = [c for r_ in results for c in r_.candidates[:1]]
            a.reasons += [r_.reason for r_ in results]
            if not all(r_.matched for r_ in results):
                _require_review(a, 0.75)

    # ---- dealers: link name to advertisers table and city to cities table
    for d in rec.lists.get("dealers", []):
        v = d.value
        dm = m.advertiser(v.get("name"))
        v["advertiser_id"] = dm.matched.id if dm.matched else None
        v["advertiser_candidates"] = [c.model_dump() for c in dm.candidates]
        cm = m.city(v.get("city")) if v.get("city") else MatchResult(None, [], "no city")
        v["city_id"] = cm.matched.id if cm.matched else None
        d.reasons.append(f"dealer: {dm.reason}; city: {cm.reason}")
        if not dm.matched:
            _require_review(d, 0.75)

    _hold_unconfirmed_ids(rec)
    return rec


def _suggest_lookalike(r: FieldResult, e: Entity | None) -> None:
    """Offer the record as a suggestion (field already in review; no ID is set)."""
    if e is None or r.status == FieldStatus.confirmed:
        return
    r.suggested_id = e.id
    r.db_match = "near_miss"
    if all(c.id != e.id for c in r.candidates):
        r.candidates = [Candidate(id=e.id, name=e.name, score=0.0)] + r.candidates
    r.reasons.append(f"possible OCR misreading of existing record '{e.name}' (#{e.id}) - suggestion: {e.name}")


def _hold_unconfirmed_ids(rec: AdRecord) -> None:
    """Fix C: an ID is only 'matched' (saveable) on a CONFIRMED field. On any field still under
    review it is kept as `suggested_id` for the reviewer to accept or change."""
    for _, r in rec.all_results():
        if r.matched_id is not None and r.status != FieldStatus.confirmed:
            r.suggested_id, r.matched_id = r.matched_id, None
            r.reasons.append(f"record #{r.suggested_id} held as a suggestion until this field is confirmed")
    for d in rec.lists.get("dealers", []):
        if d.status != FieldStatus.confirmed and isinstance(d.value, dict):
            for k in ("advertiser_id", "city_id"):
                if d.value.get(k) is not None:
                    d.value[f"suggested_{k}"] = d.value.pop(k)
                    d.value[k] = None


def _same_branch(a: str, b: str) -> bool:
    return a.split(">")[0].strip() == b.split(">")[0].strip()
