"""Accuracy pass v4: five offline fixes from the candidate_v3 error diagnosis (bench200, all 210 errors).
OFF by default; enable with ADREADER_RULES_V4=1 (v3 and the frozen pipeline stay reproducible).

Every value this module CHANGES is left in REVIEW and carries the marker LOCK, which the matcher
respects (it will not upgrade a locked field to confirmed). Nothing here can confirm a value.

  1. ad_type: primary 'manufacturer' vs verifier 'service_provider' -> use the verifier's answer.
     (set50 + holdout20 + bench200: verifier right 16/16 in this disagreement, primary alone right 0.)
  2. price parsing: MRP-only ads, '500 mL ₹28' read as 500, '₹37.6 - 52.67 Lacs' read as 38, and a price
     both models agree on that was hidden as 'too uncertain' because a unit check misfired.
  3. language: garbled-Devanagari ads (other script misread by the Devanagari OCR) -> the verifier's
     language when it names a script our OCR cannot read (Bengali, Gujarati, Gurmukhi, ...).
  4. brand: primary gave none -> the verifier's brand, only if that text is printed in the ad (OCR).
  5. headline: the verifier's headline CONTAINS the primary's (primary took a fragment) and the longer
     text is printed in the ad (OCR) -> the longer headline.
"""
from __future__ import annotations

import os
import re

from . import normalize as nz
from .schema import AdRecord, FieldResult, FieldStatus, RawExtraction

LOCK = "v4: switched value - kept in review"
REVIEW_CAP = 0.6


def enabled() -> bool:
    return os.environ.get("ADREADER_RULES_V4", "0") == "1"


def locked(r: FieldResult | None) -> bool:
    return r is not None and LOCK in r.reasons


def printed(ev, text: str) -> bool:
    """The text is printed in the ad: the whole string is found in OCR AND most of its words are
    (text_support alone can score 100 when the OCR text is SHORTER than the candidate)."""
    return ev.text_support(text) >= 90 and ev._token_coverage(text) >= 85


def _switch(r: FieldResult, value, why: str, evidence: str | None = None) -> None:
    r.value = value
    if evidence is not None:
        r.evidence = evidence
    r.confidence = min(r.confidence or REVIEW_CAP, REVIEW_CAP)
    r.status = FieldStatus.review
    r.matched_id = None
    r.reasons += [why, LOCK]


# ---------------------------------------------------------------- 1. ad type
def fix_ad_type(rec: AdRecord, ver: RawExtraction) -> None:
    r = rec.fields.get("ad_type")
    vv = (ver.ad_type.value or "") if ver.ad_type else ""
    if r is not None and r.value == "manufacturer" and str(vv).strip().lower() == "service_provider":
        _switch(r, "service_provider", "second model says service_provider; in this disagreement the second "
                                       "model has been right every time measured - confirm")


# ---------------------------------------------------------------- 2. price parsing
_CUR = re.compile(r"₹|\brs\.?|\binr\b|रु\.?|रू\.?", re.I)
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|l)\b", re.I)
_MULT = {"lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000, "l": 100_000,
         "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000}


def reparse_amount(text: str) -> int | None:
    """Corrected amount for two known misparses, else None.
    '500 mL ₹28*' -> 28 (the number after the currency sign, not the quantity before it)
    'Rs. 37.6 - 52.67 Lacs*' -> 3,760,000 (the multiplier applies to both ends of a range)"""
    t = nz.to_ascii_digits(text or "")
    m = _RANGE.search(t)
    if m:
        return int(round(float(m.group(1)) * _MULT[m.group(3).lower()]))
    c = _CUR.search(t)
    if c and c.start() > 0 and re.search(r"\d", t[:c.start()]):
        amt, _ = nz.parse_amount(t[c.start():])
        return amt
    # '10 cap pack 295/-': the amount is the number marked '/-', not the pack size before it
    m = re.search(r"(\d[\d,]*)\s*/-", t)
    # only when WORDS separate an earlier number ('10 cap pack 295/-'), never a digit group ('₹ 85 120/-')
    if m and re.search(r"\d[^\d]*[^\W\d_][^\d]*$", t[:m.start()]):
        return int(m.group(1).replace(",", ""))
    return None


def fix_price(rec: AdRecord, ver: RawExtraction) -> None:
    pr = rec.fields.get("price")
    prices = rec.lists.get("prices") or []
    # (a) misparsed amount of the current main price
    if pr is not None and pr.value is not None:
        src = next((p for p in prices if p.value and p.value.get("amount") == pr.value), None)
        new = reparse_amount(str(src.value.get("as_printed"))) if src else None
        if new and new != pr.value:
            old = pr.value
            _switch(pr, new, f"amount re-read from '{src.value.get('as_printed')}' as {new:,} (was {old:,})")
            return
    # (b) MRP is the only price printed -> it IS the selling price
    if pr is None or pr.value is None:
        sellable = [p for p in prices if p.value and p.value.get("kind") in ("price", "offer_price")]
        mrps = [p for p in prices if p.value and p.value.get("kind") == "old_price"
                and "mrp" in f"{p.value.get('label') or ''} {p.value.get('as_printed') or ''}".lower()
                and p.value.get("amount")]
        if not sellable and mrps:
            nf = mrps[0].model_copy(deep=True)
            amt = reparse_amount(str(mrps[0].value.get("as_printed"))) or mrps[0].value["amount"]
            _switch(nf, amt, "only an MRP is printed - the MRP is the selling price")
            rec.fields["price"] = nf
            return
    # (c) hidden as 'too uncertain' only because the unit check misfired, while both models read this
    #     amount with a currency sign
    if pr is not None and pr.value is not None and pr.status == FieldStatus.missing \
            and "second model agrees" in pr.reasons \
            and any("as a quantity/unit" in x for x in pr.reasons):
        src = next((p for p in prices if p.value and p.value.get("amount") == pr.value), None)
        if src and _CUR.search(str(src.value.get("as_printed") or "")):
            _switch(pr, pr.value, "both models read this amount with a ₹ sign; the 'quantity' check matched a "
                                  "nearby number - confirm")


# ---------------------------------------------------------------- 3. language (unsupported scripts)
_NO_OCR_SCRIPTS = {"bengali", "gujarati", "gurmukhi", "odia", "malayalam", "arabic"}


def fix_language(rec: AdRecord, ver: RawExtraction, ocr) -> None:
    from .accuracy_rules import devanagari_lexicon
    from .validate import LANG_CODES, LANG_SCRIPT
    r = rec.fields.get("language")
    if r is None or not isinstance(r.value, list) or not ver.language:
        return
    vlangs = [LANG_CODES.get(str(x).strip().lower(), str(x).strip().lower()).title() for x in ver.language]
    vscripts = {LANG_SCRIPT.get(x.lower()) for x in vlangs}
    pscripts = {LANG_SCRIPT.get(str(x).strip().lower()) for x in r.value}
    new_scripts = (vscripts & _NO_OCR_SCRIPTS) - pscripts
    if not new_scripts:
        return
    n_dev, _, hi, _ = devanagari_lexicon(ocr)
    if n_dev >= 15 and hi / n_dev < 0.02:      # Devanagari 'text' that is not real Hindi/Marathi
        _switch(r, vlangs, f"Devanagari reading is not real Hindi/Marathi text and the second model reads "
                           f"{', '.join(sorted(new_scripts))} script (not covered by OCR) - using its languages")


# ---------------------------------------------------------------- 4. brand (primary empty)
def fix_brand(rec: AdRecord, ver: RawExtraction, ev) -> None:
    r = rec.fields.get("brand")
    vb = ver.brand.value if ver.brand else None
    if r is None or r.value not in (None, "") or not vb or ev is None:
        return
    if printed(ev, str(vb)):
        _switch(r, str(vb), "first model gave no brand; second model's brand is printed in the ad (OCR)",
                evidence=str(vb))
        rec.record_flags = [f for f in rec.record_flags if not f.startswith("no brand identified")]


# ---------------------------------------------------------------- 5. headline fragment
def fix_headline(rec: AdRecord, ver: RawExtraction, ev) -> None:
    r = rec.fields.get("headline")
    vh = ver.headline.value if ver.headline else None
    if r is None or not isinstance(r.value, str) or not vh or ev is None:
        return
    a, b = nz.key(r.value), nz.key(str(vh))
    if a and len(b) > len(a) and a in b and printed(ev, str(vh)):
        _switch(r, str(vh), "headline looks like a fragment; second model's longer headline contains it and is "
                            "printed in the ad (OCR)", evidence=str(vh))
        r.english = ver.headline.english or None


def apply(rec: AdRecord, raw: RawExtraction, verifier: RawExtraction | None, ocr, ev) -> None:
    if not enabled():
        return
    if verifier is not None:
        fix_ad_type(rec, verifier)
        fix_language(rec, verifier, ocr)
        fix_brand(rec, verifier, ev)
        fix_headline(rec, verifier, ev)
    fix_price(rec, verifier)
