"""Stage 4: verification. Turns a RawExtraction into an AdRecord in which every value carries
a confidence, a status (confirmed / review / missing) and human-readable reasons.

Principle: a value is CONFIRMED only when independent evidence supports it -
  * its evidence quote is found in the OCR text (a separate model reading the same pixels),
  * the value is consistent with its own evidence,
  * numbers match OCR digit-for-digit and pass format rules,
  * it does not contradict other fields (and, when available, a second VLM agrees).
Anything else is REVIEW (amber) or MISSING (red). Nothing here is decided by an AI model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from rapidfuzz import fuzz

from . import normalize as nz
from .config import CONFIRM_THRESHOLD, REVIEW_THRESHOLD
from .extract import CATEGORY_NAMES
from .ocr import OCRResult, script_of
from .schema import AdRecord, AdType, Ev, FieldResult, FieldStatus, RawExtraction

OCR_SCRIPTS = {"latin", "devanagari", "tamil", "telugu", "kannada"}   # scripts our OCR can read
LANG_SCRIPT = {"english": "latin", "hindi": "devanagari", "marathi": "devanagari",
               "nepali": "devanagari", "sanskrit": "devanagari", "konkani": "devanagari",
               "tamil": "tamil", "telugu": "telugu", "kannada": "kannada", "bengali": "bengali",
               "assamese": "bengali", "gujarati": "gujarati", "punjabi": "gurmukhi",
               "malayalam": "malayalam", "odia": "odia", "oriya": "odia", "urdu": "arabic"}


# ============================================================================ OCR evidence index
class Evidence:
    """Searchable view of the OCR output (best readings + alternatives)."""

    def __init__(self, ocr: OCRResult | None):
        self.ocr = ocr
        lines = ocr.lines if ocr else []
        self.line_keys = [nz.key(l.norm) for l in lines]
        self.corpus = "".join(self.line_keys)
        self.alt_keys = [nz.key(a[0]) for l in lines for a in l.alts]
        self.alt_corpus = "".join(self.alt_keys)
        self.line_digits = [nz.digits(l.norm) for l in lines] + \
                           [nz.digits(a[0]) for l in lines for a in l.alts]
        self.all_digits = "".join(nz.digits(l.norm) for l in lines)
        # a script counts as "seen" only on lines with >= 3 letters of it (a lone '₹' misread as
        # 'र' must not make an English ad look Hindi)
        self.scripts = {l.script for l in lines if l.score >= 0.8 and _letters_of(l.norm, l.script) >= 3}
        self.available = bool(lines)

    def text_support(self, evidence: str | None) -> float:
        """0-100: how well the quoted evidence is found in the OCR text."""
        if not self.available or not evidence:
            return 0.0
        segs = [s for s in re.split(r"\n|\s\|\s|\s{3,}", evidence) if nz.key(s)]
        if not segs:
            return 0.0
        total = sum(len(nz.key(s)) for s in segs)
        contiguous = sum(self._seg_score(nz.key(s)) * len(nz.key(s)) for s in segs) / total
        return round(max(contiguous, self._token_coverage(evidence)), 1)

    def _seg_score(self, k: str) -> float:
        if not k:
            return 0.0
        if k in self.corpus or k in self.alt_corpus:
            return 100.0
        if len(k) <= 3:                       # very short: substring or nothing
            return 0.0
        return max(fuzz.partial_ratio(k, self.corpus),
                   fuzz.partial_ratio(k, self.alt_corpus) if self.alt_corpus else 0)

    def _token_coverage(self, evidence: str) -> float:
        """Order-independent: share of evidence words found (fuzzily) in the OCR text.
        Handles evidence assembled across columns, which OCR reading order splits apart."""
        toks = [nz.key(t) for t in evidence.split()]
        toks = [t for t in toks if len(t) >= 2]
        if not toks:
            return 0.0
        found = 0
        for t in toks:
            if t in self.corpus or t in self.alt_corpus:
                found += len(t)
            elif len(t) >= 5 and fuzz.partial_ratio(t, self.corpus) >= 85:
                found += len(t) * 0.8
        return 100.0 * found / sum(len(t) for t in toks)

    _CURRENCY = re.compile(r"₹|rs\.?|inr|रु|रू|रुपए|रुपये|/-|मात्र|only|price|मूल्य|कीमत|emi", re.I)
    _UNIT = re.compile(r"\d\s*(?:g|gm|gms|kg|ml|l|ltr|mm|cm|km|kmpl|%|ग्राम|गराम|ग्रा|मिली|किलो|वर्ष|साल|years?)\b|"
                       r"\d\s*(?:ग्राम|गराम|ग्रा|मिली|किलो|वर्ष|साल)", re.I)

    def price_context(self, value_digits: str) -> str:
        """For a candidate price, how do the OCR lines containing those digits look?
        'currency' | 'unit' (quantity, not a price) | 'none' | 'absent'"""
        if not self.ocr or not value_digits:
            return "absent"
        ctx = [l.norm for l in self.ocr.lines if value_digits in nz.digits(l.norm)]
        if not ctx:
            return "absent"
        if any(self._CURRENCY.search(c) for c in ctx):
            return "currency"
        if any(self._UNIT.search(c) for c in ctx):
            return "unit"
        return "none"

    def digits_status(self, value_digits: str) -> tuple[str, str | None]:
        """'exact' | 'conflict' (near-identical digits in OCR) | 'absent'; plus OCR's reading."""
        if not value_digits:
            return "absent", None
        if any(value_digits in d for d in self.line_digits) or value_digits in self.all_digits:
            return "exact", value_digits
        n = len(value_digits)
        best, best_dist = None, 99
        for d in self.line_digits:
            for i in range(0, max(1, len(d) - n + 1)):
                cand = d[i:i + n]
                if len(cand) != n:
                    continue
                dist = sum(a != b for a, b in zip(cand, value_digits))
                if dist < best_dist:
                    best, best_dist = cand, dist
        if best is not None and best_dist <= 2 and n >= 6:
            return "conflict", best
        return "absent", None


def _letters_of(text: str, script: str) -> int:
    if script == "latin":
        return sum(ch.isascii() and ch.isalpha() for ch in text)
    rng = {"devanagari": (0x0900, 0x097F), "tamil": (0x0B80, 0x0BFF), "telugu": (0x0C00, 0x0C7F),
           "kannada": (0x0C80, 0x0CFF), "bengali": (0x0980, 0x09FF), "gujarati": (0x0A80, 0x0AFF),
           "gurmukhi": (0x0A00, 0x0A7F), "malayalam": (0x0D00, 0x0D7F)}.get(script)
    return sum(rng[0] <= ord(ch) <= rng[1] for ch in text) if rng else 0


# ============================================================================ scoring helpers
@dataclass
class Check:
    conf: float
    reasons: list[str]


def _script_class(text: str | None) -> str:
    sc = script_of(text or "")
    return sc if sc != "none" else "latin"


def evidence_check(ev: Evidence, evidence: str | None, has_value: bool) -> Check:
    """Base confidence from how well the evidence quote is grounded in OCR."""
    if not has_value:
        return Check(0.0, ["not found in the ad"])
    if not evidence:
        return Check(0.35, ["model gave no evidence quote"])
    sc = _script_class(evidence)
    if sc not in OCR_SCRIPTS:
        return Check(0.6, [f"{sc} script: no OCR cross-check available"])
    if not ev.available:
        return Check(0.6, ["no OCR text to cross-check"])
    s = ev.text_support(evidence)
    strong, medium = (90, 75) if sc == "latin" else (82, 62)   # Indic OCR is noisier
    if s >= strong:
        return Check(0.92, [f"evidence verified in OCR ({s:.0f}%)"])
    if s >= medium:
        return Check(0.72, [f"evidence partly matches OCR ({s:.0f}%)"])
    if sc != "latin":
        return Check(0.55, [f"evidence not confirmed by OCR ({s:.0f}%); {sc} text is hard for OCR"])
    return Check(0.4, [f"evidence NOT found in OCR text ({s:.0f}%) - possible misreading or invention"])


def value_consistent(value: str, evidence: str | None) -> bool | None:
    """Do the value's words occur in its evidence? None = cannot tell (different scripts)."""
    if not evidence or not value:
        return None
    if _script_class(value) != _script_class(evidence):
        return None                                   # transliterated value; checked via master data
    vk = [nz.key(t) for t in re.split(r"[\s/,&+-]+", value) if len(nz.key(t)) >= 2]
    ek = nz.key(evidence)
    if not vk:
        return None
    hit = sum(1 for t in vk if t in ek or fuzz.partial_ratio(t, ek) >= 88)
    return hit / len(vk) >= 0.6


def finalize(value, evidence, english, check: Check, extra_cap: float | None = None,
             reasons: list[str] | None = None) -> FieldResult:
    conf = check.conf if extra_cap is None else min(check.conf, extra_cap)
    rs = check.reasons + (reasons or [])
    if value in (None, "", []):
        status = FieldStatus.missing
    elif conf >= CONFIRM_THRESHOLD:
        status = FieldStatus.confirmed
    elif conf >= REVIEW_THRESHOLD:
        status = FieldStatus.review
    else:
        status = FieldStatus.missing                  # too uncertain: shown as suggestion, not filled
        rs.append("too uncertain to fill - shown as a suggestion only")
    return FieldResult(value=value, evidence=evidence, english=english,
                       confidence=round(conf, 2), status=status, reasons=rs)


def _ev_val(e: Ev) -> str | None:
    if e.value is None:
        return None
    s = str(e.value).strip() if not isinstance(e.value, (list, dict)) else e.value
    return s or None


# ============================================================================ main entry
def validate(raw: RawExtraction, ocr: OCRResult | None, image: np.ndarray | None = None,
             image_name: str = "", verifier: RawExtraction | None = None) -> AdRecord:
    ev = Evidence(ocr)
    rec = AdRecord(image=image_name)
    if ocr:
        rec.quality_warnings = list(ocr.quality.warnings)

    # ---------------------------------------------------------------- simple text fields
    for name in ("brand", "product", "model_variant", "advertiser", "headline", "sub_headline",
                 "description", "warranty", "call_to_action"):
        e: Ev = getattr(raw, name)
        val = _ev_val(e)
        chk = evidence_check(ev, e.evidence, val is not None)
        cap, extra = None, []
        if val is not None and name in ("brand", "product", "model_variant", "advertiser"):
            vc = value_consistent(str(val), e.evidence)
            if vc is False:
                if _script_class(str(val)) == "latin" and ev.text_support(str(val)) >= 92:
                    chk = Check(0.86, ["value not in its evidence quote, but printed elsewhere in the ad"])
                else:
                    cap, extra = 0.5, ["value is not supported by its own evidence quote"]
            elif vc is None and e.evidence and _script_class(str(val)) != _script_class(e.evidence):
                extra = ["value transliterated from another script"]
        rec.fields[name] = finalize(val, e.evidence, e.english, chk, cap, extra)

    # ---------------------------------------------------------------- closed-set fields
    at = _ev_val(raw.ad_type)
    at_ok = at in {t.value for t in AdType}
    rec.fields["ad_type"] = finalize(
        at if at_ok else at, raw.ad_type.evidence, None,
        Check(0.88 if at_ok else 0.3, ["classified by model from ad layout"] if at_ok
              else [f"'{at}' is not a valid ad type"]))

    cat = _ev_val(raw.category)
    cat_ok = cat in CATEGORY_NAMES
    rec.fields["category"] = finalize(
        cat, raw.category.evidence, None,
        Check(0.86 if cat_ok else 0.35, ["category chosen from master list"] if cat_ok
              else [f"'{cat}' is not in the master category list"]))

    # ---------------------------------------------------------------- language vs OCR scripts
    langs = raw.language
    lang_reasons, lang_conf = [], 0.9
    for lg in langs:
        sc = LANG_SCRIPT.get(lg.strip().lower())
        if sc is None:
            lang_conf, lang_reasons = min(lang_conf, 0.6), lang_reasons + [f"unknown language '{lg}'"]
        elif sc in OCR_SCRIPTS and ev.available and sc not in ev.scripts:
            lang_conf = min(lang_conf, 0.6)
            lang_reasons.append(f"{lg}: OCR found no {sc} text")
        elif sc not in OCR_SCRIPTS:
            lang_conf = min(lang_conf, 0.7)
            lang_reasons.append(f"{lg}: script not covered by OCR, not cross-checked")
    rec.fields["language"] = finalize(langs or None, None, None,
                                      Check(lang_conf, lang_reasons or ["matches scripts seen by OCR"]))

    # ---------------------------------------------------------------- phones
    phones: list[FieldResult] = []
    seen_numbers: set[str] = set()
    for p in raw.phones + [Ev(value=x, evidence=d.evidence) for d in raw.dealers for x in d.phones]:
        val = _ev_val(p)
        if not val:
            continue
        pp_ = nz.parse_phone(str(val))
        if pp_.number in seen_numbers:
            continue
        seen_numbers.add(pp_.number)
        st, ocr_reading = ev.digits_status(nz.digits(str(val)))
        if not pp_.valid:
            chk = Check(0.3, pp_.problems)
        elif st == "exact":
            chk = Check(0.95, [f"digits match OCR exactly ({pp_.kind})"] + pp_.problems)
        elif st == "conflict":
            chk = Check(0.4, [f"DIGIT CONFLICT: model read {nz.digits(str(val))}, OCR reads {ocr_reading}"])
        elif not ev.available:
            chk = Check(0.6, ["no OCR text to cross-check digits"])
        else:
            chk = Check(0.5, ["digits not found in OCR text"])
        r = finalize(pp_.number, p.evidence, None, chk)
        r.value = {"number": pp_.number, "kind": pp_.kind, "as_printed": str(val)}
        phones.append(r)
    rec.lists["phones"] = phones

    # ---------------------------------------------------------------- prices
    prices: list[FieldResult] = []
    offers_from_prices: list[FieldResult] = []
    for p in raw.prices:
        pr = nz.parse_price(p.amount_text or "", p.label)
        chk = evidence_check(ev, p.amount_text, pr.amount is not None)
        st, ocr_reading = ev.digits_status(nz.digits(p.amount_text))
        extra, cap = list(pr.problems), None
        if pr.amount is None:
            cap = 0.3
        elif st == "exact":
            d = nz.digits(p.amount_text)
            ctx = ev.price_context(d)
            if ctx == "unit":
                cap, extra = 0.35, extra + [f"'{d}' appears in the ad as a quantity/unit, not as a price"]
            elif len(d) < 4 and ctx != "currency":
                cap, extra = 0.6, extra + [f"short number '{d}' found in OCR but without a ₹/price marker"]
            else:
                chk = Check(max(chk.conf, 0.92), [f"amount digits match OCR ({pr.amount:,})"])
        elif st == "conflict":
            cap, extra = 0.4, extra + [f"DIGIT CONFLICT with OCR reading {ocr_reading}"]
        elif ev.available:
            cap, extra = 0.55, extra + ["amount digits not found in OCR text"]
        if "several numbers in price text" in pr.problems:
            cap = min(cap or 1, 0.6)
        item = finalize({"amount": pr.amount, "currency": "INR", "kind": pr.kind, "label": p.label,
                         "item": p.item, "as_printed": p.amount_text},
                        p.amount_text, None, chk, cap, extra)
        (offers_from_prices if pr.kind in ("cashback", "discount") else prices).append(item)

    # ambiguous old/new price roles: two+ amounts for the same item without clear labels
    kinds = [x.value["kind"] for x in prices]
    if len(prices) >= 2 and kinds.count("price") >= 1 and "old_price" not in kinds:
        for x in prices:
            if x.status == FieldStatus.confirmed:
                x.status = FieldStatus.review
                x.confidence = min(x.confidence, 0.7)
            x.reasons.append("several prices: confirm which is the current price / old (struck-out) price")
    rec.lists["prices"] = prices
    # fix B: the current price is the offer price when one exists, never an old/struck-out price
    main = next((x for x in prices if x.value["kind"] == "offer_price"), None) or \
        next((x for x in prices if x.value["kind"] == "price"), None)
    rec.fields["price"] = main.model_copy(deep=True) if main else FieldResult(reasons=["no price printed"])
    if main:
        rec.fields["price"].value = main.value["amount"]

    # ---------------------------------------------------------------- websites / emails / handles
    def list_field(items: list[Ev], parser=None) -> list[FieldResult]:
        out, seen = [], set()
        for e in items:
            val = _ev_val(e)
            if not val:
                continue
            problems: list[str] = []
            if parser:
                parsed, problems = parser(str(val))
                val = parsed or val
            k = nz.key(str(val))
            if k in seen:
                continue
            seen.add(k)
            chk = evidence_check(ev, e.evidence or str(val), True)
            if chk.conf < 0.9 and e.evidence and _script_class(str(val)) == "latin":
                # the value itself may be printed even if the model's quote added unreadable context
                # (e.g. 'हमें फॉलो करें: LIC India Forever' -> OCR only read 'LIC India Forever')
                vchk = evidence_check(ev, str(val), True)
                if vchk.conf > chk.conf:
                    chk = Check(vchk.conf, [r.replace("evidence", "value") for r in vchk.reasons])
            cap = 0.3 if problems else None
            out.append(finalize(_fix_rupee(val), e.evidence, e.english, chk, cap, problems))
        return out

    rec.lists["websites"] = list_field(raw.websites, nz.parse_website)
    rec.lists["emails"] = list_field(raw.emails, nz.parse_email)
    rec.lists["social_handles"] = list_field(raw.social_handles)
    rec.lists["addresses"] = list_field(raw.addresses)
    rec.lists["features"] = list_field(raw.features)
    refs = list_field(raw.reference_numbers)
    for r in refs:                           # identifiers: every digit must match OCR exactly
        d = nz.digits(str(r.value))
        if d and ev.available:
            st_, reading = ev.digits_status(d)
            if st_ == "conflict":
                _demote(r, 0.4, f"DIGIT CONFLICT with OCR reading {reading}")
            elif st_ == "absent":
                _demote(r, 0.55, "identifier digits not found in OCR text")
    rec.lists["reference_numbers"] = refs
    rec.lists["offers"] = list_field(raw.offers) + offers_from_prices

    # ---------------------------------------------------------------- dealers
    dealers: list[FieldResult] = []
    for d in raw.dealers:
        chk = evidence_check(ev, d.evidence or d.name, bool(d.name))
        name_ok = ev.text_support(d.name) >= (85 if _script_class(d.name) == "latin" else 65)
        cap = None if name_ok else 0.55
        extra = [] if name_ok else ["dealer name not confirmed by OCR"]
        dealers.append(finalize({"name": d.name, "area": d.area, "city": d.city,
                                 "phones": [nz.parse_phone(x).number for x in d.phones]},
                                d.evidence, None, chk, cap, extra))
    rec.lists["dealers"] = dealers

    # ---------------------------------------------------------------- category attributes
    for k, e in raw.category_attributes.items():
        val = _ev_val(e)
        chk = evidence_check(ev, e.evidence, val is not None)
        cap, extra = None, []
        if val is not None and re.search(r"\d", str(val)):
            st, reading = ev.digits_status(nz.digits(str(val)))
            if st == "conflict":
                cap, extra = 0.4, [f"DIGIT CONFLICT with OCR reading {reading}"]
            elif st == "absent" and ev.available and _script_class(e.evidence) in OCR_SCRIPTS:
                cap, extra = 0.6, ["numbers not found in OCR text"]
        rec.category_attributes[k] = finalize(val, e.evidence, e.english, chk, cap, extra)

    # ---- fix A: fuel type as a list + derived EV / Non-EV classification (deterministic rule)
    ft = rec.category_attributes.get("fuel_type")
    if ft is not None and ft.value:
        fuels = nz.parse_fuel_types(" ".join([str(ft.value), ft.evidence or ""]))
        if fuels:
            ft.reasons.append(f"normalised from '{ft.value}'")
            ft.value = fuels
            ev_t = nz.ev_type(fuels)
            derived = FieldResult(value=ev_t, evidence=ft.evidence, confidence=ft.confidence,
                                  status=ft.status, reasons=[f"derived by rule from fuel types {fuels} "
                                                             "(only fully electric = EV)"] +
                                  ([] if ft.status == FieldStatus.confirmed else ["fuel type itself needs review"]))
            rec.category_attributes["ev_type"] = derived
        else:
            _demote(ft, 0.6, "fuel type not recognised (expected petrol/diesel/CNG/hybrid/electric...)")

    # ---------------------------------------------------------------- QR code / disclaimer
    qr_found, qr_data = detect_qr(image if image is not None else (ocr.image if ocr else None))
    if raw.qr_code_present and qr_found:
        qr = Check(0.97, ["QR code detected in image" + (f" ({qr_data})" if qr_data else "")])
    elif raw.qr_code_present:
        qr = Check(0.7, ["model saw a QR code; detector could not confirm (small print?)"])
    elif qr_found:
        qr = Check(0.6, ["detector found a QR code the model did not report"])
    else:
        qr = Check(0.9, ["no QR code found by model or detector"])
    rec.fields["qr_code_present"] = finalize(bool(raw.qr_code_present or qr_found), None, None, qr)
    if qr_data:
        rec.fields["qr_code_data"] = finalize(qr_data, None, None, Check(0.97, ["decoded from QR code"]))

    ocr_text = "\n".join(l.norm.lower() for l in (ocr.lines if ocr else []))
    tc = bool(re.search(r"t\s*&\s*c|terms|conditions apply|\*\s*conditions|शर्तें|नियम|दावे|^\s*\*|\*\*",
                        ocr_text, re.M))
    small_print = _has_small_print(ocr)
    disc = raw.disclaimer_present
    ocr_says = tc or small_print
    if disc == ocr_says:
        dchk = Check(0.92, ["model and OCR agree" + (" (small print detected)" if small_print else "")])
    elif disc is None:
        dchk = Check(0.75 if ocr_says else 0.9, ["model did not say; based on OCR"])
    else:
        dchk = Check(0.7, ["model and OCR disagree about small print / T&C"])
    rec.fields["disclaimer_present"] = finalize(bool(disc if disc is not None else ocr_says), None, None, dchk)

    rec.body_copy_summary = raw.body_copy_summary
    # fix F: body copy follows the same structure; it is a model-written summary, so never auto-confirmed
    rec.fields["body_copy"] = finalize(
        raw.body_copy_summary or None, None, None,
        Check(0.7, ["summary written by the model (not verbatim) - confirm it describes the ad"]))
    rec.fields["ad_message"] = finalize(
        getattr(raw, "ad_message", None) or None, None, None,
        Check(0.7, ["interpretation written by the model (not printed text) - confirm it matches the ad"]))

    # fix D: headline / description must be complete, not a fragment of a longer printed line
    for name in ("headline", "sub_headline", "description"):
        _truncation_check(rec.fields.get(name), ocr)
        _exact_text_check(rec.fields.get(name), ev, name)
    prominence_check(rec, ocr)
    for name in ("description", "warranty", "call_to_action"):
        r = rec.fields.get(name)
        if r is not None and isinstance(r.value, str):
            r.value = _fix_rupee(r.value)

    # ---------------------------------------------------------------- cross-field consistency
    _consistency(rec, raw)
    if verifier is not None:
        _second_opinion(rec, verifier)
        _drop_unsupported_language(rec, verifier)
    # Policy A (accuracy-first): WHICH printed line is the headline is a judgement the checker
    # cannot verify against OCR (the text is real either way). Without an independent second
    # model agreeing, headline / sub-headline always go to the reviewer.
    for name in ("headline", "sub_headline"):
        r = rec.fields.get(name)
        if r is not None and r.status == FieldStatus.confirmed and "second model agrees" not in r.reasons:
            _demote(r, 0.8, "headline choice is a judgement - confirm it is the main line (no second model check)")
    # Which line is the SECONDARY line is the most subjective choice of all; on unseen ads two
    # independent models agreed on the wrong line, so agreement is not enough: always reviewed.
    r = rec.fields.get("sub_headline")
    if r is not None and r.status == FieldStatus.confirmed:
        _demote(r, 0.8, "sub-headline choice is subjective - always confirmed by the reviewer")
    if raw.uncertain_notes:
        rec.record_flags += [f"model note: {n}" for n in raw.uncertain_notes]
    # accuracy pass v3: field-specific evidence gates (see accuracy_rules.py; ADREADER_RULES_V3=0 disables)
    from . import accuracy_rules
    accuracy_rules.apply(rec, ocr)

    # fix E: classify every OCR line so the reviewer sees text the model may have ignored
    recall_from_ocr(rec, ocr)
    confusable_check(rec)
    language_from_ocr(rec, ocr)
    rec.ocr_lines_review = classify_ocr_lines(rec, ocr)
    ignored = [x for x in rec.ocr_lines_review if x["category"] == "ignored"]
    if ignored:
        rec.record_flags.append(f"{len(ignored)} readable OCR line(s) not used by any field - check for missed information")
    return rec


# ============================================================================ recall + confusables
ID_PATTERNS = [
    ("RERA", re.compile(r"\b(?:P5\d{10}|PR/[A-Z]{2}/\w+/\d+|[A-Z]{2,4}RERA\w*\d{4,})\b")),
    ("UIN", re.compile(r"\bUIN[\s:.-]*([0-9]{3}[A-Z][0-9]{3}V[0-9]{2})\b", re.I)),
    ("IRDAI", re.compile(r"\bIRDAI?\s*Regn\.?\s*No\.?\s*[:.-]?\s*\d{2,4}\b", re.I)),
    ("CIN", re.compile(r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")),
    ("GSTIN", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b")),
    ("Plan", re.compile(r"\bPlan\s*No\.?\s*[:.-]?\s*\d{2,4}\b", re.I)),
]
# any digit run with spaces / hyphens (Indian ads group numbers freely: 98765 43210, 875-0039-171,
# 1800 102 2299, 0124-4384164); the phone validator decides what is a real number
_PHONE_IN_TEXT = re.compile(r"(?<![\d.,])(?:\+91[\s-]?)?\d[\d \-]{8,15}\d(?![\d.,])")


def recall_from_ocr(rec: AdRecord, ocr: OCRResult | None) -> None:
    """Phones / registration IDs printed in the ad (found by pattern in high-confidence OCR text)
    but not extracted by the model are added as REVIEW suggestions - never confirmed."""
    if not ocr:
        return
    have_phones = {r.value.get("number") for r in rec.lists.get("phones", []) if isinstance(r.value, dict)}
    have_ids = {nz.key(str(r.value)) for r in rec.lists.get("reference_numbers", [])}
    added_p, added_i = [], []
    for l in ocr.lines:
        if l.score < 0.85:
            continue
        for m in _PHONE_IN_TEXT.finditer(l.norm):
            p = nz.parse_phone(m.group(0))
            if (p.valid and p.kind in ("mobile", "toll_free", "landline") and len(p.number) >= 10
                    and p.number not in have_phones and len(added_p) < 25):
                have_phones.add(p.number)
                r = FieldResult(value={"number": p.number, "kind": p.kind, "as_printed": m.group(0)}, evidence=l.norm,
                                confidence=0.7, status=FieldStatus.review,
                                reasons=["found in the printed text by pattern; the model did not extract it - confirm"])
                added_p.append(r)
        for name, pat in ID_PATTERNS:
            for m in pat.finditer(l.norm):
                v = m.group(0).strip()
                k = nz.key(v)
                if k and not any(k in h or h in k for h in have_ids):
                    have_ids.add(k)
                    added_i.append(FieldResult(value=v, evidence=l.norm, confidence=0.7, status=FieldStatus.review,
                                               reasons=[f"{name} identifier found in the printed text by pattern; "
                                                        "the model did not extract it - confirm"]))
    rec.lists.setdefault("phones", []).extend(added_p)
    rec.lists.setdefault("reference_numbers", []).extend(added_i)
    if added_p or added_i:
        rec.record_flags.append(f"{len(added_p)} phone(s) and {len(added_i)} ID(s) found by pattern that the model missed")


_CONFUSABLE = re.compile(r"(?=\S*\d)(?=\S*[IlOoSBZ])\S*[A-Za-z]\S*")


def confusable_check(rec: AdRecord) -> None:
    """Codes mixing digits with look-alike letters (I/l/1, O/0, S/5, B/8, Z/2) are where OCR and the
    model fail TOGETHER (e.g. '1.2 I Eco TORQ' for '1.2 l'). Such values are never auto-confirmed."""
    def risky(text: str) -> bool:
        for tok in re.split(r"\s+", text):
            if re.search(r"\d", tok) and re.search(r"[IlOoSBZ]", tok):
                return True
        # a lone 'I'/'l'/'O' token directly after a number ('1.2 I', '5 O')
        return bool(re.search(r"\d[\s.,]*\b[IlO]\b", text))
    targets = [rec.fields.get("model_variant"), rec.fields.get("product")] + list(rec.lists.get("reference_numbers", []))
    for r in targets:
        if r is not None and isinstance(r.value, str) and r.status == FieldStatus.confirmed and risky(r.value):
            _demote(r, 0.75, "contains look-alike characters (I/l/1, O/0, S/5, B/8) next to digits - check against the ad")


def language_from_ocr(rec: AdRecord, ocr: OCRResult | None) -> None:
    """If OCR clearly sees an Indian script (>= 3 confident lines with >= 3 letters of it) that the
    model's language list does not cover, flag the language field."""
    r = rec.fields.get("language")
    if r is None or not r.value or not ocr:
        return
    counts: dict[str, int] = {}
    for l in ocr.lines:
        if l.score >= 0.85 and l.script not in ("latin", "none") and _letters_of(l.norm, l.script) >= 3:
            counts[l.script] = counts.get(l.script, 0) + 1
    claimed = {LANG_SCRIPT.get(str(x).strip().lower()) for x in r.value}
    missing = [sc for sc, n in counts.items() if n >= 3 and sc not in claimed]
    if missing:
        _demote(r, 0.6, f"OCR sees {', '.join(missing)} text that the language list does not include")


# ============================================================================ headline prominence
def _line_height(l) -> float:
    b = np.asarray(l.box)
    return float(min(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[1] - b[2])))


def prominence_check(rec: AdRecord, ocr: OCRResult | None) -> None:
    """Wrong-line detection: the headline is normally the most prominent marketing text.
    If OCR sees a clearly LARGER readable line that is not the brand / product / advertiser
    name and not a number or price, the model may have picked a real but wrong line."""
    r = rec.fields.get("headline")
    if r is None or not r.value or not r.evidence or not ocr or not ocr.lines:
        return
    ek = nz.key(r.evidence)
    head_lines = [l for l in ocr.lines if len(nz.key(l.norm)) >= 2 and nz.key(l.norm) in ek]
    if not head_lines:
        return
    h_head = max(_line_height(l) for l in head_lines)
    names = [nz.key(str(x)) for k in ("brand", "product", "advertiser") if rec.fields.get(k)
             for x in (rec.fields[k].value, rec.fields[k].evidence) if x]
    bigger = []
    for l in ocr.lines:
        k = nz.key(l.norm)
        letters = sum(ch.isalpha() for ch in l.norm)
        if l.score < 0.85 or letters < 4 or letters < 0.5 * len(k) or k in ek:
            continue
        if any(k in n or n in k for n in names if n):      # brand / product / advertiser logos
            continue
        if _line_height(l) >= 1.5 * h_head:
            bigger.append(l)
    if bigger:
        big = max(bigger, key=_line_height)
        _demote(r, 0.6, f"a larger printed line exists ('{big.norm[:60]}', "
                        f"{_line_height(big) / h_head:.1f}x the headline size) - check which line is the main headline")


# ============================================================================ exact wording
_RUPEE_MISREAD = re.compile(r"(?<![ऀ-ॿ])र\s?(?=\d)")


def _fix_rupee(v):
    """OCR (and models copying it) read '₹' as Devanagari 'र'. A lone 'र' directly before a
    number is never a Hindi word, so it is restored to '₹'."""
    return _RUPEE_MISREAD.sub("₹", v) if isinstance(v, str) else v


def _exact_text_check(r: FieldResult | None, ev: Evidence, name: str) -> None:
    """Headlines must be verbatim. A fuzzy pass hides spelling-level differences
    (e.g. model 'रिटायरमेण्ट' vs printed 'रिटायरमेंट'), so for headline/sub-headline anything short
    of an exact match goes to review, with OCR's reading offered as the suggestion.
    Descriptions are long and OCR noise is unavoidable, so they need >= 95 (Latin) / 90 (Indic)."""
    if r is None or not r.value or not r.evidence or not ev.available or r.status != FieldStatus.confirmed:
        return
    ek = nz.key(r.evidence)
    if not ek or ek in ev.corpus or ek in ev.alt_corpus:
        return
    sc = _script_class(r.evidence)
    if sc not in OCR_SCRIPTS:
        return
    best_txt, best = _best_ocr_window(ev, ek)
    if name == "description":
        need = 95 if sc == "latin" else 90
        if ev.text_support(r.evidence) >= need:
            return
    sugg = f" - suggestion: {best_txt}" if best_txt and best >= 75 and nz.key(best_txt) != nz.key(str(r.value)) else ""
    _demote(r, 0.75, f"wording is not an exact match to the printed text (closest OCR reading "
                     f"{best:.0f}% similar) - check spelling{sugg}")


def _best_ocr_window(ev: Evidence, ek: str) -> tuple[str | None, float]:
    """Best-matching run of 1-4 consecutive high-confidence OCR lines."""
    lines = [l for l in (ev.ocr.lines if ev.ocr else []) if l.score >= 0.85]
    best_txt, best = None, 0.0
    for i in range(len(lines)):
        for n in range(1, 5):
            win = lines[i:i + n]
            if len(win) < n:
                break
            txt = " ".join(l.norm for l in win)
            s = fuzz.ratio(nz.key(txt), ek)
            if s > best:
                best_txt, best = txt, s
    return best_txt, best


# ============================================================================ fix D: truncation
def _truncation_check(r: FieldResult | None, ocr: OCRResult | None) -> None:
    """Flag text fields whose evidence is only a fragment of a longer printed OCR line."""
    if r is None or not r.value or not r.evidence or not ocr:
        return
    ek = nz.key(r.evidence)
    if len(ek) < 4:
        return
    for l in ocr.lines:
        lk = nz.key(l.norm)
        if ek in lk and len(lk) - len(ek) >= 8 and len(lk) >= 1.4 * len(ek) and l.score >= 0.8:
            _demote(r, 0.7, f"may be incomplete - the printed line reads: '{l.norm[:120]}'")
            return


# ============================================================================ fix E: OCR line review
def classify_ocr_lines(rec: AdRecord, ocr: OCRResult | None) -> list[dict]:
    """Each OCR line -> one of:
       covered      text already used by some extracted field
       small_print  tiny / dense / low-confidence text (disclaimers, unreadable)
       decorative   logos, badges, single stray glyphs, repeats of the brand/product name
       ignored      readable text NOT used by any field -> possible missed information
    """
    if not ocr or not ocr.lines:
        return []
    evid: list[str] = []
    for _, r in rec.all_results():
        for s in (r.evidence, r.value if isinstance(r.value, str) else None):
            if s:
                evid.append(nz.key(s))
        if isinstance(r.value, dict):
            evid += [nz.key(str(v)) for v in r.value.values() if isinstance(v, str)]
    evid_corpus = "|".join(e for e in evid if e)
    names = {nz.key(str(rec.fields[k].value)) for k in ("brand", "product")
             if rec.fields.get(k) and rec.fields[k].value}
    hs = [min(np.linalg.norm(np.subtract(l.box[0], l.box[3])), np.linalg.norm(np.subtract(l.box[1], l.box[2])))
          for l in ocr.lines]
    median_h = sorted(hs)[len(hs) // 2]

    out = []
    for i, (l, h) in enumerate(zip(ocr.lines, hs), 1):
        lk = nz.key(l.norm)
        cover_thr = 90 if l.script in ("latin", "none") else 78     # Indic OCR is noisier
        if len(lk) >= 3 and (lk in evid_corpus or (len(lk) >= 6 and fuzz.partial_ratio(lk, evid_corpus) >= cover_thr)):
            cat, why = "covered", "used by an extracted field"
        elif h < 0.6 * median_h or len(l.norm) > 120 or l.score < 0.8:
            cat, why = "small_print", ("low OCR confidence" if l.score < 0.8 else "tiny or dense small print")
        elif len(lk) <= 3 or lk in names or (len(l.norm.split()) == 1 and l.score < 0.9) \
                or sum(ch.isalnum() for ch in l.norm) < 0.5 * len(l.norm):
            cat, why = "decorative", "logo / badge / stray characters / brand repeat"
        else:
            cat, why = "ignored", "readable text not used by any field"
        out.append({"line": i, "text": l.norm, "score": l.score, "category": cat, "why": why})
    return out


def _demote(r: FieldResult, cap: float, reason: str) -> None:
    r.confidence = min(r.confidence, cap)
    r.reasons.append(reason)
    if r.status == FieldStatus.confirmed and r.confidence < CONFIRM_THRESHOLD:
        r.status = FieldStatus.review


def _consistency(rec: AdRecord, raw: RawExtraction) -> None:
    f = rec.fields
    dealer_names = [d.value["name"] for d in rec.lists.get("dealers", []) if d.value.get("name")]
    at = f["ad_type"].value
    adv = f["advertiser"].value
    if at == "dealer":
        if not dealer_names:
            _demote(f["ad_type"], 0.6, "ad type 'dealer' but no dealer names were extracted")
        elif adv and not any(fuzz.partial_ratio(nz.key(n), nz.key(str(adv))) >= 85 for n in dealer_names):
            _demote(f["advertiser"], 0.55,
                    f"dealer ad: advertiser should be the dealer(s) - suggestion: {' / '.join(dealer_names)}")
    elif at == "manufacturer" and dealer_names:
        _demote(f["ad_type"], 0.65, "dealers are listed - this may be a dealer ad")
        if adv and not any(fuzz.partial_ratio(nz.key(n), nz.key(str(adv))) >= 85 for n in dealer_names):
            _demote(f["advertiser"], 0.65, "dealers are listed - if this is a dealer ad the advertiser is "
                                           f"the dealer(s) - suggestion: {' / '.join(dealer_names)}")

    head, brand, prod = f["headline"].value, f["brand"].value, f["product"].value
    if isinstance(head, str) and head[:1].isascii() and head[:1].islower():
        _demote(f["headline"], 0.6, "headline starts with a lowercase letter - a stylised first letter "
                                    "(logo/icon) may be missing")
    if head and nz.key(str(head)) in {nz.key(str(brand or "")), nz.key(f"{brand or ''}{prod or ''}"),
                                      nz.key(str(prod or ""))}:
        _demote(f["headline"], 0.55, "headline is only the brand/product name")

    if f["brand"].value is None:
        rec.record_flags.append("no brand identified - manual entry required")


LANG_CODES = {"en": "english", "hi": "hindi", "mr": "marathi", "ta": "tamil", "te": "telugu", "kn": "kannada",
              "bn": "bengali", "gu": "gujarati", "ml": "malayalam", "or": "odia", "pa": "punjabi", "ur": "urdu",
              "as": "assamese", "sa": "sanskrit", "ne": "nepali", "kok": "konkani"}


def _drop_unsupported_language(rec: AdRecord, other: RawExtraction) -> None:
    """Phase 2.5 (error audit: 'Hindi' added to English-only ads). Remove a language only when TWO
    independent signals contradict it: OCR - which has a recogniser for that script - found no text
    in it, AND the image-only second model does not list it. The field stays in review."""
    r = rec.fields.get("language")
    if r is None or not isinstance(r.value, list) or len(r.value) < 2 or not other.language:
        return
    ver = {LANG_CODES.get(str(x).strip().lower(), str(x).strip().lower()) for x in other.language}
    no_ocr = {m.group(1).strip().lower() for x in r.reasons for m in [re.match(r"(.+?): OCR found no \w+ text", x)] if m}
    drop = [lg for lg in r.value if lg.strip().lower() in no_ocr and lg.strip().lower() not in ver]
    keep = [lg for lg in r.value if lg not in drop]
    if drop and keep:
        r.value = keep
        _demote(r, 0.75, f"removed {', '.join(drop)}: OCR found no text in that script and the second model "
                         f"does not list it - confirm")


def _second_opinion(rec: AdRecord, other: RawExtraction) -> None:
    """Compare judgement fields with an INDEPENDENT model (image only, no OCR, no primary answer).
    Strict, field-specific agreement; any disagreement => REVIEW with the other reading as the
    suggestion. Agreement can lift a field to CONFIRMED only if nothing else flagged it."""
    from .matcher import default_matcher

    def brand_same(a, b):
        if nz.key(str(a)) == nz.key(str(b)):
            return True
        m = default_matcher()
        ra, rb = m.brand(str(a)).matched, m.brand(str(b)).matched
        return ra is not None and rb is not None and ra.id == rb.id

    def text_same(a, b):                      # headline wording: essentially verbatim
        ka, kb = nz.key(str(a)), nz.key(str(b))
        return ka == kb or (min(len(ka), len(kb)) >= 12 and fuzz.ratio(ka, kb) >= 97)

    def name_same(a, b):
        ka, kb = nz.key(str(a)), nz.key(str(b))
        return ka == kb or fuzz.token_set_ratio(str(a).lower(), str(b).lower()) >= 90

    def lang_same(a, b):
        # the verifier sometimes answers with ISO codes ("hi", "en") - same languages, not a disagreement
        norm = lambda x: {LANG_CODES.get(str(i).strip().lower(), str(i).strip().lower()) for i in (x or [])}  # noqa: E731
        return norm(a) == norm(b)

    def cat_same(a, b):
        # exact, or the system chose the top level and the verifier a sub-category OF THAT top level
        # ("Retail & Consumer Electronics" vs "... > Home Appliances"): both agree on the system's value
        ka, kb = nz.key(str(a)), nz.key(str(b))
        return ka == kb or (">" not in str(a) and ">" in str(b) and nz.key(str(b).split(">")[0]) == ka)

    checks = {"brand": (other.brand.value, brand_same), "product": (other.product.value, name_same),
              "ad_type": (other.ad_type.value, lambda a, b: nz.key(str(a)) == nz.key(str(b))),
              "category": (other.category.value, cat_same),
              "headline": (other.headline.value, text_same),
              "sub_headline": (other.sub_headline.value, text_same),
              "language": (other.language, lang_same)}
    for k, (ov, same) in checks.items():
        r = rec.fields.get(k)
        if r is None or r.value in (None, "", []):
            continue
        if ov in (None, "", []):
            r.reasons.append("second model did not report this field")
            if k in ("headline", "sub_headline"):
                _demote(r, 0.75, "second model found no such line")
            continue
        if same(r.value, ov):
            r.reasons.append("second model agrees")
            blocked = any(w in " ".join(r.reasons) for w in
                          ("not supported", "CONFLICT", "larger printed line", "not an exact match",
                           "may be incomplete", "lowercase", "only the brand", "dealer"))
            if not blocked and r.status != FieldStatus.missing:
                r.confidence = max(r.confidence, CONFIRM_THRESHOLD + 0.03)
                r.status = FieldStatus.confirmed
        else:
            _demote(r, 0.6, f"second model disagrees - suggestion: {ov if not isinstance(ov, list) else ', '.join(ov)}")
    other_amounts = {nz.parse_amount(p.amount_text or "")[0] for p in other.prices} - {None}
    pr = rec.fields.get("price")
    if pr and pr.value is not None:
        if pr.value in other_amounts:
            pr.reasons.append("second model agrees")
        elif other_amounts:
            _demote(pr, 0.6, f"second model read the main price as {sorted(other_amounts)}")


def _has_small_print(ocr: OCRResult | None) -> bool:
    """At least two text lines much smaller than the typical line = fine print block."""
    if not ocr or len(ocr.lines) < 4:
        return False
    hs = sorted(min(np.linalg.norm(np.subtract(l.box[0], l.box[3])), np.linalg.norm(np.subtract(l.box[1], l.box[2])))
                for l in ocr.lines)
    median = hs[len(hs) // 2]
    return sum(1 for h in hs if h < 0.6 * median) >= 2 or \
        sum(1 for l in ocr.lines if len(l.norm) > 120) >= 2       # long dense lines = T&C paragraph


# ============================================================================ QR detection
_QR_CACHE: dict[str, tuple[bool, str | None]] = {}


def detect_qr(img: np.ndarray | None) -> tuple[bool, str | None]:
    """QR code present?  Accepted only on strong evidence: the code DECODES, or at least two
    independent detector/scale combinations locate it (a single detection-only hit is ignored).
    Two OpenCV detectors (classic + ArUco-based) on colour, upscaled and greyscale images: small or
    low-contrast newspaper QR codes are missed by a single pass (7/13 found -> 12/13, 0 false
    detections on 54 labelled ads without a QR code)."""
    if img is None:
        return False, None
    import hashlib
    key = hashlib.md5(img.tobytes()).hexdigest()
    if key in _QR_CACHE:
        return _QR_CACHE[key]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    std = cv2.QRCodeDetector()
    aru = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else None
    tries = [(std, lambda: img), (std, lambda: cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
             (std, lambda: cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))]
    if aru is not None:
        tries += [(aru, lambda: g), (aru, lambda: cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))]
    hits, data = 0, None
    for det, make in tries:
        try:
            ok, decoded, points, _ = det.detectAndDecodeMulti(make())
        except cv2.error:
            continue
        if ok and points is not None and len(points):
            hits += 1
            data = data or next((d for d in decoded if d), None)
            if data or hits >= 2:
                break
    res = (bool(data) or hits >= 2, data)
    if len(_QR_CACHE) > 512:
        _QR_CACHE.clear()
    _QR_CACHE[key] = res
    return res
