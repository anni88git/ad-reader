"""Stage 3: structured extraction with a vision-language model.

The model sees the ad image AND the numbered OCR transcript. It must return every field with
a verbatim `evidence` quote so that stage 4 (validate.py) can verify each value against the
OCR text in code. The model never sees or produces database IDs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import seed_data
from .ocr import OCRResult
from .schema import CATEGORY_ATTRIBUTES, AdType, Ev, RawDealer, RawExtraction, RawPrice
from .vlm import VLMClient, parse_json

CATEGORY_NAMES = [name for _, name, _ in seed_data.CATEGORIES]

SYSTEM = """You are a meticulous data-entry operator for a newspaper-advertisement database in India.
You transcribe what is printed. You never invent, infer or "improve" information.
Accuracy matters more than completeness: an empty field is acceptable, a wrong field is not."""

INSTRUCTIONS = """Extract the advertisement into the JSON structure below.

## Hard rules
1. EVIDENCE: for every field, "evidence" must be text copied EXACTLY as printed in the ad, in its
   original script (Devanagari, Tamil, ...), including the ₹ sign, units and punctuation. Never
   translate or paraphrase evidence. If you cannot point to printed text, the value must be null.
2. NO GUESSING: if something is not printed (or is unreadable), use null / []. Do not fill a field
   from general knowledge (e.g. do not add a website, price or phone that is not printed).
3. NUMBERS: read digits one by one. Prices only from amounts that are explicitly prices (₹, Rs,
   रु, "price", "मात्र", "only", EMI...). Grams, ml, mm, km, years, percentages, dates, model
   numbers and regulatory numbers are NOT prices. A struck-out/crossed price is the old price.
   Benefit / savings / cashback / discount / exchange-bonus / "worth" amounts are OFFERS, not prices:
   list them in "offers" (or as a price with a label that says benefit / savings / cashback).
4. OCR TRANSCRIPT: a machine OCR transcript is given below as a spelling/digit reference. It can
   contain errors (lines marked "?" are low-confidence) and it misses stylised text. The IMAGE is
   the primary truth. If the OCR and the image disagree on a digit or name, re-check the image and
   add a note to "uncertain_notes".
5. "value" fields for brand/product/model_variant/advertiser/category must be in English (Latin
   script) — transliterate brand names (e.g. "हॉर्लिक्स" -> "Horlicks"). For headline,
   sub_headline, description, offers, features, warranty and call_to_action keep "value" in the
   original language and put an English translation in "english" when it is not English.
6. List fields: one item per distinct feature / offer / phone / website. Do not merge.
7. If you are unsure about any field, still give your best reading but explain in "uncertain_notes".

## Field definitions
- brand: the product brand being advertised (e.g. "Kia", "Horlicks"). Not the dealer, not the newspaper.
- product: the specific product / model / service line (e.g. "Sorento"). null if the ad is brand-level.
- model_variant: variant, trim, powertrain or pack size if printed (e.g. "Hybrid", "500 g").
- advertiser: who placed and paid for the ad. For dealer ads, the dealer name(s) joined with " / ".
  For a brand's own ad, the company (e.g. "Kia India").
- ad_type: exactly one of {ad_types}.
  "dealer" = brand creative with local dealer names/addresses/phones; "manufacturer" = brand's own
  national ad without local dealers; "retailer" = shop / multi-brand store.
- category: exactly one of these values: {categories}
- headline: the main marketing message (usually the largest slogan/claim). It must NOT be just
  the brand or product name (e.g. for "Kia Sorento / Superior tech and elegance, beyond the class."
  the headline is "Superior tech and elegance, beyond the class.").
- sub_headline: the secondary line or brand tagline (e.g. "Movement that inspires"), if any.
- description: the main descriptive paragraph (body text), verbatim, if any.
- body_copy_summary: what the ad communicates, as short ENGLISH phrases written like a data-entry
  operator would (specific, not generic). Example for a car dealer ad: ["Kia Sorento with feature
  list", "introductory price Rs 27.99 lakh", "scan QR to book", "toll-free and dealer numbers",
  "3+2 year warranty", "2 dealers in Udaipur"]. Never invent facts that are not printed.
- ad_message: 1-2 plain-English sentences explaining what the ad means: what is offered, the key
  claim or offer, and what the reader should do. Example: "Namaste India unsalted butter, 100 g now
  Rs 44 (was Rs 48), made from pure milk; call Deepak Singh on 7988859406 or order via the app."
  Use only printed facts. This is an interpretation, so it has no evidence field.
- prices: every printed price: amount_text exactly as printed, label = the words attached to it
  (e.g. "Introductory price starts at", "old price (struck out)", "offer price"), item = what it is for.
- offers: discounts, buyback, EMI, free gifts, exchange offers, festival offers.
- features: product features / claims as printed.
- warranty: warranty / guarantee text.
- phones, emails, websites, social_handles, addresses: as printed (each item value + evidence).
- dealers: each dealer / store / branch with its name, area, city, phones and evidence.
- call_to_action: the ACTION the reader is asked to take (book, call, visit, download, apply,
  scan, order...). value = the printed words that ask for it (verbatim, original script);
  english = the action in plain English, e.g. "Call Deepak Singh on 7988859406 for details".
  A slogan or product claim is NOT a call to action. null if the ad asks for no action.
- reference_numbers: plan / product numbers and regulatory or legal identifiers exactly as printed,
  e.g. "Plan No. 857", "UIN: 512N337V04", "IRDAI Regn No.: 512", "RERA No. ...", "Tender No. ...".
  Each item value = the identifier with its label.
- qr_code_present / disclaimer_present: true / false.
- category_attributes: ONLY for the attribute names listed for the chosen category's group:
{cat_attrs}
  Each attribute is {{"value": ..., "evidence": "...", "english": ...}}. Omit attributes not printed.
- language: languages of the printed text, e.g. ["Hindi", "English"].

## Output
Return ONLY one JSON object, no markdown, exactly with these keys:
{skeleton}

## OCR transcript (reference only)
{ocr}
"""

_EV = {"value": None, "evidence": None, "english": None}
SKELETON = {
    "language": [], "brand": _EV, "product": _EV, "model_variant": _EV, "advertiser": _EV,
    "ad_type": _EV, "category": _EV, "headline": _EV, "sub_headline": _EV, "description": _EV,
    "body_copy_summary": [], "ad_message": None, "prices": [{"amount_text": None, "label": None, "item": None}],
    "offers": [_EV], "features": [_EV], "warranty": _EV, "phones": [_EV], "emails": [_EV],
    "websites": [_EV], "social_handles": [_EV], "addresses": [_EV],
    "dealers": [{"name": None, "area": None, "city": None, "phones": [], "evidence": None}],
    "call_to_action": _EV, "reference_numbers": [_EV], "qr_code_present": None, "disclaimer_present": None,
    "category_attributes": {"<attribute_name>": _EV}, "uncertain_notes": [],
}

# category display name -> attribute group in CATEGORY_ATTRIBUTES
_ATTR_GROUP = {"Automobile": "automobile", "Real Estate": "real_estate", "Education": "education",
               "Jewellery": "jewellery", "Retail & Consumer Electronics": "retail_electronics",
               "Banking & Finance": "banking_finance", "Insurance": "banking_finance",
               "Healthcare": "healthcare", "Recruitment": "recruitment",
               "Government & Public Notices": "public_notice"}


def attr_group(category: str | None) -> str | None:
    if not category:
        return None
    top = category.split(">")[0].strip()
    return _ATTR_GROUP.get(top)


# Prompt v2.5 (Phase 2.5) - EVALUATED AND REJECTED (kept only for reproducibility; never default).
# set50: 90.0% vs 90.4% for a same-day re-run of v2, and 4 wrong values CONFIRMED (all model_variant:
# the product rule pushed text such as "Eye Relief Drops" into model_variant). See
# docs/EVALUATION_REPORT.md section 3d. Active only with EXTRACT_PROMPT_VERSION=2.5.
V25_RULES = """8. KNOWN OCR ERRORS - correct them from the IMAGE, never copy them into a value:
   a) a stylised, coloured or decorative FIRST letter of a line is often dropped by OCR
      ("n Duty, Always!" is printed "On Duty, Always!"; "UNIQUE VIBRANT" may be printed "Be Unique Vibrant");
   b) Indian-script vowel signs, matras, virama and nukta are often wrong or missing in OCR
      (e.g. ि/ी, ु/ू, ो/ॉ, ़; Telugu/Kannada vowel signs; Tamil ு/ூ). Copy the spelling you SEE;
   c) "I", "l" and "1" next to digits ("1.2 l engine", "1.5 l") and "O"/"0";
   d) lakh / crore / thousand words after a number in any script (e.g. "65 लाख" = 65 lakh).
   For headline, sub_headline, description and brand / product names, the image spelling wins over
   the OCR spelling. Keep evidence equal to the value you read in the image."""

V25_PRODUCT = """- product: the product line / model / service name only (e.g. "Sorento", "Magnite", "Polo").
  Edition, trim, engine and pack details go to model_variant (e.g. "Geza Special Edition", "GT TSI"),
  not into product. A slogan or campaign line (e.g. "ONE India brand legacy") is not a product.
  null if the ad is brand-level."""


def prompt_version() -> str:
    import os
    return os.getenv("EXTRACT_PROMPT_VERSION", "2").strip()


def build_prompt(ocr: OCRResult | None, version: str | None = None) -> str:
    cat_attrs = "\n".join(f"    - {top}: {', '.join(CATEGORY_ATTRIBUTES[g])}"
                          for top, g in _ATTR_GROUP.items() if g in CATEGORY_ATTRIBUTES)
    template = INSTRUCTIONS
    if (version or prompt_version()) == "2.5":
        template = template.replace(
            "7. If you are unsure about any field, still give your best reading but explain in \"uncertain_notes\".",
            "7. If you are unsure about any field, still give your best reading but explain in \"uncertain_notes\".\n"
            + V25_RULES.replace("{", "{{").replace("}", "}}"))
        template = template.replace(
            '- product: the specific product / model / service line (e.g. "Sorento"). null if the ad is brand-level.',
            V25_PRODUCT.replace("{", "{{").replace("}", "}}"))
    return template.format(
        ad_types=", ".join(t.value for t in AdType),
        categories="; ".join(CATEGORY_NAMES),
        cat_attrs=cat_attrs,
        skeleton=json.dumps(SKELETON, ensure_ascii=False),
        ocr=ocr.prompt_listing() if ocr and ocr.lines else "(no OCR text available)",
    )


# ----------------------------------------------------------------------------- lenient coercion
def _ev(x: Any) -> Ev:
    """Models sometimes return a bare string/number instead of {value, evidence}. A bare value
    is kept but has NO evidence, so validation will flag it rather than trust it."""
    if isinstance(x, Ev):
        return x
    if isinstance(x, dict):
        value, english = x.get("value"), _s(x.get("english"))
        if english is not None and english == _s(value):
            english = None                     # translation identical to value -> not needed
        return Ev(value=value, evidence=_s(x.get("evidence")), english=english)
    if x in (None, "", []):
        return Ev()
    return Ev(value=x, evidence=None)


def _s(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _ev_list(x: Any) -> list[Ev]:
    if not x:
        return []
    items = x if isinstance(x, list) else [x]
    return [e for e in (_ev(i) for i in items) if e.value not in (None, "", [])]


def _bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if isinstance(x, str) and x.strip().lower() in ("true", "yes"):
        return True
    if isinstance(x, str) and x.strip().lower() in ("false", "no"):
        return False
    return None


def coerce(data: dict) -> RawExtraction:
    def one(i):
        return i.get("value") if isinstance(i, dict) else i

    def strs(v):
        items = v if isinstance(v, list) else ([v] if v else [])
        return [str(one(i)).strip() for i in items if one(i) not in (None, "") and str(one(i)).strip()]

    prices = []
    for p in data.get("prices") or []:
        if isinstance(p, dict) and _s(p.get("amount_text")):
            prices.append(RawPrice(amount_text=_s(p.get("amount_text")), label=_s(p.get("label")),
                                   item=_s(p.get("item"))))
        elif isinstance(p, (str, int, float)) and str(p).strip():
            prices.append(RawPrice(amount_text=str(p).strip()))
    dealers = []
    for d in data.get("dealers") or []:
        if isinstance(d, dict) and _s(d.get("name")):
            dealers.append(RawDealer(name=_s(d.get("name")), area=_s(d.get("area")), city=_s(d.get("city")),
                                     phones=strs(d.get("phones")), evidence=_s(d.get("evidence"))))
    attrs = {}
    for k, v in (data.get("category_attributes") or {}).items():
        if k.startswith("<"):
            continue
        e = _ev(v)
        if e.value not in (None, "", []):
            attrs[str(k)] = e
    return RawExtraction(
        language=strs(data.get("language")),
        brand=_ev(data.get("brand")), product=_ev(data.get("product")),
        model_variant=_ev(data.get("model_variant")), advertiser=_ev(data.get("advertiser")),
        ad_type=_ev(data.get("ad_type")), category=_ev(data.get("category")),
        headline=_ev(data.get("headline")), sub_headline=_ev(data.get("sub_headline")),
        description=_ev(data.get("description")), body_copy_summary=strs(data.get("body_copy_summary")), ad_message=_s(data.get("ad_message")),
        prices=prices, offers=_ev_list(data.get("offers")), features=_ev_list(data.get("features")),
        warranty=_ev(data.get("warranty")), phones=_ev_list(data.get("phones")),
        emails=_ev_list(data.get("emails")), websites=_ev_list(data.get("websites")),
        social_handles=_ev_list(data.get("social_handles")), addresses=_ev_list(data.get("addresses")),
        dealers=dealers, call_to_action=_ev(data.get("call_to_action")),
        reference_numbers=_ev_list(data.get("reference_numbers")),
        qr_code_present=_bool(data.get("qr_code_present")),
        disclaimer_present=_bool(data.get("disclaimer_present")),
        category_attributes=attrs, uncertain_notes=strs(data.get("uncertain_notes")),
    )


# ----------------------------------------------------------------------------- main entry
@dataclass
class ExtractionRun:
    raw: RawExtraction
    model: str
    seconds: float
    reply_text: str
    tokens_in: int | None
    tokens_out: int | None


def extract(img: np.ndarray, ocr: OCRResult | None, client: VLMClient | None = None,
            model: str | None = None) -> ExtractionRun:
    client = client or VLMClient()
    prompt = build_prompt(ocr)
    t = time.perf_counter()
    reply = client.ask([img], prompt, system=SYSTEM, model=model, max_tokens=12000, max_side=2048)
    try:
        data = parse_json(reply.text)
    except ValueError:
        # one repair attempt: ask the model to re-emit valid JSON only
        reply = client.ask([img], prompt + "\n\nYour previous reply was not valid JSON. "
                           "Return ONLY the JSON object.", system=SYSTEM, model=model, max_tokens=12000)
        data = parse_json(reply.text)
    if not isinstance(data, dict):
        raise ValueError("Model did not return a JSON object")
    return ExtractionRun(coerce(data), reply.model, round(time.perf_counter() - t, 2),
                         reply.text, reply.tokens_in, reply.tokens_out)
