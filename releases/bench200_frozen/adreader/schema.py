"""Data contracts.

Two layers:
  1. RawExtraction  - what the vision-language model must return. Every scalar comes with the
                      verbatim `evidence` text it was read from, so code can verify it against OCR.
  2. AdRecord       - the final form record. Every field is a FieldResult carrying value,
                      confidence, status (confirmed / review / missing), reasons and - for
                      fields linked to master data - the matched record ID (assigned by code,
                      NEVER by the model) or a list of candidates.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------- enums
class AdType(str, Enum):
    manufacturer = "manufacturer"          # brand owner advertising (e.g. Kia India national ad)
    dealer = "dealer"                      # authorised dealer / franchise ad (brand + local dealer)
    retailer = "retailer"                  # multi-brand store / shop
    service_provider = "service_provider"  # bank, hospital, coaching, telecom ...
    real_estate_developer = "real_estate_developer"
    recruitment = "recruitment"
    classified = "classified"
    public_notice = "public_notice"        # govt / tender / legal notice
    other = "other"


class FieldStatus(str, Enum):
    confirmed = "confirmed"   # green: auto-filled, high confidence, evidence verified
    review = "review"         # amber: filled but a human must check
    missing = "missing"       # red: not found / too uncertain -> left empty


# ----------------------------------------------------------------------------- raw (VLM output)
class Ev(BaseModel):
    """A value plus the exact text in the ad that supports it."""
    value: Any = None
    evidence: str | None = None   # verbatim snippet copied from the ad (original script)
    english: str | None = None    # English translation when the original is not English


class RawPrice(BaseModel):
    amount_text: str | None = None     # exactly as printed: "₹ 27 99 000*"
    label: str | None = None           # "Introductory price starts at", "EMI", "ex-showroom"
    item: str | None = None            # what the price is for, if several prices


class RawDealer(BaseModel):
    name: str | None = None
    area: str | None = None
    city: str | None = None
    phones: list[str] = Field(default_factory=list)
    evidence: str | None = None


class RawExtraction(BaseModel):
    language: list[str] = Field(default_factory=list)          # ISO-ish names: "English", "Hindi"
    brand: Ev = Field(default_factory=Ev)                      # value normalised to English/Latin
    product: Ev = Field(default_factory=Ev)                    # e.g. "Sorento"
    model_variant: Ev = Field(default_factory=Ev)              # e.g. "Hybrid"
    advertiser: Ev = Field(default_factory=Ev)                 # who paid for the ad
    ad_type: Ev = Field(default_factory=Ev)                    # one of AdType values
    category: Ev = Field(default_factory=Ev)                   # industry, e.g. "Automobile > Cars"
    headline: Ev = Field(default_factory=Ev)
    sub_headline: Ev = Field(default_factory=Ev)
    description: Ev = Field(default_factory=Ev)                # main descriptive paragraph
    body_copy_summary: list[str] = Field(default_factory=list) # what the ad communicates (derived)
    ad_message: str | None = None                              # plain-English meaning of the ad (derived)
    prices: list[RawPrice] = Field(default_factory=list)
    offers: list[Ev] = Field(default_factory=list)
    features: list[Ev] = Field(default_factory=list)
    warranty: Ev = Field(default_factory=Ev)
    phones: list[Ev] = Field(default_factory=list)
    emails: list[Ev] = Field(default_factory=list)
    websites: list[Ev] = Field(default_factory=list)
    social_handles: list[Ev] = Field(default_factory=list)
    addresses: list[Ev] = Field(default_factory=list)
    dealers: list[RawDealer] = Field(default_factory=list)
    call_to_action: Ev = Field(default_factory=Ev)
    # plan / product numbers and regulatory IDs: UIN, IRDAI, RERA, CIN, GSTIN, tender / ref. no.
    reference_numbers: list[Ev] = Field(default_factory=list)
    qr_code_present: bool | None = None
    disclaimer_present: bool | None = None
    category_attributes: dict[str, Ev] = Field(default_factory=dict)  # see CATEGORY_ATTRIBUTES
    uncertain_notes: list[str] = Field(default_factory=list)          # model's own doubts


# Industry-specific attributes the model is asked for once it knows the category.
CATEGORY_ATTRIBUTES: dict[str, list[str]] = {
    "automobile": ["fuel_type", "mileage", "transmission", "seating_capacity", "engine",
                   "safety_features", "body_type", "buyback_or_finance"],
    "real_estate": ["project_name", "location", "configuration", "carpet_area", "rera_number",
                    "possession_date", "amenities"],
    "education": ["institution", "courses", "exam", "admission_dates", "results_claimed",
                  "scholarship", "mode"],
    "jewellery": ["metal", "making_charge_offer", "gold_rate", "festival"],
    "retail_electronics": ["store_name", "products_on_offer", "discount", "emi_or_cashback",
                           "sale_period"],
    "banking_finance": ["product_type", "interest_rate", "tenure", "eligibility"],
    "healthcare": ["hospital", "specialities", "doctors", "services"],
    "recruitment": ["organisation", "positions", "vacancies", "qualification", "last_date",
                    "how_to_apply"],
    "public_notice": ["issuing_authority", "notice_type", "reference_number", "dates",
                      "subject"],
}


# ----------------------------------------------------------------------------- final record
class Candidate(BaseModel):
    id: int
    name: str
    score: float


class FieldResult(BaseModel):
    value: Any = None
    evidence: str | None = None
    english: str | None = None
    confidence: float = 0.0
    status: FieldStatus = FieldStatus.missing
    reasons: list[str] = Field(default_factory=list)   # why flagged / how verified
    matched_id: int | None = None      # master-data ID: set ONLY when the field is confirmed
    suggested_id: int | None = None    # clear master match on a field still under review
    candidates: list[Candidate] = Field(default_factory=list)
    # master-data lookup outcome (information only - never changes status or IDs):
    # matched | unknown (nothing similar: likely a NEW entity) | near_miss (similar records: misread
    # OR new) | ambiguous (several records fit) | blocked (product: brand not linked) | None
    db_match: str | None = None


class Price(BaseModel):
    amount: int | None = None          # INR, integer rupees: 2799000
    currency: str = "INR"
    label: str | None = None
    item: str | None = None
    amount_text: str | None = None


class AdRecord(BaseModel):
    image: str
    fields: dict[str, FieldResult] = Field(default_factory=dict)          # scalar fields
    lists: dict[str, list[FieldResult]] = Field(default_factory=dict)     # phones, features, prices...
    category_attributes: dict[str, FieldResult] = Field(default_factory=dict)
    body_copy_summary: list[str] = Field(default_factory=list)
    # every OCR line classified for the reviewer: covered / ignored / small_print / decorative
    ocr_lines_review: list[dict[str, Any]] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    record_flags: list[str] = Field(default_factory=list)                 # whole-ad issues
    models_used: list[str] = Field(default_factory=list)
    timings_s: dict[str, float] = Field(default_factory=dict)

    def all_results(self) -> list[tuple[str, FieldResult]]:
        out = list(self.fields.items())
        out += [(f"{k}[{i}]", r) for k, items in self.lists.items() for i, r in enumerate(items)]
        out += [(f"attr.{k}", r) for k, r in self.category_attributes.items()]
        return out

    def flagged(self) -> list[str]:
        return [k for k, r in self.all_results()
                if r.status != FieldStatus.confirmed and r.value not in (None, "", [])]

    def summary(self) -> dict[str, int]:
        c = {s.value: 0 for s in FieldStatus}
        for _, r in self.all_results():
            c[r.status.value] += 1
        return c


# Fields that must be linked to an existing master record (never invent IDs).
MASTER_LINKED_FIELDS = {"brand": "brands", "product": "products", "category": "categories",
                        "advertiser": "advertisers"}
