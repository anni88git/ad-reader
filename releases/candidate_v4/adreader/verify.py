"""Independent second opinion for the judgement fields.

Independence by construction:
  * the verifier sees ONLY the image - no OCR transcript (so it cannot copy OCR spelling errors)
    and no answer from the primary model (so it cannot simply agree with it);
  * it should be a different model (family) from the primary extractor.
Its answers are compared field by field in validate._second_opinion(); disagreement => REVIEW.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .extract import CATEGORY_NAMES, coerce
from .schema import AdType, RawExtraction
from .vlm import VLMClient, parse_json

PROMPT = """You are checking a newspaper advertisement. Read the IMAGE carefully and answer ONLY in JSON:
{{"brand": "<brand advertised, Latin script>",
  "product": "<specific product/model or null>",
  "ad_type": "<one of: {ad_types}>",
  "category": "<exactly one of: {categories}>",
  "language": ["<languages of the printed text>"],
  "headline": "<the main marketing message (largest slogan), copied EXACTLY as printed in its original script; not just the brand name>",
  "sub_headline": "<secondary line or tagline exactly as printed, or null>",
  "main_price": "<the product's SELLING price exactly as printed, e.g. '₹ 27 99 000' - NOT a discount, cashback, savings, benefit, EMI or gift value; null if none>"}}
Copy text character by character from the image. Use null when something is not printed. Do not guess."""


@dataclass
class Verification:
    raw: RawExtraction
    model: str
    seconds: float
    reply: str


def verify(img: np.ndarray, client: VLMClient | None = None, model: str | None = None) -> Verification:
    client = client or VLMClient()
    prompt = PROMPT.format(ad_types=", ".join(t.value for t in AdType), categories="; ".join(CATEGORY_NAMES))
    t = time.perf_counter()
    reply = client.ask([img], prompt, model=model, max_tokens=1500, max_side=2048)
    d = parse_json(reply.text)
    mp = d.get("main_price")
    data = {k: d.get(k) for k in ("brand", "product", "ad_type", "category", "language", "headline", "sub_headline")}
    data["prices"] = [{"amount_text": mp, "label": "main price"}] if mp else []
    return Verification(coerce(data), reply.model, round(time.perf_counter() - t, 2), reply.text)
