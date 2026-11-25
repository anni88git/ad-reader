"""Campaign-level deduplication of newspaper ads.

Each ad gets a signature:
  * image perceptual hash (DCT pHash, 64 bit) - survives re-scanning, JPEG, small scale changes
  * creative text with all digits removed  - survives a changed price / date / phone
  * the numbers (prices, offer amounts, phones) - tell *what* changed between two runs
  * brand / product / headline              - tie different creatives to one campaign

Pair relations (conservative: a link always needs strong evidence; different brands never link):
  DUPLICATE       same creative, same numbers         -> same ad on another date / in another paper
  OFFER_CHANGED   same creative, prices/offers differ -> re-run with a changed price or offer
  NEW_VERSION     same brand + product (or headline), different creative -> new version of a campaign
  same_product    same brand + product, different creative -> reported, NOT merged into one campaign
  (unrelated)     anything else

Campaigns = connected groups of linked ads. Unique creatives = groups linked only by DUPLICATE.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from . import normalize as nz

DUPLICATE, OFFER_CHANGED, NEW_VERSION = "duplicate", "offer_changed", "new_version"
SAME_PRODUCT = "same_product"      # informational only: never merges campaigns
LINKING = {DUPLICATE, OFFER_CHANGED, NEW_VERSION}

# thresholds (64-bit pHash: 0 = identical, ~32 = unrelated)
IMG_SAME = 10          # pHash distance at or below which two images are the same creative
IMG_SIMILAR = 18       # same layout, needs text agreement too
TEXT_SAME = 90         # digit-free creative text similarity for "same creative"
TEXT_VERSION = 60      # same brand + this much shared text -> new version of the campaign
HEADLINE_VERSION = 85
CREATIVE_VERSION = 70  # headline + copy similarity (word order ignored)


@dataclass
class Signature:
    ad_id: str
    phash: int | None
    text: str                      # digit-free normalised text of the whole ad (OCR)
    creative: str                  # digit-free headline / sub-headline / description / features only
    numbers: dict                  # {"prices": set, "offers": set, "phones": set}
    brand: str
    product: str
    headline: str
    meta: dict = field(default_factory=dict)
    brand_words: tuple = ()


def phash(img: np.ndarray) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8]
    bits = (d > np.median(d.ravel()[1:])).ravel()     # median without the DC term
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else bin(a ^ b).count("1")


_DIGITS = re.compile(r"[\d०-९০-৯੦-੯૦-૯୦-୯௦-௯౦-౯೦-೯൦-൯]+")


def creative_text(lines: list[str]) -> str:
    """Words of the creative, digits and punctuation removed, so price/date/phone edits don't matter."""
    words = []
    for ln in lines:
        t = _DIGITS.sub(" ", ln.lower())
        t = re.sub(r"[^\w\s]", " ", t)
        words += [w for w in t.split() if len(w) > 1]
    return " ".join(words)


def _v(rec: dict, f: str):
    x = rec.get("fields", {}).get(f)
    return x.get("value") if isinstance(x, dict) else x


def numbers_of(rec: dict) -> dict:
    prices, offers, phones = set(), set(), set()
    p = _v(rec, "price")
    if isinstance(p, (int, float)):
        prices.add(int(p))
    for it in rec.get("lists", {}).get("prices", []):
        v = it.get("value") if isinstance(it, dict) else None
        amt = v.get("amount") if isinstance(v, dict) else v
        if isinstance(amt, (int, float)):
            prices.add(int(amt))
    for it in rec.get("lists", {}).get("offers", []):
        v = it.get("value") if isinstance(it, dict) else it
        offers |= {d for d in re.findall(r"\d[\d,. ]*\d|\d", str(v))}
    for it in rec.get("lists", {}).get("phones", []):
        v = it.get("value") if isinstance(it, dict) else it
        num = v.get("number") if isinstance(v, dict) else v
        if num:
            phones.add(nz.digits(str(num))[-10:])
    return {"prices": prices, "offers": {nz.digits(o) for o in offers if len(nz.digits(o)) >= 2}, "phones": phones}


def record_text_lines(rec: dict, creative_only: bool = False) -> list[str]:
    out = []
    fields = ("headline", "sub_headline", "description") if creative_only else         ("headline", "sub_headline", "description", "call_to_action", "warranty")
    for f in fields:
        v = _v(rec, f)
        if v:
            out.append(str(v))
    for f in (("features",) if creative_only else ("features", "offers")):
        for it in rec.get("lists", {}).get(f, []):
            v = it.get("value") if isinstance(it, dict) else it
            out.append(str(v))
    return out


def signature(ad_id: str, rec: dict, image: np.ndarray | str | Path | None = None,
              ocr_lines: list[str] | None = None, meta: dict | None = None) -> Signature:
    if isinstance(image, (str, Path)):
        image = cv2.imdecode(np.fromfile(str(image), np.uint8), cv2.IMREAD_COLOR)
    lines = ocr_lines if ocr_lines else record_text_lines(rec)
    return Signature(ad_id=ad_id, phash=phash(image) if image is not None else None,
                     text=creative_text(lines), creative=creative_text(record_text_lines(rec, True)),
                     numbers=numbers_of(rec),
                     brand=nz.key(str(_v(rec, "brand") or "")),
                     brand_words=tuple(nz.key(w) for w in str(_v(rec, "brand") or "").split() if nz.key(w)),
                     product=nz.key(str(_v(rec, "product") or "")),
                     headline=nz.key(str(_v(rec, "headline") or "")), meta=meta or {})


def _same_brand(a: Signature, b: Signature) -> bool | None:
    if not a.brand or not b.brand:
        return None
    # whole words only: "Hero" ~ "Hero MotoCorp", but "Kia" is NOT "Nokia"
    if a.brand == b.brand:
        return True
    x, y = a.brand_words, b.brand_words
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    return bool(short) and long_[:len(short)] == short


def _numbers_changed(a: Signature, b: Signature) -> list[str]:
    """A change needs the number on BOTH ads and neither set contained in the other
    (a value missing on one side is an extraction gap, e.g. a blurry copy - not a change)."""
    out = []
    for k in ("prices", "offers"):
        x, y = a.numbers[k], b.numbers[k]
        if x and y and not (x <= y or y <= x):
            out.append(f"{k}: {sorted(a.numbers[k])} -> {sorted(b.numbers[k])}")
    return out


def compare(a: Signature, b: Signature) -> dict | None:
    """Relation between two ads, or None if unrelated. Includes the evidence used."""
    brand = _same_brand(a, b)
    if brand is False:
        return None                                     # different brands never link
    img = hamming(a.phash, b.phash)
    txt = fuzz.token_set_ratio(a.text, b.text) if a.text and b.text else 0
    txt_strict = fuzz.ratio(a.text, b.text) if a.text and b.text else 0
    ev = {"image_distance": img, "text_similarity": round(txt), "text_order_similarity": round(txt_strict)}
    same_img = img is not None and img <= IMG_SAME
    similar_img = img is not None and img <= IMG_SIMILAR
    # same creative: near-identical image with some text agreement, or similar layout + same text
    same_creative = (same_img and (txt >= TEXT_VERSION or not a.text or not b.text)) or \
                    (similar_img and txt >= TEXT_SAME) or \
                    (brand and txt >= TEXT_SAME and txt_strict >= TEXT_SAME - 10)
    if same_creative:
        changed = _numbers_changed(a, b)
        if changed:
            return {"relation": OFFER_CHANGED, "changes": changed, **ev}
        return {"relation": DUPLICATE, **ev}
    if brand:
        # a new version must share the CREATIVE (headline / copy), not just the brand, product or the
        # dealer list - two different campaigns for the same car are not versions of one creative
        same_product = bool(a.product and b.product and
                            (a.product == b.product or fuzz.ratio(a.product, b.product) >= 90))
        hl = round(fuzz.ratio(a.headline, b.headline)) if a.headline and b.headline else 0
        cr = round(fuzz.token_sort_ratio(a.creative, b.creative)) if a.creative and b.creative else 0
        ev["creative_similarity"] = cr
        if hl >= HEADLINE_VERSION or cr >= CREATIVE_VERSION or (same_product and cr >= CREATIVE_VERSION - 20):
            why = [w for w, ok in ((f"headline {hl}%", hl >= HEADLINE_VERSION), (f"copy {cr}%", cr >= CREATIVE_VERSION - 20),
                                   ("same product", same_product)) if ok]
            return {"relation": NEW_VERSION, "why": why, **ev}
        if same_product:
            return {"relation": SAME_PRODUCT, "why": ["same brand + product, different creative"], **ev}
    return None


def group(sigs: list[Signature]) -> dict:
    """All pairwise relations + campaigns (connected components) + unique creatives."""
    n = len(sigs)
    parent_c, parent_u = list(range(n)), list(range(n))

    def find(p, i):
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    # blocking: only ads that could share a brand are compared (different brands never link), so
    # thousands of ads stay fast. Ads without a brand are compared with everything.
    blocks: dict[str, list[int]] = {}
    unbranded = []
    for i, sg in enumerate(sigs):
        if sg.brand_words:
            blocks.setdefault(sg.brand_words[0], []).append(i)
        else:
            unbranded.append(i)
    cand = set()
    for idx in blocks.values():
        cand |= {(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]}
    cand |= {(min(i, j), max(i, j)) for i in unbranded for j in range(n) if i != j}
    pairs = []
    for i, j in sorted(cand):
        r = compare(sigs[i], sigs[j])
        if not r:
            continue
        pairs.append({"a": sigs[i].ad_id, "b": sigs[j].ad_id, **r})
        if r["relation"] not in LINKING:
            continue
        parent_c[find(parent_c, i)] = find(parent_c, j)
        if r["relation"] == DUPLICATE:
            parent_u[find(parent_u, i)] = find(parent_u, j)
    campaigns: dict[int, list[int]] = {}
    for i in range(n):
        campaigns.setdefault(find(parent_c, i), []).append(i)
    creatives = {find(parent_u, i) for i in range(n)}
    camp_list = []
    for members in campaigns.values():
        ids = [sigs[i].ad_id for i in members]
        brand = next((sigs[i].brand for i in members if sigs[i].brand), "")
        rels = [p for p in pairs if p["a"] in ids and p["b"] in ids and p["relation"] in LINKING]
        camp_list.append({"ads": ids, "brand": brand,
                          "product": next((sigs[i].product for i in members if sigs[i].product), ""),
                          "insertions": len(ids),
                          "unique_creatives": len({find(parent_u, i) for i in members}),
                          "relations": sorted({p["relation"] for p in rels})})
    camp_list.sort(key=lambda c: -c["insertions"])
    return {"ads": n, "unique_creatives": len(creatives), "campaigns": len(camp_list),
            "pairs": pairs, "campaign_list": camp_list}
