"""Accuracy pass v3: field-specific EVIDENCE GATES applied after validation + second opinion.

Why: on the bench200 error audit every wrong-and-confirmed value was confirmed by "second model agrees".
The two models share blind spots (Marathi read as Hindi, admissions read as recruitment, nearest
sub-category picked when the master list lacks the right one). Agreement between two models is therefore
not evidence on its own for judgement fields; these gates require evidence from the AD ITSELF (OCR text,
the model's own description of the product) before such a field may stay confirmed.

Each gate only DEMOTES to review (never confirms), except two value corrections that are backed by
printed evidence and always left in review:
  * Hindi -> Marathi when the OCR text contains Marathi function words and no Hindi ones;
  * an amount labelled as a benefit / saving ("₹50,000 तक के फायदे") is not used as the price.

Switch off with ADREADER_RULES_V3=0 (gives the frozen bench200 behaviour exactly).
"""
from __future__ import annotations

import os
import re

from . import normalize as nz
from .config import CONFIRM_THRESHOLD
from .ocr import SCRIPT_RANGES, OCRResult
from .schema import AdRecord, FieldResult, FieldStatus


def enabled() -> bool:
    return os.environ.get("ADREADER_RULES_V3", "1") != "0"


def _demote(r: FieldResult, cap: float, reason: str) -> None:
    r.confidence = min(r.confidence, cap)
    r.reasons.append(reason)
    if r.status == FieldStatus.confirmed and r.confidence < CONFIRM_THRESHOLD:
        r.status = FieldStatus.review


def _ocr_text(ocr: OCRResult | None) -> list[str]:
    return [l.norm for l in ocr.lines] if ocr else []


# ============================================================================ 1. language
# Common function words. Hindi and Marathi share the Devanagari script, so the SCRIPT says nothing about
# which of the two it is; these words do. (Calibrated on set50 + holdout20 + bench200: 15/18 Marathi ads
# contain >= 1 Marathi word; 1/90 Hindi ads does, and that one has 24 Hindi words.)
MARATHI_WORDS = {"आहे", "आहेत", "आणि", "मध्ये", "करा", "पासून", "साठी", "आपल्या", "आमच्या", "तुमच्या",
                 "नाही", "होते", "झाले", "मिळवा", "जिंका", "व्हा", "येथे", "सुरू", "आपला", "आपली", "केले",
                 "असलेल्या"}
HINDI_WORDS = {"है", "हैं", "और", "के", "की", "में", "से", "का", "लिए", "करें", "पर", "यह", "को", "भी",
               "अपने", "आपके", "हम", "नहीं", "होगा", "गया", "तक", "एवं", "साथ"}
_DEV_WORD = re.compile(r"[ऀ-ॿ]+")


def devanagari_lexicon(ocr: OCRResult | None) -> tuple[int, int, int, list[str]]:
    """(devanagari tokens, Marathi-word hits, Hindi-word hits, Marathi words seen)"""
    toks = [t for line in _ocr_text(ocr) for t in _DEV_WORD.findall(line)]
    mr = [t for t in toks if t in MARATHI_WORDS]
    return len(toks), len(mr), sum(t in HINDI_WORDS for t in toks), sorted(set(mr))


def language_rules(rec: AdRecord, ocr: OCRResult | None) -> None:
    r = rec.fields.get("language")
    if r is None or not isinstance(r.value, list) or not r.value:
        return
    low = [str(x).strip().lower() for x in r.value]

    # (a) a language whose script OCR CAN read, but OCR found none of it, is never confirmed - even if
    #     the second model also lists it (both models add 'Hindi' to English ads)
    for lg in low:
        if any(x.lower().startswith(f"{lg}: ocr found no") for x in r.reasons):
            _demote(r, 0.75, f"{lg.title()} listed but OCR found no text in its script - "
                             "second-model agreement is not enough")
            break

    if not ocr:
        return
    n_dev, mr, hi, mr_words = devanagari_lexicon(ocr)
    # (b) Marathi words, no (or fewer) Hindi words -> Marathi, not Hindi
    if "hindi" in low and "marathi" not in low and mr >= 1 and mr > hi:
        if hi == 0:
            r.value = ["Marathi" if x.strip().lower() == "hindi" else x for x in r.value]
            act = "Hindi replaced by Marathi"
        else:
            r.value = list(r.value) + ["Marathi"]
            act = "Marathi added"
        _demote(r, 0.6, f"{act}: OCR text has Marathi words ({', '.join(mr_words[:4])}) and "
                        f"{'no' if hi == 0 else 'fewer'} Hindi words - Hindi and Marathi share the "
                        "Devanagari script - confirm")
    # (c) reverse: Marathi claimed, clearly Hindi wording, no Marathi words
    elif "marathi" in low and "hindi" not in low and hi >= 3 and mr == 0:
        _demote(r, 0.6, f"OCR text has {hi} common Hindi words and no Marathi words - may be Hindi")
    # (d) lots of Devanagari text but not one common Hindi word: not evidence for Hindi (Marathi without
    #     our marker words, or another script such as Gurmukhi misread by the Devanagari recogniser)
    #     Real Hindi text: >= 2.9% of words are these function words (lowest seen; most ads > 7%).
    elif "hindi" in low and n_dev >= 15 and hi / n_dev < 0.02:
        _demote(r, 0.6, f"{n_dev} Devanagari words read but {'none' if hi == 0 else 'almost none'} are common "
                        "Hindi words - could be Marathi / another language, or OCR misreading another script "
                        "(Gurmukhi, Bengali, Gujarati)")


# ============================================================================ 2. price role
BENEFIT = re.compile(r"(?<!\w)(benefits?|savings?|save|discount|cashback|cash back|bonus|worth|upto|up to|"
                     r"off|exchange)(?!\w)|फायदे|फ़ायदे|फायदा|फ़ायदा|लाभ|बचत|छूट|कैशबैक|सवलत|सूट|"
                     r"(?<![ऀ-ॿ])तक(?![ऀ-ॿ])", re.I)


def price_rules(rec: AdRecord, ocr: OCRResult | None) -> None:
    prices = rec.lists.get("prices") or []
    moved = []
    for p in list(prices):
        v = p.value or {}
        if v.get("kind") in ("price", "offer_price") and BENEFIT.search(f"{v.get('label') or ''} {v.get('item') or ''}"):
            prices.remove(p)
            v["kind"] = "discount"
            _demote(p, 0.6, "labelled as a benefit / saving / discount, not a selling price")
            moved.append(p)
    if moved:
        rec.lists.setdefault("offers", []).extend(moved)
        rec.lists["prices"] = prices
        main = next((x for x in prices if x.value["kind"] == "offer_price"), None) or \
            next((x for x in prices if x.value["kind"] == "price"), None)
        old = rec.fields.get("price")
        if main:
            nf = main.model_copy(deep=True)
            nf.value = main.value["amount"]
        else:
            nf = FieldResult(reasons=["no price printed"])
        nf.reasons.append(f"amount {', '.join(str(m.value.get('as_printed')) for m in moved)} is a benefit / "
                          "saving, not the price")
        if old is not None and old.value is not None and nf.value != old.value and main is None:
            nf.reasons.append(f"model's reading was {old.value}")
        rec.fields["price"] = nf

    # amount printed right next to benefit / discount wording (the model may have dropped that label)
    pr = rec.fields.get("price")
    if pr is None or pr.value is None or pr.status != FieldStatus.confirmed or not ocr:
        return
    d = nz.digits(str(pr.value))
    lines = _ocr_text(ocr)
    for i, line in enumerate(lines):
        if d and d in nz.digits(line):
            near = " ".join(lines[max(0, i - 1): i + 2])
            m = BENEFIT.search(near)
            if m:
                _demote(pr, 0.7, f"amount is printed next to '{m.group(0)}' - confirm it is the selling price, "
                                 "not a benefit / discount")
                return


# ============================================================================ 3. ad type
ADMISSION = re.compile(r"admissions?|enrol(l)?ment|courses offered|programmes offered|programs offered|"
                       r"दाखिला|प्रवेश", re.I)
# top-level categories whose advertisers sell a service - nobody "manufactures" a flight or a degree
# (Banking / Insurance / Food Delivery excluded: their labels treat the issuer of a policy, card or menu as
#  the 'manufacturer' of that product, so the type is not contradictory there.)
SERVICE_TOPS = {"travel & hospitality", "education", "telecom & internet", "real estate",
                "government & public notices", "recruitment", "classifieds"}


def ad_type_rules(rec: AdRecord, ocr: OCRResult | None) -> None:
    at = rec.fields.get("ad_type")
    if at is None or not at.value:
        return
    text = " ".join(_ocr_text(ocr)) + " " + " ".join(
        str(rec.fields[k].value or "") + " " + str(rec.fields[k].english or "")
        for k in ("headline", "description", "call_to_action") if k in rec.fields)
    if at.value == "recruitment":
        m = ADMISSION.search(text)
        if m:
            _demote(at, 0.6, f"ad mentions '{m.group(0)}' - an admissions / course ad is not job recruitment "
                             "(suggestion: service_provider)")
    cat = rec.fields.get("category")
    top = str(cat.value).split(">")[0].strip().lower() if cat is not None and cat.value else ""
    if at.value == "manufacturer" and top in SERVICE_TOPS:
        _demote(at, 0.65, f"'manufacturer' does not fit a {top.title()} ad - service provider / dealer?")
    if at.value == "recruitment" and top.startswith("education"):
        _demote(at, 0.6, "category is Education but ad type is recruitment - confirm")


# ============================================================================ 4. sub-category evidence
# A sub-category stays confirmed only if the ad's own words name something it covers. When the master
# list has no fitting sub-category (footwear, cement, bus travel, office copiers, supplements), both models
# pick the NEAREST one; this gate sends those to review instead of confirming them.
SUBCAT_WORDS = {
    "Automobile > Cars": r"\bcars?\b|suv|sedan|hatchback|\bmpv\b|engine|mileage|kmpl|petrol|diesel|\bcng\b|"
                         r"airbags?|ex-showroom|variant|transmission|sunroof|test drive|turbo|\bev\b",
    "Automobile > Two Wheelers": r"bikes?|motorcycles?|scooters?|two[- ]wheelers?|moped|\bcc\b|mileage|\bev\b",
    "Automobile > Commercial Vehicles": r"trucks?|tipper|pick-?up|tractors?|commercial vehicles?|cargo|\blcv\b|"
                                        r"\bhcv\b|three[- ]wheelers?|rickshaw|payload|fleet|tonn?e",
    "Automobile > Tyres & Parts": r"tyres?|tires?|batter(y|ies)|lubricants?|engine oil|spares?|\bparts?\b|"
                                  r"fasteners?|components?|brakes?",
    "Education > Coaching & Test Prep": r"coaching|test prep|neet|jee|upsc|\bias\b|\bssc\b|\bcat\b|\bgate\b|"
                                        r"classes|tuition|academy|exams?|batch",
    "Education > Schools": r"schools?|cbse|icse|nursery|kindergarten|class (xi|xii|11|12)|k-?12|pre-?primary",
    "Education > Universities & Colleges": r"universit|colleges?|admissions?|degrees?|\bmba\b|\bbba\b|b\.?tech|"
                                           r"mbbs|campus|programmes?|programs?|courses?|vidyapeeth|institut",
    "FMCG > Food & Beverages": r"food|snacks?|biscuits?|cookies?|\btea\b|coffee|milk|dairy|cheese|butter|ghee|"
                               r"\boil\b|atta|flour|rice|spices?|masala|drinks?|juice|beverage|chocolate|sweets?|"
                               r"namkeen|noodles|sauce|ketchup|pickle|water|salt|sugar|ice ?cream|bread|cake|"
                               r"toast|candy|soya|protein|mithai|churna?|taste|tasty|delicious",
    "FMCG > Personal & Home Care": r"soaps?|shampoo|hair|skin|creams?|lotion|toothpaste|teeth|gums?|mouth|"
                                  r"detergent|wash|clean|deodorant|perfume|fragrance|cosmetic|beauty|face|"
                                  r"incense|agarbatti|mosquito|repellent|sanitary|diaper|razor|henna|mehndi|"
                                  r"\bdye\b|talc|powder|balm|\boil\b|ayurvedic|herbal|laxative|stomach|stamina",
    "Retail & Consumer Electronics > Home Appliances": r"appliances?|refrigerator|fridge|washing machine|"
                                                      r"air ?condition|\bac\b|television|\btv\b|\bfans?\b|"
                                                      r"coolers?|mixer|grinder|microwave|oven|purifier|geyser|"
                                                      r"heater|chimney|stove|cooktop|\biron\b|kitchen|cooker|"
                                                      r"inverter|\bled\b",
    "Retail & Consumer Electronics > Mobiles & Computers": r"mobiles?|phones?|smartphones?|laptops?|computers?|"
                                                          r"tablets?|printers?|copiers?|\bpc\b|\b5g\b|software",
    "Fashion & Apparel > Watches & Accessories": r"watch(es)?|bags?|backpacks?|wallets?|belts?|sunglass|"
                                                 r"eyewear|accessor|handbags?|clocks?",
}


def subcategory_rules(rec: AdRecord, ocr: OCRResult | None) -> None:
    c = rec.fields.get("category")
    if c is None or c.status != FieldStatus.confirmed or str(c.value) not in SUBCAT_WORDS:
        return
    f = rec.fields
    parts = [f[k].value for k in ("brand", "product", "model_variant", "ad_message", "body_copy") if k in f]
    parts += [f[k].english for k in ("product", "headline", "sub_headline", "description") if k in f]
    parts += _ocr_text(ocr)
    text = " ".join(" ".join(map(str, p)) if isinstance(p, list) else str(p) for p in parts if p).lower()
    if re.search(SUBCAT_WORDS[str(c.value)], text):
        return
    # a brand filed under exactly this category in the master data is evidence too
    b = f.get("brand")
    if b is not None and b.value:
        from .matcher import default_matcher
        m = default_matcher()
        res = m.brand(str(b.value))
        cat_ids = {e.id for e in m.tables["categories"] if nz.key(e.name) == nz.key(str(c.value))}
        if res.matched is not None and res.matched.extra.get("category_id") in cat_ids:
            return
    _demote(c, 0.8, "nothing in the ad names a product this sub-category covers - the master list may lack the "
                    "right sub-category; confirm (or use the top level)")


# ============================================================================ 5. mixed scripts in a word
_INDIC = [(n, lo, hi) for n, (lo, hi) in SCRIPT_RANGES.items()]


def _scripts_in(word: str) -> set[str]:
    out = set()
    for ch in word:
        if ch.isalpha():
            cp = ord(ch)
            for n, lo, hi in _INDIC:
                if lo <= cp <= hi:
                    out.add(n)
                    break
    return out


def mixed_script_rules(rec: AdRecord) -> None:
    for k in ("brand", "product", "headline", "sub_headline", "description", "call_to_action", "advertiser"):
        r = rec.fields.get(k)
        if r is None or not isinstance(r.value, str):
            continue
        bad = [w for w in r.value.split() if len(_scripts_in(w)) >= 2]
        if bad:
            _demote(r, 0.6, f"letters of two different Indian scripts inside one word ('{bad[0]}') - "
                            "likely a misread character")


def apply(rec: AdRecord, ocr: OCRResult | None) -> None:
    if not enabled():
        return
    language_rules(rec, ocr)
    price_rules(rec, ocr)
    ad_type_rules(rec, ocr)
    subcategory_rules(rec, ocr)
    mixed_script_rules(rec)
