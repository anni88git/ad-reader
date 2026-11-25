"""Deterministic normalisation of prices, phone numbers, URLs, e-mails and units.
Pure functions, no AI - every rule here is testable."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .ocr import DIGIT_MAP

# ----------------------------------------------------------------------------- text keys
def to_ascii_digits(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(DIGIT_MAP)


def key(s: str | None) -> str:
    """Comparison key: NFC, ASCII digits, lowercase, only letters/marks/digits (no spaces or
    punctuation) - robust to OCR spacing and line-break differences, works for Indic scripts."""
    if not s:
        return ""
    s = to_ascii_digits(s).lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LMN")


def digits(s: str | None) -> str:
    return re.sub(r"\D", "", to_ascii_digits(s or ""))


# ----------------------------------------------------------------------------- prices
_RUPEE = r"(?:₹|rs\.?|inr|रु\.?|रू\.?|रुपए|रुपये|र(?=\s*\d))"
_MULT = {"lakh": 100_000, "lac": 100_000, "lakhs": 100_000, "lacs": 100_000, "l": 100_000,
         "लाख": 100_000, "crore": 10_000_000, "cr": 10_000_000, "crores": 10_000_000,
         "करोड़": 10_000_000, "k": 1_000, "thousand": 1_000, "हजार": 1_000, "हज़ार": 1_000,
         # regional scripts: lakh / crore / thousand
         "లక్షలు": 100_000, "లక్ష": 100_000, "லட்சம்": 100_000, "லட்ச": 100_000, "ಲಕ್ಷ": 100_000,
         "লক্ষ": 100_000, "লাখ": 100_000, "લાખ": 100_000, "ലക്ഷം": 100_000, "ਲੱਖ": 100_000,
         "కోట్లు": 10_000_000, "కోటి": 10_000_000, "கோடி": 10_000_000, "ಕೋಟಿ": 10_000_000,
         "কোটি": 10_000_000, "કરોડ": 10_000_000, "കോടി": 10_000_000, "ਕਰੋੜ": 10_000_000,
         "వేలు": 1_000, "ஆயிரம்": 1_000, "ಸಾವಿರ": 1_000, "হাজার": 1_000, "હજાર": 1_000}
_MULT_RE = "|".join(sorted((re.escape(k) for k in _MULT), key=len, reverse=True))

PRICE_KINDS = ("price", "old_price", "offer_price", "emi", "cashback", "discount", "other")
_KIND_WORDS = [
    ("cashback", ["cashback", "cash back", "कैशबैक"]),
    ("discount", ["off", "discount", "save", "savings", "saving", "benefit", "benefits", "worth", "bonus",
                  "exchange", "upto", "up to", "reward", "छूट", "बचत", "फायदा", "लाभ", "सवलत"]),
    ("emi", ["emi", "/month", "per month", "p.m", "monthly", "किस्त", "प्रति माह"]),
    ("old_price", ["struck", "crossed", "old price", "mrp", "was ", "पुराना"]),
    ("offer_price", ["offer price", "special price", "only", "मात्र", "सिर्फ", "केवल", "now"]),
]


@dataclass
class ParsedPrice:
    amount: int | None
    kind: str
    problems: list[str]


def parse_amount(text: str) -> tuple[int | None, list[str]]:
    """'₹ 27 99 000*' -> 2799000 ; '₹48/-' -> 48 ; 'Rs. 1.5 Lakh' -> 150000 ; '2.5 Cr' -> 25000000"""
    problems: list[str] = []
    t = to_ascii_digits(text).lower().replace(" ", " ")
    m = re.search(rf"(\d[\d,\s]*(?:\.\d+)?)\s*({_MULT_RE})?(?![a-z])", t)
    if not m:
        return None, ["no number found"]
    num_s, mult_s = m.group(1), m.group(2)
    if re.match(r"\s*%", t[m.end(1):]):
        return None, ["percentage, not an amount"]
    if len(re.findall(r"\d[\d,\s]*(?:\.\d+)?", t)) > 1 and not re.search(r"\d\s+\d{2,3}\s+\d{3}", t):
        problems.append("several numbers in price text")
    # Indian grouping with spaces ("27 99 000") or commas ("27,99,000")
    num_clean = re.sub(r"[,\s]", "", num_s.strip())
    try:
        val = float(num_clean)
    except ValueError:
        return None, ["unparseable number"]
    if mult_s:
        val *= _MULT.get(mult_s, 1)
    if val <= 0:
        return None, ["non-positive amount"]
    if not mult_s and "." in num_clean and val < 1000 and re.search(r"[^\x00-\x7f₹]", t[m.end(1):m.end(1) + 12]):
        problems.append("decimal amount followed by an unrecognised word - check the unit (lakh/crore?)")
    return int(round(val)), problems


def price_kind(label: str | None, amount_text: str | None) -> str:
    s = f" {(label or '').lower()} {(amount_text or '').lower()} "
    for kind, words in _KIND_WORDS:
        # whole-word match ('off' must not match 'offer'); Indic words have no \b so use spaces/punct
        if any(re.search(_word_pattern(w.strip()), s) for w in words):
            return kind
    return "price"


def _word_pattern(w: str) -> str:
    """Boundaries only where the keyword itself starts/ends with a word character, so 'off' does
    not match 'offer' while '/month' still matches '4,999/month'."""
    pre = r"(?<!\w)" if re.match(r"\w", w) else ""
    post = r"(?!\w)" if re.search(r"\w$", w) else ""
    return pre + re.escape(w) + post


def parse_price(amount_text: str, label: str | None) -> ParsedPrice:
    amount, problems = parse_amount(amount_text)
    if amount is not None and not re.search(_RUPEE, to_ascii_digits(amount_text).lower()) \
            and not re.search(_RUPEE, (label or "").lower()):
        problems.append("no currency symbol")
    return ParsedPrice(amount, price_kind(label, amount_text), problems)


# ----------------------------------------------------------------------------- phones
@dataclass
class ParsedPhone:
    number: str            # canonical digits, e.g. "9289305692", "18001085005", "02268276827"
    kind: str              # mobile / toll_free / landline / unknown
    valid: bool
    problems: list[str]


_PHONE_RUN = re.compile(r"\+?\d[\d\s\-().]{4,}\d")


def parse_phone(text: str) -> ParsedPhone:
    """Parse the phone-shaped digit run(s) in text; prefers a run that is a valid number.
    (Avoids gluing stray digits such as the '0' in 'मो0 नं0-' onto the number.)"""
    t = to_ascii_digits(text)
    runs = [m.group(0) for m in _PHONE_RUN.finditer(t)] or [t]
    parsed = [_parse_phone_digits(digits(r)) for r in runs]
    valid = [p for p in parsed if p.valid]
    if valid:
        return max(valid, key=lambda p: len(p.number))
    whole = _parse_phone_digits(digits(t))
    return whole if whole.valid else max(parsed, key=lambda p: len(p.number))


def _parse_phone_digits(d: str) -> ParsedPhone:
    problems: list[str] = []
    if d.startswith("91") and len(d) == 12:
        d = d[2:]
    elif d.startswith("0091") and len(d) == 14:
        d = d[4:]
    if re.fullmatch(r"[6-9]\d{9}", d):
        return ParsedPhone(d, "mobile", True, problems)
    if re.fullmatch(r"0[6-9]\d{9}", d):
        return ParsedPhone(d[1:], "mobile", True, problems)
    if re.fullmatch(r"1(800|860)\d{6,7}", d):
        return ParsedPhone(d, "toll_free", True, problems)
    if re.fullmatch(r"0[1-9]\d{8,9}", d):            # STD code + local number (10-11 digits)
        return ParsedPhone(d, "landline", True, problems)
    if re.fullmatch(r"[1-9]\d{8,9}", d) and len(d) in (9, 10) and not d[0] in "6789":
        return ParsedPhone("0" + d, "landline", True, [])
    if re.fullmatch(r"\d{6,8}", d):
        return ParsedPhone(d, "landline", True, ["local number without STD code"])
    problems.append(f"not a valid Indian phone number ({len(d)} digits)")
    return ParsedPhone(d, "unknown", False, problems)


# ----------------------------------------------------------------------------- fuel / EV
# order matters: more specific patterns first
_FUEL_PATTERNS = [
    ("Plug-in Hybrid", r"plug[\s-]*in\s*hybrid|\bphev\b"),
    ("Hybrid", r"hybrid|e:hev|\bhev\b|हाइब्रिड"),
    ("Electric", r"\belectric\b|\bev\b|\bbev\b|battery electric|इलेक्ट्रिक"),
    ("Diesel", r"diesel|\bcrdi\b|\btdi\b|डीजल|डीज़ल"),
    ("Petrol", r"petrol|gasoline|\bturbo[\s-]*petrol\b|पेट्रोल"),
    ("CNG", r"\bcng\b|सीएनजी"),
    ("LPG", r"\blpg\b"),
]


def parse_fuel_types(text: str | None) -> list[str]:
    """'Hybrid & Diesel Powertrains' -> ['Hybrid', 'Diesel']"""
    if not text:
        return []
    t = str(text).lower()
    out = []
    for name, pat in _FUEL_PATTERNS:
        if re.search(pat, t):
            if name == "Hybrid" and "Plug-in Hybrid" in out:
                continue
            out.append(name)
    return out


def ev_type(fuels: list[str]) -> str | None:
    """Rule used by the brief: only fully electric vehicles are 'EV'; hybrids (incl. plug-in)
    and combustion engines are 'Non-EV'. None if no fuel type is known."""
    if not fuels:
        return None
    return "EV" if set(fuels) == {"Electric"} else "Non-EV"


# ----------------------------------------------------------------------------- web / email
_DOMAIN = re.compile(r"((?:[a-z0-9-]+\.)+[a-z]{2,24})(/[^\s]*)?", re.I)
_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,24}", re.I)


def parse_website(text: str) -> tuple[str | None, list[str]]:
    t = text.strip().lower()
    t = re.sub(r"^(visit|website|web|log on to|logon to|visit us at)\s*[:\-]?\s*", "", t)
    t = re.sub(r"^https?://", "", t).rstrip(".,;")
    m = _DOMAIN.search(t)
    if not m:
        return None, ["not a valid web address"]
    return m.group(0).rstrip("/"), []


def parse_email(text: str) -> tuple[str | None, list[str]]:
    m = _EMAIL.search(text.strip())
    return (m.group(0).lower(), []) if m else (None, ["not a valid e-mail address"])
