"""Automated adversarial benchmark: plant realistic errors into EVERY ad's real extraction and
count how many come out CONFIRMED (= silent errors). Not hand-tuned per example.

  python scripts/adversarial_bench.py [--set set50] [--verifier-tag nano]

Mutations (each applied separately to a fresh copy of the ad's extraction):
  wrong_line_headline  headline := another real, readable OCR line of the same ad
  spelling_change      one character / vowel sign of the headline changed
  price_digit          one digit of the main price changed
  phone_digit          one digit of a phone number changed
  regno_digit          one digit of a registration/plan number changed (or a fake one added)
  fake_website         a website that is not printed is added
"""
import argparse
import csv
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import normalize as nz  # noqa: E402
from adreader.schema import Ev, FieldStatus, RawExtraction  # noqa: E402
from adreader.validate import validate  # noqa: E402

RNG = random.Random(13)
MATRA_SWAP = {"ा": "ी", "ी": "ि", "ि": "ी", "ु": "ू", "ू": "ु", "े": "ै", "ै": "े", "ो": "ौ", "ं": "ँ",
              "ொ": "ோ", "ோ": "ொ", "ு": "ூ", "ூ": "ு", "ి": "ీ", "ీ": "ి", "ு": "ூ", "ಿ": "ೀ"}


def change_digit(s: str) -> str | None:
    idx = [i for i, ch in enumerate(s) if ch.isdigit()]
    if not idx:
        return None
    i = RNG.choice(idx)
    return s[:i] + str((int(s[i]) + RNG.randint(1, 8)) % 10) + s[i + 1:]


def change_spelling(s: str) -> str | None:
    swaps = [i for i, ch in enumerate(s) if ch in MATRA_SWAP]
    if swaps:
        i = RNG.choice(swaps)
        return s[:i] + MATRA_SWAP[s[i]] + s[i + 1:]
    letters = [i for i, ch in enumerate(s) if ch.isascii() and ch.isalpha()]
    if len(letters) < 4:
        return None
    i = RNG.choice(letters[1:])
    repl = RNG.choice([c for c in "aeiourstnl" if c != s[i].lower()])
    return s[:i] + (repl.upper() if s[i].isupper() else repl) + s[i + 1:]


def mutations(raw: RawExtraction, ocr):
    """yield (name, mutated_raw, selector) - selector(rec) -> FieldResult or None"""
    h = raw.headline
    if h.value and h.evidence:
        hk = nz.key(h.evidence)
        others = [l.norm for l in ocr.lines if l.score >= 0.9 and len(l.norm) >= 10 and len(l.norm.split()) >= 2
                  and nz.key(l.norm) not in hk and hk not in nz.key(l.norm)]
        if others:
            m = raw.model_copy(deep=True)
            line = RNG.choice(others)
            m.headline = Ev(value=line, evidence=line)
            yield "wrong_line_headline", m, lambda rec: rec.fields.get("headline")
        sp = change_spelling(str(h.value))
        if sp and nz.key(sp) != nz.key(str(h.value)):
            m = raw.model_copy(deep=True)
            m.headline = Ev(value=sp, evidence=sp)
            yield "spelling_change", m, lambda rec: rec.fields.get("headline")
    if raw.prices:
        m = raw.model_copy(deep=True)
        p0 = m.prices[0]
        nd = change_digit(p0.amount_text or "")
        if nd:
            p0.amount_text = nd
            target = nz.parse_amount(nd)[0]
            yield "price_digit", m, lambda rec, t=target: next(
                (x for x in rec.lists.get("prices", []) if x.value.get("amount") == t), None)
    if raw.phones:
        m = raw.model_copy(deep=True)
        nd = change_digit(str(m.phones[0].value))
        if nd:
            m.phones[0] = Ev(value=nd, evidence=(m.phones[0].evidence or "").replace(str(raw.phones[0].value), nd) or nd)
            num = nz.parse_phone(nd).number
            m.dealers = [d for d in m.dealers]       # dealer phones untouched
            yield "phone_digit", m, lambda rec, n=num: next(
                (x for x in rec.lists.get("phones", []) if x.value.get("number") == n), None)
    m = raw.model_copy(deep=True)
    if raw.reference_numbers:
        nd = change_digit(str(m.reference_numbers[0].value))
        if nd:
            m.reference_numbers[0] = Ev(value=nd, evidence=nd)
            yield "regno_digit", m, lambda rec, v=nd: next(
                (x for x in rec.lists.get("reference_numbers", []) if nz.key(str(x.value)) == nz.key(v)), None)
    else:
        fake = f"RERA No. P5180{RNG.randint(1000000, 9999999)}"
        m.reference_numbers = [Ev(value=fake, evidence=fake)]
        yield "regno_digit", m, lambda rec, v=fake: next(
            (x for x in rec.lists.get("reference_numbers", []) if nz.key(str(x.value)) == nz.key(v)), None)
    m = raw.model_copy(deep=True)
    fake_web = f"www.{nz.key(str(raw.brand.value or 'brand'))[:12] or 'brand'}offers.in"
    m.websites = list(m.websites) + [Ev(value=fake_web, evidence=fake_web)]
    yield "fake_website", m, lambda rec, w=fake_web: next(
        (x for x in rec.lists.get("websites", []) if fake_web.replace("www.", "") in str(x.value)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="set50")
    ap.add_argument("--system", default="nemotron")
    ap.add_argument("--verifier-tag", default=None)
    a = ap.parse_args()
    set_dir = ROOT / "data/eval" / a.set
    rows = list(csv.DictReader((set_dir / "manifest.csv").open(encoding="utf-8")))
    stats = defaultdict(lambda: {"planted": 0, "leaked": 0, "absent": 0})
    leaks = []
    for r in rows:
        rf = set_dir / "runs" / a.system / "raw" / f"{r['id']}.json"
        if not rf.exists():
            continue
        d = json.loads(rf.read_text(encoding="utf-8"))
        if not d.get("raw"):
            continue
        raw = RawExtraction.model_validate(d["raw"])
        ocr = pickle.loads((set_dir / "ocr" / f"{r['id']}.pkl").read_bytes())
        ver = None
        if a.verifier_tag:
            vf = set_dir / "verify" / a.verifier_tag / f"{r['id']}.json"
            if vf.exists():
                ver = RawExtraction.model_validate(json.loads(vf.read_text(encoding="utf-8"))["raw"])
        for name, mraw, sel in mutations(raw, ocr):
            rec = validate(mraw, ocr, image=None, image_name=r["id"], verifier=ver)
            fr = sel(rec)
            stats[name]["planted"] += 1
            if fr is None:
                stats[name]["absent"] += 1          # dropped as invalid / deduplicated = not filled
            elif fr.status == FieldStatus.confirmed:
                stats[name]["leaked"] += 1
                leaks.append((r["id"], name, str(fr.value)[:70], "; ".join(fr.reasons)[:120]))
    total_p = sum(s["planted"] for s in stats.values())
    total_l = sum(s["leaked"] for s in stats.values())
    print(f"{'mutation':22} planted  leaked  leak%")
    for k, s in stats.items():
        print(f"{k:22} {s['planted']:7} {s['leaked']:7}  {100 * s['leaked'] / max(1, s['planted']):5.1f}")
    print(f"{'TOTAL':22} {total_p:7} {total_l:7}  {100 * total_l / max(1, total_p):5.1f}")
    for x in leaks[:30]:
        print("  LEAK", x)
    out = set_dir / "report"
    out.mkdir(exist_ok=True)
    tag = a.system + (f"+{a.verifier_tag}" if a.verifier_tag else "")
    (out / f"adversarial_{tag}.json").write_text(json.dumps({"stats": stats, "leaks": leaks}, ensure_ascii=False,
                                                            indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
