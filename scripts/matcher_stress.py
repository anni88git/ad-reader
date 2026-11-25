"""SIMULATED stress test of master-data linking under OCR-style misreadings.

For every brand / advertiser name (and alias) in the master DB, generate realistic OCR corruptions
(rn<->m, l/I/1, O/0, S/5, B/8, c/e, cl/d, vv/w, dropped / doubled letter, truncated last letter,
case changes, extra corporate suffix) and record what the matcher does:
  correct   - links to the right record
  no_link   - links nothing (goes to review as unknown / near-miss)  -> safe
  WRONG     - links to a DIFFERENT record                             -> the dangerous case
Not real-ad evidence. Writes data/outputs/matcher_stress.json.

  python scripts/matcher_stress.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader.config import OUTPUTS_DIR  # noqa: E402
from adreader.matcher import default_matcher  # noqa: E402

SWAPS = [("rn", "m"), ("m", "rn"), ("l", "I"), ("I", "l"), ("i", "l"), ("l", "1"), ("O", "0"), ("o", "0"),
         ("S", "5"), ("B", "8"), ("e", "c"), ("c", "e"), ("cl", "d"), ("d", "cl"), ("w", "vv"), ("u", "v"), ("h", "b")]


def corruptions(name: str, rng: random.Random) -> list[tuple[str, str]]:
    out = []
    for a, b in SWAPS:
        i = name.find(a)
        if i >= 0:
            out.append((f"swap {a}->{b}", name[:i] + b + name[i + len(a):]))
    letters = [i for i, ch in enumerate(name) if ch.isalpha()]
    if len(letters) > 3:
        i = rng.choice(letters[1:])
        out.append(("drop letter", name[:i] + name[i + 1:]))
        out.append(("double letter", name[:i] + name[i] + name[i:]))
        out.append(("truncate", name[:-1]))
    out.append(("upper", name.upper()))
    out.append(("suffix", name + " India"))
    return [(k, v) for k, v in out if v.strip() and v != name]


def main():
    rng = random.Random(11)
    m = default_matcher()
    res = {}
    for table, fn in (("brands", m.brand), ("advertisers", m.advertiser)):
        c, wrong = Counter(), []
        for e in m.tables[table]:
            for nm in e.names:
                for kind, bad in corruptions(nm, rng):
                    r = fn(bad)
                    if r.matched is None:
                        c["no_link"] += 1
                        la = m.lookalike(bad, m.tables[table])       # suggestion only (field stays in review)
                        if la is None:
                            c["no_link: no suggestion"] += 1
                        elif la.id == e.id:
                            c["no_link: correct suggestion"] += 1
                        else:
                            c["no_link: WRONG suggestion"] += 1
                            wrong.append({"true": e.name, "ocr_text": bad, "corruption": kind,
                                          "linked_to": "suggested " + la.name, "reason": "look-alike suggestion"})
                    elif r.matched.id == e.id:
                        c["correct"] += 1
                    else:
                        c["WRONG"] += 1
                        wrong.append({"true": e.name, "ocr_text": bad, "corruption": kind,
                                      "linked_to": r.matched.name, "reason": r.reason})
        res[table] = {"counts": dict(c), "wrong_links": wrong}
    p = OUTPUTS_DIR / "matcher_stress.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for t, v in res.items():
        n = sum(v["counts"].values())
        print(f"{t}: {n} corrupted names -> {v['counts']}")
        for w in v["wrong_links"][:15]:
            print(f"   WRONG: {w['ocr_text']!r} (true {w['true']}, {w['corruption']}) -> {w['linked_to']} | {w['reason'][:70]}")


if __name__ == "__main__":
    main()
