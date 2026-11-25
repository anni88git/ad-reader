"""Write bench200 draft labels from batch JSON files (drafted by Claude from the images only,
BEFORE looking at any system output). Never overwrites an existing label (human-verified or not).

Batch file: a JSON list of {"id": "d001", "language": [...], "brand": ..., "product": ...,
"ad_type": ..., "category": ..., "headline": ..., "price": ..., "qr_code_present": ...,
"uncertain": ["field", ...], "note": "..."}. Values follow data/labels/FORMAT.md (lists of
alternatives are written as {"any_of": [...]}).

  python data/labels/_make_bench200_labels.py <batch.json> [...]
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LABELS = ROOT / "data/labels"
FIELDS = ["language", "brand", "product", "ad_type", "category", "headline", "price", "qr_code_present"]
OPTIONAL = ["model_variant", "sub_headline", "phones", "websites", "emails"]   # labelled when legible
SKIP = ["model_variant", "advertiser", "sub_headline", "description", "body_copy", "ad_message", "features",
        "offers", "warranty", "disclaimer_present", "dealers", "addresses", "social_handles", "prices",
        "call_to_action", "phones", "websites", "emails", "reference_numbers"]


def main(paths):
    man = {r["id"]: r for r in csv.DictReader((ROOT / "data/eval/bench200/manifest.csv").open(encoding="utf-8"))}
    n = 0
    for p in paths:
        for d in json.loads(Path(p).read_text(encoding="utf-8")):
            r = man[d["id"]]
            out = LABELS / f"{Path(r['file']).stem}.json"
            if out.exists():
                print("  exists, not overwritten:", out.name)
                continue
            lab = {"_label_source": "claude_independent_read",
                   "_note": "Drafted by Claude from the ad image only, before any system output was seen. "
                            "PROVISIONAL - not human-verified. Core fields only. Format: data/labels/FORMAT.md",
                   "_verified_by": None, "_set": "bench200",
                   "_skip": [f for f in SKIP if f not in d]}
            for f in FIELDS + OPTIONAL:
                if f not in d:
                    continue
                v = d[f]
                if f in ("phones", "websites", "emails"):
                    lab[f] = v                       # lists: every item required
                else:
                    lab[f] = v if f == "language" and isinstance(v, list) and all(isinstance(x, str) for x in v) \
                        else ({"any_of": v} if isinstance(v, list) else v)
            if isinstance(d.get("language_any"), list):
                lab["language"] = {"any_of": d["language_any"]}
            lab["_uncertain_fields"] = [{"field": f, "why": "drafter unsure"} for f in d.get("uncertain", [])]
            if d.get("note"):
                lab["_drafter_note"] = d["note"]
            out.write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
            n += 1
    print(n, "labels written")


if __name__ == "__main__":
    main(sys.argv[1:])
