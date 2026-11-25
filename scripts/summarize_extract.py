"""Compact summary of saved extraction outputs:  python scripts/summarize_extract.py [glob]"""
import glob
import json
import sys
from pathlib import Path

pattern = sys.argv[1] if len(sys.argv) > 1 else "data/outputs/extract/*.json"
for f in sorted(glob.glob(pattern)):
    d = json.loads(Path(f).read_text(encoding="utf-8"))
    r = d["raw"]
    print(f"===== {Path(f).name}  {d['seconds']}s")
    for k in ["brand", "product", "model_variant", "advertiser", "ad_type", "category", "headline", "sub_headline"]:
        v = r[k]
        en = f"  en={v['english']!r}" if v.get("english") else ""
        print(f"  {k:13} {str(v['value'])!r:55} ev={str(v['evidence'])[:70]!r}{en}")
    print("  prices      ", [(p["amount_text"], p["label"]) for p in r["prices"]])
    print("  phones      ", [p["value"] for p in r["phones"]], " web", [w["value"] for w in r["websites"]])
    print("  dealers     ", [(x["name"], x["city"], x["phones"]) for x in r["dealers"]])
    print("  offers      ", [o["value"] for o in r["offers"]])
    print("  features    ", len(r["features"]), [x["value"] for x in r["features"]][:6])
    print("  attrs       ", {k: v["value"] for k, v in r["category_attributes"].items()})
    print("  language    ", r["language"], " notes", r["uncertain_notes"])
