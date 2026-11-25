"""Run the full pipeline on images and print status tables.

  python scripts/run_pipeline.py data/ads/_kia_from_screenshot.jpg [...] [--page] [--verifier MODEL] [--cached]
--page      : treat input as a full newspaper page (find ads first)
--cached    : reuse cached OCR + saved extraction (from scripts/extract_demo.py) - no API calls
Writes data/outputs/records/<name>.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adreader import ocr as ocr_mod, preprocess as pp  # noqa: E402
from adreader.config import OUTPUTS_DIR, vlm_config  # noqa: E402
from adreader.pipeline import process_ad, process_image  # noqa: E402
from adreader.schema import RawExtraction  # noqa: E402
from validate_demo import show  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--page", action="store_true")
    ap.add_argument("--verifier", default=None)
    ap.add_argument("--cached", action="store_true")
    a = ap.parse_args()
    out_dir = OUTPUTS_DIR / "records"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = vlm_config().model.split("/")[-1]
    for p in map(Path, a.images):
        if a.cached:
            img = pp.load_image(p)
            ocr = pickle.loads((OUTPUTS_DIR / "ocr_cache" / f"{p.stem}.{'-'.join(ocr_mod.DEFAULT_LANGS)}.pkl").read_bytes())
            raw = RawExtraction.model_validate(json.loads((OUTPUTS_DIR / "extract" / f"{p.stem}.{tag}.json").read_text(encoding="utf-8"))["raw"])
            results = [process_ad(img, p.name, ocr=ocr, raw=raw, verifier_model=a.verifier)]
        else:
            results = process_image(p, find_ads_on_page=a.page, verifier_model=a.verifier)
        for res in results:
            rec = res.record
            (out_dir / f"{Path(rec.image).stem.replace('#', '_')}.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")
            print(f"\n===== {rec.image}  models={rec.models_used}  timings={rec.timings_s}")
            show(rec)
            for k in ("brand", "product", "category", "advertiser"):
                r = rec.fields.get(k)
                if r and r.value:
                    print(f"      link {k:10} id={r.matched_id}  candidates={[(c.name, c.score) for c in r.candidates]}")
