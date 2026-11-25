"""Offline re-validation: re-run ONLY the validation layer on stored outputs (OCR pickle + raw model
extraction + stored verifier reply). No API calls, no new model outputs. Writes to a NEW run folder so
baseline records are never overwritten.

  python scripts/revalidate.py --set bench200 --raw nemotron --verifier nano --out cand_v3+nano
  python scripts/revalidate.py --set set50 --raw nemotron_v2fresh --verifier nano --out cand_v3+nano
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import preprocess as pp  # noqa: E402
from adreader.matcher import link_records  # noqa: E402
from adreader.pipeline import ocr_corpus_key  # noqa: E402
from adreader.schema import RawExtraction  # noqa: E402
from adreader.validate import validate  # noqa: E402

PROTECTED = {"nemotron+nano", "nemotron_v2fresh+nano", "nemotron_p25+nano", "nemotron"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--raw", default="nemotron", help="run folder holding raw/ model extractions")
    ap.add_argument("--verifier", default="nano")
    ap.add_argument("--out", required=True, help="NEW run folder name for the records")
    a = ap.parse_args()
    if a.out in PROTECTED:
        sys.exit(f"refusing to write into protected baseline run '{a.out}'")
    d = ROOT / "data/eval" / a.set
    out = d / "runs" / a.out / "records"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in csv.DictReader((d / "manifest.csv").open(encoding="utf-8")):
        rf = d / "runs" / a.raw / "raw" / f"{r['id']}.json"
        if not rf.exists():
            continue
        dd = json.loads(rf.read_text(encoding="utf-8"))
        if not dd.get("raw"):
            continue
        o = pickle.loads((d / "ocr" / f"{r['id']}.pkl").read_bytes())
        ver = None
        vf = d / "verify" / a.verifier / f"{r['id']}.json"
        if a.verifier and vf.exists():
            ver = RawExtraction.model_validate(json.loads(vf.read_text(encoding="utf-8"))["raw"])
        rec = validate(RawExtraction.model_validate(dd["raw"]), o, image=pp.load_image(ROOT / r["file"]),
                       image_name=Path(r["file"]).name, verifier=ver)
        link_records(rec, ocr_corpus_key(o))
        rec.models_used = [dd.get("model", a.raw)]
        rec.timings_s = {"ocr": o.timings_s["total"], "extract": dd.get("seconds") or 0}
        (out / f"{r['id']}.json").write_text(rec.model_dump_json(indent=1), encoding="utf-8")
        n += 1
    print(f"{a.set}: {n} records -> {out}")


if __name__ == "__main__":
    main()
