"""Run stage-4 validation on saved extraction outputs and print a status table.

  python scripts/validate_demo.py data/outputs/extract/_kia_from_screenshot.*.json [...]
Writes data/outputs/records/<image-stem>.json
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adreader import ocr as ocr_mod, preprocess as pp  # noqa: E402
from adreader.config import OUTPUTS_DIR  # noqa: E402
from adreader.schema import AdRecord, FieldStatus, RawExtraction  # noqa: E402
from adreader.validate import validate  # noqa: E402

ICON = {FieldStatus.confirmed: "OK ", FieldStatus.review: "REV", FieldStatus.missing: "---"}


def load_ocr(img_path: Path, img):
    langs = "-".join(ocr_mod.DEFAULT_LANGS)
    cache = OUTPUTS_DIR / "ocr_cache" / f"{img_path.stem}.{langs}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    return ocr_mod.run_ocr(img)


def show(rec: AdRecord) -> None:
    def short(v):
        s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        return s if len(s) <= 60 else s[:57] + "..."
    for name, r in rec.all_results():
        if r.value in (None, "", []) and r.status == FieldStatus.missing:
            continue
        print(f"  {ICON[r.status]} {r.confidence:4.2f}  {name:22} {short(r.value):62} | {'; '.join(r.reasons)[:110]}")
    print(f"  summary={rec.summary()}  quality={rec.quality_warnings}  flags={rec.record_flags}")


if __name__ == "__main__":
    out_dir = OUTPUTS_DIR / "records"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in map(Path, sys.argv[1:]):
        d = json.loads(f.read_text(encoding="utf-8"))
        img_path = Path(d["image"])
        img = pp.load_image(img_path)
        raw = RawExtraction.model_validate(d["raw"])
        rec = validate(raw, load_ocr(img_path, img), image=img, image_name=img_path.name)
        rec.models_used = [d["model"]]
        (out_dir / f"{img_path.stem}.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n===== {img_path.name}")
        show(rec)
