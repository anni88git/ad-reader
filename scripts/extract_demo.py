"""Run OCR + VLM extraction on images and save the raw structured output.

  python scripts/extract_demo.py data/ads/002_horlicks_hi.png [...] [--model MODEL] [--no-ocr]
Outputs data/outputs/extract/<name>.<model-tag>.json (+ OCR cached in data/outputs/ocr_cache/)
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adreader import ocr as ocr_mod, preprocess as pp  # noqa: E402
from adreader.config import OUTPUTS_DIR  # noqa: E402
from adreader.extract import extract  # noqa: E402
from adreader.vlm import VLMClient  # noqa: E402


def cached_ocr(path: Path, img, langs):
    cache = OUTPUTS_DIR / "ocr_cache" / f"{path.stem}.{'-'.join(langs)}.pkl"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    res = ocr_mod.run_ocr(img, langs=langs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--langs", default=",".join(ocr_mod.DEFAULT_LANGS))
    a = ap.parse_args()
    client = VLMClient()
    out_dir = OUTPUTS_DIR / "extract"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in map(Path, a.images):
        img = pp.load_image(p)
        o = None if a.no_ocr else cached_ocr(p, img, a.langs.split(","))
        run = extract(img, o, client=client, model=a.model)
        tag = (a.model or client.cfg.model).split("/")[-1] + ("-noocr" if a.no_ocr else "")
        out = {"image": str(p), "model": run.model, "seconds": run.seconds,
               "tokens_in": run.tokens_in, "tokens_out": run.tokens_out,
               "ocr_seconds": o.timings_s if o else None, "raw": run.raw.model_dump()}
        f = out_dir / f"{p.stem}.{tag}.json"
        f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n===== {p.name} | {run.model} | {run.seconds}s | tokens {run.tokens_in}->{run.tokens_out} | {f}")
        print(json.dumps(run.raw.model_dump(exclude_defaults=True), ensure_ascii=False, indent=1))
