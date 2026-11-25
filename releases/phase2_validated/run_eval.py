"""Batch runner for the evaluation set. Resumable: every stage caches per image.

  python scripts/run_eval.py ocr                       # OCR all images (shared by all systems)
  python scripts/run_eval.py prompts                   # export identical prompts for Kaggle models
  python scripts/run_eval.py extract --system nemotron # hosted model extraction (NVIDIA API)
  python scripts/run_eval.py ingest --system qwen3vl8b --replies path/to/replies.jsonl
  python scripts/run_eval.py validate --system nemotron   # verification + record linking

Layout: data/eval/<set>/ocr/<id>.pkl, prompts/<id>.txt,
        runs/<system>/raw/<id>.json, runs/<system>/records/<id>.json
"""
import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader import ocr as ocr_mod, preprocess as pp  # noqa: E402
from adreader.extract import SYSTEM, build_prompt, coerce, extract  # noqa: E402
from adreader.matcher import link_records  # noqa: E402
from adreader.pipeline import ocr_corpus_key  # noqa: E402
from adreader.schema import RawExtraction  # noqa: E402
from adreader.validate import validate  # noqa: E402
from adreader.verify import verify  # noqa: E402
from adreader.vlm import VLMClient, parse_json  # noqa: E402

SYSTEM_MODELS = {"nemotron": "nvidia/nemotron-3.5-super-vl-120b-a12b"}
OCR_LANGS = {"tamil": ("latin", "devanagari", "tamil"), "telugu": ("latin", "devanagari", "telugu"),
             "kannada": ("latin", "devanagari", "kannada")}


def items(set_dir: Path):
    with (set_dir / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stage_ocr(set_dir, rows):
    out = set_dir / "ocr"
    out.mkdir(exist_ok=True)
    for r in rows:
        f = out / f"{r['id']}.pkl"
        if f.exists():
            continue
        # script selection = what a user would pick for that ad (from the dataset's language label)
        langs = OCR_LANGS.get(r["language_label"].lower(), ocr_mod.DEFAULT_LANGS)
        img = pp.load_image(ROOT / r["file"])
        res = ocr_mod.run_ocr(img, langs=langs)
        f.write_bytes(pickle.dumps(res))
        log(f"ocr {r['id']} {len(res.lines)} lines {res.timings_s['total']}s langs={langs}")


def stage_prompts(set_dir, rows):
    out = set_dir / "prompts"
    out.mkdir(exist_ok=True)
    for r in rows:
        o = pickle.loads((set_dir / "ocr" / f"{r['id']}.pkl").read_bytes())
        (out / f"{r['id']}.txt").write_text(build_prompt(o), encoding="utf-8")
    (out / "_system.txt").write_text(SYSTEM, encoding="utf-8")
    log(f"{len(rows)} prompts -> {out}")


def stage_extract(set_dir, rows, system):
    model = SYSTEM_MODELS[system]
    out = set_dir / "runs" / system / "raw"
    out.mkdir(parents=True, exist_ok=True)
    client = VLMClient()
    for r in rows:
        f = out / f"{r['id']}.json"
        if f.exists():
            continue
        o = pickle.loads((set_dir / "ocr" / f"{r['id']}.pkl").read_bytes())
        img = pp.load_image(ROOT / r["file"])
        try:
            run = extract(img, o, client=client, model=model)
            f.write_text(json.dumps({"id": r["id"], "model": run.model, "seconds": run.seconds,
                                     "tokens_in": run.tokens_in, "tokens_out": run.tokens_out,
                                     "reply": run.reply_text, "raw": run.raw.model_dump()},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"extract {system} {r['id']} {run.seconds}s")
        except Exception as e:  # keep going; failures are reported in the evaluation
            log(f"extract {system} {r['id']} FAILED: {e}")


def stage_ingest(set_dir, rows, system, replies: Path):
    """Replies produced elsewhere (Kaggle) -> raw extraction files, parsed exactly like ours."""
    out = set_dir / "runs" / system / "raw"
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    for line in replies.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        try:
            raw = coerce(parse_json(d["reply"]))
            payload = {"id": d["id"], "model": d.get("model", system), "seconds": d.get("seconds"),
                       "reply": d["reply"], "raw": raw.model_dump()}
            n_ok += 1
        except Exception as e:
            payload = {"id": d["id"], "model": d.get("model", system), "seconds": d.get("seconds"),
                       "reply": d["reply"], "raw": None, "error": f"unparseable reply: {e}"}
            n_bad += 1
        (out / f"{d['id']}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"ingested {n_ok} ok, {n_bad} unparseable -> {out}")


def stage_verify(set_dir, rows, verifier_model, tag):
    """Independent second opinion: image only (no OCR, no primary answer)."""
    out = set_dir / "verify" / tag
    out.mkdir(parents=True, exist_ok=True)
    client = VLMClient()
    for r in rows:
        f = out / f"{r['id']}.json"
        if f.exists():
            continue
        try:
            v = verify(pp.load_image(ROOT / r["file"]), client, model=verifier_model)
            f.write_text(json.dumps({"id": r["id"], "model": v.model, "seconds": v.seconds, "reply": v.reply,
                                     "raw": v.raw.model_dump()}, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"verify {tag} {r['id']} {v.seconds}s")
        except Exception as e:
            log(f"verify {tag} {r['id']} FAILED: {e}")


def stage_validate(set_dir, rows, system, verifier_tag=None):
    out = set_dir / "runs" / (system + (f"+{verifier_tag}" if verifier_tag else "")) / "records"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        rf = set_dir / "runs" / system / "raw" / f"{r['id']}.json"
        if not rf.exists():
            continue
        d = json.loads(rf.read_text(encoding="utf-8"))
        if not d.get("raw"):
            continue
        o = pickle.loads((set_dir / "ocr" / f"{r['id']}.pkl").read_bytes())
        raw = RawExtraction.model_validate(d["raw"])
        ver = None
        if verifier_tag:
            vf = set_dir / "verify" / verifier_tag / f"{r['id']}.json"
            if vf.exists():
                ver = RawExtraction.model_validate(json.loads(vf.read_text(encoding="utf-8"))["raw"])
        rec = validate(raw, o, image=pp.load_image(ROOT / r["file"]), image_name=Path(r["file"]).name, verifier=ver)
        link_records(rec, ocr_corpus_key(o))
        rec.models_used = [d.get("model", system)]
        rec.timings_s = {"ocr": o.timings_s["total"], "extract": d.get("seconds") or 0}
        (out / f"{r['id']}.json").write_text(rec.model_dump_json(indent=1), encoding="utf-8")
    log(f"validated {system} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ocr", "prompts", "extract", "ingest", "verify", "validate"])
    ap.add_argument("--set", default="set50")
    ap.add_argument("--system", default="nemotron")
    ap.add_argument("--replies", type=Path)
    ap.add_argument("--verifier-model", default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    ap.add_argument("--verifier-tag", default=None, help="validate: use this verifier output")
    a = ap.parse_args()
    set_dir = ROOT / "data/eval" / a.set
    rows = items(set_dir)
    {"ocr": lambda: stage_ocr(set_dir, rows),
     "prompts": lambda: stage_prompts(set_dir, rows),
     "extract": lambda: stage_extract(set_dir, rows, a.system),
     "ingest": lambda: stage_ingest(set_dir, rows, a.system, a.replies),
     "verify": lambda: stage_verify(set_dir, rows, a.verifier_model, a.verifier_tag or "nano"),
     "validate": lambda: stage_validate(set_dir, rows, a.system, a.verifier_tag)}[a.stage]()
