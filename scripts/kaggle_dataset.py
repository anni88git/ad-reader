"""Build + upload the PRIVATE Kaggle dataset for the GPU model runs (images, manifest, prompts).

  python scripts/kaggle_dataset.py            # create (first time) or new version
Prompts are generated from the local OCR results so every model gets identical input.
"""
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adreader.extract import SYSTEM, build_prompt  # noqa: E402

SET = ROOT / "data/eval/set50"
OUT = ROOT / "data/eval/set50_kaggle"
SLUG = "adreader-eval-set50"


def kaggle(*args):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(((r.stdout or "") + (r.stderr or "")).strip()[-800:])
    return r


if __name__ == "__main__":
    # username from the CLI config
    cfg = subprocess.run([sys.executable, "-m", "kaggle", "config", "view"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    username = next(l.split(":", 1)[1].strip() for l in cfg.splitlines() if "username" in l)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "prompts").mkdir()
    shutil.copy(SET / "manifest.csv", OUT / "manifest.csv")
    for p in (SET / "images").iterdir():
        shutil.copy(p, OUT / "images" / p.name)
    (OUT / "prompts" / "_system.txt").write_text(SYSTEM, encoding="utf-8")
    n = 0
    for f in sorted((SET / "ocr").glob("*.pkl")):
        (OUT / "prompts" / f"{f.stem}.txt").write_text(build_prompt(pickle.loads(f.read_bytes())), encoding="utf-8")
        n += 1
    print(f"dataset folder ready: {len(list((OUT / 'images').iterdir()))} images, {n} prompts")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "adreader eval set50", "id": f"{username}/{SLUG}",
        "licenses": [{"name": "other"}]}), encoding="utf-8")

    status = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", f"{username}/{SLUG}"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "ready" in (status.stdout + status.stderr).lower():
        kaggle("datasets", "version", "-p", str(OUT), "-m", f"{n} prompts", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(OUT), "--dir-mode", "zip")   # private by default
