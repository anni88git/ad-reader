"""Create + push a Kaggle GPU kernel for one model run.

  python scripts/kaggle_kernel.py qwen3vl8b --smoke     # 2 ads, setup check
  python scripts/kaggle_kernel.py qwen3vl8b             # full set
  python scripts/kaggle_kernel.py qwen25vl7b
  python scripts/kaggle_kernel.py status qwen3vl8b      # check status
  python scripts/kaggle_kernel.py fetch qwen3vl8b       # download outputs to data/eval/set50/kaggle_out/
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"qwen3vl8b": "qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
          "qwen3vl4b": "qwen-lm/qwen-3-vl/transformers/4b-instruct/1",
          "qwen25vl7b": "qwen-lm/qwen2.5-vl/transformers/7b-instruct/2",
          "qwen25vl3b": "qwen-lm/qwen2.5-vl/transformers/3b-instruct/2"}


def run(*args):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print(out[-1500:])
    return out


def username():
    cfg = run("config", "view")
    return next(l.split(":", 1)[1].strip() for l in cfg.splitlines() if "username" in l)


def slug(kind, smoke, cpu=False):
    return f"adreader-{kind}{'-cpu' if cpu else ''}{'-smoke' if smoke else ''}"


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] in ("status", "fetch"):
        kind, smoke, cpu = args[1], "--smoke" in args, "--cpu" in args
        ref = f"{username()}/{slug(kind, smoke, cpu)}"
        if args[0] == "status":
            run("kernels", "status", ref)
        else:
            out = ROOT / "data/eval/kaggle_out" / slug(kind, smoke, cpu)
            out.mkdir(parents=True, exist_ok=True)
            run("kernels", "output", ref, "-p", str(out))
        sys.exit()

    kind, smoke, cpu = args[0], "--smoke" in args, "--cpu" in args
    dataset = "adreader-holdout20" if "--holdout" in args else "adreader-eval-set50"
    d = ROOT / "kaggle/kernels" / slug(kind, smoke, cpu)
    d.mkdir(parents=True, exist_ok=True)
    code = (ROOT / "kaggle/qwen_extract.py").read_text(encoding="utf-8")
    code = re.sub(r'^MODEL_KIND = ".*?"', f'MODEL_KIND = "{kind}"', code, flags=re.M)
    code = re.sub(r"^SMOKE = \w+", f"SMOKE = {smoke}", code, flags=re.M)
    (d / "run.py").write_text(code, encoding="utf-8")
    user = username()
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug(kind, smoke, cpu)}", "title": slug(kind, smoke, cpu), "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "false" if cpu else "true", "enable_tpu": "false", "enable_internet": "true",
        "dataset_sources": [f"{user}/{dataset}"], "competition_sources": [],
        "kernel_sources": [], "model_sources": [MODELS[kind]]}, indent=1), encoding="utf-8")
    # 8B needs both T4s (fp16 17.5 GB); smaller models run on whatever single GPU is free first
    acc = [] if cpu else ["--accelerator", "NvidiaTeslaT4"]   # explicit: default maps to a retired GPU
    run("kernels", "push", "-p", str(d), *acc)
