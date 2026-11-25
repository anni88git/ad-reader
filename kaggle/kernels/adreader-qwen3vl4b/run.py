"""Kaggle GPU job: run an open-source Qwen VL model on the evaluation set with EXACTLY the same
prompts (image + OCR transcript + instructions) as the hosted model, and save the raw replies.
Parsing, verification and scoring happen locally with the same code for every model.

Config via the two constants below (each Kaggle kernel copy sets them).
Output: /kaggle/working/replies_<MODEL_KIND>.jsonl  (one line per ad, written incrementally)
"""
import glob
import json
import os
import subprocess
import sys
import time

MODEL_KIND = "qwen3vl4b"      # qwen3vl8b | qwen3vl4b | qwen25vl7b
SMOKE = False                 # True: only 2 ads, to validate the setup

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57", "accelerate"], check=False)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

print("transformers", transformers.__version__, "| torch", torch.__version__,
      "| GPUs", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)

# ------------------------------------------------------------------ locate inputs
def find_dir(pattern_parts):
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        if all(p in cfg.lower() for p in pattern_parts):
            return os.path.dirname(cfg)
    raise SystemExit(f"model not found for {pattern_parts}: {glob.glob('/kaggle/input/*')}")


DATA = os.path.dirname(glob.glob("/kaggle/input/**/manifest.csv", recursive=True)[0])
if MODEL_KIND == "qwen3vl8b":
    from transformers import Qwen3VLForConditionalGeneration as M
    MODEL_DIR, MODEL_NAME = find_dir(["qwen-3-vl", "8b-instruct"]), "Qwen/Qwen3-VL-8B-Instruct"
elif MODEL_KIND == "qwen3vl4b":
    from transformers import Qwen3VLForConditionalGeneration as M
    MODEL_DIR, MODEL_NAME = find_dir(["qwen-3-vl", "4b-instruct"]), "Qwen/Qwen3-VL-4B-Instruct"
else:
    from transformers import Qwen2_5_VLForConditionalGeneration as M
    MODEL_DIR, MODEL_NAME = find_dir(["qwen2.5-vl", "7b-instruct"]), "Qwen/Qwen2.5-VL-7B-Instruct"
print("model dir", MODEL_DIR, "| data", DATA, flush=True)

# ------------------------------------------------------------------ load (T4 has no bf16 -> fp16)
t = time.time()
model = M.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa").eval()
MAX_PIXELS = 1280 * 28 * 28           # ~1.0 MP: enough for newspaper ads, bounded GPU memory
processor = AutoProcessor.from_pretrained(MODEL_DIR, min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS)
ip = processor.image_processor
if hasattr(ip, "size") and isinstance(ip.size, dict) and "longest_edge" in ip.size:   # Qwen3-VL style
    ip.size = {"shortest_edge": 256 * 32 * 32, "longest_edge": MAX_PIXELS}
LOAD_S = round(time.time() - t, 1)
print("loaded in", LOAD_S, "s; memory GB:",
      [round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())], flush=True)

# ------------------------------------------------------------------ run
import csv  # noqa: E402

rows = list(csv.DictReader(open(f"{DATA}/manifest.csv", encoding="utf-8")))
if SMOKE:
    rows = rows[:2]
system = open(f"{DATA}/prompts/_system.txt", encoding="utf-8").read()
out_path = f"/kaggle/working/replies_{MODEL_KIND}.jsonl"
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}

with open(out_path, "a", encoding="utf-8") as out:
    for r in rows:
        if r["id"] in done:
            continue
        img_path = os.path.join(DATA, "images", os.path.basename(r["file"]))
        prompt = open(f"{DATA}/prompts/{r['id']}.txt", encoding="utf-8").read()
        rec = {"id": r["id"], "model": MODEL_NAME}
        try:
            image = Image.open(img_path).convert("RGB")
            messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(model.device)
            t = time.time()
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=8000, do_sample=False, repetition_penalty=1.05)
            new = gen[:, inputs["input_ids"].shape[1]:]
            rec.update(seconds=round(time.time() - t, 2), prompt_tokens=int(inputs["input_ids"].shape[1]),
                       new_tokens=int(new.shape[1]),
                       reply=processor.batch_decode(new, skip_special_tokens=True)[0])
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rec.update(seconds=None, reply="", error=f"OOM: {e}"[:300])
        except Exception as e:
            rec.update(seconds=None, reply="", error=f"{type(e).__name__}: {e}"[:300])
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(r["id"], rec.get("seconds"), "s", rec.get("new_tokens"), "tokens", rec.get("error", ""), flush=True)

json.dump({"model": MODEL_NAME, "load_seconds": LOAD_S, "max_pixels": MAX_PIXELS,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "transformers": transformers.__version__}, open(f"/kaggle/working/meta_{MODEL_KIND}.json", "w"))
print("DONE", flush=True)
