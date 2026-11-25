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

MODEL_KIND = "qwen25vl3b"      # qwen3vl8b | qwen3vl4b | qwen25vl7b
SMOKE = False                 # True: only 2 ads, to validate the setup

# Qwen2.5-VL's processor config is not recognised by transformers 5.x -> pin the last 4.x release
TF_PIN = "transformers>=4.57,<5" if MODEL_KIND.startswith("qwen25") else "transformers>=4.57"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", TF_PIN, "accelerate"], check=False)

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
elif MODEL_KIND == "qwen25vl3b":
    from transformers import Qwen2_5_VLForConditionalGeneration as M
    MODEL_DIR, MODEL_NAME = find_dir(["qwen2.5-vl", "3b-instruct"]), "Qwen/Qwen2.5-VL-3B-Instruct"
else:
    from transformers import Qwen2_5_VLForConditionalGeneration as M
    MODEL_DIR, MODEL_NAME = find_dir(["qwen2.5-vl", "7b-instruct"]), "Qwen/Qwen2.5-VL-7B-Instruct"
print("model dir", MODEL_DIR, "| data", DATA, flush=True)

# ------------------------------------------------------------------ load (T4 has no bf16 -> fp16)
t = time.time()
GPU = torch.cuda.is_available()
if GPU:      # T4 has no bf16 -> fp16
    model = M.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa").eval()
else:        # CPU fallback (free GPU queue unavailable): fp32, all cores
    torch.set_num_threads(os.cpu_count() or 4)
    model = M.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
print("device:", "GPU" if GPU else f"CPU x{os.cpu_count()}", flush=True)
MAX_PIXELS = (1280 if GPU else 1024) * 28 * 28   # ~1.0 / 0.8 MP: enough for newspaper ads
try:
    processor = AutoProcessor.from_pretrained(MODEL_DIR, min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS)
except Exception as e:      # newer transformers: set the limits on the image processor instead
    print("processor kwargs rejected:", type(e).__name__, "- setting limits on image_processor", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
    except Exception as e2:  # Kaggle's Qwen2.5-VL copy: preprocessor_config.json lacks a recognised type
        print("AutoProcessor failed:", type(e2).__name__, "- building processor from parts", flush=True)
        from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLVideoProcessor, Qwen2_5_VLProcessor
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        processor = Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR),
                                        tokenizer=tok, video_processor=Qwen2VLVideoProcessor(),
                                        chat_template=getattr(tok, "chat_template", None))
ip = processor.image_processor
if hasattr(ip, "size") and isinstance(ip.size, dict) and "longest_edge" in ip.size:
    ip.size = {"shortest_edge": 256 * 28 * 28, "longest_edge": MAX_PIXELS}
for attr, val in (("min_pixels", 256 * 28 * 28), ("max_pixels", MAX_PIXELS)):
    if hasattr(ip, attr):
        setattr(ip, attr, val)
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

json.dump({"model": MODEL_NAME, "load_seconds": LOAD_S, "max_pixels": MAX_PIXELS, "device": "GPU" if GPU else "CPU",
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "transformers": transformers.__version__}, open(f"/kaggle/working/meta_{MODEL_KIND}.json", "w"))
print("DONE", flush=True)
