"""Kaggle job: native PaddleOCR (PP-OCRv5 pipeline) on the held-out images - for the OCR comparison
against RapidOCR (same PaddleOCR models via ONNX Runtime) used locally.
Output: /kaggle/working/paddle_ocr.jsonl  {id, seconds, lang, lines:[{text, score, box}]}
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paddlepaddle", "paddleocr"], check=False)
import paddle  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

print("paddle", paddle.__version__, "| GPU:", paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count(), flush=True)
DATA = os.path.dirname(glob.glob("/kaggle/input/**/manifest.csv", recursive=True)[0])
rows = list(csv.DictReader(open(f"{DATA}/manifest.csv", encoding="utf-8")))
LANG = {"english": "en", "hindi": "hi", "marathi": "mr", "tamil": "ta", "telugu": "te", "kannada": "ka",
        "bengali": "bn", "gujarati": "gu"}
engines = {}


ONLY = set(filter(None, os.environ.get("ONLY_IDS", "h17,h18,h19,h20").split(",")))  # resume after OOM


def engine(lang):
    if lang not in engines:
        engines.clear()          # one engine at a time: 5 server models in RAM got the job OOM-killed
        import gc
        gc.collect()
        try:
            # enable_mkldnn=False: PaddlePaddle 3.3 oneDNN/PIR bug on CPU ("ConvertPirAttribute2RuntimeAttribute")
            engines[lang] = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                                      use_textline_orientation=False, enable_mkldnn=False, device="cpu")
        except Exception as e:
            print("lang", lang, "unavailable:", e, "-> en", flush=True)
            engines[lang] = engine("en") if lang != "en" else None
    return engines[lang]


with open("/kaggle/working/paddle_ocr.jsonl", "w", encoding="utf-8") as out:
    for r in rows:
        if ONLY and r["id"] not in ONLY:
            continue
        lang = LANG.get(r["language_label"].lower(), "en")
        img = os.path.join(DATA, "images", os.path.basename(r["file"]))
        rec = {"id": r["id"], "lang": lang}
        try:
            eng = engine(lang)
            t = time.time()
            import cv2
            im = cv2.imread(img)
            f = min(1.0, 2500 / max(im.shape[:2]))   # 4000 px pages got the job OOM-killed
            if f < 1:
                im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            rec["scale"] = round(f, 3)
            res = list(eng.predict(im))
            rec["seconds"] = round(time.time() - t, 2)
            lines = []
            for page in res:
                for txt, sc, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
                    lines.append({"text": txt, "score": float(sc), "box": [[float(x), float(y)] for x, y in poly]})
            rec["lines"] = lines
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(r["id"], lang, rec.get("seconds"), len(rec.get("lines", [])), rec.get("error", ""), flush=True)
print("DONE", flush=True)
