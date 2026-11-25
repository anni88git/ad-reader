"""Human audit workflow for the bench200 labels (Claude-drafted -> human-verified).

  python scripts/bench200_audit.py sheets     # contact sheets, 25 ads per batch -> data/eval/bench200/audit_sheets/
  python scripts/bench200_audit.py queue      # (re)build audit_queue.csv, keeping any statuses already entered
  python scripts/bench200_audit.py apply      # stamp labels the auditor marked verified / corrected
  python scripts/bench200_audit.py status     # counts by status

How to audit (per batch):
  1. Open audit_sheets/batch_NN.jpg (ad thumbnails with id) and, for detail, the image_path in the queue.
  2. Open the label_path JSON. If it is right, set audit_status=verified. If not, edit the JSON and set
     audit_status=corrected and describe the change in notes. If unsure, audit_status=unsure.
  3. Put your name in auditor. Run `apply`: only rows with verified/corrected AND an auditor name are
     stamped (_verified_by, _audited_at, _audit_notes). Nothing is ever marked verified automatically.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data/eval/bench200"
Q = B / "audit_queue.csv"
COLS = ["image_id", "batch", "image_path", "label_path", "language", "quality", "source",
        "label_source", "uncertain_fields", "audit_status", "auditor", "audited_at", "notes"]
PER = 25


def rows():
    return list(csv.DictReader((B / "manifest.csv").open(encoding="utf-8")))


def label_path(r):
    return ROOT / "data/labels" / f"{Path(r['file']).stem}.json"


def sheets():
    out = B / "audit_sheets"
    out.mkdir(exist_ok=True)
    rs = rows()
    for b in range(0, len(rs), PER):
        tiles = []
        for r in rs[b:b + PER]:
            im = cv2.imdecode(np.fromfile(str(ROOT / r["file"]), np.uint8), cv2.IMREAD_COLOR)
            s = 300 / max(im.shape[:2])
            im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
            t = np.full((335, 300, 3), 255, np.uint8)
            t[:im.shape[0], :im.shape[1]] = im
            cv2.putText(t, f"{r['id']} {r['language_label'][:8]}", (4, 327), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            tiles.append(t)
        while len(tiles) % 5:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        cv2.imwrite(str(out / f"batch_{b // PER + 1:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{(len(rs) + PER - 1) // PER} sheets -> {out}")


def queue():
    old = {r["image_id"]: r for r in csv.DictReader(Q.open(encoding="utf-8"))} if Q.exists() else {}
    out = []
    for i, r in enumerate(rows()):
        lp = label_path(r)
        lab = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
        o = old.get(r["id"], {})
        human = bool(lab.get("_verified_by"))
        out.append({"image_id": r["id"], "batch": i // PER + 1, "image_path": r["file"],
                    "label_path": lp.relative_to(ROOT).as_posix(), "language": r["language_label"],
                    "quality": r["quality"], "source": r["source"],
                    "label_source": "human-verified" if human else lab.get("_label_source", "missing"),
                    "uncertain_fields": " ".join(u["field"] for u in lab.get("_uncertain_fields", [])),
                    "audit_status": o.get("audit_status") or ("verified" if human else "pending"),
                    "auditor": o.get("auditor") or (lab.get("_verified_by") or ""),
                    "audited_at": o.get("audited_at", ""), "notes": o.get("notes", "")})
    with Q.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {Q}")


def apply():
    q = list(csv.DictReader(Q.open(encoding="utf-8")))
    n = 0
    for r in q:
        if r["audit_status"] not in ("verified", "corrected") or not r["auditor"].strip():
            continue
        lp = ROOT / r["label_path"]
        lab = json.loads(lp.read_text(encoding="utf-8"))
        if lab.get("_verified_by") == r["auditor"].strip():
            continue
        lab["_verified_by"] = r["auditor"].strip()
        lab["_audited_at"] = r["audited_at"] or datetime.now(timezone.utc).isoformat(timespec="seconds")
        lab["_audit_status"] = r["audit_status"]
        if r["notes"]:
            lab["_audit_notes"] = r["notes"]
        lp.write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    print(f"{n} labels stamped as human-verified")
    queue()


def status():
    from collections import Counter
    q = list(csv.DictReader(Q.open(encoding="utf-8")))
    print(Counter(r["audit_status"] for r in q), "| human-verified labels:",
          sum(1 for r in rows() if json.loads(label_path(r).read_text(encoding="utf-8")).get("_verified_by")))


if __name__ == "__main__":
    {"sheets": sheets, "queue": queue, "apply": apply, "status": status}[sys.argv[1]]()
