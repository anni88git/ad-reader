"""Human audit of Claude-drafted ground-truth labels (random sample).

The audit measures how reliable the drafted labels are: every field the auditor changes is
recorded in the label (_audit_changes), and summary() reports the label error rate.
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .config import LABELS_DIR, ROOT

SET_DIR = ROOT / "data/eval/set50"
AUDIT_FILE = SET_DIR / "audit_sample.json"
FIELDS = ["brand", "product", "ad_type", "headline", "price", "phones"]   # quick audit: key fields
NOT_PRINTED = "(not printed)"
ALT_SEP = " || "


def sample(n_english=3, n_regional=2, seed=2024) -> list[dict]:
    if AUDIT_FILE.exists():
        return json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    rows = list(csv.DictReader((SET_DIR / "manifest.csv").open(encoding="utf-8")))
    rows = [r for r in rows if not r["source"].startswith("degraded")]
    eng = [r for r in rows if r["language_label"].lower() == "english"]
    reg = [r for r in rows if r["language_label"].lower() != "english"]
    rng = random.Random(seed)
    pick = rng.sample(eng, n_english) + rng.sample(reg, n_regional)
    out = [{"id": r["id"], "file": r["file"], "stem": Path(r["file"]).stem} for r in pick]
    AUDIT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------ text <-> label value
def to_text(v) -> str:
    """Label value -> editable text. Alternatives joined by ' || ', null -> '(not printed)'."""
    if isinstance(v, dict) and "any_of" in v:
        return ALT_SEP.join(to_text(a) for a in v["any_of"])
    if isinstance(v, dict) and "required" in v:
        req = ", ".join(map(str, v.get("required", [])))
        opt = ", ".join(map(str, v.get("optional", [])))
        return req + (f"  [optional: {opt}]" if opt else "")
    if v is None:
        return NOT_PRINTED
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return ", ".join(map(str, v)) if v else NOT_PRINTED
    return str(v)


def from_text(field: str, text: str):
    text = text.strip()
    alts = [a.strip() for a in text.split(ALT_SEP)] if ALT_SEP.strip() in text else [text]
    vals = [_one(field, a) for a in alts]
    return vals[0] if len(vals) == 1 else {"any_of": vals}


def _one(field, s):
    if s in ("", NOT_PRINTED):
        return [] if field in ("phones", "websites", "emails", "language") else None
    if field == "qr_code_present":
        return s.lower() in ("yes", "true", "1")
    if field == "price":
        d = "".join(ch for ch in s if ch.isdigit())
        return int(d) if d else None
    if field in ("phones", "websites", "emails", "language"):
        main, _, opt = s.partition("[optional:")
        req = [x.strip() for x in main.split(",") if x.strip()]
        if opt:
            return {"required": req, "optional": [x.strip() for x in opt.rstrip("] ").split(",") if x.strip()]}
        return req
    return s


def save_audit(stem: str, edited: dict[str, str], auditor: str) -> dict:
    path = LABELS_DIR / f"{stem}.json"
    lab = json.loads(path.read_text(encoding="utf-8"))
    changes = []
    for f, text in edited.items():
        before = to_text(lab.get(f)) if f in lab else ""
        if text.strip() != before.strip():
            changes.append({"field": f, "draft": lab.get(f), "audited": from_text(f, text)})
            lab[f] = from_text(f, text)
            if f in lab.get("_skip", []):
                lab["_skip"].remove(f)
    lab["_audited_by"] = auditor
    lab["_audited_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lab["_audit_changes"] = changes
    lab["_audit_fields_checked"] = len(edited)
    path.write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"changed": len(changes), "checked": len(edited)}


def summary() -> dict:
    s = {"audited_ads": 0, "fields_checked": 0, "fields_changed": 0, "ads_with_changes": 0, "changes": []}
    for item in sample():
        p = LABELS_DIR / f"{item['stem']}.json"
        if not p.exists():
            continue
        lab = json.loads(p.read_text(encoding="utf-8"))
        if not lab.get("_audited_by"):
            continue
        s["audited_ads"] += 1
        s["fields_checked"] += lab.get("_audit_fields_checked", 0)
        ch = lab.get("_audit_changes", [])
        s["fields_changed"] += len(ch)
        s["ads_with_changes"] += bool(ch)
        s["changes"] += [{"id": item["id"], **c} for c in ch]
    s["label_field_error_rate_pct"] = round(100 * s["fields_changed"] / s["fields_checked"], 1) if s["fields_checked"] else None
    return s
