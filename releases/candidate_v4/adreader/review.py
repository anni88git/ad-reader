"""Human review -> final record, saved submission and (optionally) a ground-truth label.

UI-independent (used by the Streamlit POC now, and by a FastAPI backend later).

Rules enforced here, not in the UI:
  * a flagged field/row that has a value cannot be saved until the reviewer has checked it;
  * a master-data ID is saved only if the field was confirmed by the system with its value
    unchanged, or the reviewer explicitly selected the record; suggestions are never saved as IDs.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import normalize as nz
from .config import LABELS_DIR
from .schema import AdRecord, FieldStatus

SCALAR_FIELDS = ["brand", "product", "model_variant", "advertiser", "ad_type", "category",
                 "headline", "sub_headline", "description", "body_copy", "ad_message", "price", "warranty",
                 "call_to_action", "language", "qr_code_present", "disclaimer_present"]
LINKED = {"brand", "product", "advertiser", "category"}
LIST_FIELDS = ["prices", "phones", "websites", "emails", "social_handles", "addresses",
               "features", "offers", "dealers", "reference_numbers"]


def initial_edits(rec: AdRecord) -> dict[str, Any]:
    """Starting point for the review form. RED (missing) values are NOT pre-filled - they are
    shown as suggestions; IDs are pre-selected only for confirmed fields."""
    fields = {}
    for name in SCALAR_FIELDS:
        r = rec.fields.get(name)
        if r is None:
            fields[name] = {"value": None, "id": None, "checked": True, "status": "missing",
                            "suggestion": None}
            continue
        show = r.value if r.status != FieldStatus.missing else None
        fields[name] = {"value": show, "id": r.matched_id, "suggested_id": r.suggested_id,
                        "status": r.status.value, "checked": r.status == FieldStatus.confirmed,
                        "suggestion": r.value if r.status == FieldStatus.missing else _reason_suggestion(r)}
    lists = {}
    for name in LIST_FIELDS:
        rows = []
        for r in rec.lists.get(name, []):
            if r.status == FieldStatus.missing:
                continue                       # too uncertain: offered separately as suggestions
            rows.append({"value": r.value, "status": r.status.value,
                         "checked": r.status == FieldStatus.confirmed})
        lists[name] = rows
    attrs = {}
    for k, r in rec.category_attributes.items():
        attrs[k] = {"value": r.value if r.status != FieldStatus.missing else None,
                    "status": r.status.value, "checked": r.status == FieldStatus.confirmed,
                    "suggestion": r.value if r.status == FieldStatus.missing else None}
    return {"fields": fields, "lists": lists, "attributes": attrs}


def _reason_suggestion(r) -> str | None:
    """A concrete alternative proposed by a verification rule, e.g. the dealer names for the
    advertiser of a dealer ad ('... - suggestion: Allied Kia / West Coast Kia')."""
    for reason in r.reasons:
        m = re.search(r"suggestion:\s*(.+)$", reason)
        if m and nz.key(m.group(1)) != nz.key(str(r.value or "")):
            return m.group(1).strip()
    return None


def pending_checks(edits: dict[str, Any]) -> list[str]:
    """Flagged items with a value that the reviewer has not ticked as checked."""
    out = []
    for k, f in edits["fields"].items():
        if f["value"] not in (None, "", []) and not f["checked"]:
            out.append(k)
    for k, rows in edits["lists"].items():
        out += [f"{k}[{i}]" for i, r in enumerate(rows) if r["value"] not in (None, "", {}) and not r["checked"]]
    out += [f"attr.{k}" for k, a in edits["attributes"].items()
            if a["value"] not in (None, "", []) and not a["checked"]]
    return out


def build_final(rec: AdRecord, edits: dict[str, Any], reviewer: str) -> dict[str, Any]:
    pend = pending_checks(edits)
    if pend:
        raise ValueError(f"unchecked flagged items: {pend}")
    final_fields, corrections = {}, []
    for k, f in edits["fields"].items():
        sys_r = rec.fields.get(k)
        sys_val = sys_r.value if sys_r else None
        val = f["value"] if f["value"] not in ("", []) else None
        fid = f.get("id")
        if k in LINKED and fid is None and sys_r and sys_r.matched_id and _same(val, sys_val):
            fid = sys_r.matched_id         # confirmed by system, unchanged by reviewer
        final_fields[k] = {"value": val, "id": fid if k in LINKED else None}
        if not _same(val, sys_val) or (sys_r and sys_r.status == FieldStatus.missing and val is not None):
            corrections.append({"field": k, "system": sys_val,
                                "system_status": sys_r.status.value if sys_r else None, "reviewer": val})
    final_lists = {k: [r["value"] for r in rows if r["value"] not in (None, "", {})]
                   for k, rows in edits["lists"].items()}
    final_attrs = {k: a["value"] for k, a in edits["attributes"].items() if a["value"] not in (None, "", [])}
    return {
        "image": rec.image, "reviewed_by": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": final_fields, "lists": final_lists, "category_attributes": final_attrs,
        "corrections": corrections, "system_summary": rec.summary(), "models_used": rec.models_used,
    }


NEW_RECORD_ORDER = ["brand", "product", "advertiser"]     # a new product may need the new brand's ID


def apply_new_records(conn: sqlite3.Connection, final: dict[str, Any], edits: dict[str, Any],
                      rec: AdRecord) -> list[dict]:
    """Create master records the reviewer explicitly asked for (edits.fields[k].new_record = {"add": True,
    "acknowledged_ids": [...]}) and put the new IDs into `final`. All requests are validated first;
    if any is refused nothing is created (MasterDataError). The system itself never calls this."""
    from . import master_data as md
    todo = [k for k in NEW_RECORD_ORDER if (edits["fields"].get(k, {}).get("new_record") or {}).get("add")]
    if not todo:
        return []
    for k in todo:
        if final["fields"][k].get("id") is not None:
            raise md.MasterDataError(f"{k}: an existing record is selected - choose either link or add new")
        if final["fields"][k].get("value") in (None, ""):
            raise md.MasterDataError(f"{k}: no value to add")
    # dry run: duplicates / unacknowledged look-alikes / missing brand for a product
    for k in todo:
        req = edits["fields"][k]["new_record"]
        brand_id = final["fields"].get("brand", {}).get("id")
        if k == "product" and brand_id is None and "brand" not in todo:
            raise md.MasterDataError("product: link or add the brand first")
        chk = md.check_new(conn, k, str(final["fields"][k]["value"]), brand_id if k == "product" else None)
        if chk["exact"]:
            raise md.MasterDataError(f"{k}: {final['fields'][k]['value']!r} already exists - link it instead", chk["exact"])
        unack = [x for x in chk["similar"] if x["id"] not in set(req.get("acknowledged_ids") or [])]
        if unack:
            raise md.MasterDataError(f"{k}: similar records exist - confirm it is none of: "
                                     + ", ".join(f"{x['name']} (#{x['id']})" for x in unack), unack)
    created = []
    cat_id = final["fields"].get("category", {}).get("id")
    for k in todo:
        req = edits["fields"][k]["new_record"]
        sys_r = rec.fields.get(k)
        new_id = md.add_entity(conn, k, str(final["fields"][k]["value"]), added_by=final["reviewed_by"],
                               source_image=rec.image or "", evidence=(sys_r.evidence or "") if sys_r else "",
                               brand_id=final["fields"].get("brand", {}).get("id") if k != "brand" else None,
                               category_id=cat_id if k == "brand" else None,
                               acknowledged_ids=req.get("acknowledged_ids") or [],
                               advertiser_type=(final["fields"].get("ad_type") or {}).get("value") if k == "advertiser" else None)
        final["fields"][k]["id"] = new_id
        final["fields"][k]["new_record"] = True
        created.append({"field": k, "id": new_id, "name": final["fields"][k]["value"]})
    final["new_records"] = created
    return created


def _same(a, b) -> bool:
    if a in (None, "", []) and b in (None, "", []):
        return True
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        return str(a) == str(b)
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)
    return nz.key(str(a)) == nz.key(str(b))


def save_submission(conn: sqlite3.Connection, final: dict[str, Any], system_record: AdRecord) -> int:
    payload = {"final": final, "system_record": json.loads(system_record.model_dump_json())}
    with conn:
        cur = conn.execute("INSERT INTO submissions(image, record_json, reviewed_by) VALUES (?,?,?)",
                           (final["image"], json.dumps(payload, ensure_ascii=False), final["reviewed_by"]))
    return cur.lastrowid


def to_label(final: dict[str, Any]) -> dict[str, Any]:
    """Reviewed record -> ground-truth label (data/labels/FORMAT.md). Empty = not printed (null)."""
    lab: dict[str, Any] = {"_note": f"Ground truth saved from the review tool for {final['image']}",
                           "_verified_by": final["reviewed_by"], "_verified_at": final["reviewed_at"],
                           "_skip": ["body_copy"]}
    for k, f in final["fields"].items():
        lab[k] = f["value"]
        if f.get("id") is not None:
            lab[f"{k}_id"] = f["id"]
    for k, items in final["lists"].items():
        if k == "phones":
            lab[k] = [i["number"] if isinstance(i, dict) else nz.parse_phone(str(i)).number for i in items]
        else:
            lab[k] = items
    lab["category_attributes"] = final["category_attributes"]
    return lab


def save_label(final: dict[str, Any], stem: str, overwrite: bool = False) -> Path:
    path = LABELS_DIR / f"{stem}.json"
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("_verified_by"):          # drafts may be replaced by a verified label
            raise FileExistsError(f"verified label already exists: {path.name}")
    path.write_text(json.dumps(to_label(final), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
