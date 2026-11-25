"""Streamlit POC: upload ad -> extract -> validate -> match records -> review flagged -> save.

    .venv\\Scripts\\python -m streamlit run app/streamlit_app.py

Temporary demo UI. All logic lives in the `adreader` package (pipeline, review), so a
FastAPI + React front end can replace this file without touching the pipeline.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Demo runs the candidate_v4 rules (5 offline fixes; switched values always stay in review).
# Set ADREADER_RULES_V4=0 before starting the app to use candidate_v3 behaviour.
os.environ.setdefault("ADREADER_RULES_V4", "1")

from adreader import db, db_link, jobs, master_data as md, ocr as ocr_mod, opsdb, preprocess as pp, providers, review, usage  # noqa: E402
from adreader.config import OUTPUTS_DIR, vlm_config  # noqa: E402
from adreader.extract import CATEGORY_NAMES, extract  # noqa: E402
from adreader.matcher import default_matcher, link_records  # noqa: E402
from adreader.normalize import PRICE_KINDS  # noqa: E402
from adreader.pipeline import AdResult, ocr_corpus_key  # noqa: E402
from adreader.schema import AdType, FieldResult, FieldStatus, RawExtraction  # noqa: E402
from adreader.segment import AdRegion, find_ads  # noqa: E402
from adreader.validate import validate  # noqa: E402
from adreader.verify import verify  # noqa: E402
from adreader.vlm import VLMClient  # noqa: E402

st.set_page_config(page_title="Indycium Ad-Reader", layout="wide")
CACHE = OUTPUTS_DIR / "app_cache"
CACHE.mkdir(parents=True, exist_ok=True)
BADGE = {"confirmed": "🟢", "review": "🟠", "missing": "🔴"}
LINE_COLOURS = {"covered": (60, 170, 60), "ignored": (0, 140, 255), "small_print": (160, 160, 160),
                "decorative": (200, 120, 200)}
EVAL_DIR = ROOT / "data/eval/set50"
SAMPLE_DIRS = [EVAL_DIR / "images", ROOT / "data/ads", ROOT / "data/madverse/images"]


# =============================================================================== helpers
@st.cache_resource
def matcher():
    return default_matcher()


@st.cache_resource
def master_options():
    """Existing master records for the pickers. Database unreadable -> empty lists (nothing can be
    linked, nothing is invented) and a warning on the review screen."""
    try:
        if db_link.open_db() is None:
            raise RuntimeError("master database unavailable")
        m = matcher()
        return {"brands": m.tables["brands"], "categories": m.tables["categories"],
                "advertisers": m.tables["advertisers"], "products": m.products, "_ok": True}
    except Exception:
        return {"brands": [], "categories": [], "advertisers": [], "products": [], "_ok": False}


def samples() -> list[Path]:
    out = []
    for d in SAMPLE_DIRS:
        out += sorted(p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "page_" not in p.name)
    pages = sorted((ROOT / "data/ads").glob("page_*"))
    return pages + out


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def cached_stage_files(stem: str):
    """Results produced earlier by scripts (OCR cache + saved extraction) for sample images,
    including the evaluation-set batch run (data/eval/set50: files keyed by the 3-digit id)."""
    if (EVAL_DIR / "images").exists() and any((EVAL_DIR / "images").glob(f"{stem}.*")):
        i = stem[:3]
        o, e = EVAL_DIR / "ocr" / f"{i}.pkl", EVAL_DIR / "runs" / "nemotron" / "raw" / f"{i}.json"
    else:
        o = OUTPUTS_DIR / "ocr_cache" / f"{stem}.{'-'.join(ocr_mod.DEFAULT_LANGS)}.pkl"
        e = OUTPUTS_DIR / "extract" / f"{stem}.{vlm_config().model.split('/')[-1]}.json"
    return (o if o.exists() else None), (e if e.exists() else None)


# =============================================================================== processing jobs
# Every ad is processed as a persistent job (adreader/jobs.py): the image is stored first, each
# stage is checkpointed, and the job survives refreshes, API failures and restarts.
STATUS_ICON = {"queued": "⏳", "processing": "⚙️", "retrying": "🔁", "completed": "✅", "failed": "❌"}
STAGE_TEXT = {"queued": "waiting", "ocr": "1/4 reading text (OCR, 15-60 s on CPU)",
              "extract": "2/4 vision model extraction", "verify": "3/4 independent second opinion",
              "validate": "4/4 checks + database linking", "done": "done"}


def job_options(langs: tuple[str, ...], stem: str | None, region: AdRegion | None = None) -> dict:
    opt: dict = {"langs": list(langs)}
    if stem:            # demo samples: reuse OCR / extraction produced earlier by the evaluation scripts
        ocr_f, ex_f = cached_stage_files(stem)
        # cached demo OCR was made with the default scripts; auto mode reuses it too (shown as cached in the header)
        if ocr_f and (tuple(langs) in (ocr_mod.DEFAULT_LANGS, (ocr_mod.AUTO,)) or EVAL_DIR in ocr_f.parents):
            opt["ocr_cache"] = str(ocr_f)
        if ex_f:
            opt["raw_cache"] = str(ex_f)
    if region is not None:
        opt["region"], opt["region_label"] = list(region.box), region.label
    return opt


def job_result(jid: int) -> AdResult | None:
    out = jobs.load_output(jid)          # re-validated with the current rules on every open
    if out is None:
        return None
    res = AdResult(out.record, out.raw, out.ocr, out.image, out.region)
    res.verifier_raw = out.verifier_raw
    return res


# ------------------------------------------------------------------ embedded background worker
@st.cache_resource(show_spinner=False)
def _embedded_worker():
    """The same loop as scripts/worker.py, in a daemon thread - one per server process. Hosts without a separate
    worker process (Streamlit Community Cloud, Hugging Face) then behave like a local setup with a worker:
    jobs start as soon as they are queued and the live card updates on its own."""
    import socket
    import threading
    name = f"{socket.gethostname()}-{os.getpid()}-embedded"

    def loop():
        c = opsdb.connect()
        while True:
            try:
                jobs.worker_seen(name, os.getpid(), c)
                jobs.recover_stale(c)
                j = jobs.process_next(name, conn=c)
            except Exception:                        # never let the thread die on one bad job / DB hiccup
                j = None
                time.sleep(5)
            if not j:
                time.sleep(2)
    t = threading.Thread(target=loop, daemon=True, name="adreader-embedded-worker")
    t.start()
    return t


def ensure_worker() -> None:
    """Start the embedded worker when no external worker is alive (not under pytest; ADREADER_EMBEDDED_WORKER=0 turns
    it off). The cache makes this a no-op after the first call."""
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("ADREADER_EMBEDDED_WORKER", "1") == "0":
        return
    if not jobs.workers_alive():
        _embedded_worker()


def worker_running() -> bool:
    """An external worker has a recent heartbeat, or this process's embedded worker thread is alive (its heartbeat
    pauses while it processes a long job)."""
    if jobs.workers_alive():
        return True
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("ADREADER_EMBEDDED_WORKER", "1") == "0":
        return False
    try:
        return _embedded_worker().is_alive()
    except Exception:
        return False


def run_here(jids: list[int]):
    """No background worker running: process in this session, through the same persistent job."""
    for jid in jids:
        with st.spinner(f"Processing job #{jid} here (no background worker running)..."):
            jobs.process_next(f"app-{os.getpid()}", job_id=jid)


PIPELINE = [("ocr", "OCR"), ("extract", "Vision extraction"), ("verify", "Verification"), ("validate", "DB matching")]
STATUS_TITLE = {"queued": "Queued", "processing": "Processing", "retrying": "Retrying", "completed": "Completed",
                "failed": "Failed"}


STALL_S = 180          # a job running longer than this gets a "taking longer than usual" note + Cancel
HEARTBEAT_LOST_S = 60  # processing job whose worker has not sent a heartbeat for this long looks stalled


def _elapsed(j: dict) -> float:
    return max(0.0, time.time() - (j.get("created_at") or time.time()))


def _fmt_elapsed(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _job_card(j: dict) -> str:
    """Live-processing card: status dot + OCR -> Vision extraction -> Verification -> DB matching, elapsed time.
    Deliberately static (no CSS animation, fixed height) so 3-second refreshes never make it jump."""
    status, stage = j["status"], (j["stage"] or "queued")
    order = [k for k, _ in PIPELINE]
    cur = order.index(stage) if stage in order else (len(order) if stage == "done" else -1)
    steps = "".join(
        f'<div class="lp-step {"done" if (i < cur or status == "completed") else ("now" if i == cur and status in jobs.ACTIVE else "")}">'
        f'<span class="lp-dot"></span><span class="lp-lbl">{label}</span></div>'
        for i, (_, label) in enumerate(PIPELINE))
    sub = f"{STAGE_TEXT.get(stage, stage)} · attempt {j['attempts']}/{j['max_attempts']}"
    if status == "retrying" and j["next_attempt_at"]:
        sub += f" · next try in {max(0, int(j['next_attempt_at'] - time.time()))} s"
    timer = f'<span class="lp-time">{_fmt_elapsed(_elapsed(j))}</span>' if status in jobs.ACTIVE else ""
    return (f'<div class="lp-card lp-{status}"><div class="lp-head"><span class="lp-pulse"></span>'
            f'<p class="lp-title">{STATUS_TITLE.get(status, status)} · Job #{j["id"]}</p>{timer}</div>'
            f'<p class="lp-name">{html.escape(str(j["name"]))}</p><div class="lp-steps">{steps}</div>'
            f'<p class="lp-sub">{html.escape(sub)}</p></div>')


def _stalled(j: dict) -> str | None:
    """Why this active job looks stuck, or None."""
    if j["status"] == "processing" and j.get("heartbeat_at") and time.time() - j["heartbeat_at"] > HEARTBEAT_LOST_S:
        return "The worker processing this job has stopped responding."
    if j["status"] in jobs.ACTIVE and _elapsed(j) > STALL_S:
        return f"This is taking longer than usual ({_fmt_elapsed(_elapsed(j))})."
    return None


def _cancel_job(jid: int) -> None:
    jobs.cancel(jid)


@st.fragment(run_every=3)
def _live_status(jids: list[int]):
    """Background worker running: only this card re-draws (every 3 s) - the ad image and the rest of the page
    stay untouched. When every job is done, the whole page reruns once to show the results."""
    rows = [jobs.get(j) for j in jids]
    for j in rows:
        st.markdown(_job_card(j), unsafe_allow_html=True)
        why = _stalled(j)
        if why:
            st.warning(f"{why} You can cancel it (it can be retried later) or keep waiting.")
            st.button("✕ Cancel job", key=f"cancel:{j['id']}", on_click=_cancel_job, args=(j["id"],))
    if not any(j["status"] in jobs.ACTIVE for j in rows):
        st.rerun(scope="app")


def job_panel(jids: list[int], img: np.ndarray | None = None) -> bool:
    """Status of the given jobs; returns True when all are finished (completed or failed). While a job is still
    active the ad stays visible next to a live processing card - the page is never left blank."""
    rows = [jobs.get(j) for j in jids]
    active = [j["id"] for j in rows if j["status"] in jobs.ACTIVE]
    if not active:
        for j in rows:
            if j["status"] == "failed":
                st.markdown(_job_card(j), unsafe_allow_html=True)
                if j["last_error"]:
                    st.caption(f"Last error ({j['error_kind']}): {j['last_error'][:300]}")
                st.warning("The ad and the finished steps are saved. Retrying continues from the failed step.")
                if st.button("🔁 Retry job", key=f"retry:{j['id']}"):
                    jobs.retry(j["id"])
                    st.rerun()
        return True
    target = st.container()
    if img is not None:
        left, target = st.columns([1.05, 1], gap="large")
        h, w = img.shape[:2]
        left.image(img, channels="BGR", width="stretch" if w >= 0.9 * h else 420)
    with target:
        if worker_running():
            _live_status(jids)
            st.caption("Processing in the background - this card updates by itself. You can also close the page; "
                       "the job continues and stays on the **Jobs** page.")
            if st.button("🔄 Refresh status", key=f"refresh:{'-'.join(map(str, active))}"):
                st.rerun()
        else:
            for j in rows:
                st.markdown(_job_card(j), unsafe_allow_html=True)
                if j["status"] == "retrying" and j["last_error"]:
                    st.caption(f"Last error ({j['error_kind']}): {j['last_error'][:300]}")
            st.info("No background worker is running (start one with `python scripts/worker.py`). "
                    "You can process the job right here - it is still saved as a job.")
            if st.button("▶ Process now in this session", key=f"here:{'-'.join(map(str, active))}", type="primary"):
                run_here(active)
                st.rerun()
    return False


def html_table(df: pd.DataFrame, max_height: int = 420) -> None:
    """Read-only table as plain dark HTML (same look as the Jobs table). Unlike the canvas grid it has no
    selectable cells, so hovering / clicking a row can never paint a light highlight."""
    e = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else html.escape(str(v))
    num = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    head = "".join(f'<th class="{"jt-num" if c in num else ""}">{e(c)}</th>' for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f'<td class="{"jt-num" if c in num else "jt-txt"}" title="{e(v)}">{e(v)}</td>'
                         for c, v in zip(df.columns, row)) + "</tr>"
        for row in df.itertuples(index=False, name=None))
    st.markdown(f'<div class="jt-wrap" style="max-height:{max_height}px"><table class="jt"><thead><tr>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


# A theme the browser stored BEFORE .streamlit/config.toml existed ("Dark" / "Light" / "System") overrides the
# custom dark theme - and with a single theme configured Streamlit hides the menu to change it. Clear that stale
# entry once and reload, so every browser uses the configured dark theme (dark grid hover / selection colours).
RESET_CACHED_THEME_JS = """<script>
try {
  var w = window.parent, ls = w.localStorage, base = 'stActiveTheme-' + w.location.pathname, hit = false;
  Object.keys(ls).forEach(function (k) { if (k === 'stActiveTheme' || k.indexOf(base) === 0) { ls.removeItem(k); hit = true; } });
  if (hit && !w.sessionStorage.getItem('indyThemeReset')) { w.sessionStorage.setItem('indyThemeReset', '1'); w.location.reload(); }
} catch (e) {}
</script>"""


def draw_ocr(res: AdResult, show: set[str]) -> np.ndarray:
    vis = res.ocr.image.copy()
    for item, line in zip(res.record.ocr_lines_review, res.ocr.lines):
        if item["category"] in show:
            pts = np.asarray(line.box, dtype=np.int32)
            cv2.polylines(vis, [pts], True, LINE_COLOURS[item["category"]], 2)
    return vis


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(map(str, v))
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def why_box(r: FieldResult | None):
    if r is None:
        return
    with st.expander("evidence & reasons", expanded=False):
        st.caption(f"confidence {r.confidence:.2f} · status {r.status.value}")
        if r.evidence:
            st.markdown(f"**Evidence (as printed):** {r.evidence}")
        if r.english:
            st.markdown(f"**English:** {r.english}")
        for reason in r.reasons:
            st.markdown(f"- {reason}")
        if r.candidates:
            st.markdown("**Candidate records:** " + ", ".join(f"{c.name} (#{c.id}, {c.score:.0f})" for c in r.candidates))


# =============================================================================== review form
def _widget_text(kind: str, v) -> str:
    if kind == "bool":
        return {None: "", True: "yes", False: "no"}.get(v, "")
    return fmt(v)


def _apply_suggestion(key: str, kind: str, value) -> None:
    """Button callback: put a system suggestion into the input (runs before widgets render)."""
    st.session_state[key] = _widget_text(kind, value)
    st.session_state[f"{key}:chk"] = False        # still has to be verified by the reviewer


def scalar_widget(rid: str, name: str, label: str, res: AdResult, edits: dict, kind: str = "text",
                  options: list | None = None):
    r = res.record.fields.get(name)
    e = edits["fields"][name]
    key = f"{rid}:{name}"
    status = e["status"]
    # widget state is initialised once from the system value, then owned by the widget
    st.session_state.setdefault(key, _widget_text(kind, e["value"]))
    title = f"{BADGE.get(status, '⚪')} {label}"
    if kind == "textarea":
        v = st.text_area(title, key=key, height=90)
    elif kind == "select":
        opts = [""] + options
        if st.session_state[key] not in opts:
            st.session_state[key] = ""
        v = st.selectbox(title, opts, key=key) or None
    elif kind == "bool":
        v = {"": None, "yes": True, "no": False}[st.selectbox(title, ["", "yes", "no"], key=key)]
    elif kind == "int":
        s = st.text_input(title, key=key).replace(",", "").replace(" ", "").strip()
        v = int(s) if s.isdigit() else None
    elif kind == "csv":
        v = [x.strip() for x in st.text_input(title + " (comma separated)", key=key).split(",") if x.strip()] or None
    else:
        v = st.text_input(title, key=key).strip() or None
    e["value"] = v

    sugg = e.get("suggestion")
    if sugg not in (None, "", []):
        txt = ("too uncertain to fill" if status == "missing" else "the system suggests")
        c1, c2 = st.columns([0.72, 0.28])
        c1.caption(f"💡 {txt}: **{fmt(sugg)}**")
        c2.button("Use suggestion", key=f"{key}:use", on_click=_apply_suggestion, args=(key, kind, sugg))
    if status != "confirmed" and v not in (None, "", []):
        st.session_state.setdefault(f"{key}:chk", e["checked"])
        e["checked"] = st.checkbox("✔ I verified this value against the ad", key=f"{key}:chk")
    elif status != "confirmed":
        e["checked"] = True                  # empty = nothing to verify (saved as 'not printed')
    if name in review.LINKED:
        link_widget(rid, name, r, e)
    why_box(r)


def link_widget(rid: str, name: str, r: FieldResult | None, e: dict):
    table = {"brand": "brands", "category": "categories", "advertiser": "advertisers", "product": "products"}[name]
    ents = master_options()[table]
    by_id = {x.id: x for x in ents}
    opts: list[int | None] = [None]
    if r is not None:
        for c in r.candidates:
            if c.id not in opts:
                opts.append(c.id)
    opts += sorted((x.id for x in ents if x.id not in opts), key=lambda i: by_id[i].name.lower())

    def label(i):
        if i is None:
            return "— not linked (new record / leave empty) —"
        tag = " ✓ matched" if r is not None and i == r.matched_id else (
            " ← suggested" if r is not None and i == r.suggested_id else "")
        return f"{by_id[i].name} (#{i}){tag}"
    k = f"{rid}:{name}:id"
    st.session_state.setdefault(k, e.get("id") if e.get("id") in opts else None)
    e["id"] = st.selectbox(f"↳ {name} → existing record", opts, format_func=label, key=k,
                           help="IDs are pre-selected only for confirmed matches. Suggestions must be picked explicitly.")
    new_record_widget(rid, name, r, e)


DB_MATCH_TEXT = {
    "unknown": "🆕 **Not in the database** - nothing similar on record. Probably a new {n}.",
    "near_miss": "⚠️ **No exact record, but similar ones exist** - this may be a misreading of an existing {n}, or a new one.",
    "ambiguous": "⚠️ **Several records fit** - pick the right one above.",
    "blocked": "🔗 **Not linked because the brand is not linked** - link or add the brand first.",
}


def new_record_widget(rid: str, name: str, r: FieldResult | None, e: dict):
    """Reviewer-only path to create a master record. Never automatic; validated and audited on save."""
    if name not in md.TABLES:
        return
    kind = r.db_match if r is not None else None
    if kind in DB_MATCH_TEXT:
        st.caption(DB_MATCH_TEXT[kind].format(n=name))
    if e.get("id") is not None or e.get("value") in (None, "", []) or kind in (None, "matched"):
        e.pop("new_record", None)
        return
    add = st.checkbox(f"➕ Add “{e['value']}” to the database as a new {name}", key=f"{rid}:{name}:new",
                      help="Only after you have checked the value. Exact duplicates are refused; similar "
                           "records must be confirmed as different. Every addition is logged and can be undone.")
    if not add:
        e.pop("new_record", None)
        return
    with db.connect() as conn:
        brand_id = None
        chk = md.check_new(conn, name, str(e["value"]), brand_id)
    if chk["exact"]:
        st.error(f"Already in the database as {', '.join(f'{x['name']} (#{x['id']})' for x in chk['exact'])} - select it above instead.")
        e.pop("new_record", None)
        return
    ack = []
    if chk["similar"]:
        st.warning("Similar records: " + ", ".join(f"{x['name']} (#{x['id']}, {x['score']}%)" for x in chk["similar"]))
        if st.checkbox("I checked these - it is a different " + name, key=f"{rid}:{name}:ack"):
            ack = [x["id"] for x in chk["similar"]]
    e["new_record"] = {"add": True, "acknowledged_ids": ack}


def list_editor(rid: str, name: str, label: str, res: AdResult, edits: dict):
    rows = edits["lists"][name]
    recs = [r for r in res.record.lists.get(name, []) if r.status != FieldStatus.missing]
    data = []
    for row, r in zip(rows, recs):
        v = row["value"]
        base = {"status": BADGE[row["status"]], "why": "; ".join(r.reasons)[:160], "ok": row["checked"]}
        if name == "prices":
            data.append({"amount": v.get("amount"), "kind": v.get("kind"), "label": v.get("label"),
                         "as_printed": v.get("as_printed"), **base})
        elif name == "phones":
            data.append({"number": v.get("number"), "type": v.get("kind"), "as_printed": v.get("as_printed"), **base})
        elif name == "dealers":
            data.append({"name": v.get("name"), "area": v.get("area"), "city": v.get("city"),
                         "phones": ", ".join(v.get("phones") or []), **base})
        else:
            data.append({"value": fmt(v), **base})
    cols = {"prices": ["amount", "kind", "label", "as_printed"], "phones": ["number", "type", "as_printed"],
            "dealers": ["name", "area", "city", "phones"]}.get(name, ["value"])
    df = pd.DataFrame(data, columns=cols + ["status", "why", "ok"])
    cfg = {"status": st.column_config.TextColumn("", disabled=True, width="small"),
           "why": st.column_config.TextColumn("reason", disabled=True),
           "ok": st.column_config.CheckboxColumn("✔ checked", default=True)}
    if name == "prices":
        cfg["kind"] = st.column_config.SelectboxColumn("kind", options=list(PRICE_KINDS))
        cfg["amount"] = st.column_config.NumberColumn("amount (₹)", format="%d")
        cfg["as_printed"] = st.column_config.TextColumn("as printed", disabled=True)
    st.markdown(f"**{label}**")
    out = st.data_editor(df, key=f"{rid}:{name}:ed", num_rows="dynamic", hide_index=True,
                         column_config=cfg, width="stretch")
    new_rows = []
    for _, x in out.iterrows():
        if name == "prices":
            if pd.isna(x["amount"]):
                continue
            v = {"amount": int(x["amount"]), "currency": "INR", "kind": x["kind"] or "price",
                 "label": x["label"] if isinstance(x["label"], str) else None, "as_printed": x["as_printed"]}
        elif name == "phones":
            if not isinstance(x["number"], str) or not x["number"].strip():
                continue
            v = {"number": x["number"].strip(), "kind": x["type"], "as_printed": x["as_printed"]}
        elif name == "dealers":
            if not isinstance(x["name"], str) or not x["name"].strip():
                continue
            v = {"name": x["name"].strip(), "area": x["area"] or None, "city": x["city"] or None,
                 "phones": [p.strip() for p in str(x["phones"] or "").split(",") if p.strip()]}
        else:
            if not isinstance(x["value"], str) or not x["value"].strip():
                continue
            v = x["value"].strip()
        status = {"🟢": "confirmed", "🟠": "review", "🔴": "missing"}.get(x["status"], "added")
        new_rows.append({"value": v, "status": status, "checked": bool(x["ok"]) if status != "added" else True})
    edits["lists"][name] = new_rows
    suggestions = [r for r in res.record.lists.get(name, []) if r.status == FieldStatus.missing]
    if suggestions:
        st.caption("🔴 rejected as too uncertain (add manually if correct): " +
                   "; ".join(f"{fmt(r.value)} — {r.reasons[0] if r.reasons else ''}" for r in suggestions))


def attributes_editor(rid: str, res: AdResult, edits: dict):
    attrs = edits["attributes"]
    if not attrs:
        st.caption("No category-specific attributes extracted.")
        return
    data = [{"attribute": k, "value": fmt(a["value"]), "status": BADGE[a["status"]],
             "why": "; ".join(res.record.category_attributes[k].reasons)[:160], "ok": a["checked"],
             "suggestion": fmt(a.get("suggestion"))} for k, a in attrs.items()]
    out = st.data_editor(pd.DataFrame(data), key=f"{rid}:attrs", hide_index=True, width="stretch",
                         column_config={"attribute": st.column_config.TextColumn(disabled=True),
                                        "status": st.column_config.TextColumn("", disabled=True),
                                        "why": st.column_config.TextColumn("reason", disabled=True),
                                        "suggestion": st.column_config.TextColumn(disabled=True),
                                        "ok": st.column_config.CheckboxColumn("✔ checked")})
    for _, x in out.iterrows():
        a = attrs[x["attribute"]]
        val = x["value"].strip() if isinstance(x["value"], str) else ""
        if x["attribute"] == "fuel_type" and val:
            a["value"] = [s.strip() for s in val.split(",") if s.strip()]
        else:
            a["value"] = val or None
        a["checked"] = bool(x["ok"]) or a["status"] == "confirmed"


def ocr_review_panel(res: AdResult):
    items = res.record.ocr_lines_review
    groups = {c: [x for x in items if x["category"] == c] for c in ("ignored", "small_print", "decorative", "covered")}
    tabs = st.tabs([f"⚠ Possibly missed ({len(groups['ignored'])})", f"Small print ({len(groups['small_print'])})",
                    f"Decorative ({len(groups['decorative'])})", f"Covered ({len(groups['covered'])})"])
    help_ = {"ignored": "Readable OCR text that no field uses - check whether information was missed.",
             "small_print": "Tiny, dense or low-confidence text (disclaimers, unreadable print).",
             "decorative": "Logos, badges, stray characters, repeats of the brand name.",
             "covered": "Text already used by an extracted field."}
    for tab, (cat, xs) in zip(tabs, groups.items()):
        with tab:
            st.caption(help_[cat])
            if xs:
                html_table(pd.DataFrame([{"#": x["line"], "text": x["text"], "ocr conf": round(x["score"], 2)} for x in xs]),
                           max_height=320)


TABS = {
    "Identity": ["brand", "product", "model_variant", "advertiser", "ad_type", "category", "language"],
    "Creative": ["headline", "sub_headline", "description", "ad_message", "body_copy", "call_to_action", "features"],
    "Price & offers": ["price", "prices", "offers", "warranty"],
    "Contacts & dealers": ["phones", "websites", "emails", "social_handles", "addresses", "dealers",
                           "reference_numbers", "qr_code_present", "disclaimer_present"],
}


def tab_of(item: str) -> str:
    if item.startswith("attr."):
        return "Attributes"
    base = item.split("[")[0]
    return next((t for t, names in TABS.items() if base in names), "Attributes")


OUTCOME_TEXT = {"matched": "✅ matched", "suggested": "🟠 suggested - confirm", "ambiguous": "🟠 ambiguous - pick one",
                "near_miss": "🟠 similar records - pick or add", "unresolved": "🆕 Unmatched - new record",
                "blocked": "🔗 brand not linked", "db_unavailable": "⛔ database unavailable", "empty": "—"}
OVERVIEW = [("brand", "Brand"), ("product", "Product"), ("advertiser", "Client / advertiser"),
            ("headline", "Headline"), ("description", "Description"), ("price", "Price"),
            ("offers", "Offers"), ("phones", "Contacts: phones"), ("websites", "Contacts: websites"),
            ("emails", "Contacts: e-mails"), ("language", "Language"), ("category", "Category"),
            ("ad_type", "Ad type")]
STATUS_TEXT = {"confirmed": "🟢 Confirmed", "review": "🟠 Needs review", "missing": "🔴 Not found / too uncertain"}


def _item_text(v) -> str:
    return fmt(v.get("number") if isinstance(v, dict) and "number" in v else v)


def overview_panel(rec):
    """Every required field at a glance: value, Confirmed / Needs review, database match."""
    dbm = {r["field"]: r for r in db_link.match_report(rec)}
    rows = []
    for k, label in OVERVIEW:
        if k in rec.fields:
            r = rec.fields[k]
            val = fmt(r.value) if r.status != FieldStatus.missing else ""
            status = STATUS_TEXT[r.status.value]
        else:
            shown = [x for x in rec.lists.get(k, []) if x.status != FieldStatus.missing]
            val = "; ".join(_item_text(x.value) for x in shown)
            if not shown:
                status = "🔴 none found"
            elif all(x.status == FieldStatus.confirmed for x in shown):
                status = f"🟢 Confirmed ({len(shown)})"
            else:
                status = f"🟠 Needs review ({sum(x.status != FieldStatus.confirmed for x in shown)}/{len(shown)})"
        m = dbm.get(k)
        db_txt = OUTCOME_TEXT.get(m["outcome"], m["outcome"]) if m else ""
        db_id = ""
        if m:
            db_id = (f"#{m['db_id']}" if m["db_id"] is not None else
                     ", ".join(f"{c['name']} (#{c['id']})" for c in m["candidates"][:3]))
        rows.append({"Field": label, "Extracted value": val[:160], "Status": status,
                     "Database match": db_txt, "DB ID / candidates": db_id})
    with st.expander("📊 Field status & database matching (overview)", expanded=True):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("DB IDs come only from existing database records (string matching - never the AI model). "
                   "An ID is pre-filled only on a confirmed match; suggestions and ambiguous matches must be "
                   "picked by you in the Identity tab. Unmatched values stay unlinked unless you add a new record.")


def form_panel(rec, edits: dict, rid: str):
    """The populated data-entry form: live preview while reviewing, final version after saving."""
    saved = st.session_state.get(f"{rid}:saved_form")
    form = saved or db_link.with_db(lambda conn: db_link.preview_form(rec, edits, conn))
    title = (f"📋 Data-entry form - SAVED (submission #{saved['submission_id']})" if saved
             else "📋 Data-entry form - live preview (fills in as you review)")
    with st.expander(title, expanded=True):
        if form["pending_review"]:
            st.warning(f"Draft: {len(form['pending_review'])} item(s) still need review: "
                       + ", ".join(form["pending_review"]))
        for fl in form["flags"]:
            st.error(fl)
        html_table(pd.DataFrame([{"Field": v["label"], "Value": fmt(v["value"]),
                                  "DB ID": "" if v.get("db_id") is None else f"#{v['db_id']}",
                                  "Resolution": v.get("db_resolution", "")}
                                 for v in form["fields"].values()]), max_height=520)
        c1, c2 = st.columns(2)
        c1.markdown("**Prices:** " + (", ".join(fmt(p.get("amount") if isinstance(p, dict) else p)
                                               for p in form["prices"]) or "—"))
        c1.markdown("**Offers:** " + ("; ".join(fmt(o) for o in form["offers"]) or "—"))
        contacts = [f"{k}: {', '.join(_item_text(x) for x in v)}" for k, v in form["contacts"].items() if v]
        c2.markdown("**Contacts:** " + ("; ".join(contacts) or "—"))
        if form["dealers"]:
            c2.markdown("**Dealers:** " + "; ".join(
                str(d["name"]) + (f" (#{d['advertiser_id']})" if d["advertiser_id"] else "") for d in form["dealers"]))
        payload = json.dumps(form, ensure_ascii=False, indent=2, default=str)
        st.download_button("⬇ Export form as JSON", payload,
                           file_name=f"{Path(rec.image.split('#')[0]).stem}_form.json", mime="application/json",
                           key=f"{rid}:dl:{'saved' if saved else 'draft'}")
        if st.checkbox("Show JSON payload", key=f"{rid}:showjson:{'saved' if saved else 'draft'}"):
            st.json(form)


def review_screen(res: AdResult, rid: str):
    rec = res.record
    if rid not in st.session_state.edits:
        st.session_state.edits[rid] = review.initial_edits(rec)
    edits = st.session_state.edits[rid]
    s = rec.summary()
    st.markdown(f"#### {rec.image}  ·  🟢 {s['confirmed']}  🟠 {s['review']}  🔴 {s['missing']}  ·  "
                f"model: `{', '.join(rec.models_used)}`" + (f"  ·  time: {t}s" if (t := rec.timings_s.get('total')) is not None else ""))
    det = getattr(res.ocr, "script_detection", None)          # older OCR results have no detection field
    used = ", ".join(getattr(res.ocr, "langs", ()) or ())
    st.caption(f"Scripts: auto → **{used}**" if det else f"Scripts: {used}")
    for w in rec.quality_warnings:
        st.warning(f"Image quality: {w}")
    for f in rec.record_flags:
        st.info(f)
    if not master_options()["_ok"]:
        st.error("Master database unavailable - no record can be linked. Values can still be reviewed; "
                 "all database IDs will stay empty.")
    overview_panel(rec)

    left, right = st.columns([0.42, 0.58])
    with left:
        show = set(st.multiselect("Show OCR boxes", list(LINE_COLOURS), default=["ignored", "covered"],
                                  key=f"{rid}:boxes"))
        st.image(draw_ocr(res, show), channels="BGR", width="stretch",
                 caption="green = used by a field · orange = possibly missed · grey = small print · purple = decorative")
        ocr_review_panel(res)

    with right:
        t1, t2, t3, t4, t5 = st.tabs(["Identity", "Creative", "Price & offers", "Contacts & dealers", "Attributes"])
        with t1:
            scalar_widget(rid, "brand", "Brand", res, edits)
            scalar_widget(rid, "product", "Product / model", res, edits)
            scalar_widget(rid, "model_variant", "Variant", res, edits)
            scalar_widget(rid, "advertiser", "Advertiser / client", res, edits)
            scalar_widget(rid, "ad_type", "Ad type", res, edits, "select", [t.value for t in AdType])
            scalar_widget(rid, "category", "Category", res, edits, "select", CATEGORY_NAMES)
            scalar_widget(rid, "language", "Language(s)", res, edits, "csv")
        with t2:
            scalar_widget(rid, "headline", "Headline", res, edits, "textarea")
            scalar_widget(rid, "sub_headline", "Sub-headline / tagline", res, edits, "textarea")
            scalar_widget(rid, "description", "Description", res, edits, "textarea")
            scalar_widget(rid, "ad_message", "Ad message (what the ad means, in English)", res, edits, "textarea")
            scalar_widget(rid, "body_copy", "Body copy (what the ad communicates)", res, edits, "csv")
            scalar_widget(rid, "call_to_action", "Call to action", res, edits)
            list_editor(rid, "features", "Features", res, edits)
        with t3:
            scalar_widget(rid, "price", "Current price (₹)", res, edits, "int")
            list_editor(rid, "prices", "All printed prices", res, edits)
            list_editor(rid, "offers", "Offers", res, edits)
            scalar_widget(rid, "warranty", "Warranty", res, edits, "textarea")
        with t4:
            list_editor(rid, "phones", "Phone numbers", res, edits)
            list_editor(rid, "websites", "Websites", res, edits)
            list_editor(rid, "emails", "E-mails", res, edits)
            list_editor(rid, "social_handles", "Social handles", res, edits)
            list_editor(rid, "addresses", "Addresses", res, edits)
            list_editor(rid, "dealers", "Dealers / stores", res, edits)
            list_editor(rid, "reference_numbers", "Plan / registration numbers (UIN, IRDAI, RERA ...)", res, edits)
            scalar_widget(rid, "qr_code_present", "QR code present", res, edits, "bool")
            scalar_widget(rid, "disclaimer_present", "Disclaimer / T&C present", res, edits, "bool")
        with t5:
            attributes_editor(rid, res, edits)

        st.divider()
        pend = review.pending_checks(edits)
        reviewer = st.text_input("Reviewer name", key="reviewer", value=st.session_state.get("reviewer", ""))
        as_label = st.checkbox("Also save as ground-truth label (for the 50-ad evaluation)", key=f"{rid}:label")
        if pend:
            by_tab: dict[str, list[str]] = {}
            for p in pend:
                by_tab.setdefault(tab_of(p), []).append(p.replace("attr.", ""))
            st.error(f"**{len(pend)} flagged item(s) still need your check** (tick ✔ after verifying):\n\n" +
                     "\n".join(f"- **{t} tab:** {', '.join(items)}" for t, items in by_tab.items()))
        if st.button("💾 Save verified record", type="primary", disabled=bool(pend) or not reviewer.strip(),
                     key=f"{rid}:save"):
            final = review.build_final(rec, edits, reviewer.strip())
            if db_link.open_db() is None:
                st.error("Not saved: the master database is unavailable.")
                st.stop()
            with db.connect() as conn:
                try:
                    created = review.apply_new_records(conn, final, edits, rec)
                except md.MasterDataError as err:
                    st.error(f"Not saved: {err}")
                    st.stop()
                final["form"] = db_link.build_form(final, conn)      # IDs verified against the database
                sid = review.save_submission(conn, final, rec)
            st.session_state[f"{rid}:saved_form"] = {**final["form"], "submission_id": sid}
            if created:
                matcher.clear()
                master_options.clear()
            msg = f"Saved submission #{sid} - {len(final['corrections'])} correction(s) vs the system."
            if created:
                msg += " New database records: " + ", ".join(f"{c['field']} “{c['name']}” (#{c['id']})" for c in created) + "."
            if as_label:
                stem = Path(rec.image.split("#")[0]).stem + ("_" + rec.image.split("#")[1] if "#" in rec.image else "")
                try:
                    p = review.save_label(final, stem)
                    msg += f" Label written: {p.relative_to(ROOT)}"
                except FileExistsError as e:
                    msg += f" (label not written: {e})"
            st.success(msg)
            if final["corrections"]:
                html_table(pd.DataFrame(final["corrections"]).astype(str))
        form_panel(rec, edits, rid)


# ------------------------------------------------------------------ pre-extraction "Ready to extract" card
# What the pipeline fills in. The examples are what is printed on the Kia reference ad (shown only for that
# image); statuses and DB IDs are never shown here - they only come from a real extraction run.
_I = ('<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
      'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>')
READY_FIELDS = [   # (title, what it is, example printed on the Kia reference ad, icon)
    ("Brand &amp; DB ID", "Brand, linked to a master-database ID", "Kia",
     _I.format('<path d="M4 7l8-4 8 4-8 4z"/><path d="M4 12l8 4 8-4M4 17l8 4 8-4"/>')),
    ("Product / Variant", "Model and variant", "Sorento",
     _I.format('<path d="M21 8l-9-5-9 5 9 5z"/><path d="M3 8v8l9 5 9-5V8"/>')),
    ("Headline", "Main headline and tagline", "Superior tech and elegance, beyond the class.",
     _I.format('<path d="M5 6h14M5 12h14M5 18h9"/>')),
    ("Price &amp; Offers", "Prices, offers, EMI, warranty", "₹27,99,000 introductory price",
     _I.format('<path d="M7 5h10M7 9h10M9 5c4 0 4 8 0 8H7l7 7"/>')),
    ("Attributes", "Fuel, mileage, engine… when printed", "Feature highlights from the ad",
     _I.format('<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>')),
    ("Contacts &amp; Dealers", "Phones, websites, dealers", "1800 108 5005 · kia.com · Allied Kia, West Coast Kia",
     _I.format('<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a1 1 0 0 1-1 1A16 16 0 0 1 4 5a1 1 0 0 1 1-1z"/>')),
]


def _stem_of(job_name: str) -> str:
    return Path(job_name.split("#")[0]).stem


def _open_job(jid: int) -> None:                    # widget callback: runs before the rerun, so state can change
    st.session_state["page"] = "Jobs"
    st.session_state["jobs_open"] = jid


def _open_saved() -> None:
    st.session_state["page"] = "Saved records"


def ready_card(name: str, stem: str | None) -> None:
    """Right-hand card before extraction: what to do, what gets extracted, and any earlier result for this ad."""
    key = stem or (Path(name).stem if name else None)
    kia = key == "_kia_from_screenshot"
    tiles = "".join(
        f'<div class="hw-tile"><div class="hw-ic">{ic}</div><div><p class="hw-tt">{t}</p>'
        f'<p class="hw-td">{ex if kia else d}</p></div></div>'
        for t, d, ex, ic in READY_FIELDS)
    st.markdown(
        f'<div class="rc-card"><p class="rc-h">{"Ready to extract" if key else "How it works"}</p>'
        '<p class="rc-t">Upload your own JPG/PNG or pick a sample, then click '
        '<b>▶ Extract &amp; verify</b> to auto-populate the data-entry form.</p>'
        '<div class="hw-pipe"><span><b>01</b> Read</span><i>→</i><span><b>02</b> Verify</span><i>→</i>'
        '<span><b>03</b> Match DB</span></div>'
        f'<p class="rc-k">{"Example · printed on this ad" if kia else "What this pipeline extracts"}</p>'
        f'<div class="hw-grid">{tiles}</div>'
        '<div class="hw-legend"><span>🟢 Confirmed</span><span>🟠 Review</span><span>🔴 Missing</span>'
        '<span class="hw-zero">Zero guessed DB IDs</span></div>'
        '<p class="rc-n">Statuses and database IDs appear only after a real run - nothing here is guessed.</p></div>',
        unsafe_allow_html=True)
    # earlier results for the same file name (jobs and saved records can carry a numeric prefix, e.g. 001_)
    if not key:
        return
    try:
        done = [r for r in jobs.list_jobs(200) if r["status"] == "completed" and _stem_of(r["name"]).endswith(key)]
        with db.connect() as conn:
            saved = [r["id"] for r in conn.execute("SELECT id, image FROM submissions ORDER BY id DESC").fetchall()
                     if _stem_of(r["image"]).endswith(key)]
    except Exception:                               # the card is a convenience - never block the page on it
        done, saved = [], []
    if done or saved:
        with st.container(key="rc_prev"):
            if done:
                j = done[0]
                st.button(f"Open completed job #{j['id']} · {j['name']}", key="rc_open_job",
                          on_click=_open_job, args=(j["id"],))
            if saved:
                st.button(f"View saved record #{saved[0]} in Saved records", key="rc_open_saved", on_click=_open_saved)
            st.caption("This file name was processed before - open it instead of running the model again.")


# =============================================================================== pages
SAMPLE_PLACEHOLDER = "Choose a sample advertisement..."


def inputs_in_sidebar() -> bool:
    """The input widgets normally live in a card on the main page. The existing UI tests (written for the old
    layout) drive them as sidebar widgets, so under pytest - or with ADREADER_SIDEBAR_INPUTS=1 - the very same
    widgets (same keys, same logic) are placed in the sidebar instead."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or os.environ.get("ADREADER_SIDEBAR_INPUTS") == "1"


def _chosen_name() -> str | None:
    """File name of the ad currently chosen in the picker (read from widget state, before the widgets are drawn)."""
    up = st.session_state.get(f"upload_{st.session_state.get('upload_n', 0)}")
    if up is not None:
        return getattr(up, "name", None) or "uploaded image"
    smp = st.session_state.get("sample")
    return Path(str(smp)).name if smp else None


def _analyze_another() -> None:                     # callback: back to the clean picker
    st.session_state.pop("proc_open", None)
    st.session_state["sample"] = None
    st.session_state["upload_n"] = st.session_state.get("upload_n", 0) + 1   # fresh (empty) uploader


def _input_widgets():
    """Upload / sample / advanced options. Returns (data, name, stem, langs, full_page)."""
    data, name, stem = None, None, None
    up = st.file_uploader("Upload an advertisement (JPG/PNG)", type=["jpg", "jpeg", "png"],
                          key=f"upload_{st.session_state.get('upload_n', 0)}")
    st.markdown('<p class="pc-or">or</p>', unsafe_allow_html=True)
    sel = st.selectbox("Select a sample advertisement", samples(), index=None, placeholder=SAMPLE_PLACEHOLDER,
                       format_func=lambda p: p.relative_to(ROOT).as_posix(), key="sample")
    with st.expander("Advanced options", expanded=False):
        auto = st.toggle("Auto-detect language / script", value=True, key="ocr_auto",
                         help="Detects Telugu, Tamil, Kannada or Devanagari text in the ad and loads only that "
                              "OCR model (always together with Latin). Turn off, or change the list below, to "
                              "choose the scripts yourself.")
        langs = tuple(st.multiselect(
            "OCR scripts (alphabets to read)", list(ocr_mod.SUPPORTED_LANGS), default=list(ocr_mod.DEFAULT_LANGS),
            key="langs", help="latin = English, devanagari = Hindi/Marathi. Keep the default unless the ad is in "
                              "Tamil, Telugu or Kannada - then ADD that script. Other Indian scripts have no OCR model."))
        full_page = st.checkbox("Full newspaper page (find ads first)", key="full_page")
    if not langs:
        st.error("Select at least one script - using the default (latin + devanagari).")
        langs = ocr_mod.DEFAULT_LANGS
    elif set(langs) != set(ocr_mod.DEFAULT_LANGS):
        missing = [l for l in ocr_mod.DEFAULT_LANGS if l not in langs]
        st.warning("Not the default setting: OCR will be re-run (slow), saved sample results are not "
                   "reused." + (f" Removing **{', '.join(missing)}** means that script is not read." if missing else ""))
    langs = tuple(l for l in ocr_mod.SUPPORTED_LANGS if l in langs)   # stable order for cache keys
    if auto and langs == ocr_mod.DEFAULT_LANGS:     # the list untouched -> auto-detect; an edited list overrides
        langs = (ocr_mod.AUTO,)
    if up:                                          # an upload wins over the sample dropdown
        data, name = up.getvalue(), up.name
    elif sel:
        data, name, stem = sel.read_bytes(), sel.name, sel.stem
        full_page = full_page or sel.name.startswith("page_")
    return data, name, stem, langs, full_page


def page_process():
    legacy = inputs_in_sidebar()
    col_left = col_right = None
    if legacy:
        with st.sidebar:
            st.header("Input")
            data, name, stem, langs, full_page = _input_widgets()
    elif st.session_state.get("proc_open") is None and _chosen_name() is None:   # nothing chosen: full picker card
        col_left, col_right = st.columns([1.05, 1], gap="large")
        with col_left:
            with st.container(key="proc_input"):
                st.markdown('<p class="pc-h">Analyze an advertisement</p>'
                            '<p class="pc-t">Upload your own ad, or pick one of the samples.</p>', unsafe_allow_html=True)
                data, name, stem, langs, full_page = _input_widgets()
    elif st.session_state.get("proc_open") is None:  # an ad is chosen: slim one-row bar, picker tucked in a popover
        with st.container(key="proc_bar"):
            c_name, c_change = st.columns([4, 1.25], vertical_alignment="center")
            c_name.markdown(f'<p class="pb-name">📄 {html.escape(_chosen_name())}</p>', unsafe_allow_html=True)
            with c_change:
                with st.popover("Change ad / options", width="stretch"):
                    data, name, stem, langs, full_page = _input_widgets()
        if not data:                                # cleared from the popover: show the full picker again
            st.rerun()
        col_left, col_right = st.columns([1.05, 1], gap="large")
    else:                                           # analysing: picker hidden (widgets kept so their values persist)
        with st.container(key="proc_input_hidden"):
            data, name, stem, langs, full_page = _input_widgets()
    key = (sha(data) + "-" + "-".join(langs)) if data else None
    if st.session_state.get("proc_open") not in (None, key):   # the chosen ad changed underneath: back to picker
        st.session_state.pop("proc_open", None)
        st.rerun()
    if not data:
        if col_right is not None:
            with col_right:
                ready_card(None, None)
        else:
            st.info("Choose a sample or upload an advertisement image to start.")
        return
    img = pp.load_image(data)
    h, w = img.shape[:2]
    width = "stretch" if w >= 0.9 * h else 420      # tall portrait ads get a capped width

    if st.session_state.get("proc_open") != key:     # ---- chosen, not analysed yet: action button + preview
        with (col_left if col_left is not None else st.container()):
            if full_page:
                if st.button("🔎 Find ads", key="find", type="primary"):
                    st.session_state["proc_open"] = key
                    st.rerun()
                st.image(img, channels="BGR", width=width,
                         caption="Newspaper page preview - click Find ads to locate the advertisements on it.")
                return
            clicked = st.button("▶ Extract & verify", type="primary", key="run")
            st.image(img, channels="BGR", width=width)
        if not clicked:
            if col_right is not None:
                with col_right:
                    ready_card(name, stem)
            return
        # Start a fresh run: continuing this one would leave the picker/preview drawn above and push the live
        # processing card below the fold - the "blank screen" users saw after clicking Extract & verify.
        st.session_state["proc_open"] = key
        st.rerun()

    # ---- analysing / reviewing this ad
    if not legacy:
        st.button("← Analyze another ad", key="proc_back", on_click=_analyze_another)
        st.caption(f"Analyzing **{name}**")
    if full_page:
        st.subheader("1 · Find advertisements on the page")
        if f"regions:{key}" not in st.session_state:
            uploads = OUTPUTS_DIR / "page_uploads"              # keep the page even if this step fails
            uploads.mkdir(parents=True, exist_ok=True)
            (uploads / f"{key}{Path(name).suffix or '.png'}").write_bytes(data)
            with st.spinner("Asking the vision model to locate ads..."):
                try:
                    regs, single = find_ads(img, VLMClient())
                except Exception as err:
                    st.session_state.pop("proc_open", None)
                    st.error(f"Finding ads failed ({type(err).__name__}): {str(err)[:200]}. The page is saved; "
                             "click **Find ads** again to retry, or untick *Full newspaper page* (Advanced options) "
                             "and crop it manually.")
                    return
            st.session_state[f"regions:{key}"] = regs
        regs = st.session_state[f"regions:{key}"]
        vis = img.copy()
        for i, r in enumerate(regs, 1):
            cv2.rectangle(vis, r.box[:2], r.box[2:], (0, 0, 255), 5)
            cv2.putText(vis, str(i), (r.box[0] + 10, r.box[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        c1, c2 = st.columns([0.5, 0.5])
        c1.image(vis, channels="BGR", width="stretch", caption="Proposed ad regions - confirm or adjust")
        df = pd.DataFrame([{"use": True, "label": r.label, "x1": r.box[0], "y1": r.box[1], "x2": r.box[2], "y2": r.box[3]}
                           for r in regs])
        with c2:
            st.caption("Untick wrong boxes or edit coordinates (pixels). Nothing is processed until you confirm.")
            ed = st.data_editor(df, key=f"regions_ed:{key}", hide_index=True, num_rows="dynamic")
            if st.button("✅ Process selected ads", type="primary", key="go_regions"):
                st.session_state[f"selected:{key}"] = [
                    AdRegion(str(x["label"]), (int(x["x1"]), int(x["y1"]), int(x["x2"]), int(x["y2"])))
                    for _, x in ed.iterrows() if x["use"]]
        selected = st.session_state.get(f"selected:{key}")
        if not selected:
            return
        jids = [jobs.enqueue(data, f"{name}#ad{i}", job_options(langs, None, r))
                for i, r in enumerate(selected, 1)]
    else:
        opts = job_options(langs, stem)
        existing = jobs.find(data, opts)            # an earlier run of the same image is reused (no new model call)
        jids = [existing if existing is not None else jobs.enqueue(data, name, opts)]

    if not job_panel(jids, None if full_page else img):
        return
    results = [r for r in (job_result(j) for j in jids) if r is not None]
    if not results:
        return
    with st.container(key="result_wrap"):                  # fades up when the results first appear
        st.markdown('<div id="extraction-result-anchor"></div>', unsafe_allow_html=True)
        st.subheader("2 · Review and save")
        names = [r.record.image for r in results]
        pick = st.radio("Advertisement", names, horizontal=True, key=f"pick:{key}") if len(results) > 1 else names[0]
        res = results[names.index(pick)]
        review_screen(res, rid=f"{key}:{pick}")
    scrolled = st.session_state.setdefault("results_scrolled", [])
    if key not in scrolled:                         # scroll once per ad - never again on later form edits
        scrolled.append(key)
        with st.container(key="scroll_js"):
            components.html(SCROLL_TO_RESULTS_JS, height=0)


# ------------------------------------------------------------------ verification & benchmark status (real figures only)
# Every number below is read from the repository at run time - releases/ snapshots, their SHA256SUMS and the pytest
# suite. If a file is missing (e.g. a deployment without releases/ or tests/), the card says "not available".
RELEASES = ROOT / "releases"
SNAPSHOTS = ("bench200_frozen", "candidate_v3", "candidate_v4")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def snapshot_integrity() -> dict:
    """Re-hash every file listed in each snapshot's SHA256SUMS (sha256sum format)."""
    out = {}
    for name in SNAPSHOTS:
        sums = RELEASES / name / "SHA256SUMS"
        if not sums.exists():
            out[name] = None
            continue
        files = bad = 0
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split(maxsplit=1)
            rel = rel.lstrip("*").strip()
            files += 1
            f = RELEASES / name / rel
            try:
                ok = hashlib.sha256(f.read_bytes()).hexdigest() == digest.lower()
            except OSError:
                ok = False
            bad += not ok
        out[name] = {"files": files, "bad": bad}
    return out


# The app never runs pytest (no subprocesses on a small CPU container). This is the result of the last full
# `pytest -q` run, verified by hand - update it together with the test suite.
VERIFIED_TESTS = {"passed": 267, "verified": "25 Nov 2026"}


def benchmark_figures() -> dict:
    import re
    frozen = _read_json(RELEASES / "bench200_frozen/baseline_results/bench200_report/summary.json") or {}
    fsys = (frozen.get("systems") or {}).get("nemotron+nano") or {}
    v4 = {s: (_read_json(RELEASES / f"candidate_v4/results/{s}_v4_vs_v3.json") or {}) for s in ("bench200", "set50", "holdout20")}
    report = RELEASES / "bench200_frozen/baseline_results/bench200_report.md"
    status = None
    if report.exists():
        m = re.search(r"\*\*Status: (\w+)\.\*\* (\d+) of (\d+) labels are human-verified", report.read_text(encoding="utf-8"))
        status = m and {"status": m.group(1), "verified": int(m.group(2)), "total": int(m.group(3))}
    return {"frozen": fsys, "frozen_ads": frozen.get("labelled_ads"),
            "v4": {s: (d.get("overall") or {}).get("cand") or {} for s, d in v4.items()},
            "v4_ads": {s: d.get("ads") for s, d in v4.items()}, "labels": status}


STATS_BANNER_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html, body { margin:0; padding:0; background:transparent; overflow:hidden; }
body { font-family:-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing:antialiased; color:#F5F5F7; }
.grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; padding:1px; box-sizing:border-box; }
.tile { box-sizing:border-box; height:126px; padding:14px 14px 12px; border-radius:14px; background:#141416;
  border:1px solid rgba(255,255,255,.08); box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
  opacity:0; transform:translateY(6px); animation:in .7s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.tile:nth-child(2) { animation-delay:.06s } .tile:nth-child(3) { animation-delay:.12s } .tile:nth-child(4) { animation-delay:.18s }
@keyframes in { to { opacity:1; transform:none; } }
.k { margin:0; font-size:10.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#8E8E93;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.v { margin:8px 0 6px; font-size:26px; font-weight:600; letter-spacing:-0.02em; line-height:1.1; white-space:nowrap;
  font-variant-numeric:tabular-nums; }
.u { font-size:12px; font-weight:500; letter-spacing:0; color:#8E8E93; margin-left:3px; }
.s { margin:0; font-size:11.5px; line-height:1.45; color:#A1A1AA; overflow:hidden; display:-webkit-box;
  -webkit-line-clamp:2; -webkit-box-orient:vertical; }
@media (prefers-reduced-motion: reduce) { .tile { animation:none; opacity:1; transform:none; } }
</style></head><body><div class="grid">__TILES__</div>
<script>
(function () {
  var nums = document.querySelectorAll('[data-to]');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function show(el, x) {
    var d = +el.dataset.dec || 0;
    el.textContent = (el.dataset.pre || '') + x.toFixed(d) + (el.dataset.suf || '');
  }
  if (reduce) { nums.forEach(function (el) { show(el, +el.dataset.to); }); return; }
  var dur = 900, t0 = null;
  function frame(t) {
    if (t0 === null) t0 = t;
    var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);          // ease-out cubic
    nums.forEach(function (el) { show(el, +el.dataset.to * e); });
    if (p < 1) requestAnimationFrame(frame);
  }
  nums.forEach(function (el) { show(el, 0); });
  requestAnimationFrame(frame);
})();
</script></body></html>"""


def _stat_tile(key: str, value: float | None, unit: str, sub: str, dec: int = 0, suffix: str = "",
               tip: str | None = None) -> str:
    """One banner tile; the number counts up from 0. A missing figure shows a dash (never a guessed number)."""
    e = html.escape
    num = (f'<span data-to="{value}" data-dec="{dec}" data-suf="{e(suffix)}">{value:.{dec}f}{e(suffix)}</span>'
           if value is not None else '<span style="color:#636366">—</span>')
    return (f'<div class="tile" title="{e(tip or sub)}"><p class="k">{e(key)}</p><p class="v">{num}'
            f'<span class="u">{e(unit)}</span></p><p class="s">{e(sub)}</p></div>')


def verification_card() -> None:
    """Headline numbers - all read from releases/ (except the verified test count, see VERIFIED_TESTS)."""
    rules = "candidate_v4" if os.environ.get("ADREADER_RULES_V4") == "1" else (
        "candidate_v3" if os.environ.get("ADREADER_RULES_V3", "1") != "0" else "frozen")
    b = benchmark_figures()
    integ = snapshot_integrity()
    f, v = b["frozen"], b["v4"]
    num = lambda d, k: d.get(k) if isinstance(d.get(k), (int, float)) else None
    fmt_pct = lambda d, k: f"{d[k]:.1f}%" if isinstance(d.get(k), (int, float)) else "—"
    present = {k: x for k, x in integ.items() if x}
    valid = sum(1 for x in present.values() if x["bad"] == 0)
    short = {"bench200_frozen": "bench200", "candidate_v3": "v3", "candidate_v4": "v4"}
    tiles = "".join([
        _stat_tile("Automated test suite", VERIFIED_TESTS["passed"], "tests",
                   "verified via pytest · unit, verifier, DB & RAG",
                   tip=f"{VERIFIED_TESTS['passed']} passed in the last full pytest run ({VERIFIED_TESTS['verified']})"),
        _stat_tile("Bench200 accuracy", num(v["bench200"], "accuracy_pct"), "",
                   f"{rules} · set50 {fmt_pct(v['set50'], 'accuracy_pct')} · holdout20 {fmt_pct(v['holdout20'], 'accuracy_pct')}",
                   dec=1, suffix="%"),
        _stat_tile("Confirmed precision", num(v["bench200"], "confirmed_precision_pct"), "",
                   f"silent errors cut from {f.get('silent_errors', '?')} → {v['bench200'].get('silent_errors', '?')}",
                   dec=1, suffix="%"),
        _stat_tile("Snapshot integrity", valid if present else None, "",
                   "checksum-valid · " + ", ".join(short.get(k, k) + ("" if x["bad"] == 0 else f" ({x['bad']} changed)")
                                                   for k, x in present.items()), suffix=f"/{len(SNAPSHOTS)}"),
    ])
    lab = b["labels"]
    caveat = (f'Benchmark labels are <b>{lab["status"].lower()}</b>: {lab["verified"]} of {lab["total"]} human-verified '
              f'(drafted from the images, pending human audit). Bench200 baseline before candidate_v4: '
              f'{fmt_pct(f, "accuracy_pct")}. candidate_v4 figures are a development retest on already-examined data - '
              'not independent proof.' if lab else
              'Benchmark labels have not all been human-verified; treat accuracy as provisional.')
    with st.container(key="vb_card"):
        st.markdown('<p class="vb-title">Automated verification &amp; benchmarks</p>'
                    '<p class="vb-intro">The records below are manual, human-in-the-loop saves. The extraction pipeline, '
                    'verifier rules and database matching behind them are measured separately against frozen benchmark '
                    'snapshots.</p>', unsafe_allow_html=True)
        components.html(STATS_BANNER_HTML.replace("__TILES__", tiles), height=130, scrolling=False)
        st.markdown(f'<p class="vb-caveat">{caveat}</p>', unsafe_allow_html=True)


def page_saved():
    verification_card()
    st.subheader("Saved (human-verified) records")
    with db.connect() as conn:
        rows = conn.execute("SELECT id, image, reviewed_by, created_at, record_json FROM submissions ORDER BY id DESC").fetchall()
    if not rows:
        st.info("Nothing saved yet.")
        return
    table = []
    for r in rows:
        final = json.loads(r["record_json"])["final"]
        f = final["fields"]
        table.append({"#": r["id"], "image": r["image"], "brand": f.get("brand", {}).get("value"),
                      "brand_id": f.get("brand", {}).get("id"), "product": f.get("product", {}).get("value"),
                      "price": f.get("price", {}).get("value"), "corrections": len(final["corrections"]),
                      "reviewer": r["reviewed_by"], "saved": r["created_at"]})
    html_table(pd.DataFrame(table))
    pick = st.selectbox("Show record", [r["id"] for r in rows])
    row = next(r for r in rows if r["id"] == pick)
    final = json.loads(row["record_json"])["final"]
    f = final["fields"]
    val = lambda k: f.get(k, {}).get("value")
    did = lambda k: f.get(k, {}).get("id")
    items = [("Brand", val("brand"), did("brand")), ("Product", val("product"), did("product")),
             ("Advertiser", val("advertiser"), did("advertiser")), ("Price", val("price"), None),
             ("Category", val("category"), did("category")), ("Language", val("language"), None)]
    cells = "".join(
        f'<div class="vr-cell"><p class="vb-k">{k}</p><p class="vr-v">{html.escape(fmt(v)) if v not in (None, "", []) else "—"}'
        f'{f" <span class=vr-id>#{i}</span>" if i is not None else ""}</p></div>' for k, v, i in items)
    st.markdown(
        f'<div class="rc-card vr-card"><div class="vr-head"><p class="rc-h">🟢 Record #{row["id"]} · {html.escape(str(row["image"]))}</p>'
        f'<p class="vr-meta">Verified by <b>{html.escape(str(row["reviewed_by"]))}</b> · {html.escape(str(row["created_at"]))} · '
        f'{len(final.get("corrections", []))} correction(s) vs the system</p></div><div class="vr-grid">{cells}</div></div>',
        unsafe_allow_html=True)
    with st.expander("View raw JSON", expanded=False):
        st.json(final)


def page_audit():
    from adreader import audit
    st.subheader("Audit the drafted correct answers (random sample of 10)")
    if not (audit.SET_DIR / "manifest.csv").is_file():   # evaluation images/labels are not shipped with every deployment
        st.info("Audit data is not included in this deployment (the evaluation ad images stay local). "
                "Run the app from a full local checkout to audit the drafted answers.")
        return
    st.caption("Claude drafted these answers by reading each ad image (not from the system's output). "
               "Check each field against the image and correct anything wrong. Alternatives are separated by "
               f"`{audit.ALT_SEP.strip()}`; `{audit.NOT_PRINTED}` means the ad does not print it. "
               "Every change is recorded and reported as the label error rate.")
    items = audit.sample()
    s = audit.summary()
    st.info(f"Audited {s['audited_ads']}/{len(items)} ads · fields changed: {s['fields_changed']} of "
            f"{s['fields_checked']} checked" + (f" ({s['label_field_error_rate_pct']}%)" if s['fields_checked'] else ""))
    labels = {i["id"]: json.loads((ROOT / "data/labels" / f"{i['stem']}.json").read_text(encoding="utf-8")) for i in items}
    pick = st.radio("Ad", [i["id"] for i in items], horizontal=True,
                    format_func=lambda i: f"{i} {'✅' if labels[i].get('_audited_by') else '⬜'}", key="audit_pick")
    item = next(i for i in items if i["id"] == pick)
    lab = labels[pick]
    left, right = st.columns([0.5, 0.5])
    left.image(str(ROOT / item["file"]), width="stretch")
    with right:
        edited = {}
        for f in audit.FIELDS:
            skipped = f in lab.get("_skip", []) or f not in lab
            val = "" if skipped else audit.to_text(lab.get(f))
            edited[f] = st.text_input(f + ("  (not scored - fill only if you want it scored)" if skipped else ""),
                                      value=val, key=f"aud:{pick}:{f}")
        edited = {f: v for f, v in edited.items() if v.strip() or (f in lab and f not in lab.get("_skip", []))}
        who = st.text_input("Auditor name", key="auditor")
        if st.button("✅ Save audit for this ad", type="primary", disabled=not who.strip(), key=f"aud:{pick}:save"):
            r = audit.save_audit(item["stem"], edited, who.strip())
            st.success(f"Saved - {r['changed']} of {r['checked']} fields changed.")
            st.rerun()
        if lab.get("_audit_changes"):
            st.caption("Recorded changes: " + "; ".join(f"{c['field']}" for c in lab["_audit_changes"]))


JOB_STAGE_SHORT = {"queued": "waiting", "ocr": "reading text", "extract": "vision extraction", "verify": "verification",
                   "validate": "DB matching", "done": "done"}


def _usage_section() -> None:
    with st.expander("Provider usage", expanded=False):
        u = usage.summary()
        if not u:
            st.caption("No model calls logged yet.")
            return
        m1, m2, m3, m4 = st.columns(4)
        calls, fails = sum(r["calls"] for r in u), sum(r["failures"] for r in u)
        m1.metric("Model calls", f"{calls:,}")
        m2.metric("Failures", f"{fails:,}", help=f"{(100 * fails / calls if calls else 0):.1f}% of calls")
        m3.metric("Tokens (in / out)", f"{sum(r['tokens_in'] for r in u) / 1e6:.2f}M / {sum(r['tokens_out'] for r in u) / 1e6:.2f}M")
        m4.metric("Cost", f"${sum(r['cost_usd'] or 0 for r in u):,.2f}")
        html_table(pd.DataFrame([{
            "provider": r["provider"], "role": r["role"], "model": r["model"], "calls": r["calls"],
            "failures": r["failures"], "failure %": r["failure_rate_pct"], "p50 s": r["latency_p50_s"],
            "p95 s": r["latency_p95_s"], "tokens in": r["tokens_in"], "tokens out": r["tokens_out"],
            "cost $": r["cost_usd"], "errors": ", ".join(f"{k} {v}" for k, v in (r.get("errors") or {}).items())}
            for r in u]))


def page_jobs():
    head, refresh = st.columns([5, 1], vertical_alignment="center")
    head.header("Processing jobs")
    if refresh.button("🔄 Refresh", key="jobs_refresh", width="stretch"):
        st.rerun()
    rd = providers.readiness()
    alive = jobs.workers_alive() or (["embedded"] if worker_running() else [])

    # ---- provider / worker status card + setup warnings
    with st.container(key="jobs_status"):
        c1, c2 = st.columns([1, 1.6], gap="large", vertical_alignment="center")
        chain = ", ".join(rd["active_chain"]) or "none"
        c1.markdown(
            '<div class="jb-stats">'
            f'<div><p class="jb-k">Active providers</p><p class="jb-v">{html.escape(chain)}</p></div>'
            f'<div><p class="jb-k">Background workers</p><p class="jb-v"><span class="jb-dot {"on" if alive else "off"}"></span>'
            f'{len(alive)} running</p></div></div>', unsafe_allow_html=True)
        notes = rd["warnings"] + rd["skipped"]
        c2.markdown('<div class="jb-warn">' + ("".join(f'<p>⚠ {html.escape(str(w))}</p>' for w in notes)
                                                 or '<p class="jb-okline">Provider setup looks fine.</p>') + '</div>',
                    unsafe_allow_html=True)

    rows = jobs.list_jobs(200)
    if not rows:
        st.info("No jobs yet - process an advertisement and it will appear here.")
        _usage_section()
        return

    # ---- status summary strip
    counts = {s: sum(r["status"] == s for r in rows) for s in STATUS_ICON}
    st.markdown('<div class="jb-strip">' + "".join(
        f'<span class="jb-pill{" zero" if not n else ""}">{STATUS_ICON[s]} {s} <b>{n}</b></span>'
        for s, n in counts.items()) + '</div>', unsafe_allow_html=True)

    # ---- jobs table (latest 200, newest first)
    # Read-only table as plain dark HTML (no canvas grid): clicking a row can never paint a light selection
    # highlight, whatever theme the browser has cached.
    e = lambda v: html.escape(str(v))
    body = "".join(
        f'<tr class="jt-{e(r["status"])}"><td class="jt-num">{r["id"]}</td>'
        f'<td>{STATUS_ICON[r["status"]]} {e(r["status"])}</td>'
        f'<td>{e(JOB_STAGE_SHORT.get(r["stage"] or "", r["stage"] or ""))}</td>'
        f'<td class="jt-num">{r["attempts"]}</td><td class="jt-ad" title="{e(r["name"])}">{e(r["name"])}</td>'
        f'<td class="jt-err" title="{e(r["last_error"] or "")}">{e((r["last_error"] or "")[:120])}</td>'
        f'<td class="jt-time">{time.strftime("%d %b %H:%M", time.localtime(r["created_at"]))}</td></tr>'
        for r in rows)
    st.markdown('<div class="jt-wrap"><table class="jt"><thead><tr><th>job</th><th>status</th><th>stage</th>'
                '<th>attempts</th><th>ad</th><th>error</th><th>created</th></tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)

    # ---- actions
    done = [r for r in rows if r["status"] == "completed"]
    failed = [r for r in rows if r["status"] == "failed"]
    active = [r["id"] for r in rows if r["status"] in jobs.ACTIVE]
    label = {r["id"]: f"#{r['id']} · {r['name']}" for r in rows}
    a1, a2 = st.columns(2, gap="large")
    with a1:
        with st.container(key="jobs_open_card"):
            st.markdown('<p class="jb-h">Open a completed job for review</p>'
                        '<p class="jb-t">Field overview, editing tabs, OCR boxes, the data-entry form and Save.</p>',
                        unsafe_allow_html=True)
            pick = st.selectbox("Completed job", [r["id"] for r in done], index=None, key="jobs_open",
                                placeholder="Choose a completed job…" if done else "No completed jobs yet",
                                format_func=lambda i: label.get(i, f"#{i}"), label_visibility="collapsed")
    with a2:
        with st.container(key="jobs_retry_card"):
            st.markdown('<p class="jb-h">Retry a failed or cancelled job</p>'
                        '<p class="jb-t">Resumes from the step that failed - finished steps are kept.</p>',
                        unsafe_allow_html=True)
            if failed:
                r1, r2 = st.columns([3, 1], vertical_alignment="bottom")
                f = r1.selectbox("Failed job", [r["id"] for r in failed], key="jobs_failed",
                                 format_func=lambda i: label.get(i, f"#{i}"), label_visibility="collapsed")
                if r2.button("🔁 Retry", key="jobs_retry", width="stretch"):
                    jobs.retry(f)
                    st.rerun()
            else:
                st.caption("No failed jobs.")
            if active and not alive:            # no worker (e.g. Hugging Face Spaces): run them in this session
                st.markdown('<p class="jb-h jb-gap">Process waiting jobs here</p>'
                            '<p class="jb-t">No background worker is running.</p>', unsafe_allow_html=True)
                if st.button(f"▶ Process {len(active)} waiting job(s) in this session", key="jobs_run_here",
                             type="primary"):
                    run_here(active)
                    st.rerun()

    if pick is not None:
        res = job_result(pick)
        if res is not None:
            st.markdown('<div class="jb-sep"></div>', unsafe_allow_html=True)
            review_screen(res, rid=f"job{pick}")
        else:
            st.warning(f"Job #{pick} has no saved result to open.")
    _usage_section()


DBA_FIELD = {"brand": "Brand", "product": "Product", "advertiser": "Advertiser", "category": "Category"}


@st.cache_data(ttl=120, show_spinner=False)
def unmatched_from_jobs(signature: tuple) -> list[dict]:
    """Values in completed jobs that the database lookup could not link - read from each job's saved record.json
    (no model call, nothing written). `signature` = ((job id, updated_at), ...) so the cache follows new jobs."""
    from adreader.schema import AdRecord
    out = []
    for jid, _ in signature:
        f = jobs.job_dir(jid) / "record.json"
        try:
            rec = AdRecord.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for m in db_link.match_report(rec):
            if m["outcome"] in ("unresolved", "near_miss"):
                out.append({"job": jid, "ad": rec.image, "field": m["field"], "value": fmt(m["value"]),
                            "outcome": m["outcome"], "similar": [c["name"] for c in m["candidates"]][:2]})
    return out


def _master_counts(conn) -> dict:
    counts = {}
    for t in ("brands", "products", "advertisers", "categories"):
        try:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            counts[t] = None
    return counts


def page_master_additions():
    st.header("Database additions")
    st.caption("Master records created by reviewers (never by the system). Undo is possible while no "
               "saved record uses them.")
    with db.connect() as conn:
        rows = md.list_additions(conn)
        counts = _master_counts(conn)

        # ---- master database at a glance (real counts)
        tile = lambda k, v, s: (f'<div class="vb-tile"><p class="vb-k">{k}</p><p class="vb-v">{v}</p>'
                                f'<p class="vb-s">{s}</p></div>')
        n = lambda t: "—" if counts.get(t) is None else f"{counts[t]:,}"
        st.markdown(
            '<div class="rc-card vb-card"><p class="rc-h">Master database</p>'
            '<p class="rc-t">The records every extracted brand, product, advertiser and category is matched against. '
            'IDs only ever come from these rows - the AI never invents or adds one.</p><div class="vb-grid">'
            + tile("Brands", n("brands"), "pre-loaded master list") + tile("Products", n("products"), "linked to their brand")
            + tile("Advertisers", n("advertisers"), "clients &amp; dealers") + tile("Reviewer additions", f"{len(rows)}",
                                                                              "added by people while saving")
            + '</div></div>', unsafe_allow_html=True)

        # ---- reviewer additions (audit trail) or a clear empty state
        if rows:
            html_table(pd.DataFrame(rows)[["id", "entity", "entity_id", "name", "added_by", "source_image", "created_at"]])
            pick = st.selectbox("Undo addition", [r["id"] for r in rows],
                                format_func=lambda i: next(f"#{r['id']} {r['entity']} “{r['name']}”" for r in rows if r["id"] == i))
            if st.button("↩️ Undo selected addition"):
                try:
                    md.undo_addition(conn, pick)
                    matcher.clear()
                    master_options.clear()
                    st.success("Removed.")
                    st.rerun()
                except md.MasterDataError as err:
                    st.error(str(err))
        else:
            st.markdown(
                '<div class="rc-card dba-empty"><p class="rc-h">No reviewer additions yet</p>'
                '<p class="rc-t">Every saved record so far linked to brands and products that already exist. A new entry '
                'appears here only when a reviewer decides a value is genuinely new:</p>'
                '<ol class="dba-steps"><li>Open a processed ad with a <b>🆕 Unmatched</b> value (see the list below).</li>'
                '<li>In the <b>Identity</b> tab, choose to add it as a new record.</li>'
                '<li>Clear the 🟠 flags, enter your name and click <b>💾 Save verified record</b>.</li></ol>'
                '<p class="rc-n">The addition is logged with who added it and from which ad, and can be undone while no '
                'saved record uses it.</p></div>', unsafe_allow_html=True)

    # ---- values in processed ads that are waiting for a reviewer's decision
    done = tuple((r["id"], r["updated_at"]) for r in jobs.list_jobs(200) if r["status"] == "completed")
    items = unmatched_from_jobs(done)
    st.subheader("Waiting for a reviewer")
    if not items:
        st.caption("Every value in the processed ads is linked to an existing database record.")
        return
    new = sum(1 for i in items if i["outcome"] == "unresolved")
    st.caption(f"{len(items)} value(s) in {len({i['job'] for i in items})} processed ad(s) could not be linked: "
               f"{new} are not in the database at all, {len(items) - new} have a similar existing record "
               "(pick that one instead of adding a duplicate). Nothing is added until a reviewer decides.")
    html_table(pd.DataFrame([{
        "job": i["job"], "ad": i["ad"], "field": DBA_FIELD.get(i["field"], i["field"]), "extracted value": i["value"],
        "database": ("🆕 not in database" if i["outcome"] == "unresolved"
                     else "🟠 similar exists: " + ", ".join(i["similar"]))} for i in items]))
    jids = sorted({i["job"] for i in items}, reverse=True)
    label = {i["job"]: f"#{i['job']} · {i['ad']}" for i in items}
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    jid = c1.selectbox("Open a processed ad to review", jids, format_func=lambda j: label.get(j, f"#{j}"),
                       key="dba_job")
    c2.button("Open job to review →", key="dba_open", on_click=_open_job, args=(jid,), width="stretch")


# Smooth one-time scroll to the results (zero-height component; same-origin parent document)
SCROLL_TO_RESULTS_JS = """<script>
setTimeout(function () {
  var el = window.parent.document.getElementById('extraction-result-anchor');
  if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
}, 250);
</script>"""

# Primary actions (Extract & verify, Process selected ads, Find ads, Save, Search) as a macOS graphite pill
# instead of Streamlit's default red - readable on both the light and the dark theme.
PRIMARY_BUTTON_CSS = """
<style>
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background-color:#2C2C2E !important; border:1px solid rgba(255,255,255,.14) !important; color:#F5F5F7 !important;
  border-radius:9px !important; box-shadow:0 1px 2px rgba(0,0,0,.25) !important;
  transition:background-color .15s ease, transform .15s ease !important; }
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color:#F5F5F7 !important; font-weight:500 !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background-color:#3A3A3C !important; transform:translateY(-1px); }
button[kind="primary"]:active, [data-testid="stBaseButton-primary"]:active { background-color:#1C1C1E !important; transform:none; }
button[kind="primary"]:focus-visible, [data-testid="stBaseButton-primary"]:focus-visible { outline:2px solid #8E8E93 !important; outline-offset:2px; }
button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled { opacity:.45 !important; transform:none; }
</style>
"""


def main():
    sys.path.insert(0, str(ROOT / "app"))
    import landing                                     # animated welcome screen (app/landing.py)
    if not landing.gate():
        return
    ensure_worker()                                    # no separate worker process (cloud hosts)? run one in-process
    with st.container(key="theme_reset_js"):           # zero-size helper; see RESET_CACHED_THEME_JS
        components.html(RESET_CACHED_THEME_JS, height=0)
    st.session_state.setdefault("edits", {})
    import mascot                                      # Indy: page-aware tips (app/mascot.py)
    import theme                                       # macOS sidebar styling (app/theme.py)
    theme.apply_sidebar()
    st.sidebar.markdown(PRIMARY_BUTTON_CSS.strip() + '\n<div class="sb-css"></div>', unsafe_allow_html=True)
    landing.home_button()
    # brand header; "Ad Data Entry" stays in the title because tests/test_app.py checks for it
    st.sidebar.title("Indycium Ad-Reader :gray[Ad Data Entry]")
    pages = ["Process advertisement", "Jobs", "Saved records", "Database additions", "Audit answers", "Search & RAG"]
    mascot.sidebar_assistant(st.session_state.get("page", pages[0]))
    page = st.sidebar.radio("Page", pages, key="page")
    cfg = vlm_config()
    rules = "candidate_v4" if os.environ.get("ADREADER_RULES_V4") == "1" else (
        "candidate_v3" if os.environ.get("ADREADER_RULES_V3", "1") != "0" else "frozen")
    try:
        if page == "Process advertisement":
            page_process()
        elif page == "Jobs":
            page_jobs()
        elif page == "Saved records":
            page_saved()
        elif page == "Database additions":
            page_master_additions()
        elif page == "Search & RAG":                   # Phase 3 MVP - separate module (app/rag_page.py)
            sys.path.insert(0, str(ROOT / "app"))
            import rag_page
            rag_page.render()
        else:
            page_audit()
    finally:                                           # model / rules metadata sits at the very bottom of the sidebar
        theme.meta_footer(cfg.model or "not configured", cfg.backend, rules)


main()
