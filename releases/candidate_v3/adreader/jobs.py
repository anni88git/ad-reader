"""Persistent, recoverable processing jobs.

* enqueue() copies the uploaded image into data/jobs/<id>/ before anything else happens, so a
  browser refresh, an API failure or a crash never loses the ad.
* Each stage writes a checkpoint (ocr.pkl, raw.json, verifier.json, record.json). A retry resumes
  at the first missing checkpoint - a model call that already succeeded is never paid for twice.
* Statuses: queued -> processing -> completed | retrying (automatic, bounded, with backoff) | failed.
  Failed jobs are never deleted; retry() puts them back in the queue.
* The extraction / validation / linking code is the unchanged pipeline code.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ocr as ocr_mod
from . import opsdb, usage
from . import preprocess as pp
from .config import DATA_DIR
from .extract import extract
from .matcher import link_records
from .schema import RawExtraction
from .segment import AdRegion
from .validate import validate
from .verify import verify
from .vlm import ProviderError, VLMClient

JOBS_DIR = DATA_DIR / "jobs"
BACKOFF_S = [30, 120, 600]          # wait before automatic retry 1, 2, 3
MAX_ATTEMPTS = 4
STALE_S = 180                        # processing job without heartbeat for this long -> recovered
ACTIVE = ("queued", "processing", "retrying")


# ------------------------------------------------------------------------------------ queue API
def _dedupe_key(data: bytes, options: dict) -> str:
    return hashlib.sha256(data + json.dumps(options, sort_keys=True).encode()).hexdigest()[:32]


def enqueue(data: bytes, name: str, options: dict | None = None, conn=None) -> int:
    """Store the image and create a job. The same image + options returns the existing job
    (so a refresh or a double click does not create duplicates)."""
    c = conn or opsdb.connect()
    options = options or {}
    key = _dedupe_key(data, options)
    row = c.execute("SELECT id FROM jobs WHERE dedupe_key = ? ORDER BY id DESC LIMIT 1", (key,)).fetchone()
    if row:
        return row["id"]
    now = time.time()
    suffix = Path(name).suffix.lower() if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") else ".png"
    cur = c.execute("INSERT INTO jobs(status, stage, name, input_path, dedupe_key, options, max_attempts, "
                    "created_at, updated_at) VALUES ('queued', 'queued', ?, '', ?, ?, ?, ?, ?)",
                    (name, key, json.dumps(options), MAX_ATTEMPTS, now, now))
    jid = cur.lastrowid
    d = job_dir(jid)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"input{suffix}"
    p.write_bytes(data)
    c.execute("UPDATE jobs SET input_path = ? WHERE id = ?", (str(p), jid))
    return jid


def find(data: bytes, options: dict | None = None, conn=None) -> int | None:
    """Existing job for this exact image + options, if any (without creating one)."""
    r = (conn or opsdb.connect()).execute("SELECT id FROM jobs WHERE dedupe_key = ? ORDER BY id DESC LIMIT 1",
                                          (_dedupe_key(data, options or {}),)).fetchone()
    return r["id"] if r else None


def job_dir(jid: int) -> Path:
    return JOBS_DIR / f"{jid:06d}"


def get(jid: int, conn=None) -> dict | None:
    r = (conn or opsdb.connect()).execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
    return _row(r) if r else None


def list_jobs(limit: int = 100, conn=None) -> list[dict]:
    return [_row(r) for r in (conn or opsdb.connect()).execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]


def _row(r) -> dict:
    d = dict(r)
    d["options"] = json.loads(d.get("options") or "{}")
    return d


def retry(jid: int, conn=None) -> None:
    """Put a failed job back in the queue (manual retry). Checkpoints are kept."""
    c = conn or opsdb.connect()
    c.execute("UPDATE jobs SET status='queued', attempts=0, next_attempt_at=NULL, error_kind=NULL, "
              "updated_at=? WHERE id=? AND status='failed'", (time.time(), jid))


def claim(worker: str, conn=None, job_id: int | None = None) -> dict | None:
    """Atomically take the next due job (or the given job, if it is waiting). Safe with several workers.
    A specific job is taken even if its retry back-off has not elapsed ("retry now")."""
    c = conn or opsdb.connect()
    now = time.time()
    c.execute("BEGIN IMMEDIATE")
    try:
        if job_id is not None:
            r = c.execute("SELECT id FROM jobs WHERE id = ? AND status IN ('queued','retrying')", (job_id,)).fetchone()
        else:
            r = c.execute("SELECT id FROM jobs WHERE status IN ('queued','retrying') AND "
                          "(next_attempt_at IS NULL OR next_attempt_at <= ?) ORDER BY id LIMIT 1", (now,)).fetchone()
        if not r:
            c.execute("COMMIT")
            return None
        c.execute("UPDATE jobs SET status='processing', attempts=attempts+1, worker=?, heartbeat_at=?, "
                  "updated_at=? WHERE id=?", (worker, now, now, r["id"]))
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    return get(r["id"], c)


def recover_stale(conn=None, stale_s: float = STALE_S) -> int:
    """Jobs left 'processing' by a crashed worker go back to the queue."""
    c = conn or opsdb.connect()
    now = time.time()
    cur = c.execute("UPDATE jobs SET status='retrying', next_attempt_at=?, last_error=COALESCE(last_error,'') || "
                    "' [worker stopped during processing - recovered]', updated_at=? "
                    "WHERE status='processing' AND heartbeat_at < ?", (now, now, now - stale_s))
    return cur.rowcount


def _set(c, jid: int, **kw) -> None:
    kw["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in kw)
    c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*kw.values(), jid))


# ------------------------------------------------------------------------------------ runner
@dataclass
class JobOutput:
    record: object
    raw: RawExtraction
    ocr: object
    image: np.ndarray
    region: AdRegion | None
    verifier_raw: RawExtraction | None


def _image_for(job: dict) -> tuple[np.ndarray, AdRegion | None]:
    img = pp.load_image(Path(job["input_path"]).read_bytes())
    reg = job["options"].get("region")
    if reg:
        region = AdRegion(job["options"].get("region_label", "ad"), tuple(int(x) for x in reg))
        return region.crop(img), region
    return img, None


class _Heartbeat:
    def __init__(self, jid: int, every: float = 15):
        self.jid, self.every, self._stop = jid, every, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        c = opsdb.connect()
        while not self._stop.wait(self.every):
            c.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (time.time(), self.jid))
        c.close()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()


def run_job(job: dict, client_factory=VLMClient, conn=None, final_attempt: bool = False) -> JobOutput:
    """Run (or resume) one job. Raises ProviderError for model-call failures."""
    c = conn or opsdb.connect()
    jid, opt, d = job["id"], job["options"], job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    img, region = _image_for(job)
    client = None
    readonly = job["status"] == "completed"        # re-opening a finished job: no status writes

    def stage(name):
        if not readonly:
            _set(c, jid, stage=name)

    def get_client():
        nonlocal client
        if client is None:
            client = client_factory()
        return client

    token = usage.current_job.set(jid)
    try:
        # 1. OCR (local CPU) - checkpoint
        f_ocr = d / "ocr.pkl"
        if f_ocr.exists():
            ocr = pickle.loads(f_ocr.read_bytes())
        elif opt.get("ocr_cache") and Path(opt["ocr_cache"]).exists():
            ocr = pickle.loads(Path(opt["ocr_cache"]).read_bytes())
            f_ocr.write_bytes(pickle.dumps(ocr))
        else:
            stage("ocr")
            ocr = ocr_mod.run_ocr(img, langs=tuple(opt.get("langs") or ocr_mod.DEFAULT_LANGS))
            f_ocr.write_bytes(pickle.dumps(ocr))
        # 2. extraction (model call) - checkpoint
        f_raw = d / "raw.json"
        if f_raw.exists():
            rj = json.loads(f_raw.read_text(encoding="utf-8"))
        elif opt.get("raw_cache") and Path(opt["raw_cache"]).exists():
            cached = json.loads(Path(opt["raw_cache"]).read_text(encoding="utf-8"))
            rj = {"raw": cached["raw"], "model": cached.get("model", "?") + " (cached)", "seconds": 0}
            f_raw.write_text(json.dumps(rj, ensure_ascii=False), encoding="utf-8")
        else:
            stage("extract")
            run = extract(img, ocr, client=get_client())
            rj = {"raw": run.raw.model_dump(), "model": run.model, "seconds": run.seconds,
                  "tokens_in": run.tokens_in, "tokens_out": run.tokens_out}
            f_raw.write_text(json.dumps(rj, ensure_ascii=False), encoding="utf-8")
        raw = RawExtraction.model_validate(rj["raw"])
        # 3. independent verifier (model call) - checkpoint. A transient failure retries the job
        #    (extraction is not repeated); on the final attempt the record is made without it,
        #    exactly as the pipeline always did (the record notes it, validation is stricter).
        f_ver = d / "verifier.json"
        ver, models = None, [rj["model"]]
        want_verifier = opt.get("verifier", True)
        if f_ver.exists():
            vj = json.loads(f_ver.read_text(encoding="utf-8"))
        elif want_verifier:
            stage("verify")
            cl = get_client()
            if not cl.cfg.verifier_model:
                vj = {"failed": "no verifier model configured"}
            else:
                try:
                    v = verify(img, cl, model=cl.cfg.verifier_model)
                    vj = {"raw": v.raw.model_dump(), "model": v.model, "seconds": v.seconds}
                except ProviderError as e:
                    if e.transient and not final_attempt:
                        raise
                    vj = {"failed": str(e)[:300]}
                except ValueError as e:                     # unparseable verifier reply
                    vj = {"failed": f"invalid verifier reply: {e}"[:300]}
            f_ver.write_text(json.dumps(vj, ensure_ascii=False), encoding="utf-8")
        else:
            vj = {"failed": "verifier disabled for this job"}
        if vj.get("raw"):
            ver = RawExtraction.model_validate(vj["raw"])
            models.append(f"verifier: {vj['model']}")
        else:
            models.append(f"verifier failed: {vj.get('failed')}")
        # 4. deterministic validation + master-data linking (unchanged pipeline code)
        stage("validate")
        from .pipeline import ocr_corpus_key
        rec = validate(raw, ocr, image=img, image_name=job["name"], verifier=ver)
        link_records(rec, ocr_corpus_key(ocr))
        rec.models_used = models
        rec.timings_s = {"extract": rj.get("seconds", 0)}
        (d / "record.json").write_text(rec.model_dump_json(indent=1), encoding="utf-8")
        return JobOutput(rec, raw, ocr, img, region, ver)
    finally:
        usage.current_job.reset(token)


def load_output(jid: int) -> JobOutput | None:
    """Artifacts of a completed job, re-validated with the current rules (as the app cache did)."""
    job = get(jid)
    if not job or job["status"] != "completed":
        return None
    return run_job(job, client_factory=_no_client)


def _no_client():
    raise RuntimeError("completed job is missing a checkpoint")


def process_next(worker: str, client_factory=VLMClient, conn=None, job_id: int | None = None) -> dict | None:
    """Claim one due job (or `job_id`) and run it. Returns the job row after processing, or None
    if nothing was claimable (idle, or the job is already being processed elsewhere)."""
    c = conn or opsdb.connect()
    job = claim(worker, c, job_id)
    if not job:
        return None
    jid = job["id"]
    final = job["attempts"] >= job["max_attempts"]
    try:
        with _Heartbeat(jid):
            run_job(job, client_factory, c, final_attempt=final)
        _set(c, jid, status="completed", stage="done", last_error=None, error_kind=None, completed_at=time.time())
    except ProviderError as e:
        if e.transient and not final:
            wait = BACKOFF_S[min(job["attempts"] - 1, len(BACKOFF_S) - 1)]
            _set(c, jid, status="retrying", next_attempt_at=time.time() + wait,
                 last_error=f"attempt {job['attempts']}: {e}"[:1000], error_kind=e.kind)
        else:
            _set(c, jid, status="failed", last_error=f"attempt {job['attempts']}: {e}"[:1000],
                 error_kind=e.kind if e.transient else "permanent")
    except Exception as e:      # unreadable image, configuration problem, bug: keep the job, stop retrying
        _set(c, jid, status="failed", last_error=f"{type(e).__name__}: {e}"[:1000],
             error_kind="config" if "provider" in str(e).lower() else "error")
    return get(jid, c)


def delete_job_files(jid: int) -> None:
    """Explicit clean-up only (never called automatically)."""
    shutil.rmtree(job_dir(jid), ignore_errors=True)


def worker_seen(name: str, pid: int, conn=None) -> None:
    (conn or opsdb.connect()).execute("INSERT INTO workers(name, pid, last_seen) VALUES (?,?,?) "
                                      "ON CONFLICT(name) DO UPDATE SET pid=excluded.pid, last_seen=excluded.last_seen",
                                      (name, pid, time.time()))


def workers_alive(within_s: float = 30, conn=None) -> list[dict]:
    return [dict(r) for r in (conn or opsdb.connect()).execute(
        "SELECT * FROM workers WHERE last_seen >= ?", (time.time() - within_s,))]
