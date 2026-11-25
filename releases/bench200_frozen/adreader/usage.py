"""Per-call provider usage log: latency, tokens, cost, failures. One row per attempt.

Logging must never break inference: any logging error is swallowed (and counted).
The current job id is taken from a context variable set by the job runner.
"""
from __future__ import annotations

import contextvars
import statistics
import threading
import time

from . import opsdb

current_job: contextvars.ContextVar[int | None] = contextvars.ContextVar("current_job", default=None)
_log_errors = 0
_local = threading.local()        # one connection per thread (Streamlit runs sessions in threads)


def _db():
    if getattr(_local, "conn", None) is None:
        _local.conn = opsdb.connect()
    return _local.conn


def log(provider: str, model: str, role: str, ok: bool, *, error_type: str | None = None,
        http_status: int | None = None, latency_s: float | None = None, tokens_in: int | None = None,
        tokens_out: int | None = None, cost_usd: float | None = None, conn=None) -> None:
    global _log_errors
    try:
        (conn or _db()).execute(
            "INSERT INTO usage(ts, provider, model, role, job_id, ok, error_type, http_status, latency_s, "
            "tokens_in, tokens_out, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), provider, model, role, current_job.get(), int(ok), error_type, http_status,
             latency_s, tokens_in, tokens_out, cost_usd))
    except Exception:
        _log_errors += 1


def _pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(q * len(v)))], 1)


def summary(since_s: float | None = None, conn=None) -> list[dict]:
    """Per provider + model: calls, failures, failure rate, latency p50/p95, tokens, cost."""
    c = conn or _db()
    q = "SELECT * FROM usage" + (" WHERE ts >= ?" if since_s else "")
    rows = [dict(r) for r in c.execute(q, (time.time() - since_s,) if since_s else ())]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["provider"], r["model"], r["role"]), []).append(r)
    out = []
    for (prov, model, role), rs in sorted(groups.items()):
        ok = [r for r in rs if r["ok"]]
        lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
        errs: dict[str, int] = {}
        for r in rs:
            if not r["ok"]:
                errs[r["error_type"] or "?"] = errs.get(r["error_type"] or "?", 0) + 1
        out.append({"provider": prov, "model": model, "role": role, "calls": len(rs), "ok": len(ok),
                    "failures": len(rs) - len(ok), "failure_rate_pct": round(100 * (len(rs) - len(ok)) / len(rs), 1),
                    "errors": errs, "latency_p50_s": round(statistics.median(lat), 1) if lat else None,
                    "latency_p95_s": _pct(lat, 0.95),
                    "tokens_in": sum(r["tokens_in"] or 0 for r in ok), "tokens_out": sum(r["tokens_out"] or 0 for r in ok),
                    "cost_usd": round(sum(r["cost_usd"] or 0 for r in ok), 4)})
    return out
