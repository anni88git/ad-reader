"""Background worker: processes queued ad jobs one at a time.

  python scripts/worker.py                 # run until Ctrl+C (start 1-2 of these)
  python scripts/worker.py --once          # process at most one job, then exit
  python scripts/worker.py status          # jobs by status, workers, provider chain, usage
  python scripts/worker.py retry <job_id>  # put a failed job back in the queue
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ADREADER_RULES_V4", "1")    # same rules as the app (candidate_v4)
from adreader import jobs, opsdb, providers, usage  # noqa: E402


def status():
    c = opsdb.connect()
    counts = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
    print("jobs:", counts or "none")
    for j in c.execute("SELECT id, status, stage, attempts, name, error_kind, substr(last_error,1,120) e "
                       "FROM jobs WHERE status != 'completed' ORDER BY id DESC LIMIT 15"):
        print(f"  #{j['id']} {j['status']:10} stage={j['stage']} attempts={j['attempts']} {j['name']} "
              f"{j['error_kind'] or ''} {j['e'] or ''}")
    print("workers alive:", [w["name"] for w in jobs.workers_alive()] or "none")
    print("providers:", json.dumps(providers.readiness(), indent=1))
    print("usage (all time):")
    for u in usage.summary():
        print("  ", json.dumps(u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "status", "retry"])
    ap.add_argument("job", nargs="?", type=int)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        return status()
    if a.cmd == "retry":
        jobs.retry(a.job)
        return print("queued again:", jobs.get(a.job)["status"])
    name = f"{socket.gethostname()}-{os.getpid()}"
    c = opsdb.connect()
    print(f"worker {name} started; providers: {providers.readiness()['active_chain']}", flush=True)
    try:
        while True:
            jobs.worker_seen(name, os.getpid(), c)
            n = jobs.recover_stale(c)
            if n:
                print(f"recovered {n} stale job(s)", flush=True)
            j = jobs.process_next(name, conn=c)
            if j:
                print(f"#{j['id']} {j['name']}: {j['status']} (attempt {j['attempts']}) "
                      f"{(j['last_error'] or '')[:150]}", flush=True)
            if a.once:
                break
            if not j:
                time.sleep(2)
    except KeyboardInterrupt:
        print("stopping", flush=True)


if __name__ == "__main__":
    main()
