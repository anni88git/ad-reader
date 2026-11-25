"""Reliability layer: provider chain, retries, usage logging, persistent recoverable jobs.
All model calls are faked - no network, no cost."""
import json
import pickle
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from adreader import jobs, opsdb, providers, usage, vlm
from adreader.ocr import OCRLine, OCRResult, normalize_text, script_of
from adreader.preprocess import ImageQuality
from adreader.providers import Provider
from adreader.vlm import AllProvidersFailed, ProviderError, VLMClient

EXTRACT_JSON = json.dumps({"brand": {"value": "Kia", "evidence": "Kia"},
                           "headline": {"value": "Superior tech", "evidence": "Superior tech"}})
VERIFY_JSON = json.dumps({"brand": "Kia", "headline": "Superior tech"})


class HTTPErr(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.status_code = code


class APITimeoutError(Exception):
    pass


def _reply(text, tin=1000, tout=200):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))


class FakeAPI:
    """Scripted behaviour per (provider, key index): list of outcomes, last one repeats."""
    def __init__(self, script):
        self.script, self.calls = script, []

    def for_client(self, name, key_idx):
        def create(model, messages, **kw):
            self.calls.append((name, key_idx, model))
            seq = self.script.get((name, key_idx)) or self.script.get(name) or ["ok"]
            n = sum(1 for c in self.calls if c[:2] == (name, key_idx))
            out = seq[min(n - 1, len(seq) - 1)]
            if isinstance(out, Exception):
                raise out
            text = VERIFY_JSON if "verifier" in model else EXTRACT_JSON
            return _reply(text)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(opsdb, "OPS_DB", tmp_path / "ops.sqlite")
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    usage._local.conn = None
    vlm._BREAKER.clear()
    monkeypatch.setattr(vlm.time, "sleep", lambda s: None)
    monkeypatch.setenv("KEY_A", "a1")
    monkeypatch.setenv("KEY_A2", "a2")
    monkeypatch.setenv("KEY_B", "b1")
    yield tmp_path
    usage._local.conn = None


def provs(validated_b=True):
    return [Provider(name="A", base_url="https://a", api_key_env="KEY_A", backup_key_env="KEY_A2",
                     model="a-extractor", verifier_model="a-verifier", validated=True,
                     price_in_per_m=1.0, price_out_per_m=2.0),
            Provider(name="B", base_url="https://b", api_key_env="KEY_B", model="b-extractor",
                     verifier_model="b-verifier", validated=validated_b, price_in_per_m=0.5, price_out_per_m=1.0)]


def client(api, plist, **kw):
    c = VLMClient(providers=plist, timeout_s=5, retries_per_provider=1, **kw)
    c._client = lambda prov: api.for_client(prov.name, c._key_idx.get(prov.name, 0))
    return c


IMG = np.full((60, 60, 3), 255, np.uint8)


# ------------------------------------------------------------------ provider chain
def test_unvalidated_provider_is_never_used(env):
    chain, skipped = providers.production_chain(provs(validated_b=False))
    assert [p.name for p in chain] == ["A"] and "not validated" in skipped[0]
    api = FakeAPI({"A": [HTTPErr(503)]})
    c = client(api, provs(validated_b=False))
    with pytest.raises(ProviderError):
        c.ask([IMG], "x")
    assert {x[0] for x in api.calls} == {"A"}                     # B never called


def test_no_validated_provider_refuses_to_start(env):
    p = provs()
    for x in p:
        x.validated = False
    with pytest.raises(RuntimeError, match="No validated"):
        VLMClient(providers=p)


def test_fallback_to_next_validated_provider_and_usage_logged(env):
    api = FakeAPI({("A", 0): [HTTPErr(503)], ("A", 1): [HTTPErr(503)], "B": ["ok"]})
    c = client(api, provs())
    r = c.ask([IMG], "x")
    assert r.model == "b-extractor"
    rows = usage.summary()
    by = {(x["provider"], x["role"]): x for x in rows}
    assert by[("A", "extract")]["failures"] == 2 and by[("A", "extract")]["errors"] == {"server_error": 2}
    assert by[("B", "extract")]["ok"] == 1 and by[("B", "extract")]["cost_usd"] == pytest.approx(0.0007)


def test_backup_key_used_on_rate_limit_same_provider(env):
    api = FakeAPI({("A", 0): [HTTPErr(429)], ("A", 1): ["ok"]})
    r = client(api, provs()).ask([IMG], "x")
    assert r.model == "a-extractor" and api.calls[-1][:2] == ("A", 1)


def test_permanent_error_is_not_retried_or_routed(env):
    api = FakeAPI({"A": [HTTPErr(400)]})
    with pytest.raises(ProviderError) as e:
        client(api, provs()).ask([IMG], "x")
    assert not e.value.transient and len(api.calls) == 1


def test_all_providers_failed_is_transient(env):
    api = FakeAPI({"A": [APITimeoutError("t")], ("A", 1): [APITimeoutError("t")], "B": [APITimeoutError("t")]})
    with pytest.raises(AllProvidersFailed) as e:
        client(api, provs()).ask([IMG], "x")
    assert e.value.transient
    assert len(api.calls) == 4          # bounded: 2 attempts on A, 2 on B


def test_verifier_routes_to_verifier_models_only(env):
    api = FakeAPI({("A", 0): [HTTPErr(503)], ("A", 1): [HTTPErr(503)]})
    c = client(api, provs())
    r = c.ask([IMG], "x", model="a-verifier")
    assert r.model == "b-verifier"
    assert all("extractor" not in m for _, _, m in api.calls)


def test_verifier_equal_to_extractor_is_never_used_as_verifier(env):
    p = provs()
    p[1].verifier_model = p[1].model
    api = FakeAPI({("A", 0): [HTTPErr(503)], ("A", 1): [HTTPErr(503)]})
    with pytest.raises(ProviderError):
        client(api, p).ask([IMG], "x", model="a-verifier")
    assert ("B", 0, "b-extractor") not in api.calls


def test_circuit_breaker_skips_failing_provider(env):
    api = FakeAPI({("A", 0): [HTTPErr(503)], ("A", 1): [HTTPErr(503)], "B": ["ok"]})
    c = client(api, provs())
    for _ in range(3):
        c.ask([IMG], "x")
    n_a = sum(1 for x in api.calls if x[0] == "A")
    c.ask([IMG], "x")
    assert sum(1 for x in api.calls if x[0] == "A") == n_a       # breaker open: A skipped


# ------------------------------------------------------------------ jobs
def _ocr_cache(tmp_path):
    lines = []
    y = 10
    for text, h in (("Kia", 40), ("Superior tech", 30)):
        lines.append(OCRLine(text=text, norm=normalize_text(text), score=0.97, lang="latin", script=script_of(text),
                             box=[[10, y], [200, y], [200, y + h], [10, y + h]]))
        y += h + 10
    o = OCRResult(lines=lines, image=np.zeros((200, 400, 3), np.uint8), scale=1.0, deskew_deg=0.0,
                  quality=ImageQuality(400, 200, 500.0, 60.0, []), langs=("latin",), timings_s={"total": 0.1})
    p = tmp_path / "ocr.pkl"
    p.write_bytes(pickle.dumps(o))
    return str(p)


def _png():
    ok, b = cv2.imencode(".png", np.full((200, 400, 3), 255, np.uint8))
    return b.tobytes()


def _factory(api, plist):
    return lambda: client(api, plist)


def test_enqueue_preserves_image_and_dedupes(env):
    j1 = jobs.enqueue(_png(), "ad.png", {"x": 1})
    j2 = jobs.enqueue(_png(), "ad.png", {"x": 1})
    job = jobs.get(j1)
    assert j1 == j2 and job["status"] == "queued"
    assert open(job["input_path"], "rb").read() == _png()


def test_job_completes_and_reopens_without_model_calls(env):
    api = FakeAPI({})
    jid = jobs.enqueue(_png(), "ad.png", {"ocr_cache": _ocr_cache(env)})
    j = jobs.process_next("w", client_factory=_factory(api, provs()))
    assert j["status"] == "completed", j["last_error"]
    n = len(api.calls)
    out = jobs.load_output(jid)
    assert out.record.fields["brand"].value == "Kia" and len(api.calls) == n
    assert jobs.get(jid)["stage"] == "done"
    assert all(r["ok"] for r in usage.summary())


def test_outage_retries_with_backoff_then_resumes_without_repeating_extraction(env):
    # extraction succeeds, then every verifier call fails -> job waits for retry, extraction is kept
    api = FakeAPI({("A", 0): ["ok", HTTPErr(503)], ("A", 1): [HTTPErr(503)], "B": [HTTPErr(503)]})
    jid = jobs.enqueue(_png(), "ad.png", {"ocr_cache": _ocr_cache(env)})
    j = jobs.process_next("w", client_factory=_factory(api, provs()))
    assert j["status"] == "retrying" and j["next_attempt_at"] > time.time() + 20
    assert j["error_kind"] == "all_providers_failed"
    assert (jobs.job_dir(jid) / "raw.json").exists() and not (jobs.job_dir(jid) / "record.json").exists()
    extract_calls = sum(1 for c in api.calls if "extractor" in c[2])
    # provider recovers; make the job due now
    api.script = {}
    opsdb.connect().execute("UPDATE jobs SET next_attempt_at = 0 WHERE id = ?", (jid,))
    vlm._BREAKER.clear()
    j = jobs.process_next("w", client_factory=_factory(api, provs()))
    assert j["status"] == "completed"
    assert sum(1 for c in api.calls if "extractor" in c[2]) == extract_calls      # not paid twice


def test_all_attempts_fail_job_is_kept_and_can_be_retried(env):
    api = FakeAPI({"A": [HTTPErr(503)], ("A", 1): [HTTPErr(503)], "B": [HTTPErr(503)]})
    jid = jobs.enqueue(_png(), "ad.png", {"ocr_cache": _ocr_cache(env)})
    c = opsdb.connect()
    for _ in range(jobs.MAX_ATTEMPTS):
        c.execute("UPDATE jobs SET next_attempt_at = 0 WHERE id = ?", (jid,))
        vlm._BREAKER.clear()
        j = jobs.process_next("w", client_factory=_factory(api, provs()))
    assert j["status"] == "failed" and j["error_kind"] == "all_providers_failed"
    assert jobs.job_dir(jid).exists() and jobs.get(jid)["input_path"]
    jobs.retry(jid)
    assert jobs.get(jid)["status"] == "queued"
    api.script = {}
    vlm._BREAKER.clear()
    assert jobs.process_next("w", client_factory=_factory(api, provs()))["status"] == "completed"


def test_final_attempt_saves_record_without_verifier(env):
    """On the last attempt a verifier outage no longer blocks the record (pipeline behaviour)."""
    api = FakeAPI({("A", 0): ["ok", HTTPErr(503)], ("A", 1): [HTTPErr(503)], "B": [HTTPErr(503)]})
    jid = jobs.enqueue(_png(), "ad.png", {"ocr_cache": _ocr_cache(env)})
    opsdb.connect().execute("UPDATE jobs SET attempts = max_attempts - 1 WHERE id = ?", (jid,))
    j = jobs.process_next("w", client_factory=_factory(api, provs()))
    assert j["status"] == "completed"
    rec = jobs.load_output(jid).record
    assert any("verifier failed" in m for m in rec.models_used)


def test_stale_processing_job_is_recovered(env):
    jid = jobs.enqueue(_png(), "ad.png", {})
    jobs.claim("crashed-worker")
    opsdb.connect().execute("UPDATE jobs SET heartbeat_at = 0 WHERE id = ?", (jid,))
    assert jobs.recover_stale() == 1 and jobs.get(jid)["status"] == "retrying"


def test_claim_is_exclusive(env):
    jobs.enqueue(_png(), "ad.png", {})
    assert jobs.claim("w1") is not None and jobs.claim("w2") is None


def test_unreadable_image_fails_permanently_but_is_kept(env):
    jid = jobs.enqueue(b"not an image", "bad.png", {})
    j = jobs.process_next("w", client_factory=_factory(FakeAPI({}), provs()))
    assert j["status"] == "failed" and j["error_kind"] == "error" and jobs.job_dir(jid).exists()
