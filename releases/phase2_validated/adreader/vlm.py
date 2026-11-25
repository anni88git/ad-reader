"""Thin client for OpenAI-compatible vision-language endpoints (NVIDIA API Catalog, vLLM on
Kaggle, DashScope). Handles image encoding, 'thinking' switches, retries and robust JSON parsing."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from . import preprocess as pp
from .config import VLMConfig, vlm_config


@dataclass
class VLMReply:
    text: str
    model: str
    seconds: float
    tokens_in: int | None = None
    tokens_out: int | None = None


class ProviderError(RuntimeError):
    """A model call failed. `transient` errors (timeout, rate limit, 5xx, empty reply) are worth
    retrying later; permanent ones (bad request) are not."""
    def __init__(self, msg: str, transient: bool = True, kind: str = "transient"):
        super().__init__(msg)
        self.transient = transient
        self.kind = kind


class AllProvidersFailed(ProviderError):
    def __init__(self, msg: str):
        super().__init__(msg, transient=True, kind="all_providers_failed")


class EmptyReply(RuntimeError):
    pass


def _classify(e: Exception) -> tuple[str, int | None, bool]:
    """(error_type, http_status, transient)."""
    code = getattr(e, "status_code", None)
    name = type(e).__name__
    if code in (401, 402, 403):
        return "auth_or_quota", code, True          # key/quota problem: another key/provider may work
    if code == 429:
        return "rate_limit", code, True
    if code is not None and code >= 500:
        return "server_error", code, True
    if code == 404:
        return "model_not_found", code, True        # provider-specific: the next provider may have it
    if code is not None:
        return "bad_request", code, False
    if "Timeout" in name:
        return "timeout", None, True
    if "Connection" in name:
        return "connection", None, True
    if isinstance(e, EmptyReply):
        return "empty_reply", None, True
    return name, None, True


# circuit breaker shared by all clients in this process: provider -> [consecutive failures, open until]
_BREAKER: dict[str, list] = {}
BREAKER_FAILS, BREAKER_OPEN_S = 3, 120


def _breaker_open(name: str) -> bool:
    st = _BREAKER.get(name)
    return bool(st and st[1] > time.time())


def _breaker_record(name: str, ok: bool) -> None:
    st = _BREAKER.setdefault(name, [0, 0.0])
    if ok:
        st[0], st[1] = 0, 0.0
    else:
        st[0] += 1
        if st[0] >= BREAKER_FAILS:
            st[1] = time.time() + BREAKER_OPEN_S


def _env_float(k: str, d: float) -> float:
    try:
        return float(os.getenv(k, d))
    except ValueError:
        return d


class VLMClient:
    """Same interface as before (`ask`). Calls go through a chain of VALIDATED providers
    (adreader/providers.py) with a bounded timeout, bounded retries per provider, a circuit breaker,
    a one-time switch to the provider's backup key on auth/quota errors, and a usage log row for
    every attempt (adreader/usage.py).

    VLMClient(cfg) with an explicit VLMConfig = a single provider (benchmarks / scripts)."""

    def __init__(self, cfg: VLMConfig | None = None, providers: list | None = None,
                 timeout_s: float | None = None, retries_per_provider: int | None = None):
        from .providers import Provider, production_chain
        self.timeout_s = timeout_s or _env_float("VLM_TIMEOUT_S", 120)
        self.retries = int(retries_per_provider if retries_per_provider is not None
                           else _env_float("VLM_RETRIES_PER_PROVIDER", 1))
        if cfg is not None:
            self.cfg = cfg
            self.providers = [Provider(name=cfg.backend, base_url=cfg.base_url, api_key_env="",
                                       model=cfg.model, verifier_model=cfg.verifier_model, validated=True)]
            self._keys = {cfg.backend: [k for k in (cfg.api_key, cfg.backup_api_key) if k]}
            self.skipped: list[str] = []
        else:
            self.providers, self.skipped = production_chain(providers)
            if not self.providers:
                raise RuntimeError("No validated inference provider available: "
                                   + "; ".join(self.skipped or ["none configured"]))
            p0 = self.providers[0]
            self.cfg = VLMConfig(backend=p0.name, base_url=p0.base_url, api_key=p0.api_key, model=p0.model,
                                 verifier_model=p0.verifier_model, backup_api_key=p0.backup_key)
            self._keys = {p.name: [k for k in (p.api_key, p.backup_key) if k] for p in self.providers}
        if not self.cfg.model:
            raise RuntimeError(f"No model configured for provider '{self.cfg.backend}'")
        if not self._keys.get(self.providers[0].name):
            raise RuntimeError(f"No API key configured for provider '{self.providers[0].name}' (see .env)")
        self._clients: dict[tuple[str, int], OpenAI] = {}
        self._key_idx: dict[str, int] = {}

    def _client(self, prov) -> OpenAI:
        i = self._key_idx.get(prov.name, 0)
        k = (prov.name, i)
        if k not in self._clients:
            self._clients[k] = OpenAI(base_url=prov.base_url, api_key=self._keys[prov.name][i],
                                      timeout=self.timeout_s, max_retries=0)
        return self._clients[k]

    def _extra_body(self, model: str) -> dict:
        # Reasoning models spend the whole token budget "thinking" and return empty content
        # for extraction tasks; we want direct, deterministic transcription instead.
        m = model.lower()
        if "nemotron" in m or "qwen3" in m:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def _route(self, model: str | None) -> list[tuple]:
        """[(provider, model, role)] in order. No model -> extraction role on every provider;
        the verifier model -> verifier role on every provider; any other explicit model
        (benchmark override) -> that model on the first provider only."""
        p0 = self.providers[0]
        if model is None or model == p0.model:
            role = "extract"
        elif model == p0.verifier_model:
            role = "verify"
        else:
            return [(p0, model, "other")]
        out = []
        for p in self.providers:
            m = p.model_for(role)
            if m and not (role == "verify" and m == p.model):     # never verify with the extractor itself
                out.append((p, m, role))
        return out

    def ask(self, images: list[np.ndarray], prompt: str, system: str | None = None,
            model: str | None = None, max_tokens: int = 4096, max_side: int = 2048,
            retries: int | None = None) -> VLMReply:
        from . import usage
        content = []
        for img in images:
            b64 = base64.b64encode(pp.to_jpeg_bytes(img, max_long_side=max_side)).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": content}]
        n_retry = self.retries if retries is None else min(retries, self.retries)
        errors: list[str] = []
        route = self._route(model)
        for prov, m, role in route:
            if _breaker_open(prov.name) and len(route) > 1:
                errors.append(f"{prov.name}: skipped (circuit open after repeated failures)")
                continue
            attempt = 0
            while attempt <= n_retry:
                t = time.perf_counter()
                try:
                    sampling = {"temperature": 0, "top_p": 1}
                    if "kimi" in m.lower():          # provider pins these for Kimi models
                        sampling = {"temperature": 0.6, "top_p": 0.95}
                    r = self._client(prov).chat.completions.create(
                        model=m, messages=messages, **sampling,
                        max_tokens=max_tokens, extra_body=self._extra_body(m))
                    text = r.choices[0].message.content or ""
                    if not text.strip():
                        raise EmptyReply(f"empty response (finish_reason={r.choices[0].finish_reason})")
                    u = r.usage
                    ti, to = getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)
                    sec = round(time.perf_counter() - t, 2)
                    usage.log(prov.name, m, role, True, latency_s=sec, tokens_in=ti, tokens_out=to,
                              cost_usd=prov.cost(ti, to))
                    _breaker_record(prov.name, True)
                    return VLMReply(text, m, sec, ti, to)
                except Exception as e:
                    etype, code, transient = _classify(e)
                    usage.log(prov.name, m, role, False, error_type=etype, http_status=code,
                              latency_s=round(time.perf_counter() - t, 2))
                    errors.append(f"{prov.name}/{m}: {etype}{' ' + str(code) if code else ''}: {str(e)[:160]}")
                    if etype in ("auth_or_quota", "rate_limit"):
                        i = self._key_idx.get(prov.name, 0)
                        if i + 1 < len(self._keys[prov.name]):      # same-provider backup key, once
                            self._key_idx[prov.name] = i + 1
                            continue
                    if not transient:
                        raise ProviderError("; ".join(errors), transient=False, kind="permanent")
                    attempt += 1
                    if attempt <= n_retry:
                        time.sleep(min(2 * attempt, 6))
            _breaker_record(prov.name, False)
        if len(route) == 1:
            raise ProviderError("model call failed: " + "; ".join(errors[-3:]))
        raise AllProvidersFailed("all providers failed: " + "; ".join(errors[-6:]))


def parse_json(text: str):
    """Extract the first JSON object/array from a model reply (handles ```json fences, prose)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in model reply: {text[:200]!r}")
    start = min(starts)
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(t[start:])
    return obj
