"""Inference providers (OpenAI-compatible vision endpoints) and the fallback chain.

A provider is used in production ONLY if `validated` is true, i.e. its extraction model (and
verifier model) passed the held-out benchmark with this pipeline. Unvalidated providers are
listed but skipped (and reported), unless ALLOW_UNVALIDATED_PROVIDERS=1 is set explicitly for a
benchmark run. Secrets are never stored here: `api_key_env` names the environment variable.

Config: config/providers.json (optional). Without it, a single provider is built from the
existing .env settings - identical to the previous behaviour.

  [{"name": "nvidia-trial", "base_url": "...", "api_key_env": "NVIDIA_API_KEY",
    "backup_key_env": "NVIDIA_API_KEY_BACKUP", "model": "...", "verifier_model": "...",
    "validated": true, "validated_on": "holdout20 ...", "production_terms_ok": false,
    "price_in_per_m": 0, "price_out_per_m": 0, "notes": "..."}]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, vlm_config

CONFIG = ROOT / "config" / "providers.json"


@dataclass
class Provider:
    name: str
    base_url: str
    api_key_env: str
    model: str
    verifier_model: str | None = None
    backup_key_env: str | None = None
    validated: bool = False
    validated_on: str = ""
    production_terms_ok: bool = False
    price_in_per_m: float = 0.0          # USD per million input tokens
    price_out_per_m: float = 0.0
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    @property
    def backup_key(self) -> str:
        return os.getenv(self.backup_key_env, "") if self.backup_key_env else ""

    def model_for(self, role: str) -> str | None:
        return self.verifier_model if role == "verify" else self.model

    def cost(self, tokens_in: int | None, tokens_out: int | None) -> float:
        return round(((tokens_in or 0) * self.price_in_per_m + (tokens_out or 0) * self.price_out_per_m) / 1e6, 6)

    def problems(self) -> list[str]:
        out = []
        if not self.api_key:
            out.append(f"no API key in ${self.api_key_env}")
        if not self.model:
            out.append("no model")
        if self.verifier_model and self.verifier_model == self.model:
            out.append("verifier model equals extraction model - second opinion would not be independent")
        return out


def _default_from_env() -> list[Provider]:
    cfg = vlm_config()
    env = {"nvidia": ("NVIDIA_API_KEY", "NVIDIA_API_KEY_BACKUP"), "dashscope": ("DASHSCOPE_API_KEY", None),
           "local": ("LOCAL_API_KEY", None)}.get(cfg.backend, ("NVIDIA_API_KEY", None))
    nvidia = cfg.backend == "nvidia"
    return [Provider(
        name=f"{cfg.backend}{'-trial' if nvidia else ''}", base_url=cfg.base_url, api_key_env=env[0],
        backup_key_env=env[1], model=cfg.model, verifier_model=cfg.verifier_model,
        validated=nvidia and "nemotron-3.5-super-vl" in (cfg.model or ""),
        validated_on="set50 + holdout20 (nemotron 120B extractor + nano verifier)" if nvidia else "",
        production_terms_ok=False,
        notes="NVIDIA API Catalog trial: free, rate-limited, terms allow testing/evaluation only" if nvidia else "")]


def load_providers(path: Path | None = None) -> list[Provider]:
    p = Path(path or CONFIG)
    if not p.exists():
        return _default_from_env()
    items = json.loads(p.read_text(encoding="utf-8"))
    known = set(Provider.__dataclass_fields__)
    return [Provider(**{k: v for k, v in it.items() if k in known},
                     extra={k: v for k, v in it.items() if k not in known}) for it in items]


def allow_unvalidated() -> bool:
    return os.getenv("ALLOW_UNVALIDATED_PROVIDERS", "") == "1"


def production_chain(providers: list[Provider] | None = None) -> tuple[list[Provider], list[str]]:
    """Providers usable now, in order, plus the reasons others were skipped."""
    chain, skipped = [], []
    for pr in providers if providers is not None else load_providers():
        probs = pr.problems()
        if not pr.validated and not allow_unvalidated():
            skipped.append(f"{pr.name}: not validated on the held-out benchmark - not used")
        elif probs and probs != ["verifier model equals extraction model - second opinion would not be independent"]:
            skipped.append(f"{pr.name}: " + "; ".join(probs))
        else:
            chain.append(pr)
    return chain, skipped


def readiness() -> dict:
    """Human-readable production status of the provider setup."""
    provs = load_providers()
    chain, skipped = production_chain(provs)
    return {
        "active_chain": [p.name for p in chain],
        "skipped": skipped,
        "independent_providers": len({p.base_url for p in chain}),
        "production_terms_ok": [p.name for p in chain if p.production_terms_ok],
        "warnings": ([] if any(p.production_terms_ok for p in chain) else
                     ["no active provider has terms that allow production use"]) +
                    ([] if len({p.base_url for p in chain}) > 1 else
                     ["only one provider endpoint - an outage or quota stop halts processing (jobs are kept and retried)"]),
    }
