"""Central configuration. Secrets come ONLY from environment / .env (never hard-coded)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
ADS_DIR = DATA_DIR / "ads"
LABELS_DIR = DATA_DIR / "labels"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "db" / "master.sqlite"

# Confidence thresholds -> field status (accuracy-first: be strict)
CONFIRM_THRESHOLD = 0.85   # >= : green, auto-filled
REVIEW_THRESHOLD = 0.50    # >= : amber, filled but flagged; below: red, left empty + flagged

# Entity matching (rapidfuzz scores 0-100)
MATCH_ACCEPT = 92          # single candidate at/above this -> link ID
MATCH_MARGIN = 6           # best must beat runner-up by this, else ambiguous -> review


@dataclass(frozen=True)
class VLMConfig:
    backend: str
    base_url: str
    api_key: str
    model: str
    verifier_model: str | None
    backup_api_key: str = ""


def vlm_config() -> VLMConfig:
    backend = os.getenv("VLM_BACKEND", "nvidia").strip().lower()
    if backend == "nvidia":
        return VLMConfig(
            backend=backend,
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.getenv("NVIDIA_API_KEY", ""),
            model=os.getenv("NVIDIA_MODEL", ""),
            verifier_model=os.getenv("NVIDIA_VERIFIER_MODEL") or None,
            backup_api_key=os.getenv("NVIDIA_API_KEY_BACKUP", ""),
        )
    if backend == "dashscope":
        return VLMConfig(
            backend=backend,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            model=os.getenv("DASHSCOPE_MODEL", "qwen3-vl-plus"),
            verifier_model=os.getenv("DASHSCOPE_VERIFIER_MODEL") or None,
        )
    if backend == "local":
        return VLMConfig(
            backend=backend,
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "not-needed"),
            model=os.getenv("LOCAL_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            verifier_model=os.getenv("LOCAL_VERIFIER_MODEL") or None,
        )
    raise ValueError(f"Unknown VLM_BACKEND={backend!r} (use 'nvidia', 'local' or 'dashscope')")
