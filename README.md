---
title: Indycium Ad-Reader
emoji: 📰
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: AI ad data entry - OCR, vision LLM, verifier, DB matching, RAG
---

# Indycium Ad-Reader

AI-powered data entry for print, newspaper and digital advertisements. Upload an ad and the system reads it,
extracts a structured record (brand, product, headline, prices, offers, contacts, dealers, attributes), checks every
value against the printed text, links brands/products/advertisers to a master database **without ever inventing an
ID**, and hands the result to a human reviewer who confirms and saves it. Saved records and machine extractions are
then searchable with a hybrid RAG search whose AI answers are fact-checked claim by claim.

## Key features

- **OCR with automatic Indian-script detection** - RapidOCR (PP-OCR models on ONNX Runtime). Latin + Devanagari by
  default; Telugu, Tamil and Kannada are detected per ad and only that model is loaded (RAM-safe on small CPUs).
- **Vision-language extraction** - an OpenAI-compatible vision LLM (default: NVIDIA API Catalog
  `nemotron-3.5-super-vl-120b-a12b`) reads the image plus the numbered OCR transcript.
- **Independent verifier + deterministic validation** - a second model (`nemotron-3-nano-omni-30b-a3b-reasoning`)
  gives an independent reading; rule-based evidence gates mark each field 🟢 Confirmed, 🟠 Needs review or 🔴 Missing.
- **Safe database matching** - IDs come only from existing master records (string matching, never the model);
  an ID is pre-filled only on a confirmed match; new brands/products are added only by a human reviewer.
- **Human-in-the-loop review** - flagged fields must be checked before saving; every correction is recorded.
- **Background job queue** - each ad is a persistent job with checkpoints, retries, a live processing card and a
  "Process now in this session" fallback when no worker is running (as on Hugging Face Spaces).
- **Hybrid Search & RAG** - BM25 + multilingual-e5-small embeddings fused with Reciprocal Rank Fusion, natural-language
  auto-filters, grounded answers with validated citations, and a mandatory judge model that verifies every claim.

## Verified results (from `releases/`)

| Metric | Value |
|---|---|
| Automated tests | 267 passing (pytest, 25 Nov 2026) |
| bench200 accuracy - frozen baseline → candidate_v4 | 86.4% → **89.0%** (1,614 scored values, 200 ads) |
| Silent errors on bench200 (wrong value auto-confirmed) | 22 → **8** |
| Confirmed precision on bench200 | 97.2% → **99.0%** |
| set50 / holdout20 accuracy (candidate_v4) | 92.7% / 95.5% |
| Snapshot integrity | `bench200_frozen`, `candidate_v3`, `candidate_v4` verified against `SHA256SUMS` |

> **Caveat:** benchmark labels are **provisional** - drafted from the images and pending human audit (0 of 200
> bench200 labels human-verified). candidate_v4 figures are a development retest on already-examined data, not
> independent proof. Details: [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

## Quickstart (local)

```bash
git clone https://github.com/anni88git/ad-reader.git
cd ad-reader
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your NVIDIA_API_KEY in .env (never commit it)
streamlit run app/streamlit_app.py
```

Optional background worker (otherwise jobs run in the browser session):

```bash
python scripts/worker.py
```

Run the tests:

```bash
pytest -q
```

Tests that need the local benchmark images/caches (`data/eval`, not in this repository) skip themselves.
The OCR models and the e5 embedding model (~118 MB) download automatically on first use.

## Deploying on Hugging Face Spaces

This repository is a ready Space. Hugging Face runs Streamlit apps through the **Docker SDK** (see the YAML header
above): the `Dockerfile` installs the Linux libraries from `packages.txt`, the Python requirements, and starts
Streamlit on port 7860. In the Space **Settings → Variables and secrets** add:

| Name | Type | Value |
|---|---|---|
| `NVIDIA_API_KEY` | **Secret** | your key from build.nvidia.com |
| `NVIDIA_API_KEY_BACKUP` | Secret (optional) | second key, used automatically if the first is rate-limited |
| `VLM_BACKEND` | Variable | `nvidia` |
| `NVIDIA_MODEL` | Variable | `nvidia/nemotron-3.5-super-vl-120b-a12b` |
| `NVIDIA_VERIFIER_MODEL` | Variable | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |

Notes for the free CPU Space: there is no background worker, so use **▶ Process now in this session** on the
processing card; the first OCR / embedding run downloads models; storage is not persistent across restarts, so saved
records and jobs reset when the Space rebuilds. Sample ad images and evaluation data are not included - upload your
own JPG/PNG to try it.

## Repository layout

| Path | Contents |
|---|---|
| `adreader/` | pipeline: OCR, extraction, verifier, validation rules, matching, jobs, RAG |
| `app/` | Streamlit UI (landing page, main app, Search & RAG page, theme, mascot) |
| `scripts/` | worker, evaluation and reporting scripts |
| `tests/` | pytest suite |
| `releases/` | frozen benchmark snapshots with `SHA256SUMS` |
| `data/` | master database, drafted labels and the search index (other data stays local) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.
