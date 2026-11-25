

# Indycium Ad-Reader

AI-powered data entry for print, newspaper and digital advertisements. Upload an ad and the system reads it,
extracts a structured record (brand, product, headline, prices, offers, contacts, dealers, attributes), checks every
value against the printed text, links brands/products/advertisers to a master database **without ever inventing an
ID**, and hands the result to a human reviewer who confirms and saves it. Saved records and machine extractions are
then searchable with a hybrid RAG search whose AI answers are fact-checked claim by claim.

| | |
|---|---|
| **Live working POC** | https://ad-reader-kejd8ymzrybmufjazbakbc.streamlit.app (Streamlit Community Cloud) |
| **Quick walkthrough video** | https://www.loom.com/share/0cf464352ff34c0da4fc91a5bff8099d |
| **Final report (PDF)** | [Indycium_Ad_Reader_Final_Report.pdf](Indycium_Ad_Reader_Final_Report.pdf) |

If the live app shows "This app has gone to sleep", click **Yes, get this app back up!** - it restarts in about a minute.

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
- **Background job queue** - each ad is a persistent job with checkpoints, retries and a live processing card
  (stages + elapsed timer, Cancel / Retry). Locally a separate worker (`scripts/worker.py`) processes jobs; on hosts
  without one (Streamlit Community Cloud) the same worker loop runs as a background thread inside the app.
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

## Performance

| Case | Typical time |
|---|---|
| Demo samples with saved results (Kia, Horlicks, Namaste) | ~5-10 s - saved OCR + extraction are reused; only the verifier and checks run |
| A new ad (upload) | ~40-60 s - OCR on CPU, then the cloud vision model (median ~22 s per call), verifier, checks |
| First action after a cold start | a little longer while OCR / embedding models finish downloading (warmed up in the background at start) |

Most of the time for a new ad is the cloud vision model's response time; progress is shown live on the processing card.

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

Optional separate background worker (otherwise the app runs one in-process automatically):

```bash
python scripts/worker.py
```

Run the tests:

```bash
pytest -q
```

Tests that need local evaluation caches that are not in this repository skip themselves.
The OCR models and the e5 embedding model (~118 MB) download automatically on first use.

## Deploying on Streamlit Community Cloud (live demo)

1. On https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**.
2. Repository `anni88git/ad-reader`, branch `main`, main file path **`app/streamlit_app.py`**.
3. **Advanced settings:** Python **3.12**, and paste the secrets in TOML format:

```toml
VLM_BACKEND = "nvidia"
NVIDIA_API_KEY = "nvapi-..."                 # your key from build.nvidia.com
NVIDIA_API_KEY_BACKUP = "nvapi-..."          # optional second key, used if the first is rate-limited
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
NVIDIA_VERIFIER_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
```

Notes: the app starts its own background worker and warms up the models after each start; OCR models download into
a writable cache (`~/.cache/adreader/rapidocr_models`, override with `ADREADER_OCR_MODEL_DIR`); storage is not
persistent, so saved records and jobs reset when the app reboots. A public app spends your NVIDIA quota on every
extraction - share the link with reviewers only.

## Docker / Hugging Face Spaces

The `Dockerfile` (Python 3.12-slim, Linux libraries from `packages.txt`, Streamlit on port 7860) runs the same app
anywhere Docker does, including Hugging Face Docker Spaces (which currently require a paid plan for CPU hosting). Set
the same variables as above as environment variables / Space secrets.

## Repository layout

| Path | Contents |
|---|---|
| `adreader/` | pipeline: OCR, extraction, verifier, validation rules, matching, jobs, RAG |
| `app/` | Streamlit UI (landing page, main app, Search & RAG page, theme, mascot) |
| `scripts/` | worker, evaluation and reporting scripts |
| `tests/` | pytest suite |
| `releases/` | frozen benchmark snapshots with `SHA256SUMS` |
| `data/` | master database, drafted labels, search index, sample ads, the images the search shows, the audit set and saved results for the 3 demo samples (job history, uploads and evaluation caches stay local) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.
