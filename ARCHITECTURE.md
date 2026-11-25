# Architecture

Indycium Ad-Reader turns an advertisement image into a verified, database-linked record. The design rule throughout:
**the AI proposes, deterministic checks and a human decide** - no value is auto-confirmed without printed evidence,
and no database ID is ever produced by a model.

## End-to-end pipeline

```
 image (upload / sample / newspaper-page region)
   │
   ▼
 1. OCR  ─────────────── adreader/ocr.py
   │  RapidOCR (PP-OCR on ONNX Runtime): detection on plain + contrast-enhanced image, deskew,
   │  recognition with Latin + Devanagari (default) or auto-detected Telugu / Tamil / Kannada,
   │  reading order, Indic digits normalised. Output: numbered transcript with boxes and confidences.
   ▼
 2. Vision-language extraction ─ adreader/extract.py, adreader/vlm.py
   │  OpenAI-compatible VLM (default NVIDIA nemotron-3.5-super-vl) sees the image + numbered OCR lines
   │  and returns a structured RawExtraction (brand, product, headline, prices, offers, contacts, …).
   ▼
 3. Independent verifier ─ adreader/verify.py
   │  A second model (nemotron-3-nano-omni reasoning) reads the image on its own; disagreements are flagged.
   ▼
 4. Deterministic validation ─ adreader/validate.py, accuracy_rules*.py
   │  Evidence gates: a value is 🟢 Confirmed only if it is grounded in the printed text (and agrees with the
   │  verifier); otherwise 🟠 Needs review or 🔴 Missing. Normalisation of prices, phones, languages, etc.
   ▼
 5. Database matching ─ adreader/matcher.py, adreader/db_link.py
   │  String matching against master tables (brands, products, advertisers, categories). Outcomes:
   │  matched (ID pre-filled) · suggested · ambiguous · similar records · unmatched. IDs only from DB rows.
   ▼
 6. Human review & save ─ app/streamlit_app.py, adreader/review.py
   │  Reviewer checks every flagged field, may pick a suggested record or add a genuinely new one
   │  (logged in Database additions, undoable), then saves. Corrections are stored with the submission.
   ▼
 7. Hybrid Search & RAG ─ adreader/rag.py, app/rag_page.py
      Index over saved (🟢 verified) records + machine extractions (🟡 unverified).
      Retrieval: BM25 + multilingual-e5-small embeddings, fused with Reciprocal Rank Fusion; natural-language
      auto-filters (e.g. "Hindi ads" → language = Hindi). Answers cite retrieved records only; a judge model
      verifies every claim (✅ verified / ⚠️ flagged). Similar-ad search uses the same text representation.
```

## Background job queue

```
 UI ──enqueue──▶ ops.sqlite (jobs table) ◀──claim── worker (scripts/worker.py)
  ▲                    │                                 │
  │   live card polls  │  status · stage · attempts      │  run_job(): OCR → extract → verify → validate
  └────────────────────┘  heartbeat · last_error         ▼
                                                   data/jobs/<id>/  checkpoints:
                                                   input · ocr.pkl · raw.json · verifier.json · record.json
```

- **Persistent & resumable** - each stage writes a checkpoint; a retry resumes from the failed stage (no repeated
  model calls).
- **Retries with back-off** for transient provider errors; permanent errors fail the job with a readable reason.
- **Stale recovery** - a job left "processing" by a crashed worker (no heartbeat) returns to the queue.
- **Cancel / retry** from the UI; jobs and results stay listed on the **Jobs** page.
- **No worker?** (e.g. Hugging Face Spaces) the processing card offers **Process now in this session**, which runs the
  same job code in the browser session.
- **Live processing card** - a Streamlit fragment polls every 3 s (only the card redraws), shows the current stage
  and elapsed time, and offers Cancel if a job stalls.

## Data stores

| Store | Purpose |
|---|---|
| `data/db/master.sqlite` | master brands / products / advertisers / categories, saved submissions, reviewer additions |
| `data/db/ops.sqlite` | job queue, worker heartbeats, provider usage log (local, not committed) |
| `data/jobs/` | per-job checkpoints (local, not committed) |
| `data/rag_index/index.json` | search index (documents + embedding vectors) |

## Configuration & secrets

All settings come from environment variables / `.env` (see `.env.example`); keys are never hard-coded. On Hugging
Face Spaces they are set as Space secrets and variables (see README).

## Quality gates

- `pytest` suite (267 tests) covering OCR script detection, validation rules, matching, review, jobs, RAG and UI flows.
- Frozen benchmark snapshots in `releases/` with `SHA256SUMS`, so every reported number can be traced to the exact
  code that produced it (see [MODEL_COMPARISON.md](MODEL_COMPARISON.md)).
