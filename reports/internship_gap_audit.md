# Internship requirements - gap audit (read-only)

*Date: 2026-11-24. Method: inspection of code, app, docs, test files and stored results. "Tested" means an
automated test or a recorded run exists; "inspected" means the code was read only. Nothing was changed
while writing this audit (the 200-image evaluation was running in the background).*

**Legend:** COMPLETE = implemented and verified by a test or a recorded run · PARTIAL = implemented,
but verification or a piece is missing · MISSING = not implemented / not produced.

## Summary

| # | Requirement | Status |
|---|---|---|
| 1 | JPG/PNG upload | PARTIAL (works; upload path through the new job queue has no automated test) |
| 2 | OCR + vision extraction | COMPLETE |
| 3 | Extraction of brand, product, client, headline, description, prices, offers, contacts, other | PARTIAL (all extracted; description/offers/features hardly evaluated) |
| 4 | English + Indian regional languages | PARTIAL (OCR for 5 scripts only; others rely on the vision model) |
| 5 | Missing / uncertain information flagged | COMPLETE |
| 6 | DB matching, IDs never invented | COMPLETE for the guarantee; PARTIAL for real-data integration (mock DB) |
| 7 | Automatic form population | COMPLETE (in the POC form); no export to an external system |
| 8 | Comparison RapidOCR / PaddleOCR / Qwen2.5-VL / Qwen3-VL / current model | PARTIAL (OCR + current model done; **Qwen not measured**) |
| 9 | Accuracy, speed, infrastructure, cost comparison | PARTIAL (no consolidated comparison document) |
| 10 | Recommended model and system architecture | MISSING as a deliverable (content exists in scattered places) |
| 11 | Working POC + complete demo workflow | PARTIAL (POC works; no demo script / walkthrough) |
| 12 | Source code + setup documentation | PARTIAL (code + requirements; **no README / setup guide**) |
| 13 | Evaluation on 50 and 200 real ads | PARTIAL (50 done, 2 labels human-verified; 200 running, 0 human-verified) |

---

## 1. JPG/PNG advertisement upload - PARTIAL

**Evidence**
- `app/streamlit_app.py:537` `st.sidebar.file_uploader("Advertisement image (JPG/PNG)", type=["jpg","jpeg","png"])` in `page_process()`.
- Full newspaper page mode: `find_ads()` (`adreader/segment.py`) + region editor in `page_process()`; the page is saved to `data/outputs/page_uploads/` before ad finding.
- Upload -> persistent job: `jobs.enqueue()` (`adreader/jobs.py`) copies the image to `data/jobs/<id>/`.
- Tested: `tests/test_app.py::test_kia_flow_review_gate_and_save` (sample image through the job flow, headless Streamlit); `tests/test_reliability.py` (enqueue preserves image bytes, dedupe, retry, crash recovery).
- Recorded real runs: worker processed a real held-out ad end-to-end (job #1) and recovered from a simulated outage (job #2). Manual uploads were done earlier with the pre-queue version.

**Remaining work**
- An automated test that drives the *uploader widget* (not the sample picker) through enqueue -> worker -> review -> save.
- Manual click-through of upload with the current queue version (not recorded since the queue was added).
- File-type / size validation messages (e.g. corrupt file, >20 MB, WEBP/PDF) are not tested; unreadable images fail the job permanently (tested), but the UI message is not.

## 2. OCR and vision-based extraction - COMPLETE

**Evidence**
- OCR: `adreader/ocr.py` `run_ocr()` (RapidOCR = PaddleOCR PP-OCR models on ONNX Runtime; plain + CLAHE detection, Latin + Indic recognisers, reading order, deskew). Preprocessing `adreader/preprocess.py`.
- Vision model extraction: `adreader/extract.py` `extract()` / `build_prompt()` (Nemotron 3.5 Super VL 120B via NVIDIA API; prompt v2 frozen, v2.5 rejected and kept off).
- Independent image-only verifier: `adreader/verify.py` `verify()`.
- Tested/measured: set50 + holdout20 runs, `docs/EVALUATION_REPORT.md` sections 2-3d; OCR comparison `data/eval/holdout20/report/ocr_comparison.json`, `ocr_fallback.json`.

**Remaining work:** none for the requirement itself (quality limits are listed under #4).

## 3. Field extraction - PARTIAL

**Evidence**
- Schema `adreader/schema.py` (`RawExtraction`, `AdRecord`): brand, product, model_variant, advertiser (= client), ad_type, category, headline, sub_headline, description, body_copy_summary, ad_message, prices (with kind: price / old price / discount / cashback / EMI), offers, features, warranty, phones, emails, websites, social handles, addresses, dealers, call_to_action, reference numbers (RERA/UIN/IRDAI/...), QR, disclaimer, language, category attributes (cars, real estate, ...).
- Price vs discount/cashback/benefit: `adreader/normalize.py` `parse_amount`, `price_kind`; tests `tests/test_accuracy_rules.py` (Indian number formats, lakh/crore in 7 scripts, benefits are not prices).
- Contacts: `normalize.parse_phone/parse_email/parse_website`; recall of missed phones / IDs from OCR `validate.recall_from_ocr` (tested).
- Measured on set50/holdout20: brand, product, advertiser, ad_type, category, headline, sub_headline, price, language, QR, phones, websites, emails, reference IDs.

**Remaining work**
- **Description, offers, features, warranty, call-to-action, body copy are mostly in `_skip`** in the labels, so their accuracy is essentially unmeasured (only a few set50 labels include them). Label and score them on a subset.
- Advertiser (client) accuracy is the weakest core field (dealer vs manufacturer judgement, `EVALUATION_REPORT.md` 3d); no fix is in place, only review flags.
- `model_variant` has weak validation (the rejected prompt v2.5 produced 4 confirmed wrong variants); with the frozen prompt it produced 0, but there is no dedicated rule.

## 4. English and Indian regional languages - PARTIAL

**Evidence**
- OCR scripts: Latin, Devanagari (Hindi/Marathi), Tamil, Telugu, Kannada (`ocr.SUPPORTED_LANGS`); Indic digit normalisation; script detection; language vs OCR-script rules (`validate.language_from_ocr`, `_drop_unsupported_language`, tested).
- Measured: set50 has 22 regional ads, holdout20 has 8, bench200 has 13 languages (results pending at the time of writing).
- PaddleOCR (native) has no Bengali/Gujarati model either (`ocr_fallback.json`).

**Remaining work**
- Bengali, Gujarati, Malayalam, Odia, Punjabi, Urdu, Meetei Mayek: no OCR cross-check, so these depend on the vision model + verifier only. Evaluate an OCR with these scripts.
- Known weak spots: regional headline spelling (Telugu/Kannada vowel signs), regional lakh/crore words next to numbers (3 real misses), see `EVALUATION_REPORT.md` 3c.
- Per-language accuracy with adequate sample sizes: bench200 gives the first breakdown; groups of 1-9 ads are indicative only.

## 5. Missing and uncertain information flagged - COMPLETE

**Evidence**
- Statuses confirmed / review / missing (`schema.FieldStatus`, thresholds `config.CONFIRM_THRESHOLD/REVIEW_THRESHOLD`), per-field reasons and evidence.
- `adreader/validate.py`: evidence grounding vs OCR, digit-exact checks, consistency rules, second-opinion disagreement, headline/sub-headline policies, prominence check, look-alike characters, OCR recall, small-print / ignored-line classification (`classify_ocr_lines`).
- Review gate: `review.pending_checks()` + `build_final()` refuse to save unchecked flagged values; red fields are not pre-filled (`initial_edits`).
- Tested: `tests/test_review.py`, `tests/test_accuracy_rules.py`, `tests/test_pipeline_outputs.py`; red team 14/14 (`scripts/redteam_validate.py`); planted-error benchmark 0/248 and 0/98 leaks (`scripts/adversarial_bench.py`); silent errors 0 on set50 (258 confirmed) and holdout20 (96 confirmed).

**Remaining work:** none mandatory. Review load is high (~34% of correct values flagged), mainly driven by the mock database and policy rules.

## 6. Database record matching - IDs never invented - COMPLETE (guarantee) / PARTIAL (real data)

**How the guarantee is enforced (inspected + tested):**
1. The model never sees or outputs IDs: `adreader/extract.py` header + prompt; IDs come only from SQLite (`adreader/db.py`).
2. Matching is string logic only: `matcher.Matcher.match()` links only on an exact name/alias, or near-identical spelling (>= `MATCH_ACCEPT` 92 with margin `MATCH_MARGIN`). Ties break only by category, otherwise no ID (`matcher.py:121-160`).
3. IDs are kept only on CONFIRMED fields: `matcher._hold_unconfirmed_ids()` (`matcher.py:329`) moves any ID on a review field to `suggested_id`.
4. Look-alike suggestions never set an ID: `matcher.lookalike()` / `_suggest_lookalike()`.
5. Saving: `review.build_final()` (`review.py:85`) saves an ID only if the system confirmed the field and the reviewer left the value unchanged, or the reviewer explicitly picked a record (`streamlit_app.link_widget`, IDs pre-selected only for confirmed matches).
6. New records only by an explicit reviewer action: `review.apply_new_records()` + `adreader/master_data.py` `add_entity()` (exact duplicates refused, look-alikes need acknowledgement, audited, undoable).

**Tests:** `test_matcher.py::test_brand_never_guessed`, `test_products_scoped_and_strict`, `test_lookalike_misreading_is_suggested_not_linked`; `test_review.py::test_red_fields_not_prefilled_and_ids_only_when_confirmed`, `test_suggestion_never_saved_as_id`; `test_pipeline_outputs.py::test_no_ids_on_unconfirmed_fields`, `test_kia_advertiser_held_as_suggestion`, `test_confirmed_links_keep_ids`; `test_master_data.py` (10 tests). Simulated stress: 2,648 OCR-corrupted names -> **0 wrong links** (`scripts/matcher_stress.py`). Real ads: 0 wrong IDs on set50 + holdout20 (`scripts/audit_phase2.py`).

**Risks / remaining work**
- **Mock master database** (`adreader/seed_data.py`: 161 brands, 230 products, 15 advertisers). Matching against the client's real master data is untested, and the high review rate for advertiser/product is caused by the mock data.
- Dealer rows: when a dealer row is not confirmed, the saved value keeps `suggested_advertiser_id` / `suggested_city_id` inside the dealer dict (`matcher._hold_unconfirmed_ids`, saved by `review.build_final` via `final_lists`). These are not stored as IDs, but a downstream consumer could misuse them. Decide whether to strip suggestions from the saved record.
- ID accuracy metric exists in the scorer (`*_id` label keys), but few labels carry IDs (8 on set50), so it is barely measured.

## 7. Automatic form population - COMPLETE (POC)

**Evidence**
- `review.initial_edits()` pre-fills the form from the record (confirmed values filled, amber values filled + flagged, red values shown as suggestions only); `streamlit_app.review_screen()` tabs (Identity, Creative, Price & offers, Contacts & dealers, Attributes), "use suggestion" buttons, OCR line panel.
- Save: `review.build_final()` -> `save_submission()` (SQLite `submissions` table), optional label export `save_label()`.
- Tested: `tests/test_app.py::test_kia_flow_review_gate_and_save` (form filled, gate blocks, reviewer ticks, saves).

**Remaining work**
- No export / API to the client's actual data-entry system (only SQLite + JSON). The planned FastAPI backend does not exist yet.
- Measured auto-fill: ~59-61% of labelled values filled as confirmed and correct (set50/holdout20).

## 8. Model / OCR comparison - PARTIAL

**Done**
- RapidOCR vs native PaddleOCR on all 20 held-out ads: fact recall 29/43 vs 28/43, 13.8 s vs 91.7 s per ad (different hardware), same script coverage. Recommendation recorded: keep RapidOCR, no PaddleOCR fallback (`EVALUATION_REPORT.md`, `scripts/compare_ocr.py`, `scripts/ocr_fallback_analysis.py`).
- Current model (Nemotron 120B + nano verifier) fully measured on set50 / holdout20 (bench200 pending).
- OCR + rules baseline without a vision model: 45.5% / 46.2% accuracy.

**Missing**
- **Qwen2.5-VL and Qwen3-VL: no results.** Code is ready (`kaggle/qwen_extract.py`, `scripts/kaggle_kernel.py`, `run_eval.py ingest`). The free Kaggle GPU job `adreader-qwen3vl4b` has been QUEUED for many hours; the CPU jobs `adreader-qwen3vl4b-cpu` and `adreader-qwen25vl3b-cpu` are RUNNING (still running at audit time); the local folders `data/eval/kaggle_out/adreader-qwen*-cpu/` contain only logs of earlier failed attempts. These are 3B/4B models on CPU, not the larger Qwen models the brief likely means.
- An earlier Kimi model test was dropped (too slow); Gemma endpoint 404.

**Remaining work:** collect the Qwen CPU outputs when they finish, then ingest -> validate -> score on holdout20 with the unchanged scorer. If unavailable, document it as a blocker with the evidence. Optionally score a hosted larger Qwen-VL via API (needs a paid account, your decision).

## 9. Accuracy, speed, infrastructure, cost comparison - PARTIAL

**Evidence**
- Accuracy: `docs/EVALUATION_REPORT.md` (all tables), per-field CSVs in `data/eval/*/report/`.
- Speed: per-stage timings in records (OCR ~15 s median on laptop CPU, extraction ~19-25 s, verifier ~3 s); usage log with latency / tokens / cost per call (`adreader/usage.py`, `scripts/worker.py status`).
- Infrastructure / cost: discussed (hosted API vs GPU server vs NVIDIA trial terms; ~$0.002-0.005 per ad estimate for hosted Qwen3-VL, server ~$15-40/month), but **only in the conversation**, not in a project document.

**Remaining work:** one comparison table (accuracy, silent errors, s/ad, hardware, cost/ad, licence/terms) covering RapidOCR, PaddleOCR, Qwen2.5-VL, Qwen3-VL, Nemotron, OCR+rules, with measured vs estimated values marked.

## 10. Recommended model and system architecture - MISSING (as a deliverable)

**Evidence of content:** pipeline description in module docstrings (`adreader/pipeline.py`, `validate.py`, `matcher.py`); OCR recommendation in `EVALUATION_REPORT.md`; architecture/production assessment (provider chain, job queue, failover, costs) given in conversation; reliability layer implemented (`jobs.py`, `providers.py`, `vlm.py`, `usage.py`).

**Remaining work:** a document with the recommended model setup (extractor + verifier + OCR), the architecture diagram (upload -> job -> OCR -> extraction -> verifier -> validation -> DB matching -> review form -> save), the rationale backed by benchmark numbers, the production path (paid provider, FastAPI, auth/multi-tenant) and the known limits. Note: the NVIDIA trial terms do not allow production use; the recommended production provider is still unvalidated.

## 11. Working POC and demo workflow - PARTIAL

**Evidence**
- Streamlit app (`app/streamlit_app.py`): pages Process advertisement, Jobs, Saved records, Database additions, Audit answers; background worker `scripts/worker.py`; app health checked running at http://localhost:8501.
- Tested headless: `tests/test_app.py` (3 tests). Last full suite: 113 passed (before the bench200 scripts were added; the pipeline is unchanged since).

**Remaining work**
- **Demo script / walkthrough** (which sample ads to show, expected flags, how ID safety and review are demonstrated, timing), plus a short screen recording if required.
- One-command start (app + worker) and a check that the NVIDIA key is configured.
- Re-verify the full manual click path after the latest changes (job queue, database additions).

## 12. Source code and setup documentation - PARTIAL

**Evidence:** code in `adreader/`, `app/`, `scripts/`, `kaggle/`, `tests/`; `requirements.txt`, `requirements-kaggle.txt`, `.env.example`, `pytest.ini`; `docs/EVALUATION_REPORT.md`, `docs/DEDUP_EVAL_PLAN.md`, `data/labels/FORMAT.md`.

**Remaining work**
- **No README** (setup, Windows notes such as Smart App Control / RapidOCR instead of PaddlePaddle, env vars, building the DB `scripts/build_db.py`, running app + worker, running tests and benchmarks).
- The project is **not a git repository**. Snapshots exist (`releases/phase2_validated/`, `releases/bench200_frozen/` with sha256 hashes); initialise git before handing in.
- Secrets: `.env` holds API keys. Keys were pasted in chat earlier and should be rotated. Make sure `.env` is not submitted.

## 13. Evaluation on 50 and 200 real ads - PARTIAL

**Evidence**
- 50: `data/eval/set50` (47 distinct ads + 3 degraded copies), results in `docs/EVALUATION_REPORT.md` (v1 88.3%, current 90.6%, 0 silent errors). **Labels: 2 human-verified, 45 Claude-drafted**; holdout20 (20 ads, Claude-drafted): 94.5%, 0 silent errors.
- 200: `data/eval/bench200` - 200 distinct creatives, 0 overlap with set50/holdout20 (`duplicate_audit.json`), 13 languages, 17 industries, quality high/medium/low; labels drafted (`_verified_by: null`); evaluation **running** at audit time; report generator `scripts/bench200_write_report.py` -> `reports/bench200_report.md`.

**Remaining work**
- Finish the bench200 run and report (provisional).
- **Human audit** of labels (set50: 45 pending, holdout20: 20, bench200: 200) using `scripts/bench200_audit.py` / the app's audit page. Until then, all numbers are provisional.

---

## Prioritised gap report

### A. Mandatory internship requirements (do first)
1. **Qwen2.5-VL / Qwen3-VL comparison (#8)**: get results or document the blocker with evidence. Required by the brief.
2. **Recommended model + architecture document with diagram (#10)**: a deliverable that does not exist yet.
3. **README / setup documentation (#12)** and **git repository** for the source code hand-in.
4. **Consolidated accuracy / speed / infrastructure / cost comparison (#9)** in one document, measured vs estimated marked.
5. **Demo workflow (#11)**: script and click-through of upload -> job -> review form -> save, with ID safety shown; record if needed.
6. **Finish the bench200 report (#13)** and state the provisional status clearly.
7. **Human audit of at least a sample of labels (#13)**: required before calling any accuracy number verified.

### B. Important for credibility (strongly recommended)
8. Automated test for the uploader -> job -> form path (#1).
9. Label and score description, offers, features, warranty, CTA on a subset (#3).
10. Decide on stripping `suggested_*` IDs from saved dealer rows (#6).
11. Test the matcher on a realistic master-data extract, if the client can provide one (#6).

### C. Optional improvements (after the deadline)
12. OCR for Bengali/Gujarati/Malayalam/Odia/Punjabi/Urdu (#4).
13. Regional lakh/crore-word OCR fix and regional headline re-read (proposals in `EVALUATION_REPORT.md` 3c).
14. Production items: validated paid provider + second provider, FastAPI backend, auth/multi-tenancy, hosting (from the reliability assessment).
15. Real multi-date dedup evaluation (data ready: 72 repeated creatives in the 400-ad pool; needs human group annotation).
