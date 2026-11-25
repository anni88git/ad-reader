# bench200 - 200-image generalisation benchmark

> **Status: PROVISIONAL.** 0 of 200 labels are human-verified. All other labels were drafted by Claude from the images (before any system output was seen) and are pending human audit (`data/eval/bench200/audit_queue.csv`). Accuracy figures below are relative to these provisional labels, not to independently verified ground truth.

## 1. Data and selection

**Integrity check (final manifest):** 200 images, 200 distinct creative clusters (all distinct); images from or clustered with set50/holdout20: 0; overlapping creatives excluded before selection: 2 (extra/e020 ~ set50/046, extra/e042 ~ set50/014).

- Downloaded: **400** Hindi/Marathi newspaper ads (4 papers, MAdVerse Epaper2) + **105** extra ads for other languages and industries (MAdVerse Epaper1/2, Advert_Gallery).
- The 400 contain only **304** distinct creatives (perceptual-hash distance <= 12; every sampled pair checked by eye was a true repeat).
- Excluded from the benchmark (logged in `duplicate_audit.json`): manual exclusion (visual duplicate or out of scope): 4; duplicate of a selected image: 1; creative already in set50/holdout20: 2. The manual exclusions are 3 language/city editions of an already selected creative that the hash missed, and 1 photo of an outdoor installation (not a newspaper ad).
- Selected: **200 distinct creatives** (one per cluster; none in set50/holdout20): 101 from the Hindi/Marathi pool and 99 from the extra downloads. Stratified by newspaper and image quality, fixed seeds (`scripts/select_bench200.py`, `scripts/fetch_bench_extra.py`).

| Source | Images |
|---|---|
| madverse_epaper2 | 62 |
| Navbharat Times | 31 |
| Amar Ujala | 30 |
| madverse_adgal | 27 |
| Maharashtra Times | 25 |
| Dainik Bhaskar | 15 |
| madverse_epaper1 | 10 |

**Language (printed, from labels)**

| Language | Images | Share |
|---|---|---|
| English | 71 | 35.5% |
| Hindi | 66 | 33.0% |
| Marathi | 15 | 7.5% |
| Malayalam | 9 | 4.5% |
| Tamil | 7 | 3.5% |
| Kannada | 7 | 3.5% |
| Telugu | 6 | 3.0% |
| Bengali | 5 | 2.5% |
| Odia | 4 | 2.0% |
| Punjabi | 4 | 2.0% |
| Gujarati | 3 | 1.5% |
| Urdu | 2 | 1.0% |
| Manipuri | 1 | 0.5% |

**Industry (from labels)**

| Industry | Images | Share |
|---|---|---|
| FMCG | 41 | 20.5% |
| Healthcare | 32 | 16.0% |
| Automobile | 21 | 10.5% |
| Fashion & Apparel | 16 | 8.0% |
| Retail & Consumer Electronics | 15 | 7.5% |
| Banking & Finance | 15 | 7.5% |
| Education | 13 | 6.5% |
| Jewellery | 8 | 4.0% |
| Real Estate | 8 | 4.0% |
| Travel & Hospitality | 7 | 3.5% |
| Government & Public Notices | 5 | 2.5% |
| Furniture & Home | 5 | 2.5% |
| Media & Entertainment | 5 | 2.5% |
| Food Delivery & Restaurants | 3 | 1.5% |
| Insurance | 2 | 1.0% |
| Unlabelled | 2 | 1.0% |
| Telecom & Internet | 1 | 0.5% |
| Recruitment | 1 | 0.5% |

**Image quality (resolution + sharpness)**

| Quality | Images | Share |
|---|---|---|
| medium | 79 | 39.5% |
| low | 65 | 32.5% |
| high | 56 | 28.0% |

Metadata issues found while labelling: the dataset's 'Assamese' ads are English / Manipuri (Meetei Mayek) ads from Manipur papers; one 'English' Advert_Gallery ad is in Hindi. Labels record what is printed.

## 2. Labels

- 200 label files in `data/labels/` (existing schema, `data/labels/FORMAT.md`), `_set: bench200`, `_label_source: claude_independent_read`, `_verified_by: null`.
- Fields: language, brand, product, ad type, category, headline, price, QR present; sub-headline, phones, websites, emails where clearly legible (otherwise in `_skip`). Long free-text fields are skipped.
- `any_of` is used for genuinely acceptable alternatives (e.g. nukta spellings, brand vs. company name, a top-level vs. sub-category).
- Fields the drafter was unsure about are listed in `_uncertain_fields`.
- **Human-verified: 0 / 200.** Audit workflow: contact sheets in batches of 25 (`data/eval/bench200/audit_sheets/`), queue `audit_queue.csv`, and `python scripts/bench200_audit.py apply` stamps only rows an auditor marked verified or corrected.

## 3. Frozen pipeline

- Extraction model `nvidia/nemotron-3.5-super-vl-120b-a12b`, verifier `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, prompt v2 (the Phase 2.5 validated version), thresholds {'CONFIRM': 0.85, 'REVIEW': 0.5, 'MATCH_ACCEPT': 92, 'MATCH_MARGIN': 6}.
- Not a git repository: sha256 of all 25 source files in `data/eval/bench200/FROZEN_PIPELINE.json`; full copy in `releases/bench200_frozen/`; previous release `releases/phase2_validated/`. No pipeline change during the evaluation.

## 4. Results (provisional)

System `nemotron+nano` (extraction + OCR checks + independent verifier). 200 ads evaluated.

Scoring (unchanged `scripts/evaluate.py`): every labelled single field and category attribute of every ad is one value. A value is correct if it matches any accepted alternative, and a red (unfilled) field counts as 'not filled'. Accuracy = (correct + correctly empty) / all values. The 95% interval is a bootstrap over ads, because fields of the same ad are correlated.

| Metric | bench200 |
|---|---|
| Values scored | 1614 |
| Accuracy | **86.4%** (1395 / 1614), 95% CI 84.7-88.2% |
| Correct extractions (incl. correctly left empty) | 1395 |
| Wrong values produced | 180 (169 wrong value, 11 false positives = value given where the label says not printed) |
| **Wrong values marked confirmed (silent errors)** | **22 / 777 confirmed (2.8%)** |
| Correct values auto-filled (confirmed) | 54.7% of values with a label |
| Values sent to human review | 574 |
| Correct values sent to review (unnecessary flags) | 35.5% |
| Wrong values caught (flagged) | 87.8% |
| Missed (label has a value, field left empty) | 39 |

Lists (precision / recall of items): websites 58/60 precise, 58/60 found; emails 24/24 precise, 24/25 found; phones 35/36 precise, 35/35 found

### Wrong values marked confirmed - the key finding

Unlike set50 and holdout20 (0 each), bench200 has **22 wrong values marked confirmed** (2.8% of confirmed values). By field: language 7, category 6, ad_type 5, headline 3, price 1. No brand, product, QR or contact value was confirmed wrongly; the confirmed errors are in judgement fields (language, category, ad type), plus 3 headlines and 1 price.

Claude's provisional reading (`silent_error_review.json`, not a human audit): **12 genuine system errors**, 10 debatable (label strictness / judgement). Genuine patterns:
- Marathi reported as Hindi (3 ads): same Devanagari script, and the language check cannot tell them apart.
- Punjabi (Gurmukhi) not reported (1).
- Gujarati headline with a mixed-in Devanagari letter (no Gujarati OCR to cross-check).
- Hindi 'fayde' (benefits) taken as a price (1).
- Admission ads classed as recruitment (2).
- A service classed as manufacturer (1).
- Two category confusions.

These were found on test data and are **not fixed** here (the pipeline is frozen); they are inputs for the next development round, to be validated on new data.

**By field**

| Field | Values | Accuracy | Wrong & confirmed | Auto-filled | Correct sent to review |
|---|---|---|---|---|---|
| ad_type | 200 | 83.0% | 5 / 123 | 59.0% | 28.9% |
| brand | 200 | 88.0% | 0 / 63 | 31.7% | 64.0% |
| category | 198 | 94.4% | 6 / 172 | 83.8% | 11.2% |
| headline | 199 | 70.4% | 3 / 46 | 21.6% | 69.3% |
| language | 200 | 89.5% | 7 / 169 | 81.0% | 9.5% |
| price | 199 | 89.4% | 1 / 9 | 22.9% | 65.2% |
| product | 200 | 77.0% | 0 / 6 | 4.5% | 93.2% |
| qr_code_present | 200 | 100.0% | 0 / 189 | 94.5% | 5.5% |
| sub_headline | 18 | 83.3% | 0 / 0 | 0.0% | 100.0% |

Groups marked ⚠ small n have fewer than 10 ads: their accuracy is indicative only (wide intervals).

**By language**

| Language | Ads | Values | Accuracy | 95% CI (ads) | Wrong & confirmed | Correct auto-filled | Correct sent to review | Missed |
|---|---|---|---|---|---|---|---|---|
| English | 71 | 582 | 89.9% | 86.9-92.5% | 10 / 298 | 58.9% | 33.5% | 11 |
| Hindi | 66 | 530 | 90.4% | 87.8-93.2% | 4 / 265 | 56.9% | 36.5% | 7 |
| Marathi | 15 | 119 | 80.7% | 73.9-87.4% | 3 / 52 | 45.8% | 41.7% | 3 |
| Malayalam ⚠ small n | 9 | 71 | 80.3% | 73.6-86.1% | 0 / 30 | 52.6% | 31.8% | 3 |
| Tamil ⚠ small n | 7 | 57 | 84.2% | 78.6-91.1% | 1 / 22 | 45.7% | 46.2% | 0 |
| Kannada ⚠ small n | 7 | 56 | 78.6% | 73.2-83.9% | 0 / 26 | 52.0% | 31.6% | 4 |
| Telugu ⚠ small n | 6 | 48 | 85.4% | 79.2-91.7% | 1 / 20 | 45.2% | 45.7% | 1 |
| Bengali ⚠ small n | 5 | 41 | 68.3% | 57.5-76.2% | 1 / 16 | 42.9% | 31.8% | 2 |
| Odia ⚠ small n | 4 | 32 | 78.1% | 65.6-87.5% | 0 / 17 | 63.0% | 22.7% | 3 |
| Punjabi ⚠ small n | 4 | 31 | 58.1% | 41.9-71.0% | 1 / 11 | 35.7% | 33.3% | 1 |
| Gujarati ⚠ small n | 3 | 24 | 70.8% | 62.5-75.0% | 1 / 10 | 47.4% | 25.0% | 1 |
| Urdu ⚠ small n | 2 | 16 | 81.2% | 75.0-87.5% | 0 / 7 | 53.8% | 30.0% | 3 |
| Manipuri ⚠ small n | 1 | 7 | 85.7% | 85.7-85.7% | 0 / 3 | 42.9% | 50.0% | 0 |

**By industry**

| Industry | Ads | Values | Accuracy | 95% CI (ads) | Wrong & confirmed | Correct auto-filled | Correct sent to review | Missed |
|---|---|---|---|---|---|---|---|---|
| FMCG | 41 | 333 | 89.2% | 85.8-92.2% | 4 / 176 | 59.5% | 32.3% | 8 |
| Healthcare | 32 | 258 | 87.2% | 82.2-91.8% | 3 / 114 | 48.1% | 43.9% | 3 |
| Automobile | 21 | 171 | 88.3% | 82.8-92.6% | 2 / 102 | 63.3% | 30.1% | 2 |
| Fashion & Apparel | 16 | 129 | 91.5% | 85.6-96.1% | 1 / 64 | 61.8% | 31.5% | 4 |
| Banking & Finance | 15 | 122 | 86.1% | 81.7-90.2% | 0 / 57 | 57.6% | 31.3% | 1 |
| Retail & Consumer Electronics | 15 | 121 | 90.1% | 84.4-94.3% | 3 / 64 | 58.1% | 35.1% | 1 |
| Education | 13 | 104 | 82.7% | 75.0-89.4% | 5 / 41 | 44.4% | 42.9% | 4 |
| Jewellery ⚠ small n | 8 | 64 | 89.1% | 82.8-95.3% | 0 / 26 | 53.1% | 40.9% | 0 |
| Real Estate ⚠ small n | 8 | 64 | 82.8% | 71.9-92.2% | 0 / 30 | 50.8% | 37.5% | 5 |
| Travel & Hospitality ⚠ small n | 7 | 57 | 80.7% | 69.6-91.5% | 2 / 25 | 48.9% | 36.1% | 2 |
| Government & Public Notices ⚠ small n | 5 | 40 | 65.0% | 62.5-70.0% | 1 / 12 | 35.5% | 35.3% | 4 |
| Furniture & Home ⚠ small n | 5 | 40 | 85.0% | 77.5-92.5% | 1 / 18 | 48.6% | 41.4% | 0 |
| Media & Entertainment ⚠ small n | 5 | 40 | 72.5% | 65.0-80.0% | 0 / 15 | 45.5% | 31.8% | 4 |
| Food Delivery & Restaurants ⚠ small n | 3 | 24 | 91.7% | 75.0-100.0% | 0 / 11 | 55.0% | 38.9% | 0 |
| Insurance ⚠ small n | 2 | 16 | 81.2% | 75.0-87.5% | 0 / 8 | 61.5% | 20.0% | 0 |
| Unlabelled ⚠ small n | 2 | 14 | 64.3% | 57.1-71.4% | 0 / 5 | 41.7% | 28.6% | 1 |
| Telecom & Internet ⚠ small n | 1 | 9 | 77.8% | 77.8-77.8% | 0 / 4 | 44.4% | 42.9% | 0 |
| Recruitment ⚠ small n | 1 | 8 | 100.0% | 100.0-100.0% | 0 / 5 | 83.3% | 16.7% | 0 |

**By quality**

| Quality | Ads | Values | Accuracy | 95% CI (ads) | Wrong & confirmed | Correct auto-filled | Correct sent to review | Missed |
|---|---|---|---|---|---|---|---|---|
| medium | 79 | 641 | 88.9% | 86.4-91.4% | 4 / 304 | 55.6% | 36.4% | 7 |
| low | 65 | 519 | 84.2% | 80.5-87.4% | 10 / 244 | 52.7% | 36.1% | 20 |
| high | 56 | 454 | 85.5% | 82.3-88.8% | 8 / 229 | 55.9% | 33.6% | 12 |

**By source**

| Source | Ads | Values | Accuracy | 95% CI (ads) | Wrong & confirmed | Correct auto-filled | Correct sent to review | Missed |
|---|---|---|---|---|---|---|---|---|
| madverse_epaper2 | 62 | 500 | 82.4% | 79.4-85.3% | 5 / 226 | 52.7% | 34.6% | 23 |
| Navbharat Times | 31 | 250 | 90.8% | 86.8-94.8% | 6 / 126 | 55.0% | 38.5% | 2 |
| Amar Ujala | 30 | 241 | 88.8% | 84.6-92.5% | 3 / 113 | 54.5% | 37.9% | 4 |
| madverse_adgal | 27 | 224 | 88.8% | 83.4-93.3% | 2 / 127 | 65.1% | 26.0% | 3 |
| Maharashtra Times | 25 | 201 | 86.6% | 81.0-92.0% | 4 / 95 | 51.7% | 38.9% | 4 |
| Dainik Bhaskar | 15 | 120 | 93.3% | 89.2-97.5% | 0 / 64 | 60.4% | 34.7% | 1 |
| madverse_epaper1 | 10 | 78 | 73.1% | 62.5-83.5% | 2 / 26 | 36.4% | 46.7% | 2 |

**Earlier benchmarks - kept separate** (not combined with bench200)

| Set | Values | Accuracy | Wrong & confirmed | Auto-filled |
|---|---|---|---|---|
| set50 (development set) | 532 | 90.6% | 0 / 258 | 58.9% |
| holdout20 (held-out) | 199 | 94.5% | 0 / 96 | 61.1% |
| **bench200 (new, provisional)** | 1614 | **86.4%** | **22 / 777** | 54.7% |

## 5. Deduplication data

The 400-ad pool has **72 repeated creatives** (126 repeat pairs, 96 surplus insertions): 31 repeat within one paper (different dates), 41 across papers, 15 are Hindi/Marathi editions. That is enough for a separate dedup evaluation (plan: `docs/DEDUP_EVAL_PLAN.md`; dedup code frozen in `data/eval/dedup_real/FROZEN.json` before this data was seen). It still needs human annotation of the groups; pHash clusters are candidates, not ground truth.

## 6. Limitations

1. **Labels are provisional** (Claude-drafted, human audit pending). Errors in labels count against the system and vice versa.
2. The labeller and the evaluated pipeline are different models, but the labeller sometimes accepts several alternatives (`any_of`); strictness differs by field.
3. Small groups (e.g. Urdu, Punjabi, Manipuri, Gujarati: 1-4 ads) have very wide intervals - indicative only.
4. OCR covers Latin, Devanagari, Tamil, Telugu and Kannada. Bengali, Gujarati, Malayalam, Odia, Punjabi, Urdu and Meetei Mayek ads rely on the vision model plus the image-only verifier.
5. Image quality is a heuristic (resolution + sharpness), not a human rating.
6. Language edition / near-identical creatives were removed where found; a few more may remain undetected.

## 7. Reproduce

```
python scripts/inspect_pool.py                      # quality + duplicate clusters
python scripts/select_bench200.py                   # manifest + duplicate_audit.json (fixed seed, lock file)
python scripts/run_eval.py ocr --set bench200
python scripts/run_eval.py extract --set bench200 --system nemotron
python scripts/run_eval.py verify --set bench200
python scripts/run_eval.py validate --set bench200 --system nemotron --verifier-tag nano
python scripts/evaluate.py --set bench200 --systems nemotron+nano
python scripts/bench200_report.py                   # breakdowns + distribution_report.json
python scripts/bench200_write_report.py             # this file
python scripts/bench200_audit.py status             # human audit progress
```
