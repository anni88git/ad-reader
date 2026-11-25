# Accuracy improvement & evaluation report

*AI-powered newspaper-advertisement data entry - POC. Date: 2026-11-24.*

## 1. Ground truth - what the numbers are measured against

| Label set | Ads | Source | Status |
|---|---|---|---|
| set50 | 2 | Human-checked in the review tool (Kia, Horlicks) | **human-verified** |
| set50 | 45 | Drafted by Claude from the images (not from system output) | machine-drafted, not human-verified |
| set50 | 3 | Degraded copies - reuse the original's label | as above |
| holdout20 | 20 | Drafted by Claude from the images **before** any system output was seen | machine-drafted, not human-verified |

Automatic label check (set50): every drafted label was compared with two independent model
readings. 6 fields were *suspicious* (both models disagreed with the label) - on inspection all 6
labels were correct and the models wrong (benefit/cashback amounts read as prices, a cropped
brand name, dealer vs manufacturer). 26 fields were *ambiguous* (models disagree with each other).
2 label errors were found and corrected earlier (a printed model series; Hindi in a logo).
**All accuracy figures below are relative to these labels; they are not human-verified ground truth.**

## 2. Results

Metrics: **accuracy** = share of scored field values correct; **silent errors** = wrong values the
system marked CONFIRMED (the critical number); **auto-fill** = correct values filled without
review; **errors caught** = wrong values that were flagged; **false flags** = correct values sent
to review.

### Baseline - set50, first run (honest, nothing tuned)
| System | Accuracy | Silent errors | Auto-fill | Errors caught | False flags |
|---|---|---|---|---|---|
| **Nemotron + OCR + checker (v1)** | **88.3%** | **14 / 312 (4.5%)** | 68.2% | 69.6% | 22.2% |
| OCR + rules, no vision model | 45.5% | 105 / 227 (46.3%) | 29.8% | 16.7% | 6.2% |

### After tuning - set50 (optimistic: these ads were used to find the fixes)
| Version | Accuracy | Silent errors | Auto-fill | Errors caught | False flags |
|---|---|---|---|---|---|
| + rule fixes | 89.3% | 7 / 308 (2.3%) | 68.7% | 82.9% | 22.4% |
| + headlines always reviewed | 89.3% | 2 / 251 (0.8%) | 56.8% | 95.1% | 35.8% |
| + independent verifier | 89.3% | 0 / 263 (0.0%) | 60.0% | 100% | 32.2% |

### Held-out - 20 unseen ads (the honest test of the fixes)
| System | Accuracy | Silent errors | Auto-fill | Errors caught | False flags |
|---|---|---|---|---|---|
| Nemotron + checker (headlines reviewed) | **94.5%** | **1 / 90 (1.1%)** | 56.7% | 88.9% | 39.0% |
| + independent verifier | 94.5% | 3 / 100 (3.0%) | 61.8% | 66.7% | 33.6% |
| OCR + rules | 46.2% | 14 / 59 (23.7%) | 28.8% | 72.5% | 11.8% |

Phones, websites, e-mails on the held-out set: 100% precision and recall. Average time ~47 s/ad
(laptop CPU OCR ~19 s + hosted model ~20 s + verifier ~3 s).

**Held-out silent errors:** h12 variant "1.2 **I** Eco TORQ" (OCR misread of "l", copied by the model);
h06 and h08 sub-headline - with the verifier, *both* models picked the same wrong line, and their
agreement confirmed it. After this finding the sub-headline is **never auto-confirmed** (applied
after measurement; the table above is the measured result before that change).

### Held-out - after changes made *because of* the held-out errors (not an unbiased test)
The held-out set has now been used twice, so these rows are post-hoc. They show that the changes
remove the observed errors without hurting accuracy. They do **not** show how often the changes help on new ads.

| Change (all only add review flags, cannot create a confirmed value) | System | Accuracy | Silent errors | Auto-fill | Caught | False flags |
|---|---|---|---|---|---|---|
| Sub-headline never auto-confirmed | + verifier | 94.5% | 1 / 95 (1.1%) | 59.9% | 88.9% | 35.6% |
| + OCR recall (missed phones / registration IDs), look-alike characters, language-vs-script | Nemotron | 94.5% | 0 / 89 | 56.7% | 100% | 39.0% |
| same | + verifier | 94.5% | 0 / 94 | 59.9% | 100% | 35.6% |

Registration / reference IDs on the held-out set: recall 6/6 after the scorer was fixed to accept
an ID printed with its label ("MahaRERA No. P5180..." vs "P5180...").

### Sensitivity to uncertain labels
`check_labels.py` marks (never changes) label fields where the two models disagree with each other
(*ambiguous*) or both disagree with the label (*suspicious*). Removing every marked field:
set50 + verifier **91.6%, 0 silent errors**, auto-fill 60.1%; held-out **95.2%, 0 silent**, auto-fill 63.3%.
The conclusions do not depend on the debatable labels.

### Adversarial benchmark (automated, every ad)
Six realistic errors planted into every ad's real extraction (wrong-line headline, one-character
spelling change, price digit, phone digit, registration-number digit or fake ID, fake website):
**0 of 248 leaked on set50, 0 of 98 leaked on the held-out set** - with and without the verifier.

## 3. Root causes of the remaining errors and what was changed

| Root cause (from set50 errors) | Fix (general, not per example) |
|---|---|
| Regional lakh/crore words not parsed ("₹14.99 లక్ష" -> ₹15) | Lakh/crore/thousand words for 7 scripts; unknown word after a decimal amount is flagged |
| Benefit / savings / cashback amounts read as prices | "benefit", "savings", "worth", "bonus", "exchange" classified as offers; both model prompts told the same |
| Advertiser contradiction not flagged (manufacturer ad listing dealers) | Consistency rule now flags the advertiser too, with the dealer names as suggestion |
| A misread "₹" counted as evidence of Hindi | A script counts as present only with >= 3 letters of it |
| Real but wrong line chosen as headline | Headline always reviewed unless an independent model agrees; prominence check flags a clearly larger unused line |
| OCR spelling copied by the model (Tamil, Telugu, "n Duty") | Independent verifier reads the image only (no OCR, no primary answer) - disagreement flags; headline starting lowercase flagged |
| Two models agreeing on the wrong secondary line | Sub-headline never auto-confirmed |
| Phones / registration numbers in small print missed by the model | OCR recall: phone-shaped and RERA/UIN/IRDAI/CIN/GSTIN-shaped text not in the record is added as REVIEW |
| OCR look-alikes copied by the model ("1.2 I Eco" for "1.2l") | Variant / product / ID values mixing digits with I, l, O, S, B, Z are never auto-confirmed |
| Language list missing a script that is clearly printed | >= 3 confident OCR lines in an Indic script missing from the language list -> REVIEW |

Regression tests: `tests/test_accuracy_rules.py` (28 cases); full suite 67 tests pass, red team 14/14.

## 3b. Batch analytics and campaign deduplication

`adreader/dedup.py` + `scripts/batch_analytics.py` (no API calls; runs on saved records).
Each ad gets an image perceptual hash, its creative text with digits removed, its numbers
(prices, offers, phones) and brand / product / headline. Pair relations:
**duplicate** (same creative re-run), **offer changed** (same creative, different price/offer, with
the change listed), **new version** (same brand, shared headline/copy, new layout), **same product**
(reported only, never merged). Different brands never link.

| Test | Result |
|---|---|
| Dedup benchmark: 70 real ads + 15 simulated re-runs (rescale, crop, blur, JPEG, brightness) + 10 re-runs with a changed price + 3 real degraded copies + 1 real repeat | 29/29 links correct, 0 missed, 0 false links |
| Same benchmark, 5 fresh random seeds (different ads / edits) | 29/29 each time |
| Real data (70 ads) | 66 unique creatives. Found an actual repeat: the same Rupa creative in *Navbharat Times* and *Dainik Bhaskar* (confirmed by eye) |

**Status: development benchmark, not production accuracy.** 28 of the 29 expected links are
**simulated** (15 re-scans, 10 price edits, 3 of our own degraded copies). One is **real** (the Rupa
pair), and the system found it itself, so it is not an independent test. The thresholds were set
while looking at the first run (it found two false links: two different Toyota Yaris campaigns and
two Renault ads with the same dealer list). Fresh seeds pass, but with simulated repeats.
The real multi-date protocol is in [DEDUP_EVAL_PLAN.md](DEDUP_EVAL_PLAN.md).

Code review fixes, made after the numbers above: brand comparison used substring matching ("Kia"
matched "No*kia*"), so it now compares whole words. The image hash median wrongly excluded a row of
coefficients instead of just the DC term. Comparisons are now blocked by brand, so thousands of ads
scale. Benchmark unchanged after the fixes (29/29, 6 seeds). The dedup code is isolated: nothing in
extraction, validation, scoring or labels imports it.

Analytics output (`data/outputs/analytics/report.md`): repeated campaigns, category / language /
ad-type mix, prices by category (confirmed prices only; EMI / service / rent prices excluded),
offer prevalence, review load per field.

**Review-load finding:** advertiser (97%) and product (85%) are the most-flagged fields, mostly
because of the **mock master database** (15 advertisers): 49 advertiser flags are "no existing
record matches", and 15 product flags are "brand not in DB". This is the never-guess-an-ID rule
working as intended. With the real master data this load should drop sharply.

### Master-data workflow (advertisers / products / brands)

Each lookup now records *why* it did or did not link (`db_match`, information only: statuses, IDs
and scores are unchanged; held-out and set50 scores re-checked as identical):

| Outcome | Meaning for the reviewer | brand | product | advertiser |
|---|---|---|---|---|
| matched | existing record found | 47 | 8 | 10 |
| unknown | nothing similar: most likely a **new** entity | 11 | 26 | 49 |
| near_miss | similar records exist: **misreading** of one of them, or new | 9 | 4 | 8 |
| ambiguous | several records fit equally: pick one | 0 | 1 | 0 |
| blocked | product not linked because its brand is not linked | - | 15 | - |

(67 ads, set50 + held-out, nemotron + verifier.) The not-found rule is unchanged: no ID without a
match, and no automatic record creation. New: the reviewer can add a checked value as a new
brand / product / advertiser (`adreader/master_data.py`, review form, "Database additions" page).
- Exact duplicates are refused ("link it instead").
- Similar records must be explicitly confirmed as different, the guard against adding a misreading as a new entity.
- A product needs a linked or just-added brand.
- All requested additions are validated before any is created.
- Every addition is audited (who, when, source ad, evidence, acknowledged look-alikes) and can be undone while no saved record uses it.

Tests: `tests/test_master_data.py` (10).

### OCR engine: should native PaddleOCR be a selective fallback? (all 20 held-out ads measured)

| | RapidOCR (pipeline default) | Native PaddleOCR | Union (both) |
|---|---|---|---|
| Facts read verbatim (headline, phone, website, price) | 29/43 (67.4%) | 28/43 (65.1%) | 30/43 |
| Headline words read exactly (mean) | 87.5% | 85.5% | - |
| Time per ad | 13.8 s (laptop CPU) | 91.7 s (Kaggle CPU) | - |
| Scripts | Latin, Devanagari, Tamil, Telugu, Kannada | same; **no Bengali or Gujarati model** either | - |

Per ad: PaddleOCR adds one fact (h06 English headline) and misses two phone numbers that RapidOCR
reads (h02, h03). On regional ads, facts are identical. The only regional difference is h17
(Marathi), where PaddleOCR read 3/4 headline words vs 2/4, a single case.
Difficulty triggers available before a second OCR (low OCR confidence, blur warning, regional
script) selected ads where PaddleOCR gained **0** facts.

**Recommendation: keep RapidOCR as the default. Do not add PaddleOCR as an automatic fallback.**
On this data there is no trigger that predicts when it helps, it does not cover the unsupported
scripts, and it costs ~6x the time (speeds are not on the same hardware). A possible narrow use is
an on-demand "second read" for a flagged regional headline, but that rests on one example and would
need its own test. Unsupported scripts (Bengali, Gujarati, ...) need a different OCR model, not PaddleOCR.

Caveats: 20 ads (8 regional); facts come from machine-drafted labels; PaddleOCR used images downscaled
to 2500 px for the two largest ads (it ran out of memory at 4000 px), while RapidOCR caps at 4000 px.
Reproduce: `python scripts/compare_ocr.py --set holdout20 --paddle data/eval/kaggle_out/paddle_ocr_all20.jsonl`
and `python scripts/ocr_fallback_analysis.py`.

## 3c. Phase 2 - extraction refinements (2026-11-24)

**Audit** (`scripts/audit_phase2.py`, 67 labelled real ads, nemotron + verifier):
- Database linking on real ads: brand 44 correct links / **0 wrong**; product 7 / 0; advertiser 3 correct
  and 1 wrong value (Kia India instead of its dealers: flagged, ID held back, so not silent).
  Missed links: 4, all because the *value* differs from the record (e.g. "Horlicks Chocolate Flavour
  Delight" vs "Horlicks Chocolate Delight"), not because of matching.
- Simulated OCR misreadings of every DB name (`scripts/matcher_stress.py`): 2,648 brand/advertiser
  corruptions produced **0 wrong links** (correct link or no link only).
- Flags on correct values, by cause:
  - Policy, kept: sub-headline always reviewed (40/40).
  - Mock-DB coverage: product (33/40) and advertiser (11/13) "no record".
  - Genuine model disagreements, kept: headline wording, ad type, regional spelling.
  - **Mechanical false alarms, fixed:** see the table below.

**Changes and before/after** (unchanged labels and scorer; set50 = development set; held-out
reported separately):

| Change | Evidence |
|---|---|
| Verifier ISO language codes ("hi", "en") normalised before comparing | parsing fix |
| Category: system top level vs verifier's sub-category *of the same top level* = agreement (child-vs-parent and other branches still flag) | logic fix |
| QR detection: two detectors x colour/upscaled/grey; accepted only if decoded or located by >= 2 methods | real codes found 7/13 -> **12/13**, false detections 0/54 (all 67 ads, held-out included, were looked at, so held-out is not independent for this rule) |
| OCR look-alike *suggestion* for unmatched brand/advertiser ('Kla' -> Kia); field stays in review, no ID set | simulated: 486/904 unlinked brand misreadings now get the correct suggestion, 0 wrong suggestions; no effect on real-ad scores (by construction) |

| | set50 + verifier: before -> after | held-out + verifier: before -> after |
|---|---|---|
| Accuracy | 90.2% -> 90.2% | 94.5% -> 94.5% |
| Silent errors | 0 -> **0** | 0 -> **0** |
| Auto-filled correct values | 249 -> 258 (56.8% -> 58.9%) | 94 -> 96 (59.9% -> 61.1%) |
| False flags | 35.8% -> 33.5% | 35.6% -> 34.2% |
| Errors caught | 100% -> 100% | 100% -> 100% |

By field (set50): category false flags 16.7% -> 10.4%, language 19.6% -> 15.2%, QR 12.0% -> 4.0%.
Safety re-checks: red team 14/14, adversarial benchmark 0/248 (set50) and 0/98 (held-out) leaks,
nemotron-only silent errors unchanged (2 on set50). Cost: QR detection adds up to ~3-5 s per ad
without a QR code (all detector passes run); cached per image.

**OCR audit by script** (share of labelled text found verbatim in the OCR output; machine-drafted labels):

| Script (ads) | Mean confidence | Brand | Headline | Product |
|---|---|---|---|---|
| Latin (34) | 0.963 | 32/34 | 21/34 | 26/28 |
| Devanagari (20) | 0.928 | 14/20 | 11/20 | 10/18 |
| Telugu (4) | **0.859** | **1/4** | 1/4 | - |
| Tamil (2), Kannada (2) | 0.95 / 0.94 | 2/2, 2/2 | 1/2, 0/2 | 1/1, 1/1 |
| Bengali (3), Gujarati (2): **no OCR model** | - | Latin brand names only | 1/3, 1/2 | - |

Prices: 5 of 8 "missing" prices are fine (printed as "9.16 Lakh", parsed correctly). The 3 real
failures are all regional lakh/crore words next to the number (Telugu read as "še", Kannada word
dropped, Marathi "65 लाख" unread): the script-specific recogniser only runs when the Latin reading
looks doubtful.

**Proposed OCR work, highest impact first (not implemented; each to be measured on set50 first):**
1. Run the script-specific recogniser on every box adjacent to a number when the ad has that
   script, to catch lakh/crore words. Target: the 3 price misses.
2. Higher-resolution re-read of the largest text block in regional ads, aimed at headline
   accuracy (Devanagari 11/20, Telugu 1/4). Measured by headline word recall.
3. Telugu: lowest confidence; compare PP-OCRv5 Telugu vs the server model on the 4 Telugu ads.
4. Bengali / Gujarati / Malayalam / Odia / Punjabi have no OCR. Evaluate an OCR with these
   scripts on the 5 ads we have, before adding it. Until then those ads rely on the image-only verifier.

## 3d. Phase 2.5 - focused accuracy pass (2026-11-24)

**Error audit**: all 63 accuracy errors (52 on set50, 11 on held-out), nemotron + verifier,
classified from the predictions, labels, OCR, verifier answers and checker reasons:

| Cause | Errors |
|---|---|
| OCR misreading copied by the model (Indic vowel signs, dropped stylised first letter, I/l, lakh words, no-OCR scripts) | 14 |
| Brand / advertiser identification (6 of them dealer-vs-manufacturer) | 10 |
| Degraded copy of the Kia ad (unverifiable values withheld, attributes not extracted) | 10 |
| Product identification (granularity, campaign line taken as product) | 8 |
| Headline / sub-headline line selection | 8 |
| Missing information | 3 |
| Hindi spelling variants (nukta) | 3 |
| Language hallucination | 3 |
| Price interpretation | 2 |
| Correct value withheld by validation | 1 |
| Category (knock-on from a missed brand) | 1 |

**Fix 1: language rule (kept).** A language is removed only when OCR (which has a recogniser for
that script) found none of its text AND the image-only verifier does not list it. The field stays in review.

| | before | after |
|---|---|---|
| set50 (development) accuracy | 90.2% | **90.6%** |
| set50 errors fixed / new | - | 2 fixed (005, 016 language) / **0 new** |
| held-out accuracy (no-harm check; the pattern does not occur there) | 94.5% | 94.5% (0 fixed / 0 new) |
| silent errors (all confirmed values, both sets) | 0 / 0 | 0 / 0 |
| auto-fill | 58.9% / 61.1% | unchanged |
| red team / planted errors | 14/14, 0/248, 0/98 | 14/14, 0/248, 0/98 |

**Fix 2: extraction prompt v2.5 (rejected).** It targeted the largest category (OCR misreadings:
known OCR failure patterns, "the image spelling wins") plus product granularity ("editions/trims go
to model_variant"). It ran on set50 only, together with a same-day re-run of the current prompt to
measure noise:

| set50 + verifier | accuracy | silent errors | auto-fill |
|---|---|---|---|
| stored v2 outputs (+ language rule) | 90.6% | 0 | 58.9% |
| v2 re-run, same day | 90.4% | 0 | 58.4% |
| **v2.5** | **90.0%** | **4** | 58.2% |

- v2.5 vs the v2 re-run: 9 errors fixed (e.g. "n Duty" -> "On Duty", 3 products), 11 new.
- All 4 silent errors are model_variant values the product rule created ("Eye Relief Drops",
  "FCFS Scheme Phase 4", ...). model_variant is confirmed when its text is in the OCR, so a
  misplaced but printed phrase passes.
- Latency +2 s per ad (24.5 s vs 22.5 s median) and about 7% more input tokens (cost $0 on the trial).
- Not evaluated on held-out: it was rejected on the development set.

**Measurement limit found:** re-running the *same* prompt changes 9 field outcomes on set50
(4 fixed, 5 broken). Prompt changes worth less than about +/-1 point cannot be told apart from
run-to-run noise with 50 ads.

**Decision.** Keep v2 prompt + language rule; freeze the extraction pipeline.
- Rollback snapshot of the Phase 2 version: `releases/phase2_validated/`.
- Remaining error categories, documented: OCR misreadings (needs better OCR, not prompting),
  dealer-vs-manufacturer judgement, degraded images, line selection.
- New residual risk exposed by v2.5: model_variant has weak validation. Under the v2 prompt it
  produced 0 silent errors on both sets.

## 4. Independent verification

- Verifier: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (different model from the 120B extractor).
- Independence by construction: it receives **only the image** - no OCR text (cannot copy OCR
  errors) and no primary answer (cannot simply agree). Field-specific strict agreement; any
  disagreement -> REVIEW with the verifier's reading as the suggestion.
- Limitation shown by the held-out set: agreement is weak evidence for subjective choices
  (which line is the sub-headline) - hence the rule above. Both models are from the same vendor;
  a different-vendor verifier (Qwen) would be more independent.

## 5. Remaining risks

1. **Labels are not human-verified** (except 2 ads). Machine checks found no label errors among the
   suspicious cases, but a human audit of a sample is still the only proof.
2. **Small held-out set** (20 ads; 8 regional). Silent-error rates of 1-3% have wide uncertainty.
3. **Scripts without OCR** (Bengali, Gujarati, Malayalam, Odia, Punjabi, Urdu): no independent text
   check - most fields go to review; spelling errors there rely only on the verifier.
4. **Character-level OCR confusions copied by both** OCR and model can still pass when they are
   not digit/letter mixes (the look-alike rule covers only those).
5. **Small print**: phones / IDs the OCR itself cannot read on low-resolution images are still missed;
   the recall rule only helps when OCR reads them.
6. More review work: ~34-39% of correct values are flagged; this is the price of the near-zero
   silent-error target.

## 6. Is the Qwen comparison still needed?

**Yes, for the assignment, but no longer for safety.** The brief explicitly asks to evaluate
Qwen3-VL / Qwen2.5-VL / PaddleOCR on accuracy, speed, infrastructure and cost; Qwen is also the best
candidate for a *different-vendor* verifier. The Kaggle GPU job (Qwen3-VL-4B, single GPU) is still
queued on the free tier; the evaluation harness (`run_eval.py ingest`) is ready to score its output
the moment it finishes. The system's safety no longer depends on it.

## Reproduce
```
python scripts/run_eval.py validate --system nemotron --verifier-tag nano
python scripts/evaluate.py --systems nemotron nemotron+nano ocr_rules            # set50
python scripts/evaluate.py --set holdout20 --systems nemotron nemotron+nano ocr_rules
python scripts/adversarial_bench.py --verifier-tag nano
python scripts/check_labels.py
python -m pytest -q
```
