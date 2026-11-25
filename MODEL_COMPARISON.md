# Model & rule-set comparison

All figures are copied from the benchmark files in `releases/` (no rounding beyond the source):

- `releases/bench200_frozen/baseline_results/bench200_report/summary.json` - frozen baseline
- `releases/candidate_v4/results/{bench200,set50,holdout20}_v4_vs_v3.json` - candidate_v3 vs candidate_v4

> **Read this first.** Benchmark labels are **provisional**: they were drafted from the images (before system output
> was seen) and are pending human audit - 0 of the 200 bench200 labels are human-verified. Accuracy is therefore
> relative to provisional labels. candidate_v4 was developed after examining these sets, so its figures are a
> development retest, not independent proof.

## Systems compared

| System | Extraction model | Verifier model | Validation rules |
|---|---|---|---|
| bench200_frozen (baseline) | nvidia/nemotron-3.5-super-vl-120b-a12b (prompt v2) | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning | Phase 2.5 validated rules (frozen 2026-11-24) |
| candidate_v3 | same | same | + stricter evidence gates |
| candidate_v4 | same | same | v3 + 5 offline fixes (`accuracy_rules_v4.py`); switched values always stay in review |

The models are identical across systems; the gains come from validation rules applied to the same model outputs.

## bench200 - 200 distinct ads, 1,614 scored values

| Metric | Frozen baseline | candidate_v3 | candidate_v4 |
|---|---|---|---|
| Accuracy | 86.4% | 87.0% | **89.0%** |
| Confirmed (auto-filled) values | 777 | 763 | 763 |
| Silent errors (wrong value auto-confirmed) | 22 | 8 | **8** |
| Silent-error rate | 2.8% | 1.0% | 1.0% |
| Confirmed precision | 97.2% | 99.0% | **99.0%** |
| Auto-fill rate | 54.7% | 54.7% | 54.6% |
| Error catch rate | 87.8% | 95.3% | 94.8% |
| Over-flag rate (correct values sent to review) | 35.5% | 35.9% | 37.9% |
| Missed values | 39 | 39 | 23 |

Frozen baseline extras: English ads 88.8% vs regional-language ads 86.0% accuracy; list fields precision / recall -
websites 96.7% / 96.7%, emails 100.0% / 96.0%, phones 97.2% / 100.0%.

### bench200 per field - candidate_v3 → candidate_v4

| Field | Values | v3 accuracy | v4 accuracy | v3 silent errors | v4 silent errors |
|---|---|---|---|---|---|
| ad_type | 200 | 83.0% | 88.5% | 2 | 2 |
| brand | 200 | 88.0% | 91.0% | 0 | 0 |
| category | 198 | 94.4% | 94.4% | 2 | 2 |
| headline | 199 | 70.4% | 71.9% | 2 | 2 |
| language | 200 | 92.5% | 95.0% | 2 | 2 |
| price | 199 | 91.0% | 95.0% | 0 | 0 |
| product | 200 | 77.0% | 77.0% | 0 | 0 |
| qr_code_present | 200 | 100.0% | 100.0% | 0 | 0 |
| sub_headline | 18 | 83.3% | 83.3% | 0 | 0 |

## set50 and holdout20

| Metric | set50 v3 | set50 v4 | holdout20 v3 | holdout20 v4 |
|---|---|---|---|---|
| Ads / scored values | 50 / 492 | 50 / 492 | 20 / 199 | 20 / 199 |
| Accuracy | 92.7% | 92.7% | 94.5% | **95.5%** |
| Silent errors | 0 | 0 | 0 | 0 |
| Confirmed precision | 100.0% | 100.0% | 100.0% | 100.0% |
| Auto-fill rate | 59.1% | 59.0% | 59.9% | 59.9% |
| Missed values | 6 | 6 | 2 | 0 |

## How to read these numbers

- **Silent errors** matter most for data entry: they are wrong values the system would have accepted without a human
  look. The rule changes cut them from 22 to 8 on bench200 while keeping the auto-fill rate essentially unchanged.
- **Over-flagging** (correct values sent to review) is the cost side: it rose slightly with v4 because switched values
  are deliberately kept in review.
- The snapshot code for each system is preserved in `releases/` and verified against its `SHA256SUMS`, so every
  number above can be reproduced from the exact code that produced it.
