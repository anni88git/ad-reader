# Campaign deduplication - real multi-date evaluation plan

*Status: plan, not run. Date: 2026-11-24.*

## 1. What exists today, and what it does and does not show

| Evidence | Pairs | Kind | What it shows |
|---|---|---|---|
| Re-scanned copies (scale, crop, blur, JPEG, brightness) | 15 | **simulated** | Image hash + text survive print/scan noise |
| Price edited in the record + re-scan | 10 | **simulated** | A changed number is reported as an offer change, not a new ad |
| set50 degraded copies (our own blur/JPEG) | 3 | **simulated** | Same as re-scans, with real extraction noise on the degraded copy |
| Rupa creative in *Navbharat Times* and *Dainik Bhaskar* | 1 | **real** | One real cross-newspaper repeat. Found by the system and confirmed afterwards, so it is **not** an independent test |
| No false links among 70 real ads (66 unique creatives) | - | real, small | No false merges on this sample |

The **29/29 is a development result.** The thresholds were set while looking at the first run, which
found two false links. It must not be reported as production accuracy. It does not measure:
- real multi-date repeats
- real price/offer edits by an advertiser (re-typeset layouts, moved boxes)
- recall on real repeats nobody pointed the system at
- scale (hundreds of ads per brand)

## 2. Definitions given to annotators

| Relation | Definition | Counted as |
|---|---|---|
| duplicate | Same creative: same layout and copy; size, crop, print quality and paper may differ | 1 creative, n insertions |
| offer changed | Same creative, but a price / offer / date / phone changed | 1 creative (versioned), n insertions |
| new version | Same campaign (brand + headline or main copy), new layout or new main visual | same campaign, new creative |
| same product | Same brand + product, unrelated creative and copy | different campaigns |
| unrelated | Anything else | - |

Edge rules: a dealer ad and the manufacturer's ad for the same car are different campaigns.
City editions of the same page are duplicates.

## 3. Data

Need: the same newspapers on consecutive dates, so real repeats occur naturally.

| Option | Source | Pros / cons |
|---|---|---|
| A (preferred) | Client's own archive: 3-4 papers x 10-14 consecutive days, incl. 1-2 regional | Real target distribution; real repeats and offer edits |
| B | Public e-paper editions, used within each site's terms | Similar, but access and terms vary per paper |
| C | MAdVerse Epaper pool (already used) | No dates, so it tests cross-newspaper repeats only; the Rupa pair shows repeats exist there |

Target size: **300-500 ads**, with at least 40 real repeat pairs and at least 10 real offer
changes. Expect roughly 15-30% of ads in a 2-week window to be repeats.

## 4. Annotation (independent of the system, low effort)

Recall cannot be measured if annotators only check pairs the system proposes. So:
1. Run extraction (brand only is needed) and print a **contact sheet per brand**: thumbnails of
   every ad of that brand, sorted by date. Ads with no brand go on a separate sheet.
2. The annotator draws groups on each sheet (same creative / offer changed / new version), about
   1 minute per brand. They never see the system's dedup output.
3. A second annotator does 20% of the brands, and agreement (Cohen's kappa) is reported. Low
   agreement means the definitions need fixing, not the system.
4. Risk: a wrong brand in extraction can hide a repeat on another sheet. The unbranded sheet plus
   a cross-brand image-hash scan (distance <= 10) gives the annotator a list of candidates to check.

## 5. Protocol

1. **Freeze** before the test data is seen: `adreader/dedup.py` thresholds and the sha256 of
   `dedup.py` and `batch_analytics.py`, written into the result file.
2. **Split** by date: the first 30% of days are *dev* (thresholds may be adjusted, and must be
   re-frozen afterwards). The last 70% are *test*, scored **once**.
3. Report test and dev separately. Simulated benchmarks stay in a separate table.

## 6. Metrics

| Metric | Why |
|---|---|
| Pairwise precision / recall per relation (duplicate, offer changed, new version) | Basic correctness |
| **False-merge rate** (different creatives or campaigns merged) | Worst error for competitive intelligence: undercounts a competitor |
| Missed-repeat rate | Overcounts a campaign |
| Unique-creative count error: abs(predicted - true) / true | What the analytics report shows |
| B-cubed precision / recall of campaign clusters | Cluster quality, insensitive to cluster size |
| Offer-change detail accuracy (old -> new value correct) | The "price changed" claim is shown to users |
| Breakdowns: language / script, print quality, brand present vs missing | Where it fails |

Proposed acceptance (to agree before testing): duplicate precision >= 98%, recall >= 90%,
false-merge rate <= 1% of campaigns.

## 7. Known weaknesses to watch

- A creative re-typeset by the advertiser (same copy, new layout) can fall between duplicate and new version.
- Dealer ads that share a manufacturer template but have different dealers: duplicate or not? The
  definitions say different campaigns; the system may link them through layout.
- A wrong brand in extraction prevents linking (blocking is by brand). The image-hash scan above is the safety net.
- Brand-less ads (e.g. teasers) depend on image and text only.

## Reproduce current (development) numbers
```
python scripts/batch_analytics.py --bench      # simulated + 1 real pair, by provenance
python scripts/batch_analytics.py              # analytics on the 70 real ads
python -m pytest -q tests/test_dedup.py
```
