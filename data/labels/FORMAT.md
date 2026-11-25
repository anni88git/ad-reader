# Ground-truth label format

One JSON file per ad image: `data/labels/<image-stem>.json`. Labels are **human-verified**;
the evaluation (chunk 7) compares the system's record against them.

## Values

| Label value | Meaning | Scoring |
|---|---|---|
| `"text"` / `123` | the correct answer | system value must match (normalised) |
| `null` | **not printed in the ad** | system must leave it empty; any value is a false positive |
| `{"any_of": ["Hybrid", null]}` | several acceptable answers (incl. "not printed") | any listed answer is correct |
| field absent / listed in `_skip` | not evaluated | ignored |

## Lists (phones, features, offers, dealers, prices, websites ...)

Either a plain list (every item required), or

```json
"features": {"required": ["Digital Key 2.0", "..."], "optional": ["24x7 Roadside Assistance"]}
```

* `required` items missing from the output count as **misses**;
* `optional` items are correct if extracted but are not misses if absent;
* output items matching neither are **false items**.
Items may themselves be `{"any_of": [...]}`.

## Normalisation used when comparing
Unicode NFC, Indic digits -> ASCII, case-insensitive, punctuation/whitespace ignored for text;
digits-only for phones; integer rupees for prices; record IDs compared exactly.
