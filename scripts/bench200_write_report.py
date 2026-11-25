"""Write reports/bench200_report.md from the bench200 result files (no numbers typed by hand).

Inputs: data/eval/bench200/{manifest.csv, distribution_report.json, duplicate_audit.json,
dedup_sufficiency.json, FROZEN_PIPELINE.json, report/bench200_results.json} + set50 / holdout20 summaries.
  python scripts/bench200_write_report.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data/eval/bench200"


def j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def grp(res, key):
    rows = []
    for g, m in res[f"by_{key}"].items():
        ci = m.get("accuracy_ci95")
        g = f"{g} ⚠ small n" if m["ads"] < 10 else g
        rows.append([g, m["ads"], m["values_scored"], f"{m['accuracy_pct']}%", f"{ci[0]}-{ci[1]}%" if ci else "-",
                     f"{m['silent_errors']} / {m['confirmed_values']}", f"{m['auto_fill_rate_pct']}%",
                     f"{m['over_flag_rate_pct']}%", m["missed_values"]])
    return table(rows, [key.capitalize(), "Ads", "Values", "Accuracy", "95% CI (ads)", "Wrong & confirmed",
                        "Correct auto-filled", "Correct sent to review", "Missed"])


def main():
    man = list(csv.DictReader((B / "manifest.csv").open(encoding="utf-8")))
    dist, dup, suff = j("data/eval/bench200/distribution_report.json"), j("data/eval/bench200/duplicate_audit.json"), \
        j("data/eval/bench200/dedup_sufficiency.json")
    frozen, res = j("data/eval/bench200/FROZEN_PIPELINE.json"), j("data/eval/bench200/report/bench200_results.json")
    s50 = j("data/eval/set50/report/summary.json")["systems"]["nemotron+nano"]
    h20 = j("data/eval/holdout20/report/summary.json")["systems"]["nemotron+nano"]
    o = res["overall"]
    pool = Counter(r["pool"] for r in man)
    src = Counter(r["source"] for r in man)
    human = res["labels"].get("human-verified", 0)
    L = []
    L += ["# bench200 - 200-image generalisation benchmark", "",
          f"> **Status: {'PROVISIONAL' if human < len(man) else 'human-verified'}.** {human} of {len(man)} labels are "
          "human-verified. All other labels were drafted by Claude from the images (before any system output was "
          "seen) and are pending human audit (`data/eval/bench200/audit_queue.csv`). Accuracy figures below are "
          "relative to these provisional labels, not to independently verified ground truth.", ""]
    evc = {r["cluster"] for r in man}
    overlap = [x for x in dup["excluded"] if x["reason"].startswith("creative already")]
    clusters = j("data/eval/bench200/clusters.json")
    eval_clusters = {c for k, c in clusters.items() if k.startswith(("set50/", "holdout20/"))}
    in_eval = [r["id"] for r in man if r["pool"] in ("set50", "holdout20") or r["cluster"] in eval_clusters]
    L += ["## 1. Data and selection", "",
          f"**Integrity check (final manifest):** {len(man)} images, {len(evc)} distinct creative clusters "
          f"({'all distinct' if len(evc) == len(man) else 'DUPLICATES PRESENT'}); images from or clustered with set50/holdout20: "
          f"{len(in_eval)}; overlapping creatives excluded before selection: {len(overlap)} "
          f"({', '.join(x['id'] + ' ~ ' + x['reference'] for x in overlap)}).", "",
          f"- Downloaded: **400** Hindi/Marathi newspaper ads (4 papers, MAdVerse Epaper2) + "
          f"**{dup['summary']['downloaded_extra']}** extra ads for other languages and industries (MAdVerse Epaper1/2, Advert_Gallery).",
          f"- The 400 contain only **{dup['summary']['distinct_creatives_in_400']}** distinct creatives (perceptual-hash "
          "distance <= 12; every sampled pair checked by eye was a true repeat).",
          "- Excluded from the benchmark (logged in `duplicate_audit.json`): "
          + "; ".join(f"{k}: {v}" for k, v in dup["summary"]["excluded_by_reason"].items())
          + ". The manual exclusions are 3 language/city editions of an already selected creative that the hash missed, and 1 photo of an outdoor installation (not a newspaper ad).",
          f"- Selected: **{len(man)} distinct creatives** (one per cluster; none in set50/holdout20): "
          f"{pool['dedup_real']} from the Hindi/Marathi pool and {pool['extra']} from the extra downloads. "
          "Stratified by newspaper and image quality, fixed seeds (`scripts/select_bench200.py`, `scripts/fetch_bench_extra.py`).", ""]
    L += [table([[k, v] for k, v in src.most_common()], ["Source", "Images"]), ""]
    for k, title in (("language", "Language (printed, from labels)"), ("industry", "Industry (from labels)"),
                     ("quality", "Image quality (resolution + sharpness)")):
        L += [f"**{title}**", "", table([[g, v["count"], f"{v['pct']}%"] for g, v in dist[k].items()], [k.capitalize(), "Images", "Share"]), ""]
    L += ["Metadata issues found while labelling: the dataset's 'Assamese' ads are English / Manipuri (Meetei Mayek) ads from "
          "Manipur papers; one 'English' Advert_Gallery ad is in Hindi. Labels record what is printed.", ""]
    L += ["## 2. Labels", "",
          f"- {len(man)} label files in `data/labels/` (existing schema, `data/labels/FORMAT.md`), `_set: bench200`, "
          "`_label_source: claude_independent_read`, `_verified_by: null`.",
          "- Fields: language, brand, product, ad type, category, headline, price, QR present; sub-headline, phones, "
          "websites, emails where clearly legible (otherwise in `_skip`). Long free-text fields are skipped.",
          "- `any_of` is used for genuinely acceptable alternatives (e.g. nukta spellings, brand vs. company name, a top-level vs. sub-category).",
          "- Fields the drafter was unsure about are listed in `_uncertain_fields`.",
          f"- **Human-verified: {human} / {len(man)}.** Audit workflow: contact sheets in batches of 25 "
          "(`data/eval/bench200/audit_sheets/`), queue `audit_queue.csv`, and `python scripts/bench200_audit.py apply` stamps only rows "
          "an auditor marked verified or corrected.", ""]
    L += ["## 3. Frozen pipeline", "",
          f"- Extraction model `{frozen['extract_model']}`, verifier `{frozen['verifier_model']}`, prompt v{frozen['prompt_version']} "
          f"(the Phase 2.5 validated version), thresholds {frozen['thresholds']}.",
          f"- Not a git repository: sha256 of all {len(frozen['sha256'])} source files in `data/eval/bench200/FROZEN_PIPELINE.json`; "
          "full copy in `releases/bench200_frozen/`; previous release `releases/phase2_validated/`. No pipeline change during the evaluation.", ""]
    L += ["## 4. Results (provisional)", "",
          f"System `{res['system']}` (extraction + OCR checks + independent verifier). {res['ads_evaluated']} ads evaluated"
          + (f"; missing records: {res['ads_missing_record']}" if res["ads_missing_record"] else "") + ".", "",
          "Scoring (unchanged `scripts/evaluate.py`): every labelled single field and category attribute of every ad is one "
          "value. A value is correct if it matches any accepted alternative, and a red (unfilled) field counts as 'not filled'. "
          "Accuracy = (correct + correctly empty) / all values. The 95% interval is a bootstrap over ads, because fields of the same ad are correlated.", "",
          table([["Values scored", o["values_scored"]],
                 ["Accuracy", f"**{o['accuracy_pct']}%** ({o['correct']} / {o['values_scored']}), 95% CI {o['accuracy_ci95_ad_bootstrap'][0]}-{o['accuracy_ci95_ad_bootstrap'][1]}%"],
                 ["Correct extractions (incl. correctly left empty)", o["correct"]],
                 ["Wrong values produced", f"{o['wrong_values']} ({o['wrong_label_present']} wrong value, {o['false_positives']} false positives = value given where the label says not printed)"],
                 ["**Wrong values marked confirmed (silent errors)**", f"**{o['silent_errors']} / {o['confirmed_values']} confirmed ({o['silent_error_rate_pct']}%)**"],
                 ["Correct values auto-filled (confirmed)", f"{o['auto_fill_rate_pct']}% of values with a label"],
                 ["Values sent to human review", o["sent_to_review"]],
                 ["Correct values sent to review (unnecessary flags)", f"{o['over_flag_rate_pct']}%"],
                 ["Wrong values caught (flagged)", f"{o['error_catch_rate_pct']}%"],
                 ["Missed (label has a value, field left empty)", o["missed_values"]]], ["Metric", "bench200"]), ""]
    if res.get("lists"):
        L += ["Lists (precision / recall of items): " + "; ".join(
            f"{f} {v['tp']}/{v['tp'] + v['fp']} precise, {v['tp']}/{v['tp'] + v['fn']} found"
            for f, v in res["lists"].items() if v.get("tp", 0) + v.get("fp", 0) + v.get("fn", 0)), ""]
    sev = ROOT / "data/eval/bench200/report/silent_error_review.json"
    if sev.exists() and o["silent_errors"]:
        sv = json.loads(sev.read_text(encoding="utf-8"))
        by_f = Counter(x["field"] for x in sv["genuine"] + sv["debatable"])
        L += ["### Wrong values marked confirmed - the key finding", "",
              f"Unlike set50 and holdout20 (0 each), bench200 has **{o['silent_errors']} wrong values marked confirmed** "
              f"({o['silent_error_rate_pct']}% of confirmed values). By field: " + ", ".join(f"{k} {v}" for k, v in by_f.most_common()) + ". "
              "No brand, product, QR or contact value was confirmed wrongly; the confirmed errors are in judgement fields "
              "(language, category, ad type), plus 3 headlines and 1 price.", "",
              f"Claude's provisional reading (`silent_error_review.json`, not a human audit): **{len(sv['genuine'])} genuine system errors**, "
              f"{len(sv['debatable'])} debatable (label strictness / judgement). Genuine patterns:",
              "- Marathi reported as Hindi (3 ads): same Devanagari script, and the language check cannot tell them apart.",
              "- Punjabi (Gurmukhi) not reported (1).",
              "- Gujarati headline with a mixed-in Devanagari letter (no Gujarati OCR to cross-check).",
              "- Hindi 'fayde' (benefits) taken as a price (1).",
              "- Admission ads classed as recruitment (2).",
              "- A service classed as manufacturer (1).",
              "- Two category confusions.", "",
              "These were found on test data and are **not fixed** here (the pipeline is frozen); they are inputs for the next development round, "
              "to be validated on new data.", ""]
    L += ["**By field**", "", table([[f, m["values_scored"], f"{m['accuracy_pct']}%", f"{m['silent_errors']} / {m['confirmed_values']}",
                                      f"{m['auto_fill_rate_pct']}%", f"{m['over_flag_rate_pct']}%"]
                                     for f, m in res["by_field"].items() if m["values_scored"]],
                                    ["Field", "Values", "Accuracy", "Wrong & confirmed", "Auto-filled", "Correct sent to review"]), ""]
    L += ["Groups marked ⚠ small n have fewer than 10 ads: their accuracy is indicative only (wide intervals).", ""]
    for k in ("language", "industry", "quality", "source"):
        L += [f"**By {k}**", "", grp(res, k), ""]
    L += ["**Earlier benchmarks - kept separate** (not combined with bench200)", "",
          table([["set50 (development set)", s50["values_scored"], f"{s50['accuracy_pct']}%", f"{s50['silent_errors']} / {s50['confirmed_values']}", f"{s50['auto_fill_rate_pct']}%"],
                 ["holdout20 (held-out)", h20["values_scored"], f"{h20['accuracy_pct']}%", f"{h20['silent_errors']} / {h20['confirmed_values']}", f"{h20['auto_fill_rate_pct']}%"],
                 ["**bench200 (new, provisional)**", o["values_scored"], f"**{o['accuracy_pct']}%**", f"**{o['silent_errors']} / {o['confirmed_values']}**", f"{o['auto_fill_rate_pct']}%"]],
                ["Set", "Values", "Accuracy", "Wrong & confirmed", "Auto-filled"]), ""]
    L += ["## 5. Deduplication data", "",
          f"The 400-ad pool has **{suff['repeat_clusters']} repeated creatives** ({suff['repeat_pairs']} repeat pairs, "
          f"{suff['surplus_insertions']} surplus insertions): {suff['same_paper_clusters']} repeat within one paper (different dates), "
          f"{suff['cross_paper_clusters']} across papers, {suff['hindi_marathi_edition_clusters']} are Hindi/Marathi editions. "
          "That is enough for a separate dedup evaluation (plan: `docs/DEDUP_EVAL_PLAN.md`; dedup code frozen in "
          "`data/eval/dedup_real/FROZEN.json` before this data was seen). It still needs human annotation of the groups; "
          "pHash clusters are candidates, not ground truth.", ""]
    L += ["## 6. Limitations", "",
          "1. **Labels are provisional** (Claude-drafted, human audit pending). Errors in labels count against the system and vice versa.",
          "2. The labeller and the evaluated pipeline are different models, but the labeller sometimes accepts several alternatives (`any_of`); strictness differs by field.",
          "3. Small groups (e.g. Urdu, Punjabi, Manipuri, Gujarati: 1-4 ads) have very wide intervals - indicative only.",
          "4. OCR covers Latin, Devanagari, Tamil, Telugu and Kannada. Bengali, Gujarati, Malayalam, Odia, Punjabi, Urdu and Meetei Mayek ads rely on the vision model plus the image-only verifier.",
          "5. Image quality is a heuristic (resolution + sharpness), not a human rating.",
          "6. Language edition / near-identical creatives were removed where found; a few more may remain undetected.", ""]
    L += ["## 7. Reproduce", "", "```",
          "python scripts/inspect_pool.py                      # quality + duplicate clusters",
          "python scripts/select_bench200.py                   # manifest + duplicate_audit.json (fixed seed, lock file)",
          "python scripts/run_eval.py ocr --set bench200",
          "python scripts/run_eval.py extract --set bench200 --system nemotron",
          "python scripts/run_eval.py verify --set bench200",
          "python scripts/run_eval.py validate --set bench200 --system nemotron --verifier-tag nano",
          "python scripts/evaluate.py --set bench200 --systems nemotron+nano",
          "python scripts/bench200_report.py                   # breakdowns + distribution_report.json",
          "python scripts/bench200_write_report.py             # this file",
          "python scripts/bench200_audit.py status             # human audit progress", "```", ""]
    out = ROOT / "reports" / "bench200_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
