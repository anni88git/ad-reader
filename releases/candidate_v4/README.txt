candidate_v4 = candidate_v3 + adreader/accuracy_rules_v4.py (5 offline fixes), OFF unless ADREADER_RULES_V4=1.
Also: hook in validate.py, review-lock check in matcher.py (no-op unless a v4-switched value is present).
Run records: data/eval/<set>/runs/cand_v4+nano. Development retest on already-examined data - not independent proof.
Rollback: unset ADREADER_RULES_V4 (=v3), ADREADER_RULES_V3=0 (=frozen), or releases/bench200_frozen.
