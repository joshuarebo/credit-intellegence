# LOOP.md — How every work session runs

```
1. ORIENT   Read CLAUDE.md → STATE.md → the doc for the module you're touching.
2. PLAN     Write the next step in STATE.md ("Next"). If it changes scope, get approval.
3. BUILD    Smallest change that moves the plan forward. Config over code.
4. VERIFY   Run tests + the pipeline on the sample. Compare against the baseline.
5. EXPLAIN  Update the plain-English doc for what you changed (why, not just what).
6. RECORD   Update STATE.md (Done / Next / Decisions / Open questions). Commit.
```

## Model-experiment loop (inner loop)
```
hypothesis → change ONE thing (feature, model, param) → train on old loans
→ score new loans → compare: AUC, KS, Brier, expected profit at chosen cutoff
→ keep if it beats the champion on expected profit AND stays explainable → log to MLflow
```

## Definition of done for any model
- Beats logistic-regression baseline on expected profit (out-of-time test).
- Calibrated (predicted 10% default ≈ 10% actually default). Brier score reported.
- Reason codes available for every single decision.
- Fairness check: approval and default rates by gender and age band reported.
- Written up in `docs/MODEL_CARD.md`.
