# CLAUDE.md — Rules for working in this repo

Read this file first, every session. Then read `STATE.md`, then `LOOP.md`.

## What this repo is
A **credit-decision intelligence microservice** for Zuri BAaaS / AIaaS (Safaricom).
It answers one question in real time: **"Should we lend to this person, how much, and why?"**
It is the first concrete piece of the *Predictive Analytics (PI-1..4)* slot in the Zuri
Intelligence Layer (see `docs/ARCHITECTURE_FIT.md`).

Target partners: digital lenders (Aventus/LendPlus, M-KOPA, Watu) and betting operators
that want to offer credit or target marketing (SportyBet, Betika, SportPesa, PepetaBet).

## Non-negotiable rules
1. **Plan before you build.** Anything non-trivial gets a short plan in `STATE.md`
   and the owner (Joshua) approves it before code is written.
2. **Update `STATE.md` at the end of every work session**: what changed, what is next,
   open decisions. If it isn't in `STATE.md`, it didn't happen.
3. **Simplest model that wins.** Every model must beat the logistic-regression baseline
   on the *business metric* (expected profit) to justify its complexity (Occam's razor).
4. **Every decision is explainable.** No model ships without per-decision reason codes
   (top features pushing the score up/down).
5. **Out-of-time validation.** Train on older loans, test on newer ones (`application_date`).
   Random splits lie in credit.
6. **Aggregate, never individual, for marketing.** Segmentation outputs are groups
   with a k ≥ 15 floor (same privacy floor as the BAaaS dashboards). Never expose a
   single subscriber's location.
7. **No raw data in git.** `data/` is git-ignored. Only code, configs, docs, small samples.
8. **Plain-English docs.** Every module gets a section in `docs/` written so a
   non-data-scientist (Marvin) can follow it.
9. **Config, not code.** Features, thresholds, model params live in `configs/*.yaml`
   (mirrors the Zuri "tenant YAML" pattern).

## Tech choices (approved defaults)
- Python 3.11+, pandas, scikit-learn, XGBoost, LightGBM, `interpret` (EBM), SHAP.
- Experiment tracking: local MLflow file store (`mlruns/`) until the Iguazio/MLRun approval lands;
  code is written so swapping to MLRun is a config change.
- Serving: FastAPI (`/score`, `/explain`, `/health`) — same pattern as the CI backend (:8040).
- Demo UI: single static HTML page calling the API (model comparison + live scoring).

## Style
- Small modules, typed functions, docstrings in plain English.
- Tests in `tests/` with pytest; a pipeline change needs a test.
- Commit messages: imperative, one line summary, body explains *why*.
