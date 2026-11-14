# STATE.md — current state of the project

_Last updated: 2026-11-13 by Claude (session 1)_

## Goal
Prototype credit-decision service (PD + decision + reason codes) on public data, shown to Marvin
as the first piece of the Zuri Intelligence Layer (PI-1). Zuri MVP is due 25 Nov; payments start 1 Dec;
this runs in parallel and does not block it.

## Done
- [x] Read Zuri architecture (`DASHBOARD_DATA_ARCHITECTURE.md`, `BAAS_PHASE5_..._SPORTYBET.md`).
      Intelligence-layer slot = "Predictive Analytics PI-1..4, planned Q4 2026". See `docs/ARCHITECTURE_FIT.md`.
- [x] Repo rules: `CLAUDE.md`, `LOOP.md`, `AGENTS.md`.
- [x] Dataset shortlist + recommendation (Home Credit). See `docs/DATASET_DECISION.md`.
- [x] Downloaded Home Credit `application_train` (307,511 × 123, default 8.07%) locally (git-ignored).
- [x] Algorithm guide with formulas, symbols, analogies, verdicts: `docs/ALGORITHMS_EXPLAINED.md`.
- [x] Meeting script: `docs/MEETING_BRIEF.md`.
- [x] Interactive stakeholder page `ui/index.html` (+ generated `ui/demo_data.js`): architecture fit,
      algorithm explainer with formulas, model lab (7 models ranked by profit, adjustable loan economics,
      ROC, calibration, fairness), live applicant scorer (LR / EBM / XGBoost running in-browser with
      reason codes), K-means segments, difference-in-differences calculator for sponsorship lift.
- [x] `scripts/build_demo_data.py`: 15 features (incl. repayment history from 13M past instalments),
      trains LR, tree, forest, XGBoost (monotonic), EBM, linear SVM, MLP on 80% and tests on 20%;
      exports LR/EBM/XGB so the page scores live; asserts the exported scorers match the libraries.

## Waiting on Joshua (green light)
- [ ] **Approve dataset** (Home Credit) or pick an alternative.
- [ ] **Approve build plan** below.
- [ ] Share the Zuri HTML prototype → used to style the demo page.

## Proposed build plan (not started)
1. `src/data/`: download + validate + out-of-time split (or holdout if dates are synthetic).
2. `src/features/`: ~25 features mapped to Safaricom-equivalents (`configs/features.yaml`),
   plus aggregates from `installments_payments` (late-payment counts, paid-ratio).
3. `src/models/`: logistic scorecard (WoE), EBM, XGBoost (monotonic), LightGBM, random forest.
   Shared train/evaluate interface. Logged to local MLflow.
4. `src/evaluate/`: AUC, Gini, KS, PR-AUC, Brier, calibration plot, confusion matrix at the
   **profit-optimal cutoff**, expected-profit curve, fairness by gender/age band.
5. `src/explain/`: SHAP reason codes (XGB/LGBM), native EBM reasons, top-3 per decision.
6. `service/`: FastAPI `/score`, `/explain`, `/models`, `/health`; Dockerfile.
7. `ui/`: one-page comparison dashboard + "score an applicant" form.
8. `docs/MODEL_CARD.md` + "how features were chosen" in plain English.
9. Phase 2: K-means/GMM segmentation demo on aggregated synthetic ward data.

## Decisions log
- 2026-11-13: Linear regression excluded for PD (unbounded output). Logistic = baseline.
- 2026-11-13: Model selection metric = expected profit at cutoff p* = M/(M+L); AUC/KS for ranking.
- 2026-11-13: Local MLflow until MLRun/Iguazio access is approved.
- 2026-11-14: Marvin has already built a logistic-regression model (legal defensibility: every feature
  has a visible weight). Our stance: his LR is the v1 production model; EBM/XGBoost run as challengers
  on Doris data. Weekly sync proposed: Thursdays 15:00, starting once access is granted.
- 2026-11-14: Next step = request Doris + Iguazio access via manager (Eneth Kubai). Features come from
  the existing feature-engineering team; new features requested through them.
- 2026-11-14: Product name is the **Decision API** (Marvin; echoes Jev, jevai.net). Jev, LLMs, JEPA assessed: not for the decision
  itself; possible later for features (sequence embeddings) and explanations. See `docs/STAKEHOLDER_CONTEXT.md`.
- 2026-11-14: Prototype LR shows a sign flip on LOAN_AMOUNT (negative weight) and AVG_DAYS_LATE, caused
  by correlated features. Check Marvin's model for the same before it goes to a regulator.

## Open questions
- Target definition with business: 30 or 60 days past due?
- Real margin M and loss L per loan for the pilot partner.
- First pilot partner: Aventus (LendPlus) or M-KOPA?
- ~~Is `application_date` real?~~ **Answered 2026-11-13: synthetic.** Default rate is flat (7.9–8.4%)
  and quarter sizes are near-identical (~25.5K each) across 2018–2020. We use a stratified holdout and
  say so; true out-of-time testing waits for Safaricom data.
