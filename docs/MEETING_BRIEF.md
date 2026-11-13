# Meeting brief: Joshua × Marvin (credit intelligence)

## The one-line position
> "It's not linear regression vs. XGBoost. Linear regression predicts amounts, not yes/no.
> The real choice is **how much accuracy we buy per unit of explainability we give up**, and
> we settle that with a **profit number on held-out loans**, not opinions."

## Script (≈10 minutes)

**1. Agree on the problem first (2 min)**
- "What exactly are we predicting? I propose: *probability this person fails to repay within 30 days
  of due date* (PD). Then a business rule turns PD into approve / smaller loan / decline."
- "Two mistakes cost different money: lending to a defaulter loses the principal (~KES 4,000 on a
  5,000 loan); rejecting a good payer loses the margin (~KES 500). So the cutoff is about 11%, not 50%."

**2. Correct the framing, kindly (1 min)**
- "Linear regression can output −20% or 130% for a probability. The linear model for yes/no is
  **logistic regression**: same 'weighted sum', squashed to 0–1. That's the bank scorecard. It
  stays as our baseline."

**3. Solve his explainability worry (3 min)**
- "Three tiers. We test all three on the same data:"
  1. **Logistic scorecard**: every weight is a sentence ("each late payment doubles the odds").
  2. **EBM (Explainable Boosting Machine)**: boosting accuracy, but glass-box: exact reasons, no approximation.
  3. **XGBoost + monotonic constraints + SHAP**: top accuracy; SHAP gives the top 3 reasons per decision.
- "Every decline comes with reason codes: 'irregular inflows in the last 60 days, SIM < 6 months'.
  That covers the customer, the regulator, and the Data Protection Act automated-decision clause."
- "We add a **grey zone**: borderline cases get a smaller starter loan or human review. That's how we
  avoid losing good customers to a model's mistake."

**4. Show it's already moving (2 min)**
- "I didn't wait. Repo is up, rules and plan documented, dataset chosen: **Home Credit**,
  307K real loans to unbanked borrowers with alternative data, 8% default. It's the public
  stand-in for Safaricom signals."
- "Within days: a small API + page where you can compare all models side by side (AUC, KS, profit at cutoff)
  and score a sample applicant with reasons. No MLflow approval needed; it's designed to drop into MLRun later."

**5. Connect to the billion-shilling story (1 min)**
- "Zuri today tells partners *what happened*. The architecture already reserves an Intelligence Layer slot
  (PI-1..4, Q4). Credit scoring is PI-1: an API partners like Aventus and M-KOPA pay per call.
  Segmentation (PI-2) tells Betika/SportyBet *where* to spend. And geo-lift (PI-3) tells them whether a
  KES 380M jersey deal actually brought customers."

**6. Ask for three things (1 min)**
1. Agreement on the target definition (PD, 30 days past due) and rough KES margin/loss numbers.
2. Which partner is the first pilot (Aventus/LendPlus or M-KOPA)?
3. Path to a *consented* Safaricom feature sample in Doris for validation after the MVP (25 Nov).

## Likely questions & answers

| He asks | You say |
|---|---|
| "Why not just XGBoost, it's most accurate?" | "It might win, and we'll know in numbers. But if EBM is within ~1 AUC point, the glass box is worth more than the last decimal." |
| "How do I explain a decline to a customer?" | "Top 3 reason codes from the model, plus a counterfactual: 'repay on time for 2 months and you qualify'." |
| "What if the model is wrong?" | "It will be, sometimes. That's why we price the error: cutoff from KES costs, grey zone to review, and monthly monitoring for drift (PSI)." |
| "Is accuracy the right metric?" | "No. 'Approve everyone' is 92% accurate. We rank models by AUC/KS and pick the cutoff by expected profit." |
| "Real time like Visa?" | "Scoring is ~1 ms per model; the latency is in fetching features. Pre-compute features in Doris daily, score on request: <100 ms end-to-end." |
| "Neural networks?" | "Not on tabular rows; boosting wins there. They earn their place later on raw transaction sequences." |
| "Data privacy?" | "Consent-based for credit. For marketing, only aggregates with the same k ≥ 15 floor as the dashboards; never an individual's location." |
| "How long?" | "Baseline + comparison page this week on public data. Safaricom-data version after MVP, once data access is approved." |
