# Stakeholder context — how Marvin is thinking about this

_Last updated: 2026-11-14_

## Naming
- The product is the **Decision API**: a service that takes a loan request and returns
  a decision (approve / starter loan / decline), the risk, and the reasons.
- The models behind it are "decision models". Use this language in docs, UI and API paths
  (`POST /v1/decision`).

## Marvin's position
- He has **already built a logistic-regression model**.
- His reason is **legal defensibility**: if a declined customer takes the company to court, we must
  show every feature, its weight, and how they produced the decision. "We can't explain it" is not
  acceptable.
- Our stance: **agree.** His logistic regression is **v1 in production**. EBM (also glass-box) and
  XGBoost run as **challengers** on the same Doris data. A challenger replaces v1 only if it wins on
  profit *and* stays explainable to the same standard.

## Working arrangement
- Weekly sync proposed: **Thursdays 15:00**, starting once access is granted.
- Pre-condition: **Doris + Iguazio access**, requested through Joshua's manager (Eneth Kubai).
- Features already exist in Doris, built by the **feature-engineering team**; new features are
  requested through them, not built ad hoc.

## Things to check in Marvin's model (our value-add)
1. **Sign checks.** Correlated features can flip a weight's sign (our prototype: bigger loan → lower
   risk). A regulator will ask. Every weight must point the business-sensible way.
2. **Straight-line assumption.** Banding (weight-of-evidence bins) keeps a readable points table
   while capturing curves, a stepping stone to EBM.
3. **Honest probabilities.** Calibration check, because the cut-off is computed from money.
4. **Reason codes.** Top-3 features pushing each decline, in plain words, logged per decision.

## Jev (jevai.net), LLMs and JEPA: do they fit?
Marvin pointed to **Jev** (https://jevai.net/), a hosted API for structured decisions: send input
(text, experimental image) plus a schema, get back typed fields such as a route, a score and a
confidence. Marketed as "Decide.", "built for automation, not conversation", trained with
"RLCD (Reinforcement Learning for Calibrated Decisions)" for honest probabilities. Early access,
priced per input token. Note: the site says it is an unofficial community project, not affiliated
with TypeSafe AI, so verify the vendor before relying on its claims. The "Decision API" name likely
comes from this framing. Also discussed: Google's **Gemini 4 Argon** (frontier LLM, 30 Sep 2026)
and **JEPA**-style encoders (Meta).

| Option | What it is | Fit for the loan decision? |
|---|---|---|
| Jev | Hosted decision-model API: input + schema → typed decision, score, confidence | ❌ Not for lending: customer M-PESA data would leave Safaricom (Data Protection Act, consent); only a confidence, no feature weights for court; vendor changes the model between versions; tabular credit data not its stated use. ✅ **Fits:** unstructured jobs (merchant-name categorisation, e.g. the AUDIT-8 `business_category` problem; ticket routing; document extraction). ✅ **Borrow the pattern:** our Decision API should mirror its interface (context + schema in → typed decision + score + reasons out) |
| Logistic regression / EBM | Small models on a table of features | ✅ **Yes.** ~1–3 ms, deterministic, every weight or curve can be shown in court |
| JEPA | Learns compact "embeddings" from raw images, video, text or time series; fast classification on top | ❌ Not for the decision itself: our input is already a clean feature table, and an embedding isn't a weight a court can read. 🟡 **Later:** a time-series JEPA could turn raw M-PESA transaction sequences into extra features that feed the EBM |
| LLM (Gemini 4 Argon, Claude, etc.) | Generates text; huge, seconds per answer, can vary between runs | ❌ Not for the decision: slow, costly per call, not reproducible, no weights to defend. ✅ **Yes for around it:** turning reason codes into customer letters, analyst Q&A (already Zuri CI), reviewing model documentation |
| GEPA (prompt optimizer for LLM classifiers) | Automatically improves an LLM's classification prompt | ❌ Same objection: still an LLM making the decision |

**Bottom line:** the fast, defensible decision is made by a small glass-box model. Modern AI
(LLMs, JEPA-style encoders) sits *around* it: producing better features from raw data and better
explanations for people, never as the unexplainable decision-maker.
