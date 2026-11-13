# Where this fits in Zuri (the simplest version)

## Zuri in one picture
```
 M-PESA + network data           (raw facts: who paid whom, how much, when, from which cell site)
        │
 L1  MLRun/KFP pipeline           (cleans, slices per client e.g. SportyBet paybill 590555)
        │
 L2  Apache Doris featurestore    (ready-made tables: daily/weekly/monthly, 643M rows daily)
        │
 L3  Superset   L4 OpenMetadata   L5 Conversational Intelligence (Text-to-SQL chat)
   (charts)      (definitions)     ("ask the data a question")
        │
 L6  Next.js API  →  L7  React dashboards (Recharts, deck.gl maps)
```
Today Zuri answers **"what happened?"** (descriptive: market share, counties, wallet split).

## The gap = the Intelligence Layer
The architecture doc already reserves a slot: **"Predictive Analytics (PI-1/2/3/4) — Planned Q4 2026"**.
Nothing is built there yet. That is where this repo goes.

| Question a partner asks | Type | Model family | Output |
|---|---|---|---|
| **Should I lend to this person?** (Aventus, M-KOPA) | Prediction (yes/no + probability) | Logistic regression → EBM / XGBoost | Probability of default + reason codes, in < 100 ms |
| **Who should I target, where?** (Betika, SportyBet) | Discovery (no label) | K-means / GMM on aggregates | Segments per county/ward with size, value, growth |
| **Will this customer leave?** (retention) | Prediction | same as credit | Churn probability per segment |
| **Is a KES 380M jersey deal worth it?** | Measurement (causal) | Geo-lift / difference-in-differences | Incremental users & revenue in exposed vs. unexposed areas |

## How the credit service plugs in
```
Lender app ──► /score (FastAPI, this repo) ──► reads features from Doris (L2)
                     │
                     ├─ probability of default
                     ├─ decision (approve / review / decline) from a profit-based cutoff
                     └─ top 3 reasons (for the customer and the regulator)
```
Same deployment pattern as the CI backend (FastAPI on Iguazio next to Doris, AD-1/AD-3).
Stateless scoring service = valid as an MLRun `application` (AD-3).
