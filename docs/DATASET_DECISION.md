# Dataset decision — which data we prototype on, and why

**Status: PROPOSED — waiting for Joshua's green light.**

## What we need from a prototype dataset
1. A **real default label** (did the borrower repay or not) — synthetic labels prove nothing.
2. **Thin-file / unbanked borrowers** — like M-KOPA, Aventus/LendPlus customers.
3. **Alternative data** (phone, region, social-circle defaults) — the closest public stand-in
   for Safaricom signals (M-PESA flows, device, cell-site region).
4. **Big and imbalanced** like real lending (few defaulters, many payers).
5. **Free and downloadable today** without approval.

## Candidates

| Dataset | Rows | Default rate | Fits our problem? | Verdict |
|---|---|---|---|---|
| **Home Credit Default Risk** (Kaggle 2018) | 307,511 loans, 122 features + 7 side tables (bureau, installments, previous loans) | 8.07% | Lender to the unbanked; uses telco & transactional data; installment-level repayment history | ✅ **Pick** |
| Zindi African Credit Scoring Challenge (2024/25) | smaller | — | Closest geography (Kenya/Ghana lenders) but needs Zindi login + rule acceptance, fewer features | 🟡 Phase 2 check ("does it hold in Africa?") |
| Give Me Some Credit (Kaggle) | 150,000 | 6.7% | Only 10 features, US banked customers | ❌ too thin |
| German Credit (UCI) | 1,000 | 30% | Textbook toy | ❌ too small |
| Lending Club | 2M+ | ~15–20% | US P2P, banked, bureau-heavy | ❌ wrong population |
| Xente (Zindi, Uganda) | mobile-money tx | — | Fraud, not credit default | ❌ wrong label |

## Why Home Credit
- It is **the** industry-standard public benchmark for "lend to people with no credit history
  using alternative data", which is exactly the M-KOPA / Aventus problem.
- Public AUC benchmarks are known (~0.75–0.80), so we can prove our numbers are honest.
- Its side table `installments_payments` (paid late? paid less?) is the closest public analogue to
  "how does this person behave with M-PESA / Fuliza / Lipa Mdogo Mdogo repayments".

## How its columns map to what Safaricom would actually have

| Home Credit column | Meaning | Safaricom equivalent (later, with consent) |
|---|---|---|
| `AMT_INCOME_TOTAL` | declared income | income proxy / LSM tier (`gaming_industry_income_proxy_trans`) |
| `AMT_CREDIT`, `AMT_ANNUITY` | loan size, instalment | requested loan, instalment |
| `REGION_RATING_CLIENT` | region quality score 1–3 | county / ward from cell-site (aggregate) |
| `DAYS_LAST_PHONE_CHANGE` | phone stability | SIM age, device change, `capability`, `dual_sim` |
| `DEF_30_CNT_SOCIAL_CIRCLE` | defaults among contacts | (not used — sensitive; documented only) |
| `EXT_SOURCE_1/2/3` | external bureau scores | CRB score (Metropol/TransUnion/Creditinfo) |
| `installments_payments` | repayment history | M-PESA repayment / Fuliza history |
| `DAYS_EMPLOYED`, `NAME_INCOME_TYPE` | job stability | regularity of inflows (salary vs. casual) |

## Sample (20 rows saved in `samples/home_credit_sample_20.csv`)

| ID | TARGET | Gender | Income | Loan | Instalment | Income type | Region rating | Phone change (days ago) | Ext score 2 | Age |
|---|---|---|---|---|---|---|---|---|---|---|
| 283554 | 0 | M | 90,000 | 113,211 | 8,010 | Pensioner | 2 | 1,482 | 0.61 | 67 |
| 198067 | 0 | F | 225,000 | 967,500 | 34,880 | State servant | 2 | 0 | 0.64 | 39 |
| 205020 | 0 | F | 90,000 | 450,000 | 16,677 | Working | 3 | 754 | 0.43 | 40 |
| 144369 | 0 | M | 166,500 | 1,006,920 | 42,791 | Commercial assoc. | 2 | 1,955 | 0.69 | 36 |
| 122106 | 0 | M | 157,500 | 323,195 | 14,364 | Pensioner | 2 | 1,023 | 0.16 | 55 |

`TARGET = 1` means the client had payment difficulties (defaulted); `0` means repaid.

## Caveats (honest)
- Not Kenyan. It proves the **method**; the **numbers** must be re-earned on Safaricom data.
- The `application_date` column exists only in the Hugging Face mirror we downloaded
  (`mohameddhameem/home-credit-default-risk`). The original Kaggle data has no dates, so it was
  probably added by whoever uploaded the mirror. **Checked: it is synthetic.** The default rate is
  flat per quarter (7.9–8.4%) and quarter sizes are near-identical, so we use a stratified holdout
  and say so plainly. True out-of-time validation happens on Safaricom data.
- Source: Kaggle competition `home-credit-default-risk` (original); downloaded via the public
  Hugging Face mirror because Kaggle needs an account + rule acceptance.
