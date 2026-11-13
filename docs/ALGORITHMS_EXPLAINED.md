# Algorithms explained from the ground up (for the credit-decision problem)

Running example throughout: **Wanjiku asks for a KES 5,000 loan.** We know (with consent)
her monthly M-PESA inflow, how regular it is, how long she's had her SIM, her past repayments.
We must answer: **will she repay?**

---

## 0. Words you must own before any formula

| Word | Plain meaning | Analogy |
|---|---|---|
| **Feature** (x) | One fact about a person: income, SIM age, late payments | One answer on an application form |
| **Label / target** (y) | The thing we want to predict. Here: 1 = defaulted, 0 = repaid | The exam result we're trying to guess |
| **n** | Number of **people (rows)** in the training data. Home Credit: n = 307,511 | Number of past loan files in the cabinet |
| **p** (or d) | Number of **features (columns)** per person | Number of questions on the form |
| **i** | Which person (row 1 … n) | File number in the cabinet |
| **j** | Which feature (column 1 … p) | Question number on the form |
| **Weight / coefficient** (β or w) | How much one feature pushes the answer | The volume knob for each question |
| **Model** | A formula with knobs that turns features into a prediction | A recipe |
| **Training** | Turning the knobs until predictions match past outcomes | Tasting and adjusting the recipe |
| **Loss function** | A number measuring how wrong one prediction is | The size of the fine for one wrong guess |
| **Objective function** | What training tries to make as small as possible = total loss + penalty for complexity | Total fines for the year + a fee for every extra ingredient |
| **Regularization** | That penalty for complexity. Stops the model memorising noise | A teacher who docks marks for over-complicated answers |
| **Overfitting** | Model memorises the training data, fails on new people | A student who memorised past papers but can't answer a new question |
| **Probability of default (PD)** | A number 0–1: chance this person won't repay | Weather forecast: "30% chance of rain" |
| **Threshold / cutoff** | PD above which we decline | "If rain chance > 40%, carry an umbrella" |

**Key reframe:** we are not predicting "yes/no". We predict a **probability**, then a
**business rule** turns it into a decision. Model ≠ decision.

---

## 1. Linear regression — the wrong tool for this job (and why)

**Formula:** ŷᵢ = β₀ + β₁xᵢ₁ + β₂xᵢ₂ + … + βₚxᵢₚ

| Symbol | Meaning |
|---|---|
| ŷᵢ | prediction for person i (the hat means "estimated") |
| β₀ | intercept: prediction when every feature is 0 (the starting point) |
| βⱼ | how much ŷ moves when feature j rises by 1, others held fixed |
| xᵢⱼ | value of feature j for person i |

**How it learns — the objective (Mean Squared Error):**
MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²  
"For every person, take the miss (residual = actual − predicted), square it so negatives don't
cancel and big misses hurt more, average over all n people. Choose β's that make this smallest."

**Analogy:** fitting a straight ruler through a cloud of dots so the total squared distance is smallest.

**Why it fails for default:** y is 0 or 1, but a straight line can predict −0.3 or 1.4 —
a "−30% chance of default" is meaningless. **Use it for amounts** (e.g. "how much will this
segment spend next month"), not yes/no.

**Verdict:** ✅ revenue/spend forecasting per segment. ❌ credit decision.

---

## 2. Logistic regression — the industry baseline (the "scorecard")

**Step 1, same weighted sum:** zᵢ = β₀ + Σⱼ βⱼ xᵢⱼ  (z can be any number, −∞ to +∞)

**Step 2, squash it into 0–1 with the sigmoid:** pᵢ = σ(zᵢ) = 1 / (1 + e^(−zᵢ))

| Symbol | Meaning |
|---|---|
| zᵢ | the "risk points" total for person i (log-odds) |
| e | Euler's number ≈ 2.718 (the constant of natural growth) |
| pᵢ | probability person i defaults |

z = 0 → p = 0.5. Big positive z → p near 1. Big negative z → p near 0.

**What a β means here:** log(p/(1−p)) = z, so e^βⱼ = **odds ratio**.
Example: β for "late payments last 3 months" = 0.7 → e^0.7 ≈ 2.0 →
**each extra late payment doubles the odds of default.** That sentence is why regulators love it.

**Objective (log-loss / binary cross-entropy):**
L = −(1/n) Σᵢ [ yᵢ·log(pᵢ) + (1 − yᵢ)·log(1 − pᵢ) ] + λ·Σⱼ βⱼ²

- If she defaulted (y=1) and we said p=0.9 → small fine (−log 0.9 = 0.1). Said p=0.1 → big fine (2.3).
  It punishes **confident wrong answers** brutally.
- λ·Σβ² is **L2 regularization** (ridge): penalises big weights so no single feature dominates.
  λ (lambda) = how strict the teacher is. **L1** (lasso) uses λ·Σ|β| instead and can push useless
  features' weights to exactly 0 — automatic feature selection.

**Scorecard trick (what banks actually ship):** bin each feature (e.g. SIM age: <6m, 6–24m, >24m),
replace bins with **Weight of Evidence** WoE = ln(%good in bin / %bad in bin), fit logistic regression,
then convert to points: Score = Offset + Factor·(−z). Classic: 600 points = 50:1 odds, +20 points
doubles the odds (PDO = 20).

**Analogy:** a points-based driving test. Each answer earns or loses points; add them; pass if above the line.

| Strengths | Limitations |
|---|---|
| Fully transparent, each weight is a sentence | Assumes each feature's effect is a straight line (on log-odds) |
| Calibrated probabilities out of the box | Misses interactions ("young AND new SIM" is worse than the sum) unless you hand-build them |
| Microsecond scoring, tiny model | Needs manual binning/feature work to compete |
| Regulator-proven for 40+ years | Usually 2–5 AUC points behind boosting |

**Real world:** almost every bank's application scorecard; CBK-supervised lenders' baseline.
**Verdict:** ✅ **Mandatory baseline and fallback.** Everything else must beat it.

---

## 3. Decision tree — the flowchart

**How it works:** ask the yes/no question that best separates payers from defaulters, then repeat
inside each group. "SIM age < 6 months?" → yes → "Late payments > 2?" → … → leaf with a default rate.

**How it picks a question — Gini impurity:** G = 1 − Σₖ pₖ²
pₖ = share of class k (payer/defaulter) in a group. A group that is 100% payers: G = 1 − 1² = 0 (pure).
50/50: G = 1 − (0.25 + 0.25) = 0.5 (maximally mixed). The tree picks the split that drops G the most.

**Analogy:** the game "20 questions".

| Strengths | Limitations |
|---|---|
| Readable as a flowchart | One tree is **unstable**: change a few rows, the whole tree changes |
| Captures interactions naturally | Overfits easily (grows a leaf per person) |
| No scaling needed | Step-shaped, coarse probabilities |

**Verdict:** building block, not a final model.

---

## 4. Random forest — bagging ("wisdom of crowds")

**How:** build B trees (e.g. B = 500). Each tree sees a **random bootstrap sample** of the rows
(drawn with replacement) and only a random subset of features at each split. Average their votes:
p̂(x) = (1/B) Σ_b tree_b(x).

**Bagging** = Bootstrap AGGregatING. Many independent noisy opinions averaged → noise cancels.

**Analogy:** ask 500 loan officers who each saw different past files; take the average opinion.

| Strengths | Limitations |
|---|---|
| Hard to overfit, robust, few knobs | Large model (500 trees), slower to score |
| Good out-of-the-box accuracy | Usually beaten by boosting on tabular data |
| | Probabilities poorly calibrated (need calibration step) |

**Verdict:** a solid sanity-check model; rarely the champion in credit.

---

## 5. Gradient boosting & XGBoost — boosting ("each student fixes the last one's mistakes")

**Bagging = trees in parallel, averaged. Boosting = trees in sequence, each correcting the previous.**

**Model:** F_M(x) = F₀ + η·f₁(x) + η·f₂(x) + … + η·f_M(x)

| Symbol | Meaning |
|---|---|
| F_M(x) | final score (log-odds) after M trees; p = σ(F_M) |
| F₀ | starting guess (the overall default rate, 8%) |
| f_m | the m-th small tree (depth 3–8) |
| η (eta) | **learning rate** (0.01–0.1): how much we trust each new tree. Small = careful steps |
| M | number of trees (hundreds to thousands) |

**How tree m is built:** compute each person's **residual-like error** (the gradient
gᵢ = pᵢ − yᵢ: "how wrong and in which direction"), and fit the next tree to predict those errors.

**XGBoost's objective (the part people hand-wave):**
Obj = Σᵢ ℓ(yᵢ, ŷᵢ) + Σₘ Ω(fₘ),   Ω(f) = γ·T + ½·λ·Σⱼ wⱼ²

| Symbol | Meaning |
|---|---|
| ℓ | log-loss (same fine as logistic regression) |
| T | number of leaves in a tree |
| wⱼ | the score output by leaf j |
| γ (gamma) | fee for every extra leaf: discourages bushy trees |
| λ | L2 penalty on leaf scores: discourages extreme outputs |

XGBoost uses gradient gᵢ **and** curvature hᵢ = pᵢ(1−pᵢ) (second derivative, "how sure we already are").
For each leaf, sum them: G = Σgᵢ, H = Σhᵢ. Then:
- **Best leaf value:** w* = −G / (H + λ)
- **Gain from a split:** ½·[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ
  Split only if gain > 0, i.e. the improvement beats the leaf fee γ. **That is regularization built in.**

**Analogy:** a relay of tutors. Tutor 1 marks the exam; tutor 2 only studies the questions tutor 1 got
wrong; tutor 3 studies what's still wrong… The final grade is the sum of their small corrections.

**Siblings:** **LightGBM** (Microsoft: faster, grows the leaf with the biggest gain first, great for
hundreds of millions of rows, like Doris tables) and **CatBoost** (Yandex: handles categorical features
like county/device type natively, strong defaults).

**Monotonic constraints (critical for credit):** you can force "more late payments → never lower risk".
This kills nonsensical wiggles and makes reasons defensible to a regulator.

| Strengths | Limitations |
|---|---|
| **Best accuracy on tabular data** (Kaggle, industry) | Black box without an explainer |
| Learns interactions & non-linearity automatically | Many knobs (η, depth, λ, γ, M) to tune |
| Handles missing values natively | Can overfit if not validated out-of-time |
| Scores in ~1 ms; production-proven | Probabilities need calibration check |

**Real world:** fraud and credit at fintechs, PayPal, many Kaggle winning solutions, and
Home Credit's own competition (top solutions were LightGBM ensembles, AUC ≈ 0.80).
**Verdict:** ✅ **Accuracy champion candidate**, shipped with monotonic constraints + SHAP.

---

## 6. EBM (Explainable Boosting Machine) — the answer to "accuracy OR explainability?"

**Formula (a GAM, Generalised Additive Model):**
g(E[y]) = β₀ + Σⱼ fⱼ(xⱼ) + Σ₍ᵢ,ⱼ₎ fᵢⱼ(xᵢ, xⱼ)

| Symbol | Meaning |
|---|---|
| g | link function (logit: same as logistic regression) |
| E[y] | expected value of y = probability of default |
| fⱼ(xⱼ) | a **learned curve** for feature j alone (not a straight line) |
| fᵢⱼ | a small number of learned **pairwise interactions** (e.g. age × SIM age) |

It's logistic regression where each straight-line weight is replaced by a **curve learned with
boosting** (cycling through features one at a time, many tiny rounds). You can **plot every curve**
and read the exact contribution of every feature for every person — no approximation.

**Analogy:** the same points-based driving test, but each question's points table was designed by an
expert who looked at millions of past drivers instead of a fixed rule.

| Strengths | Limitations |
|---|---|
| **Glass box**: exact per-person reasons, not estimates | Slower to *train* than XGBoost (fast to score) |
| Accuracy usually within ~0–1 AUC point of XGBoost on tabular | Only pairwise interactions (by design) |
| Curves can be reviewed / edited by risk team | Less known by non-specialists |

**Real world:** Microsoft InterpretML (open source), used in finance and healthcare where
decisions must be justified.
**Verdict:** ✅ **Strong candidate for production champion**: it removes the trade-off Marvin is worried about.

---

## 7. Support Vector Machine (SVM) — "the hyperplane one"

**Idea:** draw the boundary w·x + b = 0 that separates payers from defaulters with the **widest margin**
(the empty corridor between the two groups). Margin width = 2 / ‖w‖, so maximise the margin by
minimising ‖w‖.

**Objective (soft margin, hinge loss):** min ½‖w‖² + C·Σᵢ max(0, 1 − yᵢ(w·xᵢ + b))   (here yᵢ ∈ {−1, +1})

| Symbol | Meaning |
|---|---|
| w | direction perpendicular to the boundary (the weights) |
| b | shift of the boundary |
| ‖w‖ | length of w; small ‖w‖ = wide margin |
| C | how much we fine points on the wrong side (big C = strict, small C = tolerant) |
| max(0, …) | hinge: zero fine if correctly outside the corridor, growing fine otherwise |
| kernel | trick to bend the boundary (RBF kernel) without computing new features |

**Analogy:** building the widest possible road between two villages; houses in the road get fined.

| Strengths | Limitations |
|---|---|
| Elegant, strong on small/medium clean data | **Doesn't output probabilities** natively (needs Platt scaling) |
| | Kernel SVM scales badly: training ~n² to n³ → impractical at 300K–millions of rows |
| | Hard to explain |

**Verdict:** ❌ not for this problem. Mention it to show you considered it.

---

## 8. Neural networks — when do they earn their complexity?

**One neuron = logistic regression.** a = σ(w·x + b). Stack layers: h = σ(W₁x + b₁), p = σ(W₂h + b₂).
Trained by backpropagation (gradient descent on log-loss).

- **On flat tabular data (one row per applicant):** research and Kaggle consistently show boosting ≥
  neural nets (TabNet, FT-Transformer), with far less tuning. ❌ not justified now.
- **Where they *will* earn it (Phase 3):** raw **sequences** of M-PESA transactions per person
  (time, amount, merchant category) → a sequence model (GRU/Transformer) learns patterns humans can't
  hand-engineer. This is the direction large card networks and fintechs talk about publicly
  (transaction-sequence "foundation models"). Output can then be **features fed into the boosted model**.

**Verdict:** park it; revisit when we have raw transaction sequences and a proven baseline.

---

## 9. Explainability toolkit (how we say *why*)

| Tool | What it does | Formula / idea | Use |
|---|---|---|---|
| **SHAP** (Shapley values) | Splits one prediction into each feature's fair contribution | φⱼ = Σ over feature subsets S without j of [ \|S\|!(p−\|S\|−1)!/p! ]·[f(S∪{j}) − f(S)]: the average extra a feature adds, across every order features could "join the team". TreeSHAP computes it exactly & fast for trees | Reason codes for XGBoost |
| **EBM curves** | Exact, built-in | the fⱼ(xⱼ) values themselves | Reason codes for EBM |
| **LIME** | Fits a tiny linear model around one person | local approximation | Quick, but unstable; not for production |
| **Counterfactuals** (e.g. DiCE) | "What would need to change to approve?" | smallest change to x that flips the decision | Customer messaging: "repay your current loan on time for 2 months" |
| **PDP / ICE plots** | How risk moves as one feature moves | average / per-person curves | Model review with risk team |

**Analogy for SHAP:** a football team wins 3–0; Shapley fairly divides the credit among players by
averaging each player's contribution over every possible line-up.

**Why it matters legally:** Kenya's Data Protection Act 2019 gives people rights around decisions
made solely by automated processing. Reason codes + a human-review band covers this.

---

## 10. Metrics: which number actually matters

**Confusion matrix** (positive = "defaulter"):

| | Predicted default (decline) | Predicted repay (approve) |
|---|---|---|
| **Actually defaults** | TP ✅ avoided a loss | **FN 💸 we lent and lost principal** |
| **Actually repays** | **FP 😞 rejected a good customer, lost margin** | TN ✅ earned interest |

| Metric | Formula | Plain meaning | Use it? |
|---|---|---|---|
| Accuracy | (TP+TN)/all | % right | ❌ **misleading**: "approve everyone" is 92% accurate at 8% default |
| Precision | TP/(TP+FP) | Of those we declined, % truly bad | for review queues |
| Recall | TP/(TP+FN) | Of all defaulters, % we caught | for loss control |
| F1 | 2·P·R/(P+R) | balance of P and R | ❌ treats FP and FN as equal cost: they're not |
| **ROC-AUC** | P(random defaulter scored riskier than random payer) | **ranking power**, threshold-free | ✅ primary model-ranking metric |
| **Gini** | 2·AUC − 1 | what banks report | ✅ (AUC 0.78 → Gini 0.56) |
| **KS** | max gap between bad and good cumulative score distributions | separation | ✅ industry standard |
| PR-AUC | area under precision-recall | ranking focused on rare class | ✅ secondary (imbalance) |
| **Brier** | (1/n)Σ(pᵢ − yᵢ)² | are probabilities **honest**? | ✅ needed for pricing |
| **Expected profit** | see below | money | ✅ **the decision metric** |

**The cutoff comes from money, not from 0.5:**
Approve if expected gain > 0:  (1 − p)·M − p·L > 0  →  **approve when p < M / (M + L)**

- M = margin earned if she repays (interest + fees), L = loss if she defaults (principal × loss-given-default).
- Example: KES 5,000 loan, 10% fee → M = 500. Default loses L = 5,000 × 0.8 recovery-adjusted = 4,000.
  Cutoff p* = 500 / 4,500 ≈ **11%**. Approve anyone under 11% PD.
- Expected loss per loan = **PD × LGD × EAD** (probability × loss-given-default × exposure). Standard Basel language.

**This answers Marvin's FP vs FN worry directly:** the two costs aren't equal, so we put their real
KES values into the cutoff. A **"grey zone"** (e.g. 9–14% PD) goes to a smaller starter loan or human review.

---

## 11. Unsupervised: segmentation for marketing (acquisition / retention / white space)

No label here; we're discovering groups. **Always on aggregates (k ≥ 15)**, never individuals.

| Algorithm | Formula / idea | Analogy | Strength | Limit |
|---|---|---|---|---|
| **K-means** | minimise Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖² (each point's squared distance to its group centre μₖ; K groups chosen by you) | putting K shops so every customer is near one | fast on millions | round, equal-size clusters; must pick K (elbow / silhouette) |
| **GMM** (Gaussian mixture) | p(x) = Σₖ πₖ·N(x \| μₖ, Σₖ): mixture of K bell-curves, πₖ = share in group k, Σₖ = its shape | overlapping clouds | **soft** membership ("70% hustler, 30% salaried"), oval clusters | slower; sensitive to start |
| **Hierarchical** | merge closest groups repeatedly → tree (dendrogram) | family tree of customers | no K up front; nice visual | O(n²) memory: sample first |
| **PCA** | find directions of maximum variance: eigenvectors of the covariance matrix | photographing a 3D object from its most informative angle | compress 50 features → 5 before clustering; de-noise | components hard to name |
| **HDBSCAN** | density-based: clusters = dense regions, rest = noise | finding towns on a night-time satellite photo | finds odd shapes, flags outliers | needs tuning |

**Use:** PCA → K-means/GMM on ward-level (or H3-cell) aggregates: avg inflow, betting share of wallet,
device mix, age band, growth. Output: "Ward X: 4,200 young smartphone users, high betting share,
low SportyBet penetration → acquisition target."

**For the KES 380M jersey question, clustering isn't enough. That's causal:** compare new-user growth
in areas with high Nairobi United viewership vs. similar areas without (difference-in-differences /
geo-lift). The Zuri data already has county/ward × daily users × operator: the ingredients exist.

---

## 12. Final verdict for our problem

| Rank | Model | Role |
|---|---|---|
| 1 | **Logistic regression scorecard** | Baseline + regulatory fallback. Ships first. |
| 2 | **EBM** | Likely production champion: near-boosting accuracy, exact reasons |
| 3 | **XGBoost / LightGBM + monotonic constraints + SHAP** | Accuracy ceiling; champion if the profit gain over EBM is material |
| — | Random forest | sanity check |
| ✗ | Linear regression | wrong for yes/no; use for revenue forecasts |
| ✗ | SVM | doesn't scale, no probabilities |
| later | Neural sequence models | when raw transaction sequences are available |

**Decision rule:** we don't argue opinions; we run all of them on the same out-of-time test and pick the one
with the **highest expected profit**, subject to: calibrated, explainable per decision, fair across groups.
