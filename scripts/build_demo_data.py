"""Train every candidate model on Home Credit and export results for the demo page.

Plain English: we take 307K past loans, build 15 features that mirror what Safaricom
would know (income, loan size, SIM/phone stability, region, repayment habits, bureau score),
train 7 models on 80% of people, test them on the other 20% they never saw, and save the
numbers (plus small exported models that the web page can run live) to ui/demo_data.js.

Usage:  py -I scripts/build_demo_data.py <raw_data_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
REPO = Path(__file__).resolve().parents[1]

# name -> (plain label, Safaricom equivalent, expected direction of risk: +1 riskier as it rises)
FEATURES = {
    "AGE": ("Age (years)", "KYC age band", -1),
    "INCOME": ("Declared income", "Income proxy / LSM tier from M-PESA inflows", 0),
    "LOAN_AMOUNT": ("Loan amount", "Requested loan", 0),
    "ANNUITY": ("Instalment amount", "Instalment size", 0),
    "PAYMENT_RATE": ("Instalment ÷ loan", "Repayment speed of the product", 0),
    "ANNUITY_INCOME": ("Instalment ÷ income", "Affordability: instalment vs. monthly inflow", 1),
    "YEARS_EMPLOYED": ("Years employed", "Regularity / tenure of inflows", -1),
    "REGION_RATING": ("Region rating (1 best, 3 worst)", "County / ward economic tier (aggregate cell-site)", 1),
    "PHONE_YEARS": ("Years since phone change", "SIM / device tenure", -1),
    "BUREAU_SCORE_2": ("External score A", "CRB score (Metropol / TransUnion)", -1),
    "BUREAU_SCORE_3": ("External score B", "Second bureau / telco score", -1),
    "LATE_SHARE": ("Share of past instalments paid late", "Late M-PESA / Fuliza repayments", 1),
    "AVG_DAYS_LATE": ("Average days late", "How late, on average", 1),
    "UNDERPAY_SHARE": ("Share of instalments under-paid", "Partial repayments", 1),
    "N_PAST_INSTALMENTS": ("Number of past instalments", "Depth of repayment history", 0),
}
FEAT = list(FEATURES)


def build_frame(raw: Path) -> pd.DataFrame:
    app = pd.read_parquet(raw / "application_train.parquet")
    emp = app["DAYS_EMPLOYED"].where(app["DAYS_EMPLOYED"] != 365243)  # 365243 = "not employed" code
    df = pd.DataFrame({
        "SK_ID_CURR": app["SK_ID_CURR"],
        "TARGET": app["TARGET"],
        "GENDER": app["CODE_GENDER"],  # used ONLY for the fairness check, never as a feature
        "AGE": -app["DAYS_BIRTH"] / 365.25,
        "INCOME": app["AMT_INCOME_TOTAL"],
        "LOAN_AMOUNT": app["AMT_CREDIT"],
        "ANNUITY": app["AMT_ANNUITY"],
        "PAYMENT_RATE": app["AMT_ANNUITY"] / app["AMT_CREDIT"],
        "ANNUITY_INCOME": app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"],
        "YEARS_EMPLOYED": -emp / 365.25,
        "REGION_RATING": app["REGION_RATING_CLIENT"].astype(float),
        "PHONE_YEARS": -app["DAYS_LAST_PHONE_CHANGE"] / 365.25,
        "BUREAU_SCORE_2": app["EXT_SOURCE_2"],
        "BUREAU_SCORE_3": app["EXT_SOURCE_3"],
    })
    cols = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    inst = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(raw.glob("inst_*.parquet"))])
    late_days = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    agg = pd.DataFrame({
        "SK_ID_CURR": inst["SK_ID_CURR"],
        "late": (late_days > 0).astype(float),
        "days_late": late_days,
        "under": (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"] * 0.99).astype(float),
    }).groupby("SK_ID_CURR").agg(LATE_SHARE=("late", "mean"), AVG_DAYS_LATE=("days_late", "mean"),
                                 UNDERPAY_SHARE=("under", "mean"), N_PAST_INSTALMENTS=("late", "size"))
    df = df.merge(agg, left_on="SK_ID_CURR", right_index=True, how="left")
    return df


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def latency_ms(predict, row, reps=300):
    predict(row)
    t = time.perf_counter()
    for _ in range(reps):
        predict(row)
    return (time.perf_counter() - t) / reps * 1000


def main(raw: Path):
    df = build_frame(raw)
    print("frame", df.shape, "default rate", round(df.TARGET.mean(), 4))
    tr, te = train_test_split(df, test_size=0.2, stratify=df.TARGET, random_state=SEED)
    Xtr, ytr, Xte, yte = tr[FEAT], tr.TARGET.values, te[FEAT], te.TARGET.values

    imputer = SimpleImputer(strategy="median").fit(Xtr)
    scaler = StandardScaler().fit(imputer.transform(Xtr))
    Ztr, Zte = scaler.transform(imputer.transform(Xtr)), scaler.transform(imputer.transform(Xte))

    mono = tuple(FEATURES[f][2] for f in FEAT)
    rng = np.random.RandomState(SEED)
    svm_idx = rng.choice(len(Ztr), 60000, replace=False)

    models = {
        "logreg": ("Logistic regression", lambda: LogisticRegression(C=1.0, max_iter=2000), "Z"),
        "tree": ("Decision tree", lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, random_state=SEED), "Z"),
        "forest": ("Random forest", lambda: RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50,
                                                                   n_jobs=-1, random_state=SEED), "Z"),
        "xgb": ("XGBoost (monotonic)", lambda: XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                             min_child_weight=20, monotone_constraints=mono,
                                                             tree_method="hist", random_state=SEED, n_jobs=-1), "X"),
        "ebm": ("EBM (glass-box boosting)", lambda: ExplainableBoostingClassifier(interactions=0, random_state=SEED,
                                                                                  n_jobs=-1), "X"),
        "svm": ("Support vector machine", lambda: CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=5000), cv=3), "Zs"),
        "mlp": ("Neural network (MLP)", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                                              max_iter=60, random_state=SEED), "Z"),
    }

    cutoffs = np.round(np.arange(0.02, 0.405, 0.005), 3)
    age_band = pd.cut(te.AGE, [0, 30, 45, 60, 200], labels=["<30", "30–45", "45–60", "60+"]).astype(str).values
    gender = te.GENDER.values
    out_models, fitted, preds = {}, {}, {}
    for key, (label, make, space) in models.items():
        row = Xte.iloc[[0]] if space == "X" else Zte[[0]]
        # Local cache written by this script only (git-ignored). joblib uses pickle, so never point
        # this at files from elsewhere. Delete artifacts/ to force retraining.
        cache = REPO / "artifacts" / f"{key}.joblib"
        if cache.exists():
            m, secs = joblib.load(cache)
        else:
            m = make()
            t0 = time.time()
            if space == "X":
                m.fit(Xtr, ytr)
            elif space == "Zs":
                m.fit(Ztr[svm_idx], ytr[svm_idx])
            else:
                m.fit(Ztr, ytr)
            secs = time.time() - t0
            cache.parent.mkdir(exist_ok=True)
            joblib.dump((m, secs), cache)
        p = m.predict_proba(Xte if space == "X" else Zte)[:, 1]
        fitted[key], preds[key] = m, p
        grid = []
        for c in cutoffs:
            dec = p >= c  # True = decline
            tp = int(((yte == 1) & dec).sum()); fp = int(((yte == 0) & dec).sum())
            fn = int(((yte == 1) & ~dec).sum()); tn = int(((yte == 0) & ~dec).sum())
            groups = {}
            for gname, gvals in (("gender", gender), ("age", age_band)):
                for g in sorted(set(gvals)):
                    if g in ("XNA", "nan"):
                        continue
                    mask = gvals == g
                    appr = mask & ~dec
                    groups[f"{gname}:{g}"] = [int(mask.sum()), int(appr.sum()), int((appr & (yte == 1)).sum())]
            grid.append({"c": float(c), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "g": groups})
        bins = pd.qcut(p, 10, duplicates="drop")
        calib = pd.DataFrame({"p": p, "y": yte}).groupby(bins, observed=True).mean()
        fpr, tpr, _ = roc_curve(yte, p)
        sel = np.linspace(0, len(fpr) - 1, 60).astype(int)
        out_models[key] = {
            "label": label,
            "auc": float(roc_auc_score(yte, p)), "ks": ks_stat(yte, p),
            "prauc": float(average_precision_score(yte, p)), "brier": float(brier_score_loss(yte, p)),
            "logloss": float(log_loss(yte, p)), "train_s": round(secs, 1),
            "latency_ms": round(latency_ms(lambda r: m.predict_proba(r), row), 3),
            "grid": grid,
            "calib": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(calib.p, calib.y)],
            "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in sel],
        }
        print(f"{key:7s} AUC {out_models[key]['auc']:.4f}  KS {out_models[key]['ks']:.3f}  "
              f"Brier {out_models[key]['brier']:.4f}  {secs:.0f}s")

    # ---------- exports the page can run live ----------
    lr = fitted["logreg"]
    export_lr = {"median": imputer.statistics_.tolist(), "mean": scaler.mean_.tolist(),
                 "scale": scaler.scale_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}

    ebm = fitted["ebm"]
    export_ebm = {"intercept": float(np.ravel(ebm.intercept_)[0]), "terms": []}
    for i, f in enumerate(FEAT):
        cuts = ebm.bins_[i][0]
        export_ebm["terms"].append({"f": f, "cuts": [float(c) for c in cuts],
                                    "scores": [round(float(s), 5) for s in ebm.term_scores_[i]]})
    export_ebm["importance"] = dict(zip(FEAT, [float(v) for v in ebm.term_importances()]))

    booster = fitted["xgb"].get_booster()
    model_json = json.loads(booster.save_raw("json"))
    learner = model_json["learner"]
    names = learner["feature_names"]
    base = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
    export_xgb = {"base_margin": float(np.log(base / (1 - base))),
                  "trees": [slim(t, names) for t in learner["gradient_booster"]["model"]["trees"]],
                  "gain": {k: float(v) for k, v in booster.get_score(importance_type="gain").items()}}

    # sanity: re-implemented scorers must match the libraries
    sample = Xte.iloc[:500]
    js_lr = sigmoid(((sample.fillna(dict(zip(FEAT, export_lr["median"]))).values - export_lr["mean"])
                     / export_lr["scale"]) @ export_lr["coef"] + export_lr["intercept"])
    assert np.allclose(js_lr, lr.predict_proba(scaler.transform(imputer.transform(sample)))[:, 1], atol=1e-6)
    js_ebm = np.array([ebm_score(export_ebm, r) for r in sample.to_dict("records")])
    assert np.allclose(sigmoid(js_ebm), ebm.predict_proba(sample)[:, 1], atol=1e-5), "EBM export mismatch"
    js_xgb = np.array([xgb_margin(export_xgb, r) for r in sample.to_dict("records")])
    assert np.allclose(js_xgb, fitted["xgb"].predict(sample, output_margin=True), atol=1e-4), "XGB export mismatch"
    print("exports verified against library predictions")

    # ---------- preset applicants ----------
    pe, px = preds["ebm"], preds["xgb"]
    te2 = te.assign(p_ebm=pe, p_xgb=px)
    picks = {
        "Steady repayer": te2[(te2.TARGET == 0) & (te2.p_ebm < 0.03)].iloc[0],
        "Borderline": te2[(te2.p_ebm.between(0.10, 0.12))].iloc[0],
        "Late-payment history": te2[(te2.TARGET == 1) & (te2.LATE_SHARE > 0.3)].iloc[0],
        "Thin file (no bureau score)": te2[te2.BUREAU_SCORE_3.isna() & te2.N_PAST_INSTALMENTS.isna()].iloc[0],
        "High risk": te2[(te2.TARGET == 1) & (te2.p_ebm > 0.35) & (te2.LATE_SHARE.fillna(0) < 0.3)].iloc[0],
    }
    presets = [{"name": n, "actual": int(r.TARGET),
                "x": {f: (None if pd.isna(r[f]) else round(float(r[f]), 4)) for f in FEAT}} for n, r in picks.items()]

    # ---------- segmentation (illustrative, aggregate) ----------
    seg_cols = ["AGE", "INCOME", "PAYMENT_RATE", "REGION_RATING", "LATE_SHARE"]
    S = te[seg_cols].copy(); S["INCOME"] = np.log(S["INCOME"])
    S = S.fillna(S.median())
    Sz = StandardScaler().fit_transform(S)
    elbow, sil = [], []
    sub = rng.choice(len(Sz), 8000, replace=False)
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(Sz)
        elbow.append([k, float(km.inertia_)])
        sil.append([k, float(silhouette_score(Sz[sub], km.labels_[sub]))])
    best_k = max(sil, key=lambda kv: kv[1])[0]  # pick K by silhouette
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Sz)
    lab = km.labels_
    prof = []
    for c in range(best_k):
        m = lab == c
        prof.append({"id": c, "size": int(m.sum()), "age": float(te.AGE.values[m].mean()),
                     "income": float(np.median(te.INCOME.values[m])),
                     "payment_rate": float(np.nanmean(te.PAYMENT_RATE.values[m])),
                     "region": float(te.REGION_RATING.values[m].mean()),
                     "late": float(np.nanmean(te.LATE_SHARE.values[m])),
                     "default": float(yte[m].mean())})
    pca = PCA(n_components=2, random_state=SEED).fit(Sz)
    pts_idx = rng.choice(len(Sz), 1800, replace=False)
    P = pca.transform(Sz[pts_idx])
    seg = {"k": int(best_k), "cols": seg_cols, "elbow": elbow, "silhouette": sil, "profiles": prof,
           "pca_var": pca.explained_variance_ratio_.tolist(),
           "points": [[round(float(a), 3), round(float(b), 3), int(lab[i])] for (a, b), i in zip(P, pts_idx)]}

    # ---------- teaching exports: show HOW each family works, on our real data ----------
    # 1) A tiny 2-question tree on raw (median-filled) features, so people can read it.
    Xi = pd.DataFrame(imputer.transform(Xtr), columns=FEAT)
    small = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5000, random_state=SEED).fit(Xi, ytr)
    st = small.tree_

    def tree_json(i=0):
        v = st.value[i][0]
        node = {"n": int(st.n_node_samples[i]), "rate": float(v[1] / v.sum())}
        if st.children_left[i] != -1:
            node.update({"f": FEAT[st.feature[i]], "t": float(st.threshold[i]),
                         "y": tree_json(st.children_left[i]), "no": tree_json(st.children_right[i])})
        return node

    # 2) Boosting: accuracy as trees are added one after another.
    xgbm = fitted["xgb"]
    boost_curve = [[k, float(roc_auc_score(yte, xgbm.predict_proba(Xte, iteration_range=(0, k))[:, 1]))]
                   for k in (1, 3, 10, 25, 50, 100, 250, 500)]
    # 3) Bagging: accuracy as more independent trees are averaged.
    trees_p = np.array([t.predict_proba(Zte)[:, 1] for t in fitted["forest"].estimators_])
    forest_curve = [[k, float(roc_auc_score(yte, trees_p[:k].mean(0)))] for k in (1, 3, 10, 30, 100, 300)]
    # 4) Drift check (Population Stability Index) between train and test EBM scores.
    p_tr = fitted["ebm"].predict_proba(Xtr)[:, 1]
    edges = np.quantile(p_tr, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(p_tr, edges)[0] / len(p_tr); b = np.histogram(preds["ebm"], edges)[0] / len(yte)
    psi = float(np.sum((b - a) * np.log((b + 1e-6) / (a + 1e-6))))
    teach = {"tree": tree_json(), "boost_curve": boost_curve, "forest_curve": forest_curve, "psi": psi}
    print("teach", boost_curve[0], boost_curve[-1], forest_curve[0], forest_curve[-1], "psi", round(psi, 4))

    data = {
        "teach": teach,
        "meta": {"rows": int(len(df)), "train": int(len(tr)), "test": int(len(te)),
                 "default_rate": float(df.TARGET.mean()), "test_defaults": int(yte.sum()),
                 "built": time.strftime("%Y-%m-%d")},
        "features": [{"f": f, "label": FEATURES[f][0], "safaricom": FEATURES[f][1], "mono": FEATURES[f][2],
                      "missing": float(df[f].isna().mean()), "median": float(df[f].median()),
                      "p05": float(df[f].quantile(0.05)), "p95": float(df[f].quantile(0.95))} for f in FEAT],
        "models": out_models, "lr": export_lr, "ebm": export_ebm, "xgb": export_xgb,
        "lr_importance": dict(zip(FEAT, [abs(c) for c in export_lr["coef"]])),
        "presets": presets, "segments": seg,
    }
    out = REPO / "ui" / "demo_data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", out, round(out.stat().st_size / 1024), "KB")


def slim(tree, names, i=0):
    """Convert one tree from XGBoost's JSON model arrays into a small nested dict for the page."""
    left, right = tree["left_children"], tree["right_children"]
    cover = float(tree["sum_hessian"][i])
    if left[i] == -1:  # leaf: split_conditions holds the leaf value
        return {"l": float(tree["split_conditions"][i]), "c": cover}
    return {"s": names[int(tree["split_indices"][i])], "t": float(tree["split_conditions"][i]),
            "m": bool(int(tree["default_left"][i])), "c": cover,
            "y": slim(tree, names, left[i]), "n": slim(tree, names, right[i])}


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def ebm_score(exp, row):
    z = exp["intercept"]
    for t in exp["terms"]:
        v = row[t["f"]]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            z += t["scores"][0]
        else:
            z += t["scores"][int(np.searchsorted(t["cuts"], v, side="right")) + 1]
    return z


def xgb_margin(exp, row):
    z = exp["base_margin"]
    for t in exp["trees"]:
        n = t
        while "l" not in n:
            v = row[n["s"]]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                n = n["y"] if n["m"] else n["n"]
            else:
                n = n["y"] if np.float32(v) < np.float32(n["t"]) else n["n"]  # XGBoost compares in float32
        z += n["l"]
    return z


if __name__ == "__main__":
    main(Path(sys.argv[1]))
