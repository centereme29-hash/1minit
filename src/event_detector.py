"""Step 6G - early event-detection baseline (M1).

Builds causal support/resistance + 4-class event labels, joins the existing
candle features, splits chronologically (60/20/20), and trains Logistic
Regression / Random Forest / LightGBM / XGBoost.  Reports accuracy, macro-F1,
ROC-AUC and PR-AUC (one-vs-rest), binary event FPR/FNR, and a
precision-at-confidence table + signal frequency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import DATA_DIR, FEATURES_DIR, MAIN_SYMBOL, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, SYNC_DIR
from src.event_engine import CLASS_NAMES, EventConfig, add_wick_features, build_event_labels, build_levels, build_stage

RS = RANDOM_STATE

def load_dataset(cfg):
    wide = pd.read_csv(SYNC_DIR / "all_1m.csv")
    wide["time"] = pd.to_datetime(wide["time"], utc=True)
    c = f"{MAIN_SYMBOL}_close"
    o = f"{MAIN_SYMBOL}_open"
    h = f"{MAIN_SYMBOL}_high"
    l = f"{MAIN_SYMBOL}_low"
    levels = build_levels(wide, cfg, c, h, l)
    events = build_event_labels(wide, levels, cfg, c, h, l)
    stage = build_stage(levels, cfg)
    wicks = add_wick_features(wide, cfg, c, o, h, l)
    feats = pd.read_csv(FEATURES_DIR / "features.csv")
    feats["time"] = pd.to_datetime(feats["time"], utc=True)
    base_cols = [cc for cc in feats.columns if cc != "time" and "_micro_" not in cc]
    ohlc = wide[["time", o, h, l, c]].rename(columns={o: "raw_open", h: "raw_high", l: "raw_low", c: "raw_close"})
    sig = pd.concat([wide[["time"]].reset_index(drop=True), levels.reset_index(drop=True), events.reset_index(drop=True), wicks.reset_index(drop=True)], axis=1)
    sig["stage"] = stage
    m = feats[["time", *base_cols]].merge(sig, on="time", how="inner").merge(ohlc, on="time", how="inner")
    m = m.sort_values("time").reset_index(drop=True)
    feature_cols = base_cols + list(levels.columns) + list(wicks.columns) + ["stage"]
    m = m.dropna(subset=feature_cols + ["event"]).reset_index(drop=True)
    return m, feature_cols

def _train_lgb(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    n = len(ytr)
    w = {c: n / (4.0 * max(1, int((ytr == c).sum()))) for c in range(4)}
    mdl = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, n_jobs=-1, random_state=RS, objective="multiclass", num_class=4, class_weight=w, verbosity=-1)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return mdl

def _train_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    n = len(ytr)
    counts = np.bincount(ytr.astype(int), minlength=4).astype(float)
    sw = np.array([n / (4.0 * max(1.0, counts[int(y)])) for y in ytr])
    mdl = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, objective="multi:softprob", num_class=4, eval_metric="mlogloss", n_jobs=-1, random_state=RS, tree_method="hist", early_stopping_rounds=30)
    mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
    return mdl

def evaluate(y_true, pred, proba):
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    rocs = []
    prs = []
    for i in range(4):
        yb = (y_true == i).astype(int)
        if len(np.unique(yb)) < 2:
            continue
        rocs.append(roc_auc_score(yb, proba[:, i]))
        prs.append(average_precision_score(yb, proba[:, i]))
    is_ev = (y_true != 0).astype(int)
    pred_ev = (pred != 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(is_ev, pred_ev, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y_true, pred)), "precision": float(p), "recall": float(r), "f1": float(f1), "roc_auc": float(np.mean(rocs)), "pr_auc": float(np.mean(prs)), "fpr": float(fp / (fp + tn)), "fnr": float(fn / (fn + tp))}

def precision_at_conf(y_true, pred, proba, name):
    conf = proba.max(axis=1)
    rows = []
    for g in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        sel = conf >= g
        n_sig = int(sel.sum())
        act = sel & np.logical_or(pred == 1, pred == 2)
        n_act = int(act.sum())
        act_prec = float(np.mean(y_true[act] == pred[act])) if n_act else np.nan
        rows.append({"model": name, "gate": g, "n_signals": n_sig, "actionable": n_act, "action_precision": act_prec, "overall_precision": float(accuracy_score(y_true[sel], pred[sel])) if n_sig else np.nan})
    return rows

def _report(name, rep):
    print("{0} acc={1:.4f} f1={2:.4f} roc={3:.4f} fpr={4:.4f} fnr={5:.4f}".format(name, rep["accuracy"], rep["f1"], rep["roc_auc"], rep["fpr"], rep["fnr"]))

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = EventConfig()
    print("STEP 6G - early event-detection baseline")
    m, feature_cols = load_dataset(cfg)
    y = m["event"].to_numpy(dtype=np.int64)
    n = len(m)
    dist = np.bincount(y, minlength=4).tolist()
    print(f"rows={n} features={len(feature_cols)}")
    for i, nm in CLASS_NAMES.items():
        print(f"   {nm}: {dist[i]} ({100.0*dist[i]/n:.2f}%)")
    tr_end = int(n * 0.6)
    va_end = int(n * 0.8)
    Xtr = m[feature_cols].iloc[:tr_end].to_numpy(dtype=np.float32)
    Xva = m[feature_cols].iloc[tr_end:va_end].to_numpy(dtype=np.float32)
    Xte = m[feature_cols].iloc[va_end:].to_numpy(dtype=np.float32)
    ytr = y[:tr_end]
    yva = y[tr_end:va_end]
    yte = y[va_end:]
    results = []
    conf_rows = []
    mdl = _train_lgb(Xtr, ytr, Xva, yva)
    tp = mdl.predict_proba(Xte)
    rep = evaluate(yte, tp.argmax(1), tp)
    rep["model"] = "lightgbm"
    results.append(rep)
    conf_rows.extend(precision_at_conf(yte, tp.argmax(1), tp, "lightgbm"))
    _report("lightgbm", rep)
    joblib.dump(mdl, MODELS_DIR / "event_lgb.joblib")
    mdl = _train_xgb(Xtr, ytr, Xva, yva)
    tp = mdl.predict_proba(Xte)
    rep = evaluate(yte, tp.argmax(1), tp)
    rep["model"] = "xgboost"
    results.append(rep)
    conf_rows.extend(precision_at_conf(yte, tp.argmax(1), tp, "xgboost"))
    _report("xgboost", rep)
    joblib.dump(mdl, MODELS_DIR / "event_xgb.joblib")
    rng = np.random.default_rng(RS)
    sub = rng.choice(len(Xtr), 100000, replace=False)
    sub.sort()
    mdl = RandomForestClassifier(n_estimators=200, max_depth=18, class_weight="balanced", n_jobs=-1, random_state=RS)
    mdl.fit(Xtr[sub], ytr[sub])
    tp = mdl.predict_proba(Xte)
    rep = evaluate(yte, tp.argmax(1), tp)
    rep["model"] = "random_forest"
    results.append(rep)
    conf_rows.extend(precision_at_conf(yte, tp.argmax(1), tp, "random_forest"))
    _report("random_forest", rep)
    sc = StandardScaler().fit(Xtr[sub])
    mdl = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS)
    mdl.fit(sc.transform(Xtr[sub]), ytr[sub])
    tp = mdl.predict_proba(sc.transform(Xte))
    rep = evaluate(yte, tp.argmax(1), tp)
    rep["model"] = "logistic"
    results.append(rep)
    conf_rows.extend(precision_at_conf(yte, tp.argmax(1), tp, "logistic"))
    _report("logistic", rep)
    pd.DataFrame(results).to_csv(RESULTS_DIR / "event_baselines.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(RESULTS_DIR / "event_precision_at_conf.csv", index=False)
    summary = {"label_counts": {"NO_EVENT": dist[0], "SUPPORT_BREAKDOWN": dist[1], "RESISTANCE_BREAKOUT": dist[2], "FALSE_BREAK": dist[3]}, "baselines": results}
    with open(RESULTS_DIR / "event_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("DONE")

if __name__ == "__main__":
    main()
