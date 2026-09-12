"""Step 6L - retrain event model with tuning + confidence calibration.

Improvements over the M1 baseline (src/event_detector.py):
1. Small hyperparameter grid for LightGBM / XGBoost, selected by macro PR-AUC
   on the chronologically-clean validation set (the right metric for the
   imbalanced event labels, not accuracy).
2. Temperature calibration (power scaling: q = p^(1/T)) fit on validation so
   the reported confidence matches empirical accuracy; verified with ECE
   (expected calibration error) on the untouched test set.
3. Before/after report: accuracy, macro-F1, ROC-AUC, PR-AUC, ECE and
   precision-at-confidence, plus a comparison against the previous M1 run.

The calibrated wrappers replace models/event_lgb.joblib and event_xgb.joblib,
so the live dashboard (src/event_live.py) automatically gets honest
confidence.  The raw (uncalibrated) models are kept as *_raw.joblib.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import MODELS_DIR, RESULTS_DIR
from src.event_engine import EventConfig, CLASS_NAMES
from src.event_detector import load_dataset

RS = 42

GRID_LGB = [
    dict(num_leaves=31, learning_rate=0.05, min_child_samples=20),
    dict(num_leaves=63, learning_rate=0.05, min_child_samples=20),
    dict(num_leaves=63, learning_rate=0.03, min_child_samples=50),
    dict(num_leaves=127, learning_rate=0.04, min_child_samples=20),
]
GRID_XGB = [
    dict(max_depth=6, learning_rate=0.05, min_child_weight=5),
    dict(max_depth=8, learning_rate=0.05, min_child_weight=5),
    dict(max_depth=8, learning_rate=0.03, min_child_weight=10),
    dict(max_depth=10, learning_rate=0.04, min_child_weight=5),
]


class CalibratedModel:
    """Fitted classifier + temperature T (power calibration: q = p^(1/T)).

    ``predict_proba`` returns temperature-scaled probabilities, so any consumer
    (live dashboard, backtester) transparently gets calibrated confidence.
    """

    def __init__(self, model, temperature):
        self.model = model
        self.temperature = float(temperature)

    def predict_proba(self, X):
        p = np.asarray(self.model.predict_proba(X), dtype=np.float64) + 1e-12
        q = np.power(p, 1.0 / self.temperature)
        q /= q.sum(axis=1, keepdims=True)
        return q

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def ece_score(ytrue, proba, n_bins=10):
    """Expected calibration error over max-probability bins (multiclass)."""
    p = np.asarray(proba, np.float64)
    conf = p.max(1)
    pred = p.argmax(1)
    y = np.asarray(ytrue)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0] = -1e-9
    edges[-1] = 1.0 + 1e-9
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += (m.sum() / len(y)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(pval, yval):
    """Fit temperature T on validation by minimizing multiclass log loss."""
    p = np.asarray(pval, np.float64) + 1e-12
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(yval, np.int64)

    def nll(logT):
        q = np.power(p, 1.0 / np.exp(logT))
        q /= q.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1.0))))

    res = minimize_scalar(nll, bounds=(np.log(0.1), np.log(8.0)), method="bounded")
    return float(np.exp(res.x))


def evaluate(y_true, pred, proba):
    """Same semantics as M1: macro metrics + binary event FPR/FNR."""
    acc = float(accuracy_score(y_true, pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    roc = float(roc_auc_score(y_true, proba, multi_class="ovr"))
    pr = float(average_precision_score(y_true, proba, average="macro"))
    ev = np.isin(y_true, [1, 2])
    pe = np.isin(pred, [1, 2])
    fpr = float(np.mean(~ev & pe))
    fnr = float(np.mean(ev & ~pe))
    return {"accuracy": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "roc_auc": roc, "pr_auc": pr, "fpr": fpr, "fnr": fnr}


def precision_at_conf(y_true, proba, name):
    """Number of signals, actionable signals, and their precision per gate."""
    rows = []
    p = np.asarray(proba, np.float64)
    y = np.asarray(y_true)
    for g in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        sel = p.max(1) >= g
        n_sig = int(sel.sum())
        pred = p.argmax(1)
        act = sel & np.isin(pred, [1, 2])
        n_act = int(act.sum())
        act_prec = float(np.mean(y[act] == pred[act])) if n_act else np.nan
        rows.append({"model": name, "gate": g, "n_signals": n_sig, "actionable": n_act,
                     "action_precision": act_prec})
    return rows


def _weight(ytr):
    n = len(ytr)
    return {c: n / (4.0 * max(1, int((ytr == c).sum()))) for c in range(4)}


def tune_lgb(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    w = _weight(ytr)
    best, best_p = None, None
    for p in GRID_LGB:
        mdl = lgb.LGBMClassifier(n_estimators=200, num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
                                 min_child_samples=p["min_child_samples"], subsample=0.8, colsample_bytree=0.8,
                                 n_jobs=-1, random_state=RS, objective="multiclass", num_class=4,
                                 class_weight=w, verbosity=-1)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(20, verbose=False)])
        sc = float(average_precision_score(yva, mdl.predict_proba(Xva), average="macro"))
        print("  lgb %s pr_auc=%.4f" % (p, sc), flush=True)
        if best is None or sc > best:
            best, best_p = sc, p
    print("  lgb best %s (pr_auc=%.4f)" % (best_p, best), flush=True)
    return best_p


def tune_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    n = len(ytr)
    counts = np.bincount(ytr.astype(int), minlength=4).astype(float)
    sw = np.array([n / (4.0 * max(1.0, counts[int(y)])) for y in ytr])
    best, best_p = None, None
    for p in GRID_XGB:
        mdl = xgb.XGBClassifier(n_estimators=200, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                                min_child_weight=p["min_child_weight"], subsample=0.8, colsample_bytree=0.8,
                                objective="multi:softprob", num_class=4, eval_metric="mlogloss",
                                n_jobs=-1, random_state=RS, tree_method="hist", early_stopping_rounds=20)
        mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        sc = float(average_precision_score(yva, mdl.predict_proba(Xva), average="macro"))
        print("  xgb %s pr_auc=%.4f" % (p, sc), flush=True)
        if best is None or sc > best:
            best, best_p = sc, p
    print("  xgb best %s (pr_auc=%.4f)" % (best_p, best), flush=True)
    return best_p


def _final_lgb(Xtr, ytr, Xva, yva, p):
    import lightgbm as lgb
    mdl = lgb.LGBMClassifier(n_estimators=600, num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
                             min_child_samples=p["min_child_samples"], subsample=0.8, colsample_bytree=0.8,
                             n_jobs=-1, random_state=RS, objective="multiclass", num_class=4,
                             class_weight=_weight(ytr), verbosity=-1)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return mdl


def _final_xgb(Xtr, ytr, Xva, yva, p):
    import xgboost as xgb
    n = len(ytr)
    counts = np.bincount(ytr.astype(int), minlength=4).astype(float)
    sw = np.array([n / (4.0 * max(1.0, counts[int(y)])) for y in ytr])
    mdl = xgb.XGBClassifier(n_estimators=400, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                            min_child_weight=p["min_child_weight"], subsample=0.8, colsample_bytree=0.8,
                            objective="multi:softprob", num_class=4, eval_metric="mlogloss",
                            n_jobs=-1, random_state=RS, tree_method="hist", early_stopping_rounds=30)
    mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
    return mdl


def _fit_one(name, mdl, Xva, yva, Xte, yte, out_rows, conf_rows, summary):
    pva = mdl.predict_proba(Xva)
    pte = mdl.predict_proba(Xte)
    T = fit_temperature(pva, yva)
    pte_cal = CalibratedModel(mdl, T).predict_proba(Xte)
    before = evaluate(yte, pte.argmax(1), pte)
    before["ece"] = ece_score(yte, pte)
    after = evaluate(yte, pte_cal.argmax(1), pte_cal)
    after["ece"] = ece_score(yte, pte_cal)
    print("%s before: acc=%.4f f1=%.4f roc=%.4f pr=%.4f ece=%.4f | after: acc=%.4f f1=%.4f roc=%.4f pr=%.4f ece=%.4f (T=%.3f)" % (
        name, before["accuracy"], before["f1"], before["roc_auc"], before["pr_auc"], before["ece"],
        after["accuracy"], after["f1"], after["roc_auc"], after["pr_auc"], after["ece"], T), flush=True)
    out_rows.append({"model": name, "temperature": T, "version": "before", **before})
    out_rows.append({"model": name, "temperature": T, "version": "calibrated", **after})
    conf_rows.extend(precision_at_conf(yte, pte, name + "_before"))
    conf_rows.extend(precision_at_conf(yte, pte_cal, name + "_calibrated"))
    summary[name] = {"temperature": T, "before": before, "after": after}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = EventConfig()
    print("STEP 6L - retrain event model (tuning + calibration)", flush=True)
    m, feature_cols = load_dataset(cfg)
    y = m["event"].to_numpy(dtype=np.int64)
    n = len(m)
    print("rows=%d features=%d" % (n, len(feature_cols)), flush=True)
    for i, nm in CLASS_NAMES.items():
        print("  %s: %d (%.2f%%)" % (nm, int((y == i).sum()), 100.0 * (y == i).mean()), flush=True)
    tr_end, va_end = int(n * 0.6), int(n * 0.8)
    Xtr = m[feature_cols].iloc[:tr_end].to_numpy(dtype=np.float32)
    Xva = m[feature_cols].iloc[tr_end:va_end].to_numpy(dtype=np.float32)
    Xte = m[feature_cols].iloc[va_end:].to_numpy(dtype=np.float32)
    ytr, yva, yte = y[:tr_end], y[tr_end:va_end], y[va_end:]

    print("tuning lightgbm", flush=True)
    plgb = tune_lgb(Xtr, ytr, Xva, yva)
    print("tuning xgboost", flush=True)
    pxgb = tune_xgb(Xtr, ytr, Xva, yva)
    print("final fits", flush=True)
    lgb = _final_lgb(Xtr, ytr, Xva, yva, plgb)
    xgb = _final_xgb(Xtr, ytr, Xva, yva, pxgb)

    out_rows, conf_rows, summary = [], [], {}
    print("calibrating + evaluating", flush=True)
    _fit_one("lightgbm", lgb, Xva, yva, Xte, yte, out_rows, conf_rows, summary)
    _fit_one("xgboost", xgb, Xva, yva, Xte, yte, out_rows, conf_rows, summary)

    joblib.dump(CalibratedModel(lgb, summary["lightgbm"]["temperature"]), MODELS_DIR / "event_lgb.joblib")
    joblib.dump(CalibratedModel(xgb, summary["xgboost"]["temperature"]), MODELS_DIR / "event_xgb.joblib")
    joblib.dump(lgb, MODELS_DIR / "event_lgb_raw.joblib")
    joblib.dump(xgb, MODELS_DIR / "event_xgb_raw.joblib")
    print("saved models/event_{lgb,xgb}.joblib (calibrated) + *_raw.joblib", flush=True)

    pd.DataFrame(out_rows).to_csv(RESULTS_DIR / "event_retrain_metrics.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(RESULTS_DIR / "event_retrain_precision.csv", index=False)
    prev = {}
    pf = RESULTS_DIR / "event_summary.json"
    if pf.exists():
        try:
            prev = {b["model"]: b for b in json.load(open(pf)).get("baselines", [])}
        except Exception:
            prev = {}
    for name in ("lightgbm", "xgboost"):
        if name in prev:
            print("OLD %s: acc=%.4f f1=%.4f roc=%.4f pr=%.4f" % (
                name, prev[name]["accuracy"], prev[name]["f1"], prev[name]["roc_auc"], prev[name]["pr_auc"]),
                flush=True)
    with open(RESULTS_DIR / "event_retrain_summary.json", "w") as fh:
        json.dump({"previous_m1": prev, **summary, "feature_cols": feature_cols}, fh, indent=2, default=str)
    print("DONE -> results/event_retrain_{metrics,precision}.csv + event_retrain_summary.json", flush=True)


if __name__ == "__main__":
    main()
