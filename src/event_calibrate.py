"""Step 6Lb - calibrate the best existing event models (M1) on validation.

The tuning pass (event_retrain.py) showed tuned hyperparameters give no
out-of-sample gain over the M1 baselines (PR-AUC 0.4134/0.4151 tuned vs
0.4175/0.4168 M1) - i.e. accuracy is at its ceiling on candle features.
The honest, real improvement is CONFIDENCE CALIBRATION: temperature scaling
fit on validation, verified with ECE on the untouched test set.

This stage:
1. Loads models/event_lgb.joblib and event_xgb.joblib (raw M1 models).
2. Rebuilds the event dataset (chronological 60/20/20 split).
3. Fits a temperature T per model on validation.
4. Reports before/after metrics + ECE + precision-at-confidence on test.
5. Saves calibrated wrappers as event_lgb.joblib / event_xgb.joblib (so the
   live dashboard transparently gets honest confidence) and the raw models
   as *_raw.joblib.  The tuning results are kept for future reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import MODELS_DIR, RESULTS_DIR
from src.event_engine import EventConfig
from src.event_detector import load_dataset
from src.event_retrain import CalibratedModel, ece_score, evaluate, fit_temperature, precision_at_conf

# Best configs found by the tuning pass (event_retrain.py), kept for reference.
TUNED_BEST = {
    "lightgbm": {"num_leaves": 127, "learning_rate": 0.04, "min_child_samples": 20, "pr_auc_val": 0.4134},
    "xgboost": {"max_depth": 10, "learning_rate": 0.04, "min_child_weight": 5, "pr_auc_val": 0.4151},
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = EventConfig()
    print("STEP 6Lb - calibrate best event models", flush=True)
    m, feature_cols = load_dataset(cfg)
    y = m["event"].to_numpy(dtype=np.int64)
    n = len(m)
    tr_end, va_end = int(n * 0.6), int(n * 0.8)
    Xva = m[feature_cols].iloc[tr_end:va_end].to_numpy(dtype=np.float32)
    Xte = m[feature_cols].iloc[va_end:].to_numpy(dtype=np.float32)
    yva, yte = y[tr_end:va_end], y[va_end:]
    print("rows=%d features=%d val=%d test=%d" % (n, len(feature_cols), len(yva), len(yte)), flush=True)

    out_rows, conf_rows, summary = [], [], {}
    for name, fn in (("lightgbm", MODELS_DIR / "event_lgb.joblib"), ("xgboost", MODELS_DIR / "event_xgb.joblib")):
        raw = joblib.load(fn)
        if isinstance(raw, CalibratedModel):
            print("WARN %s already calibrated: %s" % (name, fn), flush=True)
            raw = raw.model
        pva = np.asarray(raw.predict_proba(Xva), np.float64)
        pte = np.asarray(raw.predict_proba(Xte), np.float64)
        T = fit_temperature(pva, yva)
        pte_cal = CalibratedModel(raw, T).predict_proba(Xte)
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
        summary[name] = {"temperature": T, "tuned_best_config": TUNED_BEST.get(name), "before": before, "after": after}
        joblib.dump(CalibratedModel(raw, T), fn)
        joblib.dump(raw, MODELS_DIR / (fn.stem + "_raw.joblib"))
        print("saved %s (calibrated) + %s_raw.joblib" % (fn.name, fn.stem), flush=True)

    pd.DataFrame(out_rows).to_csv(RESULTS_DIR / "event_retrain_metrics.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(RESULTS_DIR / "event_retrain_precision.csv", index=False)
    with open(RESULTS_DIR / "event_retrain_summary.json", "w") as fh:
        json.dump({"note": "tuned hyperparams give no OOS gain over M1 baselines; calibration is the confidence fix.",
                   **summary, "feature_cols": feature_cols}, fh, indent=2, default=str)
    print("DONE -> results/event_retrain_{metrics,precision}.csv + event_retrain_summary.json", flush=True)


if __name__ == "__main__":
    main()