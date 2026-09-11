"""Step 6B — walk-forward validation (Phase 20).

A single chronological 60/20/20 split hides regime dependence.  This module
walks forward in time instead: train on everything before a validation window,
score that window, slide the window forward, and repeat.  The result is a
per-window accuracy / F1 / PnL report plus an aggregate summary that measures
regime stability, not one lucky split.

Usage::

    python src/walk_forward.py --model lightgbm --n-windows 8 --subsample 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    CLASS_NAMES,
    FEE_RATE,
    NUM_CLASSES,
    RESULTS_DIR,
    SLIPPAGE,
    TARGET_HORIZON,
)
from src.dataset import build_matrix  # noqa: E402
from src.baselines import LABEL_MAP  # noqa: E402

COST_PER_TRADE = 2.0 * (FEE_RATE + SLIPPAGE)


def _fit_predict(model_name: str, Xtr, ytr, Xval) -> np.ndarray:
    """Train a fast baseline and return predicted class indices on the window."""
    if model_name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="multi:softprob", num_class=NUM_CLASSES, eval_metric="mlogloss",
            n_jobs=-1, random_state=42, tree_method="hist",
        )
        m.fit(Xtr, ytr, verbose=False)
        return m.predict(Xval)
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            n_jobs=-1, random_state=42, objective="multiclass", num_class=NUM_CLASSES,
            verbosity=-1,
        )
        m.fit(Xtr, ytr)
        return m.predict(Xval)


def _window_pnl(df_val: pd.DataFrame, pred: np.ndarray) -> float:
    """Simple net-of-fee directional PnL over one validation window."""
    ret5 = df_val["future_ret_5"].to_numpy(dtype="float64")
    total = 0.0
    n = len(df_val)
    i = 0
    while i < n - TARGET_HORIZON:
        p = int(pred[i])
        cls = CLASS_NAMES.get(p, "")
        if cls in ("UP", "DOWN"):
            sign = 1.0 if cls == "UP" else -1.0
            gross = ret5[i] if sign > 0 else -ret5[i] / (1.0 + ret5[i])
            total += gross - COST_PER_TRADE
            i += TARGET_HORIZON
        else:
            i += 1
    return total


def run(model_name: str = "lightgbm", n_windows: int = 8, subsample: int = 50000) -> pd.DataFrame:
    m, feature_cols = build_matrix()
    m = m.sort_values("time").reset_index(drop=True)
    n = len(m)

    # (n_windows + 1) equal blocks: block 0 = initial train, rest = val windows.
    n_blocks = n_windows + 1
    edges = np.linspace(0, n, n_blocks + 1).astype(int)

    X_all = m[feature_cols].to_numpy(dtype=np.float32)
    y_all = m["label"].map(LABEL_MAP).to_numpy()

    rows: list[dict] = []
    for k in range(n_windows):
        train_end = edges[k + 1]
        val_start = edges[k + 1]
        val_end = edges[k + 2]

        # Expanding train window (everything before the current validation block).
        idx = np.arange(0, train_end)
        if len(idx) > subsample:
            idx = np.linspace(0, len(idx) - 1, subsample).astype(int)
        Xtr, ytr = X_all[idx], y_all[idx]

        df_val = m.iloc[val_start:val_end].reset_index(drop=True)
        Xval = X_all[val_start:val_end]
        yval = y_all[val_start:val_end]

        pred = _fit_predict(model_name, Xtr, ytr, Xval)
        acc = float(accuracy_score(yval, pred))
        p, r, f1, _ = precision_recall_fscore_support(yval, pred, average="macro", zero_division=0)
        pnl = float(_window_pnl(df_val, pred))

        rows.append({
            "window": k,
            "train_end": m["time"].iloc[train_end - 1] if train_end else None,
            "val_start": m["time"].iloc[val_start],
            "val_end": m["time"].iloc[val_end - 1],
            "n_train": int(len(Xtr)),
            "n_val": int(len(Xval)),
            "accuracy": acc,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "net_pnl": pnl,
        })
        print(f"window {k:2d}: acc={acc:.4f}  f1={f1:.4f}  pnl={pnl:+.4f}  "
              f"train={len(Xtr)}  val={len(Xval)}")

    out = pd.DataFrame(rows)
    summary = {
        "mean_acc": out["accuracy"].mean(),
        "std_acc": out["accuracy"].std(),
        "mean_f1": out["f1"].mean(),
        "std_f1": out["f1"].std(),
        "total_pnl": out["net_pnl"].sum(),
        "winning_windows": int((out["net_pnl"] > 0).sum()),
        "n_windows": n_windows,
    }
    print("\nAGGREGATE:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation (6B)")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--subsample", type=int, default=50000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = run(args.model, args.n_windows, args.subsample)
    out.to_csv(RESULTS_DIR / "walk_forward.csv", index=False)
    print(f"\nsaved -> {RESULTS_DIR / 'walk_forward.csv'}")


if __name__ == "__main__":
    main()
