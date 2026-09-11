"""Step 15 / 23 — trading simulator with fees, slippage and a confidence gate.

Runs the ensemble over the chronological test set, opens LONG / SHORT positions
when the predicted class confidence exceeds the gate, holds each position for
TARGET_HORIZON minutes, and reports realistic net-of-cost metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    CONFIDENCE_GATE,
    FEE_RATE,
    RESULTS_DIR,
    SLIPPAGE,
    TARGET_HORIZON,
)
from src.baselines import load_split  # noqa: E402
from src.ensemble import ensemble_predict, load_models  # noqa: E402

CLASS_NAMES = {0: "DOWN", 1: "NO_MOVE", 2: "UP"}
COST_PER_TRADE = 2.0 * (FEE_RATE + SLIPPAGE)   # enter + exit


def simulate(df: pd.DataFrame, pred: np.ndarray, conf: np.ndarray, gate: float) -> dict:
    """Sequential (non-overlapping) backtest.  Returns per-trade + summary."""
    ret5 = df["future_ret_5"].to_numpy()
    n = len(df)
    trades: list[dict] = []
    i = 0
    while i < n - TARGET_HORIZON:
        p = int(pred[i])
        c = float(conf[i])
        if c >= gate and p != 1:          # 1 == NO_MOVE
            gross = ret5[i] if p == 2 else -ret5[i] / (1.0 + ret5[i])
            net = gross - COST_PER_TRADE
            trades.append({"time": df["time"].iloc[i], "side": CLASS_NAMES[p],
                           "confidence": c, "net_return": net})
            i += TARGET_HORIZON           # hold the full horizon, then look again
        else:
            i += 1

    if not trades:
        return {"n_trades": 0, "total_return": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "avg_return": 0.0}

    rets = np.array([t["net_return"] for t in trades])
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity / peak - 1.0).min()

    wins = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()

    return {
        "n_trades": len(trades),
        "total_return": float(equity[-1] - 1.0),
        "win_rate": float((rets > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown": float(drawdown),
        "sharpe": float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0,
        "avg_return": float(rets.mean()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 15/23 — backtest (ensemble + confidence gate)\n")

    df, X, _, _ = load_split("test")
    models = load_models()
    pred, conf = ensemble_predict(models, X)

    print(f"test rows={len(df)}  horizon={TARGET_HORIZON}m  "
          f"cost/trade={COST_PER_TRADE:.2%}")

    rows = []
    for gate in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
        rep = simulate(df, pred, conf, gate)
        rep["gate"] = gate
        rows.append(rep)
        print(f"  gate={gate:.1f}  trades={rep['n_trades']:5d}  "
              f"ret={rep['total_return']:+.2%}  win={rep['win_rate']:.2%}  "
              f"PF={rep['profit_factor']:.2f}  DD={rep['max_drawdown']:.2%}  "
              f"sharpe={rep['sharpe']:.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "backtest.csv", index=False)
    print(f"\nsaved -> {RESULTS_DIR / 'backtest.csv'}")
    print("STEP 15/23 DONE.")


if __name__ == "__main__":
    main()
