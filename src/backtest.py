"""Step 15 / 23 — trading simulator with fees, slippage and a confidence gate.

Runs the ensemble over the chronological test set, opens LONG / SHORT positions
when the predicted class confidence exceeds the gate, holds each position for
TARGET_HORIZON minutes, and reports realistic net-of-cost metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    CAPITAL,
    CLASS_NAMES,
    CONFIDENCE_GATE,
    FEE_RATE,
    RESULTS_DIR,
    SLIPPAGE,
    TARGET_HORIZON,
)
from src.baselines import load_split  # noqa: E402
from src.ensemble import ensemble_predict, load_models  # noqa: E402

COST_PER_TRADE = 2.0 * (FEE_RATE + SLIPPAGE)   # enter + exit


def _side(p: int) -> tuple[str | None, float]:
    """Map a predicted class index to (side, sign); None for NO_MOVE."""
    cls = CLASS_NAMES.get(int(p), "")
    if cls == "UP":
        return "LONG", 1.0
    if cls == "DOWN":
        return "SHORT", -1.0
    return None, 0.0


def simulate(df: pd.DataFrame, pred: np.ndarray, conf: np.ndarray, gate: float) -> dict:
    """Sequential (non-overlapping) backtest.  Returns per-trade + summary."""
    ret5 = df["future_ret_5"].to_numpy()
    n = len(df)
    trades: list[dict] = []
    i = 0
    while i < n - TARGET_HORIZON:
        p = int(pred[i])
        c = float(conf[i])
        side, sign = _side(p)
        if c >= gate and side is not None:
            gross = ret5[i] if sign > 0 else -ret5[i] / (1.0 + ret5[i])
            net = gross - COST_PER_TRADE
            trades.append({"time": df["time"].iloc[i], "side": side,
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


def simulate_portfolio(
    df: pd.DataFrame,
    pred: np.ndarray,
    conf: np.ndarray,
    gate: float = CONFIDENCE_GATE,
    capital: float = CAPITAL,
    position_fraction: float = 1.0,
    fee: float = FEE_RATE,
    slippage: float = SLIPPAGE,
    hold: int = TARGET_HORIZON,
) -> dict:
    """Full trading simulator (Phase 23): capital, sizing, fees, slippage, hold.

    Opens LONG/SHORT when confidence passes ``gate`` and the class is not
    NO_MOVE, holds for ``hold`` minutes, then re-evaluates.  Returns per-trade
    records plus portfolio-level metrics.
    """
    ret5 = df["future_ret_5"].to_numpy(dtype="float64")
    times = df["time"].to_numpy()
    n = len(df)
    equity = float(capital)
    equity_curve = [equity]
    trades: list[dict] = []
    i = 0
    while i < n - hold:
        p = int(pred[i])
        c = float(conf[i])
        side, sign = _side(p)
        if c >= gate and side is not None:
            gross = ret5[i] if sign > 0 else -ret5[i] / (1.0 + ret5[i])
            fee_paid = 2.0 * fee * equity * position_fraction
            slip_paid = 2.0 * slippage * equity * position_fraction
            net = gross - 2.0 * (fee + slippage)
            pnl = equity * position_fraction * net
            equity += pnl
            trades.append({
                "time": times[i], "side": side, "confidence": c,
                "gross_return": float(gross), "net_return": float(net),
                "pnl": float(pnl), "fees": float(fee_paid), "slippage": float(slip_paid),
                "holding_minutes": hold, "exit_time": times[i + hold],
            })
            i += hold
        else:
            i += 1
        equity_curve.append(equity)

    curve = np.asarray(equity_curve)
    peak = np.maximum.accumulate(curve)
    drawdown = float((curve / peak - 1.0).min()) if len(curve) else 0.0

    if not trades:
        return {"n_trades": 0, "total_return": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_drawdown": drawdown, "sharpe": 0.0,
                "avg_trade": 0.0, "avg_holding": 0.0, "fees_paid": 0.0,
                "slippage_paid": 0.0, "final_equity": float(equity),
                "equity_curve": curve, "trades": []}

    rets = np.array([t["net_return"] for t in trades])
    wins = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    return {
        "n_trades": len(trades),
        "total_return": float(equity / capital - 1.0),
        "final_equity": float(equity),
        "win_rate": float((rets > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown": drawdown,
        "sharpe": float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0,
        "avg_trade": float(rets.mean()),
        "avg_holding": float(hold),
        "fees_paid": float(sum(t["fees"] for t in trades)),
        "slippage_paid": float(sum(t["slippage"] for t in trades)),
        "equity_curve": curve,
        "trades": trades,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading simulator (6H)")
    parser.add_argument("--capital", type=float, default=CAPITAL)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--gates", nargs="+", type=float,
                        default=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 15/23 — trading simulator (ensemble + confidence gate)\n")

    df, X, _, _ = load_split("test")
    models = load_models()
    pred, conf = ensemble_predict(models, X)

    print(f"test rows={len(df)}  horizon={TARGET_HORIZON}m  capital={args.capital}  "
          f"sizing={args.position_fraction:.0%}")

    rows = []
    for gate in args.gates:
        rep = simulate_portfolio(df, pred, conf, gate, capital=args.capital,
                                 position_fraction=args.position_fraction)
        rep["gate"] = gate
        rows.append(rep)
        print(f"  gate={gate:.1f}  trades={rep['n_trades']:5d}  "
              f"ret={rep['total_return']:+.2%}  win={rep['win_rate']:.2%}  "
              f"PF={rep['profit_factor']:.2f}  DD={rep['max_drawdown']:.2%}  "
              f"sharpe={rep['sharpe']:.2f}  fees={rep['fees_paid']:.2f}")

    out = pd.DataFrame([{k: v for k, v in r.items() if k not in ("equity_curve", "trades")}
                        for r in rows])
    out.to_csv(RESULTS_DIR / "backtest.csv", index=False)
    print(f"\nsaved -> {RESULTS_DIR / 'backtest.csv'}")
    print("STEP 15/23 DONE.")


if __name__ == "__main__":
    main()
