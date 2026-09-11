"""Step 6I — paper trading (simulated live PnL, logged to logs/paper.csv).

Fetches the latest candles, rebuilds the exact training features, runs the
ensemble + confidence gate, and simulates LONG / SHORT / NO-TRADE fills with
fees + slippage.  Equity and fills are appended to ``logs/paper.csv``.

Usage::

    python src/paper.py --once                 # one prediction, print the signal
    python src/paper.py --loop --minutes 30    # poll every minute for 30 min
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    CAPITAL,
    CONFIDENCE_GATE,
    FEE_RATE,
    LOGS_DIR,
    MAIN_SYMBOL,
    SLIPPAGE,
    TARGET_HORIZON,
)
from src.features import build_features  # noqa: E402
from src.ensemble import ensemble_predict, gated_action, load_models  # noqa: E402
from src.live import fetch_wide  # noqa: E402

COST = 2.0 * (FEE_RATE + SLIPPAGE)
CLOSE_COL = f"{MAIN_SYMBOL}_close"


class PaperTrader:
    def __init__(self, capital: float = CAPITAL) -> None:
        self.capital = float(capital)
        self.equity = float(capital)
        self.position: dict | None = None      # {"side", "entry", "entry_time", "qty"}
        self.trades: list[dict] = []

    def on_bar(self, time, price, action: str, confidence: float) -> None:
        # Close any position that reached its holding horizon.
        if self.position is not None:
            pos = self.position
            bars_held = time - pos["entry_time"]
            if bars_held >= TARGET_HORIZON:
                self._close(time, price, pos)

        # Open a new position on a fresh signal.
        if self.position is None and action in ("LONG", "SHORT"):
            side = 1 if action == "LONG" else -1
            self.position = {"side": side, "entry": price, "entry_time": time,
                             "qty": self.equity / price, "confidence": confidence}

    def _close(self, time, price, pos) -> None:
        sign = pos["side"]
        gross = sign * (price / pos["entry"] - 1.0)
        net = gross - COST
        pnl = self.equity * net
        self.equity += pnl
        self.trades.append({
            "entry_time": pos["entry_time"], "exit_time": time,
            "side": "LONG" if sign > 0 else "SHORT", "confidence": pos["confidence"],
            "entry": pos["entry"], "exit": price, "net_return": net, "pnl": pnl,
        })
        self.position = None

    def log(self, time) -> None:
        row = {"time": time, "equity": self.equity, "position": self.position is not None}
        path = LOGS_DIR / "paper.csv"
        df = pd.DataFrame([row])
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)


def latest_features():
    wide = fetch_wide(minutes=500)
    feats = build_features(wide)
    cols = [c for c in feats.columns if c != "time"]
    return wide, feats, feats[cols].to_numpy(dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading (6I)")
    parser.add_argument("--once", action="store_true", help="single prediction")
    parser.add_argument("--loop", action="store_true", help="poll every minute")
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        models = load_models()
    except FileNotFoundError:
        print("no trained models found — run src/baselines.py first.")
        return

    if args.once:
        wide, feats, X = latest_features()
        pred, conf = ensemble_predict(models, X)
        actions, confs = gated_action(pred, conf, CONFIDENCE_GATE)
        latest_time = wide["time"].iloc[-1]
        price = wide[CLOSE_COL].iloc[-1]
        print(f"bar={latest_time}  close={price}")
        print(f"signal={actions[0]}  confidence={confs[0]:.2f}  gate={CONFIDENCE_GATE}")
        return

    trader = PaperTrader()
    print(f"paper trading: capital={trader.capital}  gate={CONFIDENCE_GATE}  "
          f"horizon={TARGET_HORIZON}m")
    end = time.time() + args.minutes * 60
    while time.time() < end:
        wide, feats, X = latest_features()
        pred, conf = ensemble_predict(models, X)
        actions, confs = gated_action(pred, conf, CONFIDENCE_GATE)
        t = wide["time"].iloc[-1]
        price = wide[CLOSE_COL].iloc[-1]
        trader.on_bar(t, price, actions[-1], confs[-1])
        trader.log(t)
        print(f"{t}  signal={actions[-1]}  conf={confs[-1]:.2f}  equity={trader.equity:.2f}")
        time.sleep(55)  # poll once per minute

    if trader.position is not None:
        # close at the latest price on shutdown (best-effort)
        wide, _, _ = latest_features()
        trader._close(wide["time"].iloc[-1], wide[CLOSE_COL].iloc[-1], trader.position)
    pd.DataFrame(trader.trades).to_csv(LOGS_DIR / "paper_trades.csv", index=False)
    print(f"\nfinal equity={trader.equity:.2f}  trades={len(trader.trades)}")
    print(f"saved -> {LOGS_DIR / 'paper.csv'}")


if __name__ == "__main__":
    main()
