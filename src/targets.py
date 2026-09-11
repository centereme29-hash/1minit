"""Step 9 — movement targets (volatility-normalized UP / DOWN / NO-MOVE).

For the MAIN symbol we compute, at each minute t:

    future_return_5  = close[t+5] / close[t] - 1
    normalized_move  = future_return_5 / atr_pct

and classify:

    normalized_move >  +threshold  ->  UP   (+1)
    normalized_move <  -threshold  ->  DOWN (-1)
    otherwise                       ->  NO MOVE (0)

We also emit the individual future returns (regression heads), the max/min
excursion over the horizon, and the future volatility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    FEATURES_DIR,
    MAIN_SYMBOL,
    SYNC_DIR,
    TARGET_HORIZON,
    TARGET_RETURNS,
    TARGET_THRESHOLD_ATR,
)

C = f"{MAIN_SYMBOL}_close"
H = f"{MAIN_SYMBOL}_high"
L = f"{MAIN_SYMBOL}_low"


def _atr(df: pd.DataFrame, h: str, l: str, c: str, period: int = 14) -> pd.Series:
    prev_close = df[c].shift()
    tr = pd.concat([df[h] - df[l], (df[h] - prev_close).abs(), (df[l] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def build_targets(wide: pd.DataFrame) -> pd.DataFrame:
    close = wide[C]
    h = TARGET_HORIZON
    out = pd.DataFrame({"time": wide["time"]})

    # Future returns (regression heads)
    for k in TARGET_RETURNS:
        out[f"future_ret_{k}"] = close.shift(-k) / close - 1.0

    # Volatility-normalized movement
    atr_pct = _atr(wide, H, L, C, 14) / close
    out["future_ret_5"] = close.shift(-h) / close - 1.0
    out["normalized_move"] = out["future_ret_5"] / atr_pct

    thr = TARGET_THRESHOLD_ATR
    out["label"] = 0
    out.loc[out["normalized_move"] > thr, "label"] = 1
    out.loc[out["normalized_move"] < -thr, "label"] = -1

    # Max / min excursion over the horizon (t+1 .. t+h)
    highs = pd.concat([wide[H].shift(-k) for k in range(1, h + 1)], axis=1)
    lows = pd.concat([wide[L].shift(-k) for k in range(1, h + 1)], axis=1)
    out["future_max_up"] = highs.max(axis=1) / close - 1.0
    out["future_max_down"] = lows.min(axis=1) / close - 1.0

    # Future volatility (std of the h future 1m returns)
    rets = pd.concat([close.shift(-k) / close - 1.0 for k in range(1, h + 1)], axis=1)
    out["future_vol"] = rets.std(axis=1)

    return out


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 9 — movement targets")

    wide = pd.read_csv(SYNC_DIR / "all_1m.csv")
    wide["time"] = pd.to_datetime(wide["time"], utc=True)

    targets = build_targets(wide)
    out_path = FEATURES_DIR / "targets.csv"
    targets.to_csv(out_path, index=False)

    print(f"rows = {len(targets)}")
    print("label distribution:")
    print(targets["label"].value_counts().sort_index().to_string())
    print(f"saved -> {out_path}")
    print("\nSTEP 9 DONE.")


if __name__ == "__main__":
    main()
