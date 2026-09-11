"""Signal generation: liquidity, support/resistance and buy/sell indications.

Everything here is computed from the candle + indicator columns produced by
``features`` and ``indicators``, so this module stays the single source of truth
for the "what to do" layer (liquidity context, key levels, entry/exit signals).
"""
from __future__ import annotations

import pandas as pd

from config import LIQUIDITY_WINDOW, SIGNAL_FAST, SIGNAL_SLOW, SUPPORT_WINDOW


def add_liquidity(df: pd.DataFrame, window: int = LIQUIDITY_WINDOW) -> pd.DataFrame:
    """Liquidity proxy from quote turnover (USDT traded per candle).

    ``liquidity`` is the raw quote turnover; ``liquidity_ma`` smooths it so the
    chart shows both the instantaneous flow and the running baseline.
    """
    out = df.copy()
    out["liquidity"] = out["turnover"]
    out["liquidity_ma"] = out["turnover"].rolling(window, min_periods=1).mean()
    return out


def add_support_resistance(df: pd.DataFrame, window: int = SUPPORT_WINDOW) -> pd.DataFrame:
    """Dynamic support/resistance as a Donchian-style band.

    ``support`` = rolling lowest low, ``resistance`` = rolling highest high over
    the last ``window`` candles.  Price often stalls or reverses near these bands.
    """
    out = df.copy()
    out["support"] = out["low"].rolling(window, min_periods=1).min()
    out["resistance"] = out["high"].rolling(window, min_periods=1).max()
    return out


def add_signals(
    df: pd.DataFrame,
    fast: int = SIGNAL_FAST,
    slow: int = SIGNAL_SLOW,
) -> pd.DataFrame:
    """Rule-based buy/sell signals.

    Buy (signal +1) when either:
      * EMA(fast) crosses above EMA(slow) — golden cross, or
      * RSI14 crosses back above 30 — oversold recovery.
    Sell (signal -1) when either:
      * EMA(fast) crosses below EMA(slow) — death cross, or
      * RSI14 crosses back below 70 — overbought fade.

    ``buy_signal`` / ``sell_signal`` are booleans, ``signal`` is +1/-1/0 and
    ``signal_reason`` records the human-readable trigger.
    """
    out = df.copy()
    ema_fast = out[f"EMA{fast}"]
    ema_slow = out[f"EMA{slow}"]

    golden = ((ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)).fillna(False)
    death = ((ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)).fillna(False)

    rsi = out["RSI14"] if "RSI14" in out.columns else pd.Series(float("nan"), index=out.index)
    rsi_cross_up = ((rsi.shift(1) < 30.0) & (rsi >= 30.0)).fillna(False)
    rsi_cross_down = ((rsi.shift(1) > 70.0) & (rsi <= 70.0)).fillna(False)

    out["buy_signal"] = golden | rsi_cross_up
    out["sell_signal"] = death | rsi_cross_down
    out["signal"] = 0
    out.loc[out["buy_signal"] & ~out["sell_signal"], "signal"] = 1
    out.loc[out["sell_signal"] & ~out["buy_signal"], "signal"] = -1

    reason = pd.Series("", index=out.index, dtype="object")
    reason = reason.mask(golden, f"EMA{fast} cross up EMA{slow}")
    reason = reason.mask(death, f"EMA{fast} cross down EMA{slow}")
    reason = reason.mask(rsi_cross_up, "RSI cross up 30")
    reason = reason.mask(rsi_cross_down, "RSI cross down 70")
    out["signal_reason"] = reason
    return out


def add_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Apply liquidity, support/resistance and signals in one pass."""
    out = df.copy()
    out = add_liquidity(out)
    out = add_support_resistance(out)
    out = add_signals(out)
    return out
