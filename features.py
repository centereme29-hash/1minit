"""Feature calculation.

Step 1 produces exactly the 12 requested fields::

    timestamp_utc, open, high, low, close, volume, turnover,
    EMA5, EMA10, EMA20, EMA50, EMA100

Later steps (returns, volatility, volume features, BTC/XAUT lags, targets,
etc.) will be added here so this module stays the single source of feature
truth.
"""
from __future__ import annotations

import pandas as pd

from config import EMA_SPANS

# Canonical Step 1 output columns (exact ordering requested).
REQUIRED_FIELDS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "EMA5",
    "EMA10",
    "EMA20",
    "EMA50",
    "EMA100",
]


def add_emas(df: pd.DataFrame, spans: tuple[int, ...] = EMA_SPANS) -> pd.DataFrame:
    """Append ``EMA5`` ... ``EMA100`` columns computed on ``close``.

    Uses the standard exponential moving average::

        ema = close.ewm(span=N, adjust=False, min_periods=N).mean()

    ``min_periods=N`` leaves the EMA as NaN until a full window of N candles
    exists, which avoids a misleading warm-up value at the very start.
    """
    out = df.copy()
    for span in spans:
        out[f"EMA{span}"] = out["close"].ewm(span=span, adjust=False, min_periods=span).mean()
    return out


def build_step1_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw candles (with ``timestamp_ms``) into the 12-field dataset."""
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True)
    out = add_emas(out)
    return out[REQUIRED_FIELDS].reset_index(drop=True)
