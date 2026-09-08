"""Lightweight sanity checks for collected 1-minute candles."""
from __future__ import annotations

import pandas as pd


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate a raw candle DataFrame.

    Expects the raw columns ``timestamp_ms, open, high, low, close, volume,
    turnover``.  Returns a small report dict with ``rows``, ``ok``, ``issues``,
    ``first`` and ``last`` timestamps.
    """
    required = ["timestamp_ms", "open", "high", "low", "close", "volume", "turnover"]

    if df.empty:
        return {"rows": 0, "ok": False, "issues": ["empty dataframe"]}

    issues = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"rows": len(df), "ok": False, "issues": [f"missing columns {missing}"]}

    # --- OHLC sanity -----------------------------------------------------
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        issues.append("high is below open/close/low")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        issues.append("low is above open/close/high")
    if (df[["open", "high", "low", "close", "volume", "turnover"]] < 0).any().any():
        issues.append("negative values detected")

    # --- Timestamp sanity -------------------------------------------------
    ts = df["timestamp_ms"].astype("int64")
    if not ts.is_monotonic_increasing:
        issues.append("timestamps not ascending")
    gaps = ts.diff().dropna()
    if (gaps <= 0).any():
        issues.append("duplicate / out-of-order timestamps")
    off = int(((gaps % 60_000) != 0).sum())
    if off:
        issues.append(f"{off} timestamp gaps not a multiple of 60s (downtime / missing data)")

    return {
        "rows": len(df),
        "ok": len(issues) == 0,
        "issues": issues,
        "first": pd.to_datetime(ts.iloc[0], unit="ms", utc=True),
        "last": pd.to_datetime(ts.iloc[-1], unit="ms", utc=True),
    }
