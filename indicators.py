"""Traditional mathematical technical indicators for trend prediction.

Pure pandas implementations (no TA-Lib dependency).  Each function takes the
candle DataFrame / series and returns a Series or DataFrame, so this module
stays the single source of truth for indicator math.

Indicators provided:

    RSI          - relative strength index (Wilder's smoothing)
    MACD         - moving-average convergence/divergence + signal + histogram
    Bollinger    - Bollinger Bands (middle, upper, lower)
    Stochastic   - slow stochastic oscillator (%K, %D)
    ATR          - average true range (Wilder's smoothing)
    OBV          - on-balance volume
    SMA          - simple moving average
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average of ``close``."""
    return close.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # A zero average loss means every candle gained -> RSI 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"MACD": macd_line, "MACD_signal": signal_line, "MACD_hist": hist}
    )


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper and lower bands."""
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"BB_mid": mid, "BB_upper": upper, "BB_lower": lower})


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    k_smooth: int = 3,
    d_period: int = 3,
) -> pd.DataFrame:
    """Slow stochastic oscillator (%K, %D)."""
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()

    denom = (high_max - low_min).replace(0.0, np.nan)
    k_raw = 100 * (df["close"] - low_min) / denom
    k_line = k_raw.rolling(k_smooth, min_periods=1).mean()
    d_line = k_line.rolling(d_period, min_periods=1).mean()
    return pd.DataFrame({"STOCH_K": k_line, "STOCH_D": d_line})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (cumulative signed volume)."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append all traditional indicators to a candle DataFrame (with ``close``)."""
    out = df.copy()

    out["SMA20"] = sma(out["close"], 20)

    bb = bollinger(out["close"], 20, 2.0)
    out["BB_upper"] = bb["BB_upper"]
    out["BB_mid"] = bb["BB_mid"]
    out["BB_lower"] = bb["BB_lower"]

    out["RSI14"] = rsi(out["close"], 14)

    m = macd(out["close"])
    out["MACD"] = m["MACD"]
    out["MACD_signal"] = m["MACD_signal"]
    out["MACD_hist"] = m["MACD_hist"]

    st = stochastic(out)
    out["STOCH_K"] = st["STOCH_K"]
    out["STOCH_D"] = st["STOCH_D"]

    out["ATR14"] = atr(out, 14)
    out["OBV"] = obv(out)

    return out
