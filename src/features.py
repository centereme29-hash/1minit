"""Step 5-8 — feature engineering (DOGE technical + cross-asset + multi-timeframe).

Reads the synchronized wide table and produces one row of features per UTC
minute for the MAIN symbol (DOGE), using BTC / XAUT as cross-asset context.
Indicators are computed inline (no external deps) and normalized where useful.
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
    MICRO_DIR,
    MICRO_FIELDS,
    SYMBOLS,
    SYNC_DIR,
)

C = f"{MAIN_SYMBOL}_close"
O = f"{MAIN_SYMBOL}_open"
H = f"{MAIN_SYMBOL}_high"
L = f"{MAIN_SYMBOL}_low"
V = f"{MAIN_SYMBOL}_volume"
BTC_C = "BTCUSDT_close"
XAUT_C = "XAUTUSDT_close"


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(avg_loss != 0.0, 100.0)


def _atr(df: pd.DataFrame, h: str, l: str, c: str, period: int = 14) -> pd.Series:
    prev_close = df[c].shift()
    tr = pd.concat([df[h] - df[l], (df[h] - prev_close).abs(), (df[l] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def load_micro_features(time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Merge ``data/microstructure/<SYMBOL>_1m_micro.csv`` onto a UTC minute index.

    Each micro column is prefixed ``<SYMBOL>_micro_`` so it never collides with
    the candle features.  Returns an empty DataFrame (index only) if no micro
    data has been collected yet, so the pipeline stays runnable without it.
    """
    index = pd.DatetimeIndex(pd.to_datetime(time_index, utc=True))
    frames: list[pd.DataFrame] = []
    for symbol in SYMBOLS:
        path = MICRO_DIR / f"{symbol}_1m_micro.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "time" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop_duplicates(subset="time", keep="last").set_index("time")
        keep = [c for c in MICRO_FIELDS if c in df.columns]
        if not keep:
            continue
        renamed = {c: f"{symbol}_micro_{c}" for c in keep}
        frames.append(df[keep].rename(columns=renamed))

    if not frames:
        return pd.DataFrame(index=index)
    micro = pd.concat(frames, axis=1)
    return micro.reindex(index)


def build_features(wide: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"time": wide["time"]})
    close = wide[C]
    eps = 1e-12

    # --- Returns (multi-timeframe) -------------------------------------------
    for h in (1, 2, 3, 5, 10, 15, 30, 60):
        out[f"ret_{h}m"] = close.pct_change(h, fill_method=None)

    # --- Volatility (rolling std of 1m returns) ------------------------------
    r1 = close.pct_change(fill_method=None)
    for w in (5, 15, 30, 60):
        out[f"vol_{w}m"] = r1.rolling(w, min_periods=w).std()

    # --- EMA / SMA mean-reversion ratios -------------------------------------
    for span in (5, 10, 20, 50, 100):
        ema = close.ewm(span=span, adjust=False).mean()
        out[f"ema_{span}"] = close / ema - 1.0
    for span in (20, 50, 100):
        sma = close.rolling(span, min_periods=span).mean()
        out[f"sma_{span}"] = close / sma - 1.0

    # --- RSI ----------------------------------------------------------------
    for p in (7, 14, 21):
        out[f"rsi_{p}"] = _rsi(close, p)

    # --- MACD (normalized by price) ------------------------------------------
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / close
    out["macd_signal"] = macd_signal / close
    out["macd_hist"] = (macd - macd_signal) / close

    # --- Bollinger %B + bandwidth --------------------------------------------
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    out["bb_pctb"] = (close - lower) / (upper - lower + eps)
    out["bb_bw"] = (upper - lower) / (mid + eps)

    # --- ATR % ----------------------------------------------------------------
    atr14 = _atr(wide, H, L, C, 14)
    out["atr_pct"] = atr14 / close


    # --- Stochastic -----------------------------------------------------------
    low_min = wide[L].rolling(14, min_periods=14).min()
    high_max = wide[H].rolling(14, min_periods=14).max()
    k_raw = 100 * (close - low_min) / (high_max - low_min + eps)
    out["stoch_k"] = k_raw.rolling(3, min_periods=1).mean()
    out["stoch_d"] = out["stoch_k"].rolling(3, min_periods=1).mean()

    # --- OBV (slopes) ----------------------------------------------------------
    obv = (np.sign(close.diff()).fillna(0.0) * wide[V]).cumsum()
    out["obv_diff_5"] = obv.diff(5)
    out["obv_diff_20"] = obv.diff(20)

    # --- Volume / price microstructure ----------------------------------------
    vol_ma20 = wide[V].rolling(20, min_periods=20).mean()
    out["vol_ratio"] = wide[V] / (vol_ma20 + eps)
    out["vol_accel"] = out["vol_ratio"].diff()
    out["price_accel"] = close.pct_change(fill_method=None).diff()
    out["range_pct"] = (wide[H] - wide[L]) / (close + eps)
    out["body_pct"] = (close - wide[O]) / (wide[H] - wide[L] + eps)

    # --- Cross-asset returns ---------------------------------------------------
    btc_c = wide[BTC_C]
    xaut_c = wide[XAUT_C]
    out["btc_ret_1m"] = btc_c.pct_change(fill_method=None)
    out["btc_ret_5m"] = btc_c.pct_change(5, fill_method=None)
    out["xaut_ret_1m"] = xaut_c.pct_change(fill_method=None)
    out["xaut_ret_5m"] = xaut_c.pct_change(5, fill_method=None)

    # Relative strength / spreads
    out["doge_btc_1m"] = out["ret_1m"] - out["btc_ret_1m"]
    out["doge_btc_5m"] = out["ret_5m"] - out["btc_ret_5m"]
    out["doge_xaut_5m"] = out["ret_5m"] - out["xaut_ret_5m"]

    # Rolling correlations (of 1m returns)
    for w in (15, 60):
        out[f"corr_doge_btc_{w}m"] = out["ret_1m"].rolling(w, min_periods=5).corr(out["btc_ret_1m"])
        out[f"corr_doge_xaut_{w}m"] = out["ret_1m"].rolling(w, min_periods=5).corr(out["xaut_ret_1m"])

    # Divergence flags
    tiny = 0.0005
    out["btc_up_doge_flat"] = ((out["btc_ret_5m"] > 0) & (out["ret_5m"].abs() < tiny)).astype(int)
    out["btc_up_doge_down"] = ((out["btc_ret_5m"] > 0) & (out["ret_5m"] < -tiny)).astype(int)
    out["btc_down_doge_up"] = ((out["btc_ret_5m"] < 0) & (out["ret_5m"] > tiny)).astype(int)
    out["xaut_up_doge_down"] = ((out["xaut_ret_5m"] > 0) & (out["ret_5m"] < -tiny)).astype(int)

    # --- Microstructure merge (Phase 6A) ------------------------------------
    micro = load_micro_features(wide["time"])
    if len(micro.columns):
        micro = micro.copy()
        micro.index.name = "time"
        out = out.merge(micro.reset_index(), on="time", how="left")

    return out


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 5-8 — feature engineering")

    wide = pd.read_csv(SYNC_DIR / "all_1m.csv")
    wide["time"] = pd.to_datetime(wide["time"], utc=True)

    feats = build_features(wide)
    out_path = FEATURES_DIR / "features.csv"
    feats.to_csv(out_path, index=False)

    print(f"features = {feats.shape[1] - 1}  rows = {feats.shape[0]}")
    print(f"saved -> {out_path}")
    print("columns:", list(feats.columns))
    print("\nSTEP 5-8 DONE.")


if __name__ == "__main__":
    main()

