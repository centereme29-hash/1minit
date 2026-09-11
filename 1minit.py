"""1minit — single-file live Bybit 1-minute collector + analysis + chart.

Everything runs from this one file: it fetches live 1-minute candles from
Bybit's public API, appends them to CSV files (never overwriting), computes the
12 core fields plus traditional indicators, liquidity, support/resistance and
buy/sell signals, then plots everything in an interactive HTML chart (UTC time,
single time column).

Usage::

    python 1minit.py                          # collect + append + chart + open
    python 1minit.py --live                   # loop, timed to the UTC candle close
    python 1minit.py --collect-only           # only collect & store (no chart)
    python 1minit.py --minutes 500 --no-open  # custom window, no browser
    python 1minit.py --symbol BTCUSDT --compare XAUTUSDT SOLUSDT
    python 1minit.py --file data/analysis/DOGEUSDT_1m_analysis.csv --no-open

Output (under ``data/``):
    raw/        <SYMBOL>_1m_raw.csv        raw OHLCV + turnover (append)
    features/   <SYMBOL>_1m_features.csv   12-field dataset (append)
    indicators/ <SYMBOL>_1m_indicators.csv all indicators (single UTC time)
    analysis/   <SYMBOL>_1m_analysis.csv   everything: indicators + liquidity +
                                           support/resistance + buy/sell signals
    <SYMBOL>_chart.html                    the interactive chart
"""
from __future__ import annotations

import argparse
import math
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"
CATEGORY = "spot"                                  # "spot" | "linear" | "inverse"
INTERVAL = "1"                                     # 1-minute candles

PRIMARY_SYMBOLS = ["DOGEUSDT"]
SUPPORT_SYMBOLS = ["BTCUSDT", "XAUTUSDT"]
DEFAULT_SYMBOLS = PRIMARY_SYMBOLS + SUPPORT_SYMBOLS

EMA_SPANS = (5, 10, 20, 50, 100)
SUPPORT_WINDOW = 50       # rolling candles for dynamic support/resistance
LIQUIDITY_WINDOW = 20     # rolling candles for the liquidity moving average
SIGNAL_FAST = 5           # fast EMA for cross signals
SIGNAL_SLOW = 20          # slow EMA for cross signals

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
INDICATORS_DIR = DATA_DIR / "indicators"
ANALYSIS_DIR = DATA_DIR / "analysis"

DEFAULT_MINUTES = 1500    # collect-only default (enough to warm EMA100)
CHART_MINUTES = 300       # chart display default


# ---------------------------------------------------------------------------
# Bybit V5 REST client (public kline endpoint, no API key)
# ---------------------------------------------------------------------------
KLINE_COLUMNS = ["timestamp_ms", "open", "high", "low", "close", "volume", "turnover"]

INTERVAL_MS = {
    "1": 60_000,
    "3": 180_000,
    "5": 300_000,
    "15": 900_000,
    "30": 1_800_000,
    "60": 3_600_000,
}


class BybitClient:
    """Fetch and parse OHLCV + turnover candles from Bybit's public API."""

    def __init__(
        self,
        base_url: str = BYBIT_BASE_URL,
        category: str = CATEGORY,
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.category = category
        self.timeout = timeout
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str = "1",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Return a single page of klines as a DataFrame, ascending by time."""
        limit = max(1, min(int(limit), 1000))
        params = {
            "category": self.category,
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)

        payload = self._request("/v5/market/kline", params=params)
        rows = payload.get("result", {}).get("list", [])

        if not rows:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        for col in KLINE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
        return df.sort_values("timestamp_ms").reset_index(drop=True)

    def fetch_history(self, symbol: str, minutes: int, interval: str = "1") -> pd.DataFrame:
        """Fetch the most recent ``minutes`` candles, paginating backwards."""
        interval_ms = INTERVAL_MS.get(interval, 60_000)
        target = max(1, int(minutes))
        frames: list[pd.DataFrame] = []
        cursor = int(time.time() * 1000)
        collected = 0

        while collected < target:
            limit = min(1000, target - collected)
            page = self.get_klines(symbol, interval=interval, end_ms=cursor, limit=limit)
            if page.empty:
                break
            frames.append(page)
            collected += len(page)

            oldest = int(page["timestamp_ms"].iloc[0])
            if oldest >= cursor:
                break
            cursor = oldest - interval_ms

        if not frames:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms")
        return out.tail(target).reset_index(drop=True)

    def _request(self, path: str, params: Optional[dict] = None, retries: int = 3) -> dict:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                data = resp.json()
                if data.get("retCode") == 0:
                    return data
                last_error = RuntimeError(
                    f"Bybit retCode={data.get('retCode')} retMsg={data.get('retMsg')}"
                )
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
            time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"Bybit request failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# CSV persistence: append + de-duplicate (live collection accumulates)
# ---------------------------------------------------------------------------
def append_csv(path: str | Path, df: pd.DataFrame, key: str) -> Path:
    """Append ``df`` to ``path``, de-duplicating on ``key`` (newest wins)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    old = pd.DataFrame()
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()

    if old.empty:
        merged = df.copy()
    else:
        cols = list(dict.fromkeys([*old.columns, *df.columns]))
        merged = pd.concat(
            [old.reindex(columns=cols), df.reindex(columns=cols)],
            ignore_index=True,
        )

    if key in merged.columns and not merged.empty:
        merged["_key"] = merged[key].astype(str)
        merged = merged.drop_duplicates(subset="_key", keep="last").drop(columns="_key")

    if key in merged.columns and not merged.empty:
        try:
            merged = merged.sort_values(key)
        except (TypeError, ValueError):
            pass
    merged = merged.reset_index(drop=True)

    merged.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Features: the 12 core Step-1 fields
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "timestamp_utc", "open", "high", "low", "close", "volume", "turnover",
    "EMA5", "EMA10", "EMA20", "EMA50", "EMA100",
]


def add_emas(df: pd.DataFrame, spans: tuple[int, ...] = EMA_SPANS) -> pd.DataFrame:
    """Append EMA5..EMA100 computed on close (min_periods=N leaves warm-up NaN)."""
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


# ---------------------------------------------------------------------------
# Traditional technical indicators (pure pandas, no TA-Lib)
# ---------------------------------------------------------------------------
def sma(close: pd.Series, period: int = 20) -> pd.Series:
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
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "MACD_signal": signal_line, "MACD_hist": hist})


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


# ---------------------------------------------------------------------------
# Signals: liquidity, support/resistance and buy/sell indications
# ---------------------------------------------------------------------------
def add_liquidity(df: pd.DataFrame, window: int = LIQUIDITY_WINDOW) -> pd.DataFrame:
    """Liquidity proxy from quote turnover (USDT traded per candle)."""
    out = df.copy()
    out["liquidity"] = out["turnover"]
    out["liquidity_ma"] = out["turnover"].rolling(window, min_periods=1).mean()
    return out


def add_support_resistance(df: pd.DataFrame, window: int = SUPPORT_WINDOW) -> pd.DataFrame:
    """Dynamic support/resistance as a Donchian-style band."""
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

    Buy (signal +1): EMA(fast) crosses above EMA(slow), or RSI14 crosses back
    above 30.  Sell (signal -1): EMA(fast) crosses below EMA(slow), or RSI14
    crosses back below 70.
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


# ---------------------------------------------------------------------------
# Lightweight sanity checks for collected candles
# ---------------------------------------------------------------------------
def validate_raw(df: pd.DataFrame) -> dict:
    """Validate a raw candle DataFrame; returns a small report dict."""
    required = ["timestamp_ms", "open", "high", "low", "close", "volume", "turnover"]

    if df.empty:
        return {"rows": 0, "ok": False, "issues": ["empty dataframe"]}

    issues = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"rows": len(df), "ok": False, "issues": [f"missing columns {missing}"]}

    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        issues.append("high is below open/close/low")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        issues.append("low is above open/close/high")
    if (df[["open", "high", "low", "close", "volume", "turnover"]] < 0).any().any():
        issues.append("negative values detected")

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


# ---------------------------------------------------------------------------
# Charting: colours, helpers, data pipeline and HTML output
# ---------------------------------------------------------------------------
BG = "#0b0e11"
PANEL = "#0b0e11"
GRID = "#1e222d"
TEXT = "#d9d9d9"
UP_GREEN = "#0ecb81"
DOWN_RED = "#f6465d"

EMA_COLORS = {
    "EMA5": "#f0b90b",
    "EMA10": "#ff8c00",
    "EMA20": "#00c3ff",
    "EMA50": "#9b59b6",
    "EMA100": "#e91e63",
}

MAIN_SYMBOL = PRIMARY_SYMBOLS[0]
CMP_COLORS = {"BTCUSDT": "#f7931a", "XAUTUSDT": "#c0c0c0", "DOGEUSDT": "#bc8d02"}

LIQUIDITY_COLOR = "#00c3ff"
LIQUIDITY_MA_COLOR = "#f0b90b"
SUPPORT_COLOR = "#0ecb81"
RESIST_COLOR = "#f6465d"
BUY_COLOR = "#0ecb81"
SELL_COLOR = "#f6465d"


def fetch_fresh(symbol: str, minutes: int) -> pd.DataFrame:
    """Fetch the latest candles live from Bybit (raw OHLCV + turnover)."""
    client = BybitClient()
    raw = client.fetch_history(symbol, minutes=max(1, int(minutes)) + 120, interval="1")
    if raw.empty:
        raise RuntimeError(f"no live data returned for {symbol}")
    return raw


def store_data(symbol: str, raw: pd.DataFrame) -> tuple[Path, Path]:
    """Persist raw + 12-field CSVs (append) and return their paths."""
    ds = build_step1_dataset(raw)
    raw_path = RAW_DIR / f"{symbol}_1m_raw.csv"
    feat_path = FEATURES_DIR / f"{symbol}_1m_features.csv"
    append_csv(raw_path, raw, key="timestamp_ms")
    append_csv(feat_path, ds, key="timestamp_utc")
    return raw_path, feat_path


def store_indicators(symbol: str, view: pd.DataFrame) -> Path:
    """Persist all candles + indicators with a single UTC time column."""
    out = view.copy()
    out["timestamp_utc"] = pd.to_datetime(out["time"], utc=True)
    out = out.drop(columns=["time", "timestamp_ms"], errors="ignore")
    cols = ["timestamp_utc"] + [c for c in out.columns if c != "timestamp_utc"]
    path = INDICATORS_DIR / f"{symbol}_1m_indicators.csv"
    append_csv(path, out[cols], key="timestamp_utc")
    return path


def store_analysis(symbol: str, view: pd.DataFrame) -> Path:
    """Persist the complete enriched dataset to ``data/analysis``."""
    out = view.copy()
    out["timestamp_utc"] = pd.to_datetime(out["time"], utc=True)
    out = out.drop(columns=["time", "timestamp_ms"], errors="ignore")
    cols = ["timestamp_utc"] + [c for c in out.columns if c != "timestamp_utc"]
    path = ANALYSIS_DIR / f"{symbol}_1m_analysis.csv"
    append_csv(path, out[cols], key="timestamp_utc")
    return path


def to_view(raw: pd.DataFrame, indicators: bool = False) -> pd.DataFrame:
    """Add EMAs, a display ``time`` column and (optionally) indicators + signals."""
    df = add_emas(raw)
    if indicators:
        df = add_indicators(df)
        df = add_analysis(df)
    df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def load_data(symbol: str, file_path: str | None, minutes: int) -> pd.DataFrame:
    """Load candles from a CSV file, or fetch them live from Bybit otherwise."""
    if file_path:
        path = Path(file_path)
        df = pd.read_csv(path)

        if "timestamp_utc" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        elif "timestamp_ms" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        else:
            raise ValueError(f"no timestamp column found in {path}")

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns {missing}")

        if "EMA5" not in df.columns:
            df = add_emas(df)
        if "RSI14" not in df.columns:
            df = add_indicators(df)
        if "support" not in df.columns:
            df = add_analysis(df)

        return df.sort_values("time").reset_index(drop=True)

    return to_view(fetch_fresh(symbol, minutes), indicators=True)


def sleep_to_candle(refresh: int) -> None:
    """Sleep until ~1s after the next UTC minute boundary (a new 1m candle)."""
    while True:
        now = time.time()
        wait = (math.floor(now / 60.0) + 1) * 60.0 + 1.0 - now
        if wait > refresh:
            time.sleep(max(1.0, float(refresh - 1)))
            continue
        time.sleep(max(0.05, wait))
        break


def indicator_snapshot(view: pd.DataFrame) -> str:
    """One-line snapshot of the latest indicator values (HTML-safe)."""
    last = view.iloc[-1]

    def val(key: str, fmt: str = ".2f") -> str:
        v = last.get(key) if key in last.index else None
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f"{key} --"
        return f"{key} {v:{fmt}}"

    parts = [
        val("RSI14"), val("MACD", ".6f"), val("MACD_signal", ".6f"),
        val("STOCH_K"), val("STOCH_D"), val("ATR14", ".6f"),
        val("Support", ".6f"), val("Resistance", ".6f"),
    ]
    obv_v = last.get("OBV")
    if obv_v is not None and not (isinstance(obv_v, float) and np.isnan(obv_v)):
        parts.append(f"OBV {obv_v:,.0f}")
    liq_v = last.get("liquidity")
    if liq_v is not None and not (isinstance(liq_v, float) and np.isnan(liq_v)):
        parts.append(f"Liquidity {liq_v:,.0f}")
    sig_v = last.get("signal")
    if sig_v == 1:
        parts.append("SIGNAL: BUY ▲")
    elif sig_v == -1:
        parts.append("SIGNAL: SELL ▼")
    return "  ·  ".join(parts)


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a bool-or-string column (CSV round-trip safe) to boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def build_figure(views: dict[str, pd.DataFrame], minutes: int, main_sym: str = MAIN_SYMBOL) -> go.Figure:
    """Build the full Bybit-styled 8-panel figure with all layers."""
    main = views[main_sym].tail(minutes).reset_index(drop=True)
    x = main["time"]

    fig = make_subplots(
        rows=8, cols=1, shared_xaxes=True,
        specs=[[{}], [{"secondary_y": True}], [{}], [{}], [{}], [{}], [{}], [{}]],
        row_heights=[0.24, 0.09, 0.08, 0.10, 0.08, 0.06, 0.15, 0.12],
        vertical_spacing=0.02,
    )

    # ----- Row 1: candles + EMAs + SMA20 + Bollinger bands ---------------------
    fig.add_trace(
        go.Candlestick(
            x=x, open=main["open"], high=main["high"],
            low=main["low"], close=main["close"], name="Price",
            increasing_line_color=UP_GREEN, increasing_fillcolor=UP_GREEN,
            decreasing_line_color=DOWN_RED, decreasing_fillcolor=DOWN_RED,
            whiskerwidth=0.4,
        ),
        row=1, col=1,
    )
    for ema, colour in EMA_COLORS.items():
        if ema in main.columns:
            fig.add_trace(
                go.Scatter(x=x, y=main[ema], name=ema, mode="lines",
                           line=dict(color=colour, width=1.1),
                           hovertemplate=f"{ema}: %{{y:.4f}}<extra></extra>"),
                row=1, col=1,
            )
    if "SMA20" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["SMA20"], name="SMA20", mode="lines",
                       line=dict(color="#e8e8e8", width=1.1, dash="dash"),
                       hovertemplate="SMA20: %{y:.4f}<extra></extra>"),
            row=1, col=1,
        )
    if "BB_upper" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["BB_upper"], name="BB upper", mode="lines",
                       line=dict(color="rgba(14,203,129,0.0)", width=0),
                       showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=main["BB_lower"], name="BB lower", mode="lines",
                       line=dict(color="rgba(14,203,129,0.0)", width=0),
                       fill="tonexty", fillcolor="rgba(14,203,129,0.06)",
                       showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )

    # ----- Support / resistance bands --------------------------------------------
    if "support" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["support"], name="Support", mode="lines",
                       line=dict(color=SUPPORT_COLOR, width=1.4, dash="dot"),
                       hovertemplate="Support: %{y:.6f}<extra></extra>"),
            row=1, col=1,
        )
    if "resistance" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["resistance"], name="Resistance", mode="lines",
                       line=dict(color=RESIST_COLOR, width=1.4, dash="dot"),
                       hovertemplate="Resistance: %{y:.6f}<extra></extra>"),
            row=1, col=1,
        )

    # ----- Buy / sell markers on the price panel ---------------------------------
    buy_mask = _as_bool(main["buy_signal"]) if "buy_signal" in main.columns else pd.Series(False, index=main.index)
    sell_mask = _as_bool(main["sell_signal"]) if "sell_signal" in main.columns else pd.Series(False, index=main.index)
    buys = main[buy_mask]
    sells = main[sell_mask]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(x=buys["time"], y=buys["low"] * 0.9995, mode="markers",
                       name="BUY", text=buys.get("signal_reason", ""),
                       marker=dict(symbol="triangle-up", size=12, color=BUY_COLOR,
                                   line=dict(width=1, color="#ffffff")),
                       hovertemplate="BUY @ %{y:.6f}<extra>%{text}</extra>"),
            row=1, col=1,
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(x=sells["time"], y=sells["high"] * 1.0005, mode="markers",
                       name="SELL", text=sells.get("signal_reason", ""),
                       marker=dict(symbol="triangle-down", size=12, color=SELL_COLOR,
                                   line=dict(width=1, color="#ffffff")),
                       hovertemplate="SELL @ %{y:.6f}<extra>%{text}</extra>"),
            row=1, col=1,
        )


    # ----- Row 2: volume bars + OBV line (secondary axis) ----------------------
    ups = main["close"] >= main["open"]
    vol_colours = [UP_GREEN if u else DOWN_RED for u in ups]
    fig.add_trace(
        go.Bar(x=x, y=main["volume"], name="Volume",
               marker_color=vol_colours, opacity=0.55,
               hovertemplate="Volume: %{y:.4f}<extra></extra>"),
        row=2, col=1,
    )
    if "OBV" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["OBV"], name="OBV", mode="lines",
                       line=dict(color="#7e57c2", width=1.4),
                       hovertemplate="OBV: %{y:,.0f}<extra></extra>"),
            row=2, col=1, secondary_y=True,
        )

    # ----- Row 3: RSI (30/70 zones) ---------------------------------------------
    if "RSI14" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["RSI14"], name="RSI14", mode="lines",
                       line=dict(color="#eed202", width=1.4),
                       hovertemplate="RSI14: %{y:.2f}<extra></extra>"),
            row=3, col=1,
        )
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(246,70,93,0.08)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(14,203,129,0.08)", line_width=0, row=3, col=1)
        fig.add_hline(y=70, line=dict(color="rgba(128,128,128,0.35)", dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="rgba(128,128,128,0.35)", dash="dash"), row=3, col=1)

    # ----- Row 4: MACD + signal + histogram --------------------------------------
    if "MACD" in main.columns:
        hist = main["MACD_hist"].fillna(0.0)
        fig.add_trace(
            go.Bar(x=x, y=hist, name="MACD hist",
                   marker_color=[UP_GREEN if v >= 0 else DOWN_RED for v in hist],
                   opacity=0.55, hovertemplate="hist: %{y:.6f}<extra></extra>"),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=main["MACD"], name="MACD", mode="lines",
                       line=dict(color="#00c3ff", width=1.4),
                       hovertemplate="MACD: %{y:.6f}<extra></extra>"),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=main["MACD_signal"], name="MACD signal", mode="lines",
                       line=dict(color="#f9a825", width=1.4),
                       hovertemplate="signal: %{y:.6f}<extra></extra>"),
            row=4, col=1,
        )
        fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.4)"), row=4, col=1)

    # ----- Row 5: Stochastic %K / %D (20/80 zones) --------------------------------
    if "STOCH_K" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["STOCH_K"], name="STOCH %K", mode="lines",
                       line=dict(color="#7b7ff0", width=1.3),
                       hovertemplate="%K: %{y:.2f}<extra></extra>"),
            row=5, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=main["STOCH_D"], name="STOCH %D", mode="lines",
                       line=dict(color="#f6465d", width=1.3),
                       hovertemplate="%D: %{y:.2f}<extra></extra>"),
            row=5, col=1,
        )
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(246,70,93,0.08)", line_width=0, row=5, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(14,203,129,0.08)", line_width=0, row=5, col=1)
        fig.add_hline(y=80, line=dict(color="rgba(128,128,128,0.3)", dash="dash"), row=5, col=1)
        fig.add_hline(y=20, line=dict(color="rgba(128,128,128,0.3)", dash="dash"), row=5, col=1)

    # ----- Row 6: ATR -------------------------------------------------------------
    if "ATR14" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["ATR14"], name="ATR14", mode="lines",
                       line=dict(color="#f77568", width=1.4),
                       hovertemplate="ATR14: %{y:.6f}<extra></extra>"),
            row=6, col=1,
        )


    # ----- Row 7: comparison — normalised % since window start ---------------------
    base_ts = main["time"].iloc[0]
    base_str = f"{base_ts:%H:%M} UTC"
    for sym in [main_sym, *views.keys()]:
        view = views[sym]
        w = view[view["time"] >= base_ts].reset_index(drop=True)
        if w.empty:
            continue
        pct = (w["close"] / w["close"].iloc[0] - 1.0) * 100.0
        fig.add_trace(
            go.Scatter(x=w["time"], y=pct, name=f"{sym} Δ%", mode="lines",
                       line=dict(color=CMP_COLORS.get(sym, "#888888"), width=1.5),
                       hovertemplate=f"{sym}: %{{y:.3f}}%<extra></extra>"),
            row=7, col=1,
        )
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.4)"), row=7, col=1)

    # ----- Row 8: Liquidity (turnover + rolling mean) ------------------------------
    if "liquidity" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["liquidity"], name="Liquidity", mode="lines",
                       line=dict(color=LIQUIDITY_COLOR, width=1.2),
                       hovertemplate="Liquidity: %{y:,.2f}<extra></extra>"),
            row=8, col=1,
        )
    if "liquidity_ma" in main.columns:
        fig.add_trace(
            go.Scatter(x=x, y=main["liquidity_ma"], name="Liquidity MA20", mode="lines",
                       line=dict(color=LIQUIDITY_MA_COLOR, width=1.2),
                       hovertemplate="Liq MA: %{y:,.2f}<extra></extra>"),
            row=8, col=1,
        )

    # ----- Layout (Bybit dark theme) ------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f"{main_sym} · 1m · Bybit  —  BTC vs XAUT  (base {base_str})",
            x=0.01, xanchor="left", font=dict(size=16, color=TEXT),
        ),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11),
        height=1400,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        showlegend=True,
    )

    for i in range(1, 9):
        fig.update_xaxes(gridcolor=GRID, showline=False, type="date", row=i, col=1)
        fig.update_yaxes(gridcolor=GRID, showline=False, zeroline=False, row=i, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="STOCH", row=5, col=1, range=[0, 100])
    fig.update_yaxes(title_text="ATR", row=6, col=1)
    fig.update_yaxes(title_text="Δ%", row=7, col=1)
    fig.update_yaxes(title_text="Liquidity", row=8, col=1)

    return fig


# Live-mode overlay: pulsing LIVE dot, real UTC clock, countdown to the next
# 1-minute candle close (UTC-based) and a one-line indicator snapshot.
_LIVE_OVERLAY = """<div id="livebar" style="position:fixed;top:12px;right:16px;z-index:9999;max-width:560px;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <div style="display:flex;align-items:center;gap:10px;background:rgba(11,14,17,.9);padding:8px 12px;border:1px solid #1e222d;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.5);">
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#0ecb81;animation:livepulse 1.5s infinite;"></span>
    <span style="color:#0ecb81;font-weight:700;font-size:14px;letter-spacing:1px;">LIVE</span>
    <span style="color:#3a4152;">|</span>
    <span id="utcclock" style="color:#00c3ff;font-size:14px;font-weight:600;">--:--:-- UTC</span>
    <span id="candledown" style="color:#f0b90b;font-size:14px;font-weight:600;">next candle in --s</span>
    <span style="color:#3a4152;">|</span>
    <span style="color:#d9d9d9;font-size:13px;">updated __UPDATED__</span>
  </div>
  <div id="indstrip" style="color:#d9d9d9;font-size:12px;margin-top:4px;background:rgba(11,14,17,.9);padding:4px 10px;border:1px solid #1e222d;border-radius:6px;">__INDICATORS__</div>
</div>
<style>@keyframes livepulse{0%{box-shadow:0 0 0 0 rgba(14,203,129,.7)}70%{box-shadow:0 0 0 8px rgba(14,203,129,0)}100%{box-shadow:0 0 0 0 rgba(14,203,129,0)}}</style>
<script>(function(){
  function pad(n){return (n<10?'0':'')+n;}
  var clock=document.getElementById('utcclock');
  var cd=document.getElementById('candledown');
  function tick(){
    var d=new Date();
    clock.textContent=pad(d.getUTCHours())+':'+pad(d.getUTCMinutes())+':'+pad(d.getUTCSeconds())+' UTC';
    var rem=60-d.getUTCSeconds();
    cd.textContent='next candle in '+rem+'s';
    cd.style.color=(rem<=10)?'#f6465d':'#f0b90b';
  }
  tick();
  setInterval(tick,250);
})();</script>
"""


def write_chart_html(
    fig: go.Figure,
    out_path: Path,
    refresh_seconds: int | None = None,
    last_ts=None,
    indicators: str = "",
) -> None:
    """Write the figure to HTML (with live UTC overlay when ``refresh_seconds`` set)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)

    if refresh_seconds:
        text = out_path.read_text(encoding="utf-8")

        meta = f'<meta http-equiv="refresh" content="{refresh_seconds}">'
        if meta not in text:
            text = text.replace("<head>", "<head>\n    " + meta + "\n", 1)

        updated = f"{last_ts:%H:%M:%S} UTC" if last_ts is not None else "now"
        overlay = (
            _LIVE_OVERLAY.replace("__UPDATED__", updated)
            .replace("__INDICATORS__", indicators)
        )
        if 'id="livebar"' not in text:
            text = text.replace("</body>", overlay + "\n</body>", 1)

        out_path.write_text(text, encoding="utf-8")


def run_collect(symbols: list[str], minutes: int, category: str) -> None:
    """Collect live data for each symbol, compute everything, and store it all."""
    client = BybitClient(category=category)
    for d in (DATA_DIR, RAW_DIR, FEATURES_DIR, INDICATORS_DIR, ANALYSIS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("COLLECT — Bybit live 1-minute data + full analysis")
    print(f"category={category}  interval=1m  minutes={minutes}")
    print(f"symbols={symbols}")

    for symbol in symbols:
        sym = symbol.upper()
        print(f"\n{'=' * 72}\n[{sym}] fetching {minutes} x 1m candles ...")

        raw = client.fetch_history(sym, minutes=minutes, interval="1")
        if raw.empty:
            print(f"[{sym}] WARNING: no data returned (check symbol / category).")
            continue

        ds = build_step1_dataset(raw)
        view = to_view(raw, indicators=True)

        store_data(sym, raw)
        store_indicators(sym, view)
        store_analysis(sym, view)

        report = validate_raw(raw)
        print(f"[{sym}] rows={report['rows']}  valid={report['ok']}")
        if report.get("first") is not None:
            print(f"    first={report['first']}  last={report['last']}")
        for issue in report["issues"]:
            print(f"    ! {issue}")

        print(f"    raw       -> {RAW_DIR / f'{sym}_1m_raw.csv'}")
        print(f"    features  -> {FEATURES_DIR / f'{sym}_1m_features.csv'}")
        print(f"    indicators-> {INDICATORS_DIR / f'{sym}_1m_indicators.csv'}")
        print(f"    analysis  -> {ANALYSIS_DIR / f'{sym}_1m_analysis.csv'}")

        print(f"\n    --- last 3 rows ({sym}) ---")
        print(ds.tail(3).to_string(index=False))

    print("\n" + "=" * 72)
    print("COLLECT DONE — 12 fields + indicators + liquidity + support/resistance + signals")
    print("=" * 72)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="1minit — single-file live Bybit collector + analysis + chart")
    parser.add_argument("--symbol", default=MAIN_SYMBOL,
                        help="main symbol for the candlestick panel (default DOGEUSDT)")
    parser.add_argument("--compare", nargs="+", default=["BTCUSDT", "XAUTUSDT"],
                        help="symbols to compare in the bottom panel (default: BTCUSDT XAUTUSDT)")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="symbols to collect in --collect-only mode (default: DOGEUSDT BTCUSDT XAUTUSDT)")
    parser.add_argument("--minutes", type=int, default=None,
                        help="number of 1m candles (default: 300 chart / 1500 collect-only)")
    parser.add_argument("--category", default=CATEGORY,
                        help="Bybit category: spot|linear|inverse (default spot)")
    parser.add_argument("--file", default=None,
                        help="optional single CSV (offline mode; skips comparison)")
    parser.add_argument("--output", default=None,
                        help="output HTML path (default: data/<SYMBOL>_chart.html)")
    parser.add_argument("--live", action="store_true",
                        help="loop: fetch + append + chart, timed to the UTC candle close")
    parser.add_argument("--refresh", type=int, default=60,
                        help="max seconds between fetches in --live mode (default 60)")
    parser.add_argument("--no-open", action="store_true",
                        help="write the HTML without opening the browser")
    parser.add_argument("--collect-only", action="store_true",
                        help="only collect & store data (no chart)")
    args = parser.parse_args(argv)

    if args.collect_only:
        minutes = args.minutes or DEFAULT_MINUTES
        run_collect([s.upper() for s in args.symbols], minutes, args.category)
        return

    main_sym = args.symbol.upper()
    compare = [s.upper() for s in args.compare if s.upper() != main_sym]
    minutes = args.minutes or CHART_MINUTES
    out_path = Path(args.output) if args.output else BASE_DIR / "data" / f"{main_sym}_chart.html"
    refresh = args.refresh if args.live else None
    opened = False

    while True:
        try:
            if args.file:
                df = load_data(main_sym, args.file, minutes)
                views = {main_sym: df}
                stored: list[Path] = []
            else:
                views: dict[str, pd.DataFrame] = {}
                stored: list[Path] = []
                for sym in [main_sym, *compare]:
                    raw = fetch_fresh(sym, minutes)
                    stored.extend(store_data(sym, raw))
                    view = to_view(raw, indicators=(sym == main_sym))
                    views[sym] = view
                    if sym == main_sym:
                        stored.append(store_indicators(sym, view))
                        stored.append(store_analysis(sym, view))

            fig = build_figure(views, minutes, main_sym)
            last_ts = views[main_sym]["time"].iloc[-1]
            snap = indicator_snapshot(views[main_sym].tail(minutes))
            write_chart_html(fig, out_path, refresh_seconds=refresh,
                             last_ts=last_ts, indicators=snap)

            print(f"[{last_ts:%Y-%m-%d %H:%M:%S} UTC] chart saved -> {out_path} "
                  f"({len(views[main_sym].tail(minutes))} candles, {len(views)} symbols)")
            if stored:
                print(f"    data stored -> {stored[0]}, {stored[1]}, ... ({len(stored)} files)")

            if not args.no_open and not opened:
                webbrowser.open(out_path.resolve().as_uri())
                opened = True
        except Exception as exc:
            print(f"ERROR: {exc}")
            if not args.live:
                raise

        if not args.live:
            break

        sleep_to_candle(refresh)


if __name__ == "__main__":
    main()










