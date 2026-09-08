"""Interactive multi-symbol candlestick chart with traditional indicators.

Live data from Bybit (1-minute candles) for DOGEUSDT (main panel) plus BTCUSDT
and XAUTUSDT (comparison), rendered like the Bybit app with every traditional
mathematical trend indicator overlaid:

    price + EMA5/10/20/50/100 + SMA20 + Bollinger Bands,
    volume + OBV, RSI, MACD (+signal +histogram), Stochastic %K/%D, ATR,
    BTC vs XAUT (and DOGE) normalised comparison panel.

The HTML overlay shows a real **UTC clock** and a countdown to the next
1-minute candle close (UTC-based, not a browser-refresh timer).

Usage::

    python chart.py                      # live DOGEUSDT + BTC vs XAUT
    python chart.py --live               # loop: fetch+store+chart, UTC-aligned
    python chart.py --symbol BTCUSDT     # change the main symbol
    python chart.py --compare XAUTUSDT SOLUSDT  # custom comparison symbols
    python chart.py --minutes 500 --no-open
"""
from __future__ import annotations

import argparse
import math
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bybit_client import BybitClient
from config import BASE_DIR, FEATURES_DIR, RAW_DIR, PRIMARY_SYMBOLS
from features import add_emas, build_step1_dataset
from indicators import add_indicators

# --- Bybit-app colour palette ------------------------------------------------
BG = "#0b0e11"          # main background
PANEL = "#0b0e11"       # plot background
GRID = "#1e222d"        # grid lines
TEXT = "#d9d9d9"        # axis text / labels
UP_GREEN = "#0ecb81"    # Bybit up candle
DOWN_RED = "#f6465d"    # Bybit down candle

EMA_COLORS = {
    "EMA5": "#f0b90b",     # amber
    "EMA10": "#ff8c00",    # orange
    "EMA20": "#00c3ff",    # cyan
    "EMA50": "#9b59b6",    # purple
    "EMA100": "#e91e63",   # pink
}

# Main symbol is the big candlestick panel; the others are overlaid for comparison.
MAIN_SYMBOL = PRIMARY_SYMBOLS[0]                       # DOGEUSDT
CMP_COLORS = {"BTCUSDT": "#f7931a", "XAUTUSDT": "#c0c0c0", "DOGEUSDT": "#bc8d02"}

INDICATORS_DIR = BASE_DIR / "data" / "indicators"


def fetch_fresh(symbol: str, minutes: int) -> pd.DataFrame:
    """Fetch the latest candles live from Bybit (raw OHLCV + turnover).

    ``minutes + 120`` candles are fetched so EMA100 / Bollinger (needs 100/20
    candles) are fully warmed before the displayed window starts.
    """
    client = BybitClient()
    raw = client.fetch_history(symbol, minutes=max(1, int(minutes)) + 120, interval="1")
    if raw.empty:
        raise RuntimeError(f"no live data returned for {symbol}")
    return raw


def store_data(symbol: str, raw: pd.DataFrame) -> tuple[Path, Path]:
    """Persist the latest candles to the Step-1 CSV files and return their paths.

    Writes the same files as ``collect_step1.py``:
      - ``data/raw/<SYMBOL>_1m_raw.csv``        (raw OHLCV + turnover)
      - ``data/features/<SYMBOL>_1m_features.csv`` (the 12-field dataset)
    """
    ds = build_step1_dataset(raw)
    raw_path = RAW_DIR / f"{symbol}_1m_raw.csv"
    feat_path = FEATURES_DIR / f"{symbol}_1m_features.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    ds.to_csv(feat_path, index=False)
    return raw_path, feat_path


def store_indicators(symbol: str, view: pd.DataFrame) -> Path:
    """Persist all candles + their indicator columns (``data/indicators``)."""
    out = view.copy()
    out["timestamp_utc"] = pd.to_datetime(out["time"], utc=True)
    path = INDICATORS_DIR / f"{symbol}_1m_indicators.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["time"]).to_csv(path, index=False)
    return path


def to_view(raw: pd.DataFrame, indicators: bool = False) -> pd.DataFrame:
    """Add EMA columns, a display ``time`` column and (optionally) indicators."""
    df = add_emas(raw)
    if indicators:
        df = add_indicators(df)
    df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def load_data(symbol: str, file_path: str | None, minutes: int) -> pd.DataFrame:
    """Load candles from a CSV file, or fetch them live from Bybit otherwise."""
    if file_path:
        path = Path(file_path)
        df = pd.read_csv(path)

        # Timestamps: features CSV has ``timestamp_utc``, raw CSV has ``timestamp_ms``.
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

        # If the CSV is raw (no EMA/indicator columns), add them on the fly.
        if "EMA5" not in df.columns:
            df = add_emas(df)
        if "RSI14" not in df.columns:
            df = add_indicators(df)

        return df.sort_values("time").reset_index(drop=True)

    return to_view(fetch_fresh(symbol, minutes), indicators=True)


def sleep_to_candle(refresh: int) -> None:
    """Sleep until ~1s after the next UTC minute boundary (a new 1m candle).

    Wakes up at least every ``refresh`` seconds so the loop stays responsive
    (Ctrl+C) and so short ``--refresh`` values still update promptly.

    The countdown the browser shows is computed from UTC in JavaScript, so the
    fetch always lands just after the candle the countdown was aiming at.
    """
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
        val("RSI14"),
        val("MACD", ".6f"),
        val("MACD_signal", ".6f"),
        val("STOCH_K"),
        val("STOCH_D"),
        val("ATR14", ".6f"),
    ]
    obv_v = last.get("OBV")
    if obv_v is not None and not (isinstance(obv_v, float) and np.isnan(obv_v)):
        parts.append(f"OBV {obv_v:,.0f}")
    return "  ·  ".join(parts)


def build_figure(
    views: dict[str, pd.DataFrame],
    minutes: int,
    main_sym: str = MAIN_SYMBOL,
) -> go.Figure:
    """Build the full Bybit-styled multi-panel figure with all indicators.

    ``views`` maps each symbol to a candle frame (with ``time`` and EMAs); the
    main symbol must also carry the indicator columns from ``add_indicators``.

    Panels (top to bottom): price + EMAs + SMA + Bollinger, volume + OBV,
    RSI, MACD, Stochastic, ATR and the BTC vs XAUT (vs DOGE) comparison.
    """
    main = views[main_sym].tail(minutes).reset_index(drop=True)
    x = main["time"]

    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=True,
        specs=[[{}], [{"secondary_y": True}], [{}], [{}], [{}], [{}], [{}]],
        row_heights=[0.30, 0.12, 0.10, 0.12, 0.10, 0.07, 0.17],
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
        fig.add_trace(  # invisible mirror so the fill band works
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

    # ----- Row 7: BTC vs XAUT vs DOGE — normalised % since window start -----------
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

    # ----- Layout (Bybit dark theme) ------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f"{main_sym} · 1m · Bybit  —  BTC vs XAUT  (base {base_str})",
            x=0.01, xanchor="left", font=dict(size=16, color=TEXT),
        ),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11),
        height=1200,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        showlegend=True,
    )

    for i in range(1, 8):
        fig.update_xaxes(gridcolor=GRID, showline=False, type="date", row=i, col=1)
        fig.update_yaxes(gridcolor=GRID, showline=False, zeroline=False, row=i, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="STOCH", row=5, col=1, range=[0, 100])
    fig.update_yaxes(title_text="ATR", row=6, col=1)
    fig.update_yaxes(title_text="Δ%", row=7, col=1)

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
    """Write the figure to HTML.

    In live mode (``refresh_seconds`` set) the page reloads via a
    ``<meta http-equiv="refresh">`` tag and carries a UTC status bar: a real
    UTC clock and a countdown to the next 1-minute candle close (computed from
    UTC in JavaScript — the refresh tag only reloads the data), plus a snapshot
    line of the latest indicator values.
    """
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multi-symbol candlestick chart + traditional indicators")
    parser.add_argument("--symbol", default=MAIN_SYMBOL,
                        help=f"main symbol for the candlestick panel, e.g. {MAIN_SYMBOL}")
    parser.add_argument("--compare", nargs="+", default=["BTCUSDT", "XAUTUSDT"],
                        help="symbols to compare in the bottom panel (default: BTCUSDT XAUTUSDT)")
    parser.add_argument("--minutes", type=int, default=300,
                        help="number of most recent 1m candles to display")
    parser.add_argument("--file", default=None,
                        help="optional single CSV (offline mode; skips comparison)")
    parser.add_argument("--output", default=None,
                        help="output HTML path (default: data/<SYMBOL>_chart.html)")
    parser.add_argument("--live", action="store_true",
                        help="loop: fetch + store data + chart, timed to the UTC candle close")
    parser.add_argument("--refresh", type=int, default=60,
                        help="max seconds between fetches in --live mode (default 60)")
    parser.add_argument("--no-open", action="store_true",
                        help="write the HTML without opening the browser")
    args = parser.parse_args(argv)

    main_sym = args.symbol.upper()
    compare = [s.upper() for s in args.compare if s.upper() != main_sym]
    out_path = Path(args.output) if args.output else BASE_DIR / "data" / f"{main_sym}_chart.html"
    refresh = args.refresh if args.live else None
    opened = False

    while True:
        try:
            if args.file:
                df = load_data(main_sym, args.file, args.minutes)
                views = {main_sym: df}
                stored: list[Path] = []
            else:
                views: dict[str, pd.DataFrame] = {}
                stored: list[Path] = []
                for sym in [main_sym, *compare]:
                    raw = fetch_fresh(sym, args.minutes)
                    stored.extend(store_data(sym, raw))
                    view = to_view(raw, indicators=(sym == main_sym))
                    views[sym] = view
                    if sym == main_sym:
                        stored.append(store_indicators(sym, view))

            fig = build_figure(views, args.minutes, main_sym)
            last_ts = views[main_sym]["time"].iloc[-1]
            snap = indicator_snapshot(views[main_sym].tail(args.minutes))
            write_chart_html(fig, out_path, refresh_seconds=refresh,
                             last_ts=last_ts, indicators=snap)

            print(f"[{last_ts:%Y-%m-%d %H:%M:%S} UTC] chart saved -> {out_path} "
                  f"({len(views[main_sym].tail(args.minutes))} candles, "
                  f"{len(views)} symbols)")
            if stored:
                print(f"    data stored -> {stored[0]}, {stored[1]}, ... "
                      f"({len(stored)} files)")

            if not args.no_open and not opened:
                webbrowser.open(out_path.resolve().as_uri())
                opened = True
        except Exception as exc:  # keep live mode alive across a bad fetch
            print(f"ERROR: {exc}")
            if not args.live:
                raise

        if not args.live:
            break

        sleep_to_candle(refresh)


if __name__ == "__main__":
    main()

