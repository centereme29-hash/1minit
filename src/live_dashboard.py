"""Live trading-prediction dashboard (self-contained Plotly HTML).

Fetches the latest candles from Bybit, rebuilds the exact training features,
runs the baseline ensemble, and renders the current LONG / SHORT / NO-TRADE
signal together with a per-model breakdown and a recent price chart.

Usage::

    python src/live_dashboard.py            # one shot: fetch -> predict -> open
    python src/live_dashboard.py --live     # loop, UTC-aligned, rewrite + reload
    python src/live_dashboard.py --transformer   # use the transformer instead
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live import fetch_wide, predict_baseline, predict_transformer  # noqa: E402
from src.features import build_features  # noqa: E402
from src.baselines import CLASS_NAMES, predict_proba  # noqa: E402
from src.ensemble import load_models  # noqa: E402
from src.params import CONFIDENCE_GATE, MAIN_SYMBOL, TARGET_HORIZON  # noqa: E402

# Root-level chart helpers (candles + indicators + support/resistance + signals)
from features import add_emas  # noqa: E402
from indicators import add_indicators  # noqa: E402
from signals import add_liquidity, add_signals, add_support_resistance  # noqa: E402

BG = "#0b0e11"
GREEN = "#0ecb81"
RED = "#f6465d"
AMBER = "#f0b90b"
CYAN = "#00c3ff"
GRAY = "#8a8f98"

OUT = ROOT / "data" / "live_prediction.html"


def _action_color(action: str) -> str:
    if action.startswith("LONG"):
        return GREEN
    if action.startswith("SHORT"):
        return RED
    return GRAY


def to_signal(cls_name: str, confidence: float) -> str:
    """Map the raw class (UP/DOWN/NO_MOVE) to a friendly trade signal."""
    if confidence < CONFIDENCE_GATE:
        return "NO TRADE"
    if cls_name == "UP":
        return "LONG"
    if cls_name == "DOWN":
        return "SHORT"
    return "NO TRADE"


def per_model_breakdown(models: dict, X_row: np.ndarray) -> list[dict]:
    rows = []
    for name, m in models.items():
        p = predict_proba(m, X_row)[0]
        cls = int(p.argmax())
        rows.append({"model": name, "class": CLASS_NAMES[cls], "confidence": float(p.max())})
    return rows


def prepare_chart(wide: pd.DataFrame) -> pd.DataFrame:
    """Extract the main symbol's candles and add indicators, S/R and signals."""
    main = MAIN_SYMBOL
    df = wide[[
        "time", f"{main}_open", f"{main}_high", f"{main}_low",
        f"{main}_close", f"{main}_volume", f"{main}_turnover",
    ]].rename(columns={
        f"{main}_open": "open", f"{main}_high": "high", f"{main}_low": "low",
        f"{main}_close": "close", f"{main}_volume": "volume", f"{main}_turnover": "turnover",
    })
    df = add_emas(df)
    df = add_indicators(df)
    df = add_support_resistance(df, window=50)
    df = add_liquidity(df, window=20)
    df = add_signals(df)
    return df.tail(120).reset_index(drop=True)


def build_figure(wide: pd.DataFrame, action: str, confidence: float,
                 breakdown: list[dict]) -> go.Figure:
    main = MAIN_SYMBOL
    df = prepare_chart(wide)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
        row_heights=[0.46, 0.14, 0.20, 0.20],
        vertical_spacing=0.04,
        subplot_titles=(
            f"{main} price + EMA + support/resistance",
            "Volume + OBV", "RSI (14)", "MACD (12, 26, 9)",
        ),
    )

    # --- 1) price + EMA + Bollinger + support/resistance -------------------
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=GREEN, decreasing_line_color=RED, name=main,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA5"], name="EMA5",
                             line=dict(color=AMBER, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA20"], name="EMA20",
                             line=dict(color=CYAN, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_upper"], name="BB upper",
                             line=dict(color="#9b59b6", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_lower"], name="BB lower",
                             line=dict(color="#9b59b6", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["support"], name="support",
                             line=dict(color=GREEN, width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["resistance"], name="resistance",
                             line=dict(color=RED, width=2, dash="dash")), row=1, col=1)
    buys = df.loc[df["buy_signal"]]
    sells = df.loc[df["sell_signal"]]
    fig.add_trace(go.Scatter(x=buys["time"], y=buys["low"] * 0.998, mode="markers",
                             name="buy", marker=dict(symbol="triangle-up", color=GREEN, size=12)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sells["time"], y=sells["high"] * 1.002, mode="markers",
                             name="sell", marker=dict(symbol="triangle-down", color=RED, size=12)),
                  row=1, col=1)

    # --- 2) volume + OBV ----------------------------------------------------
    vol_colors = [GREEN if c >= o else RED for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["time"], y=df["volume"], name="volume",
                         marker_color=vol_colors, opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["OBV"], name="OBV",
                             line=dict(color=CYAN, width=1.2)), row=2, col=1)

    # --- 3) RSI -------------------------------------------------------------
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI14"], name="RSI",
                             line=dict(color=AMBER, width=1.4)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=RED, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=GREEN, row=3, col=1)

    # --- 4) MACD ------------------------------------------------------------
    fig.add_trace(go.Bar(x=df["time"], y=df["MACD_hist"], name="MACD hist",
                         marker_color=[GREEN if v >= 0 else RED for v in df["MACD_hist"]]),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["MACD"], name="MACD",
                             line=dict(color=CYAN, width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["MACD_signal"], name="signal",
                             line=dict(color=AMBER, width=1.2)), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#d9d9d9", size=11),
        margin=dict(l=45, r=20, t=60, b=30),
        height=1050,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor="#1e222d")
    fig.update_yaxes(gridcolor="#1e222d")
    return fig


def _header_html(wide: pd.DataFrame, action: str, confidence: float,
                 breakdown: list[dict], refresh: int | None) -> str:
    main = MAIN_SYMBOL
    price = float(wide[f"{main}_close"].iloc[-1])
    ts = wide["time"].iloc[-1]
    color = _action_color(action)

    models_line = " · ".join(
        f"<b>{r['model']}</b> <span style='color:{_action_color(r['class'])}'>{r['class']}</span> {r['confidence']:.0%}"
        for r in breakdown
    ) if breakdown else ""

    refresh_tag = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""

    conf_pct = confidence * 100.0

    return f"""<div style="font-family:Segoe UI,sans-serif;background:{BG};color:#d9d9d9;padding:16px 22px 8px;">
{refresh_tag}
<div style="display:flex;align-items:center;gap:22px;flex-wrap:wrap;">
  <div>
    <div style="font-size:13px;color:#8a8f98">1minit · {main} · {ts:%Y-%m-%d %H:%M:%S} UTC · price {price:.6g}</div>
    <div style="font-size:46px;font-weight:800;color:{color};line-height:1.05">{action}</div>
    <div style="font-size:13px;color:#8a8f98;margin-top:2px">confidence {confidence:.1%} · trade gate {CONFIDENCE_GATE:.0%} · horizon {TARGET_HORIZON}m</div>
    <div style="width:340px;background:#1e222d;height:10px;border-radius:6px;margin-top:8px;position:relative">
      <div style="width:{conf_pct:.1f}%;background:{color};height:10px;border-radius:6px"></div>
      <div style="position:absolute;left:{CONFIDENCE_GATE*100:.0f}%;top:-3px;width:2px;height:16px;background:{AMBER}"></div>
    </div>
  </div>
  <div style="font-size:13px;color:#8a8f98;max-width:520px">{models_line}</div>
</div>
</div>"""


def write_dashboard(fig: go.Figure, wide: pd.DataFrame, action: str, confidence: float,
                    breakdown: list[dict], out: Path, refresh: int | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True)
    html = out.read_text(encoding="utf-8")
    header = _header_html(wide, action, confidence, breakdown, refresh)
    html = html.replace("<body>", "<body>\n" + header, 1)
    out.write_text(html, encoding="utf-8")


def run_once(models: dict, transformer: bool, out: Path, refresh: int | None,
             open_browser: bool) -> None:
    wide = fetch_wide(800 if transformer else 320)
    feats = build_features(wide)

    if transformer:
        cls_name, confidence = predict_transformer(feats)
        breakdown = []
    else:
        cols = [c for c in feats.columns if c != "time"]
        X = feats[cols].fillna(0.0).to_numpy(dtype="float32")
        cls_name, confidence = predict_baseline(feats)
        breakdown = per_model_breakdown(models, X[-1:])

    action = to_signal(cls_name, confidence)
    fig = build_figure(wide, action, confidence, breakdown)
    write_dashboard(fig, wide, action, confidence, breakdown, out, refresh)
    print(f"[{wide['time'].iloc[-1]:%Y-%m-%d %H:%M:%S} UTC] {action}  "
          f"confidence={confidence:.1%}  -> {out}")

    if open_browser:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Live trading-prediction dashboard")
    parser.add_argument("--live", action="store_true", help="loop, UTC-aligned refresh")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--refresh", type=int, default=60)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()

    models = {} if args.transformer else load_models()
    out = Path(args.output)
    opened = False

    while True:
        try:
            run_once(models, args.transformer, out,
                     refresh=args.refresh if args.live else None,
                     open_browser=(not args.no_open and not opened))
            opened = True
        except Exception as exc:
            print(f"ERROR: {exc}")
        if not args.live:
            break
        # sleep until a few seconds after the next UTC minute boundary
        now = time.time()
        time.sleep(60 - (now % 60) + 2)


if __name__ == "__main__":
    main()

