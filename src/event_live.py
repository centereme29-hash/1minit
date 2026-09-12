"""Step 6K - unified live dashboard (event model + baseline)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import MAIN_SYMBOL, MODELS_DIR, DATA_DIR
from src.event_engine import EventConfig, build_levels, build_stage, add_wick_features
from src.features import build_features
from src.live import fetch_wide, predict_baseline

EVENT_NAMES = {0: "NO_EVENT", 1: "SUPPORT_BREAKDOWN", 2: "RESISTANCE_BREAKOUT", 3: "FALSE_BREAK"}
BG = "#0b0e11"
GREEN = "#0ecb81"
RED = "#f6465d"
AMBER = "#f0b90b"
GRAY = "#8a8f98"

OUT = ROOT / "data" / "live_event_dashboard.html"

def build_event_features(wide):
    cfg = EventConfig()
    c = f"{MAIN_SYMBOL}_close"
    o = f"{MAIN_SYMBOL}_open"
    h = f"{MAIN_SYMBOL}_high"
    l = f"{MAIN_SYMBOL}_low"
    levels = build_levels(wide, cfg, c, h, l)
    stage = build_stage(levels, cfg)
    wicks = add_wick_features(wide, cfg, c, o, h, l)
    return levels, stage, wicks

def predict_event(wide, feats, model):
    levels, stage, wicks = build_event_features(wide)
    base_cols = [cc for cc in feats.columns if cc != "time" and "_micro_" not in cc]
    parts = []
    parts.append(feats[base_cols].iloc[-1].to_numpy(dtype=np.float32))
    parts.append(levels.iloc[-1].to_numpy(dtype=np.float32))
    parts.append(wicks.iloc[-1].to_numpy(dtype=np.float32))
    parts.append(np.array([float(stage[-1])], dtype=np.float32))
    x = np.concatenate(parts).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    probs = model.predict_proba(x.reshape(1, -1))[0]
    k = int(probs.argmax())
    conf = float(probs.max())
    direction = "DOWN" if k == 1 else ("UP" if k == 2 else "NEUTRAL")
    return {"event": EVENT_NAMES[k], "direction": direction, "confidence": conf, "stage": int(stage[-1]), "support": float(levels["support"].iloc[-1]), "resistance": float(levels["resistance"].iloc[-1]), "probs": {EVENT_NAMES[i]: float(probs[i]) for i in range(4)}}

def build_figure(wide, res, base_cls, base_conf, price):
    main = MAIN_SYMBOL
    cols = ["time", f"{main}_open", f"{main}_high", f"{main}_low", f"{main}_close", f"{main}_volume"]
    df = wide[cols].rename(columns={f"{main}_open": "open", f"{main}_high": "high", f"{main}_low": "low", f"{main}_close": "close", f"{main}_volume": "volume"}).tail(120).reset_index(drop=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=main, increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    fig.add_hline(y=res["support"], line_color="blue", line_width=1.2, row=1, col=1)
    fig.add_hline(y=res["resistance"], line_color="red", line_width=1.2, row=1, col=1)
    colors = [GREEN if c >= o else RED for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df["time"], y=df["volume"], marker_color=colors, name="volume"), row=2, col=1)
    evc = {"SUPPORT_BREAKDOWN": RED, "RESISTANCE_BREAKOUT": GREEN, "FALSE_BREAK": AMBER}.get(res["event"], GRAY)
    fig.add_trace(go.Scatter(x=[df["time"].iloc[-1]], y=[df["close"].iloc[-1]], mode="markers", marker=dict(symbol="circle", color=evc, size=13, line=dict(color="white", width=1)), name="event"), row=1, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=680, xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def run_once(model, out, open_browser):
    wide = fetch_wide(800)
    feats = build_features(wide)
    feats = feats[[c for c in feats.columns if '_micro_' not in c]]
    res = predict_event(wide, feats, model)
    base_cls, base_conf = predict_baseline(feats)
    price = float(wide[f"{MAIN_SYMBOL}_close"].iloc[-1])
    ts = wide["time"].iloc[-1]
    fig = build_figure(wide, res, base_cls, base_conf, price)
    title = "{main}  |  EVENT {event} ({conf:.0f}% stage {stage})  |  BASELINE {base_cls} ({base_conf:.0f}%)  |  S {sup:.6g}  R {resl:.6g}".format(main=MAIN_SYMBOL, event=res["event"], conf=100.0 * res["confidence"], stage=res["stage"], base_cls=base_cls, base_conf=100.0 * base_conf, sup=res["support"], resl=res["resistance"])
    fig.update_layout(title=title, title_font=dict(size=13))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=True)
    payload = {"event": res["event"], "direction": res["direction"], "confidence": round(res["confidence"], 4), "stage": res["stage"], "support": res["support"], "resistance": res["resistance"], "baseline": base_cls, "baseline_confidence": round(base_conf, 4), "price": price, "timestamp": str(ts)}
    print(json.dumps(payload, indent=2))
    if open_browser:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--refresh", type=int, default=60)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    model = joblib.load(MODELS_DIR / "event_lgb.joblib")
    out = Path(args.output)
    opened = False
    while True:
        try:
            run_once(model, out, open_browser=(args.open and not opened))
            opened = True
        except Exception as exc:
            print("ERROR:", repr(exc))
        if not args.live:
            break
        now = time.time()
        time.sleep(60 - (now % 60) + 2)

if __name__ == "__main__":
    main()
