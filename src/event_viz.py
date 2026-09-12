"""Step 6J - visualize the training / validation / test datasets.

Loads the event dataset, splits chronologically (60/20/20), and writes an
interactive plotly chart to data/dataset_view.html showing close + support/
resistance bands, shaded train / validation / test regions, split boundary
lines, event markers (breakdown / breakout / false break), and a second panel
with the label distribution per split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import DATA_DIR, MAIN_SYMBOL, RESULTS_DIR
from src.event_engine import EventConfig
from src.event_detector import load_dataset

CLASS_NAMES = {0: "NO_EVENT", 1: "SUPPORT_BREAKDOWN", 2: "RESISTANCE_BREAKOUT", 3: "FALSE_BREAK"}

def main():
    cfg = EventConfig()
    print("loading dataset ...")
    m, feature_cols = load_dataset(cfg)
    y = m["event"].to_numpy(np.int64)
    n = len(m)
    tr_end = int(n * 0.6)
    va_end = int(n * 0.8)
    split = np.full(n, 2, dtype=np.int8)
    split[:tr_end] = 0
    split[tr_end:va_end] = 1
    m["split"] = split
    for name, s in [("train", 0), ("val", 1), ("test", 2)]:
        sub = y[split == s]
        counts = [int((sub == i).sum()) for i in range(4)]
        print(name, "rows=", len(sub), "counts=", counts)
    step = max(1, n // 20000)
    ds = m.iloc[::step].reset_index(drop=True)
    ds_y = y[::step]
    t = ds["time"]
    t0 = m["time"].iloc[0]
    tN = m["time"].iloc[-1]
    b1 = m["time"].iloc[tr_end]
    b2 = m["time"].iloc[va_end]
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=t, y=ds["raw_close"], mode="lines", line=dict(color="#444444", width=1.0), name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=ds["support"], mode="lines", line=dict(color="blue", width=0.8), name="support"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=ds["resistance"], mode="lines", line=dict(color="red", width=0.8), name="resistance"), row=1, col=1)
    dn = ds[ds_y == 1]
    up = ds[ds_y == 2]
    fake = ds[ds_y == 3]
    fig.add_trace(go.Scatter(x=dn["time"], y=dn["raw_close"], mode="markers", marker=dict(symbol="triangle-down", color="red", size=5), name="breakdown"), row=1, col=1)
    fig.add_trace(go.Scatter(x=up["time"], y=up["raw_close"], mode="markers", marker=dict(symbol="triangle-up", color="green", size=5), name="breakout"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fake["time"], y=fake["raw_close"], mode="markers", marker=dict(symbol="circle", color="orange", size=3), name="false break"), row=1, col=1)
    fig.add_vrect(x0=t0, x1=b1, fillcolor="blue", opacity=0.08, line_width=0, row=1, col=1)
    fig.add_vrect(x0=b1, x1=b2, fillcolor="orange", opacity=0.10, line_width=0, row=1, col=1)
    fig.add_vrect(x0=b2, x1=tN, fillcolor="green", opacity=0.08, line_width=0, row=1, col=1)
    fig.add_vline(x=b1, line_dash="dash", line_color="black", line_width=1, row=1, col=1)
    fig.add_vline(x=b2, line_dash="dash", line_color="black", line_width=1, row=1, col=1)
    cats = ["NO_EVENT", "SUPPORT_BREAKDOWN", "RESISTANCE_BREAKOUT", "FALSE_BREAK"]
    colmap = ["#999999", "red", "green", "orange"]
    xs = ["train", "val", "test"]
    for i in range(4):
        vals = [int((y[split == s] == i).sum()) for s in (0, 1, 2)]
        fig.add_trace(go.Bar(name=cats[i], x=xs, y=vals, marker_color=colmap[i]), row=2, col=1)
    fig.update_layout(title=MAIN_SYMBOL + " dataset split (train 60% / val 20% / test 20%)", barmode="group", height=820, template="plotly_white", legend=dict(orientation="h"))
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="label count", row=2, col=1)
    out = DATA_DIR / "dataset_view.html"
    fig.write_html(str(out))
    print("saved ->", out)

if __name__ == "__main__":
    main()
