"""Model-results dashboard (self-contained Plotly HTML).

Runs the trained ensemble over the held-out test set, simulates the portfolio at
several confidence gates, and renders an interactive HTML report:

    * baseline model comparison (val / test accuracy + macro-F1)
    * ensemble confusion matrix (test set)
    * per-gate portfolio equity curves (net of fees + slippage)
    * walk-forward per-window accuracy / F1 / cumulative PnL

Usage::

    python src/dashboard.py               # writes results/dashboard.html
    python src/dashboard.py --open        # ... and opens it in the browser
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import CAPITAL, RESULTS_DIR  # noqa: E402
from src.baselines import CLASS_LABELS, CLASS_NAMES, load_split  # noqa: E402
from src.ensemble import ensemble_predict, load_models  # noqa: E402
from src.backtest import simulate_portfolio  # noqa: E402

# Bybit-style dark palette (matches chart.py)
BG = "#0b0e11"
PANEL = "#0b0e11"
GRID = "#1e222d"
TEXT = "#d9d9d9"
GREEN = "#0ecb81"
RED = "#f6465d"
AMBER = "#f0b90b"
CYAN = "#00c3ff"

GATES = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]


def _load_results() -> dict:
    df_test, X, y, feature_cols = load_split("test")
    models = load_models()
    pred, conf = ensemble_predict(models, X)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=CLASS_LABELS)

    curves: dict[float, np.ndarray] = {}
    sims: dict[float, dict] = {}
    for g in GATES:
        rep = simulate_portfolio(df_test, pred, conf, g)
        curves[g] = rep["equity_curve"]
        sims[g] = rep

    baselines = pd.read_csv(RESULTS_DIR / "baselines.csv")
    walk = pd.read_csv(RESULTS_DIR / "walk_forward.csv")
    backtest = pd.read_csv(RESULTS_DIR / "backtest.csv")

    return {
        "df_test": df_test, "y": y, "pred": pred, "conf": conf,
        "cm": cm, "curves": curves, "sims": sims,
        "baselines": baselines, "walk": walk, "backtest": backtest,
        "feature_cols": feature_cols,
    }


def _layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color="#ffffff"), x=0.02),
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)


def build_dashboard(res: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Baseline models (val / test acc + macro-F1)",
            "Ensemble confusion matrix (test)",
            "Portfolio equity by confidence gate (net of fees)",
            "Walk-forward per window",
        ),
        specs=[[{"type": "xy"}, {"type": "heatmap"}], [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.14, horizontal_spacing=0.12,
    )

    # --- 1) baseline comparison -------------------------------------------
    b = res["baselines"]
    models = list(b["model"])
    fig.add_trace(go.Bar(name="val accuracy", x=models, y=b["accuracy"],
                         marker_color=CYAN), row=1, col=1)
    fig.add_trace(go.Bar(name="test accuracy", x=models, y=b["test_accuracy"],
                         marker_color=AMBER), row=1, col=1)
    fig.add_trace(go.Bar(name="macro-F1 (val)", x=models, y=b["f1"],
                         marker_color=GREEN), row=1, col=1)

    # --- 2) confusion matrix ----------------------------------------------
    cm = res["cm"]
    labels = [CLASS_NAMES[i] for i in CLASS_LABELS]
    fig.add_trace(go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Tealgrn",
        text=cm, texttemplate="%{text}", textfont=dict(size=16),
        colorbar=dict(title="count"),
    ), row=1, col=2)

    # --- 3) equity curves --------------------------------------------------
    for g in GATES:
        c = res["curves"][g]
        s = res["sims"][g]
        fig.add_trace(go.Scatter(
            x=np.arange(len(c)), y=c,
            mode="lines",
            name=f"gate {g:.1f} ({s['n_trades']} trades, {s['total_return']:+.1%})",
        ), row=2, col=1)
    fig.add_hline(y=CAPITAL, line_dash="dot", line_color=GRID, row=2, col=1)

    # --- 4) walk-forward ----------------------------------------------------
    w = res["walk"]
    fig.add_trace(go.Bar(name="accuracy", x=w["window"], y=w["accuracy"],
                         marker_color=CYAN, opacity=0.8), row=2, col=2)
    fig.add_trace(go.Bar(name="macro-F1", x=w["window"], y=w["f1"],
                         marker_color=GREEN, opacity=0.8), row=2, col=2)
    fig.add_trace(go.Scatter(name="cumulative net PnL", x=w["window"],
                             y=w["net_pnl"].cumsum(), mode="lines+markers",
                             line=dict(color=AMBER, width=3)), row=2, col=2)

    _layout(fig, "1minit — model results")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-results dashboard")
    parser.add_argument("--open", action="store_true", help="open the HTML in a browser")
    parser.add_argument("--output", default=str(RESULTS_DIR / "dashboard.html"))
    args = parser.parse_args()

    print("building dashboard ...")
    res = _load_results()
    fig = build_dashboard(res)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True)

    # Console summary
    print(f"\nensemble test accuracy = {(res['pred'] == res['y']).mean():.2%}")
    bt = res["backtest"]
    best = bt.loc[bt["total_return"].idxmax()]
    print(f"best gate = {best['gate']:.1f}  return = {best['total_return']:+.2%}  "
          f"trades = {int(best['n_trades'])}  maxDD = {best['max_drawdown']:.2%}")
    print(f"\ndashboard saved -> {out}")

    if args.open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()

