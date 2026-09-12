"""Step 6H/6I - early-detection metrics + event backtester (M2 + M3).

M2: per-stage actionable precision, lead-time stats and a plotly chart of the
test segment (support/resistance bands + actual and predicted breakout markers).

M3: event-based backtester with take-profit / stop-loss / max-holding-period,
fees + slippage, and a confidence-gate sweep (profit factor, expectancy, max DD).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import DATA_DIR, FEE_RATE, MAIN_SYMBOL, RESULTS_DIR, SLIPPAGE
from src.event_engine import EventConfig
from src.event_detector import load_dataset, _train_lgb

TP = 0.003
SL = 0.002
HOLD = 30
COST = 2.0 * (FEE_RATE + SLIPPAGE)

def early_detection(m_te, yte, prob):
    stage = m_te["stage"].to_numpy()
    pred = prob.argmax(1)
    t2b = m_te["time_to_break"].to_numpy().astype(float)
    rows = []
    for s in range(5):
        sel = stage == s
        act = sel & np.logical_or(pred == 1, pred == 2)
        n_act = int(act.sum())
        if n_act:
            correct = yte[act] == pred[act]
            prec = float(correct.mean())
            lead = float(np.nanmean(t2b[act][correct]))
        else:
            prec = float("nan")
            lead = float("nan")
        rows.append({"stage": s, "n_bars": int(sel.sum()), "n_actionable": n_act, "precision": prec, "mean_lead_bars": lead})
    act_all = np.logical_or(pred == 1, pred == 2)
    correct_all = act_all & (yte == pred)
    lead_all = float(np.nanmean(t2b[correct_all])) if correct_all.any() else float("nan")
    return pd.DataFrame(rows), lead_all

def backtest(m_te, prob, gates):
    close = m_te["raw_close"].to_numpy(np.float64)
    high = m_te["raw_high"].to_numpy(np.float64)
    low = m_te["raw_low"].to_numpy(np.float64)
    p_up = prob[:, 2]
    p_dn = prob[:, 1]
    n = len(m_te)
    rows = []
    for gate in gates:
        trades = []
        i = 0
        while i < n - HOLD:
            side = 0
            if p_up[i] >= gate and p_up[i] > p_dn[i]:
                side = 1
            elif p_dn[i] >= gate and p_dn[i] > p_up[i]:
                side = -1
            if side == 0:
                i += 1
                continue
            entry = close[i]
            exit_ret = 0.0
            j = i
            for k in range(1, HOLD + 1):
                jj = i + k
                hh = high[jj] / entry - 1.0
                ll = low[jj] / entry - 1.0
                if side == 1:
                    if hh >= TP:
                        exit_ret = TP
                        j = jj
                        break
                    if ll <= -SL:
                        exit_ret = -SL
                        j = jj
                        break
                else:
                    if ll <= -TP:
                        exit_ret = TP
                        j = jj
                        break
                    if hh >= SL:
                        exit_ret = -SL
                        j = jj
                        break
            if j == i:
                j = i + HOLD
                r = close[j] / entry - 1.0
                exit_ret = r if side == 1 else -r
            net = exit_ret - COST
            trades.append(net)
            i = j + 1
        if trades:
            rets = np.array(trades)
            wins = rets[rets > 0].sum()
            losses = -rets[rets < 0].sum()
            eq = np.cumprod(1.0 + rets)
            dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
            rows.append({"gate": gate, "n_trades": len(trades), "win_rate": float((rets > 0).mean()), "total_return": float(eq[-1] - 1.0), "profit_factor": float(wins / losses) if losses > 0 else float("inf"), "expectancy": float(rets.mean()), "max_drawdown": dd})
        else:
            rows.append({"gate": gate, "n_trades": 0, "win_rate": 0.0, "total_return": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "max_drawdown": 0.0})
    return pd.DataFrame(rows)

def make_chart(m_te, yte, prob, gate):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        tail = m_te.tail(1500).reset_index(drop=True)
        yt = yte[-1500:]
        pt = prob[-1500:]
        t = tail["time"]
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=t, open=tail["raw_open"], high=tail["raw_high"], low=tail["raw_low"], close=tail["raw_close"], name=MAIN_SYMBOL, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
        fig.add_trace(go.Scatter(x=t, y=tail["support"], mode="lines", line=dict(color="blue", width=1.3), name="support"))
        fig.add_trace(go.Scatter(x=t, y=tail["resistance"], mode="lines", line=dict(color="red", width=1.3), name="resistance"))
        dn = tail[yt == 1]
        up = tail[yt == 2]
        fake = tail[yt == 3]
        fig.add_trace(go.Scatter(x=dn["time"], y=dn["raw_low"], mode="markers", marker=dict(symbol="triangle-down", color="red", size=9), name="actual breakdown"))
        fig.add_trace(go.Scatter(x=up["time"], y=up["raw_high"], mode="markers", marker=dict(symbol="triangle-up", color="green", size=9), name="actual breakout"))
        fig.add_trace(go.Scatter(x=fake["time"], y=fake["raw_close"], mode="markers", marker=dict(symbol="circle", color="yellow", size=6), name="false break"))
        pred = pt.argmax(1)
        conf = pt.max(1)
        pl = tail[(pred == 2) & (conf >= gate)]
        ps = tail[(pred == 1) & (conf >= gate)]
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["raw_close"], mode="markers", marker=dict(symbol="arrow-up", color="lime", size=11), name="predicted LONG"))
        fig.add_trace(go.Scatter(x=ps["time"], y=ps["raw_close"], mode="markers", marker=dict(symbol="arrow-down", color="magenta", size=11), name="predicted SHORT"))
        fig.update_layout(title="DOGEUSDT breakout signals (test tail, gate={})".format(gate), xaxis_rangeslider_visible=False, height=680, template="plotly_dark")
        out = DATA_DIR / "event_signals.html"
        fig.write_html(str(out))
        print("chart ->", out)
    except Exception as e:
        print("chart skipped:", repr(e))

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = EventConfig()
    print("STEP 6H/6I - early detection + backtest")
    m, feature_cols = load_dataset(cfg)
    y = m["event"].to_numpy(dtype=np.int64)
    n = len(m)
    tr_end = int(n * 0.6)
    va_end = int(n * 0.8)
    Xtr = m[feature_cols].iloc[:tr_end].to_numpy(dtype=np.float32)
    Xva = m[feature_cols].iloc[tr_end:va_end].to_numpy(dtype=np.float32)
    Xte = m[feature_cols].iloc[va_end:].to_numpy(dtype=np.float32)
    ytr = y[:tr_end]
    yva = y[tr_end:va_end]
    yte = y[va_end:]
    m_te = m.iloc[va_end:].reset_index(drop=True)
    print("training lightgbm ...")
    mdl = _train_lgb(Xtr, ytr, Xva, yva)
    prob = mdl.predict_proba(Xte)
    ed, lead_all = early_detection(m_te, yte, prob)
    ed.to_csv(RESULTS_DIR / "event_early_detection.csv", index=False)
    print(ed.to_string(index=False))
    print("overall lead (bars) for correct signals:", round(lead_all, 2))
    bt = backtest(m_te, prob, [0.5, 0.6, 0.7, 0.8, 0.9])
    bt.to_csv(RESULTS_DIR / "event_backtest.csv", index=False)
    print(bt.to_string(index=False))
    make_chart(m_te, yte, prob, 0.7)
    summary = {"early_lead_bars": lead_all, "backtest": bt.to_dict(orient="records")}
    with open(RESULTS_DIR / "event_m23_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("DONE")

if __name__ == "__main__":
    main()
