# 1minit — Results Summary

Generated on this PC (CPU-only, no GPU) with Python 3.10.

## Data (Phase 1-2)
| Asset    | 1m candles | Range |
|----------|-----------:|-------|
| DOGEUSDT | 1,051,200  | 2024-09-11 → 2026-09-11 (2 years) |
| BTCUSDT  | 1,051,200  | 2024-09-11 → 2026-09-11 (2 years) |
| XAUTUSDT |   745,620  | 2025-04-11 → 2026-09-11 (~17 months, listed later) |

Synchronized (all 3 assets present): **745,602** UTC minutes.

## Pipeline (all runnable)
```
src/download_bybit.py   Phase 1-2  bulk history
src/clean_data.py       Phase 3    cleaning + gap detection (no interpolation)
src/synchronize.py      Phase 4    shared UTC minute index
src/features.py         Phase 5-8  53 features (technical + cross-asset + MTF)
src/targets.py          Phase 9    volatility-normalized UP/DOWN/NO-MOVE (5m, 0.3 ATR)
src/dataset.py          Phase 11   60/20/20 chronological split
src/baselines.py        Phase 12   XGBoost / LightGBM / MLP
src/ensemble.py         Phase 21   weighted probability ensemble
src/backtest.py         Phase 15/23  fees + slippage + confidence gate
src/evolution.py        Phase 13-18 CPU evolution (LightGBM hyperparams)
src/live.py             Phase 17/24 live inference
src/pipeline.py         orchestrator
```

## Baselines (3-class: UP / DOWN / NO-MOVE)
| Model    | val acc | test acc | macro-F1 |
|----------|--------:|---------:|---------:|
| XGBoost  | 44.79%  | 45.55%   | 0.322    |
| LightGBM | 44.93%  | 45.63%   | 0.323    |
| MLP      | 45.16%  | 45.78%   | 0.324    |

Majority-class baseline ≈ 42.7%.  The models are only ~3 points above chance.

## Backtest (ensemble, 5m horizon, 0.24% cost/trade)
| confidence gate | trades | total return | win rate | profit factor |
|----------------:|-------:|-------------:|---------:|--------------:|
| 0.0             | 29,742 | -100.0%      | 5.98%    | 0.04          |
| 0.5             |    759 | -81.9%       | 9.49%    | 0.09          |
| 0.6             |      3 | -0.9%        | 33.3%    | 0.56          |
| 0.7             |      2 | +0.8%        | 50.0%    | 1.81          |
| 0.8             |      0 | 0.0%         | —        | —             |

## Honest conclusion
1-minute candle features (even with cross-asset context) do **not** produce a
tradable signal for 5-minute DOGE direction at this threshold.  Net of fees the
strategy loses money, and the confidence gate suppresses almost all trades.

## Recommended next steps (in priority order)
1. **Microstructure features** (best bid/ask, order-book imbalance, trade flow)
   — requires a Bybit WebSocket collector; the plan identifies this as the
   highest-value addition for small movements.
2. **Target tuning** — try 2-class (UP vs DOWN), longer horizons (15m/30m),
   and a volatility-aware threshold sweep on validation.
3. **More compute** — the transformer + 100-model evolution phases need a GPU
   (this PC is CPU-only; MLP alone took ~23 min).
4. **Regime features** — funding, open interest, BTC dominance.
