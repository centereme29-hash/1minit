# 1minit — Results Summary (2-class UP/DOWN)

Regenerated 2026-09-12 on this PC (CPU training; CUDA available but CPU used).

## Data
| Asset    | 1m candles | Range |
|----------|-----------:|-------|
| DOGEUSDT | 1,051,200  | 2024-09-11 → 2026-09-11 (2 years) |
| BTCUSDT  | 1,051,200  | 2024-09-11 → 2026-09-11 (2 years) |
| XAUTUSDT |   745,620  | 2025-04-11 → 2026-09-11 (listed later) |

Synchronized **union** index: **1,051,573** UTC minutes. After NaN-drop, the
modelling matrix is the **745,602** minutes where all three assets are present.

## Pipeline
```
src/clean_data.py      cleaning + gap detection (no interpolation)
src/synchronize.py     shared UTC minute index (wide union table)
src/features.py        81 features (technical + cross-asset + trend/regime)
src/targets.py         2-class UP/DOWN (sign of 5m future return) + regression heads
src/dataset.py         60/20/10/10 chronological split
src/baselines.py       XGBoost / LightGBM / MLP
src/ensemble.py        weighted probability ensemble
src/backtest.py        trading simulator (fees + slippage + confidence gate)
src/walk_forward.py    expanding-window walk-forward validation
```

Features grew from 53 → **81** with 28 trend/regime additions: BTC/XAUT 15/30/60m
returns, BTC EMA position (20/50/200) + cross + slope, BTC & DOGE support-break
distance from rolling 30/60-bar lows, and DOGE-BTC relative strength at 15/30/60m.

## Baselines (2-class: DOWN / UP)
| Model    | val acc | test acc | macro-F1 | time  |
|----------|--------:|---------:|---------:|-------|
| XGBoost  | 52.54%  | 52.83%   | 0.510    | 20.1s |
| LightGBM | 52.66%  | 52.59%   | 0.511    | 21.8s |
| MLP      | 53.10%  | 53.03%   | 0.494    | 581s  |

Majority class (DOWN) ≈ 50.8%. The models are only ~1–2 points above chance.

## Backtest (ensemble, 5m horizon, 0.24% cost/trade)
| confidence gate | trades | total return | win rate | profit factor |
|----------------:|-------:|-------------:|---------:|--------------:|
| 0.0             | 14,900 | -100.0%      | 5.9%     | 0.04          |
| 0.5             | 14,900 | -100.0%      | 5.9%     | 0.04          |
| 0.6             |     23 | -3.9%        | 34.8%    | 0.63          |
| 0.7             |      0 | 0.0%         | —        | —             |
| 0.8             |      0 | 0.0%         | —        | —             |

## Walk-forward (LightGBM, 8 expanding windows)
Mean accuracy **51.1%** (std 0.5%), **0 of 8** windows net-positive, total net PnL
**-315.9**. Net of fees the strategy loses in every window — regime-stable noise.

## Honest conclusion
Even with multi-timeframe trend/regime features, 1-minute candle features do not
produce a tradable 5-minute DOGE direction signal: ~52.5–53% accuracy is a coin
flip against a ~51% majority, the confidence gate suppresses almost all trades,
and every walk-forward window is net-negative after fees.

The leading signal that remains is **microstructure** (order book + trade flow).
Collect it for several days, then retrain:

```
python src/microstructure.py --symbols DOGEUSDT BTCUSDT XAUTUSDT --depth 50
```
