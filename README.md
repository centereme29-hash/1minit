# 1minit — AI Evolutionary Crypto Trading

Research pipeline that turns Bybit **1-minute candles + microstructure** into a
multi-timeframe transformer, evolved on a GPU, with the goal of a tradable
short-horizon signal. The honest finding so far (see `results/SUMMARY.md`):
candle-only features are **not** net-of-fee tradable — the edge is expected to
come from order-book / trade-flow microstructure (`src/microstructure.py`).

The repo has two layers:

1. **Step-1 collector** (repo root) — the original 12-field candle collector.
2. **ML pipeline** (`src/`) — the full research stack described in `next_work.txt`.

---

## 1. Environment setup

```powershell
python -m venv crypto_ai
crypto_ai\Scripts\activate          # or: source crypto_ai/bin/activate (macOS/Linux)

# PyTorch with CUDA (pick the wheel for your CUDA: cu118 / cu121 / cu124)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Verify GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
# -> True
```

---

## 2. Data download & core pipeline

```powershell
# full 2-year 1m history for the three symbols (data/raw/<SYMBOL>/<SYMBOL>_1m.csv)
python src/download_bybit.py --symbols DOGEUSDT BTCUSDT XAUTUSDT --days 730

# clean -> synchronize -> features -> targets -> dataset -> baselines -> backtest
python src/pipeline.py
```

| Stage | Script | What it does |
|-------|--------|--------------|
| download | `src/download_bybit.py` | bulk 1m history (resumable, backward pagination) |
| clean   | `src/clean_data.py` | dedupe + OHLC sanity + gap detection (no interpolation) |
| sync    | `src/synchronize.py` | shared UTC minute index (wide table) |
| features| `src/features.py` | 81 candle/cross-asset/multi-timeframe/trend features + micro merge |
| targets | `src/targets.py` | volatility-normalized UP/DOWN/NO-MOVE + regression heads |
| dataset | `src/dataset.py` | chronological train / val-a / val-b / test split |
| baselines| `src/baselines.py` | XGBoost / LightGBM / MLP |
| backtest| `src/backtest.py` | trading simulator (fees, slippage, capital, sizing) |

All scripts run from the project root, e.g. `python src/features.py`.

---

## 3. Remaining modules (the GPU work)

| Item | Script | Purpose |
|------|--------|---------|
| 6A   | `src/microstructure.py` | Bybit V5 WebSocket collector → `data/microstructure/<SYM>_1m_micro.csv` |
| 6B   | `src/walk_forward.py` | expanding-window walk-forward validation |
| 6D/6E| `src/model.py` + `src/train.py` | multi-timeframe transformer + multi-head targets |
| 6F   | `src/evolution_gpu.py` | GPU evolution engine (mutation/crossover/diversity) |
| 6I   | `src/paper.py` | paper trading (simulated PnL → `logs/paper.csv`) |
| 6J   | `src/live.py` | live inference (ensemble or `--transformer`) |

```powershell
# Microstructure (run for several days BEFORE retraining)
python src/microstructure.py --symbols DOGEUSDT BTCUSDT XAUTUSDT --depth 50
python src/microstructure.py --backfill --minutes 30 --poll-seconds 2   # quick bootstrap

# Walk-forward
python src/walk_forward.py --model lightgbm --n-windows 8

# Transformer
python src/train.py --epochs 5 --d-model 256 --layers 4

# Evolution (start small)
python src/evolution_gpu.py --population 12 --generations 3 --epochs 2

# Live / paper
python src/live.py --transformer
python src/paper.py --once
```

---

## 4. Configuration

`config.yaml` at the project root is the canonical configuration. Its
`data` / `target` / `split` / `trading` sections are read by `src/params.py`;
the `model` section is read by `src/train.py` (CLI flags override it).

**Pitfalls:** never touch the final test set during evolution; treat >95%
accuracy as a leakage red flag; do not blindly interpolate missing candles;
score on net-of-fee PnL, not raw accuracy.

---

## 5. Step-1 collector (legacy)

The original 12-field collector still works:

```powershell
python collect_step1.py --minutes 1500 --symbols DOGEUSDT BTCUSDT XAUTUSDT
```

Outputs `data/raw/<SYMBOL>_1m_raw.csv` and `data/features/<SYMBOL>_1m_features.csv`.

Data source: Bybit V5 public endpoint `GET /v5/market/kline` (no API key),
category `spot`.
