# 1minit — AI Evolutionary Crypto Trading (Step 1)

Step 1 only: collect **live Bybit 1-minute candles** and calculate the **12 core
fields**.

## The 12 fields

| # | Field           | Meaning                              |
|---|-----------------|--------------------------------------|
| 1 | `timestamp_utc` | candle start time (UTC)              |
| 2 | `open`          | open price                           |
| 3 | `high`          | high price                           |
| 4 | `low`           | low price                            |
| 5 | `close`         | close price                          |
| 6 | `volume`        | base-coin volume (e.g. DOGE)         |
| 7 | `turnover`      | quote-coin turnover (USDT)           |
| 8 | `EMA5`          | EMA(5) of close                      |
| 9 | `EMA10`         | EMA(10) of close                     |
| 10 | `EMA20`         | EMA(20) of close                     |
| 11 | `EMA50`         | EMA(50) of close                     |
| 12 | `EMA100`        | EMA(100) of close                    |

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
# default: DOGEUSDT BTCUSDT XAUTUSDT, last 1500 minutes
python collect_step1.py

# custom
python collect_step1.py --minutes 1500 --symbols DOGEUSDT BTCUSDT XAUTUSDT
```

## Output

- `data/raw/<SYMBOL>_1m_raw.csv`       — raw candles (incl. `timestamp_ms`)
- `data/features/<SYMBOL>_1m_features.csv` — the 12-field dataset

Data source: Bybit V5 public endpoint `GET /v5/market/kline` (no API key).
Category defaults to `spot` (so `XAUTUSDT`, which is spot-only, is covered).
