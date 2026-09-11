"""Central configuration for the 1minit evolutionary trading project.

Step 1 only needs: symbols, interval, EMA spans and output directories.
Later phases will extend this file (account, DNA, evolution params, DB, etc.).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
BYBIT_BASE_URL = "https://api.bybit.com"          # mainnet = live market data
BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"

CATEGORY = "spot"                                  # "spot" | "linear" | "inverse"
INTERVAL = "1"                                     # 1-minute candles

# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
PRIMARY_SYMBOLS = ["DOGEUSDT"]
SUPPORT_SYMBOLS = ["BTCUSDT", "XAUTUSDT"]
DEFAULT_SYMBOLS = PRIMARY_SYMBOLS + SUPPORT_SYMBOLS

# ---------------------------------------------------------------------------
# Feature parameters (Step 1)
# ---------------------------------------------------------------------------
EMA_SPANS = (5, 10, 20, 50, 100)

# Signal / analysis parameters (used by signals.py)
SUPPORT_WINDOW = 50       # rolling candles for dynamic support/resistance
LIQUIDITY_WINDOW = 20     # rolling candles for the liquidity moving average
SIGNAL_FAST = 5           # fast EMA for cross signals
SIGNAL_SLOW = 20          # slow EMA for cross signals

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # raw OHLCV + turnover (no derived fields)
FEATURES_DIR = DATA_DIR / "features"  # the 12-field Step 1 dataset
ANALYSIS_DIR = DATA_DIR / "analysis"  # full enriched dataset + liquidity/support/signals

# ---------------------------------------------------------------------------
# How many 1-minute candles to fetch by default.
# The AI input window is 1000 minutes; fetching a little more lets EMA100
# warm up with a full 100-candle window before minute 1000.
# ---------------------------------------------------------------------------
DEFAULT_MINUTES = 1500
