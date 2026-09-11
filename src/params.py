"""ML pipeline configuration and shared paths.

Single source of truth for the data-processing / modelling stages.
(Named ``params`` to avoid clashing with the root ``config.py`` used by the
live Bybit client.)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # bulk per-symbol 1m history
CLEANED_DIR = DATA_DIR / "cleaned"
SYNC_DIR = DATA_DIR / "synchronized"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

SYMBOLS = ["DOGEUSDT", "BTCUSDT", "XAUTUSDT"]
MAIN_SYMBOL = "DOGEUSDT"                  # the asset we predict

# ---------------------------------------------------------------------------
# Target definition (Phase 9)
# ---------------------------------------------------------------------------
TARGET_HORIZON = 5                         # predict 5-minute-ahead movement
TARGET_THRESHOLD_ATR = 0.3                 # normalized move threshold (UP/DOWN/NO-MOVE)
TARGET_RETURNS = (1, 2, 3, 5, 10)          # regression heads: future returns

# ---------------------------------------------------------------------------
# Chronological split (Phase 11)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
NUM_CLASSES = 3                            # UP / DOWN / NO-MOVE

# Backtest / simulation (Phase 23)
FEE_RATE = 0.001        # 0.1% per side
SLIPPAGE = 0.0002       # 0.02% slippage
CONFIDENCE_GATE = 0.6   # minimum predicted probability to act


def ensure_dirs() -> None:
    for d in (RAW_DIR, CLEANED_DIR, SYNC_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
