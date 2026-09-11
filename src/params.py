"""ML pipeline configuration and shared paths.

Single source of truth for the data-processing / modelling stages.
(Named ``params`` to avoid clashing with the root ``config.py`` used by the
live Bybit client.)

If a ``config.yaml`` exists at the project root, its values override the
defaults below, so the YAML file is the canonical place to tune the GPU-side
experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the Unicode arrows
# (→, ▲, ▼) some scripts print.  Reconfigure stdout/stderr to UTF-8 so the
# pipeline runs out-of-the-box on a Windows terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # bulk per-symbol 1m history
CLEANED_DIR = DATA_DIR / "cleaned"
SYNC_DIR = DATA_DIR / "synchronized"
FEATURES_DIR = DATA_DIR / "features"
MICRO_DIR = DATA_DIR / "microstructure"   # WebSocket-derived microstructure bars
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
# Chronological split (Phase 11 / 19)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2                            # Validation-A (evolution score target)
VAL_B_RATIO = 0.1                          # Validation-B (overfit re-check)
TEST_RATIO = 0.1                           # Final held-out test (never touched)

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
NUM_CLASSES = 3                            # UP / DOWN / NO-MOVE

# Backtest / simulation (Phase 23)
FEE_RATE = 0.001        # 0.1% per side
SLIPPAGE = 0.0002       # 0.02% slippage
CONFIDENCE_GATE = 0.7   # minimum predicted probability to act
CAPITAL = 1000.0        # default paper/backtest starting capital (USD)

# Microstructure columns produced per symbol per minute (Phase 6A).
MICRO_FIELDS = [
    "best_bid", "best_ask", "spread", "rel_spread", "mid_price",
    "ob_imbalance", "buy_volume", "sell_volume", "trade_count",
    "large_trade_count", "vwap", "trade_imbalance",
]


def load_config() -> dict:
    """Load ``config.yaml`` (if present) into a nested dict, else ``{}``."""
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # local import keeps this optional
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_config() -> None:
    """Overwrite module-level defaults from config.yaml."""
    cfg = load_config()
    if not cfg:
        return

    data = cfg.get("data") or {}
    tgt = cfg.get("target") or {}
    split = cfg.get("split") or {}
    trading = cfg.get("trading") or {}

    global SYMBOLS, MAIN_SYMBOL
    if data.get("symbols"):
        SYMBOLS = [str(s).upper() for s in data["symbols"]]
        MAIN_SYMBOL = SYMBOLS[0]

    global TARGET_HORIZON, TARGET_THRESHOLD_ATR, TARGET_RETURNS, NUM_CLASSES
    if tgt.get("horizon") is not None:
        TARGET_HORIZON = int(tgt["horizon"])
    if tgt.get("threshold_atr") is not None:
        TARGET_THRESHOLD_ATR = float(tgt["threshold_atr"])
    if tgt.get("classes") is not None:
        NUM_CLASSES = int(tgt["classes"])
    if tgt.get("returns"):
        TARGET_RETURNS = tuple(int(r) for r in tgt["returns"])

    global TRAIN_RATIO, VAL_RATIO, VAL_B_RATIO, TEST_RATIO
    if split.get("train") is not None:
        TRAIN_RATIO = float(split["train"])
    if split.get("val_a") is not None:
        VAL_RATIO = float(split["val_a"])
    if split.get("val_b") is not None:
        VAL_B_RATIO = float(split["val_b"])
    if split.get("test") is not None:
        TEST_RATIO = float(split["test"])

    global FEE_RATE, SLIPPAGE, CONFIDENCE_GATE, CAPITAL
    if trading.get("fee") is not None:
        FEE_RATE = float(trading["fee"])
    if trading.get("slippage") is not None:
        SLIPPAGE = float(trading["slippage"])
    if trading.get("confidence_gate") is not None:
        CONFIDENCE_GATE = float(trading["confidence_gate"])
    if trading.get("capital") is not None:
        CAPITAL = float(trading["capital"])


_apply_config()

# Class labels depend on the (possibly config-overridden) NUM_CLASSES.
if NUM_CLASSES == 2:
    CLASS_NAMES = {0: "DOWN", 1: "UP"}
    CLASS_LABELS = [0, 1]
else:
    CLASS_NAMES = {0: "DOWN", 1: "NO_MOVE", 2: "UP"}
    CLASS_LABELS = [0, 1, 2]


def ensure_dirs() -> None:
    for d in (RAW_DIR, CLEANED_DIR, SYNC_DIR, FEATURES_DIR, MICRO_DIR,
              MODELS_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
