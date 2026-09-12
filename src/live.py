"""Step 17 / 25 — live inference (REST one-bar prediction).

Reuses the exact training feature pipeline, feeds either the baseline ensemble
(default) or the multi-timeframe transformer (``--transformer``), and applies
the confidence gate to emit LONG / SHORT / NO TRADE for the current minute.

The full production architecture (Phase 25) layers a WebSocket collector
(``microstructure.py``) on top of this; this script remains the fast, REST-only
inference path that works without a persistent connection.

Usage::

    python src/live.py                 # baseline ensemble
    python src/live.py --transformer   # multi-timeframe transformer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bybit_client import BybitClient  # noqa: E402
from src.ensemble import ensemble_predict, load_models  # noqa: E402
from src.features import build_features  # noqa: E402
from src.params import CLASS_NAMES, CONFIDENCE_GATE, SYMBOLS, TARGET_HORIZON  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume", "turnover"]


def fetch_wide(minutes: int = 200) -> pd.DataFrame:
    client = BybitClient()
    frames = {}
    for s in SYMBOLS:
        raw = client.fetch_history(s, minutes=minutes, interval="1")
        raw["time"] = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True)
        frames[s] = raw[["time", *OHLCV]].set_index("time")

    index = None
    for df in frames.values():
        index = df.index if index is None else index.union(df.index)
    index = pd.DatetimeIndex(sorted(index))

    parts = []
    for s in SYMBOLS:
        d = frames[s].reindex(index)
        d.columns = [f"{s}_{c}" for c in OHLCV]
        parts.append(d)
    wide = pd.concat(parts, axis=1)
    wide.insert(0, "time", index)
    return wide


def _align(feats: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Return a (n, F) float32 matrix aligned to ``feature_cols`` (missing -> 0)."""
    df = feats.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols].to_numpy(dtype="float32")


def predict_baseline(feats: pd.DataFrame) -> tuple[str, float]:
    models = load_models()
    cols = [c for c in feats.columns if c != "time" and "_micro_" not in c]
    X = feats[cols].fillna(0.0).to_numpy(dtype="float32")
    pred, conf = ensemble_predict(models, X, feature_names=cols)
    p, c = int(pred[-1]), float(conf[-1])
    return CLASS_NAMES[p], c


def predict_transformer(feats: pd.DataFrame) -> tuple[str, float]:
    from src.ensemble import load_transformer_predictor  # local import
    predictor = load_transformer_predictor()
    X = _align(feats, predictor.feature_cols)
    Xs = predictor.standardize(X)
    probs = predictor.predict_proba(Xs)
    if len(probs) == 0:
        return "NO_MOVE", 0.0
    p = int(probs[-1].argmax())
    c = float(probs[-1].max())
    return CLASS_NAMES[p], c


def main() -> None:
    parser = argparse.ArgumentParser(description="Live inference (6J)")
    parser.add_argument("--transformer", action="store_true",
                        help="use the multi-timeframe transformer instead of the ensemble")
    args = parser.parse_args()

    print("LIVE INFERENCE — fetching latest candles ...")
    # Fetch enough history for feature warm-up (and the transformer context).
    minutes = 800 if args.transformer else 200
    wide = fetch_wide(minutes)
    feats = build_features(wide)

    if args.transformer:
        label, c = predict_transformer(feats)
    else:
        label, c = predict_baseline(feats)

    print(f"latest bar   : {wide['time'].iloc[-1]:%Y-%m-%d %H:%M} UTC")
    print(f"prediction   : {label}  (confidence {c:.2f})")
    print(f"horizon      : {TARGET_HORIZON} minutes  gate={CONFIDENCE_GATE}")

    if c >= CONFIDENCE_GATE and label == "UP":
        action = "LONG  ▲"
    elif c >= CONFIDENCE_GATE and label == "DOWN":
        action = "SHORT ▼"
    else:
        action = "NO TRADE"
    print(f"action       : {action}")


if __name__ == "__main__":
    main()
