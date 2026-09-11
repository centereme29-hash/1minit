"""Step 17 — live inference: fetch latest candles, build features, predict.

Reuses the same feature pipeline as training, feeds the ensemble, and applies
the confidence gate to emit LONG / SHORT / NO TRADE for the current minute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bybit_client import BybitClient  # noqa: E402
from src.ensemble import ensemble_predict, load_models  # noqa: E402
from src.features import build_features  # noqa: E402
from src.params import CONFIDENCE_GATE, MAIN_SYMBOL, SYMBOLS, TARGET_HORIZON  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume", "turnover"]
CLASS_NAMES = {0: "DOWN", 1: "NO_MOVE", 2: "UP"}


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


def main() -> None:
    print("LIVE INFERENCE — fetching latest candles ...")
    wide = fetch_wide(200)
    feats = build_features(wide)
    row = feats.drop(columns=["time"]).iloc[[-1]].fillna(0.0)

    models = load_models()
    pred, conf = ensemble_predict(models, row.to_numpy(dtype="float32"))
    p, c = int(pred[0]), float(conf[0])

    print(f"latest bar   : {wide['time'].iloc[-1]:%Y-%m-%d %H:%M} UTC")
    print(f"prediction   : {CLASS_NAMES[p]}  (confidence {c:.2f})")
    print(f"horizon      : {TARGET_HORIZON} minutes  gate={CONFIDENCE_GATE}")

    if c >= CONFIDENCE_GATE and p != 1:
        action = "LONG  ▲" if p == 2 else "SHORT ▼"
        print(f"action       : {action}")
    else:
        print("action       : NO TRADE")


if __name__ == "__main__":
    main()
