"""Step 4 — synchronize DOGE / BTC / XAUT onto one shared UTC minute index.

Produces a wide table where every row is one UTC minute and each asset
contributes its OHLCV columns (``<SYMBOL>_open``, ``<SYMBOL>_close``, ...).
Rows where an asset had no candle are left as NaN (and counted), so downstream
feature code can require all three assets to be present before building
cross-asset features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import CLEANED_DIR, SYNC_DIR, SYMBOLS  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume", "turnover"]


def load_cleaned(symbol: str) -> pd.DataFrame:
    path = CLEANED_DIR / f"{symbol}_1m.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates(subset="time", keep="last").sort_values("time")
    return df[["time", *OHLCV]].set_index("time")


def synchronize() -> pd.DataFrame:
    frames = {s: load_cleaned(s) for s in SYMBOLS}

    # Shared UTC minute index = union of all three assets' timestamps.
    index = None
    for df in frames.values():
        index = df.index if index is None else index.union(df.index)
    index = pd.DatetimeIndex(sorted(index))

    columns: list[str] = []
    parts: list[pd.DataFrame] = []
    for symbol in SYMBOLS:
        cols = [f"{symbol}_{c}" for c in OHLCV]
        aligned = frames[symbol].reindex(index)
        aligned.columns = cols
        columns += cols
        parts.append(aligned)

    wide = pd.concat(parts, axis=1)
    wide.insert(0, "time", index)

    # Completeness: how many of the three assets are present at each minute.
    present = pd.DataFrame(
        {s: (~frames[s].reindex(index)["close"].isna()).astype(int) for s in SYMBOLS}
    )
    wide["n_assets"] = present.sum(axis=1)
    wide["all_present"] = (wide["n_assets"] == len(SYMBOLS))

    return wide


def main() -> None:
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 4 — synchronize DOGE / BTC / XAUT onto one UTC minute index")

    wide = synchronize()
    out_path = SYNC_DIR / "all_1m.csv"
    wide.to_csv(out_path, index=False)

    n = len(wide)
    n_complete = int(wide["all_present"].sum())
    print(f"\nunion minutes     = {n}")
    print(f"complete minutes  = {n_complete}  ({n_complete / n:.1%})")
    for symbol in SYMBOLS:
        missing = int((wide[f"{symbol}_close"].isna()).sum())
        print(f"  {symbol}: {n - missing} present, {missing} missing")
    print(f"\nsaved -> {out_path}")
    print("STEP 4 DONE.")


if __name__ == "__main__":
    main()
