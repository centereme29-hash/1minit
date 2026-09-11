"""Step 3 — clean raw 1m candles per asset (no blind interpolation).

For each symbol: de-duplicate timestamps, drop non-positive / invalid prices,
flag OHLC inconsistencies, and detect missing-minute gaps.  We deliberately do
**not** fill gaps — missing candles are either market downtime or a collection
failure, which the synchronizer/dataset decides how to treat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import CLEANED_DIR, RAW_DIR, SYMBOLS  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume", "turnover"]


def load_raw(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / symbol / f"{symbol}_1m.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing raw file: {path}")
    return pd.read_csv(path)


def clean(symbol: str) -> tuple[pd.DataFrame, dict]:
    df = load_raw(symbol)
    n_in = len(df)

    out = df.copy()
    for c in OHLCV:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("timestamp_ms").drop_duplicates(subset="timestamp_ms", keep="last")
    n_dupes = n_in - len(out)

    # Drop invalid prices (non-positive OHLC) and negative volume/turnover.
    valid_price = (out[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_vol = (out[["volume", "turnover"]] >= 0).all(axis=1)
    n_bad_price = int((~valid_price).sum())
    n_bad_vol = int((~valid_vol).sum())
    out = out[valid_price & valid_vol]

    # OHLC consistency checks.
    bad_high = out["high"] < out[["open", "close"]].max(axis=1)
    bad_low = out["low"] > out[["open", "close"]].min(axis=1)
    n_bad_ohlc = int((bad_high | bad_low).sum())
    out = out[~(bad_high | bad_low)]

    # Timestamp column (UTC) + gap detection.
    out["time"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True)
    out = out.sort_values("time").reset_index(drop=True)

    gap = out["time"].diff()
    missing = gap[gap > pd.Timedelta(minutes=1)]
    n_gaps = int((gap > pd.Timedelta(minutes=1)).sum())
    missing_minutes = int((missing - pd.Timedelta(minutes=1)).dt.total_seconds().sum() // 60)

    report = {
        "rows_in": n_in,
        "rows_out": len(out),
        "duplicates_dropped": n_dupes,
        "bad_price_dropped": n_bad_price,
        "bad_volume_dropped": n_bad_vol,
        "bad_ohlc_dropped": n_bad_ohlc,
        "gaps_detected": n_gaps,
        "missing_minutes": missing_minutes,
    }
    return out, report


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    print("STEP 3 — clean raw candles (no interpolation)")
    for symbol in SYMBOLS:
        out, rep = clean(symbol)
        out_path = CLEANED_DIR / f"{symbol}_1m.csv"
        out.to_csv(out_path, index=False)
        print(f"\n[{symbol}] {rep['rows_in']} → {rep['rows_out']} rows")
        print(f"    dupes={rep['duplicates_dropped']}  bad_price={rep['bad_price_dropped']}  "
              f"bad_volume={rep['bad_volume_dropped']}  bad_ohlc={rep['bad_ohlc_dropped']}")
        print(f"    gaps={rep['gaps_detected']}  missing_minutes={rep['missing_minutes']}")
        print(f"    saved -> {out_path}")

    print("\nSTEP 3 DONE.")


if __name__ == "__main__":
    main()
