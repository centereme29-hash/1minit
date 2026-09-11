"""Step 1 — Bybit 1-minute collector + the 12 core fields.

For each requested symbol this script:

  1. Fetches the most recent N minutes of 1-minute candles (OHLCV + turnover)
     from Bybit's public ``/v5/market/kline`` endpoint.
  2. Calculates EMA5, EMA10, EMA20, EMA50, EMA100 on the close.
  3. Saves a raw CSV and a 12-field features CSV under ``data/``.
  4. Runs lightweight validation and prints a preview.

Usage::

    python collect_step1.py
    python collect_step1.py --minutes 1500 --symbols DOGEUSDT BTCUSDT XAUTUSDT
"""
from __future__ import annotations

import argparse

from bybit_client import BybitClient
from config import DATA_DIR, DEFAULT_MINUTES, DEFAULT_SYMBOLS, FEATURES_DIR, RAW_DIR
from features import REQUIRED_FIELDS, build_step1_dataset
from store import append_csv
from validate import validate_raw


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, FEATURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 1: Bybit 1m collector + 12 fields")
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES,
                        help="number of 1-minute candles to fetch")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="space-separated symbols, e.g. DOGEUSDT BTCUSDT XAUTUSDT")
    parser.add_argument("--category", default=None,
                        help="Bybit category (spot|linear|inverse); default from config.py")
    args = parser.parse_args(argv)

    ensure_dirs()
    client = BybitClient() if args.category is None else BybitClient(category=args.category)

    print("STEP 1 - Bybit live 1-minute data + 12 core fields")
    print(f"category={client.category}  interval=1m  minutes={args.minutes}")
    print(f"symbols={args.symbols}")

    for symbol in args.symbols:
        sym = symbol.upper()
        print(f"\n{'=' * 72}\n[{sym}] fetching {args.minutes} x 1m candles ...")

        raw = client.fetch_history(sym, minutes=args.minutes, interval="1")
        if raw.empty:
            print(f"[{sym}] WARNING: no data returned (check symbol / category).")
            continue

        ds = build_step1_dataset(raw)

        missing = [c for c in REQUIRED_FIELDS if c not in ds.columns]
        if missing:
            print(f"[{sym}] ERROR: missing fields {missing}")
            continue

        raw_path = RAW_DIR / f"{sym}_1m_raw.csv"
        feat_path = FEATURES_DIR / f"{sym}_1m_features.csv"
        append_csv(raw_path, raw, key="timestamp_ms")
        append_csv(feat_path, ds, key="timestamp_utc")

        report = validate_raw(raw)
        print(f"[{sym}] rows={report['rows']}  valid={report['ok']}")
        if report.get("first") is not None:
            print(f"    first={report['first']}  last={report['last']}")
        if report["issues"]:
            for issue in report["issues"]:
                print(f"    ! {issue}")

        print(f"    raw      -> {raw_path}")
        print(f"    features -> {feat_path}")

        print(f"\n    --- first 3 rows ({sym}) ---")
        print(ds.head(3).to_string(index=False))
        print(f"    --- last 3 rows ({sym}) ---")
        print(ds.tail(3).to_string(index=False))

    print("\n" + "=" * 72)
    print("STEP 1 DONE - 12 fields per symbol:")
    print("  timestamp_utc, open, high, low, close, volume, turnover,")
    print("  EMA5, EMA10, EMA20, EMA50, EMA100")
    print("=" * 72)


if __name__ == "__main__":
    main()
