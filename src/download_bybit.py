"""Step 1 — Bybit historical 1-minute downloader (bulk, resumable).

Downloads full 1m candle history for the requested symbols into
``data/raw/<SYMBOL>/<SYMBOL>_1m.csv``, paginating backwards by date until the
requested lookback is reached or Bybit returns no more candles.

The Bybit ``GET /v5/market/kline`` endpoint returns at most 1000 candles per
request, newest-first, and accepts ``start``/``end`` (ms).  We walk backwards
one 1000-candle page at a time, de-duplicate, sort and save a single CSV per
symbol.

Usage::

    python src/download_bybit.py --symbols DOGEUSDT BTCUSDT XAUTUSDT --days 730
    python src/download_bybit.py --symbols DOGEUSDT --days 180 --category spot
    python src/download_bybit.py --symbols BTCUSDT --days 3650 --sleep 0.15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

# Allow importing the Bybit client living in the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bybit_client import BybitClient  # noqa: E402

KLINE_COLUMNS = ["timestamp_ms", "open", "high", "low", "close", "volume", "turnover"]
PAGE_SIZE = 1000                      # Bybit hard max per request
DEFAULT_DAYS = 730                    # ~2 years of 1-minute candles


def fetch_range(
    client: BybitClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1",
    sleep: float = 0.12,
) -> pd.DataFrame:
    """Fetch all candles in ``[start_ms, end_ms]``, paginating backwards."""
    frames: list[pd.DataFrame] = []
    cursor = end_ms
    n_pages = 0

    while cursor > start_ms:
        page = client.get_klines(symbol, interval=interval, end_ms=cursor, limit=PAGE_SIZE)
        if page.empty:
            break

        frames.append(page)
        n_pages += 1

        oldest = int(page["timestamp_ms"].iloc[0])
        if n_pages % 50 == 0 or n_pages == 1:
            print(f"    [{symbol}] page {n_pages}: {PAGE_SIZE} candles "
                  f"(oldest {pd.to_datetime(oldest, unit='ms', utc=True):%Y-%m-%d %H:%M} UTC)", flush=True)

        if oldest <= start_ms:
            break
        if oldest >= cursor:  # safety: no progress would mean an infinite loop
            break
        cursor = oldest - 1  # step 1ms before the oldest candle already fetched
        time.sleep(sleep)

    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms")
    out = out[out["timestamp_ms"] >= start_ms]
    return out.reset_index(drop=True)


def download(symbol: str, days: int, category: str, out_dir: Path, sleep: float) -> Path:
    """Download ``days`` days of 1m history for one symbol and save to CSV."""
    client = BybitClient(category=category)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400 * 1000

    print(f"\n[{symbol}] downloading {days} days of 1m candles "
          f"({pd.to_datetime(start_ms, unit='ms', utc=True):%Y-%m-%d} → now) ...")

    df = fetch_range(client, symbol, start_ms=start_ms, end_ms=now_ms, sleep=sleep)

    if df.empty:
        print(f"[{symbol}] WARNING: no data returned.")
        return out_dir / f"{symbol}" / f"{symbol}_1m.csv"

    out_path = out_dir / symbol / f"{symbol}_1m.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    first = pd.to_datetime(int(df["timestamp_ms"].iloc[0]), unit="ms", utc=True)
    last = pd.to_datetime(int(df["timestamp_ms"].iloc[-1]), unit="ms", utc=True)
    print(f"[{symbol}] saved {len(df)} rows → {out_path}")
    print(f"    range: {first:%Y-%m-%d %H:%M} UTC → {last:%Y-%m-%d %H:%M} UTC")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 1: Bybit historical 1m downloader")
    parser.add_argument("--symbols", nargs="+", default=["DOGEUSDT", "BTCUSDT", "XAUTUSDT"],
                        help="space-separated symbols (default: DOGEUSDT BTCUSDT XAUTUSDT)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help="how many days of 1m history to fetch (default 730 = ~2 years)")
    parser.add_argument("--category", default="spot",
                        help="Bybit category: spot|linear|inverse (default spot)")
    parser.add_argument("--sleep", type=float, default=0.12,
                        help="seconds to sleep between requests (rate-limit friendly)")
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"),
                        help="output directory (default: data/raw)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STEP 1 — Bybit historical 1m downloader")
    print(f"category={args.category}  days={args.days}  symbols={args.symbols}")

    for symbol in args.symbols:
        sym = symbol.upper()
        try:
            download(sym, args.days, args.category, out_dir, args.sleep)
        except Exception as exc:
            print(f"[{sym}] ERROR: {exc}")

    print("\nSTEP 1 DONE.")


if __name__ == "__main__":
    main()
