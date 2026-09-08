"""Minimal Bybit V5 REST client for historical kline (candle) data.

Uses the public market endpoint ``GET /v5/market/kline`` which needs no API
key.  Reference: https://bybit-exchange.github.io/docs/v5/market/kline

Each candle array returned by Bybit has exactly this order::

    [startTime(ms), open, high, low, close, volume, turnover]

Candles are returned newest-first, so we sort ascending and paginate backwards
to collect an arbitrary number of minutes (limit per request is 1000).
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from config import BYBIT_BASE_URL, CATEGORY

# Column names (in the same order as Bybit returns them).
KLINE_COLUMNS = ["timestamp_ms", "open", "high", "low", "close", "volume", "turnover"]

# interval string -> milliseconds
INTERVAL_MS = {
    "1": 60_000,
    "3": 180_000,
    "5": 300_000,
    "15": 900_000,
    "30": 1_800_000,
    "60": 3_600_000,
}


class BybitClient:
    """Fetch and parse OHLCV + turnover candles from Bybit's public API."""

    def __init__(
        self,
        base_url: str = BYBIT_BASE_URL,
        category: str = CATEGORY,
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.category = category
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_klines(
        self,
        symbol: str,
        interval: str = "1",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Return a single page of klines as a DataFrame, ascending by time."""
        limit = max(1, min(int(limit), 1000))
        params = {
            "category": self.category,
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)

        payload = self._request("/v5/market/kline", params=params)
        rows = payload.get("result", {}).get("list", [])

        if not rows:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        for col in KLINE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
        return df.sort_values("timestamp_ms").reset_index(drop=True)

    def fetch_history(self, symbol: str, minutes: int, interval: str = "1") -> pd.DataFrame:
        """Fetch the most recent ``minutes`` candles, paginating backwards.

        The Bybit endpoint caps a page at 1000 candles, so for anything longer
        than 1000 minutes we request successive older pages and stitch them
        together without overlap or gaps.
        """
        interval_ms = INTERVAL_MS.get(interval, 60_000)
        target = max(1, int(minutes))
        frames: list[pd.DataFrame] = []
        cursor = int(time.time() * 1000)  # end of the newest candle window
        collected = 0

        while collected < target:
            limit = min(1000, target - collected)
            page = self.get_klines(symbol, interval=interval, end_ms=cursor, limit=limit)
            if page.empty:
                break

            frames.append(page)
            collected += len(page)

            oldest = int(page["timestamp_ms"].iloc[0])
            if oldest >= cursor:  # safety: no progress would mean an infinite loop
                break
            cursor = oldest - interval_ms  # step one candle before the oldest seen

        if not frames:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms")
        return out.tail(target).reset_index(drop=True)  # keep the newest `target` rows

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _request(self, path: str, params: Optional[dict] = None, retries: int = 3) -> dict:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                data = resp.json()
                if data.get("retCode") == 0:
                    return data
                last_error = RuntimeError(
                    f"Bybit retCode={data.get('retCode')} retMsg={data.get('retMsg')}"
                )
            except (requests.RequestException, ValueError) as exc:  # network / bad json
                last_error = exc
            time.sleep(1.0 * (attempt + 1))  # simple linear backoff

        raise RuntimeError(f"Bybit request failed after {retries} attempts: {last_error}")
