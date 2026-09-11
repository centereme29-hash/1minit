"""Step 6A — microstructure collector (Bybit V5 WebSocket).

Streams the public Bybit spot WebSocket and aggregates **order-book + trade-flow**
microstructure into one row per UTC minute per symbol, saved to::

    data/microstructure/<SYMBOL>_1m_micro.csv

The candle-only baselines proved net-of-fee unprofitable, so this is the single
highest-value addition: the signal for small 5-minute moves lives in the order
book / trade flow, not in OHLC candles.

Topics per symbol (one connection, comma-joined subscription):

    orderbook.1.<SYMBOL>     best bid/ask + sizes (cheap)
    orderbook.50.<SYMBOL>    depth (real imbalance, heavier)
    publicTrade.<SYMBOL>     every public trade (price, side, size, ts)

Per-minute columns (all computed inline):

    best_bid, best_ask, spread, rel_spread, mid_price,
    ob_imbalance, buy_volume, sell_volume, trade_count,
    large_trade_count, vwap, trade_imbalance

Usage::

    python src/microstructure.py --symbols DOGEUSDT BTCUSDT XAUTUSDT --minutes 10
    python src/microstructure.py --depth 50 --minutes 60
    python src/microstructure.py --backfill --minutes 5 --poll-seconds 2

Run it for several days BEFORE retraining so there is a real micro history to
merge into the feature matrix.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

WS_URL = "wss://stream.bybit.com/v5/public/spot"
API_BASE = "https://api.bybit.com"

OUT_COLUMNS = [
    "time", "best_bid", "best_ask", "spread", "rel_spread", "mid_price",
    "ob_imbalance", "buy_volume", "sell_volume", "trade_count",
    "large_trade_count", "vwap", "trade_imbalance",
]


# ---------------------------------------------------------------------------
# Aggregator (pure, synchronous, unit-testable)
# ---------------------------------------------------------------------------
class MicroAggregator:
    """Accumulate book snapshots + trades and emit one row per completed minute."""

    def __init__(
        self,
        symbols: list[str],
        depth: int = 1,
        large_k: float = 5.0,
        rolling_len: int = 2000,
        on_row: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.depth = depth
        self.large_k = large_k
        self.on_row = on_row
        self.rows: list[tuple[str, dict]] = []          # fallback sink for tests
        self._flushed_minute: int | None = None
        self._state: dict[str, dict] = {}
        for s in self.symbols:
            self._state[s] = {
                "book": {"best_bid": None, "best_ask": None,
                         "bid_size": None, "ask_size": None,
                         "bids": {}, "asks": {}},
                "trade_buckets": {},        # minute_ms -> list of trade dicts
                "recent_sizes": deque(maxlen=rolling_len),
                "seen_ids": set(),          # trade-id dedupe
            }

    # ---- ingest ----------------------------------------------------------
    def on_orderbook(self, symbol: str, data: dict, kind: str = "snapshot") -> None:
        symbol = symbol.upper()
        book = self._state[symbol]["book"]
        if self.depth > 1:
            bids = data.get("b") or []
            asks = data.get("a") or []
            if kind == "snapshot":
                book["bids"] = {float(p): float(s) for p, s in bids}
                book["asks"] = {float(p): float(s) for p, s in asks}
            else:
                for p, s in bids:
                    p, s = float(p), float(s)
                    if s == 0.0:
                        book["bids"].pop(p, None)
                    else:
                        book["bids"][p] = s
                for p, s in asks:
                    p, s = float(p), float(s)
                    if s == 0.0:
                        book["asks"].pop(p, None)
                    else:
                        book["asks"][p] = s
            if book["bids"]:
                book["best_bid"] = max(book["bids"])
                book["bid_size"] = book["bids"][book["best_bid"]]
            if book["asks"]:
                book["best_ask"] = min(book["asks"])
                book["ask_size"] = book["asks"][book["best_ask"]]
        else:
            bids = data.get("b") or []
            asks = data.get("a") or []
            if bids:
                book["best_bid"] = float(bids[0][0])
                book["bid_size"] = float(bids[0][1])
            if asks:
                book["best_ask"] = float(asks[0][0])
                book["ask_size"] = float(asks[0][1])

    def on_trade(self, symbol: str, trades: list[dict]) -> None:
        symbol = symbol.upper()
        st = self._state[symbol]
        for t in trades:
            tid = t["trade_id"]
            if tid in st["seen_ids"]:
                continue
            st["seen_ids"].add(tid)
            if len(st["seen_ids"]) > 200_000:  # bound memory on very long runs
                st["seen_ids"] = set(list(st["seen_ids"])[-50_000:])
            minute = (t["ts_ms"] // 60_000) * 60_000
            st["trade_buckets"].setdefault(minute, []).append(t)
            st["recent_sizes"].append(t["size"])

    # ---- flush -----------------------------------------------------------
    def _earliest_minute(self, now_min: int) -> int:
        """Smallest buffered minute index (so nothing gets skipped), else now_min."""
        earliest = now_min
        for st in self._state.values():
            if st["trade_buckets"]:
                earliest = min(earliest, min(st["trade_buckets"]) // 60_000)
        return earliest

    def tick(self, now_ms: int | None = None) -> None:
        """Finalize every fully-completed minute (index < ``now_min``)."""
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        now_min = now_ms // 60_000
        if self._flushed_minute is None:
            self._flushed_minute = self._earliest_minute(now_min)
        while self._flushed_minute < now_min:
            self._finalize_minute(self._flushed_minute * 60_000)
            self._flushed_minute += 1

    def force_flush(self) -> None:
        """Finalize the current (still-open) minute too, for clean shutdown."""
        now_min = int(time.time() * 1000) // 60_000
        if self._flushed_minute is None:
            self._flushed_minute = self._earliest_minute(now_min)
        for minute in range(self._flushed_minute, now_min + 1):
            self._finalize_minute(minute * 60_000)
        self._flushed_minute = now_min + 1

    def _finalize_minute(self, minute_ms: int) -> None:
        for symbol in self.symbols:
            st = self._state[symbol]
            trades = st["trade_buckets"].pop(minute_ms, [])
            row = self._build_row(symbol, minute_ms, st["book"], trades, st["recent_sizes"])
            if row is None:
                continue
            if self.on_row is not None:
                self.on_row(symbol, row)
            else:
                self.rows.append((symbol, row))

    def _build_row(self, symbol: str, minute_ms: int, book: dict, trades: list, recent) -> dict | None:
        best_bid = book.get("best_bid")
        best_ask = book.get("best_ask")
        if best_bid is None or best_ask is None:
            return None
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        rel_spread = spread / mid if mid else 0.0

        if self.depth > 1:
            bid_vol = sum(book["bids"].values())
            ask_vol = sum(book["asks"].values())
        else:
            bid_vol = book.get("bid_size") or 0.0
            ask_vol = book.get("ask_size") or 0.0
        ob_imb = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) else 0.0

        buy_vol = sum(t["size"] for t in trades if t["buy"])
        sell_vol = sum(t["size"] for t in trades if not t["buy"])
        trade_count = len(trades)

        large = 0
        if trades and recent:
            med = statistics.median(recent)
            if med and med > 0:
                large = sum(1 for t in trades if t["size"] > self.large_k * med)

        notional = sum(t["price"] * t["size"] for t in trades)
        total_size = sum(t["size"] for t in trades)
        vwap = notional / total_size if total_size else mid
        trade_imb = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) else 0.0

        return {
            "time": pd.Timestamp(minute_ms, unit="ms", tz="UTC"),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "rel_spread": rel_spread,
            "mid_price": mid,
            "ob_imbalance": ob_imb,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "trade_count": trade_count,
            "large_trade_count": large,
            "vwap": vwap,
            "trade_imbalance": trade_imb,
        }


# ---------------------------------------------------------------------------
# CSV writer (appends one row per minute, crash-safe)
# ---------------------------------------------------------------------------
class MicroWriter:
    def __init__(self, out_dir: Path | str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, symbol: str, row: dict) -> None:
        path = self.out_dir / f"{symbol.upper()}_1m_micro.csv"
        df = pd.DataFrame([row], columns=OUT_COLUMNS)
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------
def _normalize_trade(t: dict) -> dict:
    """Accept either a Bybit WS publicTrade element or a REST recent-trade row."""
    if "T" in t:  # WebSocket publicTrade
        ts_ms = int(t["T"])
        price = float(t["p"])
        size = float(t["v"])
        buy = str(t.get("S", "Buy")).lower().startswith("buy")
        tid = str(t.get("i") or f"{ts_ms}-{price}-{size}-{t.get('S')}")
    else:  # REST /v5/market/recent-trade
        ts_ms = int(t.get("time", time.time() * 1000))
        price = float(t["price"])
        size = float(t["size"])
        buy = str(t.get("side", "Buy")).lower().startswith("buy")
        tid = str(t.get("execId") or f"{ts_ms}-{price}-{size}-{t.get('side')}")
    return {"ts_ms": ts_ms, "price": price, "size": size, "buy": buy, "trade_id": tid}


# ---------------------------------------------------------------------------
# WebSocket stream
# ---------------------------------------------------------------------------
class BybitMicroStream:
    def __init__(self, symbols: list[str], aggregator: MicroAggregator,
                 url: str = WS_URL, ping_interval: int = 20) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.agg = aggregator
        self.url = url
        self.ping_interval = ping_interval
        self._stop = asyncio.Event()

    def topics(self) -> list[str]:
        topics: list[str] = []
        for s in self.symbols:
            topics.append(f"orderbook.{self.agg.depth}.{s}")
            topics.append(f"publicTrade.{s}")
        return topics

    async def run(self, minutes: int | None = None) -> None:
        import websockets  # local import: only needed for live streaming
        stop_at = time.time() + minutes * 60 if minutes else None
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, ping_interval=None) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": self.topics()}))
                    print(f"connected to {self.url}")
                    flusher = asyncio.create_task(self._flush_loop(stop_at))
                    pinger = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            await self._handle(json.loads(raw))
                            if stop_at and time.time() >= stop_at:
                                self._stop.set()
                    finally:
                        flusher.cancel()
                        pinger.cancel()
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                print(f"websocket error ({exc}); reconnecting in 3s ...")
                await asyncio.sleep(3)
            if stop_at and time.time() >= stop_at:
                break

    async def _flush_loop(self, stop_at: float | None) -> None:
        while True:
            self.agg.tick()
            await asyncio.sleep(1)
            if stop_at and time.time() >= stop_at:
                break

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(self.ping_interval)
            await ws.send(json.dumps({"op": "ping"}))

    async def _handle(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        data = msg.get("data")
        if not topic or data is None:
            return
        if topic.startswith("orderbook."):
            parts = topic.split(".")
            symbol = parts[-1] if len(parts) >= 3 else ""
            self.agg.on_orderbook(symbol, data, kind=msg.get("type", "snapshot"))
        elif topic.startswith("publicTrade."):
            symbol = topic.split(".")[-1]
            trades = data if isinstance(data, list) else [data]
            self.agg.on_trade(symbol, [_normalize_trade(t) for t in trades])


# ---------------------------------------------------------------------------
# REST backfill (bootstrap a small micro history without waiting days)
# ---------------------------------------------------------------------------
def _rest_get(path: str, params: dict) -> dict:
    import requests
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_orderbook(symbol: str, depth: int = 1) -> tuple[dict, str]:
    payload = _rest_get("/v5/market/orderbook", {
        "category": "spot", "symbol": symbol.upper(), "limit": depth,
    })
    return payload.get("result", {}), "snapshot"


def fetch_recent_trades(symbol: str, limit: int = 500) -> list[dict]:
    payload = _rest_get("/v5/market/recent-trade", {
        "category": "spot", "symbol": symbol.upper(), "limit": limit,
    })
    return payload.get("result", {}).get("list", [])


def rest_backfill(symbols: list[str], minutes: int, depth: int,
                  poll_seconds: float, out_dir: Path | str) -> None:
    writer = MicroWriter(out_dir)
    agg = MicroAggregator(symbols, depth=depth, on_row=writer.write)
    end = time.time() + minutes * 60
    print(f"REST backfill: {minutes} min @ {poll_seconds}s poll, depth={depth}")
    while time.time() < end:
        now_ms = int(time.time() * 1000)
        for s in symbols:
            try:
                book, _ = fetch_orderbook(s, depth)
                agg.on_orderbook(s, book, "snapshot")
                agg.on_trade(s, [_normalize_trade(t) for t in fetch_recent_trades(s)])
            except Exception as exc:  # noqa: BLE001
                print(f"[{s}] poll error: {exc}")
        agg.tick(now_ms)
        time.sleep(poll_seconds)
    agg.force_flush()
    print("REST backfill complete ->", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bybit microstructure collector (6A)")
    parser.add_argument("--symbols", nargs="+", default=["DOGEUSDT", "BTCUSDT", "XAUTUSDT"])
    parser.add_argument("--depth", type=int, default=1, choices=[1, 50],
                        help="orderbook depth: 1 = level-1, 50 = 50-level imbalance")
    parser.add_argument("--minutes", type=int, default=None,
                        help="how many minutes to run (None = run forever)")
    parser.add_argument("--large-k", type=float, default=5.0)
    parser.add_argument("--out", default=str(ROOT / "data" / "microstructure"))
    parser.add_argument("--backfill", action="store_true",
                        help="use REST polling (orderbook + recent trades) instead of WS")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)

    symbols = [s.upper() for s in args.symbols]

    if args.backfill:
        if not args.minutes:
            parser.error("--backfill requires --minutes")
        rest_backfill(symbols, args.minutes, args.depth, args.poll_seconds, args.out)
        return

    writer = MicroWriter(args.out)
    agg = MicroAggregator(symbols, depth=args.depth, large_k=args.large_k,
                          on_row=writer.write)
    stream = BybitMicroStream(symbols, agg)

    print("STEP 6A — microstructure WebSocket collector")
    print(f"symbols={symbols}  depth={args.depth}  minutes={args.minutes}")
    try:
        asyncio.run(stream.run(minutes=args.minutes))
    except KeyboardInterrupt:
        agg.force_flush()
        print("\nstopped; partial minute flushed.")


if __name__ == "__main__":
    main()
