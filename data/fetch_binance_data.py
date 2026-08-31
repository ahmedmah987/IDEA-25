"""Historical OHLCV data fetcher for Binance Spot.

Uses `data-api.binance.vision` — Binance's official, key-less, market-data-only
REST mirror. It serves the same endpoints as api.binance.com (klines,
exchangeInfo, ...) but is meant purely for market data and, unlike the main
trading API, is not subject to the same geographic trading restrictions.
No API key is required for any function in this file: everything here reads
public market data only.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://data-api.binance.vision"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_base", "taker_buy_quote"]

_SESSION = requests.Session()


def _get(url: str, params: dict, max_retries: int = 5, timeout: int = 20) -> object:
    """GET with basic retry/backoff for transient network or rate-limit errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code == 418:
                # Rate limited / banned momentarily — back off and retry.
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:  # noqa: PERF203 - retry loop
            last_exc = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"GET {url} failed after {max_retries} retries") from last_exc


def get_usdt_pairs(base_url: str = DEFAULT_BASE_URL, quote_asset: str = "USDT") -> list[str]:
    """Return every TRADING spot symbol quoted in `quote_asset` (default USDT)."""
    info = _get(f"{base_url}/api/v3/exchangeInfo", params={})
    out = []
    for s in info["symbols"]:
        if (
            s.get("quoteAsset") == quote_asset
            and s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed", True)
        ):
            out.append(s["symbol"])
    return sorted(out)


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> pd.DataFrame:
    """Fetch klines for one symbol between start_time_ms and end_time_ms,
    paginating past Binance's 1000-candle-per-request limit.
    """
    rows: list[list] = []
    cursor = start_time_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if cursor is not None:
            params["startTime"] = cursor
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        batch = _get(f"{base_url}/api/v3/klines", params=params)
        if not batch:
            break
        rows.extend(batch)
        last_open_time = batch[-1][0]
        if len(batch) < 1000:
            break
        cursor = last_open_time + 1
        if end_time_ms is not None and cursor >= end_time_ms:
            break
        time.sleep(0.1)  # be a polite API citizen

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if df.empty:
        return df
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[_NUMERIC_COLUMNS] = df[_NUMERIC_COLUMNS].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume", "quote_volume", "n_trades"]]


def load_or_fetch(
    symbol: str,
    interval: str = "1h",
    lookback_days: int = 365,
    cache_dir: str | Path = "data/raw",
    base_url: str = DEFAULT_BASE_URL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return cached OHLCV for `symbol`/`interval`, fetching from Binance and
    caching to Parquet on first use (or when the cache is stale/missing).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}_{interval}.parquet"

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=lookback_days)

    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        # Only reuse the cache if it actually covers the requested lookback
        # window and isn't stale — otherwise a cache built for a shorter
        # window (or an old run) would silently get reused for a longer one,
        # producing results for a smaller/older sample than requested.
        one_bar = pd.Timedelta(interval)
        covers_lookback = not df.empty and df.index.min() <= start + pd.Timedelta(days=1)
        is_fresh = not df.empty and df.index.max() >= end - 2 * one_bar - pd.Timedelta(hours=6)
        if covers_lookback and is_fresh:
            return df
    df = fetch_klines(
        symbol,
        interval=interval,
        start_time_ms=int(start.timestamp() * 1000),
        end_time_ms=int(end.timestamp() * 1000),
        base_url=base_url,
    )
    if not df.empty:
        df.to_parquet(cache_path)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch & cache Binance spot OHLCV data")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols, or 'all'")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    symbols = get_usdt_pairs() if args.symbols.strip().lower() == "all" else [
        s.strip().upper() for s in args.symbols.split(",") if s.strip()
    ]
    print(f"Fetching {len(symbols)} symbol(s) @ {args.interval}, {args.lookback_days}d lookback...")
    for i, sym in enumerate(symbols, 1):
        df = load_or_fetch(
            sym,
            interval=args.interval,
            lookback_days=args.lookback_days,
            cache_dir=args.cache_dir,
            force_refresh=args.force_refresh,
        )
        print(f"[{i}/{len(symbols)}] {sym}: {len(df)} candles")
