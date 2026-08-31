#!/usr/bin/env python3
"""Main entry point: fetch data, run the strategy sweep, print & save a
leaderboard.

Examples
--------
# quick smoke run, config.yaml defaults, sequential (safe everywhere)
python scripts/run_simulation.py --n-jobs 1

# a bigger sweep across the default symbol list
python scripts/run_simulation.py --n-random-samples 500 --n-jobs 4

# every USDT pair on Binance (slow: fetches + backtests hundreds of symbols)
python scripts/run_simulation.py --symbols all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fetch_binance_data import get_usdt_pairs, load_or_fetch  # noqa: E402
from optimizer.runner import SweepConfig, run_sweep  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Run the IDEA-25 strategy simulation sweep")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--symbols", default=None, help="Comma-separated symbols, or 'all'. Overrides config.")
    p.add_argument("--interval", default=None)
    p.add_argument("--lookback-days", type=int, default=None)
    p.add_argument("--strategies", default=None, help="Comma-separated strategy names. Overrides config.")
    p.add_argument("--method", choices=["grid", "random"], default=None)
    p.add_argument("--n-random-samples", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--out", default="reports/leaderboard.csv")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    interval = args.interval or cfg["data"]["interval"]
    lookback_days = args.lookback_days or cfg["data"]["lookback_days"]
    cache_dir = cfg["data"]["cache_dir"]
    base_url = cfg["data"]["base_url"]

    if args.symbols:
        symbols = (
            get_usdt_pairs(base_url=base_url, quote_asset=cfg["universe"]["quote_asset"])
            if args.symbols.strip().lower() == "all"
            else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        )
    else:
        symbols = cfg["universe"]["symbols"]

    strategy_names = (
        [s.strip() for s in args.strategies.split(",")] if args.strategies else cfg["strategies"]
    )

    print(f"Universe: {len(symbols)} symbol(s) | Strategies: {strategy_names}")
    print(f"Fetching/caching {interval} data ({lookback_days}d lookback) from {base_url} ...")

    data_by_symbol: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = load_or_fetch(sym, interval=interval, lookback_days=lookback_days, cache_dir=cache_dir, base_url=base_url)
        except Exception as exc:
            print(f"  [{i}/{len(symbols)}] {sym}: FAILED to fetch ({exc}) — skipping")
            continue
        if df.empty:
            print(f"  [{i}/{len(symbols)}] {sym}: no data returned — skipping")
            continue
        data_by_symbol[sym] = df
        print(f"  [{i}/{len(symbols)}] {sym}: {len(df)} candles cached")

    if not data_by_symbol:
        print("No usable data fetched — aborting.")
        sys.exit(1)

    sweep_cfg = SweepConfig(
        symbols=list(data_by_symbol.keys()),
        strategy_names=strategy_names,
        interval=interval,
        initial_capital=cfg["backtest"]["initial_capital"],
        taker_fee=cfg["backtest"]["taker_fee"],
        slippage=cfg["backtest"]["slippage"],
        method=args.method or cfg["optimizer"]["method"],
        n_random_samples=args.n_random_samples or cfg["optimizer"]["n_random_samples"],
    )
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg["optimizer"]["n_jobs"]
    top_n = args.top_n or cfg["optimizer"]["top_n"]

    print(f"\nRunning sweep: method={sweep_cfg.method}, n_jobs={n_jobs} ...")
    t0 = time.time()

    def _progress(done, total):
        if done % max(1, total // 20) == 0 or done == total:
            print(f"  {done}/{total} combinations evaluated", end="\r")

    leaderboard = run_sweep(data_by_symbol, sweep_cfg, n_jobs=n_jobs, progress_callback=_progress)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {len(leaderboard)} combinations evaluated.")

    if leaderboard.empty:
        print("No results produced.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(args.out, index=False)
    print(f"Full leaderboard saved to {args.out}\n")

    print(f"Top {top_n} (ranked by score — see backtest/metrics.py:score for how it's computed):")
    cols = ["symbol", "strategy", "score", "cagr", "sharpe", "max_drawdown", "win_rate", "n_trades", "params"]
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(leaderboard[cols].head(top_n).to_string(index=False))

    print(
        "\nReminder: these are historical-data results, not a promise of future profit. "
        "Treat the top rows as candidates for further scrutiny (out-of-sample testing, "
        "paper trading) — not as a strategy to deploy blindly with real funds."
    )


if __name__ == "__main__":
    main()
