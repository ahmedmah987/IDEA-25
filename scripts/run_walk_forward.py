#!/usr/bin/env python3
"""Validate leaderboard candidates with walk-forward testing.

Takes the top rows of an existing reports/leaderboard.csv (or explicit
--symbols/--strategies), re-optimizes on rolling train windows, and scores
the winner on each following unseen test window. Prints a verdict per
(symbol, strategy): does out-of-sample performance hold up?

Example
-------
python scripts/run_simulation.py                       # produces the leaderboard first
python scripts/run_walk_forward.py --top 10             # validate its top 10 candidates
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fetch_binance_data import load_or_fetch  # noqa: E402
from optimizer.walk_forward import WalkForwardConfig, run_walk_forward, summarize  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward validate strategy candidates")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--leaderboard", default="reports/leaderboard.csv")
    p.add_argument("--top", type=int, default=10, help="How many top leaderboard rows to validate")
    p.add_argument("--symbols", default=None, help="Comma-separated, overrides leaderboard-derived list")
    p.add_argument("--strategies", default=None, help="Comma-separated, overrides leaderboard-derived list")
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--n-random-samples", type=int, default=150)
    p.add_argument("--out", default="reports/walk_forward_summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.symbols and args.strategies:
        pairs = [
            (s.strip().upper(), st.strip())
            for s in args.symbols.split(",")
            for st in args.strategies.split(",")
        ]
    else:
        lb_path = Path(args.leaderboard)
        if not lb_path.exists():
            print(f"No leaderboard at {lb_path} and no explicit --symbols/--strategies given. "
                  f"Run scripts/run_simulation.py first, or pass both flags explicitly.")
            sys.exit(1)
        leaderboard = pd.read_csv(lb_path)
        top = leaderboard.head(args.top)
        pairs = list(dict.fromkeys(zip(top["symbol"], top["strategy"])))  # dedupe, preserve order

    print(f"Validating {len(pairs)} (symbol, strategy) pair(s) with "
          f"{args.train_days}d train / {args.test_days}d test rolling folds ...")

    wf_cfg = WalkForwardConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        n_random_samples=args.n_random_samples,
        interval=cfg["data"]["interval"],
        initial_capital=cfg["backtest"]["initial_capital"],
        taker_fee=cfg["backtest"]["taker_fee"],
        slippage=cfg["backtest"]["slippage"],
    )

    all_folds = []
    for i, (symbol, strategy_name) in enumerate(pairs, 1):
        df = load_or_fetch(
            symbol, interval=cfg["data"]["interval"],
            lookback_days=cfg["data"]["lookback_days"],
            cache_dir=cfg["data"]["cache_dir"], base_url=cfg["data"]["base_url"],
        )
        if df.empty:
            print(f"  [{i}/{len(pairs)}] {symbol}/{strategy_name}: no data, skipping")
            continue
        fold_results = run_walk_forward(df, symbol, strategy_name, wf_cfg)
        n_folds = len(fold_results)
        print(f"  [{i}/{len(pairs)}] {symbol}/{strategy_name}: {n_folds} fold(s) evaluated")
        all_folds.append(fold_results)

    if not all_folds:
        print("No folds evaluated (not enough history for the requested train+test window?).")
        sys.exit(1)

    fold_df = pd.concat(all_folds, ignore_index=True)
    summary = summarize(fold_df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    fold_df.to_csv(str(Path(args.out).with_name("walk_forward_folds.csv")), index=False)

    print(f"\nSaved fold-level detail to reports/walk_forward_folds.csv, summary to {args.out}\n")
    with pd.option_context("display.max_colwidth", 40, "display.width", 160):
        print(summary.to_string(index=False))

    print(
        "\nHow to read this: `degradation_ratio` near 1.0 means out-of-sample performance matched "
        "what the optimizer promised in-sample — a real edge. Near 0 or negative means the strategy "
        "was fit to noise in the training window and fell apart on unseen data — discard it, no "
        "matter how good its original leaderboard score looked. `pct_folds_profitable` below ~0.5 "
        "is also a red flag: the edge isn't showing up consistently across time."
    )


if __name__ == "__main__":
    main()
