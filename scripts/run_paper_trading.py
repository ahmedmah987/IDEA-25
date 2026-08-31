#!/usr/bin/env python3
"""Phase 2: paper-trade one (symbol, strategy, params) combo against live
Binance prices with fake money.

Two ways to run it:

  --once     Do exactly one tick and exit. This is the pattern for real
             unattended 24/7 operation: schedule this command with cron or
             a systemd timer to fire every `interval` (e.g. hourly for
             --interval 1h), on a machine that's actually always on. State
             is persisted to --state-file between runs, so nothing is lost.

  (default)  Loop forever in this process, sleeping --poll-seconds between
             ticks. Convenient for watching it live in one sitting, but
             only trades while this process itself keeps running.

Examples
--------
# Pick params by hand
python scripts/run_paper_trading.py --symbol BTCUSDT --strategy trend_ma_cross \\
    --params '{"fast_window": 20, "slow_window": 80, "use_ema": true}' --initial-capital 10 --once

# Pick the top walk-forward-validated row for this symbol/strategy automatically
python scripts/run_paper_trading.py --symbol XRPUSDT --strategy breakout_donchian \\
    --from-walk-forward reports/walk_forward_folds.csv --initial-capital 10 --once
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_trading.engine import PortfolioState, run_tick  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Paper-trade a strategy against live Binance prices")
    p.add_argument("--symbol", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--params", default=None, help="JSON dict of strategy params")
    p.add_argument("--from-walk-forward", default=None,
                    help="Path to a walk_forward_folds.csv; use that (symbol, strategy)'s most recent fold's params")
    p.add_argument("--interval", default="1h")
    p.add_argument("--initial-capital", type=float, default=10.0)
    p.add_argument("--taker-fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--warmup-bars", type=int, default=600)
    p.add_argument("--state-file", default=None, help="Defaults to reports/paper_state_<symbol>_<strategy>.json")
    p.add_argument("--ledger-file", default="reports/paper_trading_ledger.csv")
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--once", action="store_true", help="Do a single tick and exit (recommended for cron)")
    p.add_argument("--max-ticks", type=int, default=None, help="Stop after N ticks even in loop mode (mainly for testing)")
    return p.parse_args()


def resolve_params(args) -> dict:
    if args.params:
        return json.loads(args.params)
    if args.from_walk_forward:
        import pandas as pd
        folds = pd.read_csv(args.from_walk_forward)
        match = folds[(folds["symbol"] == args.symbol) & (folds["strategy"] == args.strategy)]
        if match.empty:
            print(f"No walk-forward rows found for {args.symbol}/{args.strategy} in {args.from_walk_forward}")
            sys.exit(1)
        # Most recent fold's in-sample-winning params -- the most up-to-date fit.
        best_row = match.sort_values("fold").iloc[-1]
        return ast.literal_eval(best_row["params"])
    print("Provide either --params or --from-walk-forward")
    sys.exit(1)


def main():
    args = parse_args()
    params = resolve_params(args)
    state_file = args.state_file or f"reports/paper_state_{args.symbol}_{args.strategy}.json"

    if Path(state_file).exists():
        state = PortfolioState.load(state_file)
        print(f"Resumed state from {state_file}: equity=${state.equity:.4f}, "
              f"held_position={state.held_position:.2f}, n_trades={state.n_trades}")
    else:
        state = PortfolioState.new(args.symbol, args.strategy, params, args.interval, args.initial_capital)
        print(f"Started new paper-trading state: {args.symbol}/{args.strategy}, "
              f"initial_capital=${args.initial_capital}, params={params}")

    def _tick():
        nonlocal state
        state, trade = run_tick(
            state, warmup_bars=args.warmup_bars, taker_fee=args.taker_fee,
            slippage=args.slippage, ledger_path=args.ledger_file,
        )
        state.save(state_file)
        pnl = state.equity - state.initial_capital
        pnl_pct = pnl / state.initial_capital * 100
        ts = state.updated_at
        print(f"[{ts}] price-driven equity=${state.equity:.4f} ({pnl_pct:+.2f}%) "
              f"held_position={state.held_position:.2f}", end="")
        if trade:
            print(f"  <- TRADE: {trade['from_position']:.2f} -> {trade['to_position']:.2f} "
                  f"@ ${trade['price']:.4f} (cost ${trade['cost_paid']:.4f})")
        else:
            print()

    if args.once:
        _tick()
        return

    tick_count = 0
    print(f"Looping every {args.poll_seconds}s. Ctrl+C to stop (state is saved after every tick).")
    try:
        while args.max_ticks is None or tick_count < args.max_ticks:
            _tick()
            tick_count += 1
            if args.max_ticks is None or tick_count < args.max_ticks:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped. State saved -- rerun the same command to resume.")


if __name__ == "__main__":
    main()
