"""Orchestrates the sweep: for every (symbol x strategy x parameter
combination), run a backtest and collect its metrics into a leaderboard.

Runs combinations in parallel across processes (CPU-bound: pandas indicator
math + the numpy loops in strategies/grid_dca.py).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from optimizer.param_grid import grid_combinations, random_combinations
from strategies import build_strategy


@dataclass
class SweepConfig:
    symbols: list[str]
    strategy_names: list[str]
    interval: str = "1h"
    initial_capital: float = 1000.0
    taker_fee: float = 0.001
    slippage: float = 0.0005
    method: str = "random"          # "grid" or "random"
    n_random_samples: int = 500      # per (symbol, strategy) when method == "random"
    n_grid_points: int = 8           # per numeric param when method == "grid"
    seed: int | None = 42


def _run_one(symbol: str, strategy_name: str, params: dict, df: pd.DataFrame, cfg: SweepConfig) -> dict:
    strategy = build_strategy(strategy_name)
    try:
        position = strategy.generate_positions(df, params)
        result = run_backtest(
            df, position,
            initial_capital=cfg.initial_capital,
            taker_fee=cfg.taker_fee,
            slippage=cfg.slippage,
        )
        metrics = compute_metrics(result, cfg.interval, cfg.initial_capital)
    except Exception as exc:  # a bad param combo shouldn't kill the whole sweep
        metrics = {
            "total_return": float("nan"), "cagr": float("nan"), "max_drawdown": float("nan"),
            "sharpe": float("nan"), "win_rate": float("nan"), "profit_factor": float("nan"),
            "n_trades": 0, "n_bars": 0, "plausible": False, "score": -999.0, "error": str(exc),
        }
    return {"symbol": symbol, "strategy": strategy_name, "params": params, **metrics}


def _combos_for(strategy_name: str, cfg: SweepConfig):
    strategy = build_strategy(strategy_name)
    if cfg.method == "grid":
        yield from grid_combinations(strategy.param_space, n_grid_points=cfg.n_grid_points)
    else:
        yield from random_combinations(strategy.param_space, cfg.n_random_samples, seed=cfg.seed)


def run_sweep(
    data_by_symbol: dict[str, pd.DataFrame],
    cfg: SweepConfig,
    n_jobs: int = 4,
    progress_callback=None,
) -> pd.DataFrame:
    """Returns a DataFrame with one row per (symbol, strategy, params) combo,
    sorted by `score` descending — the top rows are the sweep's leaderboard.
    """
    jobs = []
    for symbol in cfg.symbols:
        df = data_by_symbol.get(symbol)
        if df is None or df.empty:
            continue
        for strategy_name in cfg.strategy_names:
            for params in _combos_for(strategy_name, cfg):
                jobs.append((symbol, strategy_name, params))

    rows = []
    if n_jobs <= 1:
        for i, (symbol, strategy_name, params) in enumerate(jobs, 1):
            rows.append(_run_one(symbol, strategy_name, params, data_by_symbol[symbol], cfg))
            if progress_callback:
                progress_callback(i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_run_one, symbol, strategy_name, params, data_by_symbol[symbol], cfg): i
                for i, (symbol, strategy_name, params) in enumerate(jobs, 1)
            }
            for done, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if progress_callback:
                    progress_callback(done, len(jobs))

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values("score", ascending=False).reset_index(drop=True)
    return leaderboard
