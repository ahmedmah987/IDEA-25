"""Walk-forward validation: the honesty check on top of the sweep.

The plain optimizer sweep finds parameters that fit *one* historical
window best — which is exactly how you overfit. Walk-forward guards
against that: split the data into consecutive (train, test) folds that
roll forward through time, re-optimize on each train segment only, then
score the winning parameters on the *next*, unseen test segment. A
strategy that only looked good in-sample degrades sharply out-of-sample;
one with a real edge degrades much less.

    |--- train fold 1 ---|-- test 1 --|
              |--- train fold 2 ---|-- test 2 --|
                        |--- train fold 3 ---|-- test 3 --|

(rolling window: each fold's train segment is train_days long, slides
forward by test_days each time.)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from optimizer.param_grid import random_combinations
from strategies import build_strategy


@dataclass
class WalkForwardConfig:
    train_days: int = 90
    test_days: int = 30
    n_random_samples: int = 150   # search budget spent per fold, not overall
    seed: int = 42
    interval: str = "1h"
    initial_capital: float = 1000.0
    taker_fee: float = 0.001
    slippage: float = 0.0005


def _bars_per_day(interval: str) -> int:
    return max(1, int(pd.Timedelta(days=1) / pd.Timedelta(interval)))


def generate_folds(df: pd.DataFrame, cfg: WalkForwardConfig) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    bpd = _bars_per_day(cfg.interval)
    train_bars, test_bars = cfg.train_days * bpd, cfg.test_days * bpd
    folds = []
    start = 0
    while start + train_bars + test_bars <= len(df):
        train = df.iloc[start:start + train_bars]
        test = df.iloc[start + train_bars:start + train_bars + test_bars]
        folds.append((train, test))
        start += test_bars  # roll forward; train window resets each fold (not expanding)
    return folds


def _backtest_kwargs(cfg: WalkForwardConfig) -> dict:
    return dict(initial_capital=cfg.initial_capital, taker_fee=cfg.taker_fee, slippage=cfg.slippage)


def _best_on_segment(df_segment: pd.DataFrame, strategy_name: str, cfg: WalkForwardConfig, seed: int) -> dict | None:
    strategy = build_strategy(strategy_name)
    best = None
    for params in random_combinations(strategy.param_space, cfg.n_random_samples, seed=seed):
        try:
            position = strategy.generate_positions(df_segment, params)
            result = run_backtest(df_segment, position, **_backtest_kwargs(cfg))
            metrics = compute_metrics(result, cfg.interval, cfg.initial_capital)
        except Exception:
            continue
        if best is None or metrics["score"] > best["score"]:
            best = {"params": params, **metrics}
    return best


def _evaluate(df_segment: pd.DataFrame, strategy_name: str, params: dict, cfg: WalkForwardConfig) -> dict:
    strategy = build_strategy(strategy_name)
    position = strategy.generate_positions(df_segment, params)
    result = run_backtest(df_segment, position, **_backtest_kwargs(cfg))
    return compute_metrics(result, cfg.interval, cfg.initial_capital)


def run_walk_forward(df: pd.DataFrame, symbol: str, strategy_name: str, cfg: WalkForwardConfig) -> pd.DataFrame:
    """Returns one row per fold: best params found in-sample (train), and
    that exact strategy's out-of-sample (test) performance.
    """
    folds = generate_folds(df, cfg)
    rows = []
    for i, (train_df, test_df) in enumerate(folds):
        best = _best_on_segment(train_df, strategy_name, cfg, seed=cfg.seed + i)
        if best is None:
            continue
        oos = _evaluate(test_df, strategy_name, best["params"], cfg)
        rows.append({
            "symbol": symbol, "strategy": strategy_name, "fold": i,
            "params": best["params"],
            "is_score": best["score"], "oos_score": oos["score"],
            "is_cagr": best["cagr"], "oos_cagr": oos["cagr"],
            "is_sharpe": best["sharpe"], "oos_sharpe": oos["sharpe"],
            "oos_max_drawdown": oos["max_drawdown"],
            "oos_win_rate": oos["win_rate"], "oos_n_trades": oos["n_trades"],
        })
    return pd.DataFrame(rows)


def summarize(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (symbol, strategy) across folds into a verdict:
    does out-of-sample performance hold up, or does it collapse relative
    to in-sample (the signature of overfitting)?
    """
    if fold_results.empty:
        return fold_results

    def _agg(group: pd.DataFrame) -> pd.Series:
        is_score_mean = group["is_score"].mean()
        oos_score_mean = group["oos_score"].mean()
        # ratio of what you'd actually earn out-of-sample vs. what the
        # optimizer promised in-sample; 1.0 = no degradation, <0 = OOS lost
        # money while IS looked profitable (a classic overfit signature).
        degradation = oos_score_mean / is_score_mean if is_score_mean not in (0, None) else float("nan")
        return pd.Series({
            "n_folds": len(group),
            "oos_cagr_mean": group["oos_cagr"].mean(),
            "oos_sharpe_mean": group["oos_sharpe"].mean(),
            "oos_max_drawdown_worst": group["oos_max_drawdown"].min(),
            "pct_folds_profitable": (group["oos_cagr"] > 0).mean(),
            "is_score_mean": is_score_mean,
            "oos_score_mean": oos_score_mean,
            "degradation_ratio": degradation,
        })

    summary = fold_results.groupby(["symbol", "strategy"]).apply(_agg, include_groups=False).reset_index()
    return summary.sort_values("oos_score_mean", ascending=False).reset_index(drop=True)
