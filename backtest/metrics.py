"""Performance metrics for a BacktestResult.

None of these promise future profit — they summarize what a strategy *would
have done* on historical data, which is the honest, bounded claim a backtest
can make. Use them to compare and rank candidates, not as a guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult

_BARS_PER_YEAR = {
    "1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4,
    "1h": 365 * 24, "4h": 365 * 6, "1d": 365,
}


def annualization_factor(interval: str) -> float:
    return float(_BARS_PER_YEAR.get(interval, 365 * 24))


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def sharpe_ratio(returns: pd.Series, interval: str, risk_free: float = 0.0) -> float:
    if returns.std() == 0 or returns.empty:
        return 0.0
    excess = returns - risk_free / annualization_factor(interval)
    return float(excess.mean() / returns.std() * np.sqrt(annualization_factor(interval)))


MIN_HISTORY_DAYS = 30  # below this, annualized CAGR is a statistical fantasy, not a signal

# Above this annualized return, treat the result as a backtest artifact, not
# a real edge. Idealized bar-close fills mean any strategy that rebalances
# frequently in small fractional slices (grid trading especially) can show
# a persistent tiny per-bar edge that compounds into an absurd annualized
# number purely from compounding frequency (365*24 = 8760 independent hourly
# compounding periods/year) -- e.g. a genuinely modest +0.05%/bar edge alone
# compounds to ~7900% CAGR. Real trading has spread, latency, partial fills
# and liquidity limits this backtest doesn't model, which would crush that.
# 5000%/yr is already an extremely generous cap; anything above it needs a
# realistic execution model (or at minimum heavy live/paper-trading scrutiny)
# before it means anything -- see README > "Known limitation: grid trading".
PLAUSIBLE_CAGR_CAP = 50.0


def compute_metrics(result: BacktestResult, interval: str, initial_capital: float) -> dict:
    equity = result.equity_curve
    total_return = float(equity.iloc[-1] / initial_capital - 1.0) if len(equity) else 0.0
    n_bars = len(equity)
    years = n_bars / annualization_factor(interval) if n_bars else 0.0
    cagr = float((equity.iloc[-1] / initial_capital) ** (1 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else 0.0

    wins = [r for r in result.trade_returns if r > 0]
    losses = [r for r in result.trade_returns if r <= 0]
    win_rate = len(wins) / len(result.trade_returns) if result.trade_returns else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    mdd = max_drawdown(equity)
    sharpe = sharpe_ratio(result.returns, interval)

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": result.n_trades,
        "n_bars": n_bars,
    }
    metrics["plausible"] = bool(np.isfinite(cagr) and cagr <= PLAUSIBLE_CAGR_CAP)
    metrics["score"] = score(metrics, interval)
    return metrics


def score(metrics: dict, interval: str = "1h") -> float:
    """Single ranking number balancing return against risk and trade
    frequency. Deliberately punishes: near-zero trades (overfit/no signal),
    too little history to trust an annualized number, deep drawdowns, and
    negative Sharpe. `cagr` is capped for scoring purposes only (the
    uncapped value is still reported) — a legitimate edge doesn't need a
    triple-digit-thousand-percent number to rank at the top, and capping it
    stops any one pathological result (a strategy bug, a data glitch, an
    illiquid/low-precision symbol) from dominating a leaderboard the way an
    uncapped one silently would. Tune the weights to taste — this is a
    starting heuristic, not a magic formula.
    """
    if metrics["n_trades"] < 3:
        return -999.0  # too few trades to trust the other numbers
    min_bars = MIN_HISTORY_DAYS * annualization_factor(interval) / 365
    if metrics.get("n_bars", float("inf")) < min_bars:
        return -999.0  # too little history for CAGR to mean anything
    dd_penalty = 1.0 / (1.0 + abs(metrics["max_drawdown"]) * 5)
    pf = min(metrics["profit_factor"], 5.0) if np.isfinite(metrics["profit_factor"]) else 5.0
    capped_cagr = min(metrics["cagr"], 5.0) if np.isfinite(metrics["cagr"]) else 5.0
    capped_sharpe = min(metrics["sharpe"], 10.0) if np.isfinite(metrics["sharpe"]) else 10.0
    return float(capped_sharpe * 0.5 + capped_cagr * 2.0 + pf * 0.3) * dd_penalty
