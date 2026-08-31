"""Vectorized single-symbol backtest engine.

Every strategy (see strategies/) reduces to a target-position-fraction
Series. This engine turns that into an equity curve, charging trading fees
and slippage whenever the position changes, and never letting a bar's
decision use information from that same bar's close (positions are shifted
one bar forward before being applied to returns).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series          # per-bar strategy returns, net of costs
    position: pd.Series         # the (shifted) position actually held each bar
    n_trades: int
    trade_returns: list[float]  # % return of each completed round-trip trade


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    initial_capital: float = 1000.0,
    taker_fee: float = 0.001,
    slippage: float = 0.0005,
) -> BacktestResult:
    """Simulate holding `position` (fraction of capital in the base asset)
    against `df`'s close prices.

    Costs: every time the position changes we pay `taker_fee + slippage` on
    the *change in exposure* — i.e. moving from 0.3 to 0.8 costs as much as
    a single 0.5-sized trade, which is the right approximation for a spot
    market/limit order sized to hit the new target exposure.
    """
    if len(df) != len(position):
        raise ValueError("df and position must be the same length")

    close = df["close"]
    bar_returns = close.pct_change().fillna(0.0)

    # Decide on bar i using data through bar i, but only *earn* that decision's
    # return starting bar i+1 — i.e. execute at the next bar's open/close.
    held_position = position.shift(1).fillna(0.0)

    position_change = held_position.diff().abs()
    position_change.iloc[0] = held_position.iloc[0]  # entering from flat also costs
    trading_cost = position_change * (taker_fee + slippage)

    strategy_returns = held_position * bar_returns - trading_cost
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    n_trades, trade_returns = _extract_trades(held_position, close)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=strategy_returns,
        position=held_position,
        n_trades=n_trades,
        trade_returns=trade_returns,
    )


def _extract_trades(position: pd.Series, close: pd.Series) -> tuple[int, list[float]]:
    """Identify discrete flat->invested->flat round trips and their % return,
    used for win-rate / profit-factor style metrics. Partial position changes
    (e.g. grid/DCA scaling) are treated as one trade spanning from the first
    bar position leaves 0 to the next bar it returns to 0.
    """
    in_position = position > 0
    entries = in_position & ~in_position.shift(1, fill_value=False)
    exits = ~in_position & in_position.shift(1, fill_value=False)

    entry_idx = np.flatnonzero(entries.to_numpy())
    exit_idx = np.flatnonzero(exits.to_numpy())

    trade_returns: list[float] = []
    close_vals = close.to_numpy()
    for e_start in entry_idx:
        later_exits = exit_idx[exit_idx > e_start]
        e_end = later_exits[0] if len(later_exits) else len(close_vals) - 1
        entry_price = close_vals[e_start]
        exit_price = close_vals[e_end]
        if entry_price > 0:
            trade_returns.append(exit_price / entry_price - 1.0)

    return len(trade_returns), trade_returns
