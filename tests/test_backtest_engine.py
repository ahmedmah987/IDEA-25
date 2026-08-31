"""Sanity tests that don't need network access — synthetic price data only."""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, max_drawdown
from strategies import REGISTRY, build_strategy


def _make_synthetic_df(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0002, scale=0.01, size=n)
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": close * (1 - 0.0005),
            "high": close * (1 + 0.002),
            "low": close * (1 - 0.002),
            "close": close,
            "volume": rng.uniform(10, 100, size=n),
        },
        index=idx,
    )
    return df


def test_always_flat_preserves_capital():
    df = _make_synthetic_df()
    flat = pd.Series(0.0, index=df.index)
    result = run_backtest(df, flat, initial_capital=1000.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(1000.0)
    assert result.n_trades == 0


def test_always_invested_matches_buy_and_hold_minus_one_fee():
    df = _make_synthetic_df()
    full = pd.Series(1.0, index=df.index)
    result = run_backtest(df, full, initial_capital=1000.0, taker_fee=0.0, slippage=0.0)
    buy_hold_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
    strategy_return = result.equity_curve.iloc[-1] / 1000.0 - 1.0
    # position is shifted by 1 bar, so it lags buy-and-hold by exactly one bar's return
    assert strategy_return == pytest.approx(buy_hold_return, abs=0.02)


def test_max_drawdown_is_nonpositive():
    df = _make_synthetic_df()
    full = pd.Series(1.0, index=df.index)
    result = run_backtest(df, full)
    assert max_drawdown(result.equity_curve) <= 0.0


@pytest.mark.parametrize("name", list(REGISTRY))
def test_every_registered_strategy_runs_end_to_end(name):
    df = _make_synthetic_df(n=400, seed=hash(name) % 1000)
    strategy = build_strategy(name)
    params = strategy.default_params()
    position = strategy.generate_positions(df, params)

    assert len(position) == len(df)
    assert position.between(0.0, 1.0).all()

    result = run_backtest(df, position)
    metrics = compute_metrics(result, interval="1h", initial_capital=1000.0)
    assert np.isfinite(metrics["max_drawdown"])
    assert metrics["max_drawdown"] <= 0.0
