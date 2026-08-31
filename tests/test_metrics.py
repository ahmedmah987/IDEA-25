import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult
from backtest.metrics import PLAUSIBLE_CAGR_CAP, compute_metrics, score


def _fake_result(equity_values, n_trades=5, trade_returns=None):
    idx = pd.date_range("2024-01-01", periods=len(equity_values), freq="h", tz="UTC")
    equity = pd.Series(equity_values, index=idx)
    returns = equity.pct_change().fillna(0.0)
    position = pd.Series(1.0, index=idx)
    return BacktestResult(
        equity_curve=equity, returns=returns, position=position,
        n_trades=n_trades, trade_returns=trade_returns or [0.05, -0.02, 0.03, 0.01, -0.01],
    )


def test_plausible_flag_true_for_reasonable_returns():
    # ~30% CAGR over a full year of hourly bars
    n = 24 * 365
    equity = 1000 * (1.3 ** (np.arange(n) / n))
    metrics = compute_metrics(_fake_result(equity), interval="1h", initial_capital=1000.0)
    assert metrics["plausible"] is True
    assert metrics["score"] > -999.0


def test_plausible_flag_false_for_absurd_compounding():
    # A tiny per-bar edge compounded over a year of hourly bars -> absurd CAGR,
    # exactly the failure mode this guard exists to catch.
    n = 24 * 365
    equity = 1000 * (1.0008 ** np.arange(n))
    metrics = compute_metrics(_fake_result(equity), interval="1h", initial_capital=1000.0)
    assert metrics["cagr"] > PLAUSIBLE_CAGR_CAP
    assert metrics["plausible"] is False
    # score is still finite/bounded (capped), not literally 1e10 -- it just won't
    # win a ranking against a real strategy the way an uncapped score silently would
    assert np.isfinite(metrics["score"])


def test_score_rejects_short_history_even_with_good_return():
    # 5 days of hourly bars with a strong return -- CAGR from annualizing 5 days
    # is meaningless regardless of how good the raw return looks.
    n = 24 * 5
    equity = np.linspace(1000, 1100, n)
    metrics = compute_metrics(_fake_result(equity), interval="1h", initial_capital=1000.0)
    assert metrics["score"] == -999.0


def test_score_rejects_too_few_trades():
    equity = np.linspace(1000, 2000, 24 * 200)
    metrics = compute_metrics(_fake_result(equity, n_trades=1, trade_returns=[0.5]), interval="1h", initial_capital=1000.0)
    assert metrics["score"] == -999.0
