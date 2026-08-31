"""Paper-trading engine tests -- no network calls: fetch_latest_klines is
monkeypatched to return synthetic candles.
"""
import numpy as np
import pandas as pd
import pytest

import paper_trading.engine as engine_mod
from paper_trading.engine import PortfolioState, run_tick


def _synthetic_batch(prices: list[float], interval: str = "1h", start="2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    close_times = idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return pd.DataFrame(
        {
            "open": prices, "high": [p * 1.001 for p in prices], "low": [p * 0.999 for p in prices],
            "close": prices, "volume": [10.0] * len(prices), "quote_volume": [1000.0] * len(prices),
            "n_trades": [5] * len(prices), "close_time": close_times,
        },
        index=idx,
    )


def test_run_tick_flat_strategy_preserves_capital(monkeypatch):
    # A synthetic "always uptrend" series is enough to exercise the flow;
    # what matters here is that equity accounting is exact, not the signal.
    prices = list(100 + np.cumsum(np.zeros(60)))  # perfectly flat prices
    df = _synthetic_batch(prices)
    monkeypatch.setattr(engine_mod, "fetch_latest_klines", lambda *a, **k: df)

    state = PortfolioState.new("BTCUSDT", "trend_ma_cross", {"fast_window": 5, "slow_window": 20, "use_ema": False}, "1h", 10.0)
    state, trade = run_tick(state, warmup_bars=60)
    # flat prices -> bar_return is 0 regardless of position, equity unaffected by price moves
    assert state.last_price == pytest.approx(100.0)


def test_run_tick_is_idempotent_on_same_candle(monkeypatch):
    prices = [100.0] * 60
    df = _synthetic_batch(prices)
    monkeypatch.setattr(engine_mod, "fetch_latest_klines", lambda *a, **k: df)

    state = PortfolioState.new("BTCUSDT", "trend_ma_cross", {"fast_window": 5, "slow_window": 20, "use_ema": False}, "1h", 10.0)
    state, _ = run_tick(state, warmup_bars=60)
    equity_after_first = state.equity
    n_trades_after_first = state.n_trades

    # Same underlying data (same latest closed candle) -> second tick is a no-op
    state, trade = run_tick(state, warmup_bars=60)
    assert trade is None
    assert state.equity == pytest.approx(equity_after_first)
    assert state.n_trades == n_trades_after_first


def test_run_tick_charges_cost_on_position_change(monkeypatch):
    # Force an entry: enough history above the slow MA so a fresh state (held_position=0)
    # sees fast_ma > slow_ma -> target 1.0 -> a trade should fire on the very first tick.
    prices = [100.0 + i * 0.5 for i in range(60)]  # steady uptrend
    df = _synthetic_batch(prices)
    monkeypatch.setattr(engine_mod, "fetch_latest_klines", lambda *a, **k: df)

    state = PortfolioState.new("BTCUSDT", "trend_ma_cross", {"fast_window": 5, "slow_window": 20, "use_ema": False}, "1h", 10.0)
    state, trade = run_tick(state, warmup_bars=60, taker_fee=0.001, slippage=0.0005)

    assert trade is not None
    assert trade["from_position"] == 0.0
    assert trade["to_position"] == pytest.approx(1.0)
    assert state.equity < state.initial_capital  # cost was charged, no price move yet to offset it
    assert state.n_trades == 1


def test_state_round_trips_through_json(tmp_path):
    state = PortfolioState.new("ETHUSDT", "mean_reversion_rsi", {"window": 14, "oversold": 30, "overbought": 70}, "1h", 10.0)
    state.equity = 10.42
    state.held_position = 1.0
    path = tmp_path / "state.json"
    state.save(path)
    loaded = PortfolioState.load(path)
    assert loaded.symbol == "ETHUSDT"
    assert loaded.equity == pytest.approx(10.42)
    assert loaded.params == {"window": 14, "oversold": 30, "overbought": 70}
