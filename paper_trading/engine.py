"""Phase 2: paper trading.

Same math as backtest/engine.py (equity moves by held_position * bar_return,
minus taker_fee + slippage on every position change), but driven one *live*
closed candle at a time instead of a full historical DataFrame — so a
strategy validated in Phase 1 can be watched against real, unfolding prices
with fake money before Phase 3 ever risks real funds.

This module has no built-in scheduler. `run_tick()` does one unit of work
(fetch the latest closed candle, update the portfolio, persist state) and
returns. Driving it in a loop (`scripts/run_paper_trading.py`, default mode)
works for a session you keep open; for genuine 24/7 unattended operation,
run it with `--once` from cron / a systemd timer instead — state is
JSON-persisted between calls, so nothing is lost between ticks either way.
See the README's Phase 2 section for why a suspend/resume container can't
itself provide "24/7".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from data.fetch_binance_data import DEFAULT_BASE_URL, fetch_latest_klines
from strategies import build_strategy


@dataclass
class PortfolioState:
    symbol: str
    strategy: str
    params: dict[str, Any]
    interval: str
    initial_capital: float
    equity: float
    held_position: float = 0.0
    last_price: float | None = None
    last_candle_open_ms: int | None = None
    started_at: str = ""
    updated_at: str = ""
    n_trades: int = 0

    @classmethod
    def new(cls, symbol: str, strategy: str, params: dict, interval: str, initial_capital: float) -> "PortfolioState":
        now = pd.Timestamp.now(tz="UTC").isoformat()
        return cls(
            symbol=symbol, strategy=strategy, params=params, interval=interval,
            initial_capital=initial_capital, equity=initial_capital,
            started_at=now, updated_at=now,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PortfolioState":
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def _append_ledger(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_tick(
    state: PortfolioState,
    warmup_bars: int = 600,
    taker_fee: float = 0.001,
    slippage: float = 0.0005,
    base_url: str = DEFAULT_BASE_URL,
    ledger_path: str | Path = "reports/paper_trading_ledger.csv",
) -> tuple[PortfolioState, dict | None]:
    """One unit of work: pull the latest closed candle, roll equity forward
    by whatever happened since the last tick, recompute the strategy's
    signal, rebalance if it changed. Returns (updated_state, trade_or_None).

    NOTE: if this is called less often than one `interval` (e.g. polling
    every 10 minutes against a 1h interval), only the *latest* closed candle
    is acted on — any candles that closed in between are skipped. Poll at
    least as often as `interval` to avoid missing signals.
    """
    df = fetch_latest_klines(state.symbol, interval=state.interval, limit=warmup_bars, base_url=base_url)
    if df.empty:
        return state, None

    now = pd.Timestamp.now(tz="UTC")
    closed = df[df["close_time"] <= now]
    if closed.empty:
        return state, None  # no fully-closed candle available yet

    latest = closed.iloc[-1]
    latest_open_ms = int(closed.index[-1].timestamp() * 1000)
    if state.last_candle_open_ms is not None and latest_open_ms <= state.last_candle_open_ms:
        return state, None  # already processed this candle, nothing new

    new_price = float(latest["close"])
    strategy = build_strategy(state.strategy)
    positions = strategy.generate_positions(closed.drop(columns=["close_time"]), state.params)
    new_target = float(positions.iloc[-1])

    if state.last_price is not None:
        bar_return = new_price / state.last_price - 1.0
        state.equity *= (1 + state.held_position * bar_return)

    trade = None
    if abs(new_target - state.held_position) > 1e-9:
        cost_fraction = abs(new_target - state.held_position) * (taker_fee + slippage)
        equity_before = state.equity
        state.equity *= (1 - cost_fraction)
        trade = {
            "time": now.isoformat(), "symbol": state.symbol, "strategy": state.strategy,
            "price": new_price, "from_position": state.held_position, "to_position": new_target,
            "cost_paid": equity_before - state.equity, "equity_after": state.equity,
        }
        state.n_trades += 1

    state.held_position = new_target
    state.last_price = new_price
    state.last_candle_open_ms = latest_open_ms
    state.updated_at = now.isoformat()

    _append_ledger(ledger_path, {
        "time": now.isoformat(), "candle_open": closed.index[-1].isoformat(),
        "symbol": state.symbol, "strategy": state.strategy, "price": new_price,
        "target_position": new_target, "equity": state.equity,
        "traded": trade is not None,
    })

    return state, trade
