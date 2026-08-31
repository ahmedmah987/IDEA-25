# IDEA-25 — Binance Spot Strategy Simulation Lab

A framework for **screening a very large number of trading-strategy
variants** against real historical Binance spot data, before any of them
ever touches real money.

## ⚠️ Read this first

**No backtest, optimizer, or bot can guarantee daily profit.** Every
strategy here — no matter how well it scores on historical data — carries
real risk of loss, and past performance is not predictive of future
returns. What this project *can* honestly do is:

- Test thousands of strategy/parameter/symbol combinations against real
  historical price data in minutes.
- Rank them with risk-aware metrics (Sharpe ratio, max drawdown, win rate,
  profit factor) instead of raw return alone.
- Give you a reproducible, inspectable process for deciding what's even
  worth paper-trading — instead of guessing.

Treat every leaderboard this project produces as a **shortlist for further
scrutiny**, not a plan to deploy with real funds.

## What this is (and isn't) — 3 phases

| Phase | Status | What it does |
|---|---|---|
| **1. Simulation Lab** | ✅ this repo, today | Fetch historical OHLCV, backtest strategies, sweep parameters, rank results |
| **2. Paper Trading** | 🔜 not built yet | Run shortlisted strategies against *live* prices with fake money, to validate out-of-sample |
| **3. Live Bot** | 🔜 not built yet | Only after Phase 2 holds up — real orders on Binance Spot, with hard risk limits |

This repo is Phase 1 only. Do not wire real API keys or real funds into
anything here.

## Architecture

```
data/fetch_binance_data.py   Pull & cache OHLCV candles (no API key needed)
strategies/                  One file per strategy family, all behind the
                              same Strategy interface (base.py)
backtest/engine.py           Turns a position series into an equity curve
                              (fees + slippage charged on every position change)
backtest/metrics.py          Sharpe, CAGR, max drawdown, win rate, profit
                              factor, and a combined ranking `score`
optimizer/param_grid.py      Expands a strategy's parameter space into grid
                              or random-sampled combinations
optimizer/runner.py          Runs the full (symbol × strategy × params) sweep
                              in parallel, returns a ranked leaderboard
scripts/run_simulation.py    CLI entry point tying it all together
config/config.yaml           Default symbols, timeframe, fees, sweep size
```

### Why `data-api.binance.vision`?

Binance's main trading API (`api.binance.com`) geo-blocks some locations
even for public market data. `data-api.binance.vision` is Binance's
official, key-less, **market-data-only** mirror — same endpoints
(`/api/v3/klines`, `/api/v3/exchangeInfo`), no API key required, and not
subject to the same trading-eligibility geo-block. It's what `config.yaml`
points at by default. If you're somewhere `api.binance.com` works fine for
you too, you can swap `data.base_url` back to it.

### Why every strategy reduces to a 0–1 "position fraction"

Spot trading has no shorting, so every strategy — trend-following,
mean-reversion, grid trading, DCA, breakout — is expressed as *"what
fraction of capital should be in the base asset right now"*, a single float
per bar in `[0, 1]`. That one contract (`strategies/base.py`) is what lets
`backtest/engine.py` stay one small, strategy-agnostic simulator, and lets
the optimizer sweep every strategy family with the same code path.

Grid trading and DCA are inherently stateful (they track open levels /
accumulated buys), so those two run an explicit bar-by-bar loop over numpy
arrays internally — still fast (tens of thousands of bars in well under a
second) — while still exposing the same `generate_positions()` contract.

### Strategies included

| Name | Family | Idea |
|---|---|---|
| `trend_ma_cross` | Trend-following | Invested while fast MA > slow MA |
| `trend_macd` | Trend-following | Invested while MACD line > signal line |
| `mean_reversion_rsi` | Mean-reversion | Buy oversold RSI, exit on overbought |
| `mean_reversion_bollinger` | Mean-reversion | Buy lower Bollinger touch, exit at midline |
| `grid_trading` | Grid | Scale in as price falls through a rolling range, scale out as it rises |
| `dca_dip_buyer` | DCA | Buy fixed tranches on each further dip, sell all at a take-profit target |
| `breakout_donchian` | Breakout | Buy N-bar high breakout, exit on M-bar low breakdown (turtle-style) |

### "An infinite number of strategies" — how the sweep actually covers that

Each strategy declares a `param_space` (e.g. `trend_ma_cross`'s fast/slow MA
windows). `optimizer/param_grid.py` expands that into either:

- **`grid`** — every combination (exhaustive, explodes fast on wide ranges)
- **`random`** (default) — N uniformly sampled points from the same space,
  including continuous ranges — the practical way to cover a huge or
  effectively-infinite parameter space within a fixed compute budget

Increase `optimizer.n_random_samples` in `config.yaml` (or `--n-random-samples`)
to search harder; it scales linearly with runtime.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fast smoke run: 2 symbols, 30 days, sequential
python scripts/run_simulation.py --symbols BTCUSDT,ETHUSDT --lookback-days 30 --n-random-samples 20 --n-jobs 1

# A real sweep using config.yaml's defaults (5 symbols, 1 year, parallel)
python scripts/run_simulation.py

# Every USDT pair on Binance — slow, fetches + backtests hundreds of symbols
python scripts/run_simulation.py --symbols all
```

Results are written to `reports/leaderboard.csv` and the top rows printed
to the console, ranked by `score` (a blended Sharpe/CAGR/drawdown/profit-factor
metric — see `backtest/metrics.py:score`).

Run the test suite (no network needed, uses synthetic price data):

```bash
pip install pytest
python -m pytest tests/ -v
```

## Configuration

All defaults live in `config/config.yaml`: symbol universe, timeframe,
lookback window, trading fees/slippage assumptions, and sweep size. Every
value has a matching CLI flag on `scripts/run_simulation.py` that overrides
it for a single run.

## Roadmap

- [x] Phase 1: data fetcher, strategy library, vectorized backtest engine,
      parameter sweep + leaderboard
- [ ] Out-of-sample / walk-forward validation (train on one window, verify
      on the next) so leaderboard results aren't just overfit to one period
- [ ] Phase 2: paper-trading runner against Binance's live WebSocket feed
- [ ] Phase 3: live execution bot with hard position-size and daily-loss
      limits, only after Phase 2 results hold up
