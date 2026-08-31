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
| **1. Simulation Lab** | ✅ done | Fetch historical OHLCV, backtest strategies, sweep parameters, rank results |
| **1b. Walk-Forward Validation** | ✅ done | Re-optimize on rolling train windows, score on unseen test windows — catches overfit leaderboard entries |
| **2. Paper Trading** | ✅ done | Run one strategy against *live* Binance prices with fake money |
| **3. Live Bot** | 🔜 not built yet | Only after Phase 2 holds up for a meaningful stretch — real orders on Binance Spot, with hard risk limits |

Do not wire real API keys or real funds into anything here yet — Phase 3
hasn't been built, and Phase 2's results need time to accumulate before
that's a reasonable next step.

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

### Walk-forward validation (do this before trusting any leaderboard row)

```bash
python scripts/run_walk_forward.py --top 10
```

Re-optimizes each of the leaderboard's top 10 (symbol, strategy) pairs on
rolling 90-day train windows, then scores the winning parameters on the
following unseen 30-day test window. Read the printed `degradation_ratio`
and `pct_folds_profitable` before taking anything further — a strategy that
only looked good in-sample will show a collapsed or negative
`degradation_ratio` here. See `optimizer/walk_forward.py`'s module docstring
for how to read the numbers.

### Paper trading (Phase 2) — live prices, fake money

```bash
python scripts/run_paper_trading.py \
    --symbol BTCUSDT --strategy trend_ma_cross \
    --params '{"fast_window": 20, "slow_window": 80, "use_ema": true}' \
    --initial-capital 10 --once
```

Or point it at a walk-forward-validated candidate directly:

```bash
python scripts/run_paper_trading.py --symbol XRPUSDT --strategy breakout_donchian \
    --from-walk-forward reports/walk_forward_folds.csv --initial-capital 10 --once
```

Each `--once` run does one tick (fetch the latest closed candle, roll
virtual equity forward, rebalance if the strategy's signal changed) and
persists its state to `reports/paper_state_<symbol>_<strategy>.json` — safe
to stop and resume any time, nothing is lost between runs. Every tick is
also appended to `reports/paper_trading_ledger.csv` for a full audit trail.

**For real 24/7 operation**, schedule `--once` with cron (or a systemd
timer) at your chosen `--interval` on a machine that's actually always
on — a laptop that sleeps, or this sandboxed session, cannot provide that.
Example crontab entry for an hourly strategy:

```cron
0 * * * * cd /path/to/idea-25 && .venv/bin/python scripts/run_paper_trading.py \
    --symbol BTCUSDT --strategy trend_ma_cross --params '...' --initial-capital 10 --once \
    >> logs/paper_trading.log 2>&1
```

(Drop `--once` and pass `--poll-seconds` instead if you'd rather run it as
one long-lived loop in a terminal you keep open — same state file either
way.)

## Configuration

All defaults live in `config/config.yaml`: symbol universe, timeframe,
lookback window, trading fees/slippage assumptions, and sweep size. Every
value has a matching CLI flag on `scripts/run_simulation.py` that overrides
it for a single run.

## Roadmap

- [x] Phase 1: data fetcher, strategy library, vectorized backtest engine,
      parameter sweep + leaderboard
- [x] Walk-forward validation (train on rolling windows, verify on the next
      unseen one) so leaderboard results aren't just overfit to one period
- [x] Phase 2: paper-trading engine against live Binance prices, with
      persisted state for real unattended (cron-driven) operation
- [ ] Phase 3: live execution bot with hard position-size and daily-loss
      limits, only after Phase 2 results hold up over a meaningful stretch
      of real time (not just one favorable day)
