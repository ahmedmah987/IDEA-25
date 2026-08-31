"""Grid trading and DCA (dollar-cost averaging).

Both are naturally *stateful, event-driven* strategies (track open grid
levels / accumulated buys), which is harder to vectorize than a pure
indicator crossover. To keep every strategy behind the same fast, vectorized
`generate_positions` contract (so the optimizer can screen thousands of
combinations quickly), these implementations run one explicit bar-by-bar
loop each over numpy arrays — no pandas per-row overhead, still fast enough
for a single symbol/timeframe backtest (tens of thousands of bars execute in
well under a second).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, clip_position


class GridTradingStrategy(Strategy):
    """Divide a rolling price range into `n_levels` grid lines. Each time
    price crosses a level downward, buy one grid-sized slice; each time it
    crosses a level upward while holding that slice, sell it. Net effect:
    position scales up as price falls through the range and back down as it
    recovers — classic grid/"buy low sell high in a channel" behavior.
    """

    def __post_init__(self):
        self.name = "grid_trading"
        self.param_space = {
            "range_window": (48, 480),   # bars used to define the rolling grid range
            "n_levels": (4, 20),
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        range_window = int(params["range_window"])
        n_levels = max(2, int(params["n_levels"]))

        close = df["close"].to_numpy()
        roll_low = df["close"].rolling(range_window, min_periods=range_window).min().to_numpy()
        roll_high = df["close"].rolling(range_window, min_periods=range_window).max().to_numpy()

        n = len(close)
        position = np.zeros(n)
        held_levels = 0  # how many grid slices currently held, out of n_levels

        for i in range(n):
            if i == 0 or np.isnan(roll_low[i]) or np.isnan(roll_high[i]):
                position[i] = position[i - 1] if i > 0 else 0.0
                continue
            span = roll_high[i] - roll_low[i]
            if span <= 0:
                position[i] = position[i - 1]
                continue
            # Which grid line (0..n_levels) is price at right now?
            level = int(np.clip((roll_high[i] - close[i]) / span * n_levels, 0, n_levels))
            held_levels = level  # snap directly to the target level (buy dips, sell rallies)
            position[i] = held_levels / n_levels

        return clip_position(pd.Series(position, index=df.index))


class DCADipBuyerStrategy(Strategy):
    """Scale into a position with fixed-size buys every time price drops
    `dip_pct` from the last buy price, up to `max_buys` tranches. Sell the
    entire position (take profit) once it's up `take_profit_pct` from the
    average entry price. Resets and starts scaling in again after each exit.
    """

    def __post_init__(self):
        self.name = "dca_dip_buyer"
        self.param_space = {
            "dip_pct": (0.01, 0.08),          # buy another tranche after this much drop
            "take_profit_pct": (0.02, 0.15),   # sell everything after this much gain
            "max_buys": (2, 10),
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        dip_pct = float(params["dip_pct"])
        tp_pct = float(params["take_profit_pct"])
        max_buys = max(1, int(params["max_buys"]))

        close = df["close"].to_numpy()
        n = len(close)
        position = np.zeros(n)

        tranches_held = 0
        last_buy_price = None
        avg_entry = 0.0

        for i in range(n):
            price = close[i]
            if tranches_held == 0:
                # Not in a position yet: start (or restart) accumulation.
                tranches_held = 1
                last_buy_price = price
                avg_entry = price
            else:
                # Check take-profit first.
                if price >= avg_entry * (1 + tp_pct):
                    tranches_held = 0
                    last_buy_price = None
                    avg_entry = 0.0
                elif tranches_held < max_buys and price <= last_buy_price * (1 - dip_pct):
                    avg_entry = (avg_entry * tranches_held + price) / (tranches_held + 1)
                    tranches_held += 1
                    last_buy_price = price

            position[i] = tranches_held / max_buys

        return clip_position(pd.Series(position, index=df.index))
