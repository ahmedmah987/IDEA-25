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
    """A static ladder of `n_levels` price lines spanning the trailing
    `range_window`-bar high/low, recomputed periodically (every
    range_window // 4 bars) rather than every single bar. Each level is
    bought (once) the first time price dips to or below it, and sold (once)
    only after price recovers a full grid step above it — position scales
    up as price falls through the ladder, back down as it climbs out.

    Earlier version note: a prior implementation recomputed the whole grid
    from the *rolling* high/low and snapped the position directly to
    "wherever price sits in that range" on every single bar. That let the
    backtest capture the full round-trip of every tiny price wiggle with
    perfect timing every bar, compounding into absurd (billions-of-percent)
    backtested returns that no real order-fill process could ever achieve.
    This version only trades a level on an actual crossing event, and holds
    a level between crossings — the grid is a real, temporarily-fixed
    ladder, not an oracle repriced every bar.
    """

    def __post_init__(self):
        self.name = "grid_trading"
        self.param_space = {
            "range_window": (48, 480),   # bars used to define each grid ladder
            "n_levels": (4, 20),
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        range_window = int(params["range_window"])
        n_levels = max(2, int(params["n_levels"]))
        recalc_every = max(1, range_window // 4)

        close = df["close"].to_numpy()
        n = len(close)
        position = np.zeros(n)

        grid_low = grid_high = None
        levels_bought = np.zeros(n_levels, dtype=bool)
        next_recalc_at = range_window  # first bar with a full range_window of history

        for i in range(n):
            if i >= next_recalc_at:
                window = close[max(0, i - range_window):i]
                grid_low, grid_high = float(window.min()), float(window.max())
                next_recalc_at = i + recalc_every

            if grid_low is None or grid_high <= grid_low:
                position[i] = position[i - 1] if i > 0 else 0.0
                continue

            price = close[i]
            step = (grid_high - grid_low) / n_levels
            for lvl in range(n_levels):
                level_price = grid_high - step * (lvl + 1)  # levels 0..n_levels-1, top to bottom
                if not levels_bought[lvl] and price <= level_price:
                    levels_bought[lvl] = True
                elif levels_bought[lvl] and price > level_price + step:
                    levels_bought[lvl] = False  # take profit one grid step above where it was bought

            position[i] = levels_bought.sum() / n_levels

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
