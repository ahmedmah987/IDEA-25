from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, clip_position
from strategies.indicators import donchian_channel


class DonchianBreakoutStrategy(Strategy):
    """Buy when price breaks above the highest high of the last
    `entry_window` bars; exit when it breaks below the lowest low of the
    last `exit_window` bars (a trailing-stop-style exit). Classic
    trend-following breakout / turtle-trading style logic.
    """

    def __post_init__(self):
        self.name = "breakout_donchian"
        self.param_space = {
            "entry_window": (10, 100),
            "exit_window": (5, 60),
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        entry_w = int(params["entry_window"])
        exit_w = int(params["exit_window"])

        entry_lower, entry_upper = donchian_channel(df["high"], df["low"], entry_w)
        exit_lower, _exit_upper = donchian_channel(df["high"], df["low"], exit_w)

        # Breakout above yesterday's N-bar high triggers entry; breakdown
        # below the (shorter) M-bar low triggers exit. Shift by 1 so the
        # signal is based on the *prior* bar's channel, avoiding lookahead.
        breakout_up = df["close"] > entry_upper.shift(1)
        breakdown = df["close"] < exit_lower.shift(1)

        position = pd.Series(index=df.index, dtype=float)
        position[breakout_up] = 1.0
        position[breakdown] = 0.0
        position = position.ffill().fillna(0.0)
        return clip_position(position)
