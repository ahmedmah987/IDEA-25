from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, clip_position
from strategies.indicators import ema, macd, sma


class MACrossStrategy(Strategy):
    """Classic moving-average crossover: fully invested while the fast MA is
    above the slow MA, flat otherwise.
    """

    def __post_init__(self):
        self.name = "trend_ma_cross"
        self.param_space = {
            "fast_window": (5, 50),      # int range for random search
            "slow_window": (20, 200),
            "use_ema": [True, False],     # grid-style categorical choice
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        fast_w, slow_w = int(params["fast_window"]), int(params["slow_window"])
        if fast_w >= slow_w:
            fast_w, slow_w = min(fast_w, slow_w - 1) if slow_w > 1 else 1, max(slow_w, fast_w + 1)
        ma_fn = ema if params.get("use_ema") else sma
        fast_ma = ma_fn(df["close"], fast_w)
        slow_ma = ma_fn(df["close"], slow_w)
        position = (fast_ma > slow_ma).astype(float)
        return clip_position(position)


class MACDStrategy(Strategy):
    """Invested while the MACD line is above its signal line."""

    def __post_init__(self):
        self.name = "trend_macd"
        self.param_space = {
            "fast": (5, 20),
            "slow": (21, 60),
            "signal": (5, 20),
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        fast, slow, signal = int(params["fast"]), int(params["slow"]), int(params["signal"])
        if fast >= slow:
            fast, slow = min(fast, slow - 1) if slow > 1 else 1, max(slow, fast + 1)
        macd_line, signal_line = macd(df["close"], fast=fast, slow=slow, signal=signal)
        position = (macd_line > signal_line).astype(float)
        return clip_position(position)
