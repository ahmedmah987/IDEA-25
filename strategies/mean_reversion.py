from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, clip_position
from strategies.indicators import bollinger_bands, rsi


class RSIStrategy(Strategy):
    """Buy when RSI drops below `oversold`, hold until it rises back above
    `overbought` (or `exit_level` if provided) — a classic mean-reversion bet
    that price snaps back after being oversold.
    """

    def __post_init__(self):
        self.name = "mean_reversion_rsi"
        self.param_space = {
            "window": (7, 30),
            "oversold": (15, 40),
            "overbought": (60, 85),
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        window = int(params["window"])
        oversold, overbought = float(params["oversold"]), float(params["overbought"])
        r = rsi(df["close"], window)

        # Stateful hold-until-exit logic, vectorized via a forward-filled
        # "regime" signal: +1 marks a fresh entry, -1 a fresh exit, then we
        # forward-fill the last signal to know what's held between events.
        entries = r < oversold
        exits = r > overbought
        signal = pd.Series(np.nan, index=df.index)
        signal[entries] = 1.0
        signal[exits] = 0.0
        position = signal.ffill().fillna(0.0)
        return clip_position(position)


class BollingerBandStrategy(Strategy):
    """Buy on a touch of the lower band, exit on a touch of the mid/upper band."""

    def __post_init__(self):
        self.name = "mean_reversion_bollinger"
        self.param_space = {
            "window": (10, 50),
            "n_std": (1.5, 3.0),
        }

    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        window = int(params["window"])
        n_std = float(params["n_std"])
        lower, mid, _upper = bollinger_bands(df["close"], window, n_std)

        entries = df["close"] <= lower
        exits = df["close"] >= mid
        signal = pd.Series(np.nan, index=df.index)
        signal[entries] = 1.0
        signal[exits] = 0.0
        position = signal.ffill().fillna(0.0)
        return clip_position(position)
