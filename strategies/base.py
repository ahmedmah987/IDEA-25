"""Common interface every strategy implements.

A strategy turns an OHLCV DataFrame + a dict of parameters into a *target
position fraction* series aligned to the same index: 0.0 = fully in cash/USDT,
1.0 = fully in the base asset. Spot trading only, so values are clipped to
[0, 1] — no shorting/leverage.

Keeping every strategy family (trend, mean-reversion, grid, DCA, breakout)
behind this one contract is what lets backtest/engine.py stay a single,
vectorized, strategy-agnostic simulator, and what lets the optimizer sweep
thousands of (strategy x symbol x parameter) combinations with the same code
path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Strategy(ABC):
    """Subclass and implement `generate_positions`.

    `param_space` describes the search space the optimizer draws from:
    each value is a list of candidates (grid search) or a (low, high) tuple
    of numbers (random search draws uniformly, int or float depending on
    the tuple's type).
    """

    name: str = field(init=False)
    param_space: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Subclasses override this to set self.name / self.param_space.

        Defining this stub here (even as a no-op) is required so that
        @dataclass's generated __init__ actually calls self.__post_init__()
        at all — it only wires that call in when __post_init__ exists on the
        class being decorated, not on subclasses defined later.
        """

    @abstractmethod
    def generate_positions(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Return a float Series in [0, 1], same index as df, one value per bar."""
        raise NotImplementedError

    def default_params(self) -> dict:
        """A single representative point from param_space, used for smoke tests."""
        out = {}
        for k, v in self.param_space.items():
            if isinstance(v, tuple):
                lo, hi = v
                out[k] = (lo + hi) / 2 if isinstance(lo, float) else (lo + hi) // 2
            else:
                out[k] = v[len(v) // 2]
        return out


def clip_position(pos: pd.Series) -> pd.Series:
    return pos.clip(lower=0.0, upper=1.0).fillna(0.0)
