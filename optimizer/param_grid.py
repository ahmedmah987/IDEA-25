"""Turn a strategy's param_space into concrete parameter combinations.

"An infinite number of strategies" isn't literal — it's a near-unbounded
combinatorial space (every parameter x every value x every symbol). This
module is how we actually cover a meaningful slice of it in finite time:

- `grid_combinations`: exhaustive enumeration. Fine for small/discrete
  spaces, explodes fast otherwise (e.g. 3 params x 20 values each = 8000
  combos *per symbol per strategy*).
- `random_combinations`: uniformly samples N points from the same space
  (including continuous ranges) — the practical way to cover a huge or
  continuous space with a fixed compute budget. This is what config.yaml's
  `optimizer.method: random` uses by default.
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Iterator


def _grid_values(spec: Any, n_grid_points: int = 8) -> list:
    """Expand one param_space entry into a finite list of candidate values."""
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, tuple):
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            step = max(1, (hi - lo) // n_grid_points)
            return list(range(lo, hi + 1, step))
        return [lo + (hi - lo) * i / (n_grid_points - 1) for i in range(n_grid_points)]
    raise TypeError(f"Unsupported param_space entry: {spec!r}")


def grid_combinations(param_space: dict[str, Any], n_grid_points: int = 8) -> Iterator[dict]:
    keys = list(param_space)
    value_lists = [_grid_values(param_space[k], n_grid_points) for k in keys]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def random_combinations(param_space: dict[str, Any], n: int, seed: int | None = None) -> Iterator[dict]:
    rng = random.Random(seed)
    keys = list(param_space)
    for _ in range(n):
        params = {}
        for k in keys:
            spec = param_space[k]
            if isinstance(spec, list):
                params[k] = rng.choice(spec)
            elif isinstance(spec, tuple):
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    params[k] = rng.randint(lo, hi)
                else:
                    params[k] = rng.uniform(lo, hi)
            else:
                raise TypeError(f"Unsupported param_space entry: {spec!r}")
        yield params
