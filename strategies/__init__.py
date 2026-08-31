from strategies.base import Strategy
from strategies.breakout import DonchianBreakoutStrategy
from strategies.grid_dca import DCADipBuyerStrategy, GridTradingStrategy
from strategies.mean_reversion import BollingerBandStrategy, RSIStrategy
from strategies.trend_following import MACDStrategy, MACrossStrategy

# Central registry: name (as used in config.yaml) -> Strategy class.
REGISTRY: dict[str, type[Strategy]] = {
    "trend_ma_cross": MACrossStrategy,
    "trend_macd": MACDStrategy,
    "mean_reversion_rsi": RSIStrategy,
    "mean_reversion_bollinger": BollingerBandStrategy,
    "grid_trading": GridTradingStrategy,
    "dca_dip_buyer": DCADipBuyerStrategy,
    "breakout_donchian": DonchianBreakoutStrategy,
}


def build_strategy(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["Strategy", "REGISTRY", "build_strategy"]
