"""Small, dependency-free technical indicator library (pandas/numpy only)."""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return lower, mid, upper


def donchian_channel(high: pd.Series, low: pd.Series, window: int = 20):
    upper = high.rolling(window, min_periods=window).max()
    lower = low.rolling(window, min_periods=window).min()
    return lower, upper
