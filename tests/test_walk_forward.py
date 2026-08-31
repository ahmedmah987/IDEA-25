import numpy as np
import pandas as pd

from optimizer.walk_forward import WalkForwardConfig, generate_folds, run_walk_forward, summarize


def _make_synthetic_df(n=24 * 200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0001, scale=0.008, size=n)
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close * (1 - 0.0005), "high": close * (1 + 0.002),
            "low": close * (1 - 0.002), "close": close,
            "volume": rng.uniform(10, 100, size=n),
        },
        index=idx,
    )


def test_generate_folds_are_chronological_and_non_overlapping_in_test():
    df = _make_synthetic_df()
    cfg = WalkForwardConfig(train_days=90, test_days=30)
    folds = generate_folds(df, cfg)
    assert len(folds) >= 2
    for train, test in folds:
        assert train.index.max() < test.index.min()
        assert len(train) == cfg.train_days * 24
        assert len(test) == cfg.test_days * 24


def test_run_walk_forward_produces_one_row_per_fold():
    df = _make_synthetic_df()
    cfg = WalkForwardConfig(train_days=90, test_days=30, n_random_samples=10)
    result = run_walk_forward(df, "SYNTH", "trend_ma_cross", cfg)
    expected_folds = len(generate_folds(df, cfg))
    assert len(result) == expected_folds
    assert {"is_score", "oos_score", "oos_cagr", "params"}.issubset(result.columns)


def test_summarize_computes_degradation_ratio():
    df = _make_synthetic_df()
    cfg = WalkForwardConfig(train_days=90, test_days=30, n_random_samples=10)
    fold_results = run_walk_forward(df, "SYNTH", "mean_reversion_rsi", cfg)
    summary = summarize(fold_results)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["symbol"] == "SYNTH"
    assert row["strategy"] == "mean_reversion_rsi"
    assert np.isfinite(row["degradation_ratio"]) or row["is_score_mean"] == 0
