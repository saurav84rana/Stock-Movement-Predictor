import numpy as np
import pandas as pd
import pytest
from src.data.features import engineer_features

# --- create a synthetic mock OHLCV ---
def mock_ohlcv(n=100, seed=0) -> pd.DataFrame:
    """Make a mock valid OHLCV, of `n` days, with rng `seed`"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")

    # simulate close wandering around 100
    close_price = 100 + np.cumsum(rng.normal(0, 1, n))  
    open_price = close_price + rng.normal(0, 0.5, n)

    # so that high >= open or close, and low <= open or close
    high = np.maximum(open_price, close_price) + rng.uniform(0, 1, n)
    low = np.minimum(open_price, close_price) - rng.uniform(0, 1, n)
    
    volume = rng.integers(1_000_000, 5_000_000, n)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_price, 
        "High": high,
        "Low": low,
        "Close": close_price, 
        "Volume": volume
    }) 


# --- test cases ---
@pytest.fixture
def raw_df() -> pd.DataFrame:
    return mock_ohlcv()

def test_features_do_not_use_future_data(raw_df: pd.DataFrame):
    """Changing future market data, 
    must not change features for earlier dates"""
    original = engineer_features(raw_df.copy())
    cutoff = 60

    # mutate features after row 60
    mutated_raw = raw_df.copy()
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    mutated_raw.loc[cutoff:, ohlcv_cols] = (
        (mutated_raw.loc[cutoff:, ohlcv_cols] + 50) * 1000
    )  # add a constant, so features that are directly ratios are also affected
    mutated = engineer_features(mutated_raw)

    # join so it is easier to compare
    merged = original.merge(mutated, on="Date", suffixes=("_orig", "_mutated"))
    cutoff_date = raw_df["Date"].iloc[cutoff]
    before = merged[merged["Date"] < cutoff_date]

    feat_col = [c for c in original.columns if c not in ["Date", "target"]]
    for c in feat_col:
        pd.testing.assert_series_equal(
            before[f"{c}_orig"],
            before[f"{c}_mutated"],
            check_names=False
        )

def test_target_changes_after_cutoff(raw_df: pd.DataFrame):
    """Sanity check: changing future market data, 
    should change targets for earlier dates"""

    original = engineer_features(raw_df.copy())
    cutoff = 60

    # mutate features after row 60
    mutated_raw = raw_df.copy()
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    mutated_raw.loc[cutoff:, ohlcv_cols] = (
        mutated_raw.loc[cutoff:, ohlcv_cols] * 1000 + 50000
    )  # add a constant, so features that are directly ratios are also affected
    mutated = engineer_features(mutated_raw)

    merged = original.merge(mutated, on="Date", suffixes=("_orig", "_mutated"))

    assert not merged["target_orig"].equals(merged["target_mutated"])
    