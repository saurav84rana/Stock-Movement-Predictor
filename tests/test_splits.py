import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.validation.splits import walk_forward_splits

@pytest.fixture()
def processed_df() -> pd.DataFrame:
    # stock-predictor/data/processed/processed.csv
    PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "processed.csv"
    if not PROCESSED_PATH.exists():
        pytest.skip("processed.csv not found, please run features.py first")
    return pd.read_csv(PROCESSED_PATH, parse_dates=["Date"])

def test_panel_dates_are_monotonically_increasing(processed_df: pd.DataFrame):
    """Sanity check: make sure all dates in processed.csv are monotonically increasing
    for walk_forward_split"""
    dates = pd.to_datetime(processed_df["Date"])
    assert dates.is_monotonic_increasing

def test_train_date_is_before_test_date(processed_df: pd.DataFrame):
    """Make sure the last (latest) training date, is always smaller (sooner),
    than the first (earliest) testing date"""
    splits = walk_forward_splits(processed_df, 5, 25)

    for train_idx, test_idx in splits:
        last_train_date = processed_df["Date"].iloc[train_idx].max()
        first_test_date = processed_df["Date"].iloc[test_idx].min()
        assert last_train_date < first_test_date

def test_n_gap_days_match_n_trading_days(processed_df: pd.DataFrame):
    """The gap (number of days) between last (latest) training date, 
    and the first (earliest) testing date, 
    should equal to number of trading days between them"""
    gap = 25
    splits = walk_forward_splits(processed_df, 5, gap)

    dates = pd.to_datetime(processed_df["Date"])
    unique_dates = pd.to_datetime(np.sort(dates.unique()))

    for train_idx, test_idx in splits:
        # dates is a series that has kept processed_df's rows as is
        # so that can be used directly for assertion
        last_train_date = dates.iloc[train_idx].max()
        first_test_date = dates.iloc[test_idx].min()

        train_pos = unique_dates.get_loc(last_train_date)
        test_pos = unique_dates.get_loc(first_test_date)

        assert test_pos - train_pos - 1 == gap
