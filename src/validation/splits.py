import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_splits(df: pd.DataFrame, n_split: int = 5, gap: int = 25
                        ) -> list[tuple[pd.Index, pd.Index]]:
    """Returns a list of tuples of **positional indexes** of passed `df`, 
    in form [(train_indexes, test_indexes), ...], each tuple for each fold.
    This is a `TimeSeriesSplit`'s `.split()` wrapper."""

    tss = TimeSeriesSplit(n_splits=n_split, gap=gap)

    unique_dates: pd.Series = pd.to_datetime(
        np.sort(pd.to_datetime(df["Date"]).unique()))

    splits: list[tuple[pd.Index, pd.Index]] = []

    for train_unique_dates_idx, test_unique_dates_idx in tss.split(unique_dates):
        # find the dates corresponding to the index
        train_dates = unique_dates[train_unique_dates_idx]
        test_dates = unique_dates[test_unique_dates_idx]

        # find all the indexes of df, corresponding to the dates
        train_mask = df["Date"].isin(train_dates)
        test_mask = df["Date"].isin(test_dates)

        # return raw positional indexes, not labels (row index) already given
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        splits.append((train_idx, test_idx))

    return splits
