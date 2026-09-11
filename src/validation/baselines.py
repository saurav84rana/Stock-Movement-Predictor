import numpy as np
import pandas as pd

def always_up(n: int) -> np.ndarray:
    """Return an ndarray of n ones, i.e predict 1 (up) for every observation"""
    return np.ones(n, dtype=int)

def buy_and_hold(df: pd.DataFrame, start_date, end_date) -> float:
    """Return equal weighted, buy and hold return for duration 
    `start_date` to `end_date`, across all tickers"""
    # get ["ticker", "Close"] for start_date and end_date
    start: pd.DataFrame = df[df["Date"] == pd.to_datetime(start_date)][["ticker", "Close"]]
    end: pd.DataFrame = df[df["Date"] == pd.to_datetime(end_date)][["ticker", "Close"]]

    price = start.merge(
        end, 
        on="ticker", 
        suffixes=("_start", "_end")
    )
    returns = price["Close_end"] / price["Close_start"] - 1

    return returns.mean()