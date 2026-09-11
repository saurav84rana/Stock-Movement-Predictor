import numpy as np
import pandas as pd
import pytest
from src.validation.baselines import always_up, buy_and_hold

def test_always_up():
    n = 5
    assert len(always_up(n)) == n

def test_buy_and_hold():
    dates = pd.date_range(start="2018-01-01", end="2018-01-03").to_list()
    dates = pd.Series(dates*2)
    tickers = ["AAA"] * 3 + ["BBB"] * 3

    closes = [ 
        # ticker AAA
        98, 100, 101,
        # ticker BBB
        102, 100, 101
    ]
    
    df = pd.DataFrame({
        "Close": closes,
        "Date": dates,
        "ticker": tickers
    })

    assert buy_and_hold(df, dates.iloc[0], dates.iloc[-1]) == pytest.approx(0.0104, abs=1e-4)
