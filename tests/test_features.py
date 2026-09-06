import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.data.features import rsi, stochastic, aroon

NON_FEATURE_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "ticker"]

def _feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


@pytest.fixture(scope="module")
def processed_df() -> pd.DataFrame:
    # stock-predictor/data/processed/processed.csv
    PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "processed.csv"
    if not PROCESSED_PATH.exists():
        pytest.skip("processed.csv not found, please run features.py first")
    return pd.read_csv(PROCESSED_PATH)


def test_features_are_float64(processed_df: pd.DataFrame):
    """All engineered features must be dtype float64 (except ticker and target)"""
    cols = _feature_cols(processed_df)
    cols.remove("target")  # dont compare target
    assert all(processed_df[c].dtype == "float64" for c in cols)

def test_no_missing_values(processed_df: pd.DataFrame):
    """Feature columns must not have any NaN, None, etc."""
    cols = _feature_cols(processed_df)
    assert processed_df[cols].isna().sum().sum() == 0 

def test_no_infinites(processed_df: pd.DataFrame):
    """Feature columns must not have any infinites (division by zeros)"""
    cols = _feature_cols(processed_df)
    assert np.isinf(processed_df[cols]).sum().sum() == 0

def test_features_are_bounded(processed_df: pd.DataFrame):
    """Feature columns must be in range [0,100], inclusive"""
    cols = [ "stoch_k_14d", "stoch_d_14d", "rsi_14d",
            "aroon_up_25d", "aroon_down_25d"]
    assert processed_df[cols].ge(0).all().all()
    assert processed_df[cols].le(100).all().all()


def test_rsi_with_wilder_smooth():
    """Test RSI, and if Wilder's Smoothing's seed correctly uses SMA,
    and also correctly uses recursion thereafter, for one iteration
    """
    df_c = pd.DataFrame({
        "Close": [ 99,  97,  99, 101, 100, 101, 102, 104, 
                  106, 105, 105, 105, 106, 106, 104, 103]
    })

    results = rsi(df_c, 14)

    # SMA, and one iteration of recursion
    assert results.iloc[-2] == pytest.approx(64.7059)
    assert results.iloc[-1] == pytest.approx(60.8511)

def test_stochastic():
    """Test stochastic oscillators %K (first 3)
    and %D (first 1)"""
    df_hlc = pd.DataFrame({
        "High": [100, 100, 103, 103, 104, 105, 106, 108, 
                 107, 109, 108, 107, 109, 107, 107, 106],

        "Low": [ 99,  98, 101,  99, 103, 103, 104, 107, 
                106, 107, 107, 105, 106, 105, 106, 103],

        "Close": [100,  99, 101, 102, 103, 104, 106, 107, 
                  107, 108, 108, 106, 107, 105, 106, 105]
    })

    stoch_k, stoch_d = stochastic(df_hlc, 14, 3)

    # warmup must be NaN
    assert stoch_k.iloc[:13].isna().all()
    assert stoch_d.iloc[:15].isna().all()

    # test 3 %K
    np.testing.assert_allclose(
        stoch_k.iloc[13:],
        [63.6364, 72.7273, 60.0],
        atol=1e-4,
    )
    # and 1 %D
    assert stoch_d.iloc[-1] == pytest.approx(65.4545)

def test_aroon():
    """Test First 3 Aroon Up and Down"""
    df_hl = pd.DataFrame({
        "High": [101, 100, 103, 104, 103, 104, 109, 107, 107, 
                 109, 110, 106, 108, 106, 106, 105, 107, 106, 
                 107, 104, 107, 106, 104, 105, 106, 105, 103],

        "Low": [ 99,  99, 100, 102, 102, 102, 105, 104, 107, 
                107, 107, 105, 105, 102, 105, 103, 104, 103, 
                104, 103, 105, 104, 101, 101, 102, 102, 102]
    })

    aroon_up, aroon_down = aroon(df_hl, 25)

    np.testing.assert_allclose(
        aroon_up.iloc[-3:],
        [41.6667, 37.5, 33.3333],
        atol=1e-4,
    )

    np.testing.assert_allclose(
        aroon_down.iloc[-3:],
        [4.1667, 0.0, 0.0],
        atol=1e-4,
    )
