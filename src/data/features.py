import numpy as np
import pandas as pd
from pathlib import Path

# --- Feature engineering helpers ---
def _wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing, seeded with SMA of first `window` values"""
    result = series.copy().dropna()  # RSI's diff()'s 1st value is NaN

    seed = result.iloc[:window].mean()
    result.iloc[:window - 1] = np.nan
    result.iloc[window - 1] = seed

    result.iloc[window - 1:] = (
        result.iloc[window - 1:]
        .ewm(alpha=1/window, adjust=False)  # use recursive version
        .mean()
    )
    # result reindexed so that result.iloc[i] = series.iloc[i]
    return result.reindex(series.index)

def atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Returns Average True Range, as a percentage of close (ATR%) 
    of lookback period `window`"""
    # true range = max(high - low, |high - prev_close|, |low - prev_close|)
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_raw = _wilder_smooth(true_range, window)
    return 100 * atr_raw / df["Close"]

def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Returns Relative Strength Index of lookback period `window`"""
    delta = df["Close"].diff()
    gains = delta.clip(lower=0) 
    losses = -delta.clip(upper=0) # negatives treated as positive for adding

    avg_gain = _wilder_smooth(gains, window)
    avg_loss = _wilder_smooth(losses, window)

    rs = avg_gain / avg_loss
    result = 100 - (100 / (1 + rs))

    # to replace any NaN from 0/0, treat as rsi = 50 instead
    flat = (avg_gain == 0) & (avg_loss == 0)
    result[flat] = 50.0

    return result

def stochastic(df: pd.DataFrame, window_k: int = 14, window_d: int = 3
    ) -> tuple[pd.Series, pd.Series]:
    """Returns Stochastic %K (fast, `window_k` day lookback period), 
    and %D (slow, `window_d` day period SMA of %K)"""
    lowest_win = df["Low"].rolling(window_k).min()
    highest_win = df["High"].rolling(window_k).max()

    diff = highest_win - lowest_win
    stoch_k = 100 * (df["Close"] - lowest_win) / diff

    # replace any division by zeros with %K = 50
    flat = (diff == 0)
    stoch_k[flat] = 50.0

    stoch_d = stoch_k.rolling(window_d).mean()
    return stoch_k, stoch_d

def aroon(df: pd.DataFrame, window: int = 25) -> tuple[pd.Series, pd.Series]:
    """Returns Aroon Up and Aroon Down of lookback period `window`"""
    # the index of extreme from reversed list (newest to oldest) 
    # is the same as period since last extreme, picking most recent one
    period_since_high = df["High"].rolling(window).apply(
        lambda x: np.argmax(x[::-1]), 
        raw=True
    )
    period_since_low = df["Low"].rolling(window).apply(
        lambda x: np.argmin(x[::-1]),
        raw=True
    )
    up = 100 * (window - 1 - period_since_high) / (window - 1)
    down = 100 * (window - 1 - period_since_low) / (window - 1)
    return up, down


# --- feature engineering ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and a prediction target to `df`:

    ROC, VROC, Disparity Index (MA's ratio), rolling volatility,
    ATR%, RSI, Stochastic %K and %D, Aroon Up and Aroon Down,
    and target
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # over each windows, calculate: ROC, VROC, Disparity Index, rolling volatility
    window = [1, 5, 10, 20]
    for k in window:
        df[f"return_{k}d"] = df["Close"].pct_change(k)          # ROC
        df[f"volume_change_{k}d"] = df["Volume"].pct_change(k)  # VROC

        if k == 1: continue

        df[f"ma_ratio_{k}d"] = df["Close"] / df["Close"].rolling(k).mean() - 1
        df[f"volatility_{k}d"] = df["return_1d"].rolling(k).std()

    df["atr_pct_14d"] = atr_pct(df, 14)
    df["rsi_14d"] = rsi(df, 14)
    df["stoch_k_14d"], df["stoch_d_14d"] = stochastic(df, 14)
    df["aroon_up_25d"], df["aroon_down_25d"] = aroon(df, 25)

    # target = 1 when tmr's close is higher than today's close, else 0
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # final row can't have valid target because has no next day close
    df = df.iloc[:-1]  

    df = df.dropna()
    return df


# --- panel maker ---
def make_feature_panel(raw_dir: Path) -> pd.DataFrame:
    """Make a feature panel from all OHLCV csvs in `raw_dir`"""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        ticker = csv_path.stem  # e.g. "NVDA.csv" -> "NVDA"
        raw = pd.read_csv(csv_path, parse_dates=["Date"])
        raw = engineer_features(raw)
        raw["ticker"] = ticker
        frames.append(raw)

    if not frames:
        raise FileNotFoundError(f"No csvs found in: {raw_dir}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["Date", "ticker"]).reset_index(drop=True)
    return panel
        

if __name__ == "__main__":
    # DATA_DIR = .../stock-predictor/data/
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    panel = make_feature_panel(RAW_DIR)
    panel.to_csv(PROCESSED_DIR / "processed.csv", index=False)
