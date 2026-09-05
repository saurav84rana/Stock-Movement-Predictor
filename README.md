# Stock Movement Predictor

Predict next day stock direction (up/down) using historical price/technical features

And have time aware validation to ensure no lookahead bias, walk forward splits 
Then compare against naive baselines like always predict up

This will be served via a FastAPI endpoint, dockerised, tested, version controlled.

## Features
Engineered from daily OHLCV, only using data available on or before their respective dates.

I used the following features:
- ROC, VROC, Disparity Index (MA's ratio), rolling volatility
- ATR%, RSI, Stochastic %K and %D, Aroon Up and Aroon Down

| Category | Features | What |
| --- | --- | --- |
| Momentum (x7) | `return_*d`, `rsi_pct_14d`, `stoch_k/_d` | Shows short/medium term direction (velocity) of price |
| Trend (x5) | `ma_ratio_*d`, `aroon_up/_down` | How current value stands against the general trend (direction) of previous prices | 
| Volatility (x4) | `volatility_*d`,  `atr_14d` | How much current prices varies from recent prices, how turbulent (varied) trading has been. More of a reassurance indicator | 
| Volume (x4) | `volume_change_*d`  | Helps confirm movement | 

You can find my rough notes including derivations, formulas, and interpretations here: [`features_notes.md`](./notebooks/features_notes.md)