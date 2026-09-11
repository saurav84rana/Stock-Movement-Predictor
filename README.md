# Stock Movement Predictor

Predict next day stock direction (up/down) using historical price/technical features

And have time aware validation to ensure no lookahead bias, walk forward splits 
Then compare against naive baselines like always predict up

This will be served via a FastAPI endpoint, dockerised, tested, version controlled.

## Stock Universe:
Includes 24 liquid large cap US stocks, equally distributed across 4 categories, for 8 years of continuous daily OHLCV data. (Trading days from 1st January 2018 to 31st December 2025, inclusive)

| Field | Tickers |
| --- | --- |
| Tech (x6) | NVDA, AAPL, MSFT, AMD, GOOGL, META |
| Automotive / industrial (x6) | F, GM, CAT, BA, TSLA, RACE |
| Finance (x6) | JPM, GS, V, MA, BAC, WFC |
| Consumer (x6) | NKE, MCD, SBUX, KO, PEP, DIS |


## Features
Engineered from daily OHLCV, only using data available on or before their respective dates.

I used the following features:
- [1, 5, 10, 20] day windows: ROC, VROC
- [5, 10, 20] day windows: Disparity Index (MA's ratio), rolling volatility
- 14 day window: ATR%, RSI, Stochastic %K and %D
- 25 day window: Aroon up/down

| Category | Features | What |
| --- | --- | --- |
| Momentum (x7) | `return_*d`, `rsi_14d`, `stoch_k_14d`/`stoch_d_14d` | Shows short/medium term direction (velocity) of price |
| Trend (x5) | `ma_ratio_*d`, `aroon_up_25d`/`aroon_down_25d` | How current value stands against the general trend (direction) of previous prices | 
| Volatility (x4) | `volatility_*d`,  `atr_pct_14d` | How much current prices varies from recent prices, how turbulent (varied) trading has been. More of a reassurance indicator | 
| Volume (x4) | `volume_change_*d`  | Helps confirm movement | 

You can find my rough notes including derivations, formulas, and interpretations here: [`features_notes.md`](./notebooks/features_notes.md)


## Walk forward validation (Time Series Split)

### Why use Time Series Split and not others:
Normal k fold or shuffled 80/20 split, relies on the fact that rows are independent of order and can occur (be placed) anywhere.

But this data is time dependent, because a row from March can only exist after events reperesented by rows from January and Februray have already happened.

This means shuffling or k fold split would mean, training on later dates and testing on earlier dates. This lets the model learn from information that wouldn't be available at the date the test prediction was supposedly made.

Essentially, the validation score could give a misleading optimistic estimate of how well the model would generalise to real future data.

So, instead of those, we use walk forward validation via TimeSeriesSplit, which guarantees all training dates come before any testing dates.

### TimeSeriesSplit `.split()`, extreamly fast:

TimeSeriesSplit `.split(x)` returns **INDEXES** of the passed list `x` on what to train and what to test:
```py
tss = TimeSeriesSplit(n_splits=5, gap=0)
x = [7, 8, 9, 10, 11, 12]
for i, (train_indexes, test_indexes) in enumerate(tss.split(x)):
    print(f"fold {i}:")
    print("\t", train_indexes)
    print("\t", test_indexes)
```
```py
fold 0:
    [0]
    [1]
fold 1:
    [0, 1]
    [2]
fold 2:
    [0, 1, 2]
    [3]
fold 3:
    [0, 1, 2, 3]
    [4]
fold 4:
    [0, 1, 2, 3, 4]
    [5]
```
Imagine a `partition_size = (len(x) - 1) / n_splits`. The length of `train_indexes` at fold `i`, is `partition_size * i`, whereas the length of `test_indexes` stays constant at `partition_size`.

If gap is passed, thats taken away from length of train_idx, effectively becoming: `train_indexes[:len(train_idx) - gap]`

Basically, `gap` number of items are directly taken off from the ends of `train_indexes`.


### What's happening in `walk_forward_splits()`:
Pass down the processed_df (processed.csv) into `walk_forward_splits()`:

- Then extract the dates, keep only unique ones. Thats the `unique_dates` lookup, which includes all trading days. (It is tested so that all dates have all tickers)

- Next pass `unique_dates` into `tss.split(unique_dates)`, which basically returns an array of tuples, (`train_indexes`, `test_indexes`), where indexes correspond to `unique_dates`'s.

- Then grab the dates by simply plugging `train_indexes` into `unique_dates`: \
`train_dates = unique_dates[train_indexes]`, same for `test_date`.

Now get the indexes of the original processed_df, where dates match:

- This was done by creating a mask, `df["Date"]` preserves original processed_df's row order, so performing `df["date"].isin(train_dates)` works and also perserves row order. This returns column of True and False if the date matches or not.

- Then to get positional indexes, simply doing `np.flatnonzero(mask)` of mask, which returns only indexes of items in mask that were not 0/False. Because row order was perserved, this is directly a raw positional index.

The reason we get positional index, and not just grab the index column of processed_df like `df.index[mask]`, is because a DataFrame's index label row may differ to its physical row position.

`np.flatnonzero(mask)` gives us positional index, so that mean `.iloc[]` can safely use the indices later.

### Why gap, and why gap=25:

#### First reason:
Because of how target was computed: `target = tomorrow's close > today's close`. It is the only feature that looks ahead for 1 day.

For example, with `gap = 0`, the last row of train uses its next day close price, which is the first row of test. 

This is lookahead leakage.

#### Second reason:
The feature with the longest look back window is `Aroon_up/_down`, with 25 trading days.

Although no features (other than target) looks into the future, if gap is smaller than a given feature's rolling window, then that rolling feature near test/train boundary, shares some of the same price histories.

For example, with `gap = 0`, for `Aroon_up/_down`, the first row of test will have 24 out of 25 shared rows with last row of train, which the features were derived from. i.e., their feature space are nearly identical.

Without a gap, this means there is a dependency between training and test set near/at boundary, meaning it can make the test set less independent from the training set.

It can also cause a slight optimistic bias, which shows that models are performing better than they actually would.

Therefore, I chose `gap = 25` between train and test sets, which separates the test period's rolling window feature from training features, whiles also avoiding lookahead leakage.
