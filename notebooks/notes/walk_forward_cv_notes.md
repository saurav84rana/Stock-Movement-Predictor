# TimeSeriesSplit `.split()`, extreamly fast:

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
    [0]             # [7]
    [1]             # [8]
fold 1:
    [0, 1]          # [7, 8]
    [2]             # [9]
fold 2:
    [0, 1, 2]       # [7, 8, 9]
    [3]             # [10]
fold 3:
    [0, 1, 2, 3]    # and so on....
    [4]
fold 4:
    [0, 1, 2, 3, 4] 
    [5]
```
Imagine a `partition_size = (len(x) - 1) / n_splits`. The length of `train_indexes` at fold `i`, is `partition_size * i`, whereas the length of `test_indexes` stays constant at `partition_size`.

If gap is passed, thats taken away from length of train_idx, effectively becoming: `train_indexes[:len(train_idx) - gap]`

Basically, `gap` number of items are directly taken off from the ends of `train_indexes`.


# What's happening in `walk_forward_splits()`:

For reference: [`splits.py`](../../src/validation/splits.py)

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

# Why even have a gap, and why is gap = 25:

## First reason:
Because of how target was computed: `target = tomorrow's close > today's close`. It is the only feature that looks ahead for 1 day.

For example, with `gap = 0`, the last row of train uses its next day close price, which is the first row of test. 

This is lookahead leakage.

## Second reason:
The feature with the longest look back window is `Aroon_up/_down`, with 25 trading days.

Although no features (other than target) looks into the future, if gap is smaller than a given feature's rolling window, then that rolling feature near test/train boundary, shares some of the same price histories.

For example, with `gap = 0`, for `Aroon_up/_down`, the first row of test will have 24 out of 25 shared rows with last row of train, which the features were derived from. i.e., their feature space are nearly identical.

Without a gap, this means there is a dependency between training and test set near/at boundary, meaning it can make the test set less independent from the training set.

It can also cause a slight optimistic bias, which shows that models are performing better than they actually would.

Therefore, I chose `gap = 25` between train and test sets, which separates the test period's rolling window feature from training features, whiles also avoiding lookahead leakage.
