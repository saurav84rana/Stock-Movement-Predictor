# Stock markets, extreamly fast:
Requests look like this (order book):
```
BUYERS                         SELLERS

$109.00   500 shares           $110.00   900 shares
$108.00   800 shares           $111.00   600 shares
$107.00   1,000 shares         $112.00   300 shares
```
if your buy price >= someone's sell price, the exchange's matching engine fills it instantly (price-time priority). If not, it just waits in the book.

So,\
Submit orders -> order gets match -> trade happens -> prices logged.

At night, market close, but the orders build up and gets processed next at open, and, things like company profit announcement, news, foreign market still open like ADR, etc., mean that, sometimes values from yesterday's close and today's open differs.

# Plan: what features to use

## ~~within the day: (replaced by atr)~~
- ~~intra volitility: (high - low) / close~~

## For windows k = [1, 5, 10, 20] (1 day, 1 week, 2 week, 4 week)
- **(lagged) return**:\
(close[t]/ close[t-k]) - 1, or `.pct_change(k)`
- **moving avg**: (k=1 not applicable, `.rolling(1)` is just itself so its always 0)\
`close[t] / close[t-k:t].mean() -1`, or for denominator `.rolling(k).mean()`
- **rolling volatility**: (`k=1` not applicable, std() needs at least 2 values)\
"How volatile (spread out, (std)) it is in a rolling window": `return.rolling(k).std()`
- **volume change**:\
`volume[t]/ volume[t-k] - 1`, or `.pct_change(k)`

## Additionally
- RSI (=100 - (1/(1+RS)), RS = (avg_gain/ avg_loss))
- Stochastic Oscillator, Aroon
- Maybe even OBV, ADX?


## Revised plan (items added/removed):
\+ ATR%: replaces intra day volitility. Intra day volitility doesn't show any new information in the row, whereas ATR% does because it looks back.

\- intra day volitility (replaced by ATR%), ADX, OBV:\
Too many per category and already reached 20 features

# Catagorised: 

(repeat from README.md)

| Category | Features | What |
| --- | --- | --- |
| Momentum (x7: 4 + 3 bounded) | `return_*d`, `rsi_14d`, `stoch_k/_d` (bounded 0-100) | Shows short/medium term direction (velocity) of price |
| Trend (x5) | `ma_ratio_*d`, `aroon_up/_down` | How current value stands against the general trend (direction) of previous prices | 
| Volatility (x4) | `volatility_*d`,  `atr_pct_14d` | How much current prices varies from recent prices, how turbulent (varied) trading has been. More of a reassurance indicator | 
| Volume (x4) | `volume_change_*d`  | Helps confirm movement | 


# Feature breakdown:

## Momentum

### `return_*d`: Rate of Change (ROC) 

close[t] / close[t-k] - 1

Shows how far above or below current day is punching from k days ago, don't care about any in between.


### `rsi_14d`: Relative Strength Index
$RSI = 100 - \frac{100}{1 + rs}$, answer is bounded in 0-100

$rs = \frac{\text{Average of Gains}}{\text{Average of Losses}}$, "How much stronger is the gain than the loss"

#### For first entry: 14th day:
Average of Gains = sum of all gains / 14 \
(gains = positive values from `.diff()`, `.diff()` is `today's close - yesterday's close`, and 14 is standard window)

Average of Losses = sum of all ||losses|| / 14 \
(loss = negatives values from `.diff()`, `.diff()` is `today's close - yesterday's close`, and 14 is standard window)

#### For the rest of the entries: (using Average of Gains for example)
Todays's Average of Gains = $\frac{\text{yesterday's average of gain * 13 + today's gain}}{14}$, if no gain, or was a loss, then today's gain = 0

This can be rewritten as: $\frac{13}{14} \cdot$ yesterday's Average of Gain + $\frac{1}{14} \cdot $ today's gain

And thats the formula of `gains_list.ewm(alpha=1/window, adjust=False).mean()`\
`gains_list` is dataset items long, it returns a window, so doing `.mean()` tells it what to do, `.ewm()`: `exponentially weighted moving average (EWMA)`. Pandas simply uses the first values of gain to start, which should be manually calculated

Putting it together its precisely as written from the rearrangement:

Todays's Average of Gains \
= $\alpha \cdot$ current index (which is today's gain) + $(1-\alpha) \cdot$ previous index (which is yesterday's Average of Gain)


#### Understanding it
$RSI = 100 - \frac{100}{1 + rs}$ can be rewritten as: $100 \cdot \frac{rs}{1+rs}$

and, \
$rs = \frac{G}{L}$ so, (G = avg gain, L = avg loss)

so, \
$RSI = 100 \cdot \frac{G}{G+L}$

$\boxed{\text{Essentially: "How much does gain contribute in all these movement?"}}$

or

$\boxed{\text{How much of the movement is just gain?}}$ \
(movement = gain and loss)


#### Interpreting it
If G >> L: rsi -> 100 \
If G << L: rsi -> 0 \
If G = L: rsi = 50, not much/no average movement at all

Rsi > 70: overbought
- Price rose too far too quickly: may due a pullback (price fall)

Rsi < 30: oversold
- Price dropped too far too quickly: may due a rebound (price rise)

Just a general rule, because for exmaple a rising company may stay at high rsi for a long time

"overbought" doesnt mean prices will fall, it just an observation that strong upward momentum pushed prices towards top of its recent range, and vice versa for "oversold"

### `stoch_k`, `stoch_d`: sotchastic %K (fast) and %D (slow)
#### $\%K$:
$\%K = 100 \cdot \frac{C-L_n}{H_n - L_n}$, "where does current close sit in range: Highest high - Lowest low (of the window)?"

Where: \
C = today's close \
$L_n$ = lowest low over period n \
$H_n$ = highest high over period n

$\%K$ near 100 : price is near the top of its recent range
$\%K$ near 0: price is enar the bottom of its recent range

#### $\%D$:
$\%D = SMA_3(\%K)$, where $SMA_n$ = simple moving average of n-period (previous n days including today)

So if $\%K$ = [80, 80, 60, 70, 80], 

$\%D$ is (60+70+80)/3 = 70, which reacts slower.

#### Interpreting it
IRL both are used together:

%K crosses over %D -> bullish (price go up) momentum signal \
%K crosses under %D -> bearish (price go down) momentum signal \
both above 80: usually considered overbought \
both under 20: usually considered oversold

```
100 ┤
 80 ┤────── Overbought zone
    │     /
    │    / %K
 50 ┤---/---%D
    │  /
 20 ┤────── Oversold zone
  0 ┤
```
If %K crosses above %D below 20, its often considered a stronger bullish setup than crossing over randomly in the range (oversold and potentially increasing) \
likewise \
If %K crosses below %D above 80, it could be viewed as a strong bearish setup (overbought and potentially declining)

Thats just the general rule, when in strong uptrend, %K can remain above 80 for a long time. Vice versa too.

%K shows where today's close percentile is in the window $H_n - L_n$. (Think boxplots/candles). \
That can be choppy/jumpy so, \
%D just smooths that out so the trend is easier to see direction and persistance of a change/changes in that position





## Trend

### `ma_ratio_*d`: Disparity Index (Moving average but take ratio)

ratio: close[t] / close[t-k+1:t+1].mean() - 1

It shows how much above or below it (today's close) is punching from the average of window k.

Take ratio because raw values bad and pretty meaningless for ML

### `aroon_up/_down`: Aroon Indicator

aroon up = 100 $ \cdot \frac{\text{25 - 1 - period since last highest [high] in 25-period window}}{25 - 1}$ \
aroon down is that but $\text{lowest [low]}$

Minus one because if today's index is 0, then at period N, its index is 24 (24 periods ago), which means it can now reach 0-100 range.


Essentially, how long ago was latest extreme (highest high/ lowest low):

#### Interpreting it
aroon up/down = 0: it was 25 days ago \
aroon up/down = 100: its today

If aroon up = high, aroon down = low: prices are strongly trending up \
If aroon up = low, aroon down = high: prices are strongly trending down

Both high: frequent new highs and lows, getting more volatile, look at other features for better picture\
Both low: weak trend, sideways market

Crossovers could mean: \
Aroon up crossing above Aroon down: upward trend is developing \
Aroon up crossing below Aroon down: downward trend is developing


## Volatility

### `volatility_*d`: 	Historical Volatility (unannualised)

Find percentage change of today vs yesterday (already part of `return_d` ROC for 1 day)\
Then find its standard deviation in a given window \
`df["volatility_{k}d"] =  df["return_1d"].rolling(k).std()`,\
And for annulised just $\times \sqrt{252}$, where 252 days is 1 trading year.

It returns 1 $\sigma$ (standard deviation) of the pct in window k.\
Recalling empricial rule (1sd, 2sd, 3sd) = (68%, 95%, 99.7%) \
So df["volatility_20d"] = 1.5%, means: \
 - "68% of the pct doesnt deviate more than 1.5% from the mean $\mu$ (mean of the window)"
 - "95% of the pct doesnt deviate more than 3% from the mean $\mu$."
 - "99.7% ... 4.5% ..."

High HV = recent prices jumps quite a bit\
Low HV = recent prices are steady

Usually in finance data, bell curves tend to have fat tails so its not exactly interpretable like that.

### `atr_14d`:

"How much did the price vary?"/ "How volatile (avg of largest gaps) are the prices in 14 day window?" \
Start by finding true range: (imagine candle + yesterday's close bar, find the biggest gap)
```txt
tr = max([
   Today's high - Todays' low,  # intra day range
   | Today's high - yesterday's close |,
   | Today's low - yesterday's close |
])
```
This accounts for price gap overnight that intra day range misses. (Prices can change overnight)

`atr` average true range is just `.rolling(14).mean()` (SMA: simple moving average), except its upgraded, with Wilder's smoothing.

`atr_raw = tr.ewm(alpha=1/window, adjust=False).mean()` uses recursive formula seen in rsi section. 13/14 weight for previous atr, and 1/14 weight on today's `tr`. Because old SMA are completely forgotten if out of window. This one carries its weight and gradually disperses as we move forth in date.

Of course, the first `atr` is just `tr`, but actually should be `tr[:14].mean()`

#### Usage irl:
Traders use ATR for threshold for stop loss (auto sell when price fall below threshold), and ATR% for analysis and comparison (normalised by close, so it has fair comparisons with other stocks)

#### ATR raw:
1. Most common: Stoploss placement:\
If ATR = $3, entry = $100: \
Instead of putting it arbitarily $1 away (sell at $99), you can put it `1.5 x ATR` away, thats the distance of $4.5 (sell when below $95.5).\
This is because $1 may be hit by ordinary volatility.

2. position sizing:\
Maybe you want to risk $200, and decide stop at `2 x ATR`, ATR = $2,\
stop distance = `2 x ATR` = $4,\
`position size (how many shares to buy) = Maximum risk / stop-distance`, so\
`position size` = $200/$4 = 50 shares, because if the stop (`2 x ATR`) is hit, then maximum loss is `50 shares * $4 = $200`\
If it was more volatile (higher ATR), it would mean higher stop distance, and lower postition size  

3. trailing stops: Upgrading stop loss as prices rises:\
entry = $100, price = $110, ATR = $2 (assume its the same for both days), stop distance = `2 x ATR`,\
Old stop loss = $96, new stop loss = 110 - 4 = 106
new price = 115, new stop loss = 115-4 = $111

4. breakout interpretability:
if ATR = $2, but prices break above that (resistance) by $0.50, thats `0.5/2 = 0.25 ATR`, so a pretty weak/small breakout.
if break resistance by $3, thats `3/2 = 1.5 ATR`, significant breakout.

#### ATR% (ATR normalised by `close`):

5. Comparisons:

| Asset | Price | ATR | ATR% |
| --- | --- | --- | --- |
| Stock A | $20 | $1 | 5% |
| Stock B |	$200 | $4 | 2% |

Stock B ATR is $4, but relative to price, its actually less volatile than A. You get a clearer picture of the volatility and can fairly compare between stocks.

Higher ATR% = more opportunity AND more risks\
ATR never tells what direction, just how volatile it is.

## Volume:

### `volume_change_*d`: (VROC) Volume Rate of change
`df["Volume"].pct_change()`: exact same as ROC but for volume
to show how much volume has changed relative to k days ago. (Not window, just constant t)

---