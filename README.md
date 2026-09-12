# Daily Stock Prediction (S&P 500, AAPL, PLTR)

Predicts the next trading day's return/direction for the S&P 500 index (`^GSPC`),
Apple (`AAPL`), and Palantir (`PLTR`) using classic ML on engineered technical
features. No paid data or API keys required (uses `yfinance`).

## What it actually predicts

Raw closing price is almost perfectly autocorrelated (tomorrow's price ≈
today's price), so a model that predicts price directly looks deceptively
accurate while learning nothing useful. Instead, every model here predicts:

- **Regression target**: next day's close-to-close **% return**
- **Classification target**: next day's **direction** (up/down)

The predicted price shown in `predict.py` is just `last_close * (1 + predicted_return)`,
reconstructed for readability.

## Setup

```bash
cd /Users/minsoochoi/Desktop/stock
source venv/bin/activate   # already created; recreate with `python3 -m venv venv` if missing
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py       # fetches data, engineers features, tunes + evaluates models, saves the best ones
python src/predict.py     # loads saved models, prints tomorrow's prediction + live news sentiment for each ticker
python src/news_pulse.py  # just the live news-sentiment readout, on its own
```

`train.py` re-downloads fresh history every run, so re-run it periodically
(e.g. weekly) to keep models current — daily market microstructure drifts.

## Pipeline

1. **`src/fetch_data.py`** — downloads ~10 years of daily OHLCV via `yfinance`, caches to `data/*.csv`.
2. **`src/features.py`** — builds ~20 "own" features per day from *only* past data (no
   look-ahead): lagged returns, SMA ratios (5/10/20/50d), rolling volatility, RSI(14), MACD,
   Bollinger Band position/width, volume change, day-of-week.
3. **`src/macro_features.py`** — builds 21 cross-market/"world" features shared across all
   three tickers: VIX level/z-score, 10Y Treasury yield level/change, the 10Y-vs-13-week yield
   curve slope (recession-risk signal), dollar index return, oil/gold returns, Dow/Nasdaq/Russell
   2000 returns, Nikkei/FTSE/DAX returns (how Asia and Europe already traded before the US session
   opens), Bitcoin return/volatility (trades weekends, so it reflects global news equities can't
   price in until Monday), a credit-stress proxy (high-yield bond ETF return minus Treasury bond
   ETF return), and QQQ return (tech-sector proxy). All same-day (close of day *t*), which is
   valid information for predicting day *t+1* — no look-ahead.
4. **`src/train.py`** — chronological (not shuffled) train/test split, last ~15% of days held out
   as the final, untouched evaluation set. For each ticker, trains and compares two feature sets —
   **baseline** (own features only, 23) and **enhanced** (own + macro/world, 44) — and for each,
   runs the optimization pipeline described below. Saves a test-period actual-vs-predicted chart to
   `reports/`, and full metrics to `reports/model_comparison.csv`.
5. **`src/predict.py`** — recomputes features on the latest available day and prints tomorrow's
   predicted return, direction, reconstructed price, and a live news-sentiment readout for context.
6. **`src/news_pulse.py`** — live news-sentiment snapshot (see "World-events layer" below).

## Optimization pipeline (per ticker, per task, per feature set)

1. **Feature selection**: when a feature set exceeds 20 columns, a quick Random Forest fit
   ranks importances and only the top 20 are kept — with only ~1,200-2,100 training rows,
   more features than that adds noise faster than signal (this is exactly what let the macro
   features hurt some models before this pass — see below).
2. **Hyperparameter tuning**: Random Forest and HistGradientBoosting are tuned with
   `RandomizedSearchCV` over `TimeSeriesSplit` folds (never shuffled k-fold — a fold's "future"
   must never leak into its own training data). 10 candidate configs × 3 time-ordered folds each.
3. **Ensembling**: a soft-voting ensemble of the tuned Random Forest + tuned HistGradientBoosting
   + the linear model (Ridge/Logistic, left untuned — it's already fast, stable, and low-variance)
   is added as one more candidate.
4. **Final selection**: baseline vs. enhanced feature sets, and every candidate model within each,
   are all compared on the *same* held-out test window from step 4 above — the CV folds in step 2
   are only ever used to pick hyperparameters, never to report the final metric. The regressor and
   classifier for a ticker can end up using different feature sets and different selected columns;
   exactly which columns each saved model expects is recorded in `models/{ticker}_features.json`.

## Reading the results honestly

Daily stock returns are close to a random walk — beating a coin flip
consistently is genuinely hard, and these results reflect that:

- Directional accuracy sits around 50-54%, only modestly above the ~50% baseline.
- RMSE is close to (sometimes worse than) the "predict zero return" baseline.
- None of this is investment advice, and nothing here accounts for transaction
  costs, slippage, or overfitting risk from repeated model comparisons.

Treat this as a demonstration of a correct, leakage-free ML pipeline for
financial time series, not a working trading signal. Ideas for improvement
are in the "Extending this" section below.

### Did the optimization actually help?

Yes, modestly and unevenly — the honest result, ticker by ticker, after
feature selection + tuning + ensembling + world-market features
(full numbers in `reports/model_comparison.csv`):

| Ticker | Regressor RMSE | Classifier accuracy |
|---|---|---|
| SP500 | 0.01029 (tuned baseline wins) | **53.9% → 56.1%** (tuned + world features) |
| AAPL | 0.01936 (tuned baseline wins, ensemble) | 52.6% → 53.0% (tuned + world features) |
| PLTR | 0.03954 → 0.03943 (tuned + world features, ensemble) | 51.4% → 52.3% (tuning alone; world features didn't help here) |

SP500's classifier is the strongest result across this whole project: **56.1%
directional accuracy**, up from the original untuned 53.9%. That gain is a mix
of tuning (better-regularized trees generalize better than the original fixed
hyperparameters) and the world-market features actually contributing —
feature importances confirm Treasury yield, VIX, credit stress, and the
overseas-index returns get real weight there.

It's still not uniform: PLTR's classifier improved from tuning but *not* from
the world features (shortest price history of the three, so more features
means more overfitting risk), and no regressor beat ~2% RMSE improvement
anywhere — daily returns remain close to a random walk regardless of how much
tuning you throw at it. `train.py` evaluates every combination on the same
held-out window and keeps whichever wins per ticker per task, so it never
defaults to the fancier setup just because it's newer.

## World-events layer: what's actually feasible without a paid data source

Two different things answer "analyze what's happening in the world," and only
one of them can honestly be trained into the model:

- **Backtestable** (used by the trained models, via `macro_features.py`): global
  market indices (Nikkei/FTSE/DAX — how the rest of the world already traded
  before the US opens), Bitcoin (trades weekends, so it reflects news equities
  can't price in until Monday), the yield curve, VIX, credit spreads. These
  have full historical daily price archives, so they can be properly
  backtested with no look-ahead — same standard as every other feature here.
- **Live-only, shown as context, NOT trained into the model** (`news_pulse.py`,
  surfaced in `predict.py`'s `live_news_sentiment` column): current headlines
  for each ticker plus general market news, scored with VADER (a free,
  offline, lexicon-based sentiment analyzer). This is a genuine read of
  today's news tone — but every free news source we found (including
  `yfinance`'s own news feed) only returns *current* headlines with no
  point-in-time historical archive. Without that archive there's nothing to
  backtest a "trained on news" model against, so training on it would mean
  quietly claiming a track record that doesn't exist. A real historical
  news-sentiment feature would need a paid archive (NewsAPI's historical
  tier, Refinitiv, RavenPack, Bloomberg) — worth doing if this ever needs to
  be more than a demo, but out of scope for a free, no-API-key project.

## Same-day prediction (predict today's 4pm ET close)

The pipeline above predicts *tomorrow's* close using *yesterday's* daily bar.
`train_intraday.py` / `predict_intraday.py` instead predict **today's** close using
intraday price action from earlier the same day — useful for a running estimate
throughout the trading session.

```bash
python src/train_intraday.py     # trains the same-day-close model (run occasionally, e.g. weekly)
python src/predict_intraday.py   # run any time 9:30am-4:00pm ET for a live estimate of today's close
```

**How it works:** Yahoo Finance only retains 5-minute bars for 60 days but hourly
bars for ~2 years, so the model is *trained* on ~2 years of hourly "snapshots"
(one per hour of each trading day: given what's happened since the open, what's
the remaining return to the 4pm close?), then *queried* at prediction time with a
fresh 5-minute snapshot of today. Features are all ratios/time-remaining based
(minutes until close, % return since open, where price sits in today's range,
volume so far vs. typical, overnight gap), not raw bar counts, so the
train/inference resolution mismatch doesn't break it.

Outside market hours, `predict_intraday.py` reports the last session's actual
close instead of a live prediction.

**Honest results** (`reports/intraday_model_comparison.csv`, `reports/*_intraday_accuracy.png`):
directional accuracy is close to a coin flip for SP500/PLTR, and moderately
better (~60-65%) for AAPL, generally improving somewhat as the close gets nearer
— consistent with markets getting more efficient to predict at short lags but
still not a reliable trading edge.

## Extending this

- Pay for a historical news/sentiment archive (NewsAPI historical tier,
  RavenPack, Refinitiv) to properly backtest a news-driven feature instead of
  only showing it live.
- Multi-fold walk-forward evaluation (several rolling train/test windows,
  not just one final holdout) for the *reported* metric, not just for
  hyperparameter tuning — would show how stable these accuracy numbers are
  across different market regimes.
- Add a sequence model (LSTM/Transformer via PyTorch) to capture temporal
  patterns the tree/linear models can't.
- Backtest with realistic position sizing, fees, and slippage rather than
  just directional accuracy.
- Options-implied volatility (skew/term structure) as a forward-looking
  volatility feature, complementing VIX's backward-looking realized measure.
