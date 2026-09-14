# Daily Stock Prediction (S&P 500, AAPL, PLTR)

[![GitHub repo](https://img.shields.io/badge/GitHub-stock--price--predictor-181717?logo=github)](https://github.com/minsoo070420-pixel/stock-price-predictor)

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
python src/train.py           # fetches data, engineers features, tunes + evaluates models, saves the best ones
python src/predict.py         # loads saved models, prints tomorrow's prediction + live news sentiment for each ticker
python src/news_pulse.py      # just the live news-sentiment readout, on its own
python src/implied_vol.py     # just the live single-stock implied-vol/skew readout (AAPL, PLTR), on its own
python src/backtest_dates.py  # walk-forward accuracy check + transaction-cost reality check over recent days
python src/leakage_check.py   # verifies no feature depends on future data (see the checklist below)
python src/long_horizon_drift.py    # NOT day-to-day forecasting -- see "The 80-90% question" below
python src/predict_long_horizon.py  # ditto -- read that section before running either
```

`train.py` re-downloads fresh history every run, so re-run it periodically
(e.g. weekly) to keep models current — daily market microstructure drifts.

## Pipeline

1. **`src/fetch_data.py`** — downloads ~10 years of daily OHLCV via `yfinance`, caches to `data/*.csv`.
2. **`src/features.py`** — builds ~20 "own" features per day from *only* past data (no
   look-ahead): lagged returns, SMA ratios (5/10/20/50d), rolling volatility, RSI(14), MACD,
   Bollinger Band position/width, volume change, day-of-week.
3. **`src/macro_features.py`** — builds 26 cross-market/"world" features shared across all
   three tickers: VIX level/z-score (S&P 500's own options-implied vol), the VIX/VIX3M term
   structure and VXN (Nasdaq-100 implied vol) for more implied-vol signal, 10Y Treasury yield
   level/change, the 10Y-vs-13-week yield curve slope (recession-risk signal), dollar index
   return, oil/gold returns, Dow/Nasdaq/Russell 2000 returns, Nikkei/FTSE/DAX returns (how Asia
   and Europe already traded before the US session opens), Bitcoin return/volatility (trades
   weekends, so it reflects global news equities can't price in until Monday), a credit-stress
   proxy (high-yield bond ETF return minus Treasury bond ETF return), and QQQ return (tech-sector
   proxy). All same-day (close of day *t*), which is valid information for predicting day *t+1*
   — no look-ahead.
4. **`src/implied_vol.py`** — live single-stock options-implied-vol/skew snapshot for AAPL/PLTR
   (see "Options-implied volatility" below for why this is live-only, not trained).
5. **`src/train.py`** — chronological (not shuffled) train/test split, last ~15% of days held out
   as the final, untouched evaluation set. For each ticker, trains and compares two feature sets —
   **baseline** (own features only, 23) and **enhanced** (own + macro/world, 49) — and for each,
   runs the optimization pipeline described below. Saves a test-period actual-vs-predicted chart to
   `reports/`, and full metrics to `reports/model_comparison.csv`.
6. **`src/predict.py`** — recomputes features on the latest available day and prints tomorrow's
   predicted return, direction, reconstructed price, live news sentiment, and live implied vol/skew
   for context.
7. **`src/news_pulse.py`** — live news-sentiment snapshot (see "World-events layer" below).
8. **`src/baselines.py`** — finance-specific dumb baselines (momentum persistence), eligible to
   actually win model selection, not just serve as a floor (see the checklist below — this matters).
9. **`src/leakage_check.py`** — rebuilds every feature from truncated history and verifies it's
   identical to the production version, to actually verify (not assume) there's no look-ahead.

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

### The classification decision threshold, and why it's 0.51 not 0.50

Backtesting turned up something concrete: SP500's classifier predicted "UP" on
21 out of 21 days over a real month (8/13-9/11), because its predicted
probability of "up" never once dropped below 0.50 (it ranged 0.50-0.68) — a
soft bias from training data where ~56% of days were historically up, not a
hard-coded rule, but functionally the same result. Sweeping the decision
threshold showed that ~0.65 would have maximized accuracy *for that specific
month* (38%→67%) — but that number is fit in hindsight on the very data it's
being tested against, which is overfitting, not a fix, and would not
generalize to the next month.

`CLASSIFICATION_THRESHOLD = 0.51` in `config.py` is the honest, defensible
version: a fixed, modest 1-point shift off 50/50, applied uniformly in
training/evaluation, live prediction, and backtesting (previously the
default `.predict()` used 0.50 implicitly and inconsistently against what the
backtest reported). It's small enough not to be curve-fit to any one test
window. Measured effect: it flipped only 1 of SP500's 21 calls that month
(20 UP / 1 DOWN) and left accuracy roughly unchanged (~33-38%, within normal
noise for n=21) — which itself is an honest finding: the bias is baked in
deeply enough that a 1-point nudge can't meaningfully counter it. The
structural fix that actually worked is below.

### The structural fix: class-weighted training + balanced-accuracy selection

A threshold nudge patches the symptom at inference time. The actual cause was
upstream, in training and model selection:

- **Training**: `RandomForestClassifier`, `HistGradientBoostingClassifier`,
  and `LogisticRegression` now all use `class_weight="balanced"`, so getting a
  DOWN day wrong costs the model exactly as much as getting an UP day wrong,
  instead of the training data's ~56% historical up-rate making "lean UP" the
  path of least loss.
- **Selection**: `RandomizedSearchCV` and the final baseline-vs-enhanced /
  model-vs-model comparisons now score classifiers on **balanced accuracy**
  (mean of per-class recall) instead of raw accuracy. Raw accuracy is exactly
  the metric a model that only ever predicts the majority class can win for
  free — balanced accuracy scores that same model at precisely 50%, matching
  the majority-class baseline instead of beating it.

**Measured effect — the bias is genuinely gone**: over the same real month
that motivated this fix, predicted calls went from SP500 21/21 UP to a
realistic 17 DOWN / 13 UP, and AAPL from 29/30 UP to an even 15/15 split.
`reports/model_comparison.csv` also now carries a `pct_predicted_up` column
per candidate specifically so a collapse like this is visible again
immediately, without needing to re-derive it from a backtest by hand.

**The honest cost of removing the exploit**: held-out balanced accuracy for
all three tickers dropped to 49.6-52.3% — indistinguishable from a coin flip.
The previous ~54-56% figures were, in part, the model getting rewarded for
leaning on the base rate rather than reading real daily signal; once that
shortcut was closed off, what's left is close to the honest ceiling this
category of data actually supports. That's not a regression from this
change — it's this change correcting an inflated number from before it.

## Six-point ML-hygiene checklist, applied and verified

A standard list of things that quietly wreck financial ML pipelines, checked
against this one — not just asserted, actually tested where that's possible:

1. **Leakage is the #1 silent killer.** Verified, not assumed: `src/leakage_check.py`
   rebuilds every feature (own + macro) from data truncated at several cutoff
   dates and confirms each cutoff's last row is bit-for-bit identical to that
   date's row computed from the full history. If truncating the future ever
   changed a past feature value, that would be leakage. Run: `python src/leakage_check.py`
   — currently passes for all three tickers, both feature families.
2. **Never shuffle time series data.** Already true throughout: `TimeSeriesSplit`
   for hyperparameter CV, a strictly chronological split for the final held-out
   test, nowhere a shuffled k-fold. Verified by grepping for `shift(-` (only the
   target uses it) and confirming no `KFold`/`shuffle` anywhere in the codebase.
3. **Predict returns, not price levels.** Already true from the start of this
   project (see "What it actually predicts" above) — targets are `pct_change`,
   never raw price.
4. **Always compare against a dumb baseline — and this one bit us.** Added a
   `baseline_persistence` candidate (`src/baselines.py`: "tomorrow repeats
   today's direction/return") alongside the existing zero-return/majority-class
   floors. It was initially *excluded* from ever winning the model-selection
   step, same as the other baselines — until checking revealed it actually
   **beats every tuned/ensembled model's balanced accuracy, for all three
   tickers**. That's now fixed structurally: `run_task` in `train.py` only
   excludes the true know-nothing floors (zero-return, majority-class) from
   winning; a real strategy like persistence is left eligible, and now wins
   the classifier slot for SP500, AAPL, *and* PLTR. Concretely: no amount of
   feature engineering, tuning, or ensembling in this pipeline beats "assume
   today's direction continues." The regressors still handily beat persistence
   on RMSE (predicting an exact return via naive continuation is much worse
   than via a fitted model), so that slot is unaffected.
5. **A good backtest score ≠ a profitable strategy.** Operationalized instead
   of just asserted: `backtest_dates.py` now simulates actually trading each
   call (long on UP, short on DOWN) against a rough round-trip transaction-cost
   assumption per ticker (`TRANSACTION_COST_BPS` in `config.py` — 2bps SP500,
   3bps AAPL, 8bps PLTR; not researched for any specific broker, just
   plausible for a liquid retail-sized order) and reports gross vs. net bps
   per trade. Result on the last 30 trading days: AAPL's already-thin edge
   flips negative after costs (+1.8 → **-1.2 bps/trade**); SP500 and PLTR stay
   net positive, but PLTR's number is dominated by one outsized single-day
   move inside the window (a +29% day), so a 30-day average there is not a
   reliable estimate — exactly the kind of fat-tail sensitivity that makes
   "backtest looked profitable" a weak claim on its own.
6. **Missing values in event data are often informative, not random.**
   `news_pulse.py` already avoided imputing a neutral score for "no news";
   tightened further to distinguish three states explicitly: `had_news=True`
   (scored), `had_news=False` (genuinely no recent coverage — itself a
   signal, a quiet news day), and `had_news=None` (the fetch failed —
   technical missingness, unrelated to whether news exists, and conflating it
   with "no news" would have been its own quiet bug).

## Reading the results honestly

Daily stock returns are close to a random walk — beating a coin flip
consistently is genuinely hard, and these results reflect that:

- Directional accuracy sits around 50-58%, only modestly above the ~50% baseline
  — and for all three tickers, the classifier that actually wins is
  `baseline_persistence` (see the checklist above), not a tuned model.
- RMSE is close to (sometimes worse than) the "predict zero return" baseline.
- None of this is investment advice; `backtest_dates.py`'s cost-adjusted reality
  check shows the edge (such as it is) doesn't clearly survive transaction costs
  for at least one ticker, and says nothing about slippage, capacity, or the
  fact that other people are running similar models on the same public data.

Treat this as a demonstration of a correct, leakage-free ML pipeline for
financial time series, not a working trading signal. Ideas for improvement
are in the "Extending this" section below.

### Did the optimization actually help?

For the regressors: modestly. Feature selection, tuning, and ensembling
bought a small RMSE improvement in some tickers (PLTR: 0.03964→0.03943) and
made no meaningful difference in others — daily returns remain close to a
random walk regardless of how much tuning you throw at it. No regressor
anywhere beat roughly a 2% RMSE improvement over the untuned starting point.

For the classifiers: this is where the story is more interesting, and more
honest than it first looked, in two stages. An earlier pass (tuning +
world-market features, scored on *raw* accuracy) reported SP500 at 56.1% —
the best number in this whole project. But raw accuracy rewards a model for
leaning on the training data's ~56% historical up-day rate, and backtesting
caught it doing exactly that (predicting UP on 21 of 21 real days in a row).
Fixing that at the source — `class_weight="balanced"` plus scoring on
*balanced* accuracy — dropped the honest number to 49.6-52.3%. Then, adding
the persistence baseline as an eligible competitor (checklist item #4 above)
revealed the real punchline: that simple "tomorrow repeats today" rule beats
every tuned/ensembled candidate outright, for all three tickers (SP500 ~52%,
AAPL ~51%, PLTR ~58% balanced accuracy) — so it's now what's actually
deployed. All the feature engineering, hyperparameter tuning, and ensembling
in this project's classifiers, combined, do not beat one line of momentum
logic. That's the most useful single finding in this whole README, and it
held even after adding options-implied volatility (below) — the next
section's honest numbers make the same point again with a different feature.

`train.py` still evaluates every feature-set/model combination on the same
held-out window and keeps whichever wins per ticker per task, so it never
defaults to the fancier setup just because it's newer — that part of the
methodology didn't change, only the metric being optimized for did.

## Options-implied volatility: the same free-data ceiling as news, again

Implied vol is forward-looking in a way price-derived technical indicators
aren't — it's the market's own current bet on future volatility, priced into
options today, rather than a description of what already happened. Same
two-tier split as the world-events layer below:

- **Backtestable** (trained into the models, `macro_features.py`): VIX *is*
  already an options-implied-vol feature — it's literally the S&P 500's own
  30-day implied volatility, derived from SPX options, and has been in this
  pipeline since macro features were added. New this pass: **VIX3M** (the
  3-month sibling) gives the **vol term structure** (`vix_term_structure` =
  VIX3M/VIX — below 1, "backwardation," is a well-known near-term-stress
  signal), and **VXN** (Nasdaq-100's own 30-day implied vol) is a
  sector-relevant proxy for AAPL/PLTR specifically. All three have full
  historical daily archives via `yfinance`, so — unlike single-stock IV below
  — they can be properly backtested with no look-ahead.
- **Live-only, shown as context, NOT trained into the model**
  (`implied_vol.py`, surfaced in `predict.py`'s `live_atm_iv`/`live_iv_skew`
  columns for AAPL/PLTR): current at-the-money implied vol (~30 days out) and
  put/call skew, read directly from each stock's live options chain. Same
  constraint as news: Yahoo (and every other free source checked) only
  exposes *current* options chains for individual equities, no historical
  archive, so there's nothing to backtest a "trained on AAPL's own IV"
  feature against. A real version of this would need a paid options-data
  vendor (OptionMetrics, CBOE DataShop, ORATS).

**Honest result**: adding VIX3M term structure and VXN did not move the
needle. Classifier balanced accuracy barely changed (SP500 ~52.0%→51.9%,
PLTR unchanged at ~58.3%; AAPL's small uptick to ~51.2% turned out to be a
test-window artifact, not signal — see caveat below), and `baseline_persistence`
still wins the classifier slot for all three tickers regardless. This is the
same story as the original macro features and the world-market additions:
another financially sensible, theoretically forward-looking signal that
doesn't break through the wall this category of data runs into.

*Caveat on comparing feature sets when persistence wins*: `baseline_persistence`
only ever looks at `ret_1d` — it doesn't use VIX3M, VXN, or anything else in
the "enhanced" feature set. When it wins both the baseline and enhanced runs
(as it usually does), any small difference between those two results reflects
a slightly different test-window date range (macro data's rolling-window
warmup trims a few different rows) rather than the extra features actually
mattering. The comparison is only informative when the winning model
genuinely depends on the feature set — true for every regressor here, not
always true for these classifiers.

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

## The 80-90% question: can this hit 80-90% accuracy?

Not at daily-direction prediction, no — and that's not a limitation of this
particular pipeline, it's close to a hard ceiling for anyone. Every lever
pulled in this project (hyperparameter tuning, ensembling, 26 macro/world
features, options-implied volatility, class-weighting) moved balanced
accuracy by low single digits at best, and the strongest model found for any
of the three tickers was a **one-line momentum rule**, which beat every
tuned model built here. Professional quant funds with vastly more data,
compute, and expertise report hit rates around 52-58% on their best signals
— 80-90% on a liquid, publicly-traded asset's next-day direction would imply
a near risk-free money machine sitting in plain sight, which markets don't
leave lying around.

`src/long_horizon_drift.py` and `src/predict_long_horizon.py` answer a
**different, much weaker question** that genuinely can hit 80%+: not "what
will tomorrow's direction be," but "if you always bet UP and hold for N
trading days, how often would that have worked historically?" A high hit
rate there reflects the equity market's well-documented long-run upward
drift (the equity risk premium) — it is a reframed buy-and-hold backtest, not
a discovered signal, and it says nothing about tomorrow's price.

**What the analysis actually found, empirically, out-of-sample** (`reports/long_horizon_drift.csv`
has the full horizon-by-horizon table):

| Ticker | Shortest horizon crossing 80% OOS | OOS hit rate | Independent samples |
|---|---|---|---|
| SP500 | 2 months (42 trading days) | 86.6% | ~7 |
| AAPL | 6 months (126 trading days) | 97.6% | **~1** |
| PLTR | *none* | — | — |

Read this carefully, not just the headline numbers:

- **The overlapping-window trap**: a 6-month window and the next day's 6-month
  window share 125 of 126 days — they are not independent evidence. AAPL's
  "251 test windows" is really **about one independent 6-month period**. That
  97.6% is closer to "AAPL happened to keep going up during the one stretch
  we have" than a validated statistic. SP500's 86.6% (~7 independent windows)
  is on firmer ground but still thin by any normal statistical standard.
- **PLTR breaks the whole premise.** At longer horizons its out-of-sample hit
  rate actually *drops* to 27-33% — the held-out test period was a genuine
  down-trending stretch for PLTR, so "always bet UP" would have been wrong
  most of the time. `predict_long_horizon.py` deliberately makes no call for
  PLTR rather than force an 80% claim that the data doesn't support — a
  script that found a way to claim 80%+ for all three tickers regardless of
  what actually happened would be fitting the conclusion, not reporting it.
- **This only "works" as long as the drift continues.** Every one of these
  numbers is a statement about the past, extrapolated forward on the
  assumption the trend holds. When it doesn't — as it apparently didn't for
  PLTR in this exact test window — the approach fails, and it fails in
  exactly the direction that matters (calling UP right before a decline).

Net: 80-90% is achievable *as a number*, but only by predicting something
much less useful than "will the stock go up tomorrow," and even that weaker
claim's evidence is thinner than the headline percentage suggests.

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
