# Daily Stock Prediction (S&P 500 + 7 large-caps)

[![GitHub repo](https://img.shields.io/badge/GitHub-stock--price--predictor-181717?logo=github)](https://github.com/minsoo070420-pixel/stock-price-predictor)

Predicts the next trading day's return/direction for `SP500` (`^GSPC`), `AAPL`,
`PLTR`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, and `META` using classic ML on
engineered technical features. No paid data or API keys required (uses `yfinance`).

**A note on scope while reading this README**: the project started with just
SP500/AAPL/PLTR, and most of the narrative sections below — the six-point ML
checklist, the 80-90% question, the published-research comparison — were
written and verified against those three specifically. MSFT, GOOGL, AMZN,
NVDA, and META were added later, trained through the same pipeline, and
independently leakage-checked (`leakage_check.py` passes for all eight), but
weren't individually re-litigated through every prior finding. Treat the
original three as the deeply-audited core and the other five as covered by
the same methodology, not as separately re-proven case by case.

## Two pieces

The project is split into two clearly separated parts, each with its own job:

- **`src/engine/`** — Part 1, the prediction/backtest/research engine. Everything
  that fetches data, engineers features, trains and tunes models, verifies
  there's no leakage, and backtests results. This is all of the work described
  in the rest of this README.
- **`src/comparison/`** — Part 2, a neutral descriptive-statistics comparison
  across all eight tickers, built entirely from Part 1's own outputs. It
  reports how each ticker's model has performed historically — held-out test
  accuracy, recent backtest accuracy, cost-adjusted returns, long-horizon
  drift, a live 3-month walk-forward check, raw fundamentals — side by side.
  **It does not train anything new, and it does not recommend a trade.** See
  "Descriptive ticker comparison" below for why that line matters and isn't
  just a formality.

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

To train with NYT news-sentiment features (`news_features.py`, see below),
put a free API key from [developer.nytimes.com](https://developer.nytimes.com)
(Archive API enabled on the app) in a `.env` file at the project root:

```
NYT_API_KEY=your-key-here
```

`.env` is gitignored, so the key never gets committed. Without it, the news
feature set is simply unavailable and the pipeline falls back to
baseline/macro features only.

## Usage

```bash
python src/engine/train.py           # fetches data, engineers features (incl. NYT news history on first run), tunes + evaluates models, saves the best ones
python src/engine/predict.py         # loads saved models, prints tomorrow's prediction + live news sentiment for each ticker
python src/engine/news_pulse.py      # just the live news-sentiment readout, on its own
python src/engine/implied_vol.py     # just the live single-stock implied-vol/skew readout (AAPL, PLTR), on its own
python src/engine/backtest_dates.py  # walk-forward accuracy check + transaction-cost reality check over recent days
python src/engine/leakage_check.py   # verifies no feature depends on future data (see the checklist below)
python src/engine/long_horizon_drift.py    # NOT day-to-day forecasting -- see "The 80-90% question" below
python src/engine/predict_long_horizon.py  # ditto -- read that section before running either

python src/comparison/compare_tickers.py   # Part 2: neutral side-by-side stats across tickers, NOT investment advice
```

`train.py` re-downloads fresh history every run, so re-run it periodically
(e.g. weekly) to keep models current — daily market microstructure drifts.
The first `train.py` run also builds the full NYT news history (~120 monthly
archive calls, cached to `data/news_cache/` afterward so subsequent runs are fast).

## Pipeline

1. **`src/engine/fetch_data.py`** — downloads ~10 years of daily OHLCV via `yfinance`, caches to `data/*.csv`.
2. **`src/engine/features.py`** — builds ~20 "own" features per day from *only* past data (no
   look-ahead): lagged returns, SMA ratios (5/10/20/50d), rolling volatility, RSI(14), MACD,
   Bollinger Band position/width, volume change, day-of-week.
3. **`src/engine/macro_features.py`** — builds 24 cross-market/"world" features shared across all
   tickers: VIX level/z-score (S&P 500's own options-implied vol), VXN (Nasdaq-100 implied vol)
   for more implied-vol signal, 10Y Treasury yield level/change, the 10Y-vs-13-week yield curve
   slope (recession-risk signal), dollar index return, oil/gold returns, Dow/Nasdaq/Russell 2000
   returns, Nikkei/FTSE/DAX returns (how Asia and Europe already traded before the US session
   opens), Bitcoin return/volatility (trades weekends, so it reflects global news equities can't
   price in until Monday), a credit-stress proxy (high-yield bond ETF return minus Treasury bond
   ETF return), and QQQ return (tech-sector proxy). All same-day (close of day *t*), which is
   valid information for predicting day *t+1* — no look-ahead.

   **Temporarily missing**: the VIX/VIX3M term structure feature. On 2026-09-16, `^VIX3M`'s Yahoo
   feed started returning exactly 1 row (today's date only) regardless of period or retries,
   confirmed broken via two independent yfinance access paths — a genuine provider-side outage,
   not a bug in this code. It corrupted the cached history (data files are gitignored, so there
   was no backup to restore from) and collapsed the entire feature table for every ticker on the
   next backtest run, since a single all-NaN macro column makes `make_dataset`'s `dropna` remove
   every row. Fixed two ways: (1) `fetch_data.py` now refuses to overwrite a cached file with a
   fresh fetch that's suspiciously much shorter (`_save_with_outage_guard`) — this specific
   failure mode won't recur even after ^VIX3M's data resumes normally, for this or any other
   ticker; (2) the term-structure feature itself is commented out (not deleted) in `config.py`
   and `macro_features.py` until the feed recovers. Re-enabling it is a two-line uncomment.
4. **`src/engine/implied_vol.py`** — live single-stock options-implied-vol/skew snapshot for AAPL/PLTR
   (see "Options-implied volatility" below for why this is live-only, not trained).
5. **`src/engine/train.py`** — chronological (not shuffled) train/test split, last ~15% of days held out
   as the final, untouched evaluation set. For each ticker, trains and compares two feature sets —
   **baseline** (own features only, 23) and **enhanced** (own + macro/world, 47) — and for each,
   runs the optimization pipeline described below. Saves a test-period actual-vs-predicted chart to
   `reports/`, and full metrics to `reports/model_comparison.csv`.
6. **`src/engine/predict.py`** — recomputes features on the latest available day and prints tomorrow's
   predicted return, direction, reconstructed price, live news sentiment, and live implied vol/skew
   for context.
7. **`src/engine/news_pulse.py`** — live news-sentiment snapshot (see "World-events layer" below).
8. **`src/engine/baselines.py`** — finance-specific dumb baselines (momentum persistence), eligible to
   actually win model selection, not just serve as a floor (see the checklist below — this matters).
9. **`src/engine/leakage_check.py`** — rebuilds every feature from truncated history and verifies it's
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

### The classification decision threshold, and why it's not 0.50

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

**Update**: the margin was later raised from 0.51 to `CLASSIFICATION_THRESHOLD
= 0.52` (a 2-point shift off 50/50, still fixed and not fit to any test
window). This is a small extra safety buffer, not a re-attempt at the fix
above — the structural class-weighting fix below is what actually corrected
the always-UP bias; this threshold is a secondary, modest tightening on top
of it. Because the threshold feeds directly into model selection (classifiers
are scored on balanced accuracy *at this threshold*, not at 0.50), changing
it required retraining all 8 tickers, and did shift several: MSFT, GOOGL,
AMZN, NVDA, and META all moved from the persistence baseline to a tuned
`HistGradientBoosting` classifier at 0.52 where some had picked persistence
at 0.51 — a real, threshold-sensitive change in what wins, not noise.

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

1. **Leakage is the #1 silent killer.** Verified, not assumed: `src/engine/leakage_check.py`
   rebuilds every feature (own + macro) from data truncated at several cutoff
   dates and confirms each cutoff's last row is bit-for-bit identical to that
   date's row computed from the full history. If truncating the future ever
   changed a past feature value, that would be leakage. Run: `python src/engine/leakage_check.py`
   — currently passes for all three tickers, both feature families.
2. **Never shuffle time series data.** Already true throughout: `TimeSeriesSplit`
   for hyperparameter CV, a strictly chronological split for the final held-out
   test, nowhere a shuffled k-fold. Verified by grepping for `shift(-` (only the
   target uses it) and confirming no `KFold`/`shuffle` anywhere in the codebase.
3. **Predict returns, not price levels.** Already true from the start of this
   project (see "What it actually predicts" above) — targets are `pct_change`,
   never raw price.
4. **Always compare against a dumb baseline — and this one bit us.** Added a
   `baseline_persistence` candidate (`src/engine/baselines.py`: "tomorrow repeats
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

Two different things answer "analyze what's happening in the world":

- **Backtestable** (used by the trained models, via `macro_features.py`): global
  market indices (Nikkei/FTSE/DAX — how the rest of the world already traded
  before the US opens), Bitcoin (trades weekends, so it reflects news equities
  can't price in until Monday), the yield curve, VIX, credit spreads. These
  have full historical daily price archives, so they can be properly
  backtested with no look-ahead — same standard as every other feature here.
- **Live-only, shown as context, NOT trained into the model** (`news_pulse.py`,
  surfaced in `predict.py`'s `live_news_sentiment` column): current headlines
  for each ticker plus general market news, scored with VADER. `yfinance`'s
  news feed (and most free news sources) only return *current* headlines with
  no point-in-time historical archive, so nothing here can be backtested — see
  `news_features.py` below for the source that changed this.

**Update**: the NYT Archive API turned out to be a genuine exception to "no
free source has point-in-time history" — see the next section. `news_pulse.py`
stays as-is for the same reason `implied_vol.py` does: it reads *today's*
options chain / news feed, which has no historical archive of its own, so it's
still correctly live-only context, not a training input.

## NYT news-sentiment features (`news_features.py`)

Unlike `yfinance`'s news feed, the [NYT Archive API](https://developer.nytimes.com)
returns every article NYT published in a given month, *as NYT tagged it at the
time* — back to 1851, free, ~2,000 requests/day. That "as tagged at the time"
part is what makes it trainable: at row `t` the pipeline only ever uses
articles published on or before day `t`, so there's no look-ahead risk, unlike
a live-only feed with no historical record to anchor to.

**Two signals, both scored with VADER (headline + abstract):**
- `news_market_sentiment_mean` / `news_market_count`: daily average sentiment
  and article volume across Business/Technology-desk coverage generally — a
  shared "market mood" signal, applied to every ticker including SP500.
- `news_company_sentiment_mean` / `news_company_count` /
  `news_company_had_news`: same, filtered to articles that specifically name
  that ticker's company. Primary match is NYT's own `organizations` keyword
  tag (e.g. `"Apple Inc"`, `"Nvidia Corp"` — high precision), with a
  business/tech-context-only text fallback so a quiet-tagging month doesn't
  silently zero out the signal. Restricting the fallback to business/tech
  articles specifically avoids generic-word collisions (`"apple"` the fruit,
  `"amazon"` the rainforest) that a naive full-text search would hit.

**How it's evaluated**: added as a third A/B leg in `train.py`, alongside the
existing baseline-vs-macro comparison — `baseline` (own technical features)
vs. `with_macro` (+ cross-market features) vs. `with_macro_and_news` (+ NYT
sentiment) — and only kept per ticker/task if it actually wins on held-out
balanced accuracy / RMSE, same discipline as every other feature added here.
`leakage_check.py` verifies this the same way it verifies macro alignment:
rebuild the daily news aggregate from articles truncated to `pub_date <= d`
and confirm it's bit-for-bit identical to the same date's row built from the
full archive.

**Honest result**: unlike VIX3M/VXN and the world-market additions above, this
one actually moved the needle for some tickers — the first feature addition
since recency-weighting to do that. `macro+news` won outright for: **NVDA**
(both tasks — RMSE 0.0255→0.0231, balanced accuracy 52.7%→55.4%, the largest
single-feature jump seen in this project), **SP500**'s classifier (52.0%→54.4%
balanced accuracy), and the regressors for **MSFT** and **GOOGL** (small RMSE
improvements). It did *not* win for AAPL, PLTR, AMZN, or META, where baseline
or macro-only stayed ahead — consistent with this project's running finding
that no single feature category helps every ticker uniformly. Kept only where
it won, per the same rule as everything else here.

## Same-day prediction (predict today's 4pm ET close)

The pipeline above predicts *tomorrow's* close using *yesterday's* daily bar.
`train_intraday.py` / `predict_intraday.py` instead predict **today's** close using
intraday price action from earlier the same day — useful for a running estimate
throughout the trading session.

```bash
python src/engine/train_intraday.py     # trains the same-day-close model (run occasionally, e.g. weekly)
python src/engine/predict_intraday.py   # run any time 9:30am-4:00pm ET for a live estimate of today's close
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

`src/engine/long_horizon_drift.py` and `src/engine/predict_long_horizon.py` answer a
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

## Expected vs. actual return: using a year-old snapshot, checked against reality

A natural follow-up to the hit-rate work above: instead of just "was the
direction right," what if the algorithm had to commit to an actual expected
*return*, and that commitment was checked against what really happened?
`expected_vs_actual_return()` in `long_horizon_drift.py` does exactly that,
walk-forward and with no lookahead:

1. Pick an anchor ~1 year (252 trading days) back from today.
2. Using **only price history strictly before that anchor date**, compute a
   point estimate of the historical forward return at 3/6/9/12 months — "what
   the algorithm would have expected," formed with the same information that
   would genuinely have been available at that point in time.
3. Compare to the **actual** return realized over that exact window — fully
   knowable now, since even the 12-month window ends today.

Run: `python src/engine/long_horizon_drift.py` (prints this after the hit-rate
tables; saved to `reports/expected_vs_actual_return.csv`).

### Reducing the error: median beats mean, exactly as forecasting theory predicts

The first version of this check used the raw historical **mean** forward
return as the expectation, giving 22/32 (68.8%) direction matches but a mean
absolute error of **27.85 percentage points** — a large miss even when the
sign was right. Checking published forecasting theory rather than guessing at
a fix: minimizing MAE means forecasting the **median**, not the mean —
minimizing RMSE is what targets the mean instead (Hyndman & Athanasopoulos,
*Forecasting: Principles and Practice*). Financial returns are also
fat-tailed, and Kaggle's own competition write-ups (Optiver, Jane Street) use
trimmed means and Tukey-IQR winsorization as standard countermeasures for
exactly that. `_point_estimators()` now computes all four from the same
historical sample and reports which actually wins — not assumed, tested:

| Estimator | MAE (pct points) | Direction match rate |
|---|---|---|
| **median** | **24.99** | 65.6% |
| trimmed mean (10%) | 26.05 | 65.6% |
| winsorized mean (Tukey IQR) | 27.47 | 65.6% |
| mean (original) | 27.78 | 65.6% |

Median wins, as the theory says it should — a ~10% relative reduction in
error, with PLTR's worst case (9-month horizon) improving from a 95-point
miss down to a -0.55pt miss on the 3-month row specifically, because the
median isn't dragged around by PLTR's handful of explosive historical
quarters the way the mean is. Direction-match rate barely moves (65.6% vs.
68.8%) — this fix targets *magnitude* error, which is a different claim from
*direction* accuracy, and it was never going to move the second one much.

Per-ticker detail using the median estimator (anchor ≈ 2025-09-12):

| Ticker | 3mo error | 6mo error | 9mo error | 12mo error | Direction matched |
|---|---|---|---|---|---|
| SP500 | +0.5pts | -5.6pts | +4.4pts | +0.9pts | 4/4 |
| AAPL | +12.2pts | -4.5pts | +5.6pts | +15.8pts | 4/4 |
| GOOGL | +23.3pts | +17.7pts | +37.0pts | +19.5pts | 4/4 |
| AMZN | -5.7pts | -17.9pts | -8.6pts | -14.6pts | 3/4 |
| NVDA | -14.1pts | -31.5pts | -34.9pts | -61.8pts | 4/4 |
| MSFT | -12.6pts | -37.5pts | -47.1pts | -36.2pts | 0/4 |
| PLTR | -0.6pts | -33.7pts | -81.2pts | -78.7pts | 2/4 |
| META | -20.7pts | -30.0pts | -45.0pts | -40.7pts | 0/4 |

The pattern from before still holds even with the better estimator: SP500 and
AAPL stay well-calibrated, GOOGL's errors are large but one-directional
(consistently under-estimating a stronger-than-typical year), and
MSFT/META/PLTR still carry errors of 30-80+ points at the longer horizons —
this wasn't a bug in the estimator, it's those tickers genuinely breaking
from their own historical pattern in this specific year. No estimator choice
fixes that; median just stops the *typical* case from being thrown off by the
*extreme* historical cases the way a raw mean does.

This remains the sharpest illustration in this README of why a hit-rate
percentage alone understates the risk: even "4/4 direction matched" tickers
(SP500, AAPL, GOOGL, NVDA) carry errors of 5 to 62 percentage points on the
actual number. Getting the sign right and getting the magnitude right are
different claims, and no amount of estimator-tuning changes that — it only
improves how wrong the wrong ones are.

## How this compares to published research

Before treating this project's ~50-58% ceiling as specific to its own
methodology, it's worth checking what the broader field actually finds — not
from memory, but by reading the papers. There turn out to be two very
different tiers of published result:

**Tier 1 — papers claiming 80-93%+ directional accuracy.** These are common:
a Vietnamese-market LSTM at 93%, an S&P 500 study at 83.6%, Random Forest at
91.3%, ANNs averaging 83.4% across G-7 indices. On the surface these make
this project's numbers look mediocre.

**Tier 2 — rigorous, walk-forward-validated studies converge almost exactly
on this project's numbers.** A controlled comparison of 918 experiments
across transformer, TCN, and LSTM architectures for financial forecasting,
using proper walk-forward validation, found directional accuracy essentially
flat at 50.08% — "no combination deviating meaningfully from 50%," across
every model and horizon tested. A separate study found predictive accuracy
normally distributed around a 52% mean. An LSTM on the Brazilian exchange
topped out at 55.9%. SP500 (~52%), AAPL (~51%), and PLTR (~58%) in this
project land right inside that same band.

**Why the gap between tiers — checked directly, not assumed**: the Tier 1
papers tend to have exactly the flaws this project was built to avoid.
Random (non-time-series) train/test splits are a documented, common source
of leakage in financial ML papers — precisely what `TimeSeriesSplit` and
chronological splitting throughout `train.py` prevent, and what
`leakage_check.py` *verifies* rather than assumes. LSTM is repeatedly flagged
in the literature as unstable and overfitting-prone on financial data ("as
epoch increases, training accuracy improves, but validation accuracy stays
constant"); one widely-cited headline result (R² = 0.997) is called out by
reviewers as a red flag, not a success — a near-perfect fit on stock data
usually means the model is trivially tracking yesterday's price *level*, the
exact trap this project avoided from its first design decision (predict
returns, not price levels — see "What it actually predicts" above). And
backtest-to-live-trading failures are broadly attributed to unrealistic
execution assumptions (no slippage, perfect fills) and no cost adjustment —
exactly why `backtest_dates.py`'s transaction-cost reality check exists, and
exactly why it's shown real, uncomfortable negative numbers here (AAPL net-negative
across multiple checked windows this session).

**Where real hedge funds fit in**: Renaissance Technologies' Medallion Fund
returns roughly 66% gross annually — but that's *return*, not *directional
accuracy*, earned through massive diversification and leverage across
probably thousands of individually weak (~51-55%) signals, not high-confidence
single-stock calls. That's a fundamentally different strategy than "predict
AAPL's direction tomorrow with high accuracy," and it's the strongest
available evidence that professional practice treats a thin, diversified
edge — not a strong single-name signal — as the realistic path to profit.

**Conclusion**: this project's results aren't a sign of a weaker
implementation. They match the most rigorously validated published findings,
and by several concrete measures (leakage verification, walk-forward
validation, cost-adjusted reality checking, an eligible-to-win dumb baseline)
this pipeline is more methodologically careful than a meaningful share of the
literature claiming much higher numbers.

Sources: [How effective is machine learning in stock market predictions? (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10826674/) ·
[A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: 918 Experiments](https://arxiv.org/pdf/2603.16886) ·
[Validating Weak-form Market Efficiency in US Stock Markets with Trend Deterministic Price Data and ML](https://arxiv.org/pdf/1909.05151) ·
[Applying machine learning algorithms to predict the stock price trend — Vietnam (Nature)](https://www.nature.com/articles/s41599-024-02807-x) ·
[Over-Fit or Not: That is the Question (Springer)](https://link.springer.com/chapter/10.1007/978-981-96-7742-9_26) ·
[Renaissance Tech and Two Sigma lead 2024 quant gains (Hedgeweek)](https://www.hedgeweek.com/renaissance-tech-and-two-sigma-lead-2024-quant-gains/) ·
[Common Backtesting Mistakes — Why Strategies Fail in Live Trading (Gainium)](https://gainium.io/blog/common-backtesting-problems) ·
[Why 90% of Profitable Backtests Are Statistically Invalid (Medium)](https://daviddtech.medium.com/the-three-deadly-sins-of-backtesting-overfitting-look-ahead-bias-and-p-hacking-a68c6345e668)

## Recency-weighted training: a genuinely new lever, not a rehash

Every improvement attempt up to this point — hyperparameter tuning, feature
selection, ensembling, macro/world-market features, implied volatility,
class-weighting — had been tried and landed in the same ~50-58% band. Rather
than tune something already shown not to matter much, this pass added a lever
untested elsewhere in the project: **recency weighting**. Markets are
non-stationary — a relationship that held in 2016 isn't guaranteed to hold in
2026 — so `recency_weights()` in `train.py` exponentially down-weights older
training rows (half-life 504 trading days, ~2 years) instead of treating a
decade of history as equally relevant. Recency-weighted Random Forest and
HistGradientBoosting variants were added as two *more* candidates competing
in the same held-out-test selection every other model already goes through —
not assumed to help, tested.

**Result: a genuine, if modest, improvement.** Average held-out balanced
accuracy across all 8 tickers moved from ~52.8% to ~53.2%. The gain wasn't
spread evenly — it concentrated where it should, in tickers whose recent
behavior differs most from their decade-long average:

| Ticker | Before | After | Winning candidate |
|---|---|---|---|
| AMZN | 50.4% | **54.5%** | `random_forest_recency_weighted` |
| NVDA | 53.6% | **54.9%** | (weighted variant competitive, unweighted RF still won here) |
| META | 52.8% | 53.3% | `random_forest_recency_weighted` |
| SP500 / AAPL / PLTR / MSFT / GOOGL | ~unchanged | ~unchanged | mostly still `baseline_persistence` |

AMZN is the clearest case: its recency-weighted classifier beat every other
candidate including persistence, a first for that ticker. This doesn't change
the project's overall conclusion — 53.2% is still solidly inside the same
range published research and real hedge funds report, not a breakthrough —
but it's a real, measured gain from a lever that hadn't been pulled yet,
which is different from the many additions before it that tested and didn't
move the needle.

## Descriptive ticker comparison (Part 2)

`src/comparison/compare_tickers.py` is Part 2 of the project: a single,
neutral, side-by-side table comparing all eight tickers on the numbers that
actually matter, all pulled from Part 1's own outputs — it trains nothing new.
(Example tables below show a subset of rows for brevity, not the full eight.)

```
                       held_out_test_rmse   held_out_test_balanced_accuracy   recent_accuracy   recent_net_bps_per_trade   long_horizon_call
SP500                  0.0104                0.5209                          0.4333            +8.1                       UP over 2 months (86.6% OOS)
AAPL                   0.0193                0.5110                          0.5000            -23.8                      UP over 6 months (97.6% OOS, ~1 sample)
PLTR                   0.0394                0.5806                          0.5333            +29.9                      none reliable
```

Three different numbers, three different meanings, deliberately kept
separate rather than blended into one score:

- **`held_out_test_*`** — the most trustworthy number (hundreds of days),
  freshly re-evaluated against each ticker's actual deployed model.
- **`recent_*`** — the last 30 trading days. Small sample, genuinely noisy
  (see the standard-error math earlier in this README) — don't read a
  30-day swing as the model getting better or worse.
- **`long_horizon_*`** — a *different, much weaker* claim (buy-and-hold
  drift over months, not day-to-day prediction; see "The 80-90% question").
  Blending this in with the daily numbers would be comparing two different
  questions as if they were one.

The tool also prints (and saves to `reports/four_horizon_comparison.csv`) the
out-of-sample UP hit rate at exactly **1 month, 3 months, 6 months, and 1
year**, per ticker — a fixed horizon breakdown reusing `long_horizon_drift.py`'s
own walk-forward analysis rather than computing anything new:

```
horizon              1 month            3 months            6 months               1 year
AAPL     63.8%  [~16 indep.]  79.0%  [~4 indep.]  97.6%  [~1 indep.]  100.0%  [~0 indep.]
PLTR      39.7%  [~9 indep.]  27.2%  [~2 indep.]  29.3%  [~0 indep.]                  n/a
SP500    74.2%  [~16 indep.]  90.4%  [~4 indep.]  95.2%  [~1 indep.]  100.0%  [~0 indep.]
```

The `[~N indep.]` figure is the point of showing this at all: it's the
effective *independent* sample count once overlapping windows are accounted
for, and it shrinks toward zero exactly where the headline percentages look
most impressive (6-12 months). PLTR's numbers *decline* at longer horizons
instead of climbing — a genuine, structurally different result, not
something smoothed over to make the table look consistent across tickers.
This table was requested as an input to "optimize the investment strategy" —
that specific framing was declined for the same reason as the ticker ranking
above: it's investment advice regardless of which horizons back it. What's
shown here is the same descriptive statistic, at the four horizons asked
for, with nothing built on top of it that says what to do with it.

The tool also prints (and saves to `reports/recent_3month_check.csv`) a
direct answer to "run the algorithm on data from 3 months ago": the single
most recent completed 3-month window for each ticker — price 63 trading days
ago vs. today — checked against what actually happened:

```
         start_date    end_date  start_price  end_price  return_pct predicted  correct
SP500    2026-06-12  2026-09-14     7,431.46    7,619.98      +2.54%        UP     True
AAPL     2026-06-11  2026-09-11       295.38      332.27     +12.49%        UP     True
GOOGL    2026-06-11  2026-09-11       357.54      338.50      -5.33%        UP    False
...      7/8 tickers were actually UP over their most recent 3-month window
```

This is **one instance per ticker, not a statistic** — read it as an anecdote
about what just happened, not as validated evidence of a rate; the aggregate
hit-rate table above it is the actual evidence, thin as that evidence is at
these horizons. Building this surfaced a real data-quality bug worth noting:
Yahoo Finance occasionally returns a trailing row with `Volume` populated but
`Close`/`Open`/`High`/`Low` still `NaN` — a not-yet-settled bar for an
individual stock, sometimes hours after that same day's index-level close
had already posted. `fetch_data.py` now drops any row with a `NaN` close at
the source, so every downstream consumer is protected rather than each one
needing its own defense against it.

The tool also prints raw fundamental data (`reports/fundamentals_comparison.csv`)
— trailing/forward P/E, market cap, price-to-book, dividend yield, beta,
profit margin, revenue growth, sector, 52-week range — pulled directly from
`yfinance`, with no verdict attached and nothing labeled "cheap" or
"expensive":

```
        trailing_pe  forward_pe  market_cap_billions  price_to_book  profit_margin_pct  revenue_growth_pct
SP500           n/a         n/a                   n/a            n/a                n/a                 n/a   (index, not a company)
AAPL          38.39       34.90              4,880.59          45.44              27.62               16.40
PLTR         147.57       74.23                414.91          42.44              49.01               92.80
```

SP500 is genuinely `n/a` for nearly every column — an index isn't a company
and structurally has no P/E or market cap, which is different from a missing
data point that should be filled in. This was added after a request to
recommend specific stocks to buy, phrased as wanting "a good company cheap,
not a good company at an expensive price" — which was declined (see below)
independent of whether the ticker list was widened, because giving
investment advice isn't something that changes with scope or phrasing. What
*can* be given honestly is the raw numbers a value-investing judgment would
actually be based on, with the judgment itself left to the reader — PLTR's
147x trailing P/E next to AAPL's 38x is shown here exactly as data, not as
an answer to "which one is the better buy."

**Why this exists, and why it stops exactly here**: a user asked this
project to "give advice for which stocks we should invest in." That's
personalized investment advice, which this project does not provide — not as
a policy choice that could be relaxed, but because it isn't something to
give regardless of how the request is framed. What *is* useful and honest is
letting someone see the same backtested numbers the rest of this README
already reports, side by side instead of scattered across sections. The
table above is exactly that and nothing more: it does not rank tickers by
"best," does not output a buy/sell/hold call, and every run reprints the
same disclaimer before and after the table. If a ticker looks better on one
row here, that is a historical statistic, not a forecast, and specifically
not a recommendation — the whole rest of this README is the explanation of
why that distinction actually matters here, not boilerplate.

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
