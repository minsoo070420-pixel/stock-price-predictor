from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# yfinance ticker -> friendly name used for files/output
TICKERS = {
    "^GSPC": "SP500",
    "AAPL": "AAPL",
    "PLTR": "PLTR",
}

# Cross-market / macro tickers used as extra features (shared across all three
# targets above) -- volatility, rates, currency, commodities, other indices,
# a credit-stress proxy, and a tech-sector proxy relevant to AAPL/PLTR.
MACRO_TICKERS = {
    "^VIX": "VIX",     # equity volatility / fear gauge
    "^TNX": "TNX",     # 10-year Treasury yield
    "UUP": "DXY",      # US dollar index proxy (ETF)
    "CL=F": "OIL",     # WTI crude oil
    "GC=F": "GOLD",    # gold
    "^DJI": "DJI",     # Dow Jones Industrial Average
    "^IXIC": "IXIC",   # Nasdaq Composite
    "^RUT": "RUT",     # Russell 2000 (small caps)
    "HYG": "HYG",      # high-yield corporate bond ETF
    "IEF": "IEF",      # 7-10yr Treasury bond ETF (HYG/IEF spread = credit stress proxy)
    "QQQ": "QQQ",      # Nasdaq-100 ETF, tech-sector proxy
    "^IRX": "IRX",     # 13-week T-bill yield (short end, for yield curve slope)
    "^N225": "NIKKEI", # Japan -- how Asia traded overnight before the US session
    "^FTSE": "FTSE",   # UK -- how Europe traded before the US session
    "^GDAXI": "DAX",   # Germany -- ditto
    "BTC-USD": "BTC",  # trades 24/7 incl. weekends -- captures risk sentiment/news
                       # equities can't price in until the next session
}

# Hyperparameter search / feature-selection settings for train.py
HYPERPARAM_CV_SPLITS = 3       # TimeSeriesSplit folds used only for tuning, never for the final reported metric
FEATURE_SELECT_TOP_K = 20      # when a feature set exceeds this, keep only the top-K by importance

HISTORY_PERIOD = "10y"   # how much history to download
INTERVAL = "1d"
TEST_FRACTION = 0.15     # last 15% of trading days held out as test set, chronologically
RANDOM_STATE = 42

# Intraday (same-day close prediction) settings.
# Yahoo Finance retention limits: 60m bars ~730 days, 5m bars ~60 days.
# We train on the longer hourly history and feed the model fresh 5-minute
# snapshots at prediction time (features are ratio/time-based, not tied to bar count).
INTRADAY_TRAIN_INTERVAL = "60m"
INTRADAY_TRAIN_PERIOD = "730d"
INTRADAY_LIVE_INTERVAL = "5m"
INTRADAY_LIVE_PERIOD = "5d"
