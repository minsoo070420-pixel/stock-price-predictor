"""Feature engineering: turn raw OHLCV history into a model-ready table.

Every feature at row t is computed only from data up to and including day t
(no look-ahead). The targets predict day t+1 from day t's features:
  - target_return: next day's close-to-close percent return
  - target_direction: 1 if next day's return > 0 else 0
"""
import numpy as np
import pandas as pd

from macro_features import MACRO_FEATURE_COLUMNS, align_macro_to_ticker

OWN_FEATURE_COLUMNS = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d",
    "sma_5_ratio", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "vol_5d", "vol_10d", "vol_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width",
    "vol_change", "vol_ratio_20d",
    "hl_range",
    "dow_sin", "dow_cos",
]

FEATURE_COLUMNS = OWN_FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    out = pd.DataFrame(index=df.index)

    daily_ret = close.pct_change()
    out["ret_1d"] = daily_ret
    out["ret_2d"] = close.pct_change(2)
    out["ret_3d"] = close.pct_change(3)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)

    for w in (5, 10, 20, 50):
        out[f"sma_{w}_ratio"] = close / close.rolling(w).mean() - 1

    for w in (5, 10, 20):
        out[f"vol_{w}d"] = daily_ret.rolling(w).std()

    out["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / close
    out["macd_signal"] = macd_signal / close
    out["macd_hist"] = (macd - macd_signal) / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid

    out["vol_change"] = volume.pct_change()
    out["vol_ratio_20d"] = volume / volume.rolling(20).mean()

    out["hl_range"] = (high - low) / close

    dow = df.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Targets: next trading day, derived from data NOT available at feature time.
    out["target_return"] = daily_ret.shift(-1)
    out["target_direction"] = (out["target_return"] > 0).astype(int)
    out["close"] = close

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def make_dataset(df: pd.DataFrame, macro_df: pd.DataFrame | None = None, feature_columns: list[str] | None = None):
    """Return (X, y_return, y_direction, close) aligned and with NaNs dropped.

    If macro_df is given (see macro_features.build_macro_features), its columns
    are joined on by date -- aligned to this ticker's own trading calendar --
    and included alongside the ticker's own technical features. `feature_columns`
    controls which columns end up in X (default: OWN + macro if macro_df is
    given, else OWN only) -- pass it explicitly to compare feature sets.
    """
    feats = build_features(df)
    if macro_df is not None:
        aligned = align_macro_to_ticker(macro_df, feats.index)
        feats = feats.join(aligned)

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS if macro_df is not None else OWN_FEATURE_COLUMNS

    feats = feats.dropna(subset=feature_columns + ["target_return", "target_direction"])
    X = feats[feature_columns]
    y_return = feats["target_return"]
    y_direction = feats["target_direction"]
    close = feats["close"]
    return X, y_return, y_direction, close
