"""Cross-market / macro features shared across all three prediction targets.

Each series contributes same-day (close-of-day t) information, which is valid
for predicting day t+1 -- there is no look-ahead as long as these are joined
onto a ticker's own feature table by date and used only to predict the next day.
"""
import numpy as np
import pandas as pd

MACRO_FEATURE_COLUMNS = [
    "vix_level", "vix_zscore", "vix_ret_1d",
    "tnx_level", "tnx_chg_1d_bp",
    "yield_curve_slope", "yield_curve_chg_1d_bp",
    "dxy_ret_1d",
    "oil_ret_1d", "oil_vol_10d",
    "gold_ret_1d",
    "dji_ret_1d", "ixic_ret_1d", "rut_ret_1d",
    "credit_stress", "credit_stress_chg_1d",
    "qqq_ret_1d",
    "nikkei_ret_1d", "ftse_ret_1d", "dax_ret_1d",
    "btc_ret_1d", "btc_vol_10d",
    "vix_term_structure", "vix_term_structure_chg_1d",
    "vxn_level", "vxn_zscore", "vxn_ret_1d",
]


def _close(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


def build_macro_features(macro_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    vix = _close(macro_data["VIX"])
    tnx = _close(macro_data["TNX"])
    dxy = _close(macro_data["DXY"])
    oil = _close(macro_data["OIL"])
    gold = _close(macro_data["GOLD"])
    dji = _close(macro_data["DJI"])
    ixic = _close(macro_data["IXIC"])
    rut = _close(macro_data["RUT"])
    hyg = _close(macro_data["HYG"])
    ief = _close(macro_data["IEF"])
    qqq = _close(macro_data["QQQ"])
    irx = _close(macro_data["IRX"])
    nikkei = _close(macro_data["NIKKEI"])
    ftse = _close(macro_data["FTSE"])
    dax = _close(macro_data["DAX"])
    btc = _close(macro_data["BTC"])
    vix3m = _close(macro_data["VIX3M"])
    vxn = _close(macro_data["VXN"])

    out = pd.DataFrame(index=vix.index)
    out["vix_level"] = vix
    out["vix_zscore"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
    out["vix_ret_1d"] = vix.pct_change()

    out["tnx_level"] = tnx
    out["tnx_chg_1d_bp"] = tnx.diff() * 10  # ^TNX is quoted in tenths of a percent; *10 = basis points

    yield_curve = tnx - irx  # 10y minus 13-week: negative/flat = classic recession-risk signal
    out["yield_curve_slope"] = yield_curve
    out["yield_curve_chg_1d_bp"] = yield_curve.diff() * 10

    out["dxy_ret_1d"] = dxy.pct_change()

    oil_ret = oil.pct_change()
    out["oil_ret_1d"] = oil_ret
    out["oil_vol_10d"] = oil_ret.rolling(10).std()

    out["gold_ret_1d"] = gold.pct_change()

    out["dji_ret_1d"] = dji.pct_change()
    out["ixic_ret_1d"] = ixic.pct_change()
    out["rut_ret_1d"] = rut.pct_change()

    credit_stress = hyg.pct_change() - ief.pct_change()  # risk-on/off proxy: high-yield vs. Treasury returns
    out["credit_stress"] = credit_stress
    out["credit_stress_chg_1d"] = credit_stress.diff()

    out["qqq_ret_1d"] = qqq.pct_change()

    # Overseas sessions close before the US session opens -- their same-calendar-day
    # return is genuinely known before the US market trades that day.
    out["nikkei_ret_1d"] = nikkei.pct_change()
    out["ftse_ret_1d"] = ftse.pct_change()
    out["dax_ret_1d"] = dax.pct_change()

    # BTC trades every day including weekends, so its Monday return already reflects
    # anything that happened globally over the weekend that equities can't price until Monday.
    btc_ret = btc.pct_change()
    out["btc_ret_1d"] = btc_ret
    out["btc_vol_10d"] = btc_ret.rolling(10).std()

    # Options-implied volatility (genuinely forward-looking, unlike everything else
    # above which is derived from past prices): VIX is the S&P 500's own 30-day
    # implied vol, so it already is an options-implied-vol feature. VIX3M/VIX gives
    # the vol term structure -- >1 (contango) is the normal state; <1 (backwardation,
    # near-term IV pricier than farther-out) is a well-known near-term-stress signal.
    # VXN is the Nasdaq-100's equivalent, more relevant to AAPL/PLTR specifically.
    term_structure = vix3m / vix
    out["vix_term_structure"] = term_structure
    out["vix_term_structure_chg_1d"] = term_structure.diff()

    out["vxn_level"] = vxn
    out["vxn_zscore"] = (vxn - vxn.rolling(60).mean()) / vxn.rolling(60).std()
    out["vxn_ret_1d"] = vxn.pct_change()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def align_macro_to_ticker(macro_df: pd.DataFrame, ticker_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex macro features onto a ticker's own trading calendar, forward-filling
    small mismatches (e.g. futures settle on days equities don't trade)."""
    idx = pd.to_datetime(ticker_index).tz_localize(None).normalize()
    aligned = macro_df.reindex(idx.union(macro_df.index)).sort_index().ffill()
    aligned = aligned.reindex(idx)
    aligned.index = ticker_index
    return aligned
