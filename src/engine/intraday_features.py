"""Build 'as of time T during the trading day' snapshots that predict the 4pm close.

Every historical trading day contributes several snapshot rows: one for each
intraday bar time T, using only data known up to and including T. The target
is always the day's actual closing price, expressed as the remaining return
from price(T) to close. Features are deliberately scale/resolution agnostic
(ratios, minutes-until-close) so a model trained on hourly bars still works
when fed a live 5-minute snapshot at prediction time.
"""
import numpy as np
import pandas as pd

MARKET_CLOSE = (16, 0)   # 4:00pm ET
MARKET_OPEN = (9, 30)    # 9:30am ET
SESSION_MINUTES = 390    # 6.5 hours

SNAPSHOT_FEATURE_COLUMNS = [
    "minutes_until_close",
    "frac_of_day_elapsed",
    "ret_since_open",
    "ret_last_bar",
    "pos_in_range",
    "range_so_far_pct",
    "gap_from_prev_close",
    "prev_day_return",
    "vol_ratio_vs_typical",
    "dow_sin",
    "dow_cos",
]


def _minutes_until_close(ts: pd.Timestamp) -> float:
    close_ts = ts.normalize() + pd.Timedelta(hours=MARKET_CLOSE[0], minutes=MARKET_CLOSE[1])
    return max((close_ts - ts).total_seconds() / 60.0, 0.0)


def _prep_daily_context(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Per-day context available without look-ahead: previous close, previous
    day's return, and a trailing 20-day average daily volume (shifted so the
    current day's own volume never leaks into its own typical-volume baseline)."""
    d = daily_df.copy()
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    prev_close = d["Close"].shift(1)
    prev_day_return = d["Close"].pct_change().shift(1)
    avg_volume_20d = d["Volume"].rolling(20).mean().shift(1)
    ctx = pd.DataFrame({
        "prev_close": prev_close,
        "prev_day_return": prev_day_return,
        "avg_volume_20d": avg_volume_20d,
    })
    return ctx


def build_snapshots(intraday_df: pd.DataFrame, daily_df: pd.DataFrame, include_last_bar: bool = False) -> pd.DataFrame:
    """Return one row per (day, bar) snapshot with features + target_return_to_close."""
    df = intraday_df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    df["day"] = df.index.tz_localize(None).normalize()

    ctx = _prep_daily_context(daily_df)

    rows = []
    for day, day_bars in df.groupby("day"):
        day_bars = day_bars.sort_index()
        n = len(day_bars)
        if n < 3:
            continue
        if day not in ctx.index or pd.isna(ctx.loc[day, "prev_close"]):
            continue

        prev_close = ctx.loc[day, "prev_close"]
        prev_day_return = ctx.loc[day, "prev_day_return"]
        avg_vol_20d = ctx.loc[day, "avg_volume_20d"]

        day_open = float(day_bars["Open"].iloc[0])
        day_close = float(day_bars["Close"].iloc[-1])
        gap_from_prev_close = day_open / prev_close - 1

        last_idx = n if include_last_bar else n - 1
        for i in range(1, last_idx):
            window = day_bars.iloc[: i + 1]
            ts = window.index[-1]
            price_now = float(window["Close"].iloc[-1])
            high_so_far = float(window["High"].max())
            low_so_far = float(window["Low"].min())
            rng = high_so_far - low_so_far
            pos_in_range = (price_now - low_so_far) / rng if rng > 0 else 0.5

            minutes_left = _minutes_until_close(ts)
            frac_elapsed = 1 - minutes_left / SESSION_MINUTES

            cum_volume = float(window["Volume"].sum())
            typical_cum_volume = avg_vol_20d * frac_elapsed if pd.notna(avg_vol_20d) and frac_elapsed > 0 else np.nan
            vol_ratio = cum_volume / typical_cum_volume if typical_cum_volume and typical_cum_volume > 0 else np.nan

            dow = ts.dayofweek
            rows.append({
                "date": day,
                "timestamp": ts,
                "minutes_until_close": minutes_left,
                "frac_of_day_elapsed": frac_elapsed,
                "ret_since_open": price_now / day_open - 1,
                "ret_last_bar": price_now / float(window["Close"].iloc[-2]) - 1,
                "pos_in_range": pos_in_range,
                "range_so_far_pct": rng / price_now if price_now else np.nan,
                "gap_from_prev_close": gap_from_prev_close,
                "prev_day_return": prev_day_return,
                "vol_ratio_vs_typical": vol_ratio,
                "dow_sin": np.sin(2 * np.pi * dow / 5),
                "dow_cos": np.cos(2 * np.pi * dow / 5),
                "price_now": price_now,
                "day_close": day_close,
                "target_return_to_close": day_close / price_now - 1,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.set_index("timestamp")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=SNAPSHOT_FEATURE_COLUMNS + ["target_return_to_close"])
    return out


def build_live_snapshot(intraday_today: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame | None:
    """Build a single-row feature snapshot 'as of now' from partial intraday
    bars for the current trading day. Returns None if there isn't enough
    data yet (e.g. market just opened) or no prior-day context."""
    df = intraday_today.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    if len(df) < 2:
        return None

    today = df.index[-1].tz_localize(None).normalize()
    ctx = _prep_daily_context(daily_df)
    if today not in ctx.index or pd.isna(ctx.loc[today, "prev_close"]):
        return None

    prev_close = ctx.loc[today, "prev_close"]
    prev_day_return = ctx.loc[today, "prev_day_return"]
    avg_vol_20d = ctx.loc[today, "avg_volume_20d"]

    day_open = float(df["Open"].iloc[0])
    ts = df.index[-1]
    price_now = float(df["Close"].iloc[-1])
    high_so_far = float(df["High"].max())
    low_so_far = float(df["Low"].min())
    rng = high_so_far - low_so_far
    pos_in_range = (price_now - low_so_far) / rng if rng > 0 else 0.5

    minutes_left = _minutes_until_close(ts)
    frac_elapsed = 1 - minutes_left / SESSION_MINUTES

    cum_volume = float(df["Volume"].sum())
    typical_cum_volume = avg_vol_20d * frac_elapsed if pd.notna(avg_vol_20d) and frac_elapsed > 0 else np.nan
    vol_ratio = cum_volume / typical_cum_volume if typical_cum_volume and typical_cum_volume > 0 else np.nan

    dow = ts.dayofweek
    row = {
        "minutes_until_close": minutes_left,
        "frac_of_day_elapsed": frac_elapsed,
        "ret_since_open": price_now / day_open - 1,
        "ret_last_bar": price_now / float(df["Close"].iloc[-2]) - 1,
        "pos_in_range": pos_in_range,
        "range_so_far_pct": rng / price_now if price_now else np.nan,
        "gap_from_prev_close": day_open / prev_close - 1,
        "prev_day_return": prev_day_return,
        "vol_ratio_vs_typical": vol_ratio,
        "dow_sin": np.sin(2 * np.pi * dow / 5),
        "dow_cos": np.cos(2 * np.pi * dow / 5),
    }
    out = pd.DataFrame([row], index=[ts])
    out.attrs["price_now"] = price_now
    out.attrs["as_of"] = ts
    return out
