# src/features.py

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features required for the machine learning and anomaly analysis pipeline.

    The feature set includes:
    - Daily and market returns
    - Forward (next-day) returns
    - Binary target variable (outperformance)
    - Rolling volatility and moving averages
    - Classical calendar effects (day-of-week, month, turn-of-month, sell-in-May)
    - Extended calendar anomalies (Christmas, Thanksgiving, New Year, First day of quarter)
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Basic returns
    # ------------------------------------------------------------------
    df["daily_return"] = df.groupby("ticker")["close_price"].pct_change()
    df["market_return"] = df["SP500_Close"].pct_change()

    # ------------------------------------------------------------------
    # 2. Next-day returns
    # ------------------------------------------------------------------
    df["daily_return_tomorrow"] = df.groupby("ticker")["daily_return"].shift(-1)
    df["market_return_tomorrow"] = df["market_return"].shift(-1)

    # ------------------------------------------------------------------
    # 3. Binary outperformance indicator (based on next-day returns)
    # ------------------------------------------------------------------
    df["outperform_tomorrow"] = (
        df["daily_return_tomorrow"] > df["market_return_tomorrow"]
    ).astype(int)
    df["outperform"] = df["outperform_tomorrow"]

    # ------------------------------------------------------------------
    # 4. Rolling volatility (20 trading days)
    # ------------------------------------------------------------------
    df["volatility_20d"] = (
        df.groupby("ticker")["daily_return"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------------
    # 5. Moving average (20 trading days)
    # ------------------------------------------------------------------
    df["ma20"] = (
        df.groupby("ticker")["close_price"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------------
    # 6. Classical calendar effects
    # ------------------------------------------------------------------
    df["day_of_week"] = df["Date"].dt.weekday
    df["month"] = df["Date"].dt.month
    df["turn_of_month"] = (df["Date"].dt.day <= 3).astype(int)
    df["sell_in_may"] = df["Date"].dt.month.isin([5, 6, 7, 8, 9, 10]).astype(int)

    # Pre-holiday effect (day preceding selected U.S. holidays)
    holidays = [
        "2020-12-24", "2021-12-24", "2022-12-23", "2023-12-25",
        "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-02",
    ]
    holidays = pd.to_datetime(holidays)
    df["pre_holiday"] = df["Date"].isin(holidays - pd.Timedelta(days=1)).astype(int)

    # ------------------------------------------------------------------
    # 7. Extended calendar anomalies
    # ------------------------------------------------------------------

    # Christmas effect (December 25)
    df["is_christmas"] = (
        (df["Date"].dt.month == 12) &
        (df["Date"].dt.day == 25)
    ).astype(int)

    # New Year effect (January 1)
    df["is_new_year"] = (
        (df["Date"].dt.month == 1) &
        (df["Date"].dt.day == 1)
    ).astype(int)

    # Thanksgiving effect (fourth Thursday of November)
    is_november = df["Date"].dt.month == 11
    is_thursday = df["Date"].dt.weekday == 3
    is_fourth_week = df["Date"].dt.day.between(22, 28)
    df["is_thanksgiving"] = (is_november & is_thursday & is_fourth_week).astype(int)

    # First day of the quarter (January 1, April 1, July 1, October 1)
    df["is_first_day_quarter"] = (
        (df["Date"].dt.day == 1) &
        (df["Date"].dt.month.isin([1, 4, 7, 10]))
    ).astype(int)

    return df
git add src/features.py