# src/features.py

import pandas as pd
import numpy as np
import holidays as holidays_lib


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature set used for:
    - calendar anomaly analysis
    - machine learning models

    Expected minimum columns:
        ["Date", "ticker", "price"] (+ "sector" if available)
    """

    df = df.copy()

    # --------------------------------------------------------------
    # 0) Basic validation and harmonization
    # --------------------------------------------------------------
    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    if "ticker" not in df.columns:
        raise ValueError("Missing required column: 'ticker'.")
    if "price" not in df.columns:
        raise ValueError("Missing required column: 'price'.")
    if "Date" not in df.columns:
        raise ValueError("Missing required column: 'Date'.")

    df["Date"] = pd.to_datetime(df["Date"])

    # Required for correct shift/rolling computations
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    # --------------------------------------------------------------
    # 1) Base returns and target construction
    # --------------------------------------------------------------
    if "daily_return" not in df.columns:
        df["daily_return"] = df.groupby("ticker")["price"].pct_change()

    if "market_return" not in df.columns:
        # Cross-sectional mean return as a simple market proxy
        df["market_return"] = df.groupby("Date")["daily_return"].transform("mean")

    # Next-day returns per ticker
    if "daily_return_tomorrow" not in df.columns:
        df["daily_return_tomorrow"] = df.groupby("ticker")["daily_return"].shift(-1)

    # Next-day market return aligned by date (robust to missing ticker days)
    if "market_return_tomorrow" not in df.columns:
        market_by_date = (
            df.drop_duplicates("Date")[["Date", "market_return"]]
            .sort_values("Date")
        )
        market_by_date["market_return_tomorrow"] = market_by_date["market_return"].shift(-1)
        df = df.merge(
            market_by_date[["Date", "market_return_tomorrow"]],
            on="Date",
            how="left",
        )

    # Binary target: outperform the market on the next trading day
    if "outperform_tomorrow" not in df.columns:
        df["outperform_tomorrow"] = (
            df["daily_return_tomorrow"] > df["market_return_tomorrow"]
        ).astype(int)

    # Alias used across the project
    df["outperform"] = df["outperform_tomorrow"]

    # Excess returns (useful for descriptive statistics)
    df["excess_return"] = df["daily_return"] - df["market_return"]
    df["excess_return_tomorrow"] = df["daily_return_tomorrow"] - df["market_return_tomorrow"]

    # --------------------------------------------------------------
    # 2) Momentum features
    # --------------------------------------------------------------
    for window in [5, 10, 20]:
        df[f"momentum_{window}d"] = df.groupby("ticker")["price"].pct_change(periods=window)

    # --------------------------------------------------------------
    # 3) Volatility and moving averages
    # --------------------------------------------------------------
    for window in [10, 20]:
        df[f"volatility_{window}d"] = (
            df.groupby("ticker")["daily_return"]
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
        )

    for window in [20, 50]:
        df[f"ma_{window}d"] = (
            df.groupby("ticker")["price"]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    df["price_over_ma20"] = df["price"] / df["ma_20d"]

    # --------------------------------------------------------------
    # 4) Standard calendar effects
    # --------------------------------------------------------------
    df["day_of_week"] = df["Date"].dt.weekday  # 0 = Monday
    df["month"] = df["Date"].dt.month

    # Explicit January indicator (used in models.py)
    df["is_january"] = (df["month"] == 1).astype(int)

    # --------------------------------------------------------------
    # Turn-of-the-month effect (finance convention):
    # - first 3 trading days of the month
    # - last 3 trading days of the month
    # Computed per ticker to handle holidays / missing days.
    # --------------------------------------------------------------
    df["turn_of_month"] = 0

    g = df.groupby(["ticker", df["Date"].dt.to_period("M")])

    # Trading-day rank within month: 1, 2, 3, ...
    df["dom_rank"] = g.cumcount() + 1

    # Rank from month-end: -1 = last trading day, -2 = second last, etc.
    month_size = g["Date"].transform("size")
    df["eom_rank"] = df["dom_rank"] - month_size - 1

    df.loc[(df["dom_rank"] <= 3) | (df["eom_rank"] >= -3), "turn_of_month"] = 1

    # Drop temporary columns
    df = df.drop(columns=["dom_rank", "eom_rank"])

    # Sell-in-May indicator: May–October = 1, otherwise 0
    df["sell_in_may"] = df["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)

    # --------------------------------------------------------------
    # 5) Extended calendar effects (simple definitions)
    # --------------------------------------------------------------
    df["is_christmas"] = (
        (df["Date"].dt.month == 12) & (df["Date"].dt.day == 25)
    ).astype(int)

    df["is_new_year"] = (
        (df["Date"].dt.month == 1) & (df["Date"].dt.day == 1)
    ).astype(int)

    # Thanksgiving: 4th Thursday of November (standard approximation)
    is_november = df["Date"].dt.month == 11
    is_thursday = df["Date"].dt.weekday == 3
    is_fourth_week = df["Date"].dt.day.between(22, 28)
    df["is_thanksgiving"] = (is_november & is_thursday & is_fourth_week).astype(int)

    # First day of quarter: Jan/Apr/Jul/Oct 1st
    df["is_first_day_quarter"] = (
        (df["Date"].dt.day == 1) & (df["Date"].dt.month.isin([1, 4, 7, 10]))
    ).astype(int)

    # --------------------------------------------------------------
    # 6) Pre-holiday effect (US holidays) — inferred from dates present in df
    # --------------------------------------------------------------
    years = df["Date"].dt.year.unique().tolist()
    us_holidays = holidays_lib.US(years=years)

    holiday_dates = pd.to_datetime(list(us_holidays.keys()))

    df["pre_holiday"] = df["Date"].isin(holiday_dates - pd.Timedelta(days=1)).astype(int)

    return df
