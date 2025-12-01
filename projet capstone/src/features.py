import pandas as pd
import numpy as np

def create_features(df):
    """
    Création de toutes les features nécessaires pour le pipeline :
    - Rendements
    - Volatilité
    - Mouvements du marché
    - Effets calendaires (Day-of-week, Month, Sell-in-May, Pre-holiday)
    - Nouveaux effets : Christmas, Thanksgiving, New Year, First-day-of-quarter
    """

    df = df.copy()

    # ---------------------------
    # 1. Rendements basiques
    # ---------------------------
    df["daily_return"] = df.groupby("ticker")["close_price"].pct_change()
    df["market_return"] = df["SP500_Close"].pct_change()

    # ---------------------------
    # 2. Rendements du lendemain
    # ---------------------------
    df["daily_return_tomorrow"] = df.groupby("ticker")["daily_return"].shift(-1)
    df["market_return_tomorrow"] = df["market_return"].shift(-1)

    # ---------------------------
    # 3. Indicateur outperform (dépendant du lendemain)
    # ---------------------------
    df["outperform_tomorrow"] = (df["daily_return_tomorrow"] > df["market_return_tomorrow"]).astype(int)
    df["outperform"] = df["outperform_tomorrow"]

    # ---------------------------
    # 4. Volatilité rolling
    # ---------------------------
    df["volatility_20d"] = df.groupby("ticker")["daily_return"].rolling(20).std().reset_index(0, drop=True)

    # ---------------------------
    # 5. Moyenne mobile
    # ---------------------------
    df["ma20"] = df.groupby("ticker")["close_price"].rolling(20).mean().reset_index(0, drop=True)

    # ---------------------------
    # 6. Effets calendaires classiques
    # ---------------------------
    df["day_of_week"] = df["Date"].dt.weekday
    df["month"] = df["Date"].dt.month
    df["turn_of_month"] = (df["Date"].dt.day <= 3).astype(int)
    df["sell_in_may"] = df["Date"].dt.month.isin([5, 6, 7, 8, 9, 10]).astype(int)

    # Pre-holiday : jour précédent un jour férié US
    holidays = [
        "2020-12-24", "2021-12-24", "2022-12-23", "2023-12-25",
        "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-02"
    ]
    holidays = pd.to_datetime(holidays)
    df["pre_holiday"] = df["Date"].isin(holidays - pd.Timedelta(days=1)).astype(int)

    # ---------------------------
    # 7. NOUVELLES ANOMALIES
    # ---------------------------

    # 🎄 Christmas effect (25 décembre)
    df["is_christmas"] = (
        (df["Date"].dt.month == 12) &
        (df["Date"].dt.day == 25)
    ).astype(int)

    # 🎆 New year effect (1 janvier)
    df["is_new_year"] = (
        (df["Date"].dt.month == 1) &
        (df["Date"].dt.day == 1)
    ).astype(int)

    # 🦃 Thanksgiving effect (4e jeudi de novembre)
    is_nov = df["Date"].dt.month == 11
    is_thu = df["Date"].dt.weekday == 3
    is_4th = df["Date"].dt.day.between(22, 28)
    df["is_thanksgiving"] = (is_nov & is_thu & is_4th).astype(int)

    # 📅 First day of quarter (1 janvier, 1 avril, 1 juillet, 1 octobre)
    df["is_first_day_quarter"] = (
        (df["Date"].dt.day == 1) &
        (df["Date"].dt.month.isin([1, 4, 7, 10]))
    ).astype(int)

    return df
