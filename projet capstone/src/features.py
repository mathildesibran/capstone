import pandas as pd

# ============================================================================
# PARTIE 5 : CALCUL DES FEATURES
# ============================================================================

def create_features(df):
    """
    Ajoute les colonnes nécessaires pour le machine learning :
    - daily_return        : rendement journalier de l'action
    - market_return       : rendement journalier du marché
    - volatility_20d      : volatilité roulante sur 20 jours
    - ma20                : moyenne mobile sur 20 jours
    - day_of_week         : 0 = lundi
    - month               : 1 = janvier
    - outperform (target) : 1 si l’action fait mieux que le marché
    """

    print("\n🛠️ Création des features...")

    # Rendement journalier de l'action
    df["daily_return"] = df.groupby("ticker")["close_price"].pct_change()

    # Rendement journalier du marché (S&P 500)
    df["market_return"] = df["SP500_Close"].pct_change()

    # Volatilité roulante 20 jours
    df["volatility_20d"] = df.groupby("ticker")["daily_return"].transform(
        lambda x: x.rolling(window=20).std()
    )

    # Moyenne mobile 20 jours
    df["ma20"] = df.groupby("ticker")["close_price"].transform(
        lambda x: x.rolling(window=20).mean()
    )

    # Variables calendrier
    df["day_of_week"] = df["Date"].dt.weekday   # 0 = lundi
    df["month"] = df["Date"].dt.month          # 1 = janvier

    # Target : 1 si l’action fait mieux que le marché
    df["outperform"] = (df["daily_return"] > df["market_return"]).astype(int)

    print("✅ Features créées :")
    print(df.head())

    return df
