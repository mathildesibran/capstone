import os
import pandas as pd
import yfinance as yf
import numpy as np

# ============================================================================
# PARTIE 1 : CHARGEMENT DES DONNÉES EXCEL
# ============================================================================

def load_excel_data(filepath="data/raw/market_anomalie.xlsx"):
    """
    Charge les 2 onglets du fichier Excel :
    - DAILY
    - MONTHLY
    """
    print("📊 Chargement du fichier Excel...")

    sp500_daily = pd.read_excel(
        filepath,
        sheet_name="DAILY",
        index_col=0,      # Date en index
        parse_dates=True
    )
    sp500_monthly = pd.read_excel(
        filepath,
        sheet_name="MONTHLY",
        index_col=0,      # Date en index
        parse_dates=True
    )

    print("✅ Données chargées :")
    print(f"   - S&P 500 DAILY:   {sp500_daily.shape}")
    print(f"   - S&P 500 MONTHLY: {sp500_monthly.shape}")

    return {
        "sp500_daily": sp500_daily,
        "sp500_monthly": sp500_monthly
    }


# ============================================================================
# PARTIE 2 : TÉLÉCHARGEMENT DES INDICES
# ============================================================================

def download_market_indices(start_date="2010-01-01", end_date="2025-11-01"):
    """
    Télécharge les indices S&P500 (^GSPC) et STOXX 600 (^STOXX)
    et renvoie les DataFrames bruts de yfinance.
    """
    print("\n📈 Téléchargement des indices de marché...")

    sp500_index = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    stoxx_index = yf.download("^STOXX", start=start_date, end=end_date, progress=False)

    print(f"   - S&P 500 Index téléchargé :  {sp500_index.shape[0]} jours")
    print(f"   - STOXX 600 Index téléchargé : {stoxx_index.shape[0]} jours")

    return {
        "sp500_index": sp500_index,   # on garde TOUTES les colonnes
        "stoxx_index": stoxx_index,
    }


# ============================================================================
# PARTIE 3 : SÉLECTION DES 40 MEILLEURES ACTIONS
# ============================================================================

def select_top_stocks(df, n_stocks=40):
    """
    Sélectionne les 40 meilleures actions à partir du dataframe DAILY.
    Critère : moins de 5% de données manquantes puis plus forte volatilité.
    """
    print("\n🎯 Sélection des meilleures actions...")

    # % de valeurs manquantes par colonne (par ticker)
    missing_pct = df.isnull().sum() / len(df) * 100
    valid_stocks = missing_pct[missing_pct < 5].index.tolist()

    print(f"   - {len(valid_stocks)} actions avec <5% de données manquantes")

    # Volatilité des rendements
    returns = df[valid_stocks].pct_change()
    volatility = returns.std().sort_values(ascending=False)

    selected_stocks = volatility.head(n_stocks).index.tolist()

    print(f"✅ {len(selected_stocks)} actions sélectionnées")

    print("\n📌 TICKERS UTILISÉS PAR LE PROJET :")
    for t in selected_stocks:
        print("   •", t)

    return selected_stocks


# ============================================================================
# PARTIE 4 : NETTOYAGE + STRUCTURATION + SECTEURS
# ============================================================================

def clean_and_structure_data(excel_daily, selected_tickers, sp500_index, sector_mapping):
    """
    - Garde uniquement les tickers sélectionnés
    - Met les données au format long (Date, ticker, close_price)
    - Ajoute l'index S&P 500 (SP500_Close)
    - Ajoute le secteur de chaque ticker
    """
    print("\n🧹 Nettoyage et structuration des données...")

    # 1) On garde seulement les colonnes des tickers sélectionnés
    df = excel_daily[selected_tickers].copy()

    # L’index est la date → on le remet comme vraie colonne "Date"
    df = df.reset_index()
    if "Date" not in df.columns:  # au cas où la colonne s'appelle "index"
        df = df.rename(columns={df.columns[0]: "Date"})

    # Passage en format long
    df = df.melt(
        id_vars="Date",
        var_name="ticker",
        value_name="close_price"
    )

    # On enlève les lignes sans prix
    df = df.dropna(subset=["close_price"])

    # 2) Préparer le S&P 500
    sp500 = sp500_index.copy()

    # Si colonnes MultiIndex (('Close','^GSPC'), ...) → on garde le 1er niveau
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    print("   Colonnes SP500 :", sp500.columns.tolist())

    possible_cols = ["Close", "close", "Adj Close", "adjclose"]
    price_col = None
    for c in possible_cols:
        if c in sp500.columns:
            price_col = c
            break

    if price_col is None:
        raise ValueError(
            "Impossible de trouver une colonne de prix dans sp500_index "
            "(cherché : 'Close', 'close', 'Adj Close', 'adjclose')."
        )

    sp500_df = (
        sp500[[price_col]]
        .rename(columns={price_col: "SP500_Close"})
        .reset_index()              # index → colonne Date
        .rename(columns={sp500.index.name or "Date": "Date"})
    )

    # 3) Fusion actions + index
    df = df.merge(sp500_df, on="Date", how="left")

    # 4) Ajouter les secteurs
    df = df.merge(sector_mapping, on="ticker", how="left")

    print("   Ajout des secteurs effectué.")
    print("   Lignes sans secteur :", df["sector"].isna().sum())

    print("   Données structurées :")
    print(df.head())
    print(f"\n✅ Données structurées prêtes ({len(df)} lignes)")

    return df


def load_sector_mapping(filepath="data/raw/sector_mapping.csv"):
    """
    Charge le mapping des secteurs pour chaque ticker.
    CSV attendu avec au moins les colonnes : 'ticker', 'sector'
    (ou 'Ticker', 'Sector' qu'on renomme).
    """
    print("\n📂 Chargement des secteurs...")
    sector_df = pd.read_csv(filepath)

    # Harmonisation des noms de colonnes possibles
    rename_cols = {}
    if "Ticker" in sector_df.columns:
        rename_cols["Ticker"] = "ticker"
    if "Sector" in sector_df.columns:
        rename_cols["Sector"] = "sector"
    if rename_cols:
        sector_df = sector_df.rename(columns=rename_cols)

    print(f"   - {sector_df.shape[0]} mappings trouvés")
    return sector_df
