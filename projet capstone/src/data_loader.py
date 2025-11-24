import pandas as pd
import yfinance as yf

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
        index_col=0,
        parse_dates=True
    )
    sp500_monthly = pd.read_excel(
        filepath,
        sheet_name="MONTHLY",
        index_col=0,
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
    """
    print("\n📈 Téléchargement des indices de marché...")

    sp500_index = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    stoxx_index = yf.download("^STOXX", start=start_date, end=end_date, progress=False)

    print(f"   - S&P 500 Index téléchargé :  {sp500_index.shape[0]} jours")
    print(f"   - STOXX 600 Index téléchargé : {stoxx_index.shape[0]} jours")

    return {
        "sp500_index": sp500_index[["Close"]].rename(columns={"Close": "SP500_Close"}),
        "stoxx_index": stoxx_index[["Close"]].rename(columns={"Close": "STOXX_Close"}),
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

    # % de valeurs manquantes
    missing_pct = df.isnull().sum() / len(df) * 100
    valid_stocks = missing_pct[missing_pct < 5].index.tolist()

    print(f"   - {len(valid_stocks)} actions avec <5% de données manquantes")

    # Volatilité des rendements
    returns = df[valid_stocks].pct_change()
    volatility = returns.std().sort_values(ascending=False)

    selected_stocks = volatility.head(n_stocks).index.tolist()

    print(f"✅ {len(selected_stocks)} actions sélectionnées")
    return selected_stocks


# ============================================================================
# PARTIE 4 : NETTOYAGE + STRUCTURATION
# ============================================================================

def clean_and_structure_data(prices_df, selected_tickers, index_df):
    """
    Nettoyage + passage en format long + merge avec l'indice de marché
    Sortie : DataFrame avec colonnes Date, ticker, close_price, SP500_Close
    """
    print("\n🧹 Nettoyage et structuration des données...")

    # 1) Nettoyer les prix
    prices_clean = prices_df[selected_tickers].copy()
    prices_clean = prices_clean.ffill().bfill()

    # 2) Format long : une ligne = (Date, ticker, close_price)
    prices_long = prices_clean.reset_index().melt(
        id_vars=prices_clean.index.name or "Date",
        var_name="ticker",
        value_name="close_price"
    )

    # Corriger le nom de la colonne Date si besoin
    if "index" in prices_long.columns:
        prices_long.rename(columns={"index": "Date"}, inplace=True)

    # 3) Préparer l’indice
    index_df_reset = index_df.reset_index()

    # Aplatir MultiIndex si présent
    if isinstance(index_df_reset.columns, pd.MultiIndex):
        index_df_reset.columns = [col[0] for col in index_df_reset.columns]

    # Renommer la colonne de date si nécessaire
    if "Date" not in index_df_reset.columns:
        date_col = index_df_reset.columns[0]
        index_df_reset.rename(columns={date_col: "Date"}, inplace=True)

    # 4) Fusion sur la Date
    final_df = prices_long.merge(index_df_reset, on="Date", how="left")

    # Nettoyage final
    if "SP500_Close" in final_df.columns:
        final_df["SP500_Close"] = final_df["SP500_Close"].ffill().bfill()

    # S'assurer que Date est bien de type datetime
    final_df["Date"] = pd.to_datetime(final_df["Date"])

    print("✅ Données structurées :")
    print(final_df.head())

    return final_df
