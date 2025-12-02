# src/data_loader.py

import os
from typing import Dict, List

import pandas as pd
import yfinance as yf


# ============================================================================
# PART 1: EXCEL DATA LOADING
# ============================================================================


def load_excel_data(filepath: str = "data/raw/market_anomalie.xlsx") -> Dict[str, pd.DataFrame]:
    """
    Load the two sheets from the Excel file:

    - DAILY   : daily prices for S&P 500 constituents
    - MONTHLY : monthly prices for S&P 500 constituents
    """
    print("Loading Excel file...")

    sp500_daily = pd.read_excel(
        filepath,
        sheet_name="DAILY",
        index_col=0,      # Date as index
        parse_dates=True,
    )
    sp500_monthly = pd.read_excel(
        filepath,
        sheet_name="MONTHLY",
        index_col=0,      # Date as index
        parse_dates=True,
    )

    print("Excel data loaded:")
    print(f"   - S&P 500 DAILY:   {sp500_daily.shape}")
    print(f"   - S&P 500 MONTHLY: {sp500_monthly.shape}")

    return {
        "sp500_daily": sp500_daily,
        "sp500_monthly": sp500_monthly,
    }


# ============================================================================
# PART 2: MARKET INDEX DOWNLOAD
# ============================================================================


def download_market_indices(
    start_date: str = "2010-01-01",
    end_date: str = "2025-11-01",
) -> Dict[str, pd.DataFrame]:
    """
    Download S&P 500 (^GSPC) and STOXX 600 (^STOXX) indices from Yahoo Finance
    and return the raw DataFrames.
    """
    print("\nDownloading market indices...")

    sp500_index = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    stoxx_index = yf.download("^STOXX", start=start_date, end=end_date, progress=False)

    print(f"   - S&P 500 index downloaded:  {sp500_index.shape[0]} trading days")
    print(f"   - STOXX 600 index downloaded: {stoxx_index.shape[0]} trading days")

    return {
        "sp500_index": sp500_index,  # keep all columns for flexibility
        "stoxx_index": stoxx_index,
    }


# ============================================================================
# PART 3: SELECTION OF TOP 40 STOCKS
# ============================================================================


def select_top_stocks(df: pd.DataFrame, n_stocks: int = 40) -> List[str]:
    """
    Select the top N stocks from the DAILY price DataFrame.

    Selection criteria:
    1. Less than 5% missing observations.
    2. Among those, highest return volatility.
    """
    print("\nSelecting top stocks based on data availability and volatility...")

    # Percentage of missing values per column (per ticker)
    missing_pct = df.isnull().sum() / len(df) * 100
    valid_stocks = missing_pct[missing_pct < 5].index.tolist()

    print(f"   - {len(valid_stocks)} stocks with less than 5% missing data")

    # Volatility of daily returns
    returns = df[valid_stocks].pct_change()
    volatility = returns.std().sort_values(ascending=False)

    selected_stocks = volatility.head(n_stocks).index.tolist()

    print(f"   - {len(selected_stocks)} stocks selected for the analysis")
    print("\nTickers used in the project:")
    for t in selected_stocks:
        print(f"   • {t}")

    return selected_stocks


# ============================================================================
# PART 4: DATA CLEANING, STRUCTURING AND SECTOR MERGE
# ============================================================================


def clean_and_structure_data(
    excel_daily: pd.DataFrame,
    selected_tickers: List[str],
    sp500_index: pd.DataFrame,
    sector_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the main analysis DataFrame:

    - Keep only the selected tickers from the daily price panel.
    - Reshape the data to long format: (Date, ticker, close_price).
    - Merge S&P 500 index level as SP500_Close.
    - Merge sector information for each ticker.
    """
    print("\nCleaning and structuring data...")

    # 1) Keep only selected tickers
    df = excel_daily[selected_tickers].copy()

    # Reset index so that the date becomes an explicit column
    df = df.reset_index()
    if "Date" not in df.columns:  # fallback if the first column is unnamed
        df = df.rename(columns={df.columns[0]: "Date"})

    # Long format: one row per (Date, ticker)
    df = df.melt(
        id_vars="Date",
        var_name="ticker",
        value_name="close_price",
    )

    # Remove rows without a valid price
    df = df.dropna(subset=["close_price"])

    # 2) Prepare the S&P 500 index series
    sp500 = sp500_index.copy()

    # If columns are a MultiIndex ((Close, ^GSPC), ...), keep only the first level
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    possible_cols = ["Close", "close", "Adj Close", "adjclose"]
    price_col = None
    for c in possible_cols:
        if c in sp500.columns:
            price_col = c
            break

    if price_col is None:
        raise ValueError(
            "Could not find a price column in sp500_index "
            "(searched for: 'Close', 'close', 'Adj Close', 'adjclose')."
        )

    sp500_df = (
        sp500[[price_col]]
        .rename(columns={price_col: "SP500_Close"})
        .reset_index()  # index → Date column
        .rename(columns={sp500.index.name or "Date": "Date"})
    )

    # 3) Merge stock prices with S&P 500 index level
    df = df.merge(sp500_df, on="Date", how="left")

    # 4) Merge sector information
    df = df.merge(sector_mapping, on="ticker", how="left")

    print("   Sector information merged.")
    print(f"   Rows without sector information: {df['sector'].isna().sum()}")
    print(f"   Structured dataset ready with {len(df)} rows.")

    return df


# ============================================================================
# PART 5: SECTOR MAPPING
# ============================================================================


def load_sector_mapping(filepath: str = "data/raw/sector_mapping.csv") -> pd.DataFrame:
    """
    Load the sector mapping for each ticker.

    Expected CSV structure:
    - Either columns ('ticker', 'sector')
    - Or ('Ticker', 'Sector'), which are renamed accordingly.
    """
    print("\nLoading sector mapping...")
    sector_df = pd.read_csv(filepath)

    # Harmonize potential column names
    rename_cols = {}
    if "Ticker" in sector_df.columns:
        rename_cols["Ticker"] = "ticker"
    if "Sector" in sector_df.columns:
        rename_cols["Sector"] = "sector"
    if rename_cols:
        sector_df = sector_df.rename(columns=rename_cols)

    print(f"   - {sector_df.shape[0]} ticker–sector mappings loaded")
    return sector_df
git add src/data_loader.py