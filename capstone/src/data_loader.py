# src/data_loader.py

import pandas as pd


def load_and_prepare_data(excel_path: str, min_obs: int = 1500) -> dict:
    """
    Load a wide-format Excel price file and convert it to a clean long-format
    daily panel:

    - One row per (Date, ticker, price)
    - Filter out tickers with too few observations
    - Remove missing prices
    - Apply light winsorization to reduce the impact of extreme values

    Returns:
        {"daily_panel": panel}
    """

    raw = pd.read_excel(excel_path)

    # Identify the date column (supports both "Date" and "date")
    if "Date" in raw.columns:
        date_col = "Date"
    elif "date" in raw.columns:
        date_col = "date"
    else:
        raise ValueError(
            f"Unable to find a date column named 'Date' or 'date' in {excel_path}. "
            f"Columns found: {list(raw.columns)}"
        )

    # All non-date columns are interpreted as price series
    price_cols = [c for c in raw.columns if c != date_col]
    if len(price_cols) == 0:
        raise ValueError(
            f"No price columns detected (excluding '{date_col}') in {excel_path}."
        )

    # Convert wide format to long format: one observation per (Date, ticker)
    panel = raw.melt(
        id_vars=[date_col],
        value_vars=price_cols,
        var_name="ticker",
        value_name="price",
    )

    # Basic cleaning: ensure datetime and drop missing prices
    panel[date_col] = pd.to_datetime(panel[date_col])
    panel = panel.dropna(subset=["price"])

    # Keep only tickers with sufficient history
    counts = panel.groupby("ticker")["price"].transform("count")
    panel = panel[counts >= min_obs].copy()

    # Light winsorization by ticker to mitigate outliers / data errors
    def _winsorize(s: pd.Series) -> pd.Series:
        lower = s.quantile(0.001)
        upper = s.quantile(0.999)
        return s.clip(lower=lower, upper=upper)

    panel["price"] = panel.groupby("ticker")["price"].transform(_winsorize)

    # Standardize the date column name across the project
    panel = panel.rename(columns={date_col: "Date"})
    panel = panel.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return {"daily_panel": panel}


def load_sector_mapping(csv_path: str) -> pd.DataFrame:
    """
    Load and clean the sector mapping file.

    Expected minimum columns:
        - 'ticker'
        - 'sector'

    Cleaning steps:
    - Remove empty / missing sector labels
    - Deduplicate tickers to avoid many-to-one merges (row inflation)
    - Normalize sector naming to avoid inconsistent labels
    """
    df = pd.read_csv(csv_path)

    required_cols = {"ticker", "sector"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Sector mapping must contain columns {required_cols}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df.copy()

    # Standardize formatting
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()

    # Normalize sector names to prevent duplicates due to naming variants
    df["sector"] = df["sector"].replace({"Healthcare": "Health Care"})

    # Drop missing/empty sector labels
    df = df[df["sector"].notna() & (df["sector"] != "")].copy()

    # Ensure a one-to-one mapping ticker -> sector
    df = df.drop_duplicates(subset=["ticker"], keep="first")

    return df
