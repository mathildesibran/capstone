import pandas as pd

# Paths
EXCEL_PATH = "data/raw/market_anomalie.xlsx"
SECTOR_PATH = "data/raw/sector_mapping.csv"
OUTPUT_EXCEL_PATH = "data/raw/market_anomalie_40.xlsx"

# Target selection rule: approximately 8 sectors × 5 tickers = 40 tickers
N_PER_SECTOR = 5


def main():
    # --- 1) Load the daily price sheet ---
    daily = pd.read_excel(EXCEL_PATH, sheet_name="DAILY")

    # Ensure the first column is named "Date"
    if "Date" not in daily.columns:
        daily = daily.rename(columns={daily.columns[0]: "Date"})

    price_cols = [c for c in daily.columns if c != "Date"]

    # --- 2) Load and standardize the sector mapping file ---
    sector_map = pd.read_csv(SECTOR_PATH)
    sector_map = sector_map.rename(
        columns={
            "Company": "company",
            "company": "company",
            "Ticker": "ticker",
            "ticker": "ticker",
            "Sector": "sector",
            "sector": "sector",
        }
    )

    # Keep only tickers that exist in the Excel file
    sector_map = sector_map[sector_map["ticker"].isin(price_cols)].copy()

    # --- 3) Compute per-ticker metrics: number of observations and volatility ---
    metrics = []
    for t in price_cols:
        s = daily[t]
        if s.notna().sum() == 0:
            continue

        returns = s.pct_change()
        metrics.append(
            {
                "ticker": t,
                "n_obs": int(s.notna().sum()),
                "vol": returns.std(),
            }
        )

    metrics_df = pd.DataFrame(metrics)

    # Merge with sector information
    metrics_df = metrics_df.merge(
        sector_map[["ticker", "sector"]],
        on="ticker",
        how="left",
    )

    # Drop tickers without sector labels
    metrics_df = metrics_df.dropna(subset=["sector"])

    # --- 4) Select N_PER_SECTOR tickers per sector ---
    selected_rows = []
    for sector, sub in metrics_df.groupby("sector"):
        # Prioritize longer histories, then lower volatility
        sub = sub.sort_values(["n_obs", "vol"], ascending=[False, True])
        selected_rows.append(sub.head(N_PER_SECTOR))

    selected = pd.concat(selected_rows, ignore_index=True)

    # If more than 40 are selected, keep the 40 with the most observations
    if selected.shape[0] > 40:
        selected = selected.sort_values("n_obs", ascending=False).head(40)

    print("Selected tickers (≈40):")
    print(selected[["ticker", "sector", "n_obs", "vol"]])

    tickers_40 = selected["ticker"].tolist()
    cols_keep = ["]()_
