import pandas as pd

# Paths
excel_path = "data/raw/market_anomalie.xlsx"
output_path = "data/raw/sector_mapping.csv"

# Read only the header row from the DAILY sheet
df = pd.read_excel(excel_path, sheet_name="DAILY", nrows=1)

# Extract tickers (all columns except Date)
tickers = [col for col in df.columns if col != "Date"]

# Initialize an empty sector mapping table
sector_mapping = pd.DataFrame({
    "company": tickers,   # placeholder (set equal to ticker initially)
    "ticker": tickers,
    "sector": [""] * len(tickers),
})

# Export to CSV
sector_mapping.to_csv(output_path, index=False)

print(f"sector_mapping.csv generated with {len(tickers)} tickers:")
print(output_path)
