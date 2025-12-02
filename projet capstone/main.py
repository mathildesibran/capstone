iimport sys
import os

# Add the src/ directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import (
    load_excel_data,
    download_market_indices,
    select_top_stocks,
    clean_and_structure_data,
    load_sector_mapping,
)
from features import create_features
from anomalies import analyze_weekday_effect, analyze_january_effect
from models import (
    run_logistic_regression,
    run_random_forest,
    run_gradient_boosting,
    run_neural_network,
)
from anomaly_analysis import run_anomaly_return_analysis, analyze_sectors


# If set to True, machine learning models are skipped to speed up execution
FAST_MODE = False


def run_all_models(df):
    """
    Run all machine learning models sequentially.
    """
    run_logistic_regression(df)
    run_random_forest(df)
    run_gradient_boosting(df)
    run_neural_network(df)


def main():
    """
    Main execution pipeline for the market anomaly analysis project.
    """
    print("Starting full pipeline...")

    # 1. Load Excel input data
    DATA_PATH = "data/raw/market_anomalie.xlsx"
    excel_data = load_excel_data(DATA_PATH)

    # 1b. Load sector mapping
    print("Loading sector mapping...")
    sector_mapping = load_sector_mapping()

    # 2. Download market indices
    print("Downloading market indices...")
    market_indices = download_market_indices()

    # 3. Select top stocks from the S&P 500 universe
    print("Selecting top stocks...")
    selected_tickers = select_top_stocks(excel_data["sp500_daily"])

    # 4. Clean and structure the dataset
    print("Cleaning and structuring data...")
    df = clean_and_structure_data(
        excel_data["sp500_daily"],
        selected_tickers,
        market_indices["sp500_index"],
        sector_mapping,
    )

    # 5. Feature engineering
    print("Creating features...")
    df = create_features(df)

    # Safety check: stop if dataset is empty
    if df.empty:
        print("Dataset is empty after feature engineering. Pipeline stopped.")
        return

    # 6. Descriptive anomaly analysis
    print("Running descriptive anomaly analysis...")
    analyze_weekday_effect(df)
    analyze_january_effect(df)

    # 6b. Anomaly-based return analysis
    run_anomaly_return_analysis(df)

    # 6c. Sector-level analysis
    print("Running sector analysis...")
    analyze_sectors(df)

    # 7. Machine learning models
    if not FAST_MODE:
        print("Running machine learning models...")
        run_all_models(df)
    else:
        print("FAST MODE enabled: machine learning models skipped.")

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
