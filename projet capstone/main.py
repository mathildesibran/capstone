import sys
import os

# ajouter src/ au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_excel_data, download_market_indices, select_top_stocks, clean_and_structure_data
from features import create_features
from anomalies import analyze_weekday_effect, analyze_january_effect
from models import run_logistic_regression, run_random_forest, run_gradient_boosting


def main():
    print("🚀 Starting full pipeline...")

    # 1. Charger Excel
    excel_data = load_excel_data("data/raw/market_anomalie.xlsx")

    # 2. Télécharger indices
    market_indices = download_market_indices()

    # 3. Sélection des tickers
    selected_tickers = select_top_stocks(excel_data["sp500_daily"])

    # 4. Structuration
    df = clean_and_structure_data(
        excel_data["sp500_daily"],
        selected_tickers,
        market_indices["sp500_index"]
    )

    # 5. Features
    df = create_features(df)

    # 6. Anomalies
    analyze_weekday_effect(df)
    analyze_january_effect(df)

    # 7. Machine Learning
    run_logistic_regression(df)
    run_random_forest(df)
    run_gradient_boosting(df)

    print("\n🎉 Pipeline COMPLET terminé !")


if __name__ == "__main__":
    main()
