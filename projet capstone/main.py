import sys
import os

# ajouter src/ au PYTHONPATH
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


def main():
    print("🚀 Starting full pipeline...")

    # 1. Charger Excel
    DATA_PATH = "data/raw/market_anomalie.xlsx"
    excel_data = load_excel_data(DATA_PATH)


    # 1b. Charger le mapping secteurs
    sector_mapping = load_sector_mapping()

    # 2. Télécharger indices
    market_indices = download_market_indices()

    # 3. Sélection des tickers
    selected_tickers = select_top_stocks(excel_data["sp500_daily"])

    # 4. Structuration
    df = clean_and_structure_data(
        excel_data["sp500_daily"],
        selected_tickers,
        market_indices["sp500_index"],
        sector_mapping,
    )

    # 5. Features
    df = create_features(df)

    # 6. Anomalies descriptives
    analyze_weekday_effect(df)
    analyze_january_effect(df)

    # 6b. Analyse des rendements par anomalie
    run_anomaly_return_analysis(df)

    # 6c. 🔎 Analyse sectorielle
    analyze_sectors(df)

    # 7. Machine Learning
    run_logistic_regression(df)
    run_random_forest(df)
    run_gradient_boosting(df)
    run_neural_network(df)

    print("\n🎉 Pipeline COMPLET terminé !")


if __name__ == "__main__":
    main()

