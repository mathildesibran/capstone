import os
import sys

# Allow imports from the src/ directory
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_and_prepare_data, load_sector_mapping
from features import create_features

from anomalies import (
    analyze_weekday_effect,       # A1
    analyze_january_effect,       # A2
    analyze_turn_of_month_effect, # A3
    analyze_sell_in_may_effect,   # A4
    analyze_pre_holiday_effect,   # A5
)

from models import run_all_models

from visualization import (
    plot_day_of_week_global,
    plot_monday_effect_by_sector,
    plot_january_by_sector,
    plot_turn_of_month_by_sector,
    plot_sell_in_may_by_sector,
    plot_pre_holiday_by_sector,
    plot_model_performance,
)


# ===========================================
# BASELINE — Majority class classifier
# ===========================================
def compute_baseline(df):
    """
    Baseline classifier that always predicts the majority class.
    Used as a reference for machine learning performance.
    """
    proportion_of_ones = df["outperform"].mean()
    baseline_acc = max(proportion_of_ones, 1 - proportion_of_ones)

    print("\n=== BASELINE (majority class) ===")
    print(f"Share of positive outcomes (outperform = 1): {proportion_of_ones:.4f}")
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    return baseline_acc


# ===========================================
# MAIN PIPELINE
# ===========================================
def main():

    print("=== STEP 1: Build the clean daily panel from Excel ===")

    # 1) Load Excel file (already reduced to ~40 tickers)
    data = load_and_prepare_data("data/raw/market_anomalie_40.xlsx")
    df = data["daily_panel"]
    print(f"Raw daily panel: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2) Load sector mapping
    sector_map = load_sector_mapping("data/raw/sector_mapping.csv")

    # 3) Keep only tickers present in the sector mapping
    mapped_tickers = sector_map["ticker"].unique()
    df = df[df["ticker"].isin(mapped_tickers)].copy()
    print(f"After ticker filtering: {df.shape[0]} rows")

    # 4) Merge sector information
    df = df.merge(
        sector_map[["ticker", "sector"]],
        on="ticker",
        how="left",
    )

    print("\nSector distribution:")
    print(df["sector"].value_counts())

    # 5) Create all features (calendar + technical indicators)
    df = create_features(df)

    # 6) Save the structured dataset
    os.makedirs("results", exist_ok=True)
    out_path = "results/structured_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nStructured dataset saved to {out_path}")

    # ==========================================================
    # STEP 2 — CALENDAR ANOMALY ANALYSIS
    # ==========================================================
    print("\n=== STEP 2: Calendar anomaly analysis ===")

    analyze_weekday_effect(df)            # A1
    analyze_january_effect(df)            # A2 (includes by_sector sheet)
    analyze_turn_of_month_effect(df)      # A3
    analyze_sell_in_may_effect(df)        # A4
    analyze_pre_holiday_effect(df)        # A5

    print("\nCalendar anomaly analysis completed.")

    # ==========================================================
    # STEP 3 — MACHINE LEARNING BASELINE
    # ==========================================================
    print("\n=== STEP 3: Machine learning baseline ===")
    compute_baseline(df)

    # ==========================================================
    # STEP 4 — GLOBAL MACHINE LEARNING MODELS
    # ==========================================================
    print("\n=== STEP 4: Global machine learning models ===")
    run_all_models(df, suffix="")
    print("\nGlobal models completed.")

    # ==========================================================
    # STEP 5 — SECTOR-LEVEL MACHINE LEARNING MODELS
    # ==========================================================
    print("\n=== STEP 5: Sector-level machine learning models ===")

    for sector in df["sector"].unique():
        print(f"\n--- Sector: {sector} ---")
        df_sector = df[df["sector"] == sector].copy()

        # Skip sectors with insufficient observations
        if df_sector.shape[0] < 1000:
            print("Not enough observations. Sector skipped.")
            continue

        suffix = "_" + str(sector).replace(" ", "_")
        run_all_models(df_sector, suffix=suffix)

    print("\nSector-level models completed.")

    # ==========================================================
    # STEP 6 — VISUALIZATIONS
    # ==========================================================
    print("\n=== STEP 6: Visualizations ===")

    plot_day_of_week_global()
    plot_monday_effect_by_sector()
    plot_january_by_sector()
    plot_turn_of_month_by_sector()
    plot_sell_in_may_by_sector()
    plot_pre_holiday_by_sector()
    plot_model_performance()

    print("\nFull pipeline completed.")


if __name__ == "__main__":
    main()
