import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import calendar

def auto_width_excel(path):
    """Ajuste automatiquement la largeur des colonnes dans un Excel."""
    wb = load_workbook(path)
    ws = wb.active

    for col in ws.columns:
        max_length = 0
        column = col[0].column

        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except Exception:
                pass

        adjusted_width = max_length + 2
        ws.column_dimensions[get_column_letter(column)].width = adjusted_width

    wb.save(path)


# ============================================================================
# A1 — Effet Day-of-Week
# ============================================================================

def analyze_weekday_effect(df):
    print("\n📊 Analyse A1 : Day-of-Week effect")

    grouped = df.groupby(["ticker", "day_of_week"]).agg(
        mean_return=("daily_return", "mean"),
        std_return=("daily_return", "std"),
        prob_outperform=("outperform", "mean"),
        n_obs=("daily_return", "count")
    ).reset_index()

    grouped["day_name"] = grouped["day_of_week"].map(lambda x: calendar.day_name[x])

    global_stats = grouped.groupby(["day_of_week", "day_name"]).agg(
        mean_return=("mean_return", "mean"),
        std_return=("std_return", "mean"),
        prob_outperform=("prob_outperform", "mean"),
        n_obs=("n_obs", "sum")
    ).reset_index()

    os.makedirs("results/tables", exist_ok=True)
    detail_path = "results/tables/anomaly_A1_by_ticker.xlsx"
    global_path = "results/tables/anomaly_A1_global.xlsx"

    grouped.to_excel(detail_path, index=False)
    global_stats.to_excel(global_path, index=False)
    auto_width_excel(detail_path)
    auto_width_excel(global_path)

    print(f"📁 A1 sauvegardé : {detail_path}")
    print(f"📁 A1 global      : {global_path}")

    return grouped, global_stats



# ============================================================================
# A2 — Effet January
# ============================================================================

def analyze_january_effect(df):
    print("\n📊 Analyse A2 : January effect")

    grouped = df.groupby(["ticker", "month"]).agg(
        mean_return=("daily_return", "mean"),
        std_return=("daily_return", "std"),
        prob_outperform=("outperform", "mean"),
        n_obs=("daily_return", "count")
    ).reset_index()

    global_stats = grouped.copy()
    global_stats["is_january"] = (global_stats["month"] == 1).astype(int)

    global_stats = global_stats.groupby("is_january").agg(
        mean_return=("mean_return", "mean"),
        std_return=("std_return", "mean"),
        prob_outperform=("prob_outperform", "mean"),
        n_obs=("n_obs", "sum")
    ).reset_index()

    global_stats["label"] = global_stats["is_january"].map(
        {1: "January", 0: "Other months"}
    )

    os.makedirs("results/tables", exist_ok=True)
    detail_path = "results/tables/anomaly_A2_by_ticker.xlsx"
    global_path = "results/tables/anomaly_A2_global.xlsx"

    grouped.to_excel(detail_path, index=False)
    global_stats.to_excel(global_path, index=False)
    auto_width_excel(detail_path)
    auto_width_excel(global_path)

    print(f"📁 A2 sauvegardé : {detail_path}")
    print(f"📁 A2 global      : {global_path}")

    return grouped, global_stats
