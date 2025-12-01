# src/anomaly_analysis.py

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Liste des colonnes d’anomalies qu’on analyse
ANOMALY_COLS = [
    "pre_holiday",
    "turn_of_month",
    "sell_in_may",
    "is_christmas",
    "is_thanksgiving",
    "is_new_year",
    "is_first_day_quarter",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_excess_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    S'assure qu'on a une colonne 'excess_return' = daily_return - market_return.
    """
    df = df.copy()
    if "excess_return" in df.columns:
        return df

    if {"daily_return", "market_return"}.issubset(df.columns):
        df["excess_return"] = df["daily_return"] - df["market_return"]
        return df

    raise ValueError(
        "Impossible de calculer 'excess_return' : "
        "colonnes 'daily_return' et 'market_return' manquantes."
    )


def t_test_difference(sample_anom: pd.Series, sample_non: pd.Series) -> float:
    """
    t-stat pour la différence de moyennes entre jours anomalie et jours normaux.
    Si on n'a pas assez d'observations, renvoie NaN.
    """
    sample_anom = sample_anom.dropna()
    sample_non = sample_non.dropna()

    if len(sample_anom) < 5 or len(sample_non) < 5:
        return np.nan

    t_stat, _ = stats.ttest_ind(sample_anom, sample_non, equal_var=False)
    return float(t_stat)


# ---------------------------------------------------------------------------
# 1. Statistiques globales par anomalie
# ---------------------------------------------------------------------------

def compute_global_anomaly_stats(df: pd.DataFrame,
                                 anomaly_cols=ANOMALY_COLS) -> pd.DataFrame:
    """
    Pour chaque anomalie :
      - nb d'observations en anomalie / hors anomalie
      - rendement moyen excédentaire (excess_return) les jours d'anomalie
      - idem jours normaux
      - différence des deux
      - t-stat (test de Student)
    """

    rows = []
    for col in anomaly_cols:
        mask = df[col] == 1

        excess_anom = df.loc[mask, "excess_return"]
        excess_non  = df.loc[~mask, "excess_return"]

        mean_anom = excess_anom.mean()
        mean_non  = excess_non.mean()
        diff      = mean_anom - mean_non
        t_stat    = t_test_difference(excess_anom, excess_non)

        rows.append({
            "anomaly": col,
            "n_anomaly": int(mask.sum()),
            "n_non_anomaly": int((~mask).sum()),
            "mean_excess_anomaly": mean_anom,
            "mean_excess_non_anomaly": mean_non,
            "diff": diff,
            "t_stat": t_stat,
        })

    global_df = pd.DataFrame(rows)
    return global_df.sort_values("anomaly").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Statistiques par ticker (quels titres profitent de chaque anomalie ?)
# ---------------------------------------------------------------------------

def compute_ticker_anomaly_stats(df: pd.DataFrame,
                                 anomaly_cols=ANOMALY_COLS) -> pd.DataFrame:
    """
    Pour chaque anomalie et pour chaque ticker:
      - rendement moyen excédentaire les jours d'anomalie
    """
    all_rows = []

    for col in anomaly_cols:
        mask = df[col] == 1
        sub = df.loc[mask]

        if sub.empty:
            continue

        tmp = (
            sub.groupby("ticker")["excess_return"]
               .mean()
               .reset_index()
               .rename(columns={"excess_return": "mean_excess_return_anomaly_days"})
        )
        tmp["anomaly"] = col
        all_rows.append(tmp)

    if not all_rows:
        return pd.DataFrame(columns=["ticker", "anomaly", "mean_excess_return_anomaly_days"])

    ticker_df = pd.concat(all_rows, ignore_index=True)
    # On ordonne pour que ce soit lisible dans Excel
    ticker_df = ticker_df[["ticker", "anomaly", "mean_excess_return_anomaly_days"]]
    return ticker_df.sort_values(["anomaly", "mean_excess_return_anomaly_days"],
                                 ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Statistiques par secteur (effet par secteur)
# ---------------------------------------------------------------------------

def compute_sector_anomaly_stats(df: pd.DataFrame,
                                 anomaly_cols=ANOMALY_COLS) -> pd.DataFrame:
    """
    Pour chaque anomalie et chaque secteur:
      - nb d'observations avec anomalie / hors anomalie
      - rendements moyens excédentaires
      - différence
      - t-stat
    Nécessite une colonne 'sector' dans df.
    """
    if "sector" not in df.columns:
        raise ValueError("La colonne 'sector' est manquante dans df.")

    rows = []

    for col in anomaly_cols:
        for sector, g in df.groupby("sector"):
            mask = g[col] == 1
            if mask.sum() < 5:
                # Pas assez de jours d'anomalie dans ce secteur -> on saute
                continue

            excess_anom = g.loc[mask, "excess_return"]
            excess_non  = g.loc[~mask, "excess_return"]

            mean_anom = excess_anom.mean()
            mean_non  = excess_non.mean()
            diff      = mean_anom - mean_non
            t_stat    = t_test_difference(excess_anom, excess_non)

            rows.append({
                "anomaly": col,
                "sector": sector,
                "n_anomaly": int(mask.sum()),
                "n_non_anomaly": int((~mask).sum()),
                "mean_excess_anomaly": mean_anom,
                "mean_excess_non_anomaly": mean_non,
                "diff": diff,
                "t_stat": t_stat,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "anomaly", "sector", "n_anomaly", "n_non_anomaly",
            "mean_excess_anomaly", "mean_excess_non_anomaly",
            "diff", "t_stat"
        ])

    sector_df = pd.DataFrame(rows)
    return sector_df.sort_values(["anomaly", "sector"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Graphiques
# ---------------------------------------------------------------------------

def plot_global_anomaly_stats(global_df: pd.DataFrame, out_dir: str):
    """
    Barplot de la différence de rendement (diff) par anomalie.
    """
    if global_df.empty:
        return

    plt.figure(figsize=(8, 4))
    plt.bar(global_df["anomaly"], global_df["diff"])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Différence moyenne\n(excess_return anomaly - normal)")
    plt.title("Impact moyen des anomalies sur le rendement excédentaire")
    plt.tight_layout()

    path = os.path.join(out_dir, "global_anomaly_diff.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Graphique global sauvegardé : {path}")


def plot_sector_heatmap(sector_df: pd.DataFrame, out_dir: str):
    """
    Heatmap des diff par secteur × anomalie.
    """
    if sector_df.empty:
        return

    pivot = sector_df.pivot(index="sector", columns="anomaly", values="diff")

    plt.figure(figsize=(10, max(4, 0.4 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        annot=False
    )
    plt.title("Différence de rendement excédentaire par secteur et anomalie")
    plt.ylabel("Secteur")
    plt.xlabel("Anomalie")
    plt.tight_layout()

    path = os.path.join(out_dir, "sector_anomaly_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Heatmap secteurs sauvegardée : {path}")


# ---------------------------------------------------------------------------
# 5. Fonction principale appelée depuis main.py
# ---------------------------------------------------------------------------

def run_anomaly_return_analysis(df: pd.DataFrame,
                                results_dir: str = "results/anomalies"):
    """
    1) Global : effet moyen de chaque anomalie sur le rendement excédentaire
    2) Ticker : quels titres profitent le plus de chaque anomalie
    3) Secteur : quels secteurs bénéficient le plus de chaque anomalie
    4) Graphiques : barplot global + heatmap secteurs
    """

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = ensure_excess_return(df)

    # ---- Global
    global_stats = compute_global_anomaly_stats(df)
    global_path = os.path.join(results_dir, "anomaly_global_returns.xlsx")
    global_stats.to_excel(global_path, index=False)
    print(f"💾 Statistiques globales sauvegardées : {global_path}")

    # ---- Par ticker
    ticker_stats = compute_ticker_anomaly_stats(df)
    ticker_path = os.path.join(results_dir, "anomaly_ticker_returns.xlsx")
    ticker_stats.to_excel(ticker_path, index=False)
    print(f"💾 Statistiques par ticker sauvegardées : {ticker_path}")

    # ---- Par secteur
    try:
        sector_stats = compute_sector_anomaly_stats(df)
        sector_path = os.path.join(results_dir, "anomaly_sector_returns.xlsx")
        sector_stats.to_excel(sector_path, index=False)
        print(f"💾 Statistiques par secteur sauvegardées : {sector_path}")
    except ValueError as e:
        print(f"⚠️ Analyse sectorielle non effectuée : {e}")
        sector_stats = pd.DataFrame()

    # ---- Graphiques
    plot_global_anomaly_stats(global_stats, plots_dir)
    if not sector_stats.empty:
        plot_sector_heatmap(sector_stats, plots_dir)

    print("✅ Analyse des anomalies (rendements + secteurs + graphiques) terminée.")

def analyze_sectors(df, save_path="results/anomalies/anomaly_sector_returns.xlsx"):
    """
    Analyse les anomalies par secteur :
    - moyenne des rendements
    - comparaison secteurs
    - ranking
    """
    print("\n📊 Analyse sectorielle...")

    if "sector" not in df.columns:
        print("⚠️ Colonne 'sector' absente → analyse ignorée.")
        return

    sector_stats = (
        df.groupby(["sector"])
          .agg(
              avg_return=("daily_return", "mean"),
              volatility=("daily_return", "std"),
              count=("daily_return", "size")
          )
          .sort_values(by="avg_return", ascending=False)
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sector_stats.to_excel(save_path)

    print(f"   ➤ Analyse sectorielle sauvegardée dans : {save_path}")
    print(sector_stats)
