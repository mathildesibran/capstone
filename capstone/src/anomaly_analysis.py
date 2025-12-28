# src/anomaly_analysis.py

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# List of anomaly indicator columns used in the analysis
ANOMALY_COLS: List[str] = [
    "pre_holiday",
    "turn_of_month",
    "sell_in_may",
    "is_christmas",
    "is_thanksgiving",
    "is_new_year",
    "is_first_day_quarter",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def ensure_excess_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the DataFrame contains an 'excess_return' column
    defined as daily_return - market_return.

    If the column already exists, the input is returned unchanged.
    If daily_return or market_return are missing, an error is raised.
    """
    df = df.copy()
    if "excess_return" in df.columns:
        return df

    if {"daily_return", "market_return"}.issubset(df.columns):
        df["excess_return"] = df["daily_return"] - df["market_return"]
        return df

    raise ValueError(
        "Unable to compute 'excess_return': "
        "columns 'daily_return' and 'market_return' are required."
    )


def t_test_difference(sample_anom: pd.Series, sample_non: pd.Series) -> float:
    """
    Compute the t-statistic for the difference in means between two samples:
    anomaly days vs non-anomaly days.

    If there are fewer than 5 observations in one of the samples,
    the function returns NaN.
    """
    sample_anom = sample_anom.dropna()
    sample_non = sample_non.dropna()

    if len(sample_anom) < 5 or len(sample_non) < 5:
        return np.nan

    t_stat, _ = stats.ttest_ind(sample_anom, sample_non, equal_var=False)
    return float(t_stat)


# ---------------------------------------------------------------------------
# 1. Global statistics per anomaly
# ---------------------------------------------------------------------------


def compute_global_anomaly_stats(
    df: pd.DataFrame, anomaly_cols: List[str] = ANOMALY_COLS
) -> pd.DataFrame:
    """
    Compute global statistics for each anomaly indicator.

    For each anomaly dummy column, the function reports:
    - Number of observations on anomaly vs non-anomaly days
    - Mean excess return on anomaly days
    - Mean excess return on non-anomaly days
    - Difference in means (anomaly - non-anomaly)
    - t-statistic for the difference in means
    """
    rows = []
    for col in anomaly_cols:
        mask = df[col] == 1

        excess_anom = df.loc[mask, "excess_return"]
        excess_non = df.loc[~mask, "excess_return"]

        mean_anom = excess_anom.mean()
        mean_non = excess_non.mean()
        diff = mean_anom - mean_non
        t_stat = t_test_difference(excess_anom, excess_non)

        rows.append(
            {
                "anomaly": col,
                "n_anomaly": int(mask.sum()),
                "n_non_anomaly": int((~mask).sum()),
                "mean_excess_anomaly": mean_anom,
                "mean_excess_non_anomaly": mean_non,
                "diff": diff,
                "t_stat": t_stat,
            }
        )

    global_df = pd.DataFrame(rows)
    return global_df.sort_values("anomaly").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Statistics by ticker
# ---------------------------------------------------------------------------


def compute_ticker_anomaly_stats(
    df: pd.DataFrame, anomaly_cols: List[str] = ANOMALY_COLS
) -> pd.DataFrame:
    """
    Compute anomaly-related statistics at the ticker level.

    For each anomaly and each ticker, the function computes:
    - Mean excess return on anomaly days
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
            .rename(
                columns={"excess_return": "mean_excess_return_anomaly_days"}
            )
        )
        tmp["anomaly"] = col
        all_rows.append(tmp)

    if not all_rows:
        return pd.DataFrame(
            columns=["ticker", "anomaly", "mean_excess_return_anomaly_days"]
        )

    ticker_df = pd.concat(all_rows, ignore_index=True)
    ticker_df = ticker_df[["ticker", "anomaly", "mean_excess_return_anomaly_days"]]

    return ticker_df.sort_values(
        ["anomaly", "mean_excess_return_anomaly_days"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Statistics by sector
# ---------------------------------------------------------------------------


def compute_sector_anomaly_stats(
    df: pd.DataFrame, anomaly_cols: List[str] = ANOMALY_COLS
) -> pd.DataFrame:
    """
    Compute anomaly-related statistics at the sector level.

    For each anomaly and each sector, the function reports:
    - Number of observations on anomaly vs non-anomaly days
    - Mean excess return on anomaly days
    - Mean excess return on non-anomaly days
    - Difference in means
    - t-statistic for the difference in means

    Requires the DataFrame to contain a 'sector' column.
    """
    if "sector" not in df.columns:
        raise ValueError("Column 'sector' is missing from the DataFrame.")

    rows = []

    for col in anomaly_cols:
        for sector, g in df.groupby("sector"):
            mask = g[col] == 1
            if mask.sum() < 5:
                # Not enough anomaly days in this sector
                continue

            excess_anom = g.loc[mask, "excess_return"]
            excess_non = g.loc[~mask, "excess_return"]

            mean_anom = excess_anom.mean()
            mean_non = excess_non.mean()
            diff = mean_anom - mean_non
            t_stat = t_test_difference(excess_anom, excess_non)

            rows.append(
                {
                    "anomaly": col,
                    "sector": sector,
                    "n_anomaly": int(mask.sum()),
                    "n_non_anomaly": int((~mask).sum()),
                    "mean_excess_anomaly": mean_anom,
                    "mean_excess_non_anomaly": mean_non,
                    "diff": diff,
                    "t_stat": t_stat,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "anomaly",
                "sector",
                "n_anomaly",
                "n_non_anomaly",
                "mean_excess_anomaly",
                "mean_excess_non_anomaly",
                "diff",
                "t_stat",
            ]
        )

    sector_df = pd.DataFrame(rows)
    return sector_df.sort_values(["anomaly", "sector"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------


def plot_global_anomaly_stats(global_df: pd.DataFrame, out_dir: str) -> None:
    """
    Create a bar plot of the mean excess-return difference (anomaly vs normal)
    for each anomaly indicator.
    """
    if global_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.bar(global_df["anomaly"], global_df["diff"])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean difference\n(excess return: anomaly - normal)")
    plt.title("Average impact of anomalies on excess returns")
    plt.tight_layout()

    path = os.path.join(out_dir, "global_anomaly_diff.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Global anomaly bar plot saved to: {path}")


def plot_sector_heatmap(sector_df: pd.DataFrame, out_dir: str) -> None:
    """
    Create a heatmap of excess-return differences by sector and anomaly.
    """
    if sector_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)

    pivot = sector_df.pivot(index="sector", columns="anomaly", values="diff")

    plt.figure(figsize=(10, max(4, 0.4 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        annot=False,
    )
    plt.title("Excess return differences by sector and anomaly")
    plt.ylabel("Sector")
    plt.xlabel("Anomaly")
    plt.tight_layout()

    path = os.path.join(out_dir, "sector_anomaly_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Sector anomaly heatmap saved to: {path}")


# ---------------------------------------------------------------------------
# 5. Main function called from main.py
# ---------------------------------------------------------------------------


def run_anomaly_return_analysis(
    df: pd.DataFrame, results_dir: str = "results/anomalies"
) -> None:
    """
    Run the full anomaly return analysis workflow:

    1) Global impact of each anomaly on excess returns.
    2) Ticker-level statistics: which stocks benefit most from each anomaly.
    3) Sector-level statistics: which sectors benefit most from each anomaly.
    4) Plots: global bar plot + sector heatmap.
    """
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = ensure_excess_return(df)

    # Global statistics
    global_stats = compute_global_anomaly_stats(df)
    global_path = os.path.join(results_dir, "anomaly_global_returns.xlsx")
    global_stats.to_excel(global_path, index=False)
    print(f"Global anomaly statistics saved to: {global_path}")

    # Ticker-level statistics
    ticker_stats = compute_ticker_anomaly_stats(df)
    ticker_path = os.path.join(results_dir, "anomaly_ticker_returns.xlsx")
    ticker_stats.to_excel(ticker_path, index=False)
    print(f"Ticker-level anomaly statistics saved to: {ticker_path}")

    # Sector-level statistics
    try:
        sector_stats = compute_sector_anomaly_stats(df)
        sector_path = os.path.join(results_dir, "anomaly_sector_returns.xlsx")
        sector_stats.to_excel(sector_path, index=False)
        print(f"Sector-level anomaly statistics saved to: {sector_path}")
    except ValueError as e:
        print(f"Sector-level analysis not performed: {e}")
        sector_stats = pd.DataFrame()

    # Plots
    plot_global_anomaly_stats(global_stats, plots_dir)
    if not sector_stats.empty:
        plot_sector_heatmap(sector_stats, plots_dir)

    print("Anomaly return analysis completed.")


def analyze_sectors(
    df: pd.DataFrame,
    save_path: str = "results/anomalies/anomaly_sector_summary.xlsx",
) -> None:
    """
    Compute a simple sector-level summary:

    - Average daily return
    - Volatility of daily returns
    - Number of observations

    This is intended as a descriptive complement to the anomaly analysis.
    """
    print("\nRunning sector-level descriptive analysis...")

    if "sector" not in df.columns:
        print("Column 'sector' is missing; sector analysis is skipped.")
        return

    sector_stats = (
        df.groupby("sector")
        .agg(
            avg_return=("daily_return", "mean"),
            volatility=("daily_return", "std"),
            count=("daily_return", "size"),
        )
        .sort_values(by="avg_return", ascending=False)
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sector_stats.to_excel(save_path)

    print(f"Sector-level summary saved to: {save_path}")
    print(sector_stats)
