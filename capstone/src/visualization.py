# src/visualization.py

import os
import pandas as pd
import matplotlib.pyplot as plt


FIG_DIR = "results/figures"


def _ensure_fig_dir():
    """Create the output directory for figures if it does not exist."""
    os.makedirs(FIG_DIR, exist_ok=True)


def _save_and_close(fig, filename: str):
    """Save the figure to disk and close it to free memory."""
    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {path}")


# ======================================================================
# 1) Day-of-week effect (GLOBAL) — A1
# ======================================================================

def plot_day_of_week_global():
    """
    Line plot of mean next-day excess return by weekday (global).
    Input: results/tables/anomaly_A1.xlsx (sheet: 'global')
    """
    path = "results/tables/anomaly_A1.xlsx"
    df = pd.read_excel(path, sheet_name="global")

    df["mean_excess_return_pct"] = df["mean_excess_return_tomorrow"] * 100

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        df["day_name"],
        df["mean_excess_return_pct"],
        marker="o",
        linewidth=1.5,
    )

    for x, y in zip(df["day_name"], df["mean_excess_return_pct"]):
        ax.text(
            x,
            y,
            f"{y:.3f}%",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
        )

    ax.set_xlabel("Weekday")
    ax.set_ylabel("Mean next-day excess return (%)")
    ax.set_title("Day-of-week effect (global)")

    _save_and_close(fig, "day_of_week_global.png")


# ======================================================================
# 2) Monday effect by sector — A1
# ======================================================================

def plot_monday_effect_by_sector():
    """
    Bar plot of mean next-day excess return on Mondays, by sector.
    Input: results/tables/anomaly_A1.xlsx (sheet: 'by_sector')
    """
    path = "results/tables/anomaly_A1.xlsx"
    df = pd.read_excel(path, sheet_name="by_sector")

    monday = df[df["day_name"] == "Monday"].copy().sort_values("sector")
    monday["mean_excess_return_pct"] = monday["mean_excess_return_tomorrow"] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    x = range(len(monday))
    heights = monday["mean_excess_return_pct"]

    ax.bar(x, heights)

    for xi, yi in zip(x, heights):
        ax.text(
            xi,
            yi,
            f"{yi:.3f}%",
            ha="center",
            va="bottom" if yi >= 0 else "top",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(monday["sector"], rotation=45, ha="right")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean next-day excess return (%)")
    ax.set_title("Monday effect by sector")

    _save_and_close(fig, "monday_effect_by_sector.png")


# ======================================================================
# 3) January effect by sector — A2
# ======================================================================

def plot_january_by_sector():
    """
    Bar plot of mean next-day excess return in January, by sector.
    Input: results/tables/anomaly_A2.xlsx (sheet: 'by_sector')
    """
    path = "results/tables/anomaly_A2.xlsx"
    df = pd.read_excel(path, sheet_name="by_sector")

    january = df[df["is_january"] == 1].copy().sort_values("sector")
    january["mean_excess_return_pct"] = january["mean_excess_return_tomorrow"] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    x = range(len(january))
    heights = january["mean_excess_return_pct"]

    ax.bar(x, heights)

    for xi, yi in zip(x, heights):
        ax.text(
            xi,
            yi,
            f"{yi:.3f}%",
            ha="center",
            va="bottom" if yi >= 0 else "top",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(january["sector"], rotation=45, ha="right")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean next-day excess return (%)")
    ax.set_title("January effect by sector")

    _save_and_close(fig, "january_by_sector.png")


# ======================================================================
# 4) Turn-of-the-month effect by sector — A3
# ======================================================================

def plot_turn_of_month_by_sector():
    """
    Bar plot of mean next-day excess return on turn-of-the-month days, by sector.
    Input: results/tables/anomaly_A3.xlsx (sheet: 'by_sector')
    """
    path = "results/tables/anomaly_A3.xlsx"
    df = pd.read_excel(path, sheet_name="by_sector")

    tom = df[df["turn_of_month"] == 1].copy().sort_values("sector")
    tom["mean_excess_return_pct"] = tom["mean_excess_return_tomorrow"] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    x = range(len(tom))
    heights = tom["mean_excess_return_pct"]

    ax.bar(x, heights)

    for xi, yi in zip(x, heights):
        ax.text(
            xi,
            yi,
            f"{yi:.3f}%",
            ha="center",
            va="bottom" if yi >= 0 else "top",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tom["sector"], rotation=45, ha="right")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean next-day excess return (%)")
    ax.set_title("Turn-of-the-month effect by sector")

    _save_and_close(fig, "turn_of_month_by_sector.png")


# ======================================================================
# 5) Sell-in-May effect by sector — A4
# ======================================================================

def plot_sell_in_may_by_sector():
    """
    Bar plot of mean next-day excess return for May–October, by sector.
    Input: results/tables/anomaly_A4.xlsx (sheet: 'by_sector')
    """
    path = "results/tables/anomaly_A4.xlsx"
    df = pd.read_excel(path, sheet_name="by_sector")

    sim = df[df["sell_in_may"] == 1].copy().sort_values("sector")
    sim["mean_excess_return_pct"] = sim["mean_excess_return_tomorrow"] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    x = range(len(sim))
    heights = sim["mean_excess_return_pct"]

    ax.bar(x, heights)

    for xi, yi in zip(x, heights):
        ax.text(
            xi,
            yi,
            f"{yi:.3f}%",
            ha="center",
            va="bottom" if yi >= 0 else "top",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sim["sector"], rotation=45, ha="right")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean next-day excess return (%)")
    ax.set_title("Sell-in-May effect by sector")

    _save_and_close(fig, "sell_in_may_by_sector.png")


# ======================================================================
# 6) Pre-holiday effect by sector — A5
# ======================================================================

def plot_pre_holiday_by_sector():
    """
    Bar plot of mean next-day excess return on pre-holiday days, by sector.
    Input: results/tables/anomaly_A5.xlsx (sheet: 'by_sector')
    """
    path = "results/tables/anomaly_A5.xlsx"
    df = pd.read_excel(path, sheet_name="by_sector")

    ph = df[df["pre_holiday"] == 1].copy().sort_values("sector")
    ph["mean_excess_return_pct"] = ph["mean_excess_return_tomorrow"] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    x = range(len(ph))
    heights = ph["mean_excess_return_pct"]

    ax.bar(x, heights)

    for xi, yi in zip(x, heights):
        ax.text(
            xi,
            yi,
            f"{yi:.3f}%",
            ha="center",
            va="bottom" if yi >= 0 else "top",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ph["sector"], rotation=45, ha="right")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean next-day excess return (%)")
    ax.set_title("Pre-holiday effect by sector")

    _save_and_close(fig, "pre_holiday_by_sector.png")


# ======================================================================
# 7) Model performance (global)
# ======================================================================

def plot_model_performance():
    """
    Bar plot of global model ROC AUC scores.
    Input: results/models/model_performance.csv
    """
    path = "results/models/model_performance.csv"
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = range(len(df))
    heights = df["roc_auc"]

    ax.bar(x, heights)

    for xi, yi in zip(x, heights):
        ax.text(
            xi,
            yi,
            f"{yi:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=45, ha="right")

    ax.set_ylim(0.45, max(0.6, heights.max() + 0.02))

    ax.set_xlabel("Model")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Model performance (global)")

    _save_and_close(fig, "model_performance.png")
