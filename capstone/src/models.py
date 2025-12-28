# src/models.py

import os
import re
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# XGBoost may not be available depending on the environment.
# The import is handled gracefully to keep the pipeline runnable.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False


# ==========================
# 1. Dataset preparation
# ==========================

def prepare_ml_dataset(
    df_features: pd.DataFrame,
    date_col: str | None = None,
    target_col: str = "outperform_tomorrow",
    train_end: str = "2018-12-31",
    test_start: str = "2019-01-01",
):
    """
    Prepare the ML dataset using a strict time-based split:
      - Train: dates <= train_end (default: 2010–2018)
      - Test : dates >= test_start (default: 2019–2024)

    Assumptions:
    - df_features contains a date column ("Date" or "date")
    - df_features contains the target column target_col
    """
    df = df_features.copy()

    # 1) Resolve the date column name
    if date_col is None:
        if "Date" in df.columns:
            date_col = "Date"
        elif "date" in df.columns:
            date_col = "date"
        else:
            raise ValueError("No date column found (expected: 'Date' or 'date').")

    # 2) Parse dates and sort chronologically
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 3) Validate target column
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: '{target_col}'.")

    # 4) One-hot encode sector on the full dataset to keep train/test columns consistent
    if "sector" in df.columns:
        df = pd.get_dummies(df, columns=["sector"], prefix="sector")

    # 5) Drop rows with missing values (excluding the date column)
    cols_without_date = [c for c in df.columns if c != date_col]
    df = df.dropna(subset=cols_without_date)

    # 6) Time split
    train_mask = df[date_col] <= pd.to_datetime(train_end)
    test_mask = df[date_col] >= pd.to_datetime(test_start)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or test set is empty after the time split. "
            f"Check train_end={train_end}, test_start={test_start}, and the dataset date range."
        )

    # 7) Remove remaining text columns (e.g., ticker, company name) for model compatibility
    text_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    if text_cols:
        train_df = train_df.drop(columns=text_cols)
        test_df = test_df.drop(columns=text_cols)

    # 8) Define the expected feature set
    base_features = [
        # returns at time t
        "daily_return",
        "market_return",
        "excess_return",

        # momentum
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",

        # volatility
        "volatility_10d",
        "volatility_20d",

        # moving averages / ratios
        "ma_20d",
        "ma_50d",
        "price_over_ma20",

        # calendar indicators
        "day_of_week",
        "month",
        "turn_of_month",
        "sell_in_may",
        "pre_holiday",
        "is_january",
    ]

    # Add sector dummies if present
    sector_cols = [c for c in train_df.columns if c.startswith("sector_")]
    feature_cols = [c for c in (base_features + sector_cols) if c in train_df.columns]

    # 9) Leakage prevention: exclude target and future-looking variables
    forbidden = {
        target_col,
        "outperform",
        "daily_return_tomorrow",
        "market_return_tomorrow",
        "excess_return_tomorrow",
        "Date",
        "date",
    }
    feature_cols = [c for c in feature_cols if c not in forbidden]

    if not feature_cols:
        raise ValueError("No usable features found after filtering.")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    # Minimal split summary for reproducibility checks
    print(
        f"Train years: {train_df[date_col].dt.year.min()}–{train_df[date_col].dt.year.max()} | size = {len(train_df)}"
    )
    print(
        f"Test years : {test_df[date_col].dt.year.min()}–{test_df[date_col].dt.year.max()} | size = {len(test_df)}"
    )
    print(f"Number of features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


# ==========================
# 2. Generic model runner
# ==========================

def _run_model(
    name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    use_scaler: bool = True,
) -> dict:
    """
    Fit a model (optionally with StandardScaler) and report:
    - accuracy
    - ROC AUC (when probability estimates are available)
    - macro precision/recall/F1
    """
    if use_scaler:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    else:
        pipe = Pipeline([("clf", model)])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    auc = float("nan")
    try:
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
    except Exception:
        pass

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    if auc == auc:
        print(f"ROC AUC : {auc:.3f}")
    print("----- Classification report -----")
    print(classification_report(y_test, y_pred, digits=3))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    return {
        "model": name,
        "accuracy": acc,
        "roc_auc": auc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


# ==========================
# 3. Models
# ==========================

def run_logistic_regression(df: pd.DataFrame, **prep_kwargs) -> dict:
    X_train, X_test, y_train, y_test, _ = prepare_ml_dataset(df, **prep_kwargs)
    model = LogisticRegression(max_iter=2000)
    return _run_model(
        "Logistic Regression",
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        use_scaler=True,
    )


def run_random_forest(df: pd.DataFrame, **prep_kwargs) -> dict:
    X_train, X_test, y_train, y_test, _ = prepare_ml_dataset(df, **prep_kwargs)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    # Tree-based models do not require feature scaling
    return _run_model(
        "Random Forest",
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        use_scaler=False,
    )


def run_xgboost(df: pd.DataFrame, **prep_kwargs) -> dict:
    if not _HAS_XGB:
        print("\nXGBoost is not available in the current environment.")
        print("Install with: conda install -c conda-forge xgboost  (or pip install xgboost)")
        return {
            "model": "XGBoost",
            "accuracy": float("nan"),
            "roc_auc": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
        }

    X_train, X_test, y_train, y_test, _ = prepare_ml_dataset(df, **prep_kwargs)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    return _run_model(
        "XGBoost",
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        use_scaler=False,
    )


def run_neural_network(df: pd.DataFrame, **prep_kwargs) -> dict:
    X_train, X_test, y_train, y_test, _ = prepare_ml_dataset(df, **prep_kwargs)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    return _run_model(
        "Neural Network (MLP)",
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        use_scaler=True,
    )


def run_gradient_boosting(df: pd.DataFrame, **prep_kwargs) -> dict:
    """
    Optional benchmark model (kept for completeness).
    """
    X_train, X_test, y_train, y_test, _ = prepare_ml_dataset(df, **prep_kwargs)
    model = GradientBoostingClassifier(random_state=42)
    return _run_model(
        "Gradient Boosting",
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        use_scaler=False,
    )


# ==========================
# 4. Run all models
# ==========================

def run_all_models(df: pd.DataFrame, suffix: str = "", **prep_kwargs) -> pd.DataFrame:
    """
    Run the required models (Logit, RF, XGB, MLP).

    Output:
      results/models/model_performance{suffix}.csv
    """
    results = [
        run_logistic_regression(df, **prep_kwargs),
        run_random_forest(df, **prep_kwargs),
        run_xgboost(df, **prep_kwargs),
        run_neural_network(df, **prep_kwargs),
        # run_gradient_boosting(df, **prep_kwargs),
    ]

    results_df = pd.DataFrame(results)

    os.makedirs("results/models", exist_ok=True)
    out_path = f"results/models/model_performance{suffix}.csv"
    results_df.to_csv(out_path, index=False)

    print("\n=== Performance summary ===")
    print(results_df)
    print(f"\nCSV saved to: {out_path}")

    return results_df


def sanitize_suffix(name: str) -> str:
    """
    Convert a sector name into a filesystem-safe suffix.
    """
    s = re.sub(r"\s+", "_", str(name).strip())
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s
