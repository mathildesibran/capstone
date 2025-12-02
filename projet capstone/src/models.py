# src/models.py

from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


# --------------------------------------------------------------------
# 1. Temporal train–test split (train = 2010–2018 / test = 2019–2024+)
# --------------------------------------------------------------------
def temporal_train_test_split(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a temporal train–test split on the input DataFrame.

    - Drops rows with missing values in the selected features, target, or Date.
    - Ensures that Date is a datetime column.
    - Sorts observations by Date.
    - Uses data up to 2018 as the training set and data from 2019 onwards as the test set.
    """
    # Remove rows with missing values in features, target or Date
    ml_df = df.dropna(subset=feature_cols + ["outperform", "Date"]).copy()

    # Ensure Date is in datetime format
    ml_df["Date"] = pd.to_datetime(ml_df["Date"])

    # Sort by date
    ml_df = ml_df.sort_values("Date")

    # Temporal train / test split
    train_df = ml_df[ml_df["Date"].dt.year <= 2018]
    test_df = ml_df[ml_df["Date"].dt.year >= 2019]

    print("\nTemporal train–test split:")
    print(
        f"  Train period: {train_df['Date'].min().date()} → {train_df['Date'].max().date()} "
        f"({len(train_df)} observations)"
    )
    print(
        f"  Test  period: {test_df['Date'].min().date()} → {test_df['Date'].max().date()} "
        f"({len(test_df)} observations)"
    )

    X_train = train_df[feature_cols]
    y_train = train_df["outperform"]
    X_test = test_df[feature_cols]
    y_test = test_df["outperform"]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------
# 2. Common feature list
# --------------------------------------------------------------------
FEATURE_COLS = [
    "daily_return",
    "market_return",
    "volatility_20d",
    "ma20",
    "day_of_week",
    "month",
    "turn_of_month",
    "sell_in_may",
    "pre_holiday",
    "is_christmas",
    "is_thanksgiving",
    "is_new_year",
    "is_first_day_quarter",
]


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to the training data.
    """
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        "SMOTE applied: train size "
        f"{len(y_train)} → {len(y_train_res)} "
        f"(class 1 share: before {y_train.mean():.3f}, "
        f"after {y_train_res.mean():.3f})"
    )
    return X_train_res, y_train_res


def print_classification_results(
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> None:
    """
    Print standard classification metrics for a given model.
    """
    print(f"\n{model_name} results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"AUC:      {auc:.4f}")
        except Exception:
            print("AUC:      nan")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))


# --------------------------------------------------------------------
# 3. Logistic Regression + SMOTE
# --------------------------------------------------------------------
def run_logistic_regression(df: pd.DataFrame) -> LogisticRegression:
    """
    Train and evaluate a logistic regression classifier with SMOTE oversampling.
    """
    print("\nStep ML1: Logistic Regression")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # Baseline: always predict the majority class in the test set
    baseline_acc = max((y_test == 0).mean(), (y_test == 1).mean())
    print(f"Baseline (majority class) accuracy: {baseline_acc:.4f}")

    # Apply SMOTE on the training sample only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print_classification_results("LOGISTIC REGRESSION", y_test, y_pred, y_proba)

    return model


# --------------------------------------------------------------------
# 4. Random Forest + SMOTE
# --------------------------------------------------------------------
def run_random_forest(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train and evaluate a random forest classifier with SMOTE oversampling.
    """
    print("\nStep ML2: Random Forest")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    # Use predicted probabilities for AUC if available
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    print_classification_results("RANDOM FOREST", y_test, y_pred, y_proba)

    print("\nRandom forest feature importance:")
    for name, score in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{name:<22}: {score:.3f}")

    return model


# --------------------------------------------------------------------
# 5. Gradient Boosting + SMOTE
# --------------------------------------------------------------------
def run_gradient_boosting(df: pd.DataFrame) -> GradientBoostingClassifier:
    """
    Train and evaluate a gradient boosting classifier with SMOTE and grid search.
    """
    print("\nStep ML3: Gradient Boosting")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    base_model = GradientBoostingClassifier(random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("Best hyperparameters:", grid.best_params_)
    print_classification_results("GRADIENT BOOSTING", y_test, y_pred, y_proba)

    print("\nGradient boosting feature importance:")
    for name, score in sorted(
        zip(FEATURE_COLS, best_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{name:<22}: {score:.3f}")

    return best_model


# --------------------------------------------------------------------
# 6. Neural Network (MLP) + SMOTE
# --------------------------------------------------------------------
def run_neural_network(df: pd.DataFrame) -> MLPClassifier:
    """
    Train and evaluate a feed-forward neural network (MLP) with SMOTE oversampling.
    """
    print("\nStep ML4: Neural Network (MLPClassifier)")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    mlp.fit(X_train_res, y_train_res)
    y_pred = mlp.predict(X_test)

    try:
        y_proba = mlp.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    print_classification_results("NEURAL NETWORK", y_test, y_pred, y_proba)

    return mlp
