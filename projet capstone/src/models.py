# src/models.py

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE
from typing import List, Tuple
import numpy as np

# --------------------------------------------------------------------
# 1. Split temporel (train = 2010–2018 / test = 2019–2024+)
# --------------------------------------------------------------------
def temporal_train_test_split(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    # On enlève les lignes avec NaN sur les features / target / Date
    ml_df = df.dropna(subset=feature_cols + ["outperform", "Date"]).copy()

    # S'assurer que Date est bien en datetime
    ml_df["Date"] = pd.to_datetime(ml_df["Date"])

    # Trier par date
    ml_df = ml_df.sort_values("Date")

    # Train / Test temporel
    train_df = ml_df[ml_df["Date"].dt.year <= 2018]
    test_df = ml_df[ml_df["Date"].dt.year >= 2019]

    print("\n🕒 Temporal split :")
    print(
        f"  Train period : {train_df['Date'].min().date()} → {train_df['Date'].max().date()} "
        f"({len(train_df)} observations)"
    )
    print(
        f"  Test period  : {test_df['Date'].min().date()} → {test_df['Date'].max().date()} "
        f"({len(test_df)} observations)"
    )

    X_train = train_df[feature_cols]
    y_train = train_df["outperform"]
    X_test = test_df[feature_cols]
    y_test = test_df["outperform"]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------
# 2. Liste des features communes
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
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"📊 SMOTE appliqué : train size {len(y_train)} → {len(y_train_res)} "
        f"(proportion classe 1 : avant {y_train.mean():.3f}, après {y_train_res.mean():.3f})"
    )
    return X_train_res, y_train_res

# --------------------------------------------------------------------
# 3. Logistic Regression + SMOTE
# --------------------------------------------------------------------
def run_logistic_regression(df: pd.DataFrame) -> LogisticRegression:

    print("\n🤖 Étape ML1 : Logistic Regression...")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # Baseline : always predict majority class
    baseline_acc = max((y_test == 0).mean(), (y_test == 1).mean())
    print(f"❌ Baseline (always majority class) accuracy : {baseline_acc:.4f}")

    # ---------- SMOTE (sur l'échantillon d'entraînement uniquement) ----------
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    # -------------------------------------------------------------------------

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 LOGISTIC REGRESSION RESULTS :")
    print(f"✔️ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"✔️ AUC      : {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion matrix :")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report :")
    print(classification_report(y_test, y_pred))

    return model


# --------------------------------------------------------------------
# 4. Random Forest + SMOTE
# --------------------------------------------------------------------
def run_random_forest(df: pd.DataFrame) -> RandomForestClassifier:

    print("\n🌲 Étape ML2 : Random Forest...")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # ---------- SMOTE ----------
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    # ---------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    # Pour l'AUC, on prend les proba si possible
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    print("\n📊 RANDOM FOREST RESULTS :")
    print(f"✔️ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"✔️ AUC      : {auc:.4f}")
    print("Confusion matrix :")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report :")
    print(classification_report(y_test, y_pred))

    print("\n🌟 Variable importance :")
    for name, score in sorted(
        zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True
    ):
        print(f"{name:<22} : {score:.3f}")

    return model


# --------------------------------------------------------------------
# 5. Gradient Boosting + SMOTE
# --------------------------------------------------------------------
def run_gradient_boosting(df: pd.DataFrame) -> GradientBoostingClassifier:

    print("\n🔥 Étape ML3 : Gradient Boosting...")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # ---------- SMOTE ----------
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    )
    # ---------------------------

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

    print(f"\n🔍 Best params : {grid.best_params_}")

    print("\n📊 GRADIENT BOOSTING RESULTS :")
    print(f"✔️ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"✔️ AUC      : {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion matrix :")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report :")
    print(classification_report(y_test, y_pred))

    print("\n🌟 Variable importance :")
    for name, score in sorted(
        zip(FEATURE_COLS, best_model.feature_importances_), key=lambda x: x[1], reverse=True
    ):
        print(f"{name:<22} : {score:.3f}")

    return best_model


# --------------------------------------------------------------------
# 6. Neural Network (MLP) + SMOTE
# --------------------------------------------------------------------
def run_neural_network(df: pd.DataFrame) -> MLPClassifier:

    print("\n🔮 Étape ML4 : Neural Network (MLPClassifier)...")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, FEATURE_COLS)

    # ---------- SMOTE ----------
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    # ---------------------------

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
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    print("\n📊 NEURAL NETWORK RESULTS :")
    print(f"✔️ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"✔️ AUC      : {auc:.4f}")
    print("Confusion matrix :")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report :")
    print(classification_report(y_test, y_pred))

    return mlp
