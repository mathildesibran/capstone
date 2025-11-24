import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================================
# ML — Régression Logistique
# ============================================================================

def run_logistic_regression(df):

    print("\n🤖 Étape ML1 : Logistic Regression...")

    feature_cols = [
        "daily_return",
        "market_return",
        "volatility_20d",
        "ma20",
        "day_of_week",
        "month",
    ]

    ml_df = df.dropna(subset=feature_cols + ["outperform"]).copy()

    X = ml_df[feature_cols]
    y = ml_df["outperform"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 LOGISTIC REGRESSION RESULTS :")
    print(f"✔️ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model



# ============================================================================
# ML — Random Forest
# ============================================================================

def run_random_forest(df):

    print("\n🌲 Étape ML2 : Random Forest...")

    feature_cols = [
        "daily_return",
        "market_return",
        "volatility_20d",
        "ma20",
        "day_of_week",
        "month",
    ]

    ml_df = df.dropna(subset=feature_cols + ["outperform"]).copy()

    X = ml_df[feature_cols]
    y = ml_df["outperform"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📊 RANDOM FOREST RESULTS :")
    print(f"✔️ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("\n🌟 Variable importance :")
    for name, score in sorted(zip(feature_cols, model.feature_importances_),
                              key=lambda x: x[1], reverse=True):
        print(f"{name:<15} : {score:.3f}")

    return model



# ============================================================================
# ML — Gradient Boosting (B2)
# ============================================================================

def run_gradient_boosting(df):

    print("\n🔥 Étape ML3 : Gradient Boosting...")

    feature_cols = [
        "daily_return",
        "market_return",
        "volatility_20d",
        "ma20",
        "day_of_week",
        "month",
    ]

    ml_df = df.dropna(subset=feature_cols + ["outperform"]).copy()

    X = ml_df[feature_cols]
    y = ml_df["outperform"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    base_model = GradientBoostingClassifier(random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3]
    }

    grid = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n🔍 Best params :", grid.best_params_)
    print("\n📊 GRADIENT BOOSTING RESULTS :")
    print(f"✔️ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("\n🌟 Variable importance :")
    for name, score in sorted(zip(feature_cols, best_model.feature_importances_),
                              key=lambda x: x[1], reverse=True):
        print(f"{name:<15} : {score:.3f}")

    return best_model
