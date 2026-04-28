import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.inspection import permutation_importance


# Settings


MODEL_FIGURES_DIR = "./model_figures"
FEATURE_IMPORTANCE_DIR = "./features_importance"

os.makedirs(MODEL_FIGURES_DIR, exist_ok=True)
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

path_template = "./data/{split}_model_{suffix}_aus.csv"

dataset_suffixes = {
    "1a": "before mandates",
    "1b": "after mandates",
}

model_factories = {
    "xgboost": lambda: XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        seed=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        random_state=42
    ),
}

grid_params = {
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.3, 0.6],
        "subsample": [0.1, 0.3, 0.5, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    },
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Helper functions


def safe_filename(value):
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-zA-Z0-9_\-]+", "", value)
    return value


def get_output_stem(model_name, suffix=None):
    model_name = safe_filename(model_name)

    if suffix is None:
        return f"aus_{model_name}"

    return f"aus_{model_name}_{suffix}"


def get_positive_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if proba.shape[1] == 2:
            return proba[:, 1]

        return proba

    return model.decision_function(X)


def get_permutation_importance(model, X, y, feature_cols):
    perm = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": perm.importances_mean,
        "importance_std": perm.importances_std
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    return importance_df


def plot_top10_importance(importance_df, suffix, label, model_name, is_best):
    top10 = importance_df.head(10).copy()
    top10 = top10.sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(
        top10["feature"],
        top10["importance"]
    )

    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")

    best_tag = " [BEST]" if is_best else ""
    plt.title(
        f"Top 10 Feature Importance - {suffix} ({label})\n"
        f"Model: {model_name}{best_tag}"
    )

    plt.tight_layout()
    output_stem = get_output_stem(model_name, suffix)
    output_path = os.path.join(
        FEATURE_IMPORTANCE_DIR,
        f"{output_stem}_features_importance.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Feature importance figure saved to: {output_path}")


def plot_model_performance(y_test, y_score, model_name, suffix, label, test_auc, accuracy, is_best):
    output_stem = get_output_stem(model_name, suffix)
    output_path = os.path.join(MODEL_FIGURES_DIR, f"{output_stem}.png")

    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        y_score,
        name=f"{model_name} ROC",
        ax=plt.gca()
    )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)

    best_tag = " [BEST]" if is_best else ""
    plt.title(
        f"Model Performance - {suffix} ({label})\n"
        f"Model: {model_name}{best_tag}"
    )
    plt.text(
        0.62,
        0.18,
        f"Test AUC: {test_auc:.4f}\nAccuracy: {accuracy:.4f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", alpha=0.15)
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Model performance figure saved to: {output_path}")


# Main loop


importance_results = {}

for suffix, label in dataset_suffixes.items():
    print("=" * 70)
    print(f"Dataset {suffix}: {label}")
    print("=" * 70)

    # Load data

    X_train = pd.read_csv(path_template.format(split="X_train", suffix=suffix))
    X_test  = pd.read_csv(path_template.format(split="X_test",  suffix=suffix))

    y_train = pd.read_csv(
        path_template.format(split="y_train", suffix=suffix)
    ).squeeze()

    y_test = pd.read_csv(
        path_template.format(split="y_test", suffix=suffix)
    ).squeeze()

    # Validate y shapes

    assert isinstance(y_train, pd.Series), \
        f"y_train for {suffix} should be a Series, got {type(y_train)}"
    assert isinstance(y_test, pd.Series), \
        f"y_test for {suffix} should be a Series, got {type(y_test)}"

    # Remove state_ features

    feature_cols = [
        c for c in X_train.columns
        if not c.startswith("state_")
    ]

    X_train = X_train[feature_cols]
    X_test  = X_test[feature_cols]

    # Split out a validation set for permutation importance

    val_idx = list(cv.split(X_train, y_train))[-1][1]
    X_val   = X_train.iloc[val_idx]
    y_val   = y_train.iloc[val_idx]

    # Train all models, collect results

    model_results = {}  # name -> {model, cv_auc, test_auc, ...}

    for name, factory in model_factories.items():
        print(f"\n{'-' * 40}")
        print(f"Model: {name}")
        print(f"{'-' * 40}")

        search = RandomizedSearchCV(
            estimator=factory(),
            param_distributions=grid_params[name],
            n_iter=50,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
            random_state=42
        )

        search.fit(X_train, y_train)

        print(f"  Best CV AUC: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")

        model_results[name] = {
            "model":      search.best_estimator_,
            "cv_auc":     search.best_score_,
            "best_params": search.best_params_,
        }

    # Identify best model by CV AUC

    best_model_name = max(model_results, key=lambda n: model_results[n]["cv_auc"])

    print(f"\n{'=' * 40}")
    print(f"Best model: {best_model_name} "
          f"(CV AUC: {model_results[best_model_name]['cv_auc']:.4f})")
    print(f"{'=' * 40}\n")

    # Evaluate and output every model

    importance_results[suffix] = {}

    for name, result in model_results.items():
        is_best = (name == best_model_name)
        model   = result["model"]

        print(f"\nModel: {name}{' [BEST]' if is_best else ''}")

        y_pred  = model.predict(X_test)
        y_score = get_positive_score(model, X_test)

        test_auc = roc_auc_score(y_test, y_score)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  CV AUC:   {result['cv_auc']:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

        plot_model_performance(
            y_test=y_test,
            y_score=y_score,
            model_name=name,
            suffix=suffix,
            label=label,
            test_auc=test_auc,
            accuracy=accuracy,
            is_best=is_best
        )

        # Feature importance on validation set

        importance_df = get_permutation_importance(
            model=model,
            X=X_val,
            y=y_val,
            feature_cols=feature_cols
        )

        importance_results[suffix][name] = importance_df

        print("\nTop 10 feature importance:")
        print(
            importance_df[
                ["feature", "importance", "importance_std"]
            ].head(10).to_string(index=False, float_format=lambda x: f"{x:.5f}")
        )

        feature_output_stem = get_output_stem(name, suffix)
        feature_csv_path = os.path.join(
            FEATURE_IMPORTANCE_DIR,
            f"{feature_output_stem}_features_importance.csv"
        )

        importance_df.to_csv(feature_csv_path, index=False)
        print(f"Feature importance CSV saved to: {feature_csv_path}")

        plot_top10_importance(
            importance_df=importance_df,
            suffix=suffix,
            label=label,
            model_name=name,
            is_best=is_best
        )

        print()