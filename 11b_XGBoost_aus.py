# %% Packages
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score, 
    confusion_matrix
)
import matplotlib.pyplot as plt

# %% Load the data
X_train = pd.read_csv("./data/X_train_model_1b_aus.csv")
X_test = pd.read_csv("./data/X_test_model_1b_aus.csv")
y_train = pd.read_csv("./data/y_train_model_1b_aus.csv")
y_test = pd.read_csv("./data/y_test_model_1b_aus.csv")

# %% Check the shapes of the data

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %% Baseline XGBoost Classifier
baseline_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# %% Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_cv_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("Baseline CV AUC Scores:", baseline_cv_scores)
print("Baseline CV AUC Mean:", baseline_cv_scores.mean())
print("Baseline CV AUC Std:", baseline_cv_scores.std())

# %% Baseline model training
baseline_model.fit(X_train, y_train)

# %% Baseline predictions & evaluation
baseline_y_pred = baseline_model.predict(X_test)
baseline_y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]

baseline_accuracy = accuracy_score(y_test, baseline_y_pred)
baseline_classification_report = classification_report(y_test, baseline_y_pred)
baseline_roc_auc = roc_auc_score(y_test, baseline_y_pred_proba)
baseline_confusion_matrix = confusion_matrix(y_test, baseline_y_pred)

print("Baseline Accuracy:", baseline_accuracy)
print("Baseline Classification Report:\n", baseline_classification_report)
print("Baseline ROC AUC:", baseline_roc_auc)
print("Baseline Confusion Matrix:\n", baseline_confusion_matrix)

# %% Manual Hyperparameter Tuning
param_grid = [
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
    {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300},
]

results = []

for params in param_grid:
    model = XGBClassifier(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    mean_auc = cv_scores.mean()
    std_auc = cv_scores.std()
    result = {
        "params": params,
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc
    }
    results.append(result)

    print(f"Params: {params}, CV AUC Mean: {mean_auc:.4f}, CV AUC Std: {std_auc:.4f}")


# %% Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by="cv_auc_mean", ascending=False
                                               ).reset_index(drop=True)
print("\nManual Tuning Results:")
print(results_df)

# %% Select best parameters
best_params = results_df.loc[0, "params"]
print("\nBest Parameters:", best_params)

# %% Train best model
best_model = XGBClassifier(
    **best_params,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
best_model.fit(X_train, y_train)

# %% Best model predictions & evaluation
best_y_pred = best_model.predict(X_test)
best_y_pred_proba = best_model.predict_proba(X_test)[:, 1]
best_accuracy = accuracy_score(y_test, best_y_pred)
best_classification_report = classification_report(y_test, best_y_pred)
best_roc_auc = roc_auc_score(y_test, best_y_pred_proba)
best_confusion_matrix = confusion_matrix(y_test, best_y_pred)  
print("Best Model Accuracy:", best_accuracy)
print("Best Model Classification Report:\n", best_classification_report)
print("Best Model ROC AUC:", best_roc_auc)
print("Best Model Confusion Matrix:\n", best_confusion_matrix)

# %% Feature importance
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_model.feature_importances_
}).sort_values(by="importance", ascending=False)
print("\nFeature Importance:\n", feature_importance)

top10 = feature_importance.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top10["feature"], top10["importance"], color="skyblue")
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importance - XGBoost")
plt.gca().invert_yaxis()
plt.savefig("./figures/aus_model1b_xgboost_feature_importance.png", bbox_inches="tight")
plt.show()

non_state_df = feature_importance[~feature_importance["feature"].str.startswith("state_")]
top10_non_state = non_state_df.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top10_non_state["feature"], top10_non_state["importance"], color="salmon")
plt.xlabel("Feature Importance")
plt.title("Top 10 Non-State Feature Importance - XGBoost")
plt.gca().invert_yaxis()
plt.savefig("./figures/aus_model1b_xgboost_non_state_feature_importance.png", bbox_inches="tight")
plt.show()

# %%
