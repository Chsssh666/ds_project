# %% Packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

# %% Data
X_train = pd.read_csv("./data/X_train_model_1_aus.csv")
X_test = pd.read_csv("./data/X_test_model_1_aus.csv")
y_train = pd.read_csv("./data/y_train_model_1_aus.csv")
y_test = pd.read_csv("./data/y_test_model_1_aus.csv")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %% Model
baseline_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    max_features='sqrt',
    n_jobs=-1
)

# %% Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_cv_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train.values.ravel(),
    cv=cv,
    scoring='roc_auc'
    n_jobs=-1
)
print("Baseline CV AUC scores:", baseline_cv_scores)
print("Baseline CV AUC mean:", baseline_cv_scores.mean())
print("Baseline CV AUC std:", baseline_cv_scores.std())

# %% Train and evaluate
baseline_model.fit(X_train, y_train.values.ravel())
baseline_y_pred = baseline_model.predict(X_test)
baseline_y_proba = baseline_model.predict_proba(X_test)[:, 1]

baseline_accuracy = accuracy_score(y_test, baseline_y_pred)
baseline_classification_report = classification_report(y_test, baseline_y_pred)
baseline_roc_auc = roc_auc_score(y_test, baseline_y_proba)
baseline_confusion_matrix = confusion_matrix(y_test, baseline_y_pred)

print("Baseline Accuracy:", baseline_accuracy)
print("Baseline Classification Report:\n", baseline_classification_report)
print("Baseline ROC AUC:", baseline_roc_auc)
print("Baseline Confusion Matrix:\n", baseline_confusion_matrix)

# %% Manual Hyperparameter Tuning
param_grid = [

    {"n_estimators": 100, "max_depth": 8, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 4},
    {"n_estimators": 500, "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 4},
]

results = []
for params in param_grid:
    model = RandomForestClassifier(
        **params,
        random_state=42,
        max_features='sqrt',
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train.values.ravel(),
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    mean_auc = cv_scores.mean()
    std_auc = cv_scores.std()
    results = {
        "params": params,
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc
    }
    results.append(results)
    print(f"Params: {params}, CV AUC Mean: {mean_auc:.4f}, CV AUC Std: {std_auc:.4f}")

# %% Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(
    by="cv_auc_mean",
    ascending=False
).reset_index(drop=True)
print(results_df)

# %% Select best params
best_params = results_df.loc[0, "params"]
print("Best Hyperparameters:", best_params)

# %% Train best model
best_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    max_features='sqrt',
    n_jobs=-1
)
best_model.fit(X_train, y_train.values.ravel())

# %% Evaluate best model
best_y_pred = best_model.predict(X_test)
best_y_proba = best_model.predict_proba(X_test)[:, 1]

best_accuracy = accuracy_score(y_test, best_y_pred)
best_classification_report = classification_report(y_test, best_y_pred)
best_roc_auc = roc_auc_score(y_test, best_y_proba)
best_confusion_matrix = confusion_matrix(y_test, best_y_pred)

print("Best Model Accuracy:", best_accuracy)
print("Best Model Classification Report:\n", best_classification_report)
print("Best Model ROC AUC:", best_roc_auc)
print("Best Model Confusion Matrix:\n", best_confusion_matrix)

# %% Feature Importance
feature_importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("Feature Importances:\n", feature_importances)

top10 = feature_importances.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top10["feature"], top10["importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.savefig("./figures/aus_model1_random_forest_feature_importance.png", bbox_inches="tight")
plt.show()

non_state_df = feature_importances[~feature_importances["feature"].str.startswith("state_")]
top10_non_state = non_state_df.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top10_non_state["feature"], top10_non_state["importance"], color="salmon")
plt.xlabel("Importance")
plt.title("Top 10 Non-State Feature Importances")
plt.gca().invert_yaxis()
plt.savefig("./figures/aus_model1_random_forest_non_state_feature_importance.png", bbox_inches="tight")
plt.show()