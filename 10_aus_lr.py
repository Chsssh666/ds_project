# %% Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import pickle

# %% parameters
n_splits = 5
seed = 19980707
kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=seed)
lr = LogisticRegression(max_iter=5000)

# %% cv lr
def cross_validate_model(group_number):
    X_train = pd.read_csv(f'./data/aus_X_train_{group_number}.csv', keep_default_na=False)
    y_train = pd.read_csv(f'./data/aus_y_train_{group_number}.csv', keep_default_na=False).values.ravel()

    cv_scores = {'test_precision': [], 'test_recall': [],
                 'test_roc_auc': [], 'test_accuracy': [], 'test_f1': []}

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Upsample minority class within fold
        X_tr, y_tr = RandomOverSampler().fit_resample(X_tr, y_tr)

        clf = lr.fit(X_tr, y_tr)
        pred = clf.predict(X_val)
        prob = clf.predict_proba(X_val)[:, 1]

        cv_scores['test_precision'].append(precision_score(y_val, pred))
        cv_scores['test_recall'].append(recall_score(y_val, pred))
        cv_scores['test_roc_auc'].append(roc_auc_score(y_val, prob))
        cv_scores['test_accuracy'].append(accuracy_score(y_val, pred))
        cv_scores['test_f1'].append(f1_score(y_val, pred))

    for key in cv_scores:
        cv_scores[key] = np.array(cv_scores[key])

    print(f"\nGroup {group_number} - Pre/Post vaccination")
    for metric, key in zip(['Precision', 'Recall', 'AUC', 'Accuracy', 'F1'],
                           ['test_precision', 'test_recall', 'test_roc_auc', 'test_accuracy', 'test_f1']):
        print(f"  {metric}: {cv_scores[key].mean():.3f} ± {cv_scores[key].std():.3f}")

    with open(f'./results/group_{group_number}_logistic_reg.pkl', 'wb') as f:
        pickle.dump(cv_scores, f)

cross_validate_model(0)
cross_validate_model(1)