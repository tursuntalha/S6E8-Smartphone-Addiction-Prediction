import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, SUB
SEED = 42
RUN_NAME = 'lgbm_ismissing'

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

feats = [c for c in train.columns if c not in ['id', 'addicted_label']]

for c in feats:
    train[f'{c}_is_missing'] = train[c].isna().astype(np.int8)
    test[f'{c}_is_missing'] = test[c].isna().astype(np.int8)

all_feats = feats + [f'{c}_is_missing' for c in feats]

X = train[all_feats].copy()
X_test = test[all_feats].copy()

cat_cols = ['gender', 'stress_level', 'academic_work_impact']
for c in cat_cols:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')
    cats = X[c].cat.categories.union(X_test[c].cat.categories)
    X[c] = X[c].cat.set_categories(cats)
    X_test[c] = X_test[c].cat.set_categories(cats)

y = train['addicted_label'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = dict(
    objective='binary',
    metric='auc',
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.9,
    colsample_bytree=0.9,
    n_estimators=3000,
    random_state=SEED,
    verbosity=-1,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    print(f'Fold {fold+1} AUC: {roc_auc_score(y_va, oof[va_idx]):.5f}')

cv_auc = roc_auc_score(y, oof)
print(f'\nCV OOF AUC: {cv_auc:.5f}')

imp = pd.Series(model.feature_importances_, index=all_feats).sort_values(ascending=False)
print('\nTop 10 feature importance:')
print(imp.head(10).round(0).to_string())

sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
sub_path = f'{SUB}/{RUN_NAME}_2026-08-11.csv'
sub.to_csv(sub_path, index=False)
print(f'\nSaved: {sub_path}')
