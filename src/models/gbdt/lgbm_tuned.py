import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, SUB, CONFIGS
SEED = 42
RUN_NAME = 'lgbm_tuned'

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

for frame in (train, test):
    frame['sum_components'] = frame[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame['ratio_weekend_daily'] = frame['weekend_screen_time'] / frame['daily_screen_time_hours'].replace(0, np.nan)

feats = [c for c in train.columns if c not in ['id', 'addicted_label']]
cat_cols = ['gender', 'stress_level', 'academic_work_impact']

for c in cat_cols:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')
    cats = train[c].cat.categories.union(test[c].cat.categories)
    train[c] = train[c].cat.set_categories(cats)
    test[c] = test[c].cat.set_categories(cats)

X = train[feats]
X_test = test[feats]
y = train['addicted_label'].values

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned = json.load(f)

params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])],
              eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    print(f'Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.5f} (iter {model.best_iteration_})')

cv_auc = roc_auc_score(y, oof)
print(f'\nCV OOF AUC (tuned): {cv_auc:.5f}')

sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
sub_path = f'{SUB}/{RUN_NAME}_2026-08-11.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
