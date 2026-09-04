import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, SUB
SEED = 42
RUN_NAME = 'lgbm_sum_feature'

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

# Düz matematik özelliği: bileşenlerin toplamı (hepsi NaN ise NaN)
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

X = train[feats].copy()
X_test = test[feats].copy()
y = train['addicted_label'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
              min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
              n_estimators=3000, random_state=SEED, verbosity=-1)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])],
              eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    print(f'Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.5f}')

cv_auc = roc_auc_score(y, oof)
print(f'\nCV OOF AUC: {cv_auc:.5f}')

imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
print('\nÖzellik önemleri (top 8):')
print(imp.head(8).round(0).to_string())

sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
sub_path = f'{SUB}/{RUN_NAME}_2026-08-11.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
