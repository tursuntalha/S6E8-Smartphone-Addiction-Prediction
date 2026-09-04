import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, SUB, CONFIGS
SEED = 42
N_FOLDS_TE = 10          # target encoding fold sayısı (yazar 10 kullanıyor)
SMOOTH = 20.0            # m-estimate smoothing
RUN_NAME = 'lgbm_te10'

t0 = time.time()
train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

for frame in (train, test):
    frame['sum_components'] = frame[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame['ratio_weekend_daily'] = frame['weekend_screen_time'] / frame['daily_screen_time_hours'].replace(0, np.nan)

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

# NaN -> 'missing' kategorisi, hepsi -> string
for frame in (train, test):
    for c in all_cats:
        frame[c] = frame[c].astype('object').where(frame[c].notna(), 'missing').astype(str)

y = train['addicted_label'].values
prior = y.mean()

te_skf = StratifiedKFold(n_splits=N_FOLDS_TE, shuffle=True, random_state=SEED)

X_enc = pd.DataFrame(index=train.index)
X_test_enc = pd.DataFrame(index=test.index)

for c in all_cats:
    col_tr = train[c].values
    col_te = test[c].values

    # global istatistik (test encoding + smoothing için)
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + SMOOTH * prior) / (g['count'] + SMOOTH)

    # test encoding: global harita
    map_te = g['enc'].to_dict()
    X_test_enc[c] = pd.Series(col_te).map(map_te).fillna(prior).values

    # OOF training encoding: her fold'da diğer 9 fold'dan hesapla
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + SMOOTH * prior) / (gk['count'] + SMOOTH)
        mapk = gk['enc'].to_dict()
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(mapk).fillna(prior).values
    X_enc[c] = oof_enc

feats = all_cats
print(f'Encoding bitti: {time.time()-t0:.0f}s')

# ---- LightGBM 5-fold CV (tuned params, numerik özellikler) ----
with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
              random_state=SEED, **tuned)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X_enc))
test_pred = np.zeros(len(X_test_enc))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_enc, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(X_enc.iloc[tr_idx], y[tr_idx], eval_set=[(X_enc.iloc[va_idx], y[va_idx])],
              eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = model.predict_proba(X_enc.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test_enc)[:, 1] / skf.n_splits
    print(f'Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.5f} (iter {model.best_iteration_})')

cv_auc = roc_auc_score(y, oof)
print(f'\nCV OOF AUC (target-encoded, 10-fold): {cv_auc:.5f}')

imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
print('\nÖnem (top 8):')
print(imp.head(8).round(0).to_string())

sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
sub_path = f'{SUB}/{RUN_NAME}_2026-08-11.csv'
sub.to_csv(sub_path, index=False)
print(f'\nSaved: {sub_path}')
print(f'Elapsed: {time.time()-t0:.0f}s')
