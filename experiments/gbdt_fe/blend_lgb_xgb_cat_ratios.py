import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, SUB, CONFIGS
SEED = 42
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

for frame in (train, test):
    frame['sum_components'] = frame[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame['ratio_weekend_daily'] = frame['weekend_screen_time'] / frame['daily_screen_time_hours'].replace(0, np.nan)
    frame['ratio_screen_sleep'] = frame['daily_screen_time_hours'] / frame['sleep_hours']
    frame['ratio_work_daily'] = frame['work_study_hours'] / frame['daily_screen_time_hours']
    frame['ratio_social_daily'] = frame['social_media_hours'] / frame['daily_screen_time_hours']
    frame['ratio_opens_daily'] = frame['app_opens_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_social_sleep'] = frame['social_media_hours'] / frame['sleep_hours']
    frame['ratio_weekend_sleep'] = frame['weekend_screen_time'] / frame['sleep_hours']
    frame['ratio_gaming_daily'] = frame['gaming_hours'] / frame['daily_screen_time_hours']

ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
              'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

raw_tr = train[all_cats + ratio_cols].copy()
raw_te = test[all_cats + ratio_cols].copy()
for c in all_cats + ratio_cols:
    for frame in (raw_tr, raw_te):
        frame[c] = frame[c].astype('object').where(frame[c].notna(), np.nan)
        frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

enc_tr = pd.DataFrame(index=train.index)
enc_te = pd.DataFrame(index=test.index)
for c in all_cats:
    col_tr = train[c].astype(str).values
    col_te = test[c].astype(str).values
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + 3.0 * prior) / (g['count'] + 3.0)
    enc_te[c] = pd.Series(col_te).map(g['enc'].to_dict()).fillna(prior).values
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + 3.0 * prior) / (gk['count'] + 3.0)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    enc_tr[c] = oof_enc

X = pd.concat([raw_tr, enc_tr.add_prefix('te_')], axis=1).values
X_test = pd.concat([raw_te, enc_te.add_prefix('te_')], axis=1).values
print(f'Features: {X.shape[1]} (28 baseline + {len(ratio_cols)} ratio) | enc: {time.time()-t0:.0f}s')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx = list(skf.split(X, y))

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned = json.load(f)

# ---------- LightGBM (CPU) ----------
oof_lgb = np.zeros(len(X)); pred_lgb = np.zeros(len(X_test))
params_lgb = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
                  random_state=SEED, **tuned)
for tr, va in fold_idx:
    m = lgb.LGBMClassifier(**params_lgb)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va] = m.predict_proba(X[va])[:, 1]
    pred_lgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
print(f'LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.5f} ({time.time()-t0:.0f}s)')

# ---------- XGBoost (GPU) ----------
with open(f'{CONFIGS}/best_params_xgb.json') as f:
    tuned_xgb = json.load(f)
oof_xgb = np.zeros(len(X)); pred_xgb = np.zeros(len(X_test))
params_xgb = dict(objective='binary:logistic', eval_metric='auc', n_estimators=5000,
                  tree_method='hist', device='cuda', random_state=SEED, **tuned_xgb)
for tr, va in fold_idx:
    m = xgb.XGBClassifier(**params_xgb)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    oof_xgb[va] = m.predict_proba(X[va])[:, 1]
    pred_xgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
print(f'XGBoost OOF AUC: {roc_auc_score(y, oof_xgb):.5f} ({time.time()-t0:.0f}s)')

# ---------- CatBoost (GPU) ----------
with open(f'{CONFIGS}/best_params_cat.json') as f:
    tuned_cat = json.load(f)
oof_cat = np.zeros(len(X)); pred_cat = np.zeros(len(X_test))
for tr, va in fold_idx:
    m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', iterations=5000,
                           task_type='GPU', devices='0', random_seed=SEED, verbose=0, **tuned_cat)
    m.fit(X[tr], y[tr], eval_set=Pool(X[va], y[va]), early_stopping_rounds=100)
    oof_cat[va] = m.predict_proba(X[va])[:, 1]
    pred_cat += m.predict_proba(X_test)[:, 1] / len(fold_idx)
print(f'CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.5f} ({time.time()-t0:.0f}s)')

# ---------- Rank-average blend ----------
oof_rank = (rankdata(oof_lgb) + rankdata(oof_xgb) + rankdata(oof_cat)) / 3
pred_rank = (rankdata(pred_lgb) + rankdata(pred_xgb) + rankdata(pred_cat)) / 3

blend_auc = roc_auc_score(y, oof_rank)
print(f'\nBlend (rank-average) OOF AUC: {blend_auc:.5f}')
print(f'Karşılaştırma -> v3 blend (ratio yok, LB 0.96862): 0.96762 | v4 blend (+ratio): {blend_auc:.5f}')

sub = pd.DataFrame({'id': test['id'], 'addicted_label': pred_rank / max(pred_rank)})
sub_path = f'{SUB}/lgbm_xgb_cat_rankblend_v4_ratios_2026-08-12.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
print(f'Elapsed: {time.time()-t0:.0f}s')
