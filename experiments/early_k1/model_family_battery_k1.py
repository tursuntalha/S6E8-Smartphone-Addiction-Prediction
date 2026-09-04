"""
Farkli model ailesi denemesi: LGB/XGB/CatBoost (hepsi boosting, OOF korelasyonu >0.99)
disinda genuine cesitlilik kaynagi arayisi. K1 bundle (48 ozellik, seed=42, 5-fold)
uzerinde RandomForest, ExtraTrees, HistGradientBoosting (sklearn) test edilip:
  1) standalone OOF AUC
  2) mevcut LGB baseline OOF ile Pearson korelasyonu
  3) LGB + aday modelin 2-model rank-blend'i (referans LGB=0.96831'i geciyor mu)
olculur. Amac: dusuk-korelasyonlu, blend'e gercekten katki yapabilecek bir aday bulmak.
"""
import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from scipy.stats import rankdata, pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, SUB, CONFIGS
SEED = 42
SMOOTH = 3.0
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

RESID_COLS = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']
ACT3 = ['social_media_hours', 'gaming_hours', 'work_study_hours']

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
    frame['ratio_notif_daily'] = frame['notifications_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_notif_sleep'] = frame['notifications_per_day'] / frame['sleep_hours']
    frame['ratio_opens_sleep'] = frame['app_opens_per_day'] / frame['sleep_hours']
    frame['ratio_work_sleep'] = frame['work_study_hours'] / frame['sleep_hours']
    frame['ratio_sum_daily'] = frame['sum_components'] / frame['daily_screen_time_hours']
    frame['diff_weekend_daily'] = frame['weekend_screen_time'] - frame['daily_screen_time_hours']
    mask4 = frame[RESID_COLS].notna().all(axis=1)
    clean = frame['daily_screen_time_hours'] - (frame['social_media_hours'] + frame['gaming_hours'] + frame['work_study_hours'])
    frame['diff_daily_sum_clean'] = np.where(mask4, clean, np.nan)
    mask3 = frame[ACT3].notna().all(axis=1)
    mx = frame[ACT3].max(axis=1)
    mn = frame[ACT3].min(axis=1)
    frame['max_activity3'] = np.where(mask3, mx, np.nan)
    frame['range_activity3'] = np.where(mask3, mx - mn, np.nan)
    frame['gap_social_to_max'] = np.where(mask3, mx - frame['social_media_hours'], np.nan)
    frame['gap_gaming_to_max'] = np.where(mask3, mx - frame['gaming_hours'], np.nan)
    frame['gap_work_to_max'] = np.where(mask3, mx - frame['work_study_hours'], np.nan)
    dominant = frame[ACT3].idxmax(axis=1)
    dominant = dominant.where(mask3, np.nan)
    frame['dominant_activity'] = dominant.map({'social_media_hours': 'social', 'gaming_hours': 'gaming',
                                                 'work_study_hours': 'work'})

extra_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
              'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily',
              'ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
              'diff_weekend_daily', 'diff_daily_sum_clean',
              'max_activity3', 'range_activity3', 'gap_social_to_max', 'gap_gaming_to_max', 'gap_work_to_max']

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols
te_only_cats = ['dominant_activity']

raw_tr = train[all_cats + extra_cols].copy()
raw_te = test[all_cats + extra_cols].copy()
for c in all_cats + extra_cols:
    for frame in (raw_tr, raw_te):
        frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

y = train['addicted_label'].values
prior = y.mean()

te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
enc_tr = pd.DataFrame(index=train.index)
enc_te = pd.DataFrame(index=test.index)
for c in all_cats + te_only_cats:
    if c == 'dominant_activity':
        col_tr = train[c].fillna('missing').astype(str).values
        col_te = test[c].fillna('missing').astype(str).values
    else:
        col_tr = train[c].astype(str).values
        col_te = test[c].astype(str).values
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + SMOOTH * prior) / (g['count'] + SMOOTH)
    enc_te[c] = pd.Series(col_te).map(g['enc'].to_dict()).fillna(prior).values
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + SMOOTH * prior) / (gk['count'] + SMOOTH)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    enc_tr[c] = oof_enc

X = pd.concat([raw_tr, enc_tr.add_prefix('te_')], axis=1)
X_test = pd.concat([raw_te, enc_te.add_prefix('te_')], axis=1)
print(f'K1 feature set hazir: {X.shape[1]} ozellik, {time.time()-t0:.0f}s')

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned_lgb = json.load(f)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx = list(skf.split(X, y))

oof_store = {}

# --- LGB referans (korelasyon icin) ---
ts = time.time()
params_lgb = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
                  random_state=SEED, **tuned_lgb)
oof_lgb = np.zeros(len(X))
for tr, va in fold_idx:
    m = lgb.LGBMClassifier(**params_lgb)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va] = m.predict_proba(X.iloc[va])[:, 1]
lgb_auc = roc_auc_score(y, oof_lgb)
oof_store['lgb_baseline'] = oof_lgb
print(f'[lgb_baseline] OOF AUC = {lgb_auc:.5f}  ({time.time()-ts:.0f}s)')

results = {'lgb_baseline': lgb_auc}


def run_sklearn_model(name, model_fn):
    ts = time.time()
    oof = np.zeros(len(X))
    for tr, va in fold_idx:
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    auc = roc_auc_score(y, oof)
    corr = pearsonr(oof, oof_lgb)[0]
    blend = (rankdata(oof) + rankdata(oof_lgb)) / 2
    blend_auc = roc_auc_score(y, blend)
    results[name] = auc
    oof_store[name] = oof
    print(f'[{name}] OOF AUC = {auc:.5f}  corr(vs LGB)={corr:.5f}  '
          f'2-model rank-blend AUC={blend_auc:.5f} (vs LGB alone {lgb_auc:.5f}, '
          f'delta={blend_auc-lgb_auc:+.5f})  ({time.time()-ts:.0f}s)')


run_sklearn_model('random_forest', lambda: RandomForestClassifier(
    n_estimators=500, max_depth=16, min_samples_leaf=30, max_features='sqrt',
    n_jobs=-1, random_state=SEED))

run_sklearn_model('extra_trees', lambda: ExtraTreesClassifier(
    n_estimators=500, max_depth=16, min_samples_leaf=30, max_features='sqrt',
    n_jobs=-1, random_state=SEED))

run_sklearn_model('hist_gb_sklearn', lambda: HistGradientBoostingClassifier(
    max_iter=500, max_depth=8, learning_rate=0.05, l2_regularization=1.0,
    random_state=SEED))

print('\n' + '=' * 50)
print('OZET (referans: lgb_baseline)')
print('=' * 50)
for k, v in results.items():
    print(f'  {k:20s}: {v:.5f}  (delta vs LGB={v-lgb_auc:+.5f})')

print('\nOOF korelasyon matrisi:')
names = list(oof_store.keys())
for n1 in names:
    for n2 in names:
        if n1 < n2:
            c = pearsonr(oof_store[n1], oof_store[n2])[0]
            print(f'  corr({n1}, {n2}) = {c:.5f}')

for name in ['random_forest', 'extra_trees', 'hist_gb_sklearn']:
    np.save(f'{SUB}/oof_{name}_k1_2026-08-14.npy', oof_store[name])

print(f'\nToplam sure: {time.time()-t0:.0f}s')
