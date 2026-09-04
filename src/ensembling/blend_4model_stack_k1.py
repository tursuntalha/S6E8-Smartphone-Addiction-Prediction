"""
Bugunku meta-stack mekanizma testinin (basit blend katkisiz ama nonlinear
polinom-etkileşimli meta-model +0.0012 verdi) bizim GUNCEL EN IYI modellerimize
uygulanmasi: K1 (48 ozellik) LGB+XGB+CatBoost+HistGB, tek seed=42, 5-fold.
Karsilastirma: (a) 4-model basit rank-blend, (b) nonlinear (LGB+etkilesim) meta-stack,
referans: tek-seed 3-model (LGB+XGB+Cat) rank-blend = 0.96849 (blend_multiseed_K1.py,
2026-08-14), cok-seedli 3-model blend = 0.96874.
"""
import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from itertools import combinations

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, SUB, CONFIGS
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
with open(f'{CONFIGS}/best_params_xgb.json') as f:
    tuned_xgb = json.load(f)
with open(f'{CONFIGS}/best_params_cat.json') as f:
    tuned_cat = json.load(f)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx = list(skf.split(X, y))

oof = {}
pred = {}

ts = time.time()
oof_lgb = np.zeros(len(X)); pred_lgb = np.zeros(len(X_test))
params_lgb = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned_lgb)
for tr, va in fold_idx:
    m = lgb.LGBMClassifier(**params_lgb)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va] = m.predict_proba(X.iloc[va])[:, 1]
    pred_lgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
oof['lgb'] = oof_lgb; pred['lgb'] = pred_lgb
print(f'[lgb] OOF AUC = {roc_auc_score(y, oof_lgb):.5f}  ({time.time()-ts:.0f}s)')

ts = time.time()
oof_xgb = np.zeros(len(X)); pred_xgb = np.zeros(len(X_test))
params_xgb = dict(objective='binary:logistic', eval_metric='auc', n_estimators=5000,
                  tree_method='hist', device='cuda', random_state=SEED, **tuned_xgb)
for tr, va in fold_idx:
    m = xgb.XGBClassifier(**params_xgb)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    oof_xgb[va] = m.predict_proba(X.iloc[va])[:, 1]
    pred_xgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
oof['xgb'] = oof_xgb; pred['xgb'] = pred_xgb
print(f'[xgb] OOF AUC = {roc_auc_score(y, oof_xgb):.5f}  ({time.time()-ts:.0f}s)')

ts = time.time()
oof_cat = np.zeros(len(X)); pred_cat = np.zeros(len(X_test))
for tr, va in fold_idx:
    m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', iterations=5000,
                           task_type='GPU', devices='0', random_seed=SEED, verbose=0, **tuned_cat)
    m.fit(X.iloc[tr], y[tr], eval_set=Pool(X.iloc[va], y[va]), early_stopping_rounds=100)
    oof_cat[va] = m.predict_proba(X.iloc[va])[:, 1]
    pred_cat += m.predict_proba(X_test)[:, 1] / len(fold_idx)
oof['cat'] = oof_cat; pred['cat'] = pred_cat
print(f'[cat] OOF AUC = {roc_auc_score(y, oof_cat):.5f}  ({time.time()-ts:.0f}s)')

ts = time.time()
oof_hgb = np.zeros(len(X)); pred_hgb = np.zeros(len(X_test))
for tr, va in fold_idx:
    m = HistGradientBoostingClassifier(max_iter=500, max_depth=8, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=SEED)
    m.fit(X.iloc[tr], y[tr])
    oof_hgb[va] = m.predict_proba(X.iloc[va])[:, 1]
    pred_hgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
oof['histgb'] = oof_hgb; pred['histgb'] = pred_hgb
print(f'[histgb] OOF AUC = {roc_auc_score(y, oof_hgb):.5f}  ({time.time()-ts:.0f}s)')

names = list(oof.keys())
print('\nOOF korelasyon matrisi:')
for a, b in combinations(names, 2):
    c = np.corrcoef(oof[a], oof[b])[0, 1]
    print(f'  corr({a},{b}) = {c:.5f}')

rank3 = (rankdata(oof['lgb']) + rankdata(oof['xgb']) + rankdata(oof['cat'])) / 3
print(f'\n[3-model (LGB+XGB+Cat) rank-blend] OOF AUC = {roc_auc_score(y, rank3):.5f}  (referans tek-seed: 0.96849)')

rank4 = sum(rankdata(oof[n]) for n in names) / len(names)
print(f'[4-model (+HistGB) basit rank-blend] OOF AUC = {roc_auc_score(y, rank4):.5f}')

base = pd.DataFrame({n: oof[n] for n in names})
inter = pd.DataFrame(index=base.index)
for a, b in combinations(names, 2):
    inter[f'{a}_x_{b}'] = base[a] * base[b]
    inter[f'{a}_m_{b}'] = base[a] - base[b]
X_meta = pd.concat([base, inter], axis=1)

oof_meta = np.zeros(len(X_meta))
test_base = pd.DataFrame({n: pred[n] for n in names})
test_inter = pd.DataFrame(index=test_base.index)
for a, b in combinations(names, 2):
    test_inter[f'{a}_x_{b}'] = test_base[a] * test_base[b]
    test_inter[f'{a}_m_{b}'] = test_base[a] - test_base[b]
X_meta_test = pd.concat([test_base, test_inter], axis=1)
pred_meta = np.zeros(len(X_meta_test))

ts = time.time()
for tr, va in fold_idx:
    m = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=2000,
                           learning_rate=0.03, num_leaves=15, min_child_samples=200,
                           verbosity=-1, random_state=SEED)
    m.fit(X_meta.iloc[tr], y[tr], eval_set=[(X_meta.iloc[va], y[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_meta[va] = m.predict_proba(X_meta.iloc[va])[:, 1]
    pred_meta += m.predict_proba(X_meta_test)[:, 1] / len(fold_idx)
meta_auc = roc_auc_score(y, oof_meta)
print(f'[polinom-etkilesimli 4-model meta-stack (LGB)] OOF AUC = {meta_auc:.5f}  ({time.time()-ts:.0f}s)')
print(f'  (referans: tek-seed 3-model blend 0.96849, cok-seedli 3-model blend 0.96874)')

sub = pd.DataFrame({'id': test['id'], 'addicted_label': pred_meta})
sub_path = f'{SUB}/lgbm_metastack_4model_k1_2026-08-14.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
