"""
Izole test E: K1 bundle (48 ozellik) + jeneratorun bilinen hard-rule bolge flag'leri.
Kaggle discussion'inda (broccoli beef, "Generation model of the missing original
dataset") jeneratorun COZULMUS kurali:
  p=1 KESIN eger daily_screen_time_hours>8 VEYA social_media_hours>4
  p=0 KESIN eger daily_screen_time_hours<=6 VE social_media_hours<=4
  (geri kalan dar bant orijinal kucuk veride saf gurultu)
Bu iki kesin kural aciik gen_hard_pos / gen_hard_neg flag'leri olarak ekleniyor (NaN
korunuyor - ilgili sutunlardan biri eksikse kural degerlendirilemiyor, NaN birakiliyor).
Agac modelleri bunu zaten 2 split ile yaklasik ogrenebilir ama tam esik (8.0/4.0/6.0)
ve OR/AND mantigi acik feature olarak verilince split arama isini kisaltiyor + tam
sinirdaki (8.0, 4.0, 6.0 degerlerinin kendisi) belirsizligi ortadan kaldiriyor.
Tek-seed (42), 3-model rank-blend, K1 referansiyla (LB 0.96957) karsilastirilacak.
"""
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
    # YENI: jenerator hard-rule bolge flag'leri
    d, s = frame['daily_screen_time_hours'], frame['social_media_hours']
    d_gt8 = (d > 8).where(d.notna())
    s_gt4 = (s > 4).where(s.notna())
    d_le6 = (d <= 6).where(d.notna())
    s_le4 = (s <= 4).where(s.notna())
    hard_pos = pd.Series(np.nan, index=frame.index)
    hard_pos[(d_gt8 == True) | (s_gt4 == True)] = 1.0
    hard_pos[(d_gt8 == False) & (s_gt4 == False)] = 0.0
    hard_neg = pd.Series(np.nan, index=frame.index)
    hard_neg[(d_le6 == True) & (s_le4 == True)] = 1.0
    hard_neg[(d_le6 == False) | (s_le4 == False)] = 0.0
    frame['gen_hard_pos'] = hard_pos
    frame['gen_hard_neg'] = hard_neg

region_cols = ['gen_hard_pos', 'gen_hard_neg']

extra_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
              'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily',
              'ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
              'diff_weekend_daily', 'diff_daily_sum_clean',
              'max_activity3', 'range_activity3', 'gap_social_to_max', 'gap_gaming_to_max', 'gap_work_to_max'] + region_cols

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
print(f'gen_hard_pos dagilim: {train["gen_hard_pos"].value_counts(dropna=False).to_dict()}')
print(f'gen_hard_neg dagilim: {train["gen_hard_neg"].value_counts(dropna=False).to_dict()}')

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned_lgb = json.load(f)
with open(f'{CONFIGS}/best_params_xgb.json') as f:
    tuned_xgb = json.load(f)
with open(f'{CONFIGS}/best_params_cat.json') as f:
    tuned_cat = json.load(f)

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

X = pd.concat([raw_tr, enc_tr.add_prefix('te_')], axis=1).values
X_test = pd.concat([raw_te, enc_te.add_prefix('te_')], axis=1).values
print(f'Ozellik sayisi: {X.shape[1]} (K1 48 + {len(region_cols)} generator-region)')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx = list(skf.split(X, y))

oof_lgb = np.zeros(len(X)); pred_lgb = np.zeros(len(X_test))
params_lgb = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned_lgb)
for tr, va in fold_idx:
    m = lgb.LGBMClassifier(**params_lgb)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va] = m.predict_proba(X[va])[:, 1]
    pred_lgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
print(f'LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.5f} ({time.time()-t0:.0f}s)')

oof_xgb = np.zeros(len(X)); pred_xgb = np.zeros(len(X_test))
params_xgb = dict(objective='binary:logistic', eval_metric='auc', n_estimators=5000,
                  tree_method='hist', device='cuda', random_state=SEED, **tuned_xgb)
for tr, va in fold_idx:
    m = xgb.XGBClassifier(**params_xgb)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    oof_xgb[va] = m.predict_proba(X[va])[:, 1]
    pred_xgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
print(f'XGBoost OOF AUC: {roc_auc_score(y, oof_xgb):.5f} ({time.time()-t0:.0f}s)')

oof_cat = np.zeros(len(X)); pred_cat = np.zeros(len(X_test))
for tr, va in fold_idx:
    m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', iterations=5000,
                           task_type='GPU', devices='0', random_seed=SEED, verbose=0, **tuned_cat)
    m.fit(X[tr], y[tr], eval_set=Pool(X[va], y[va]), early_stopping_rounds=100)
    oof_cat[va] = m.predict_proba(X[va])[:, 1]
    pred_cat += m.predict_proba(X_test)[:, 1] / len(fold_idx)
print(f'CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.5f} ({time.time()-t0:.0f}s)')

blend_oof = (rankdata(oof_lgb) + rankdata(oof_xgb) + rankdata(oof_cat)) / 3
print(f'\n3-model blend OOF AUC: {roc_auc_score(y, blend_oof):.5f}  (K1 tek-seed referans: 0.96843)')

blend_pred = (rankdata(pred_lgb) + rankdata(pred_xgb) + rankdata(pred_cat)) / 3
sub = pd.DataFrame({'id': test['id'], 'addicted_label': blend_pred / max(blend_pred)})
sub_path = f'{SUB}/2026-08-19/lgbm_xgb_cat_generatorregion_2026-08-19.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
print(f'Elapsed: {time.time()-t0:.0f}s')
