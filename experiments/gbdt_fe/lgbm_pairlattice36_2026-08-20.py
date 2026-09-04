"""
Izole test: A+B+D referans (84 ozellik, LightGBM OOF=0.96856) uzerine TUM C(9,2)=36
sayisal-sutun cifti icin 2-yonlu (joint) target encoding eklendi ("wide pair lattice",
74-model OOF-library notebook'undaki en buyuk tek FE kaldiraci - LGB/XGB/CatBoost'ta
tutarli +0.00023/+0.00020/+0.00017 vermisti). Daha once TEK bir cift (notif x opens,
2026-08-13) katki vermemisti (-0.00001) - o notebook'un bulgusuyla tutarli: "4 hand-picked
pair" de tek basina zayifti, oda genislik (36 ciftin TAMAMI) idi.
Cift kardinalitesi COK degisken (avg 3 satir/hucre - 80 satir/hucre arasi, olculdu) - sabit
JOINT_SMOOTH=10 ile smoothing formulu (count*mean+smooth*prior)/(count+smooth) seyrek
hucreleri otomatik prior'a cekiyor, ekstra rezolusyon/binning bu ilk turda eklenmedi.
Hiz icin sadece LightGBM (isole), tam blend'e tasima kararindan once.
"""
import itertools
import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, CONFIGS
SEED = 42
SMOOTH = 3.0
JOINT_SMOOTH = 10.0
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')
n_train = len(train)

RESID_COLS = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']
ACT3 = ['social_media_hours', 'gaming_hours', 'work_study_hours']
DECIMAL_COLS = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                 'work_study_hours', 'sleep_hours', 'weekend_screen_time']

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
    for c in DECIMAL_COLS:
        v = frame[c]
        frame[f'frac_{c}'] = v - np.floor(v)
        frame[f'd1_{c}'] = np.floor(v * 10) % 10

decimal_feat_cols = [f'frac_{c}' for c in DECIMAL_COLS] + [f'd1_{c}' for c in DECIMAL_COLS]

extra_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
              'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily',
              'ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
              'diff_weekend_daily', 'diff_daily_sum_clean',
              'max_activity3', 'range_activity3', 'gap_social_to_max', 'gap_gaming_to_max', 'gap_work_to_max'] + decimal_feat_cols

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols
te_only_cats = ['dominant_activity']

# 36 cift icin kullanilan 9 HAM sayisal sutun (jenerator forensics'te lookup-table/monotonik
# arketiplerin oldugu, GBDT'de zaten kanitlanmis 9 raw sutun - 74-model notebook'undaki
# "36 numeric pairs" ile birebir ayni sayida sutun/cift)
JOINT_BASE_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                    'work_study_hours', 'sleep_hours', 'notifications_per_day',
                    'app_opens_per_day', 'weekend_screen_time']
JOINT_PAIRS = list(itertools.combinations(JOINT_BASE_COLS, 2))
print(f'{len(JOINT_PAIRS)} cift olusturulacak')

# ---- B: imputation-augment (transduktif, raw korunuyor, 'imp_' kopya ekleniyor) ----
IMPUTE_TARGETS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                   'work_study_hours', 'sleep_hours', 'notifications_per_day',
                   'app_opens_per_day', 'weekend_screen_time']
full = pd.concat([train[cont_cols + cat_cols], test[cont_cols + cat_cols]], axis=0, ignore_index=True)
for c in cat_cols:
    full[c] = full[c].astype('category')

imp_full = pd.DataFrame(index=full.index)
for target_col in IMPUTE_TARGETS:
    preds = [c for c in cont_cols + cat_cols if c != target_col]
    mask_known = full[target_col].notna()
    reg = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                             verbosity=-1, random_state=SEED)
    reg.fit(full.loc[mask_known, preds], full.loc[mask_known, target_col])
    predicted = reg.predict(full[preds])
    imp_full[f'imp_{target_col}'] = np.where(mask_known, full[target_col], predicted)
imp_tr = imp_full.iloc[:n_train].reset_index(drop=True)
imp_te = imp_full.iloc[n_train:].reset_index(drop=True)
print(f'Imputation hazir: {time.time()-t0:.0f}s')

raw_tr = train[all_cats + extra_cols].copy()
raw_te = test[all_cats + extra_cols].copy()
for c in all_cats + extra_cols:
    for frame in (raw_tr, raw_te):
        frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

y = train['addicted_label'].values
prior = y.mean()

te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)


def make_te(col_tr, col_te, smooth):
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + smooth * prior) / (g['count'] + smooth)
    enc_te_col = pd.Series(col_te).map(g['enc'].to_dict()).fillna(prior).values
    oof_enc = np.zeros(len(col_tr))
    for tr_idx, va_idx in te_skf.split(col_tr, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + smooth * prior) / (gk['count'] + smooth)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    return oof_enc, enc_te_col


enc_tr = pd.DataFrame(index=train.index)
enc_te = pd.DataFrame(index=test.index)
freq_tr = pd.DataFrame(index=train.index)
freq_te = pd.DataFrame(index=test.index)

for c in all_cats + te_only_cats:
    if c == 'dominant_activity':
        col_tr = train[c].fillna('missing').astype(str).values
        col_te = test[c].fillna('missing').astype(str).values
    else:
        col_tr = train[c].astype(str).values
        col_te = test[c].astype(str).values

    oof_enc, enc_te_col = make_te(col_tr, col_te, SMOOTH)
    enc_tr[c] = oof_enc
    enc_te[c] = enc_te_col

    combined = np.concatenate([col_tr, col_te])
    counts = pd.Series(combined).value_counts()
    freq_tr[c] = pd.Series(col_tr).map(counts).values
    freq_te[c] = pd.Series(col_te).map(counts).values
print(f'Tekil TE + frekans hazir: {time.time()-t0:.0f}s')

# ---- Wide pair lattice: 36 ciftin TAMAMI icin joint TE ----
joint_tr = pd.DataFrame(index=train.index)
joint_te = pd.DataFrame(index=test.index)
for a, b in JOINT_PAIRS:
    name = f'te_joint_{a}_{b}'
    key_tr = (train[a].astype(str) + '_' + train[b].astype(str)).values
    key_te = (test[a].astype(str) + '_' + test[b].astype(str)).values
    oof_enc, enc_te_col = make_te(key_tr, key_te, JOINT_SMOOTH)
    joint_tr[name] = oof_enc
    joint_te[name] = enc_te_col
print(f'36-cift joint TE hazir: {time.time()-t0:.0f}s')

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned_lgb = json.load(f)
params_lgb = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned_lgb)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx = list(skf.split(train, y))


def run(name, use_joint):
    t1 = time.time()
    parts_tr = [raw_tr, enc_tr.add_prefix('te_'), freq_tr.add_prefix('freq_'), imp_tr]
    parts_te = [raw_te, enc_te.add_prefix('te_'), freq_te.add_prefix('freq_'), imp_te]
    if use_joint:
        parts_tr.append(joint_tr)
        parts_te.append(joint_te)
    X = pd.concat(parts_tr, axis=1).values
    X_test = pd.concat(parts_te, axis=1).values

    oof = np.zeros(len(X))
    for tr, va in fold_idx:
        m = lgb.LGBMClassifier(**params_lgb)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f'[{name}] CV OOF AUC = {auc:.5f}  ({time.time()-t1:.0f}s, {X.shape[1]} ozellik)')
    return auc


auc_baseline = run('baseline_ABD_84feat', use_joint=False)
auc_pairlattice = run('ABD_plus_36pair_te', use_joint=True)

print('\n' + '=' * 50)
print('OZET')
print('=' * 50)
print(f'  baseline_ABD_84feat     : {auc_baseline:.5f}')
print(f'  ABD_plus_36pair_te      : {auc_pairlattice:.5f}  (delta={auc_pairlattice-auc_baseline:+.5f})')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
