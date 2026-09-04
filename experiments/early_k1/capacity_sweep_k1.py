"""
Kaggle discussion'da (WOWTIMWOW, "Model capacity was worth 18x my feature engineering")
gorulen bulgu: num_leaves dusukken (kucuk/eski bir problemden devralinan varsayilan)
engineered ratio/diff ozellikleri CV'de kazanc gibi gorunuyor, ama kapasite yeterince
artirilinca (num_leaves 15->31) kazanc sifirlaniyor - agac zaten kendi basina kurabiliyormus.

Bizim num_leaves=43, Optuna ile GUNLERDIR raw+TE (28-ozellik) seti uzerinde tunelendi
(2026-08-12), o zamandan beri ozellik sayisi 28->48'e cikti (ratio aileleri + K1) ama
kapasite hic yeniden ayarlanmadi. Bu script iki soruyu birlikte cevaplar:
  1) K1'in base(raw+TE)'e karsi kazanci (gap) num_leaves arttikca kucaluyor mu?
  2) Kapasite artisinin kendisi (base VEYA K1 icin) ek bir CV kazanci veriyor mu?
Referans: base(28 ozellik) num_leaves=43 -> bilinen ~0.96726 (2026-08-11/12),
K1(48 ozellik) num_leaves=43 -> bilinen 0.96831 (2026-08-13/14).
"""
import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, CONFIGS
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

k1_extra_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
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

raw_tr_full = train[all_cats + k1_extra_cols].copy()
raw_te_full = test[all_cats + k1_extra_cols].copy()
for c in all_cats + k1_extra_cols:
    for frame in (raw_tr_full, raw_te_full):
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

X_base = pd.concat([raw_tr_full[all_cats], enc_tr[all_cats].add_prefix('te_')], axis=1)
X_k1 = pd.concat([raw_tr_full, enc_tr.add_prefix('te_')], axis=1)
print(f'base={X_base.shape[1]} ozellik, K1={X_k1.shape[1]} ozellik, hazirlik {time.time()-t0:.0f}s')

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned_lgb = json.load(f)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx_base = list(skf.split(X_base, y))
fold_idx_k1 = list(skf.split(X_k1, y))  # ayni SEED -> ayni satirlar, ayni fold'lar

results = {}


def run(name, X, num_leaves):
    ts = time.time()
    params = dict(tuned_lgb)
    params['num_leaves'] = num_leaves
    params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
                  random_state=SEED, **params)
    oof = np.zeros(len(X))
    for tr, va in fold_idx_base:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    auc = roc_auc_score(y, oof)
    results[name] = auc
    print(f'[{name}] num_leaves={num_leaves}  OOF AUC = {auc:.5f}  ({time.time()-ts:.0f}s)')


LEAVES = [31, 43, 63, 90, 127]
for nl in LEAVES:
    run(f'base_nl{nl}', X_base, nl)
    run(f'K1_nl{nl}', X_k1, nl)
    gap = results[f'K1_nl{nl}'] - results[f'base_nl{nl}']
    print(f'  -> gap (K1-base) at num_leaves={nl}: {gap:+.5f}')

print('\n' + '=' * 60)
print('OZET')
print('=' * 60)
for nl in LEAVES:
    b, k = results[f'base_nl{nl}'], results[f'K1_nl{nl}']
    print(f'  num_leaves={nl:4d}   base={b:.5f}   K1={k:.5f}   gap={k-b:+.5f}')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
