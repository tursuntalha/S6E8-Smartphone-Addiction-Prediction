"""
TabM (Yandex Research, resmi 'tabm' pip paketi) icin veri hazirlik. GBDT'nin ayni 84
ozellikli feature set'ini (raw+TE+frekans+imputation-augment, ABD kombinasyonu) yeniden
uretir - TabM'in kendi onerisi zaten "her sutunu (surekli dahil) target-encode et" oldugu
icin bu proven feature pipeline'i doğrudan reuse ediyoruz. GBDT ile BIREBIR AYNI
StratifiedKFold(n_splits=5, seed=42) fold atamasi (nn_data_prep_kfold.py'deki gibi) -
boylece TabM-OOF, GBDT-OOF ve NN-OOF ayni satirlarda karsilastirilabilir/stack edilebilir.

Cikti: nn_cache/tabm_prepped.npz
"""
import pandas as pd
import numpy as np
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

DATA = 'data'
SEED = 42
SMOOTH = 3.0
CACHE_DIR = 'nn_cache'
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

y = train['addicted_label'].values.astype(np.float32)
prior = y.mean()

te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
enc_tr = pd.DataFrame(index=train.index)
enc_te = pd.DataFrame(index=test.index)
freq_tr = pd.DataFrame(index=train.index)
freq_te = pd.DataFrame(index=test.index)


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

X = pd.concat([raw_tr, enc_tr.add_prefix('te_'), freq_tr.add_prefix('freq_'), imp_tr], axis=1).values.astype(np.float32)
X_test = pd.concat([raw_te, enc_te.add_prefix('te_'), freq_te.add_prefix('freq_'), imp_te], axis=1).values.astype(np.float32)
print(f'Ozellik sayisi: {X.shape[1]} (ABD 84 ile ayni)')

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_id = np.full(n_train, -1, dtype=np.int64)
for fold, (_, va_idx) in enumerate(skf.split(np.zeros(n_train), y)):
    fold_id[va_idx] = fold
assert (fold_id >= 0).all()

np.savez(
    f'{CACHE_DIR}/tabm_prepped.npz',
    X=X_s, X_test=X_test_s, y=y, fold_id=fold_id, test_id=test['id'].values,
)
print(f'Saved: {CACHE_DIR}/tabm_prepped.npz  X={X_s.shape}  X_test={X_test_s.shape}')
print(f'Toplam sure: {time.time()-t0:.0f}s')
