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

DATA = 'data'
SEED = 42
SMOOTH = 3.0
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
    frame['ratio_notif_daily'] = frame['notifications_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_notif_sleep'] = frame['notifications_per_day'] / frame['sleep_hours']
    frame['ratio_opens_sleep'] = frame['app_opens_per_day'] / frame['sleep_hours']
    frame['ratio_work_sleep'] = frame['work_study_hours'] / frame['sleep_hours']
    frame['ratio_sum_daily'] = frame['sum_components'] / frame['daily_screen_time_hours']
    frame['diff_daily_sum'] = frame['daily_screen_time_hours'] - frame['sum_components']
    frame['diff_weekend_daily'] = frame['weekend_screen_time'] - frame['daily_screen_time_hours']

R1 = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
      'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']
R2 = ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
      'diff_daily_sum', 'diff_weekend_daily']

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

raw_tr = train[all_cats + R1 + R2].copy()
raw_te = test[all_cats + R1 + R2].copy()
for c in all_cats + R1 + R2:
    for frame in (raw_tr, raw_te):
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
    g['enc'] = (g['count'] * g['mean'] + SMOOTH * prior) / (g['count'] + SMOOTH)
    enc_te[c] = pd.Series(col_te).map(g['enc'].to_dict()).fillna(prior).values
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + SMOOTH * prior) / (gk['count'] + SMOOTH)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    enc_tr[c] = oof_enc

# ucu feature subset: her model farkli bir kombinasyon goruyor (RAW+TE cekirdek her zaman var)
core_tr = pd.concat([raw_tr[all_cats], enc_tr.add_prefix('te_')], axis=1)
core_te = pd.concat([raw_te[all_cats], enc_te.add_prefix('te_')], axis=1)

X_full = pd.concat([core_tr, raw_tr[R1 + R2]], axis=1)          # LGB: tam 42 ozellik
X_full_test = pd.concat([core_te, raw_te[R1 + R2]], axis=1)

X_noR2 = pd.concat([core_tr, raw_tr[R1]], axis=1)                # XGB: RAW+TE+R1 (35 ozellik, R2 yok)
X_noR2_test = pd.concat([core_te, raw_te[R1]], axis=1)

X_noR1 = pd.concat([core_tr, raw_tr[R2]], axis=1)                # CatBoost: RAW+TE+R2 (35 ozellik, R1 yok)
X_noR1_test = pd.concat([core_te, raw_te[R2]], axis=1)

print(f'X_full: {X_full.shape[1]} ozellik | X_noR2 (XGB): {X_noR2.shape[1]} | X_noR1 (Cat): {X_noR1.shape[1]}')
print(f'Encoding hazir: {time.time()-t0:.0f}s')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx = list(skf.split(X_full, y))  # ayni fold split'i tum modellerde kullaniliyor (adil karsilastirma)

with open('sub/best_params_lgbm.json') as f:
    tuned = json.load(f)
with open('sub/best_params_xgb.json') as f:
    tuned_xgb = json.load(f)
with open('sub/best_params_cat.json') as f:
    tuned_cat = json.load(f)

# ---------- LightGBM: tam 42 ozellik ----------
oof_lgb = np.zeros(len(X_full)); pred_lgb = np.zeros(len(X_full_test))
params_lgb = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
                  random_state=SEED, **tuned)
for tr, va in fold_idx:
    m = lgb.LGBMClassifier(**params_lgb)
    m.fit(X_full.iloc[tr], y[tr], eval_set=[(X_full.iloc[va], y[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va] = m.predict_proba(X_full.iloc[va])[:, 1]
    pred_lgb += m.predict_proba(X_full_test)[:, 1] / len(fold_idx)
print(f'LightGBM (full 42) OOF AUC: {roc_auc_score(y, oof_lgb):.5f} ({time.time()-t0:.0f}s)')

# ---------- XGBoost: RAW+TE+R1 (R2 yok) ----------
oof_xgb = np.zeros(len(X_noR2)); pred_xgb = np.zeros(len(X_noR2_test))
params_xgb = dict(objective='binary:logistic', eval_metric='auc', n_estimators=5000,
                  tree_method='hist', device='cuda', random_state=SEED, **tuned_xgb)
for tr, va in fold_idx:
    m = xgb.XGBClassifier(**params_xgb)
    m.fit(X_noR2.iloc[tr], y[tr], eval_set=[(X_noR2.iloc[va], y[va])], verbose=False)
    oof_xgb[va] = m.predict_proba(X_noR2.iloc[va])[:, 1]
    pred_xgb += m.predict_proba(X_noR2_test)[:, 1] / len(fold_idx)
print(f'XGBoost (RAW+TE+R1, 35) OOF AUC: {roc_auc_score(y, oof_xgb):.5f} ({time.time()-t0:.0f}s)')

# ---------- CatBoost: RAW+TE+R2 (R1 yok) ----------
oof_cat = np.zeros(len(X_noR1)); pred_cat = np.zeros(len(X_noR1_test))
for tr, va in fold_idx:
    m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', iterations=5000,
                           task_type='GPU', devices='0', random_seed=SEED, verbose=0, **tuned_cat)
    m.fit(X_noR1.iloc[tr], y[tr], eval_set=Pool(X_noR1.iloc[va], y[va]), early_stopping_rounds=100)
    oof_cat[va] = m.predict_proba(X_noR1.iloc[va])[:, 1]
    pred_cat += m.predict_proba(X_noR1_test)[:, 1] / len(fold_idx)
print(f'CatBoost (RAW+TE+R2, 35) OOF AUC: {roc_auc_score(y, oof_cat):.5f} ({time.time()-t0:.0f}s)')

# ---------- Rank-average blend ----------
oof_rank = (rankdata(oof_lgb) + rankdata(oof_xgb) + rankdata(oof_cat)) / 3
pred_rank = (rankdata(pred_lgb) + rankdata(pred_xgb) + rankdata(pred_cat)) / 3
blend_auc = roc_auc_score(y, oof_rank)

# korelasyon (cesitlilik olcumu)
corr_lx = np.corrcoef(oof_lgb, oof_xgb)[0, 1]
corr_lc = np.corrcoef(oof_lgb, oof_cat)[0, 1]
corr_xc = np.corrcoef(oof_xgb, oof_cat)[0, 1]

print(f'\nFeature-subset diversity blend OOF AUC: {blend_auc:.5f}')
print(f'Referans (ayni-feature-set blend, seed 42): 0.96843')
print(f'Delta: {blend_auc-0.96843:+.5f}')
print(f'OOF korelasyonlari (lgb-xgb, lgb-cat, xgb-cat): {corr_lx:.5f}, {corr_lc:.5f}, {corr_xc:.5f}')
print('(karsilastirma icin: ayni-feature-set blend genelde >0.99 korelasyonlu olurdu)')

sub = pd.DataFrame({'id': test['id'], 'addicted_label': pred_rank / max(pred_rank)})
sub_path = 'sub/lgbm_xgb_cat_featuresubset_2026-08-13.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
print(f'Elapsed: {time.time()-t0:.0f}s')
