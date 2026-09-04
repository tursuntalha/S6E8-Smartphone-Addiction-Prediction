import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
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
    # dogrulanmis 7 ratio (bugun LB 0.96900 ile onaylandi) - hepsinde baseline olarak kaliyor
    frame['ratio_screen_sleep'] = frame['daily_screen_time_hours'] / frame['sleep_hours']
    frame['ratio_work_daily'] = frame['work_study_hours'] / frame['daily_screen_time_hours']
    frame['ratio_social_daily'] = frame['social_media_hours'] / frame['daily_screen_time_hours']
    frame['ratio_opens_daily'] = frame['app_opens_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_social_sleep'] = frame['social_media_hours'] / frame['sleep_hours']
    frame['ratio_weekend_sleep'] = frame['weekend_screen_time'] / frame['sleep_hours']
    frame['ratio_gaming_daily'] = frame['gaming_hours'] / frame['daily_screen_time_hours']
    # --- YENI KATEGORILER (2026-08-12) ---
    # A: ek oranlar
    frame['ratio_notif_daily'] = frame['notifications_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_notif_sleep'] = frame['notifications_per_day'] / frame['sleep_hours']
    frame['ratio_opens_sleep'] = frame['app_opens_per_day'] / frame['sleep_hours']
    frame['ratio_work_sleep'] = frame['work_study_hours'] / frame['sleep_hours']
    frame['ratio_sum_daily'] = frame['sum_components'] / frame['daily_screen_time_hours']
    # B: farklar
    frame['diff_daily_sum'] = frame['daily_screen_time_hours'] - frame['sum_components']
    frame['diff_weekend_daily'] = frame['weekend_screen_time'] - frame['daily_screen_time_hours']
    # C: uyanik-gun butcesi (daily+sleep hicbir zaman 24'u asmiyor, dogrulandi)
    frame['waking_hours'] = 24 - frame['sleep_hours']
    frame['ratio_daily_waking'] = frame['daily_screen_time_hours'] / frame['waking_hours']
    frame['other_life_hours'] = frame['waking_hours'] - frame['daily_screen_time_hours']
    # D: ters-yon oranlar
    frame['session_len_opens'] = frame['daily_screen_time_hours'] / frame['app_opens_per_day']
    frame['session_len_notif'] = frame['daily_screen_time_hours'] / frame['notifications_per_day']

base_ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
                   'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']

CATEGORIES = {
    'A_moreratios': ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily'],
    'B_diffs': ['diff_daily_sum', 'diff_weekend_daily'],
    'C_wakingbudget': ['ratio_daily_waking', 'other_life_hours'],
    'D_reciprocal': ['session_len_opens', 'session_len_notif'],
}

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# TE (all_cats icin sabit, kategoriler arasinda degismiyor) - bir kere hesapla
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
print(f'TE encoding hazir: {time.time()-t0:.0f}s')

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}
for cat_name, extra_cols in CATEGORIES.items():
    t1 = time.time()
    all_raw_cols = all_cats + base_ratio_cols + extra_cols
    raw_tr = train[all_raw_cols].copy()
    raw_te = test[all_raw_cols].copy()
    for c in all_raw_cols:
        for frame in (raw_tr, raw_te):
            frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

    Xc = pd.concat([raw_tr, enc_tr.add_prefix('te_')], axis=1)
    Xc_test = pd.concat([raw_te, enc_te.add_prefix('te_')], axis=1)

    oof = np.zeros(len(Xc))
    test_pred = np.zeros(len(Xc_test))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xc, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(Xc.iloc[tr_idx], y[tr_idx], eval_set=[(Xc.iloc[va_idx], y[va_idx])],
                  eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = model.predict_proba(Xc.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(Xc_test)[:, 1] / skf.n_splits

    cv_auc = roc_auc_score(y, oof)
    results[cat_name] = cv_auc
    imp = pd.Series(model.feature_importances_, index=Xc.columns)
    print(f'[{cat_name}] CV OOF AUC = {cv_auc:.5f} (referans: 0.96771)  ({time.time()-t1:.0f}s)')
    print(f'  yeni özellik importance: {imp[extra_cols].sort_values(ascending=False).to_dict()}')

    sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
    sub_path = f'{SUB}/lgbm_fe_{cat_name}_2026-08-12.csv'
    sub.to_csv(sub_path, index=False)
    print(f'  Saved: {sub_path}\n')

print('=' * 50)
print('ÖZET (referans: baseline+7ratio = 0.96771)')
print('=' * 50)
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k:20s}: {v:.5f}  (Δ={v-0.96771:+.5f})')
print(f'\nToplam süre: {time.time()-t0:.0f}s')
