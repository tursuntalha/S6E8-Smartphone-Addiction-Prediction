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
SMOOTH = 3.0
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

for frame in (train, test):
    frame['sum_components'] = frame[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame['ratio_weekend_daily'] = frame['weekend_screen_time'] / frame['daily_screen_time_hours'].replace(0, np.nan)
    # baseline 7 ratio
    frame['ratio_screen_sleep'] = frame['daily_screen_time_hours'] / frame['sleep_hours']
    frame['ratio_work_daily'] = frame['work_study_hours'] / frame['daily_screen_time_hours']
    frame['ratio_social_daily'] = frame['social_media_hours'] / frame['daily_screen_time_hours']
    frame['ratio_opens_daily'] = frame['app_opens_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_social_sleep'] = frame['social_media_hours'] / frame['sleep_hours']
    frame['ratio_weekend_sleep'] = frame['weekend_screen_time'] / frame['sleep_hours']
    frame['ratio_gaming_daily'] = frame['gaming_hours'] / frame['daily_screen_time_hours']
    # A+B (dogrulanmis, LB'de kaldi -> AB baseline'in parcasi)
    frame['ratio_notif_daily'] = frame['notifications_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_notif_sleep'] = frame['notifications_per_day'] / frame['sleep_hours']
    frame['ratio_opens_sleep'] = frame['app_opens_per_day'] / frame['sleep_hours']
    frame['ratio_work_sleep'] = frame['work_study_hours'] / frame['sleep_hours']
    frame['ratio_sum_daily'] = frame['sum_components'] / frame['daily_screen_time_hours']
    frame['diff_daily_sum'] = frame['daily_screen_time_hours'] - frame['sum_components']
    frame['diff_weekend_daily'] = frame['weekend_screen_time'] - frame['daily_screen_time_hours']
    # --- YENI: Kategori E (denenmemis oran/fark kombinasyonlari) ---
    frame['ratio_gaming_sleep'] = frame['gaming_hours'] / frame['sleep_hours']
    frame['ratio_social_gaming'] = frame['social_media_hours'] / frame['gaming_hours']
    frame['ratio_notif_opens'] = frame['notifications_per_day'] / frame['app_opens_per_day']
    frame['ratio_weekend_gaming'] = frame['weekend_screen_time'] / frame['gaming_hours']
    frame['ratio_weekend_work'] = frame['weekend_screen_time'] / frame['work_study_hours']
    frame['diff_weekend_sum'] = frame['weekend_screen_time'] - frame['sum_components']
    frame['diff_social_gaming'] = frame['social_media_hours'] - frame['gaming_hours']

base_ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
                   'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']
ab_cols = ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
           'diff_daily_sum', 'diff_weekend_daily']

CATEGORIES = {
    'E_newratios': ['ratio_gaming_sleep', 'ratio_social_gaming', 'ratio_notif_opens',
                     'ratio_weekend_gaming', 'ratio_weekend_work', 'diff_weekend_sum', 'diff_social_gaming'],
}

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# single-column TE (all_cats icin sabit, kategoriler arasinda degismiyor) - bir kere hesapla
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
print(f'TE encoding hazir: {time.time()-t0:.0f}s')

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# once AB baseline (42 ozellik) referansini bu script icinde de olcelim (tutarlilik icin)
baseline_cols = all_cats + base_ratio_cols + ab_cols
results = {}


def run(name, extra_cols):
    t1 = time.time()
    all_raw_cols = baseline_cols + extra_cols
    raw_tr = train[all_raw_cols].copy()
    raw_te = test[all_raw_cols].copy()
    for c in all_raw_cols:
        for frame in (raw_tr, raw_te):
            frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

    Xc = pd.concat([raw_tr, enc_tr.add_prefix('te_')], axis=1)
    Xc_test = pd.concat([raw_te, enc_te.add_prefix('te_')], axis=1)

    oof = np.zeros(len(Xc))
    test_pred = np.zeros(len(Xc_test))
    model = None
    for tr_idx, va_idx in skf.split(Xc, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(Xc.iloc[tr_idx], y[tr_idx], eval_set=[(Xc.iloc[va_idx], y[va_idx])],
                  eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = model.predict_proba(Xc.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(Xc_test)[:, 1] / skf.n_splits

    cv_auc = roc_auc_score(y, oof)
    results[name] = cv_auc
    print(f'[{name}] CV OOF AUC = {cv_auc:.5f}  ({time.time()-t1:.0f}s, {len(all_raw_cols)+len(all_cats)} ozellik)')
    if extra_cols:
        imp = pd.Series(model.feature_importances_, index=Xc.columns)
        print(f'  yeni özellik importance: {imp[extra_cols].sort_values(ascending=False).to_dict()}')
    sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
    sub_path = f'{SUB}/lgbm_fe_{name}_2026-08-13.csv'
    sub.to_csv(sub_path, index=False)
    print(f'  Saved: {sub_path}\n')


run('baseline_AB_42feat', [])
for cat_name, extra_cols in CATEGORIES.items():
    run(cat_name, extra_cols)

print('=' * 50)
print('ÖZET')
print('=' * 50)
ref = results['baseline_AB_42feat']
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k:20s}: {v:.5f}  (Δ vs baseline={v-ref:+.5f})')
print(f'\nToplam süre: {time.time()-t0:.0f}s')
