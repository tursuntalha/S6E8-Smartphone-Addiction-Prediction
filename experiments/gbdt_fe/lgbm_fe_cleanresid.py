import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = 'data'
SEED = 42
SMOOTH = 3.0
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

RESID_COLS = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']

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
    frame['diff_daily_sum'] = frame['daily_screen_time_hours'] - frame['sum_components']  # mevcut (seyreltilmis)
    frame['diff_weekend_daily'] = frame['weekend_screen_time'] - frame['daily_screen_time_hours']
    # --- YENI: temiz residual (sadece 4 sutun da doluyken, aksi halde NaN->-999) ---
    mask4 = frame[RESID_COLS].notna().all(axis=1)
    clean = frame['daily_screen_time_hours'] - (frame['social_media_hours'] + frame['gaming_hours'] + frame['work_study_hours'])
    frame['diff_daily_sum_clean'] = np.where(mask4, clean, np.nan)

base_ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
                   'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']
ab_cols = ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
           'diff_daily_sum', 'diff_weekend_daily']

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

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
print(f'TE encoding hazir: {time.time()-t0:.0f}s')

with open('sub/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

baseline_cols = all_cats + base_ratio_cols + ab_cols
results = {}


def run(name, cols):
    t1 = time.time()
    raw_tr = train[cols].copy()
    raw_te = test[cols].copy()
    for c in cols:
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
    print(f'[{name}] CV OOF AUC = {cv_auc:.5f}  ({time.time()-t1:.0f}s, {Xc.shape[1]} ozellik)')
    sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
    sub_path = f'sub/lgbm_fe_{name}_2026-08-13.csv'
    sub.to_csv(sub_path, index=False)
    print(f'  Saved: {sub_path}\n')


run('baseline_AB_42feat_v3', baseline_cols)

cols_replace = [c for c in baseline_cols if c != 'diff_daily_sum'] + ['diff_daily_sum_clean']
run('G_replace_cleanresid', cols_replace)

cols_add = baseline_cols + ['diff_daily_sum_clean']
run('G_add_cleanresid', cols_add)

print('=' * 50)
print('OZET')
print('=' * 50)
ref = results['baseline_AB_42feat_v3']
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k:24s}: {v:.5f}  (delta vs baseline={v-ref:+.5f})')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
