"""
Bugunun iki kucuk-pozitif bulgusunu birlestirip additiflik kontrolu:
  H2: diff_daily_sum -> diff_daily_sum_clean (Δ=+0.00006 tek basina)
  I:  dominant-activity/rank ailesi (Δ=+0.00005 tek basina)
Beklenti: farkli bilgi kaynaklarindan geldikleri icin ~additive (~+0.0001) olabilirler,
ama ayni feature havuzunda etkilesip sonup sonmediklerini gormek lazim.
"""
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
    frame['diff_daily_sum'] = frame['daily_screen_time_hours'] - frame['sum_components']
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

base_ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
                   'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']
ab_cols_noresid = ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
                    'diff_weekend_daily']
CAT_I = ['max_activity3', 'range_activity3', 'gap_social_to_max', 'gap_gaming_to_max', 'gap_work_to_max']

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)


def te_fit(col_tr, col_te, smooth=SMOOTH):
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + smooth * prior) / (g['count'] + smooth)
    enc_te = pd.Series(col_te).map(g['enc'].to_dict()).fillna(prior).values
    oof_enc = np.zeros(len(col_tr))
    for tr_idx, va_idx in te_skf.split(col_tr, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + smooth * prior) / (gk['count'] + smooth)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    return oof_enc, enc_te


enc_tr = pd.DataFrame(index=train.index)
enc_te = pd.DataFrame(index=test.index)
for c in all_cats:
    oof_enc, enc_te_col = te_fit(train[c].astype(str).values, test[c].astype(str).values)
    enc_tr[c] = oof_enc
    enc_te[c] = enc_te_col
dom_tr = train['dominant_activity'].fillna('missing').astype(str).values
dom_te = test['dominant_activity'].fillna('missing').astype(str).values
oof_dom, te_dom = te_fit(dom_tr, dom_te)
enc_tr['dominant_activity'] = oof_dom
enc_te['dominant_activity'] = te_dom
print(f'TE encoding hazir: {time.time()-t0:.0f}s')

with open('sub/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}


def run(name, raw_cols, te_extra=None):
    t1 = time.time()
    raw_tr = train[raw_cols].copy()
    raw_te = test[raw_cols].copy()
    for c in raw_cols:
        for frame in (raw_tr, raw_te):
            frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

    parts_tr = [raw_tr, enc_tr[all_cats].add_prefix('te_')]
    parts_te = [raw_te, enc_te[all_cats].add_prefix('te_')]
    if te_extra:
        parts_tr.append(enc_tr[te_extra].add_prefix('te_'))
        parts_te.append(enc_te[te_extra].add_prefix('te_'))
    Xc = pd.concat(parts_tr, axis=1)
    Xc_test = pd.concat(parts_te, axis=1)

    oof = np.zeros(len(Xc))
    test_pred = np.zeros(len(Xc_test))
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
    sub.to_csv(f'sub/lgbm_fe_{name}_2026-08-13.csv', index=False)
    print(f'  Saved: sub/lgbm_fe_{name}_2026-08-13.csv\n')


base = all_cats + base_ratio_cols + ab_cols_noresid
run('K0_current_production', base + ['diff_daily_sum'])
run('K1_bundle_clean_dominant', base + ['diff_daily_sum_clean'] + CAT_I, te_extra=['dominant_activity'])

print('=' * 50)
print('OZET (referans: K0 = mevcut production)')
print('=' * 50)
ref = results['K0_current_production']
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k:26s}: {v:.5f}  (delta vs K0={v-ref:+.5f})')
print(f'  (beklenen additive: H2 +0.00006 + I +0.00005 = ~+0.00011)')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
