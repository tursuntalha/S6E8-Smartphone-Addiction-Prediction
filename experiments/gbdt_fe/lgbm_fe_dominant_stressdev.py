"""
Kategori I: dominant-activity / rank ozellikleri (social/gaming/work arasinda).
Hipotez: sabit ikili oranlarin aksine (trees birkac split ile yaklasik yakalayabiliyor),
3 sutun arasinda "hangisi en yuksek" (argmax/rank) bilgisini kucuk sayida axis-aligned
split ile kurmak kombinatorik olarak zor - trees'in zayif oldugu bir eksen olabilir.

Kategori J: stress_level grup-ortalamasindan sapma (OOF, TE ile ayni nested-fold
metodolojisi, feature uzerinde - target degil, sizinti riski yok ama tutarlilik icin
ayni sema kullanildi).
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
from config import DATA, SUB, CONFIGS
SEED = 42
SMOOTH = 3.0
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

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

    # --- Kategori I: dominant-activity / rank (social/gaming/work) ---
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

# --- Kategori J: stress_level grup sapmasi (feature-uzeri, target degil, sizinti riski yok;
#      grup ortalamasi SADECE train'den ogreniliyor, test'e ayni sabit harita uygulaniyor) ---
for col in ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']:
    grp_mean_map = train.groupby('stress_level')[col].mean()
    train[f'{col}_dev_stress'] = train[col] - train['stress_level'].map(grp_mean_map)
    test[f'{col}_dev_stress'] = test[col] - test['stress_level'].map(grp_mean_map)

base_ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
                   'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']
ab_cols = ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
           'diff_daily_sum', 'diff_weekend_daily']

CAT_I = ['max_activity3', 'range_activity3', 'gap_social_to_max', 'gap_gaming_to_max', 'gap_work_to_max']
CAT_I_CAT = ['dominant_activity']  # kategorik, TE'ye eklenecek
CAT_J = ['daily_screen_time_hours_dev_stress', 'social_media_hours_dev_stress',
         'gaming_hours_dev_stress', 'work_study_hours_dev_stress']

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

# dominant_activity icin de TE (kategorik, NaN'lari 'missing' stringine cevir)
dom_tr = train['dominant_activity'].fillna('missing').astype(str).values
dom_te = test['dominant_activity'].fillna('missing').astype(str).values
oof_dom, te_dom = te_fit(dom_tr, dom_te)
enc_tr['dominant_activity'] = oof_dom
enc_te['dominant_activity'] = te_dom
print(f'TE encoding hazir: {time.time()-t0:.0f}s')

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

baseline_cols = all_cats + base_ratio_cols + ab_cols
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
    new_cols = [c for c in raw_cols if c not in baseline_cols]
    if new_cols:
        imp = pd.Series(model.feature_importances_, index=Xc.columns)
        relevant = [c for c in imp.index if any(c.endswith(nc) for nc in new_cols)]
        print(f'  yeni ozellik importance: {imp[relevant].sort_values(ascending=False).to_dict()}')
    sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
    sub.to_csv(f'{SUB}/lgbm_fe_{name}_2026-08-13.csv', index=False)
    print(f'  Saved: sub/lgbm_fe_{name}_2026-08-13.csv\n')


run('baseline_AB_42feat_v4', baseline_cols)
run('I_dominant_activity', baseline_cols + CAT_I, te_extra=['dominant_activity'])
run('J_stress_deviation', baseline_cols + CAT_J)
run('IJ_combined', baseline_cols + CAT_I + CAT_J, te_extra=['dominant_activity'])

print('=' * 50)
print('OZET')
print('=' * 50)
ref = results['baseline_AB_42feat_v4']
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k:24s}: {v:.5f}  (delta vs baseline={v-ref:+.5f})')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
