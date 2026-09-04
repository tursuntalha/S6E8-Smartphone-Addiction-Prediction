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
JOINT_SMOOTH = 10.0  # hucre basina ort. ~15 satir (notif x opens), daha guclu smoothing
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

base_ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
                   'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']
ab_cols = ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
           'diff_daily_sum', 'diff_weekend_daily']

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols
baseline_cols = all_cats + base_ratio_cols + ab_cols

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


# single-column TE for all_cats
enc_tr = pd.DataFrame(index=train.index)
enc_te = pd.DataFrame(index=test.index)
for c in all_cats:
    oof_enc, enc_te_col = make_te(train[c].astype(str).values, test[c].astype(str).values, SMOOTH)
    enc_tr[c] = oof_enc
    enc_te[c] = enc_te_col
print(f'Tekil TE encoding hazir: {time.time()-t0:.0f}s')

# joint (2-yonlu) TE cifti: notifications_per_day x app_opens_per_day
JOINT_PAIRS = [('notifications_per_day', 'app_opens_per_day')]

joint_tr = pd.DataFrame(index=train.index)
joint_te = pd.DataFrame(index=test.index)
joint_cols = []
for a, b in JOINT_PAIRS:
    name = f'te_joint_{a}_{b}'
    key_tr = train[a].astype(str) + '_' + train[b].astype(str)
    key_te = test[a].astype(str) + '_' + test[b].astype(str)
    oof_enc, enc_te_col = make_te(key_tr.values, key_te.values, JOINT_SMOOTH)
    joint_tr[name] = oof_enc
    joint_te[name] = enc_te_col
    joint_cols.append(name)
print(f'Joint TE encoding hazir ({joint_cols}): {time.time()-t0:.0f}s')

with open('sub/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}


def run(name, use_joint):
    t1 = time.time()
    raw_tr = train[baseline_cols].copy()
    raw_te = test[baseline_cols].copy()
    for c in baseline_cols:
        for frame in (raw_tr, raw_te):
            frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

    parts_tr = [raw_tr, enc_tr.add_prefix('te_')]
    parts_te = [raw_te, enc_te.add_prefix('te_')]
    if use_joint:
        parts_tr.append(joint_tr)
        parts_te.append(joint_te)
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
    if use_joint:
        imp = pd.Series(model.feature_importances_, index=Xc.columns)
        print(f'  joint TE importance: {imp[joint_cols].sort_values(ascending=False).to_dict()}')
    sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
    sub_path = f'sub/lgbm_fe_{name}_2026-08-13.csv'
    sub.to_csv(sub_path, index=False)
    print(f'  Saved: {sub_path}\n')


run('baseline_AB_42feat_v2', False)
run('F_jointte_notif_opens', True)

print('=' * 50)
print('ÖZET')
print('=' * 50)
ref = results['baseline_AB_42feat_v2']
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k:24s}: {v:.5f}  (delta vs baseline={v-ref:+.5f})')
print(f'\nToplam süre: {time.time()-t0:.0f}s')
