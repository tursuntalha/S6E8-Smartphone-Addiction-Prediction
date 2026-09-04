"""
Focal loss (Lin et al. 2017) custom objective denemesi, LightGBM icin.
FL(p_t) = -(1-p_t)^gamma * log(p_t), p_t = t*p + (1-t)*(1-p), p = sigmoid(raw_margin).
alpha=0.5 (sinif dengeleme YOK) kullanildi çünkü class_weight/scale_pos_weight/
is_unbalance testleri (2026-08-14, imbalance_test_k1.py) UCÜNUN DE net negatif çiktigini
gostermisti - focal loss'un class-balancing kismi da ayni mekanizmayla zarar verebilir,
bu yuzden sadece "zor ornek" vurgusu (gamma) izole test ediliyor.

Gradyan/hessian merkezi sonlu-fark (numerical differentiation) ile hesaplaniyor -
analitik turev yerine, hata riskini azaltmak icin (dx=1e-3, cift-kesinlik icin guvenli).
K1 bundle (48 ozellik), seed=42, 5-fold. Referans: baseline (standart logloss) = 0.96831.
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

extra_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
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

raw_tr = train[all_cats + extra_cols].copy()
raw_te = test[all_cats + extra_cols].copy()
for c in all_cats + extra_cols:
    for frame in (raw_tr, raw_te):
        frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

y = train['addicted_label'].values
prior = y.mean()

te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
enc_tr = pd.DataFrame(index=train.index)
for c in all_cats + te_only_cats:
    if c == 'dominant_activity':
        col_tr = train[c].fillna('missing').astype(str).values
    else:
        col_tr = train[c].astype(str).values
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + SMOOTH * prior) / (gk['count'] + SMOOTH)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    enc_tr[c] = oof_enc

X = pd.concat([raw_tr, enc_tr.add_prefix('te_')], axis=1)
print(f'K1 feature set hazir: {X.shape[1]} ozellik, {time.time()-t0:.0f}s')

with open('sub/best_params_lgbm.json') as f:
    tuned_lgb = json.load(f)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_idx = list(skf.split(X, y))


def make_focal_loss(gamma, alpha=0.5, dx=1e-3):
    def fl_value(x, t):
        p = 1.0 / (1.0 + np.exp(-x))
        pt = np.where(t == 1, p, 1 - p)
        at = np.where(t == 1, alpha, 1 - alpha)
        pt = np.clip(pt, 1e-9, 1 - 1e-9)
        return -at * (1 - pt) ** gamma * np.log(pt)

    def focal_obj(preds, train_data):
        t = train_data.get_label()
        f_plus = fl_value(preds + dx, t)
        f_0 = fl_value(preds, t)
        f_minus = fl_value(preds - dx, t)
        grad = (f_plus - f_minus) / (2 * dx)
        hess = (f_plus - 2 * f_0 + f_minus) / (dx ** 2)
        hess = np.clip(hess, 1e-6, None)
        return grad, hess
    return focal_obj


def focal_eval(preds, train_data, gamma, alpha=0.5):
    t = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    pt = np.where(t == 1, p, 1 - p)
    at = np.where(t == 1, alpha, 1 - alpha)
    pt = np.clip(pt, 1e-9, 1 - 1e-9)
    loss = -at * (1 - pt) ** gamma * np.log(pt)
    return 'focal_loss', float(np.mean(loss)), False


results = {}


def run_baseline():
    ts = time.time()
    params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
                  random_state=SEED, **tuned_lgb)
    oof = np.zeros(len(X))
    for tr, va in fold_idx:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    auc = roc_auc_score(y, oof)
    results['baseline_logloss'] = auc
    print(f'[baseline_logloss] OOF AUC = {auc:.5f}  ({time.time()-ts:.0f}s)')


def run_focal(gamma):
    ts = time.time()
    focal_obj = make_focal_loss(gamma)
    params = dict(tuned_lgb)
    params = dict(objective=focal_obj, n_estimators=5000, verbosity=-1, random_state=SEED, **params)
    oof = np.zeros(len(X))
    for tr, va in fold_idx:
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr])
        dva = lgb.Dataset(X.iloc[va], label=y[va], reference=dtr)
        booster = lgb.train(
            params, dtr, valid_sets=[dva],
            feval=lambda p, d: focal_eval(p, d, gamma),
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        raw = booster.predict(X.iloc[va], raw_score=True)
        oof[va] = 1.0 / (1.0 + np.exp(-raw))
    auc = roc_auc_score(y, oof)
    results[f'focal_gamma{gamma}'] = auc
    print(f'[focal_gamma{gamma}] OOF AUC = {auc:.5f}  ({time.time()-ts:.0f}s)')


run_baseline()
for g in [1.0, 2.0, 3.0]:
    run_focal(g)

print('\n' + '=' * 50)
print('OZET (referans: baseline_logloss)')
print('=' * 50)
ref = results['baseline_logloss']
for k, v in results.items():
    print(f'  {k:20s}: {v:.5f}  (delta={v-ref:+.5f})')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
