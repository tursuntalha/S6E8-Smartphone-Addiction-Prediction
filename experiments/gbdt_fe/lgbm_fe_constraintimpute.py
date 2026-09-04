"""
Residual (diff_daily_sum) kapsamini genisletme denemesi.

Onceki versiyon (medyan-sabit back-solve) matematiksel olarak yanlisti: constraint
denklemini SABIT bir residual varsayimiyla geri cozmek, geri kazanilan degeri otomatik
olarak o sabite esitler (bilgi katmaz). Duzeltilmis yaklasim: exactly-1-missing
satirlarda eksik bileseni (veya daily'yi), o sutunun COMPLETE oldugu satirlar uzerinde
egitilmis bir LGBMRegressor ile TUM diger sutunlardan (constraint sutunlari dahil)
tahmin et - bu eda_imputability_r2.py'nin olcmus oldugu R^2'leri (daily=0.808,
social=0.544, gaming=0.413, work=0.439) kullanir, boylece per-row varyans korunur
(sabit degil).

Bu, 2026-08-11'deki basarisiz genel-amacli imputasyon denemesinden (CV 0.96134,
BOZDU) 3 onemli noktada farkli:
  1. RAW sutunlari degistirmiyor (daily/social/gaming/work orijinal NaN/-999 haliyle
     kaliyor, LightGBM native NaN routing'i korunuyor) - sadece YENI bir turetilmis
     'residual' ozelligi icin kullaniliyor.
  2. Sadece 4 sutundan TAM OLARAK BIRI eksikse tahmin yapiliyor (n=159020, %23) -
     2+ eksikse (%16) NaN birakiliyor, zincirlemeli hata-yayilimi yok.
  3. Ayni tuned hiperparametreler ve ayni 5-fold CV protokolu kullaniliyor (day-1
     denemesi farkli/tune-edilmemis hiperparametrelerle karisikti).
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
OTHER_PRED_COLS = ['age', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
                    'weekend_screen_time', 'gender', 'stress_level', 'academic_work_impact']

mask4_train = train[RESID_COLS].notna().all(axis=1)
clean_resid_train = (train.loc[mask4_train, 'daily_screen_time_hours']
                      - train.loc[mask4_train, ['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1))
print(f'Clean-row residual (train): median={clean_resid_train.median():.4f} mean={clean_resid_train.mean():.4f} '
      f'std={clean_resid_train.std():.4f}  (n={mask4_train.sum()})')

# --- eksik bilesen tahmini: sadece 4 sutundan TAM OLARAK BIRI eksik olan satirlar icin ---
cat_pred_cols = ['gender', 'stress_level', 'academic_work_impact']
train_enc = train.copy()
test_enc = test.copy()
for c in cat_pred_cols:
    train_enc[c] = train_enc[c].astype('category')
    test_enc[c] = test_enc[c].astype('category')
    cats = train_enc[c].cat.categories.union(test_enc[c].cat.categories)
    train_enc[c] = train_enc[c].cat.set_categories(cats)
    test_enc[c] = test_enc[c].cat.set_categories(cats)

imputed_value = {}  # target_col -> {'train': array-aligned-to-full-index, 'test': ...}
for target in RESID_COLS:
    predictors = [c for c in RESID_COLS if c != target] + OTHER_PRED_COLS
    fit_mask = train_enc[target].notna() & train_enc[predictors].notna().all(axis=1)
    reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                            verbosity=-1, random_state=SEED)
    reg.fit(train_enc.loc[fit_mask, predictors], train_enc.loc[fit_mask, target])

    pred_full_tr = pd.Series(np.nan, index=train.index)
    pred_full_te = pd.Series(np.nan, index=test.index)
    need_tr = train_enc[target].isna() & train_enc[predictors].notna().all(axis=1)
    need_te = test_enc[target].isna() & test_enc[predictors].notna().all(axis=1)
    if need_tr.sum() > 0:
        pred_full_tr[need_tr] = reg.predict(train_enc.loc[need_tr, predictors])
    if need_te.sum() > 0:
        pred_full_te[need_te] = reg.predict(test_enc.loc[need_te, predictors])
    imputed_value[target] = (pred_full_tr, pred_full_te)
    print(f'  {target}: {need_tr.sum()} train / {need_te.sum()} test satiri tahmin edildi '
          f'({time.time()-t0:.0f}s)')

print(f'Bilesen tahminleri hazir: {time.time()-t0:.0f}s')


def build_extended_residual(frame, pred_tr_or_te):
    d, s, g, w = (frame['daily_screen_time_hours'].copy(), frame['social_media_hours'].copy(),
                  frame['gaming_hours'].copy(), frame['work_study_hours'].copy())
    miss = frame[RESID_COLS].isna()
    n_miss = miss.sum(axis=1)

    resid = pd.Series(np.nan, index=frame.index)
    m0 = n_miss == 0
    resid[m0] = d[m0] - (s[m0] + g[m0] + w[m0])

    d_hat = d.copy(); d_hat[miss['daily_screen_time_hours']] = pred_tr_or_te['daily_screen_time_hours']
    s_hat = s.copy(); s_hat[miss['social_media_hours']] = pred_tr_or_te['social_media_hours']
    g_hat = g.copy(); g_hat[miss['gaming_hours']] = pred_tr_or_te['gaming_hours']
    w_hat = w.copy(); w_hat[miss['work_study_hours']] = pred_tr_or_te['work_study_hours']

    m1 = n_miss == 1
    resid[m1] = d_hat[m1] - (s_hat[m1] + g_hat[m1] + w_hat[m1])
    return resid


pred_tr = {k: v[0] for k, v in imputed_value.items()}
pred_te = {k: v[1] for k, v in imputed_value.items()}
train['diff_daily_sum_extended'] = build_extended_residual(train, pred_tr)
test['diff_daily_sum_extended'] = build_extended_residual(test, pred_te)

n_ext_valid = train['diff_daily_sum_extended'].notna().sum()
print(f'Extended residual kapsam: {n_ext_valid} / {len(train)} ({n_ext_valid/len(train)*100:.1f}%)')

y_all = train['addicted_label'].values
mask4_check = train[RESID_COLS].notna().all(axis=1)
only_new = train['diff_daily_sum_extended'].notna() & (~mask4_check)
print(f'Sadece extended ile kazanilan (tam 1 eksik) satirlar: {only_new.sum()}')
if only_new.sum() > 100:
    auc_new = roc_auc_score(y_all[only_new.values], train.loc[only_new, 'diff_daily_sum_extended'])
    print(f'  Bu satirlarda extended residual standalone AUC: {auc_new:.4f} (referans clean AUC: 0.7649)')

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

base_ratio_cols = ['ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
                   'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily']
ab_cols_noresid = ['ratio_notif_daily', 'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
                    'diff_weekend_daily']

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


base = all_cats + base_ratio_cols + ab_cols_noresid
run('H1_diluted_current', base + ['diff_daily_sum'])
run('H2_clean_only', base + ['diff_daily_sum_clean'])
run('H3_extended_regression', base + ['diff_daily_sum_extended'])
run('H5_extended_plus_diluted', base + ['diff_daily_sum', 'diff_daily_sum_extended'])

print('=' * 50)
print('OZET (referans: H1 = mevcut production ozelligi)')
print('=' * 50)
ref = results['H1_diluted_current']
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k:26s}: {v:.5f}  (delta vs H1={v-ref:+.5f})')
print(f'\nToplam sure: {time.time()-t0:.0f}s')
