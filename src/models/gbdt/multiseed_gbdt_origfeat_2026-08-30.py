"""
Acik madde #2 (08-21'den beri ertelenmisti, 08-29'da atlandi, bu gece siraya alindi):
3-seed bagging, ama artik GUNCEL en iyi tarif uzerinde (K1+A+B+D+29 ORIG-CDF, 113 ozellik -
lgbm_orig_features_lb_2026-08-21.py'nin AYNISI), 08-14'teki blend_multiseed_K1.py deseni
takip edilerek. Gecmis kanit: K0->K1 gecisinde 3-seed bagging CV+0.00026 -> LB+0.00005
(dusuk ama POZITIF transfer, ~0.19-0.4x). Bu gece ucuz/dusuk-riskli bir kaldirac olarak
calistiriliyor.

Verimlilik: ORIG-CDF hesaplamasi (KDE ~580s) ve imputasyon (9 LGBMRegressor) SEED'DEN
BAGIMSIZ (dis referans veriye dayanıyor) - bir kez hesaplanip tum seed'lerde paylasiliyor.
Sadece TE/freq-encoding (10-fold, seed'e bagli) ve 3-model 5-fold egitimi (seed'e bagli)
her seed icin tekrarlaniyor.

Cikti:
  nn_cache/gbdt_abd_origfeat_multiseed_oof.npy, nn_cache/gbdt_abd_origfeat_multiseed_test_pred.npy
  sub/2026-08-30/lgbm_xgb_cat_ABD_origfeat_multiseed_2026-08-30.csv
"""
import pandas as pd
import numpy as np
import json
import os
import time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, NN_CACHE as CACHE_DIR, SUB, CONFIGS
SEEDS = [42, 43, 44]
SMOOTH = 3.0
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')
orig = pd.read_csv(f'{DATA}/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')
n_train = len(train)

# ================= ORIG-CDF (lgbm_orig_features_lb_2026-08-21.py ile BIREBIR AYNI, seed-bagimsiz) =================
RAW_COLS_ORIG = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                  'work_study_hours', 'sleep_hours', 'notifications_per_day',
                  'app_opens_per_day', 'weekend_screen_time', 'gender', 'stress_level',
                  'academic_work_impact']
NUM_ORIG = RAW_COLS_ORIG[:9]
ORIG_CDF_SOURCE_COLS = ['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours']
ORIG_Q50_SOURCE_COLS = ['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours',
                         'notifications_per_day', 'app_opens_per_day']
ORIG_Q50_Y1_SOURCE_COLS = ['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours',
                            'app_opens_per_day']
ORIG_MEAN_SOURCE_COLS = ['daily_screen_time_hours', 'weekend_screen_time',
                          'notifications_per_day', 'app_opens_per_day']
ORIG_CLASS_CDF_GAP_SOURCE_COLS = ['daily_screen_time_hours', 'weekend_screen_time',
                                    'social_media_hours', 'notifications_per_day', 'app_opens_per_day']
KDE_SOURCE_COLS = ['weekend_screen_time', 'notifications_per_day', 'app_opens_per_day']
ORIG_N_BINS = 20


def row_hash(df, cols):
    parts = []
    for c in cols:
        if c in NUM_ORIG:
            parts.append(df[c].astype(float).round(8).fillna(-999999.0).astype(str))
        else:
            parts.append(df[c].astype(str).fillna('__MISSING__'))
    return pd.Series(list(zip(*parts))).astype(str)


train_hash = set(row_hash(train, RAW_COLS_ORIG))
orig_hash = row_hash(orig, RAW_COLS_ORIG)
orig_clean = orig.loc[~orig_hash.isin(train_hash)].drop_duplicates(subset=RAW_COLS_ORIG).reset_index(drop=True)
orig_y = orig_clean['addicted_label'].values.astype(np.int8)
global_rate = float(orig_y.mean())


def empirical_cdf(values, sorted_ref):
    values = np.asarray(values, dtype=np.float64)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if len(sorted_ref) > 0:
        result[valid] = np.searchsorted(sorted_ref, values[valid], side='right') / len(sorted_ref)
    return result


def make_quantile_edges(values, n_bins):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_bins(values, edges):
    values = np.asarray(values, dtype=np.float64)
    bins = np.full(len(values), -1, dtype=np.int32)
    valid = np.isfinite(values)
    bins[valid] = np.clip(np.searchsorted(edges, values[valid], side='right') - 1, 0, len(edges) - 2)
    return bins


def silverman_bw(values):
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return 0.30
    std = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    iqr_scale = float((q75 - q25) / 1.34)
    scales = [s for s in [std, iqr_scale] if np.isfinite(s) and s > 1e-12]
    scale = min(scales) if scales else 1.0
    bw = 0.9 * scale * (n ** (-1.0 / 5.0))
    return float(np.clip(bw, 0.10, 1.00))


refs = {'cdf': {}, 'class_cdf': {}, 'q50': {}, 'mean': {}, 'kde': {}}
for col in ORIG_CDF_SOURCE_COLS:
    v = orig_clean[col].astype(float).values
    refs['cdf'][col] = np.sort(v[np.isfinite(v)])
for col in ORIG_CLASS_CDF_GAP_SOURCE_COLS:
    v = orig_clean[col].astype(float).values
    refs['class_cdf'][col] = {
        'y0': np.sort(v[(orig_y == 0) & np.isfinite(v)]),
        'y1': np.sort(v[(orig_y == 1) & np.isfinite(v)]),
    }
for col in ORIG_Q50_SOURCE_COLS:
    v = orig_clean[col].astype(float).values
    valid = np.isfinite(v)
    ref = {'all': float(np.median(v[valid])), 'y0': float(np.median(v[valid & (orig_y == 0)]))}
    if col in ORIG_Q50_Y1_SOURCE_COLS:
        ref['y1'] = float(np.median(v[valid & (orig_y == 1)]))
    refs['q50'][col] = ref
for col in ORIG_MEAN_SOURCE_COLS:
    v = orig_clean[col].astype(float).values
    edges = make_quantile_edges(v, ORIG_N_BINS)
    bins = assign_bins(v, edges)
    n_bins = len(edges) - 1
    bin_means = np.full(n_bins, global_rate)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            bin_means[b] = float(orig_y[mask].mean())
    refs['mean'][col] = {'edges': edges, 'bin_means': bin_means}
for col in KDE_SOURCE_COLS:
    v = orig_clean[col].astype(float).values
    valid_v = v[np.isfinite(v)]
    mean, std = float(np.mean(valid_v)), float(np.std(valid_v))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    y0 = ((v[(orig_y == 0) & np.isfinite(v)] - mean) / std).reshape(-1, 1)
    y1 = ((v[(orig_y == 1) & np.isfinite(v)] - mean) / std).reshape(-1, 1)
    kde0 = KernelDensity(kernel='gaussian', bandwidth=silverman_bw(y0)).fit(y0)
    kde1 = KernelDensity(kernel='gaussian', bandwidth=silverman_bw(y1)).fit(y1)
    refs['kde'][col] = {'mean': mean, 'std': std, 'kde0': kde0, 'kde1': kde1}
print(f'ORIG referanslari fit edildi ({time.time()-t0:.0f}s)')


def add_orig_features(df):
    out = {}
    for col in ORIG_CDF_SOURCE_COLS:
        v = df[col].astype(float).values
        out[f'{col}__orig_cdf'] = empirical_cdf(v, refs['cdf'][col])
    for col in ORIG_CLASS_CDF_GAP_SOURCE_COLS:
        v = df[col].astype(float).values
        c0 = empirical_cdf(v, refs['class_cdf'][col]['y0'])
        c1 = empirical_cdf(v, refs['class_cdf'][col]['y1'])
        out[f'{col}__orig_cdf_gap'] = c0 - c1
    for col in ORIG_Q50_SOURCE_COLS:
        v = df[col].astype(float).values
        r = refs['q50'][col]
        out[f'{col}__orig_q50_dist'] = np.abs(v - r['all'])
        out[f'{col}__orig_q50_dist_y0'] = np.abs(v - r['y0'])
        if col in ORIG_Q50_Y1_SOURCE_COLS:
            out[f'{col}__orig_q50_dist_y1'] = np.abs(v - r['y1'])
    for col in ORIG_MEAN_SOURCE_COLS:
        v = df[col].astype(float).values
        r = refs['mean'][col]
        bins = assign_bins(v, r['edges'])
        result = np.full(len(v), global_rate)
        valid = bins >= 0
        result[valid] = r['bin_means'][bins[valid]]
        out[f'{col}__orig_mean'] = result
    for col in KDE_SOURCE_COLS:
        v = df[col].astype(float).values
        r = refs['kde'][col]
        result = np.full(len(v), np.nan)
        valid = np.isfinite(v)
        if valid.any():
            scaled = ((v[valid] - r['mean']) / r['std']).reshape(-1, 1)
            log0 = r['kde0'].score_samples(scaled)
            log1 = r['kde1'].score_samples(scaled)
            result[valid] = np.clip(log1 - log0, -20.0, 20.0)
        out[f'{col}__orig_kde_llr'] = result
    return pd.DataFrame(out, index=df.index)


orig_feat_tr = add_orig_features(train)
orig_feat_te = add_orig_features(test)
print(f'ORIG feature sayisi: {orig_feat_tr.shape[1]}  ({time.time()-t0:.0f}s)')

# ================= Turetilmis oran/fark/decimal-lattice (seed-bagimsiz) =================
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

# ================= Imputasyon (seed=42 sabit, seed-bagimsiz kabul edildi - bkz. modul docstring) =================
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
                             verbosity=-1, random_state=42)
    reg.fit(full.loc[mask_known, preds], full.loc[mask_known, target_col])
    predicted = reg.predict(full[preds])
    imp_full[f'imp_{target_col}'] = np.where(mask_known, full[target_col], predicted)
print(f'Imputation tamam ({time.time()-t0:.0f}s)')
imp_tr = imp_full.iloc[:n_train].reset_index(drop=True)
imp_te = imp_full.iloc[n_train:].reset_index(drop=True)

raw_tr = train[all_cats + extra_cols].copy()
raw_te = test[all_cats + extra_cols].copy()
for c in all_cats + extra_cols:
    for frame in (raw_tr, raw_te):
        frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

y = train['addicted_label'].values
prior = y.mean()

with open(f'{CONFIGS}/best_params_lgbm.json') as f:
    tuned_lgb = json.load(f)
with open(f'{CONFIGS}/best_params_xgb.json') as f:
    tuned_xgb = json.load(f)
with open(f'{CONFIGS}/best_params_cat.json') as f:
    tuned_cat = json.load(f)

all_oof = {}
all_pred = {}

for SEED in SEEDS:
    ts = time.time()
    print(f'\n=== SEED {SEED} ===  toplam gecen: {(ts-t0)/60:.1f} dk')
    te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    enc_tr = pd.DataFrame(index=train.index)
    enc_te = pd.DataFrame(index=test.index)
    freq_tr = pd.DataFrame(index=train.index)
    freq_te = pd.DataFrame(index=test.index)

    for c in all_cats + te_only_cats:
        if c == 'dominant_activity':
            col_tr = train[c].fillna('missing').astype(str).values
            col_te = test[c].fillna('missing').astype(str).values
        else:
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

        combined = np.concatenate([col_tr, col_te])
        counts = pd.Series(combined).value_counts()
        freq_tr[c] = pd.Series(col_tr).map(counts).values
        freq_te[c] = pd.Series(col_te).map(counts).values

    X = pd.concat([raw_tr, enc_tr.add_prefix('te_'), freq_tr.add_prefix('freq_'), imp_tr, orig_feat_tr], axis=1).values
    X_test = pd.concat([raw_te, enc_te.add_prefix('te_'), freq_te.add_prefix('freq_'), imp_te, orig_feat_te], axis=1).values
    print(f'  Toplam ozellik: {X.shape[1]}  ({time.time()-ts:.0f}s)')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_idx = list(skf.split(X, y))

    oof_lgb = np.zeros(len(X)); pred_lgb = np.zeros(len(X_test))
    params_lgb = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
                      random_state=SEED, **tuned_lgb)
    for tr, va in fold_idx:
        m = lgb.LGBMClassifier(**params_lgb)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb[va] = m.predict_proba(X[va])[:, 1]
        pred_lgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
    print(f'  LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.5f} ({time.time()-ts:.0f}s)')
    all_oof[('lgb', SEED)] = oof_lgb
    all_pred[('lgb', SEED)] = pred_lgb

    oof_xgb = np.zeros(len(X)); pred_xgb = np.zeros(len(X_test))
    params_xgb = dict(objective='binary:logistic', eval_metric='auc', n_estimators=5000,
                      tree_method='hist', device='cuda', random_state=SEED, **tuned_xgb)
    for tr, va in fold_idx:
        m = xgb.XGBClassifier(**params_xgb)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof_xgb[va] = m.predict_proba(X[va])[:, 1]
        pred_xgb += m.predict_proba(X_test)[:, 1] / len(fold_idx)
    print(f'  XGBoost OOF AUC: {roc_auc_score(y, oof_xgb):.5f} ({time.time()-ts:.0f}s)')
    all_oof[('xgb', SEED)] = oof_xgb
    all_pred[('xgb', SEED)] = pred_xgb

    oof_cat = np.zeros(len(X)); pred_cat = np.zeros(len(X_test))
    for tr, va in fold_idx:
        m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', iterations=5000,
                               task_type='GPU', devices='0', random_seed=SEED, verbose=0, **tuned_cat)
        m.fit(X[tr], y[tr], eval_set=Pool(X[va], y[va]), early_stopping_rounds=100)
        oof_cat[va] = m.predict_proba(X[va])[:, 1]
        pred_cat += m.predict_proba(X_test)[:, 1] / len(fold_idx)
    print(f'  CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.5f} ({time.time()-ts:.0f}s)')
    all_oof[('cat', SEED)] = oof_cat
    all_pred[('cat', SEED)] = pred_cat

    seed_rank_oof = (rankdata(oof_lgb) + rankdata(oof_xgb) + rankdata(oof_cat)) / 3
    print(f'  Seed {SEED} 3-model blend OOF AUC: {roc_auc_score(y, seed_rank_oof):.5f}  '
          f'(seed suresi: {(time.time()-ts)/60:.1f} dk)')
    np.save(f'{CACHE_DIR}/multiseed_origfeat_partial_{SEED}.npy', seed_rank_oof)

ref_oof = (rankdata(all_oof[('lgb', SEEDS[0])]) + rankdata(all_oof[('xgb', SEEDS[0])]) + rankdata(all_oof[('cat', SEEDS[0])])) / 3
ref_auc = roc_auc_score(y, ref_oof)
print(f'\nTek-seed ({SEEDS[0]}) 3-model blend OOF AUC: {ref_auc:.5f}  (referans: tek-seed ABD+ORIG 0.96889, LB 0.96998)')

avg_oof_lgb = np.mean([all_oof[('lgb', s)] for s in SEEDS], axis=0)
avg_oof_xgb = np.mean([all_oof[('xgb', s)] for s in SEEDS], axis=0)
avg_oof_cat = np.mean([all_oof[('cat', s)] for s in SEEDS], axis=0)
multiseed_oof = (rankdata(avg_oof_lgb) + rankdata(avg_oof_xgb) + rankdata(avg_oof_cat)) / 3
multiseed_auc = roc_auc_score(y, multiseed_oof)
print(f'Cok-seedli (model-bazli ortalama, sonra blend) OOF AUC: {multiseed_auc:.5f}  (delta vs tek-seed={multiseed_auc-ref_auc:+.5f})')

all_ranks_oof = sum(rankdata(all_oof[k]) for k in all_oof) / len(all_oof)
allrank_auc = roc_auc_score(y, all_ranks_oof)
print(f'Tum {len(all_oof)} (model x seed) dogrudan rank-blend OOF AUC: {allrank_auc:.5f}  (delta={allrank_auc-ref_auc:+.5f})')

avg_pred_lgb = np.mean([all_pred[('lgb', s)] for s in SEEDS], axis=0)
avg_pred_xgb = np.mean([all_pred[('xgb', s)] for s in SEEDS], axis=0)
avg_pred_cat = np.mean([all_pred[('cat', s)] for s in SEEDS], axis=0)
multiseed_pred = (rankdata(avg_pred_lgb) + rankdata(avg_pred_xgb) + rankdata(avg_pred_cat)) / 3

os.makedirs(f'{SUB}/2026-08-30', exist_ok=True)
sub = pd.DataFrame({'id': test['id'], 'addicted_label': multiseed_pred / max(multiseed_pred)})
sub_path = f'{SUB}/2026-08-30/lgbm_xgb_cat_ABD_origfeat_multiseed_2026-08-30.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')

np.save(f'{CACHE_DIR}/gbdt_abd_origfeat_multiseed_oof.npy', multiseed_oof)
np.save(f'{CACHE_DIR}/gbdt_abd_origfeat_multiseed_test_pred.npy', multiseed_pred)
print(f'Saved: {CACHE_DIR}/gbdt_abd_origfeat_multiseed_oof.npy, {CACHE_DIR}/gbdt_abd_origfeat_multiseed_test_pred.npy')
print(f'Elapsed: {time.time()-t0:.0f}s')
