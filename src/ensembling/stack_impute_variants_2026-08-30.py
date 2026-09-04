"""
Kullanici fikri (08-30): bos degerleri sadece LGBM ile degil, XGB ve CatBoost ile de
tamhinleyip 3 imputasyon varyanti + koklu multi-view olustur. Hepsi ayni guclu raw
pipeline urezerinde (multiseed_gbdt_origfeat tarifi, tek seed=42).

Varyantlar:
  variant_lgb   : mevcut imp_lgb (referans, ~0.96889)
  variant_xgb   : XGB-imputed degerler
  variant_cat   : CatBoost-imputed degerler
  variant_multi : 3 imputasyon seti de birden (multi-view / anlasmazlik bilgisi)

Verimlilik: ORIG-CDF, raw/extra, TE/freq encodings, imputasyonlar SEED-BAGIMSIZ ve
paylasimli; sadece 3-model 5-fold egitimi varyant basina tekrar eder.

Cikti:
  nn_cache/imp_var_{lgb,xgb,cat,multi}_oof.npy / _test_pred.npy
  sub/2026-08-30/imp_var_{lgb,xgb,cat,multi}_2026-08-30.csv
"""
import pandas as pd
import numpy as np
import json
import os
import time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, NN_CACHE, NN_CACHE as CACHE_DIR, SUB, CONFIGS
SEED = 42
SMOOTH = 3.0
t0 = time.time()
CX = f'{NN_CACHE}/impvar'
os.makedirs(CX, exist_ok=True)

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')
orig = pd.read_csv(f'{DATA}/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')
n_train = len(train)

# ================= ORIG-CDF (seed-bagimsiz) =================
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
print(f'ORIG referanslari fit edildi ({time.time()-t0:.0f}s)', flush=True)


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


if os.path.exists(f'{CX}/orig_tr.pkl') and os.path.exists(f'{CX}/orig_te.pkl'):
    orig_feat_tr = pd.read_pickle(f'{CX}/orig_tr.pkl')
    orig_feat_te = pd.read_pickle(f'{CX}/orig_te.pkl')
    print(f'ORIG feature cache okundu ({time.time()-t0:.0f}s)', flush=True)
else:
    orig_feat_tr = add_orig_features(train)
    orig_feat_te = add_orig_features(test)
    orig_feat_tr.to_pickle(f'{CX}/orig_tr.pkl')
    orig_feat_te.to_pickle(f'{CX}/orig_te.pkl')
    print(f'ORIG feature sayisi: {orig_feat_tr.shape[1]}  ({time.time()-t0:.0f}s)', flush=True)

# ================= Turetilmis ozellikler (seed-bagimsiz) =================
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

full = pd.concat([train[cont_cols + cat_cols], test[cont_cols + cat_cols]], axis=0, ignore_index=True)
for c in cat_cols:
    full[c] = full[c].astype('category')

# ================= 3 imputer (LGB/XGB/Cat) =================
IMPUTE_TARGETS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                   'work_study_hours', 'sleep_hours', 'notifications_per_day',
                   'app_opens_per_day', 'weekend_screen_time']
imp_sets = {k: pd.DataFrame(index=full.index) for k in ['lgb', 'xgb', 'cat']}
for k, mk_im in imp_sets.items():
    if os.path.exists(f'{CX}/imp_{k}.pkl'):
        mk_im = pd.read_pickle(f'{CX}/imp_{k}.pkl')
        imp_sets[k] = mk_im
        print(f'Imputation {k} cache okundu ({time.time()-t0:.0f}s)', flush=True)
        continue
    for target_col in IMPUTE_TARGETS:
        preds = [c for c in cont_cols + cat_cols if c != target_col]
        mask_known = full[target_col].notna()
        if k == 'lgb':
            reg = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                    verbosity=-1, random_state=42)
            reg.fit(full.loc[mask_known, preds], full.loc[mask_known, target_col])
            predicted = reg.predict(full[preds])
        elif k == 'xgb':
            reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                   subsample=0.8, colsample_bytree=0.8, verbosity=0,
                                   random_state=42, tree_method='hist', device='cuda')
            reg.fit(full.loc[mask_known, preds], full.loc[mask_known, target_col])
            predicted = reg.predict(full[preds])
        else:
            reg = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                                    random_seed=42, verbose=0)
            cat_idx = [i for i, cc in enumerate(preds) if cc in cat_cols]
            dfc = full.loc[mask_known, preds].copy()
            for cc in cat_cols:
                dfc[cc] = dfc[cc].astype(str)
            reg.fit(dfc, full.loc[mask_known, target_col], cat_features=cat_idx)
            dfc2 = full[preds].copy()
            for cc in cat_cols:
                dfc2[cc] = dfc2[cc].astype(str)
            predicted = reg.predict(dfc2)
        mk_im[f'imp_{target_col}'] = np.where(mask_known, full[target_col], predicted)
    mk_im.to_pickle(f'{CX}/imp_{k}.pkl')
    print(f'Imputation {k} tamam ({time.time()-t0:.0f}s)', flush=True)

imp_tr = {k: v.iloc[:n_train].reset_index(drop=True) for k, v in imp_sets.items()}
imp_te = {k: v.iloc[n_train:].reset_index(drop=True) for k, v in imp_sets.items()}

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

# ================= TE + freq (1 kez) =================
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
if os.path.exists(f'{CX}/tefreq_tr.pkl'):
    enc_tr = pd.read_pickle(f'{CX}/tefreq_tr.pkl')
    enc_te = pd.read_pickle(f'{CX}/tefreq_te.pkl')
    freq_tr = pd.read_pickle(f'{CX}/freq_tr.pkl')
    freq_te = pd.read_pickle(f'{CX}/freq_te.pkl')
    print(f'TE/freq cache okundu ({time.time()-t0:.0f}s)', flush=True)
else:
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
        counts = pd.Series(np.concatenate([col_tr, col_te])).value_counts()
        freq_tr[c] = pd.Series(col_tr).map(counts).values
        freq_te[c] = pd.Series(col_te).map(counts).values
    enc_tr.to_pickle(f'{CX}/tefreq_tr.pkl')
    enc_te.to_pickle(f'{CX}/tefreq_te.pkl')
    freq_tr.to_pickle(f'{CX}/freq_tr.pkl')
    freq_te.to_pickle(f'{CX}/freq_te.pkl')
    print(f'TE/freq tamam ({time.time()-t0:.0f}s)', flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def train_variant(name, imp_tr_k, imp_te_k):
    ts = time.time()
    if os.path.exists(f'{CACHE_DIR}/imp_var_{name}_oof.npy'):
        oof = np.load(f'{CACHE_DIR}/imp_var_{name}_oof.npy')
        auc = roc_auc_score(y, oof)
        print(f'  [{name}] cache okundu OOF: {auc:.5f}', flush=True)
        return auc
    X = pd.concat([raw_tr, enc_tr.add_prefix('te_'), freq_tr.add_prefix('freq_'), imp_tr_k, orig_feat_tr], axis=1).values
    X_test = pd.concat([raw_te, enc_te.add_prefix('te_'), freq_te.add_prefix('freq_'), imp_te_k, orig_feat_te], axis=1).values
    print(f'  [{name}] ozellik: {X.shape[1]}', flush=True)
    fold_idx = list(skf.split(X, y))
    oof = np.zeros(len(X)); pred = np.zeros(len(X_test))
    for mdl, fitx in [('lgb', 'lgb'), ('xgb', 'xgb'), ('cat', 'cat')]:
        oof_m = np.zeros(len(X)); pred_m = np.zeros(len(X_test))
        if mdl == 'lgb':
            params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1,
                          random_state=SEED, **tuned_lgb)
            for tr, va in fold_idx:
                m = lgb.LGBMClassifier(**params)
                m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='auc',
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                oof_m[va] = m.predict_proba(X[va])[:, 1]
                pred_m += m.predict_proba(X_test)[:, 1] / len(fold_idx)
        elif mdl == 'xgb':
            params = dict(objective='binary:logistic', eval_metric='auc', n_estimators=5000,
                          tree_method='hist', device='cuda', random_state=SEED, **tuned_xgb)
            for tr, va in fold_idx:
                m = xgb.XGBClassifier(**params)
                m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
                oof_m[va] = m.predict_proba(X[va])[:, 1]
                pred_m += m.predict_proba(X_test)[:, 1] / len(fold_idx)
        else:
            for tr, va in fold_idx:
                m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', iterations=5000,
                                       task_type='GPU', devices='0', random_seed=SEED, verbose=0, **tuned_cat)
                m.fit(X[tr], y[tr], eval_set=Pool(X[va], y[va]), early_stopping_rounds=100)
                oof_m[va] = m.predict_proba(X[va])[:, 1]
                pred_m += m.predict_proba(X_test)[:, 1] / len(fold_idx)
        print(f'    [{name}] {mdl} OOF: {roc_auc_score(y, oof_m):.5f}', flush=True)
        oof += rankdata(oof_m) / 3
        pred += rankdata(pred_m) / 3
    auc = roc_auc_score(y, oof)
    print(f'  [{name}] 3-model blend OOF: {auc:.5f}  ({(time.time()-ts)/60:.1f} dk)', flush=True)
    np.save(f'{CACHE_DIR}/imp_var_{name}_oof.npy', oof)
    np.save(f'{CACHE_DIR}/imp_var_{name}_test_pred.npy', pred)
    sub = pd.DataFrame({'id': test['id'], 'addicted_label': pred / max(pred)})
    path = f'{SUB}/2026-08-30/imp_var_{name}_2026-08-30.csv'
    sub.to_csv(path, index=False)
    print(f'Saved: {path}', flush=True)
    return auc

results = {}
results['lgb'] = train_variant('lgb', imp_tr['lgb'], imp_te['lgb'])
results['xgb'] = train_variant('xgb', imp_tr['xgb'], imp_te['xgb'])
results['cat'] = train_variant('cat', imp_tr['cat'], imp_te['cat'])
multi_tr = pd.concat([imp_tr['lgb'].add_prefix('l_'), imp_tr['xgb'].add_prefix('x_'), imp_tr['cat'].add_prefix('c_')], axis=1)
multi_te = pd.concat([imp_te['lgb'].add_prefix('l_'), imp_te['xgb'].add_prefix('x_'), imp_te['cat'].add_prefix('c_')], axis=1)
results['multi'] = train_variant('multi', multi_tr, multi_te)

# korelasyon karsilastirmasi
ref_oof = np.load(f'{CACHE_DIR}/gbdt_abd_origfeat_oof.npy')
nn_oof = np.load(f'{CACHE_DIR}/nn_missingaug_featfull_kfold_oof.npy')
print('\nSonuclar / korelasyon:')
print(f'  referans gbdt_abd_origfeat OOF: {roc_auc_score(y, ref_oof):.5f}')
for k, auc in results.items():
    o = np.load(f'{CACHE_DIR}/imp_var_{k}_oof.npy')
    print(f'  imp_var_{k}: OOF={auc:.5f}  corr(ref_origfeat)={np.corrcoef(o, ref_oof)[0,1]:.4f}  '
          f'corr(nn_featfull)={np.corrcoef(o, nn_oof)[0,1]:.4f}')
print(f'Elapsed: {(time.time()-t0)/60:.1f} dk')