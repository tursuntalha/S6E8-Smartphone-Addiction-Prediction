"""
Acik madde #3: ResNet-tarzi tabular NN (attention'siz, residual MLP bloklari) icin veri
hazirlik. GBDT'nin PRODUCTION feature muhendisligini (K1+A+B+D+29 ORIG-CDF, 113 ozellik,
lgbm_orig_features_lb_2026-08-21.py'den BIREBIR kopyalandi - ayni TE/freq/impute/ORIG kodu)
aynen kullaniyoruz: amac GBDT'nin GORDUGU AYNI bilgiye, farkli bir mimariyle (attention'siz
residual-MLP, Lookup-Transformer'in tersine kategorik/surekli ayrimi yapmadan tek bir
StandardScaler'li vektor olarak) erismek - gercek cesitlilik mimari seviyesinde aranacak
(2026-08-21 gun sonu bulgusu: feature-seviyesi denemeler tukendi, cesitlilik icin
GBDT'nin gormedigi bilgi kanali degil, farkli bir cozum yolu gerekiyor).

X_orig zaten tamamen sayisal (raw -999 sentinel + te_* + freq_* + imp_* + orig_*_cdf/...),
kategorik embedding'e gerek yok - ResNet dogrudan StandardScaler'li vektoru alacak.

Fold ataması GBDT/NN ile BIREBIR AYNI: StratifiedKFold(n_splits=5, shuffle=True,
random_state=42) - sklearn split() sadece y'nin shape'ine bakiyor, X'ten bagimsiz, bu yuzden
GBDT'nin X_orig'i ile burasi ayni satirlarda ayni fold'a dusuyor.

Cikti: nn_cache/resnet_prepped.npz (X_tr, X_te, y, fold_id, test_id, feature_names)
"""
import pandas as pd
import numpy as np
import json
import os
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, NN_CACHE as CACHE_DIR
SEED = 42
SMOOTH = 3.0
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')
orig = pd.read_csv(f'{DATA}/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')
n_train = len(train)

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
print(f'ORIG: {len(orig)} -> temizlik sonrasi {len(orig_clean)} satir')

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

print(f'Referanslar fit edildi ({time.time()-t0:.0f}s)')


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
ORIG_FEATURE_NAMES = list(orig_feat_tr.columns)
print(f'ORIG feature sayisi: {len(ORIG_FEATURE_NAMES)}  ({time.time()-t0:.0f}s)')

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
                             verbosity=-1, random_state=SEED)
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

X_orig = pd.concat([raw_tr, enc_tr.add_prefix('te_'), freq_tr.add_prefix('freq_'), imp_tr, orig_feat_tr], axis=1)
X_test_orig = pd.concat([raw_te, enc_te.add_prefix('te_'), freq_te.add_prefix('freq_'), imp_te, orig_feat_te], axis=1)
feature_names = list(X_orig.columns)
print(f'Toplam ozellik sayisi: {X_orig.shape[1]}  ({time.time()-t0:.0f}s)  (GBDT referansi: 113)')

# ---- ResNet icin: eksik NaN kalabilir (orig_feat KDE/cdf disi-domain'de NaN dondurebilir) - medyanla doldur, sonra StandardScaler ----
medians = X_orig.median()
X_orig_filled = X_orig.fillna(medians)
X_test_filled = X_test_orig.fillna(medians)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_orig_filled).astype(np.float32)
X_te_scaled = scaler.transform(X_test_filled).astype(np.float32)

# ---- GBDT/NN ile BIREBIR AYNI fold atamasi (split() sadece y'ye bakiyor, X-bagimsiz) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_id = np.full(n_train, -1, dtype=np.int64)
for fold, (_, va_idx) in enumerate(skf.split(np.zeros(n_train), y)):
    fold_id[va_idx] = fold
assert (fold_id >= 0).all()
print('Fold dagilimi:', np.bincount(fold_id))

os.makedirs(CACHE_DIR, exist_ok=True)
np.savez(
    f'{CACHE_DIR}/resnet_prepped.npz',
    X_tr=X_tr_scaled, X_te=X_te_scaled,
    y=y.astype(np.float32), fold_id=fold_id,
    test_id=test['id'].values,
)
with open(f'{CACHE_DIR}/resnet_feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print(f'Saved: {CACHE_DIR}/resnet_prepped.npz  (X_tr shape={X_tr_scaled.shape})')
print(f'Elapsed: {time.time()-t0:.0f}s')
