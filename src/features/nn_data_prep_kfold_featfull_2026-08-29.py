"""
Acik madde #1: NN'e GBDT'nin (K1+A+B+D+29 ORIG-CDF, 113 ozellik) daha eksiksiz feature setini
vermek - blend cesitliligini bozmadan (NN'in kendine ozgu lookup+PLR+attention mekanizmasi
korunuyor, 9 ham surekli sutun HALA exact-deger lookup+PLR aliyor, degisen sadece PLR-only
turetilmis sutun sayisi: eski 8 -> yeni 62).

Eklenen PLR-only sutunlar (nn_data_prep_kfold.py'nin 8'ine ek):
  - Kategori A'nin NN'de eksik kalan oranlari (8): ratio_opens_daily, ratio_social_sleep,
    ratio_weekend_sleep, ratio_notif_daily, ratio_notif_sleep, ratio_opens_sleep,
    ratio_work_sleep, ratio_sum_daily
  - Kategori B fark (1): diff_daily_sum_clean
  - gap/range (4): range_activity3, gap_social_to_max, gap_gaming_to_max, gap_work_to_max
  - decimal lattice (12): frac_*/d1_* (6 sutun x 2)
  - ORIG-CDF (29): lgbm_orig_features_lb_2026-08-21.py ile BIREBIR AYNI kod (empirical CDF,
    class-conditional CDF gap, medyan uzakligi, quantile-binned target ortalamasi, KDE LLR)

TE/freq-encoded sutunlar BILEREK eklenmedi: bunlar zaten NN'in exact-deger lookup embedding'i
ile ayni bilgiyi (deger->hedef-orani) baska bir yoldan kodluyor, eklemek GBDT-NN korelasyonunu
daha da arttirabilir (2026-08-20/21 deseni: "NN'i iyilestir != blend'e katki").

Cikti: nn_cache/prepped_kfold_featfull.npz
"""
import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity

SEED = 42
DATA = 'data'
CACHE_DIR = 'nn_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')
orig = pd.read_csv(f'{DATA}/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')
n_train = len(train)
print(f'train: {train.shape}  test: {test.shape}')

CONT_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']

# ---- Ham surekli sutunlar: exact-deger lookup index (0=missing) + PLR icin olcekli deger (DEGISMEDI) ----
cont_missing_tr = train[CONT_COLS].isna().values.astype(np.float32)
cont_missing_te = test[CONT_COLS].isna().values.astype(np.float32)

cont_idx_tr = np.zeros((len(train), len(CONT_COLS)), dtype=np.int64)
cont_idx_te = np.zeros((len(test), len(CONT_COLS)), dtype=np.int64)
cont_vocab_sizes = []
for j, c in enumerate(CONT_COLS):
    combined = pd.concat([train[c], test[c]]).dropna()
    vocab = {v: i + 1 for i, v in enumerate(sorted(combined.unique()))}
    cont_idx_tr[:, j] = train[c].map(vocab).fillna(0).astype(np.int64)
    cont_idx_te[:, j] = test[c].map(vocab).fillna(0).astype(np.int64)
    cont_vocab_sizes.append(len(vocab) + 1)
cont_vocab_sizes = np.array(cont_vocab_sizes, dtype=np.int64)

medians = train[CONT_COLS].median()
scaler = StandardScaler()
cont_scaled_tr = scaler.fit_transform(train[CONT_COLS].fillna(medians)).astype(np.float32)
cont_scaled_te = scaler.transform(test[CONT_COLS].fillna(medians)).astype(np.float32)

# ================= ORIG-CDF feature'lari (lgbm_orig_features_lb_2026-08-21.py ile BIREBIR AYNI) =================
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
ORIG_FEATURE_NAMES = list(orig_feat_tr.columns)
print(f'ORIG feature sayisi: {len(ORIG_FEATURE_NAMES)}  ({time.time()-t0:.0f}s)')

# ================= Turetilmis oran/fark/decimal-lattice feature'lari (eski 8 + yeni) =================
ACT3 = ['social_media_hours', 'gaming_hours', 'work_study_hours']
RESID_COLS = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']
DECIMAL_COLS = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                 'work_study_hours', 'sleep_hours', 'weekend_screen_time']

for frame in (train, test):
    frame['sum_components'] = frame[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame['ratio_weekend_daily'] = frame['weekend_screen_time'] / frame['daily_screen_time_hours'].replace(0, np.nan)
    frame['ratio_screen_sleep'] = frame['daily_screen_time_hours'] / frame['sleep_hours']
    frame['ratio_social_daily'] = frame['social_media_hours'] / frame['daily_screen_time_hours']
    frame['ratio_gaming_daily'] = frame['gaming_hours'] / frame['daily_screen_time_hours']
    frame['ratio_work_daily'] = frame['work_study_hours'] / frame['daily_screen_time_hours']
    frame['diff_weekend_daily'] = frame['weekend_screen_time'] - frame['daily_screen_time_hours']
    mask3 = frame[ACT3].notna().all(axis=1)
    mx = frame[ACT3].max(axis=1)
    mn = frame[ACT3].min(axis=1)
    frame['max_activity3'] = np.where(mask3, mx, np.nan)
    dominant = frame[ACT3].idxmax(axis=1)
    dominant = dominant.where(mask3, np.nan)
    frame['dominant_activity'] = dominant.map({'social_media_hours': 'social', 'gaming_hours': 'gaming',
                                                 'work_study_hours': 'work'})
    # ---- YENI: kategori A eksik oranlar ----
    frame['ratio_opens_daily'] = frame['app_opens_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_social_sleep'] = frame['social_media_hours'] / frame['sleep_hours']
    frame['ratio_weekend_sleep'] = frame['weekend_screen_time'] / frame['sleep_hours']
    frame['ratio_notif_daily'] = frame['notifications_per_day'] / frame['daily_screen_time_hours']
    frame['ratio_notif_sleep'] = frame['notifications_per_day'] / frame['sleep_hours']
    frame['ratio_opens_sleep'] = frame['app_opens_per_day'] / frame['sleep_hours']
    frame['ratio_work_sleep'] = frame['work_study_hours'] / frame['sleep_hours']
    frame['ratio_sum_daily'] = frame['sum_components'] / frame['daily_screen_time_hours']
    # ---- YENI: kategori B fark ----
    mask4 = frame[RESID_COLS].notna().all(axis=1)
    clean = frame['daily_screen_time_hours'] - (frame['social_media_hours'] + frame['gaming_hours'] + frame['work_study_hours'])
    frame['diff_daily_sum_clean'] = np.where(mask4, clean, np.nan)
    # ---- YENI: gap/range ----
    frame['range_activity3'] = np.where(mask3, mx - mn, np.nan)
    frame['gap_social_to_max'] = np.where(mask3, mx - frame['social_media_hours'], np.nan)
    frame['gap_gaming_to_max'] = np.where(mask3, mx - frame['gaming_hours'], np.nan)
    frame['gap_work_to_max'] = np.where(mask3, mx - frame['work_study_hours'], np.nan)
    # ---- YENI: decimal lattice ----
    for c in DECIMAL_COLS:
        v = frame[c]
        frame[f'frac_{c}'] = v - np.floor(v)
        frame[f'd1_{c}'] = np.floor(v * 10) % 10

decimal_feat_cols = [f'frac_{c}' for c in DECIMAL_COLS] + [f'd1_{c}' for c in DECIMAL_COLS]

PLR_ONLY_COLS = (
    ['sum_components', 'ratio_weekend_daily', 'ratio_screen_sleep', 'ratio_social_daily',
     'ratio_gaming_daily', 'ratio_work_daily', 'diff_weekend_daily', 'max_activity3']
    + ['ratio_opens_daily', 'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_notif_daily',
       'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily']
    + ['diff_daily_sum_clean']
    + ['range_activity3', 'gap_social_to_max', 'gap_gaming_to_max', 'gap_work_to_max']
    + decimal_feat_cols
)

plr_tr_df = pd.concat([train[PLR_ONLY_COLS], orig_feat_tr], axis=1)
plr_te_df = pd.concat([test[PLR_ONLY_COLS], orig_feat_te], axis=1)
PLR_ALL_NAMES = list(plr_tr_df.columns)

plr_missing_tr = plr_tr_df.isna().values.astype(np.float32)
plr_missing_te = plr_te_df.isna().values.astype(np.float32)
plr_medians = plr_tr_df.median()
plr_scaler = StandardScaler()
plr_scaled_tr = plr_scaler.fit_transform(plr_tr_df.fillna(plr_medians)).astype(np.float32)
plr_scaled_te = plr_scaler.transform(plr_te_df.fillna(plr_medians)).astype(np.float32)
print(f'PLR-only turetilmis feature sayisi: {len(PLR_ALL_NAMES)}  (eski: 8)')

# ---- Kategorik sutunlar (DEGISMEDI) ----
CAT_COLS_EXT = CAT_COLS + ['dominant_activity']
cat_maps = {}
for c in CAT_COLS_EXT:
    train[c] = train[c].fillna('missing').astype(str)
    test[c] = test[c].fillna('missing').astype(str)
    vocab = sorted(set(train[c]) | set(test[c]))
    cat_maps[c] = {v: i for i, v in enumerate(vocab)}
    print(c, '->', len(vocab), 'kategori')

cat_idx_tr = np.stack([train[c].map(cat_maps[c]).values for c in CAT_COLS_EXT], axis=1).astype(np.int64)
cat_idx_te = np.stack([test[c].map(cat_maps[c]).values for c in CAT_COLS_EXT], axis=1).astype(np.int64)
cat_vocab_sizes = np.array([len(cat_maps[c]) for c in CAT_COLS_EXT], dtype=np.int64)

y = train['addicted_label'].values.astype(np.float32)

# ---- GBDT ile BIREBIR AYNI fold atamasi ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_id = np.full(len(train), -1, dtype=np.int64)
for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    fold_id[va_idx] = fold
assert (fold_id >= 0).all()
print('Fold dagilimi:', np.bincount(fold_id))

np.savez(
    f'{CACHE_DIR}/prepped_kfold_featfull.npz',
    cont_idx_tr=cont_idx_tr, cont_idx_te=cont_idx_te,
    cont_scaled_tr=cont_scaled_tr, cont_scaled_te=cont_scaled_te,
    cont_missing_tr=cont_missing_tr, cont_missing_te=cont_missing_te,
    cont_vocab_sizes=cont_vocab_sizes,
    plr_scaled_tr=plr_scaled_tr, plr_scaled_te=plr_scaled_te,
    plr_missing_tr=plr_missing_tr, plr_missing_te=plr_missing_te,
    cat_idx_tr=cat_idx_tr, cat_idx_te=cat_idx_te,
    cat_vocab_sizes=cat_vocab_sizes,
    y=y, fold_id=fold_id,
    test_id=test['id'].values,
)
print(f'Saved: {CACHE_DIR}/prepped_kfold_featfull.npz')
print(f'Elapsed: {time.time()-t0:.0f}s')
