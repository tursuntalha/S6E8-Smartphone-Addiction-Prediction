"""Shared, model-agnostic helpers used by both the GBDT and NN feature builders
(src/features.py) and by the ensembling step (src/ensembling.py).

Consolidating these here removes a duplication that existed across the original
per-experiment scripts: the ORIG-CDF reference-fitting code, for instance, used to be
copy-pasted near-identically between the GBDT feature script and the NN data-prep
script. Here it's fit once and reused by both.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity


# ---------------------------------------------------------------------------
# Target / frequency encoding
# ---------------------------------------------------------------------------

def target_encode_oof(col_train, col_test, y, skf=None, n_splits=10, smooth=3.0, seed=42):
    """Out-of-fold smoothed mean-encoding (m-estimate smoothing) of a single column.

    Returns (oof_encoded, test_encoded). The out-of-fold scheme avoids leaking a row's
    own label into its own encoded value. Pass an already-built `skf` (StratifiedKFold)
    when encoding several columns so they all use the exact same fold assignment;
    otherwise one is built from `n_splits`/`seed`.
    """
    from sklearn.model_selection import StratifiedKFold

    prior = y.mean()
    if skf is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(col_train))
    for tr_idx, va_idx in skf.split(col_train, y):
        g = pd.DataFrame({'v': col_train[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        g['enc'] = (g['count'] * g['mean'] + smooth * prior) / (g['count'] + smooth)
        oof[va_idx] = pd.Series(col_train[va_idx]).map(g['enc'].to_dict()).fillna(prior).values

    g_full = pd.DataFrame({'v': col_train, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g_full['enc'] = (g_full['count'] * g_full['mean'] + smooth * prior) / (g_full['count'] + smooth)
    test_enc = pd.Series(col_test).map(g_full['enc'].to_dict()).fillna(prior).values
    return oof, test_enc


def frequency_encode(col_train, col_test):
    """Count of each value across train+test combined."""
    combined = np.concatenate([col_train, col_test])
    counts = pd.Series(combined).value_counts()
    return pd.Series(col_train).map(counts).values, pd.Series(col_test).map(counts).values


# ---------------------------------------------------------------------------
# ORIG-CDF: placing rows against the small real (pre-synthesis) dataset's
# class-conditional distributions. See the "Feature engineering" section of the
# main README for what each feature means.
# ---------------------------------------------------------------------------

def row_hash(df, cols, numeric_cols):
    """String hash of a row over `cols`, used to detect exact train/orig duplicates."""
    parts = []
    for c in cols:
        if c in numeric_cols:
            parts.append(df[c].astype(float).round(8).fillna(-999999.0).astype(str))
        else:
            parts.append(df[c].astype(str).fillna('__MISSING__'))
    return pd.Series(list(zip(*parts))).astype(str)


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


# Which source columns feed each ORIG-CDF sub-feature. Fixed by what was found to help
# during feature selection (see experiments/README.md) rather than applied uniformly to
# every column.
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


def fit_orig_cdf_refs(orig_clean, orig_y):
    """Fit every reference distribution needed by add_orig_features(), once, from the
    small real dataset (after excluding rows that are exact duplicates of training rows
    — see clean_orig_reference() in features.py)."""
    global_rate = float(orig_y.mean())
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

    return refs, global_rate


def add_orig_features(df, refs, global_rate):
    """Apply the fitted ORIG-CDF references (fit_orig_cdf_refs) to a train or test frame."""
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


# ---------------------------------------------------------------------------
# Ensembling
# ---------------------------------------------------------------------------

def rank_average(*preds):
    """Average of rank-transformed predictions, normalized to [0, 1]. Rank-averaging
    the GBDT family (LightGBM/XGBoost/CatBoost) is the blend used throughout this
    project instead of averaging raw probabilities, since it's insensitive to each
    model's calibration."""
    ranks = [rankdata(p) for p in preds]
    return np.mean(ranks, axis=0) / len(preds[0])


def search_blend_weight(oof_a, oof_b, y, coarse_step=0.01, fine_step=0.002, fine_radius=0.02):
    """Two-stage (coarse then fine) search over the rank-blend weight for model A vs.
    model B that maximizes out-of-fold AUC. This is how every blend weight in this
    project was chosen — scanning the OOF AUC surface costs nothing, unlike scanning
    weights on the leaderboard."""
    r_a, r_b = rankdata(oof_a), rankdata(oof_b)

    best_w, best_auc = None, -1.0
    for w in np.arange(0.0, 1.0 + 1e-9, coarse_step):
        auc = roc_auc_score(y, w * r_a + (1 - w) * r_b)
        if auc > best_auc:
            best_w, best_auc = w, auc

    lo, hi = max(0.0, best_w - fine_radius), min(1.0, best_w + fine_radius)
    for w in np.arange(lo, hi + 1e-9, fine_step):
        auc = roc_auc_score(y, w * r_a + (1 - w) * r_b)
        if auc > best_auc:
            best_w, best_auc = w, auc

    return float(best_w), float(best_auc)


def apply_blend_weight(test_a, test_b, weight):
    r_a, r_b = rankdata(test_a), rankdata(test_b)
    blend = weight * r_a + (1 - weight) * r_b
    return blend / blend.max()
