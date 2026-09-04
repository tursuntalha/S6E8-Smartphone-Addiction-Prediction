"""Feature engineering for both the GBDT ensemble and the neural network.

Two entry points:
  - build_gbdt_feature_matrix(...) -> the ~113-column table LightGBM/XGBoost/CatBoost train on
  - build_nn_arrays(...)           -> the tensors the Lookup-Transformer NN trains on

Both share the same derived ratio/decimal-lattice features and the same ORIG-CDF
reference distributions (src/utils.py) so the two views of the data stay consistent,
even though the GBDT gets them as plain columns (+ target/frequency encoding) and the
NN gets them as a "PLR-only" (periodic-linear, no exact-value lookup) token group.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import DATA, CONFIGS
from src import utils

CONT_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']

RAW_COLS_ORIG = CONT_COLS + CAT_COLS
NUM_ORIG = CONT_COLS

ACT3 = ['social_media_hours', 'gaming_hours', 'work_study_hours']
RESID_COLS = ['daily_screen_time_hours'] + ACT3
DECIMAL_COLS = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                'work_study_hours', 'sleep_hours', 'weekend_screen_time']

IMPUTE_TARGETS = CONT_COLS


def load_raw_data():
    """Reads train.csv, test.csv, and the small real (pre-synthesis) reference dataset
    from DATA. See the main README for where to get these from Kaggle."""
    train = pd.read_csv(f'{DATA}/train.csv')
    test = pd.read_csv(f'{DATA}/test.csv')
    orig = pd.read_csv(f'{DATA}/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')
    return train, test, orig


def clean_orig_reference(train, orig):
    """Drops rows from the real reference dataset that are exact duplicates of a
    training row, so the ORIG-CDF features can't leak a row's own label."""
    train_hash = set(utils.row_hash(train, RAW_COLS_ORIG, NUM_ORIG))
    orig_hash = utils.row_hash(orig, RAW_COLS_ORIG, NUM_ORIG)
    orig_clean = orig.loc[~orig_hash.isin(train_hash)].drop_duplicates(subset=RAW_COLS_ORIG).reset_index(drop=True)
    orig_y = orig_clean['addicted_label'].values.astype(np.int8)
    return orig_clean, orig_y


def add_derived_features(frame):
    """Adds every ratio/difference/gap/dominant-activity/decimal-lattice column to
    `frame` in place. Called once per (train, test) frame with identical logic, so
    there's no risk of the two diverging."""
    frame['sum_components'] = frame[ACT3].sum(axis=1, min_count=1)
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
    clean = frame['daily_screen_time_hours'] - frame[ACT3].sum(axis=1)
    frame['diff_daily_sum_clean'] = np.where(mask4, clean, np.nan)

    mask3 = frame[ACT3].notna().all(axis=1)
    mx, mn = frame[ACT3].max(axis=1), frame[ACT3].min(axis=1)
    frame['max_activity3'] = np.where(mask3, mx, np.nan)
    frame['range_activity3'] = np.where(mask3, mx - mn, np.nan)
    frame['gap_social_to_max'] = np.where(mask3, mx - frame['social_media_hours'], np.nan)
    frame['gap_gaming_to_max'] = np.where(mask3, mx - frame['gaming_hours'], np.nan)
    frame['gap_work_to_max'] = np.where(mask3, mx - frame['work_study_hours'], np.nan)

    dominant = frame[ACT3].idxmax(axis=1).where(mask3, np.nan)
    frame['dominant_activity'] = dominant.map({'social_media_hours': 'social', 'gaming_hours': 'gaming',
                                                'work_study_hours': 'work'})

    for c in DECIMAL_COLS:
        v = frame[c]
        frame[f'frac_{c}'] = v - np.floor(v)
        frame[f'd1_{c}'] = np.floor(v * 10) % 10


RATIO_EXTRA_COLS = [
    'ratio_screen_sleep', 'ratio_work_daily', 'ratio_social_daily', 'ratio_opens_daily',
    'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_gaming_daily', 'ratio_notif_daily',
    'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
    'diff_weekend_daily', 'diff_daily_sum_clean', 'max_activity3', 'range_activity3',
    'gap_social_to_max', 'gap_gaming_to_max', 'gap_work_to_max',
] + [f'frac_{c}' for c in DECIMAL_COLS] + [f'd1_{c}' for c in DECIMAL_COLS]

PLR_ONLY_COLS = [
    'sum_components', 'ratio_weekend_daily', 'ratio_screen_sleep', 'ratio_social_daily',
    'ratio_gaming_daily', 'ratio_work_daily', 'diff_weekend_daily', 'max_activity3',
    'ratio_opens_daily', 'ratio_social_sleep', 'ratio_weekend_sleep', 'ratio_notif_daily',
    'ratio_notif_sleep', 'ratio_opens_sleep', 'ratio_work_sleep', 'ratio_sum_daily',
    'diff_daily_sum_clean', 'range_activity3', 'gap_social_to_max', 'gap_gaming_to_max',
    'gap_work_to_max',
] + [f'frac_{c}' for c in DECIMAL_COLS] + [f'd1_{c}' for c in DECIMAL_COLS]


def impute_with_lightgbm(train, test, predictor_cols, seed=42, n_estimators=300):
    """Model-based imputation used as an *additional* feature (`imp_<col>`), alongside
    the raw (sentinel-filled) value — not as a replacement for it. Each target column is
    predicted from every other column in `predictor_cols` via a small LightGBM regressor
    fit on rows where it's observed."""
    import lightgbm as lgb

    n_train = len(train)
    full = pd.concat([train[predictor_cols], test[predictor_cols]], axis=0, ignore_index=True)
    for c in CAT_COLS:
        full[c] = full[c].astype('category')

    imp_full = pd.DataFrame(index=full.index)
    for target_col in IMPUTE_TARGETS:
        preds = [c for c in predictor_cols if c != target_col]
        mask_known = full[target_col].notna()
        reg = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=31, learning_rate=0.05,
                                 verbosity=-1, random_state=seed)
        reg.fit(full.loc[mask_known, preds], full.loc[mask_known, target_col])
        predicted = reg.predict(full[preds])
        imp_full[f'imp_{target_col}'] = np.where(mask_known, full[target_col], predicted)

    return imp_full.iloc[:n_train].reset_index(drop=True), imp_full.iloc[n_train:].reset_index(drop=True)


def build_gbdt_feature_matrix(train, test, refs, global_rate, seed=42, te_smooth=3.0, impute_n_estimators=300):
    """Builds the full GBDT feature table: raw values (NaN -> -999 sentinel) + derived
    ratio/decimal-lattice features + out-of-fold target encoding + frequency encoding +
    LightGBM-imputed values + ORIG-CDF features.

    Returns (X_train, X_test, feature_names). `train`/`test` are mutated in place by
    add_derived_features (adds the ratio/decimal-lattice columns).
    """
    add_derived_features(train)
    add_derived_features(test)

    orig_feat_tr = utils.add_orig_features(train, refs, global_rate)
    orig_feat_te = utils.add_orig_features(test, refs, global_rate)

    cont_cols_gbdt = CONT_COLS + ['sum_components', 'ratio_weekend_daily']
    all_cats = cont_cols_gbdt + CAT_COLS
    te_only_cats = ['dominant_activity']

    raw_tr = train[all_cats + RATIO_EXTRA_COLS].copy()
    raw_te = test[all_cats + RATIO_EXTRA_COLS].copy()
    for c in all_cats + RATIO_EXTRA_COLS:
        for frame in (raw_tr, raw_te):
            frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

    y = train['addicted_label'].values
    te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    enc_tr, enc_te = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    freq_tr, freq_te = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    for c in all_cats + te_only_cats:
        if c == 'dominant_activity':
            col_tr = train[c].fillna('missing').astype(str).values
            col_te = test[c].fillna('missing').astype(str).values
        else:
            col_tr = train[c].astype(str).values
            col_te = test[c].astype(str).values
        # Shared `te_skf` so every column's OOF encoding uses the same fold assignment.
        enc_tr[c], enc_te[c] = utils.target_encode_oof(col_tr, col_te, y, skf=te_skf, smooth=te_smooth)
        freq_tr[c], freq_te[c] = utils.frequency_encode(col_tr, col_te)

    imp_tr, imp_te = impute_with_lightgbm(train, test, all_cats, seed=seed, n_estimators=impute_n_estimators)

    X_train = pd.concat([raw_tr, enc_tr.add_prefix('te_'), freq_tr.add_prefix('freq_'), imp_tr, orig_feat_tr], axis=1)
    X_test = pd.concat([raw_te, enc_te.add_prefix('te_'), freq_te.add_prefix('freq_'), imp_te, orig_feat_te], axis=1)
    return X_train, X_test, list(X_train.columns)


def build_nn_arrays(train, test, refs, global_rate, seed=42, n_splits=5):
    """Builds the tensors the Lookup-Transformer NN trains on (src/model_nn.py):
      - the 9 raw continuous columns as exact-value lookup index + PLR-scaled value
      - every derived ratio/decimal-lattice/ORIG-CDF column as a PLR-only token
        (too high-cardinality for an exact-value lookup to be meaningful)
      - the 3 raw categoricals + `dominant_activity` as lookup-embedding tokens

    `train`/`test` must already have add_derived_features applied (build_gbdt_feature_matrix
    does this as a side effect if called first; call add_derived_features directly
    otherwise).
    """
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

    orig_feat_tr = utils.add_orig_features(train, refs, global_rate)
    orig_feat_te = utils.add_orig_features(test, refs, global_rate)

    plr_tr_df = pd.concat([train[PLR_ONLY_COLS], orig_feat_tr], axis=1)
    plr_te_df = pd.concat([test[PLR_ONLY_COLS], orig_feat_te], axis=1)

    plr_missing_tr = plr_tr_df.isna().values.astype(np.float32)
    plr_missing_te = plr_te_df.isna().values.astype(np.float32)
    plr_medians = plr_tr_df.median()
    plr_scaler = StandardScaler()
    plr_scaled_tr = plr_scaler.fit_transform(plr_tr_df.fillna(plr_medians)).astype(np.float32)
    plr_scaled_te = plr_scaler.transform(plr_te_df.fillna(plr_medians)).astype(np.float32)

    cat_cols_ext = CAT_COLS + ['dominant_activity']
    cat_maps = {}
    for c in cat_cols_ext:
        train[c] = train[c].fillna('missing').astype(str)
        test[c] = test[c].fillna('missing').astype(str)
        vocab = sorted(set(train[c]) | set(test[c]))
        cat_maps[c] = {v: i for i, v in enumerate(vocab)}
    cat_idx_tr = np.stack([train[c].map(cat_maps[c]).values for c in cat_cols_ext], axis=1).astype(np.int64)
    cat_idx_te = np.stack([test[c].map(cat_maps[c]).values for c in cat_cols_ext], axis=1).astype(np.int64)
    cat_vocab_sizes = np.array([len(cat_maps[c]) for c in cat_cols_ext], dtype=np.int64)

    y = train['addicted_label'].values.astype(np.float32)

    # Same StratifiedKFold(seed) as build_gbdt_feature_matrix's te_skf/model_gbdt's CV
    # split, so GBDT and NN out-of-fold predictions line up row-for-row for blending.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_id = np.full(len(train), -1, dtype=np.int64)
    for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_id[va_idx] = fold

    return dict(
        cont_idx_tr=cont_idx_tr, cont_idx_te=cont_idx_te,
        cont_scaled_tr=cont_scaled_tr, cont_scaled_te=cont_scaled_te,
        cont_missing_tr=cont_missing_tr, cont_missing_te=cont_missing_te,
        cont_vocab_sizes=cont_vocab_sizes,
        plr_scaled_tr=plr_scaled_tr, plr_scaled_te=plr_scaled_te,
        plr_missing_tr=plr_missing_tr, plr_missing_te=plr_missing_te,
        cat_idx_tr=cat_idx_tr, cat_idx_te=cat_idx_te,
        cat_vocab_sizes=cat_vocab_sizes,
        y=y, fold_id=fold_id, test_id=test['id'].values,
    )


def load_best_params():
    """Loads the Optuna-tuned hyperparameters for each GBDT model (src/model_gbdt.py),
    found by experiments/tuning/ and cached under configs/."""
    with open(f'{CONFIGS}/best_params_lgbm.json') as f:
        tuned_lgb = json.load(f)
    with open(f'{CONFIGS}/best_params_xgb.json') as f:
        tuned_xgb = json.load(f)
    with open(f'{CONFIGS}/best_params_cat.json') as f:
        tuned_cat = json.load(f)
    return tuned_lgb, tuned_xgb, tuned_cat
