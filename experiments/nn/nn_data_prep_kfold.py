"""
Yapay sinir agi pipeline'i - BOLUM 1b: Veri Hazirlik (K-FOLD versiyonu).
nn_data_prep.py ile AYNI feature seti (2026-08-19 kazanan konfigurasyon: 9 ham + 8 turetilmis
surekli sutun + 4 kategorik). Tek fark: 90/10 tek split yerine, GBDT ile BIREBIR AYNI
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) fold atamasi kaydediliyor
(bkz. scripts/lgbm_xgb_cat_ABD_combined_2026-08-19.py satir 159) - boylece NN-OOF ve
GBDT-OOF (nn_cache/gbdt_abd_oof.npy) ayni satirlarda, dogrudan karsilastirilabilir/stack
edilebilir olacak.

Cikti: nn_cache/prepped_kfold.npz (fold_id alani disinda prepped.npz ile ayni icerik)
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

SEED = 42
import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, NN_CACHE as CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')
print(f'train: {train.shape}  test: {test.shape}')

CONT_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']

# ---- Ham surekli sutunlar: exact-deger lookup index (0=missing) + PLR icin olcekli deger ----
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

# ---- Turetilmis oran/fark feature'lari (GBDT'de kanitlanmis, PLR-only token) ----
ACT3 = ['social_media_hours', 'gaming_hours', 'work_study_hours']
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
    frame['max_activity3'] = np.where(mask3, mx, np.nan)
    dominant = frame[ACT3].idxmax(axis=1)
    dominant = dominant.where(mask3, np.nan)
    frame['dominant_activity'] = dominant.map({'social_media_hours': 'social', 'gaming_hours': 'gaming',
                                                 'work_study_hours': 'work'})

PLR_ONLY_COLS = ['sum_components', 'ratio_weekend_daily', 'ratio_screen_sleep', 'ratio_social_daily',
                  'ratio_gaming_daily', 'ratio_work_daily', 'diff_weekend_daily', 'max_activity3']

plr_missing_tr = train[PLR_ONLY_COLS].isna().values.astype(np.float32)
plr_missing_te = test[PLR_ONLY_COLS].isna().values.astype(np.float32)
plr_medians = train[PLR_ONLY_COLS].median()
plr_scaler = StandardScaler()
plr_scaled_tr = plr_scaler.fit_transform(train[PLR_ONLY_COLS].fillna(plr_medians)).astype(np.float32)
plr_scaled_te = plr_scaler.transform(test[PLR_ONLY_COLS].fillna(plr_medians)).astype(np.float32)
print(f'PLR-only turetilmis feature sayisi: {len(PLR_ONLY_COLS)}')

# ---- Kategorik sutunlar -> embedding icin tamsayi kod (train+test birlesik kelime dagarcigi) ----
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

# ---- GBDT ile BIREBIR AYNI fold atamasi (ayni n_splits/shuffle/random_state, ayni row sirasi) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_id = np.full(len(train), -1, dtype=np.int64)
for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    fold_id[va_idx] = fold
assert (fold_id >= 0).all()
print('Fold dagilimi:', np.bincount(fold_id))

np.savez(
    f'{CACHE_DIR}/prepped_kfold.npz',
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
print(f'Saved: {CACHE_DIR}/prepped_kfold.npz')
