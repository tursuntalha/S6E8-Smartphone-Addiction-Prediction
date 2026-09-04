"""
NN deneyi: missingness-augmentation (2026-08-20 gun sonu bulgusunun ilk-iş adayi).
Egitim sirasinda rastgele EK deger maskeleme (zaten eksik olmayan hucreleri de eksik
gibi gostermek) - bir egitim MEKANIZMASI degisikligi, hiperparametre degil. GBDT'nin
NN'e ustunlugunun eksik-veri arttikca buyumesiyle motive edildi (tam veri +0.0025,
4+ eksik +0.0068, 2026-08-20 bulgusu). Mimari/diger hiperparametreler PRODUCTION
(nn_model.py) ile AYNI - SADECE mask_prob degisiyor, izole test, tek 90/10 split.
Referans (mask_prob=0, mevcut production): val AUC=0.96507.
"""
import numpy as np
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, train_model, device

CACHE_DIR = 'nn_cache'
SEEDS_TRIED = [0.05, 0.10, 0.20]
REFERENCE_AUC = 0.96507

data = np.load(f'{CACHE_DIR}/prepped.npz')
tr_idx, va_idx = data['tr_idx'], data['va_idx']
y = data['y']

train_ds = LookupDataset(
    data['cont_idx_tr'][tr_idx], data['cont_scaled_tr'][tr_idx], data['cont_missing_tr'][tr_idx],
    data['plr_scaled_tr'][tr_idx], data['plr_missing_tr'][tr_idx], data['cat_idx_tr'][tr_idx], y[tr_idx],
)
val_ds = LookupDataset(
    data['cont_idx_tr'][va_idx], data['cont_scaled_tr'][va_idx], data['cont_missing_tr'][va_idx],
    data['plr_scaled_tr'][va_idx], data['plr_missing_tr'][va_idx], data['cat_idx_tr'][va_idx], y[va_idx],
)

results = {}
for mask_prob in SEEDS_TRIED:
    print(f'\n===== mask_prob={mask_prob} =====')
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = LookupTransformerNet(
        lookup_vocab_sizes=data['cont_vocab_sizes'],
        plr_only_count=data['plr_scaled_tr'].shape[1],
        cat_vocab_sizes=data['cat_vocab_sizes'],
    ).to(device)

    model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=25,
                                   missing_aug_prob=mask_prob)
    results[mask_prob] = best_auc
    print(f'[mask_prob={mask_prob}] En iyi val AUC: {best_auc:.5f}  (referans: {REFERENCE_AUC:.5f}, '
          f'delta={best_auc - REFERENCE_AUC:+.5f})')

print('\n===== OZET =====')
print(f'Referans (mask_prob=0): {REFERENCE_AUC:.5f}')
for mp, auc in results.items():
    print(f'mask_prob={mp}: {auc:.5f}  (delta={auc - REFERENCE_AUC:+.5f})')
