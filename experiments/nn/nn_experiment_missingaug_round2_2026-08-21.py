"""
NN deneyi: missingness-augmentation, TUR 2 - daha yuksek mask_prob taramasi.
Tur 1 sonucu (0.05/0.10/0.20 hepsi pozitif, monoton artan, 0.20 25-epoch sinirinda
HALA iyileşiyordu, early-stop tetiklenmedi): mask_prob=0.20 -> val AUC=0.96619
(+0.00112, referans 0.96507'ye gore, projenin en buyuk tek NN kazanci simdiye kadar).
Bu turda: daha yuksek mask_prob (0.3/0.4/0.5) + daha uzun epoch butcesi (40) + daha
sabirli early-stop (patience=8) - tepe noktasini bulmak icin.
"""
import numpy as np
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, train_model, device

import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR
MASK_PROBS = [0.30, 0.40, 0.50]
REFERENCE_AUC = 0.96507
ROUND1_BEST = 0.96619  # mask_prob=0.20

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
for mask_prob in MASK_PROBS:
    print(f'\n===== mask_prob={mask_prob} =====')
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = LookupTransformerNet(
        lookup_vocab_sizes=data['cont_vocab_sizes'],
        plr_only_count=data['plr_scaled_tr'].shape[1],
        cat_vocab_sizes=data['cat_vocab_sizes'],
    ).to(device)

    model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=40,
                                   patience=8, missing_aug_prob=mask_prob)
    results[mask_prob] = best_auc
    print(f'[mask_prob={mask_prob}] En iyi val AUC: {best_auc:.5f}  (referans: {REFERENCE_AUC:.5f}, '
          f'delta={best_auc - REFERENCE_AUC:+.5f})')

print('\n===== OZET (tur 2) =====')
print(f'Referans (mask_prob=0): {REFERENCE_AUC:.5f}')
print(f'Tur 1 en iyisi (mask_prob=0.20, 25 epoch/patience=5): {ROUND1_BEST:.5f}')
for mp, auc in results.items():
    print(f'mask_prob={mp}: {auc:.5f}  (delta vs referans={auc - REFERENCE_AUC:+.5f})')
