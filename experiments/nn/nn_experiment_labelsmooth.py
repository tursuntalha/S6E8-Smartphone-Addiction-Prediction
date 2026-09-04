"""
NN deneyi ADIM 3: label smoothing (eps=0.02). Feature seti VE mimari baseline (nn_model.py)
ile AYNI (9 ham surekli+3 kategorik, d_token=64, 2 katman, lr=1e-3) - SADECE loss fonksiyonu
degisti, izole test.
Referans: baseline (label smoothing yok) val AUC=0.96346.
"""
import numpy as np
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, train_model, device

import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR
data = np.load(f'{CACHE_DIR}/prepped.npz')

tr_idx, va_idx = data['tr_idx'], data['va_idx']
y = data['y']
n_plr_only = 0
plr_scaled_tr = np.zeros((len(y), n_plr_only), dtype=np.float32)
plr_missing_tr = np.zeros((len(y), n_plr_only), dtype=np.float32)

train_ds = LookupDataset(
    data['cont_idx_tr'][tr_idx], data['cont_scaled_tr'][tr_idx], data['cont_missing_tr'][tr_idx],
    plr_scaled_tr[tr_idx], plr_missing_tr[tr_idx], data['cat_idx_tr'][tr_idx], y[tr_idx],
)
val_ds = LookupDataset(
    data['cont_idx_tr'][va_idx], data['cont_scaled_tr'][va_idx], data['cont_missing_tr'][va_idx],
    plr_scaled_tr[va_idx], plr_missing_tr[va_idx], data['cat_idx_tr'][va_idx], y[va_idx],
)
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

model = LookupTransformerNet(
    lookup_vocab_sizes=data['cont_vocab_sizes'],
    plr_only_count=n_plr_only,
    cat_vocab_sizes=data['cat_vocab_sizes'],
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Parametre sayisi: {n_params:,}  (label_smoothing=0.02)')

model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=25,
                               label_smoothing=0.02)
print(f'\n[ADIM 3: label smoothing] En iyi val AUC: {best_auc:.5f}  (baseline referans: 0.96346)')
