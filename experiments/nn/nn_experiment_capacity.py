"""
NN deneyi ADIM 2: kapasite artirma (d_token 64->128, katman 2->3) + LR warmup (3 epoch,
5e-4'e kadar). Feature seti baseline (nn_model.py) ile AYNI (9 ham surekli+3 kategorik,
plr_only=0) - SADECE mimari/optimizasyon degisti, izole test.
Referans: baseline (d_token=64, 2 katman, lr=1e-3 sabit) val AUC=0.96346.
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
    d_token=128, n_layers=3,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Parametre sayisi: {n_params:,}  (d_token=128, n_layers=3)')

model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=25,
                               base_lr=5e-4, warmup_epochs=3)
print(f'\n[ADIM 2: kapasite+warmup] En iyi val AUC: {best_auc:.5f}  (baseline referans: 0.96346)')
