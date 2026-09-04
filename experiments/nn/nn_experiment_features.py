"""
NN deneyi ADIM 1: turetilmis oran/fark feature'lari (PLR-only token) + dominant_activity
eklenince val AUC nasil degisiyor. Mimari/hiperparametreler baseline (nn_model.py) ile
AYNI (d_token=64, 2 katman, lr=1e-3, sabit warmup/label-smoothing yok) - SADECE feature
seti degisti, izole test.
Referans: baseline (9 ham surekli+3 kategorik) val AUC=0.96346.
"""
import numpy as np
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, train_model, device

CACHE_DIR = 'nn_cache'
data = np.load(f'{CACHE_DIR}/prepped_extended.npz')

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
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

model = LookupTransformerNet(
    lookup_vocab_sizes=data['cont_vocab_sizes'],
    plr_only_count=data['plr_scaled_tr'].shape[1],
    cat_vocab_sizes=data['cat_vocab_sizes'],
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Parametre sayisi: {n_params:,}  (plr_only token sayisi: {data["plr_scaled_tr"].shape[1]})')

model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=25)
print(f'\n[ADIM 1: feature genisletme] En iyi val AUC: {best_auc:.5f}  (baseline referans: 0.96346)')
