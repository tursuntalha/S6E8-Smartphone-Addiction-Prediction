"""
Yapay sinir agi pipeline'i - BOLUM 2: Modelleme (PRODUCTION - kazanan konfigurasyon).
2026-08-19 izole hiperparametre denemelerinin (nn_experiment_*.py) sonucuna gore:
  - feature genisletme (turetilmis oran/fark + dominant_activity): +0.00193 val AUC -> ALINDI
  - kapasite artirma (d_token 128, 3 katman) + LR warmup: -0.00085 -> REDDEDILDI, kullanilmiyor
  - label smoothing (eps=0.02): izolede +0.00019 ama feature genisletmeyle BIRLIKTE test
    edilince -0.00033 (0.96539->0.96506, etkiler toplanmadi) -> REDDEDILDI, kullanilmiyor
Mimari (nn_common.py'den): 9 ham surekli sutun icin exact-deger lookup+PLR, 8 turetilmis
oran/fark sutunu icin sadece PLR, 4 kategorik sutun icin lookup embedding, CLS+21 token ->
TransformerEncoder (d=64, 2 katman, 8 head) -> ResidualBlock'lu head.

`nn_data_prep.py`'nin uretttigi nn_cache/prepped.npz'i okur, egitir, en iyi checkpoint'i
nn_cache/best_model.pt'ye kaydeder.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, train_model, device

SEED = 42
import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR
LABEL_SMOOTHING = 0.0

np.random.seed(SEED)
torch.manual_seed(SEED)
print('device:', device)

if __name__ == '__main__':
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
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = LookupTransformerNet(
        lookup_vocab_sizes=data['cont_vocab_sizes'],
        plr_only_count=data['plr_scaled_tr'].shape[1],
        cat_vocab_sizes=data['cat_vocab_sizes'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parametre sayisi: {n_params:,}')

    model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=25,
                                   label_smoothing=LABEL_SMOOTHING)
    print(f'\nEn iyi val AUC: {best_auc:.5f}')

    torch.save(model.state_dict(), f'{CACHE_DIR}/best_model.pt')
    print(f'Saved: {CACHE_DIR}/best_model.pt')
