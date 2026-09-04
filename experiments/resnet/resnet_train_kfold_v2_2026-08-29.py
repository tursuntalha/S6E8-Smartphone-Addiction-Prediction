"""
ResNet-tabular v2: baseline (resnet_train_kfold_2026-08-29.py) solo OOF=0.96452, GBDT'den
(0.96889) 0.00437 uzak - blend'e katkisi yetersizdi (W_GBDT=0.98 optimuma cikti, delta=-0.00025).
Korelasyon Lookup-Transformer'dan DUSUKTU (0.9791 vs 0.9814, mimari-cesitlilik hipotezi
dogru yonde) ama solo kalite acigi bunu yiyor. Bu versiyon: daha buyuk kapasite
(hidden_dim 256->384, n_blocks 5->7), daha uzun patience/epoch, hafif dusuk weight_decay -
amac solo kaliteyi GBDT'ye yaklastirip korelasyon avantajini blend'e yansitmak.

Cikti:
  nn_cache/resnet_v2_model_fold{K}.pt
  nn_cache/resnet_v2_kfold_oof.npy, nn_cache/resnet_v2_kfold_test_pred.npy
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from resnet_model import ResNetTabular, device

SEED = 42
import sys, os
sys.path.insert(0, os.getcwd())
from src.config import NN_CACHE as CACHE_DIR
N_FOLDS = 5
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 2048
BASE_LR = 1e-3
HIDDEN_DIM = 384
N_BLOCKS = 7
DROPOUT = 0.25

np.random.seed(SEED)
torch.manual_seed(SEED)
print('device:', device)

if __name__ == '__main__':
    t_start = time.time()
    data = np.load(f'{CACHE_DIR}/resnet_prepped.npz')
    X_tr_all, X_te = data['X_tr'], data['X_te']
    y = data['y']
    fold_id = data['fold_id']
    n_features = X_tr_all.shape[1]
    n_test = len(data['test_id'])
    print(f'n_features={n_features}  train={len(y)}  test={n_test}')

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred_sum = np.zeros(n_test, dtype=np.float64)

    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(X_te_t), batch_size=8192, shuffle=False)

    fold_times = []
    for fold in range(N_FOLDS):
        t_fold = time.time()
        tr_idx = np.where(fold_id != fold)[0]
        va_idx = np.where(fold_id == fold)[0]
        print(f'\n===== Fold {fold+1}/{N_FOLDS}  (train={len(tr_idx)}  val={len(va_idx)}) '
              f'=====  toplam gecen: {(t_fold - t_start)/60:.1f} dk')

        train_ds = TensorDataset(torch.tensor(X_tr_all[tr_idx], dtype=torch.float32),
                                  torch.tensor(y[tr_idx], dtype=torch.float32))
        val_x = torch.tensor(X_tr_all[va_idx], dtype=torch.float32).to(device)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = ResNetTabular(n_features=n_features, hidden_dim=HIDDEN_DIM,
                               n_blocks=N_BLOCKS, dropout=DROPOUT).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=5e-6)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
        bce = nn.BCEWithLogitsLoss()
        best_auc, best_state, no_improve = 0.0, None, 0

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = bce(model(xb), yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(model(val_x)).cpu().numpy()
            val_auc = roc_auc_score(y[va_idx], val_preds)
            sched.step(val_auc)
            print(f'Epoch {epoch+1}/{EPOCHS}  val AUC: {val_auc:.5f}  lr={opt.param_groups[0]["lr"]:.2e}')

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print('Early stopping.')
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va_idx] = torch.sigmoid(model(val_x)).cpu().numpy()
            fold_test_preds = []
            for (xb,) in test_loader:
                xb = xb.to(device)
                fold_test_preds.append(torch.sigmoid(model(xb)).cpu().numpy())
        test_pred_sum += np.concatenate(fold_test_preds)

        torch.save(model.state_dict(), f'{CACHE_DIR}/resnet_v2_model_fold{fold}.pt')
        np.save(f'{CACHE_DIR}/resnet_v2_kfold_oof_partial.npy', oof)

        dt = time.time() - t_fold
        fold_times.append(dt)
        avg_fold_time = sum(fold_times) / len(fold_times)
        remaining = avg_fold_time * (N_FOLDS - fold - 1)
        print(f'--- Fold {fold+1} bitti: val AUC={best_auc:.5f}  sure={dt/60:.1f} dk  '
              f'tahmini kalan sure={remaining/60:.1f} dk ---')

    test_pred = test_pred_sum / N_FOLDS
    oof_auc = roc_auc_score(y, oof)
    total_min = (time.time() - t_start) / 60
    print(f'\n===== TAMAMLANDI: 5-fold ResNet-tabular v2 OOF AUC = {oof_auc:.5f}  '
          f'(v1 referans: 0.96452)  toplam sure={total_min:.1f} dk =====')

    np.save(f'{CACHE_DIR}/resnet_v2_kfold_oof.npy', oof)
    np.save(f'{CACHE_DIR}/resnet_v2_kfold_test_pred.npy', test_pred)
    print(f'Saved: {CACHE_DIR}/resnet_v2_kfold_oof.npy, {CACHE_DIR}/resnet_v2_kfold_test_pred.npy')
