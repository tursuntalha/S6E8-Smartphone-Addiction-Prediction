"""
Yapay sinir agi pipeline'i - BOLUM 2b: K-FOLD Modelleme.
Ayni Lookup-Transformer mimarisini (nn_model.py'nin uretim konfigurasyonu, nn_common.py)
GBDT ile ayni 5 fold'da (nn_data_prep_kfold.py'nin urettigi fold_id) egitir. Her fold:
train edilir, validation fold'u icin OOF tahmini uretilir, test tahmini biriktirilir
(5 fold-modelinin ortalamasi). CHECKPOINT: her fold sonunda model + o ana kadarki kismi
OOF diske kaydedilir (kesinti olursa is kaybolmaz), sure/kalan-sure tahmini yazdirilir.

Cikti:
  nn_cache/kfold_model_fold{K}.pt      -> her fold'un en iyi checkpoint'i
  nn_cache/nn_kfold_oof.npy            -> tam 691370 satirlik NN-OOF (GBDT-OOF ile hizali)
  nn_cache/nn_kfold_test_pred.npy      -> 5 fold-modelinin ortalama test tahmini
  nn_cache/nn_kfold_oof_partial.npy    -> ara checkpoint (calisirken guncellenir)
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from nn_common import LookupTransformerNet, LookupDataset, train_model, predict_probabilities, device

SEED = 42
import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR
N_FOLDS = 5

np.random.seed(SEED)
torch.manual_seed(SEED)
print('device:', device)

if __name__ == '__main__':
    t_start = time.time()
    data = np.load(f'{CACHE_DIR}/prepped_kfold.npz')
    fold_id = data['fold_id']
    y = data['y']
    n_train = len(y)
    n_test = len(data['test_id'])

    oof = np.zeros(n_train, dtype=np.float64)
    test_pred_sum = np.zeros(n_test, dtype=np.float64)

    fold_times = []
    for fold in range(N_FOLDS):
        t_fold = time.time()
        tr_idx = np.where(fold_id != fold)[0]
        va_idx = np.where(fold_id == fold)[0]
        print(f'\n===== Fold {fold+1}/{N_FOLDS}  (train={len(tr_idx)}  val={len(va_idx)}) '
              f'=====  toplam gecen: {(t_fold - t_start)/60:.1f} dk')

        train_ds = LookupDataset(
            data['cont_idx_tr'][tr_idx], data['cont_scaled_tr'][tr_idx], data['cont_missing_tr'][tr_idx],
            data['plr_scaled_tr'][tr_idx], data['plr_missing_tr'][tr_idx], data['cat_idx_tr'][tr_idx], y[tr_idx],
        )
        val_ds = LookupDataset(
            data['cont_idx_tr'][va_idx], data['cont_scaled_tr'][va_idx], data['cont_missing_tr'][va_idx],
            data['plr_scaled_tr'][va_idx], data['plr_missing_tr'][va_idx], data['cat_idx_tr'][va_idx], y[va_idx],
        )
        test_ds = LookupDataset(
            data['cont_idx_te'], data['cont_scaled_te'], data['cont_missing_te'],
            data['plr_scaled_te'], data['plr_missing_te'], data['cat_idx_te'],
        )
        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

        model = LookupTransformerNet(
            lookup_vocab_sizes=data['cont_vocab_sizes'],
            plr_only_count=data['plr_scaled_tr'].shape[1],
            cat_vocab_sizes=data['cat_vocab_sizes'],
        ).to(device)

        model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=25,
                                       label_smoothing=0.0, verbose=True)

        oof[va_idx] = predict_probabilities(model, val_loader)
        test_pred_sum += predict_probabilities(model, test_loader)

        torch.save(model.state_dict(), f'{CACHE_DIR}/kfold_model_fold{fold}.pt')
        # Ara checkpoint: kesinti olursa o ana kadarki fold'lar kaybolmaz.
        np.save(f'{CACHE_DIR}/nn_kfold_oof_partial.npy', oof)

        dt = time.time() - t_fold
        fold_times.append(dt)
        avg_fold_time = sum(fold_times) / len(fold_times)
        remaining = avg_fold_time * (N_FOLDS - fold - 1)
        print(f'--- Fold {fold+1} bitti: val AUC={best_auc:.5f}  sure={dt/60:.1f} dk  '
              f'ortalama fold suresi={avg_fold_time/60:.1f} dk  '
              f'tahmini kalan sure={remaining/60:.1f} dk ---')

    test_pred = test_pred_sum / N_FOLDS
    oof_auc = roc_auc_score(y, oof)
    total_min = (time.time() - t_start) / 60
    print(f'\n===== TAMAMLANDI: 5-fold NN OOF AUC = {oof_auc:.5f}  '
          f'(90/10 tek-split referans: 0.96507)  toplam sure={total_min:.1f} dk =====')

    np.save(f'{CACHE_DIR}/nn_kfold_oof.npy', oof)
    np.save(f'{CACHE_DIR}/nn_kfold_test_pred.npy', test_pred)
    print(f'Saved: {CACHE_DIR}/nn_kfold_oof.npy, {CACHE_DIR}/nn_kfold_test_pred.npy')
