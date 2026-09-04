"""
Acik madde #1: Lookup-Transformer NN, GENISLETILMIS feature seti (62 PLR-only turetilmis
sutun, nn_data_prep_kfold_featfull_2026-08-29.py) + missingness-augmentation (mask_prob=0.40,
2026-08-21'de dogrulanmis kalici teknik) ile 5-fold egitim. Mimari/epoch/patience/mask_prob
PRODUCTION missingaug script'iyle (nn_train_kfold_missingaug_2026-08-21.py) AYNI - tek
degisken feature-set genisligi.

Cikti:
  nn_cache/missingaug_featfull_model_fold{K}.pt
  nn_cache/nn_missingaug_featfull_kfold_oof.npy, nn_cache/nn_missingaug_featfull_kfold_test_pred.npy
"""
import os
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
MASK_PROB = 0.40

np.random.seed(SEED)
torch.manual_seed(SEED)
print('device:', device)

if __name__ == '__main__':
    t_start = time.time()
    data = np.load(f'{CACHE_DIR}/prepped_kfold_featfull.npz')
    fold_id = data['fold_id']
    y = data['y']
    n_train = len(y)
    n_test = len(data['test_id'])
    print(f'PLR-only feature sayisi: {data["plr_scaled_tr"].shape[1]}')

    partial_path = f'{CACHE_DIR}/nn_missingaug_featfull_kfold_oof_partial.npy'
    oof = np.load(partial_path) if os.path.exists(partial_path) else np.zeros(n_train, dtype=np.float64)
    test_pred_sum = np.zeros(n_test, dtype=np.float64)

    fold_times = []
    for fold in range(N_FOLDS):
        t_fold = time.time()
        tr_idx = np.where(fold_id != fold)[0]
        va_idx = np.where(fold_id == fold)[0]

        test_ds = LookupDataset(
            data['cont_idx_te'], data['cont_scaled_te'], data['cont_missing_te'],
            data['plr_scaled_te'], data['plr_missing_te'], data['cat_idx_te'],
        )
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

        ckpt_path = f'{CACHE_DIR}/missingaug_featfull_model_fold{fold}.pt'
        if os.path.exists(ckpt_path):
            print(f'\n===== Fold {fold+1}/{N_FOLDS}: checkpoint bulundu, atlaniyor '
                  f'(sadece test tahmini icin yukleniyor) =====')
            model = LookupTransformerNet(
                lookup_vocab_sizes=data['cont_vocab_sizes'],
                plr_only_count=data['plr_scaled_tr'].shape[1],
                cat_vocab_sizes=data['cat_vocab_sizes'],
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            fold_val_auc = roc_auc_score(y[va_idx], oof[va_idx])
            print(f'--- Fold {fold+1} (onceden egitilmis): val AUC={fold_val_auc:.5f} ---')
            test_pred_sum += predict_probabilities(model, test_loader)
            fold_times.append(0.0)
            continue

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
        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

        model = LookupTransformerNet(
            lookup_vocab_sizes=data['cont_vocab_sizes'],
            plr_only_count=data['plr_scaled_tr'].shape[1],
            cat_vocab_sizes=data['cat_vocab_sizes'],
        ).to(device)

        model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=40,
                                       patience=8, missing_aug_prob=MASK_PROB, verbose=True)

        oof[va_idx] = predict_probabilities(model, val_loader)
        test_pred_sum += predict_probabilities(model, test_loader)

        torch.save(model.state_dict(), ckpt_path)
        np.save(partial_path, oof)

        dt = time.time() - t_fold
        fold_times.append(dt)
        avg_fold_time = sum(fold_times) / max(1, len([t for t in fold_times if t > 0]))
        remaining = avg_fold_time * (N_FOLDS - fold - 1)
        print(f'--- Fold {fold+1} bitti: val AUC={best_auc:.5f}  sure={dt/60:.1f} dk  '
              f'ortalama fold suresi={avg_fold_time/60:.1f} dk  '
              f'tahmini kalan sure={remaining/60:.1f} dk ---')

    test_pred = test_pred_sum / N_FOLDS
    oof_auc = roc_auc_score(y, oof)
    total_min = (time.time() - t_start) / 60
    print(f'\n===== TAMAMLANDI: 5-fold FEATFULL MISSINGAUG NN OOF AUC = {oof_auc:.5f}  '
          f'(eski (dar feature) missingaug NN referans: 0.96717)  toplam sure={total_min:.1f} dk =====')

    np.save(f'{CACHE_DIR}/nn_missingaug_featfull_kfold_oof.npy', oof)
    np.save(f'{CACHE_DIR}/nn_missingaug_featfull_kfold_test_pred.npy', test_pred)
    print(f'Saved: {CACHE_DIR}/nn_missingaug_featfull_kfold_oof.npy, '
          f'{CACHE_DIR}/nn_missingaug_featfull_kfold_test_pred.npy')
