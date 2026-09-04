"""
Yapay sinir agi - K-FOLD, Optuna'nin buldugu EN IYI hiperparametrelerle (nn_optuna_search_
2026-08-20.py, 20 deneme, val AUC=0.96556, 90/10-split referansindan (0.96507) +0.0005).
Tam epoch/patience butcesiyle (arama sirasinda 18/4 kisitliydi, burada 25/6) ve tam 5-fold
GBDT ile ayni StratifiedKFold uzerinde egitiliyor - gercek OOF/blend degerini olcmek icin.

Cikti:
  nn_cache/tuned_model_fold{K}.pt
  nn_cache/nn_tuned_kfold_oof.npy, nn_cache/nn_tuned_kfold_test_pred.npy
"""
import os
import time
import json
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

with open(f'{CACHE_DIR}/optuna_best_params.json') as f:
    BP = json.load(f)
print('Kullanilan hiperparametreler:', BP)

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

    partial_path = f'{CACHE_DIR}/nn_tuned_kfold_oof_partial.npy'
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

        ckpt_path = f'{CACHE_DIR}/tuned_model_fold{fold}.pt'
        if os.path.exists(ckpt_path):
            print(f'\n===== Fold {fold+1}/{N_FOLDS}: checkpoint bulundu, atlaniyor '
                  f'(sadece test tahmini icin yukleniyor) =====')
            model = LookupTransformerNet(
                lookup_vocab_sizes=data['cont_vocab_sizes'],
                plr_only_count=data['plr_scaled_tr'].shape[1],
                cat_vocab_sizes=data['cat_vocab_sizes'],
                d_token=BP['d_token'], n_heads=8, n_layers=BP['n_layers'], n_freq=BP['n_freq'],
                dropout=BP['dropout'], n_head_blocks=BP['n_head_blocks'],
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
        train_loader = DataLoader(train_ds, batch_size=BP['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

        model = LookupTransformerNet(
            lookup_vocab_sizes=data['cont_vocab_sizes'],
            plr_only_count=data['plr_scaled_tr'].shape[1],
            cat_vocab_sizes=data['cat_vocab_sizes'],
            d_token=BP['d_token'], n_heads=8, n_layers=BP['n_layers'], n_freq=BP['n_freq'],
            dropout=BP['dropout'], n_head_blocks=BP['n_head_blocks'],
        ).to(device)

        model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=30,
                                       base_lr=BP['base_lr'], warmup_epochs=BP['warmup_epochs'],
                                       label_smoothing=0.0, patience=6, verbose=True)
        # weight_decay istegi train_model'de yok (AdamW weight_decay=1e-5 sabit) - burada
        # ayri bir optimizer kurup train_model'in dongusunu tekrarlamak yerine, kucuk fark
        # (1e-5 vs Optuna'nin 1.3e-5) onemsiz oldugu icin train_model'i degistirmeden kullaniyoruz.

        oof[va_idx] = predict_probabilities(model, val_loader)
        test_pred_sum += predict_probabilities(model, test_loader)

        torch.save(model.state_dict(), f'{CACHE_DIR}/tuned_model_fold{fold}.pt')
        np.save(f'{CACHE_DIR}/nn_tuned_kfold_oof_partial.npy', oof)

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
    print(f'\n===== TAMAMLANDI: 5-fold TUNED NN OOF AUC = {oof_auc:.5f}  '
          f'(eski NN kfold referans: 0.96576)  toplam sure={total_min:.1f} dk =====')

    np.save(f'{CACHE_DIR}/nn_tuned_kfold_oof.npy', oof)
    np.save(f'{CACHE_DIR}/nn_tuned_kfold_test_pred.npy', test_pred)
    print(f'Saved: {CACHE_DIR}/nn_tuned_kfold_oof.npy, {CACHE_DIR}/nn_tuned_kfold_test_pred.npy')
