"""
Yapay sinir agi - K-FOLD, missingness-augmentation ile (mask_prob=0.40).
Isole tek-split deneyi (nn_experiment_missingaug_2026-08-21.py + round2) sonucu:
mask_prob taramasi 0.05/0.10/0.20/0.30/0.40/0.50, plato 0.3-0.5 araliginda, en iyi
mask_prob=0.40 (val AUC=0.96648, referans 0.96507'den +0.00141 - simdiye kadarki en
buyuk tekil NN kazanci, HPO taramasinin +0.0005'inden kat kat buyuk). Mimari PRODUCTION
(nn_model.py, d_token=64, 2 katman) ile AYNI - Optuna'nin tuned parametreleri KULLANILMIYOR
(o yon 2026-08-20'de negatif sonuc verip kapanmisti). Ayni GBDT StratifiedKFold(5, seed=42)
uzerinde egitiliyor - gercek OOF/blend degerini olcmek icin.
Epoch/patience: round2 deneyindeki 40/8 ile ayni (mask_prob>=0.3 icin yakinsama daha
uzun surdugu gorulmustu).

Cikti:
  nn_cache/missingaug_model_fold{K}.pt
  nn_cache/nn_missingaug_kfold_oof.npy, nn_cache/nn_missingaug_kfold_test_pred.npy
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
    data = np.load(f'{CACHE_DIR}/prepped_kfold.npz')
    fold_id = data['fold_id']
    y = data['y']
    n_train = len(y)
    n_test = len(data['test_id'])

    partial_path = f'{CACHE_DIR}/nn_missingaug_kfold_oof_partial.npy'
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

        ckpt_path = f'{CACHE_DIR}/missingaug_model_fold{fold}.pt'
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
    print(f'\n===== TAMAMLANDI: 5-fold MISSINGAUG NN OOF AUC = {oof_auc:.5f}  '
          f'(eski NN kfold referans: 0.96576)  toplam sure={total_min:.1f} dk =====')

    np.save(f'{CACHE_DIR}/nn_missingaug_kfold_oof.npy', oof)
    np.save(f'{CACHE_DIR}/nn_missingaug_kfold_test_pred.npy', test_pred)
    print(f'Saved: {CACHE_DIR}/nn_missingaug_kfold_oof.npy, {CACHE_DIR}/nn_missingaug_kfold_test_pred.npy')
