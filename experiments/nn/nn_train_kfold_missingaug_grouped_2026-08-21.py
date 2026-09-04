"""
Yapay sinir agi - K-FOLD, GRUP-BAZLI (kaskad) missingness-augmentation ile (mask_prob=0.30).
Hucre-bazli versiyonun (mask_prob=0.40) solo NN kalitesini artirdigini (+0.00141) ama
GBDT blend'ine katkisinin sinirda kaldigini (+0.00014 OOF, +0.00018 LB) bulmustuk - sebep:
GBDT-NN Spearman korelasyonu 0.9806'ya yukselmisti (cesitlilik degil, sadece kalite artmisti).
Hipotez: hucre-bazli maskeleme, turetilmis oran sutunlarini (ratio_social_daily vb.) ham
sutundan BAGIMSIZ maskeliyordu - model ham sutun eksikken turetilmis orandan gercek degeri
"kacak" olarak cikarabiliyordu (gercek eksiklikte boyle olmaz). Grup-bazli versiyon
(augment_missingness_grouped, nn_common.py) turetilmis sutunlarin missing bayragini ham
sutunlardan PLR_DEPENDENCIES kurallarina (any/all) gore KASKAD hesapliyor - kacagi kapatiyor.
Izole tek-split sweep (nn_experiment_missingaug_grouped_2026-08-21.py): en iyi mask_prob=0.30
(val AUC=0.96646, hucre-bazli en iyiye (0.96648) neredeyse esit - solo kalite degismedi,
asil soru bu k-fold + blend testinde: GBDT korelasyonu dustu mu, gercek cesitlilik geldi mi?
Mimari PRODUCTION (nn_model.py, d_token=64, 2 katman) ile AYNI. Epoch/patience: 40/8
(hucre-bazli k-fold ile ayni butce, karsilastirilabilir olsun diye).

Cikti:
  nn_cache/missingaug_grouped_model_fold{K}.pt
  nn_cache/nn_missingaug_grouped_kfold_oof.npy, nn_cache/nn_missingaug_grouped_kfold_test_pred.npy
"""
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from nn_common import LookupTransformerNet, LookupDataset, train_model, predict_probabilities, device

SEED = 42
CACHE_DIR = 'nn_cache'
N_FOLDS = 5
MASK_PROB = 0.30

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

    partial_path = f'{CACHE_DIR}/nn_missingaug_grouped_kfold_oof_partial.npy'
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

        ckpt_path = f'{CACHE_DIR}/missingaug_grouped_model_fold{fold}.pt'
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
                                       patience=8, missing_aug_prob=MASK_PROB,
                                       missing_aug_grouped=True, verbose=True)

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
    print(f'\n===== TAMAMLANDI: 5-fold GRUPLU-MISSINGAUG NN OOF AUC = {oof_auc:.5f}  '
          f'(hucre-bazli missingaug referans: 0.96717)  toplam sure={total_min:.1f} dk =====')

    np.save(f'{CACHE_DIR}/nn_missingaug_grouped_kfold_oof.npy', oof)
    np.save(f'{CACHE_DIR}/nn_missingaug_grouped_kfold_test_pred.npy', test_pred)
    print(f'Saved: {CACHE_DIR}/nn_missingaug_grouped_kfold_oof.npy, '
          f'{CACHE_DIR}/nn_missingaug_grouped_kfold_test_pred.npy')
