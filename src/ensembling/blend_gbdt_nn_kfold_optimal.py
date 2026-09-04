"""
GBDT (K1+A+B+D, 5-fold OOF) + NN (Lookup-Transformer, ayni 5-fold OOF) - artik ikisi de
GERCEK k-fold OOF, ayni StratifiedKFold(n_splits=5, seed=42) uzerinde, tum 691370 satirda
hizali. Onceki ad-hoc yontem (NN'in sadece %10'luk val-split'i ile agirlik taramasi) yerine
tam OOF uzerinde rank-blend agirligini tara -> cok daha guvenilir tek-parametre kalibrasyonu
(74-model OOF-library notebook'undaki C-grid mantigina benzer, sadece 1 boyutlu).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR, SUB

y = np.load(f'{CACHE_DIR}/prepped_kfold.npz')['y']
gbdt_oof = np.load(f'{CACHE_DIR}/gbdt_abd_oof.npy')
nn_oof = np.load(f'{CACHE_DIR}/nn_kfold_oof.npy')
gbdt_test = np.load(f'{CACHE_DIR}/gbdt_abd_test_pred.npy')
nn_test = np.load(f'{CACHE_DIR}/nn_kfold_test_pred.npy')
test_id = np.load(f'{CACHE_DIR}/prepped_kfold.npz')['test_id']

print(f'GBDT OOF AUC: {roc_auc_score(y, gbdt_oof):.5f}')
print(f'NN   OOF AUC: {roc_auc_score(y, nn_oof):.5f}')
print(f'Spearman(GBDT,NN) OOF corr: {np.corrcoef(rankdata(gbdt_oof), rankdata(nn_oof))[0,1]:.4f}')

r_gbdt = rankdata(gbdt_oof)
r_nn = rankdata(nn_oof)

best_w, best_auc = None, -1
for w in np.arange(0.0, 1.001, 0.01):
    blend = w * r_gbdt + (1 - w) * r_nn
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f'\nEn iyi agirlik (tam 691K OOF taramasi): W_GBDT={best_w:.2f}  OOF AUC={best_auc:.5f}')

# Iyi bulunan agirligin etrafinda ince tarama
fine_best_w, fine_best_auc = best_w, best_auc
for w in np.arange(max(0, best_w - 0.02), min(1, best_w + 0.02) + 0.001, 0.002):
    blend = w * r_gbdt + (1 - w) * r_nn
    auc = roc_auc_score(y, blend)
    if auc > fine_best_auc:
        fine_best_auc, fine_best_w = auc, w
print(f'Ince tarama:                            W_GBDT={fine_best_w:.3f}  OOF AUC={fine_best_auc:.5f}')

W_GBDT = fine_best_w
r_gbdt_test = rankdata(gbdt_test)
r_nn_test = rankdata(nn_test)
blend_test_rank = W_GBDT * r_gbdt_test + (1 - W_GBDT) * r_nn_test

sub = pd.DataFrame({'id': test_id, 'addicted_label': blend_test_rank / blend_test_rank.max()})
sub_path = f'{SUB}/2026-08-20/blend_gbdt_nn_kfold_w{int(round(W_GBDT*100))}_2026-08-20.csv'
import os
os.makedirs(f'{SUB}/2026-08-20', exist_ok=True)
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')

print(f'\nKarsilastirma: eski ad-hoc 90/10-split yontemi W_GBDT=0.85 -> LB=0.96996')
print(f'Yeni tam-OOF taramasi: W_GBDT={W_GBDT:.3f} -> OOF AUC={fine_best_auc:.5f}')
