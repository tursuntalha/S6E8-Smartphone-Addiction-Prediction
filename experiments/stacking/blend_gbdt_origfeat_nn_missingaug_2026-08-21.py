"""
GBDT+ORIG-CDF (5-fold OOF, LB=0.96998 solo) + missingaug-NN (mask_prob=0.40, k-fold
OOF=0.96717) - tam OOF uzerinde rank-blend agirligini tara. Referans: eski (ORIG'siz)
GBDT+missingaug-NN blend OOF=0.96909 (W_GBDT=0.718), LB=0.97016 (mevcut production).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import NN_CACHE as CACHE_DIR, SUB

y = np.load(f'{CACHE_DIR}/prepped_kfold.npz')['y']
gbdt_oof = np.load(f'{CACHE_DIR}/gbdt_abd_origfeat_oof.npy')
nn_oof = np.load(f'{CACHE_DIR}/nn_missingaug_kfold_oof.npy')
gbdt_test = np.load(f'{CACHE_DIR}/gbdt_abd_origfeat_test_pred.npy')
nn_test = np.load(f'{CACHE_DIR}/nn_missingaug_kfold_test_pred.npy')
test_id = np.load(f'{CACHE_DIR}/prepped_kfold.npz')['test_id']

print(f'GBDT+ORIG OOF AUC: {roc_auc_score(y, gbdt_oof):.5f}  (LB dogrulandi: 0.96998)')
print(f'NN (missingaug) OOF AUC: {roc_auc_score(y, nn_oof):.5f}')
corr = np.corrcoef(rankdata(gbdt_oof), rankdata(nn_oof))[0, 1]
print(f'Spearman(GBDT+ORIG,NN) OOF corr: {corr:.4f}  (eski GBDT-NN referans: 0.9806)')

r_gbdt = rankdata(gbdt_oof)
r_nn = rankdata(nn_oof)

best_w, best_auc = None, -1
for w in np.arange(0.0, 1.001, 0.01):
    blend = w * r_gbdt + (1 - w) * r_nn
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f'\nEn iyi agirlik (tam 691K OOF taramasi): W_GBDT={best_w:.2f}  OOF AUC={best_auc:.5f}')

fine_best_w, fine_best_auc = best_w, best_auc
for w in np.arange(max(0, best_w - 0.02), min(1, best_w + 0.02) + 0.001, 0.002):
    blend = w * r_gbdt + (1 - w) * r_nn
    auc = roc_auc_score(y, blend)
    if auc > fine_best_auc:
        fine_best_auc, fine_best_w = auc, w
print(f'Ince tarama:                            W_GBDT={fine_best_w:.3f}  OOF AUC={fine_best_auc:.5f}')

print(f'\nKarsilastirma: eski (ORIG-siz) blend OOF=0.96909 (W_GBDT=0.718), LB=0.97016 (production)')
print(f'Yeni (ORIG-eklenmis) blend:        OOF={fine_best_auc:.5f}  (delta={fine_best_auc-0.96909:+.5f})')

W_GBDT = fine_best_w
r_gbdt_test = rankdata(gbdt_test)
r_nn_test = rankdata(nn_test)
blend_test_rank = W_GBDT * r_gbdt_test + (1 - W_GBDT) * r_nn_test

sub = pd.DataFrame({'id': test_id, 'addicted_label': blend_test_rank / blend_test_rank.max()})
os.makedirs(f'{SUB}/2026-08-21', exist_ok=True)
sub_path = f'{SUB}/2026-08-21/blend_gbdt_origfeat_nn_missingaug_w{int(round(W_GBDT*100))}_2026-08-21.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
