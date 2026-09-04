"""
GBDT (K1+A+B+D, 5-fold OOF) + GRUPLU-MISSINGAUG NN (mask_prob=0.30, kaskad maskeleme,
nn_train_kfold_missingaug_grouped_2026-08-21.py) - tam OOF uzerinde rank-blend agirligini
tara. Karsilastirma: hucre-bazli missingaug blend OOF=0.96909 (W_GBDT=0.718, Spearman=0.9806,
LB=0.97016). Bu scriptin amaci: gruplu maskeleme GBDT korelasyonunu dusurup gercek cesitlilik
kazandirdi mi, yoksa solo kalite kaybi + ayni/benzer korelasyon mu (net kayip)?
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
nn_oof = np.load(f'{CACHE_DIR}/nn_missingaug_grouped_kfold_oof.npy')
gbdt_test = np.load(f'{CACHE_DIR}/gbdt_abd_test_pred.npy')
nn_test = np.load(f'{CACHE_DIR}/nn_missingaug_grouped_kfold_test_pred.npy')
test_id = np.load(f'{CACHE_DIR}/prepped_kfold.npz')['test_id']

print(f'GBDT OOF AUC: {roc_auc_score(y, gbdt_oof):.5f}')
print(f'NN (gruplu missingaug) OOF AUC: {roc_auc_score(y, nn_oof):.5f}  '
      f'(hucre-bazli referans: 0.96717)')
corr = np.corrcoef(rankdata(gbdt_oof), rankdata(nn_oof))[0, 1]
print(f'Spearman(GBDT,NN) OOF corr: {corr:.4f}  (hucre-bazli referans: 0.9806)')

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

print(f'\nKarsilastirma: hucre-bazli missingaug blend OOF=0.96909 (W_GBDT=0.718), LB=0.97016')
print(f'Gruplu missingaug blend:                 OOF={fine_best_auc:.5f}  '
      f'(delta vs hucre-bazli={fine_best_auc-0.96909:+.5f})')

W_GBDT = fine_best_w
r_gbdt_test = rankdata(gbdt_test)
r_nn_test = rankdata(nn_test)
blend_test_rank = W_GBDT * r_gbdt_test + (1 - W_GBDT) * r_nn_test

sub = pd.DataFrame({'id': test_id, 'addicted_label': blend_test_rank / blend_test_rank.max()})
import os
os.makedirs(f'{SUB}/2026-08-21', exist_ok=True)
sub_path = f'{SUB}/2026-08-21/blend_gbdt_nn_missingaug_grouped_w{int(round(W_GBDT*100))}_2026-08-21.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
