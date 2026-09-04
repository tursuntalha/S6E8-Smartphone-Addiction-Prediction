"""
Acik madde #3 sonucu: GBDT+ORIG (K1+A+B+D+29 ORIG-CDF, LB=0.96998 solo) + ResNet-tabular
(attention'siz residual-MLP, ayni 113-ozellik seti) - tam OOF uzerinde rank-blend agirligini
tara. Amac: mimari-seviyesi cesitlilik (GBDT agac-ensemble vs. duz feed-forward residual net,
ikisi de AYNI feature'lari goruyor) - eger blend katkisi Lookup-Transformer'dan (Spearman
korelasyonu 0.98+) daha buyukse, hipotez ("mimari farki feature farkindan daha etkili
cesitlilik kaynagi") dogrulanmis olur.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR, SUB

y = np.load(f'{CACHE_DIR}/resnet_prepped.npz')['y']
gbdt_oof = np.load(f'{CACHE_DIR}/gbdt_abd_origfeat_oof.npy')
resnet_oof = np.load(f'{CACHE_DIR}/resnet_kfold_oof.npy')
gbdt_test = np.load(f'{CACHE_DIR}/gbdt_abd_origfeat_test_pred.npy')
resnet_test = np.load(f'{CACHE_DIR}/resnet_kfold_test_pred.npy')
test_id = np.load(f'{CACHE_DIR}/resnet_prepped.npz')['test_id']

print(f'GBDT+ORIG OOF AUC: {roc_auc_score(y, gbdt_oof):.5f}  (LB dogrulandi: 0.96998)')
print(f'ResNet-tabular OOF AUC: {roc_auc_score(y, resnet_oof):.5f}')
corr = np.corrcoef(rankdata(gbdt_oof), rankdata(resnet_oof))[0, 1]
print(f'Spearman(GBDT+ORIG,ResNet) OOF corr: {corr:.4f}  '
      f'(Lookup-Transformer NN referansi: 0.9814 - daha DUSUKse mimari cesitliligi ise yaramis demek)')

r_gbdt = rankdata(gbdt_oof)
r_resnet = rankdata(resnet_oof)

best_w, best_auc = None, -1
for w in np.arange(0.0, 1.001, 0.01):
    blend = w * r_gbdt + (1 - w) * r_resnet
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w
print(f'\nEn iyi agirlik (tam 691K OOF taramasi): W_GBDT={best_w:.2f}  OOF AUC={best_auc:.5f}')

fine_best_w, fine_best_auc = best_w, best_auc
for w in np.arange(max(0, best_w - 0.02), min(1, best_w + 0.02) + 0.001, 0.002):
    blend = w * r_gbdt + (1 - w) * r_resnet
    auc = roc_auc_score(y, blend)
    if auc > fine_best_auc:
        fine_best_auc, fine_best_w = auc, w
print(f'Ince tarama:                            W_GBDT={fine_best_w:.3f}  OOF AUC={fine_best_auc:.5f}')

print(f'\nKarsilastirma: production (GBDT+ORIG+Lookup-Transformer-NN) OOF=0.96914, LB=0.97025')
print(f'Yeni (GBDT+ResNet) blend:                              OOF={fine_best_auc:.5f}  '
      f'(delta={fine_best_auc-0.96914:+.5f})')

W_GBDT = fine_best_w
r_gbdt_test = rankdata(gbdt_test)
r_resnet_test = rankdata(resnet_test)
blend_test_rank = W_GBDT * r_gbdt_test + (1 - W_GBDT) * r_resnet_test

sub = pd.DataFrame({'id': test_id, 'addicted_label': blend_test_rank / blend_test_rank.max()})
os.makedirs(f'{SUB}/2026-08-29', exist_ok=True)
sub_path = f'{SUB}/2026-08-29/blend_gbdt_origfeat_resnet_w{int(round(W_GBDT*100))}_2026-08-29.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
