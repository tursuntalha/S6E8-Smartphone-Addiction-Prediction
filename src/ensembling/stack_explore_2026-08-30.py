"""
Gece oturumu: OOF kütüphanesi (74 model) + bizim 2 OOF'umuz ile stacking altyapisi.
1) Her seyi yukle, hizalamayi dogrula (README quick-start OOF 0.96943 reproduce).
2) Bizim gbdt_abd_origfeat + nn_missingaug_featfull OOF'larini ekle -> tam blend degisimi.
3) Korelasyon matrisi -> en cok cesitlilik getiren uyeleri raporla.
"""
import numpy as np
import pandas as pd
import os, glob, time
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

t0 = time.time()
import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, NN_CACHE as CACHE
L = f'{DATA}/oof_library/oof'

# ---- yukle etiketler ----
train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values
print(f'train yuklendi: {len(y)} satir, prior={y.mean():.4f}')

# ---- kutuphane OOF/test ----
names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(f'{L}/oof_*.npy'))
print(f'kutuphane model sayisi: {len(names)}')
O = np.column_stack([np.load(f'{L}/oof_{n}.npy') for n in names]).astype('float64')
T = np.column_stack([np.load(f'{L}/test_{n}.npy') for n in names]).astype('float64')
print(f'[{(time.time()-t0):.0f}s] kutuphane O={O.shape} T={T.shape} yuklendi')

# ---- bizim OOF'lar ----
ours_oof_names = ['gbdt_abd_origfeat', 'nn_missingaug_featfull']
ours_oof = np.column_stack([
    np.load(f'{CACHE}/gbdt_abd_origfeat_oof.npy'),
    np.load(f'{CACHE}/nn_missingaug_featfull_kfold_oof.npy'),
]).astype('float64')
ours_test = np.column_stack([
    np.load(f'{CACHE}/gbdt_abd_origfeat_test_pred.npy'),
    np.load(f'{CACHE}/nn_missingaug_featfull_kfold_test_pred.npy'),
]).astype('float64')
print(f'[{(time.time()-t0):.0f}s] bizim OOF={ours_oof.shape} Test={ours_test.shape}')

assert O.shape[0] == len(y) == ours_oof.shape[0], 'OOF satir sayisi uyusmuyor!'
assert T.shape[0] == ours_test.shape[0], 'test satir sayisi uyusmuyor!'

# ---- solo AUC'ler ----
solos = {n: roc_auc_score(y, O[:, i]) for i, n in enumerate(names)}
for on, o in zip(ours_oof_names, [ours_oof[:, 0], ours_oof[:, 1]]):
    solos[on] = roc_auc_score(y, o)
top = sorted(solos.items(), key=lambda kv: -kv[1])[:15]
print('\n--- Top-15 solo OOF AUC ---')
for n, a in top:
    print(f'  {n:28s} {a:.5f}')

# ---- README quick-start reproduce (74 member logit LR) ----
OL = logit(np.clip(O, 1e-6, 1 - 1e-6))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for itr, iva in skf.split(OL, y):
    oof[iva] = LogisticRegression(max_iter=3000).fit(OL[itr], y[itr]).predict_proba(OL[iva])[:, 1]
print(f'\n[{(time.time()-t0):.0f}s] README 74-model blend OOF AUC: {roc_auc_score(y, oof):.5f}  (beklenen 0.96943)')

# ---- bizim 2 OOF eklenince ----
OL2 = logit(np.clip(np.hstack([O, ours_oof]), 1e-6, 1 - 1e-6))
oof2 = np.zeros(len(y))
for itr, iva in skf.split(OL2, y):
    oof2[iva] = LogisticRegression(max_iter=3000).fit(OL2[itr], y[itr]).predict_proba(OL2[iva])[:, 1]
a2 = roc_auc_score(y, oof2)
print(f'[{(time.time()-t0):.0f}s] 76-model (74+kutuphane+ours) blend OOF AUC: {a2:.5f}  (delta={a2-0.96943:+.5f})')

# ---- korelasyon: her modelin geri kalaniyla max/min korelasyonu ----
Rn = rankdata(O, axis=1)
corr = np.corrcoef(Rn.T)
np.save(f'{CACHE}/stack_corr_74.npy', corr)
self = np.eye(len(names))
maxc = np.where(self, 0, corr).max(axis=1)
minc = np.where(self, 0, corr).min(axis=1)
print('\n--- En DUSUK max-korelasyon (en cesitli) ---')
for i in np.argsort(maxc)[:12]:
    print(f'  {names[i]:28s} max_corr={maxc[i]:.4f} solo={solos[names[i]]:.5f}')
print('\n--- En YUKSEK solo -- korelasyon ikilisi ---')
i = int(np.argmax(solos[names[i]] for i in range(len(names))))
print(f'  en iyi solo model: {names[int(np.argmax([solos[n] for n in names]))]}')

print(f'\nToplam sure: {(time.time()-t0)/60:.1f} dk')