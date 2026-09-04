"""
Kaggle discussion'daki "meta-feature + polinom etkileşim, tek model" fikrinin
mekanizma testi (Hazmah'in yontemi, broccoli beef'in dedigi gibi aslinda
StackingClassifier(passthrough=True)). Elimizdeki 6 farkli OOF'u (2 farkli
pipeline vintage'i: eski 42-ozellik LGB/XGB/Cat 2026-08-12 + yeni K1 48-ozellik
RF/ExtraTrees/HistGB 2026-08-14) meta-feature olarak birlestirip polinom
etkilesimli tek bir LGB meta-model ile CV alinir. Amac: sadece basit rank-blend
DEGIL, dogrusal-olmayan bir kombinasyonun bu 6 (kismen korele, kismen farkli
vintage/algoritma) tahminden ekstra bilgi cikarip cikaramadigini gormek.
NOT: RF/ET/HistGB icin test tahmini kaydedilmemisti, bu yuzden bu sadece CV
mekanizma testi - submission uretilemez.
"""
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from itertools import combinations
import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, SUB

t0 = time.time()
train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values

names = ['lgb_old42', 'xgb_old42', 'cat_old42', 'rf_k1', 'et_k1', 'histgb_k1']
files = [f'{SUB}/oof_lgb.npy', f'{SUB}/oof_xgb.npy', f'{SUB}/oof_cat.npy',
         f'{SUB}/oof_random_forest_k1_2026-08-14.npy', f'{SUB}/oof_extra_trees_k1_2026-08-14.npy',
         f'{SUB}/oof_hist_gb_sklearn_k1_2026-08-14.npy']
oofs = {n: np.load(f) for n, f in zip(names, files)}

print('Standalone AUC ve korelasyonlar:')
for n in names:
    print(f'  {n:12s}: AUC={roc_auc_score(y, oofs[n]):.5f}')
print()
for a, b in combinations(names, 2):
    c = np.corrcoef(oofs[a], oofs[b])[0, 1]
    print(f'  corr({a}, {b}) = {c:.4f}')

# basit rank-blend (referans)
rank_blend = sum(rankdata(oofs[n]) for n in names) / len(names)
print(f'\n[basit 6-model rank-blend] AUC = {roc_auc_score(y, rank_blend):.5f}')

best_pair_blend = (rankdata(oofs['lgb_old42']) + rankdata(oofs['histgb_k1'])) / 2
print(f'[lgb_old42 + histgb_k1 rank-blend] AUC = {roc_auc_score(y, best_pair_blend):.5f}')

# Meta-feature matrisi: 6 taban + ikili carpim etkilesimleri (polinom derece-2, sadece etkilesim)
base = pd.DataFrame({n: oofs[n] for n in names})
inter = pd.DataFrame(index=base.index)
for a, b in combinations(names, 2):
    inter[f'{a}_x_{b}'] = base[a] * base[b]
    inter[f'{a}_m_{b}'] = base[a] - base[b]
X_meta = pd.concat([base, inter], axis=1)
print(f'\nMeta-feature matrisi: {X_meta.shape[1]} ozellik ({X_meta.shape[1]})')

SEED = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_meta = np.zeros(len(X_meta))
ts = time.time()
for tr, va in skf.split(X_meta, y):
    m = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=2000,
                           learning_rate=0.03, num_leaves=15, min_child_samples=200,
                           verbosity=-1, random_state=SEED)
    m.fit(X_meta.iloc[tr], y[tr], eval_set=[(X_meta.iloc[va], y[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_meta[va] = m.predict_proba(X_meta.iloc[va])[:, 1]
meta_auc = roc_auc_score(y, oof_meta)
print(f'[polinom-etkilesimli meta-model (LGB)] OOF AUC = {meta_auc:.5f}  ({time.time()-ts:.0f}s)')

print(f'\nToplam sure: {time.time()-t0:.0f}s')
