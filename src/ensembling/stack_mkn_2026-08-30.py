"""mkn 3 uyeyi aligned 98'e ekle: exp_a101. Ayrica final tablo + test korrelasyon."""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit
import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, save_submission, CACHE

SMOOTH = 1e-6
train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values.astype(np.uint8)
n = len(y)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))
test_id = pd.read_csv(f'{DATA}/test.csv')['id'].values

import importlib.util
_spec = importlib.util.spec_from_file_location('se', os.path.join(os.path.dirname(__file__), 'stack_expand_2026-08-30.py'))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)

names, O, T, _, _ = load_all()
LO = logit(np.clip(O, SMOOTH, 1 - SMOOTH))
LT = logit(np.clip(T, SMOOTH, 1 - SMOOTH))
for gn, gO, gT in [_se.load_group_dariush(), _se.load_group_rayk(), _se.load_group_najiblends()]:
    LO = np.hstack([LO, gO]); LT = np.hstack([LT, gT])

# mkn 3
for d, tag in [('s6e8-xgb-oof', 'xgb'), ('s6e8-lgb-dart-oof', 'lgb'), ('s6e8-cat-mlp-oof', 'cat')]:
    o = np.load(f'{DATA}/extra_oof/mkn/u/{d}/oof_{tag}_v3.npy')
    t = np.load(f'{DATA}/extra_oof/mkn/u/{d}/test_{tag}_v3.npy')
    LO = np.hstack([LO, logit(np.clip(o, SMOOTH, 1 - SMOOTH))[:, None]])
    LT = np.hstack([LT, logit(np.clip(t, SMOOTH, 1 - SMOOTH))[:, None]])
print(f'[matris] {LO.shape[1]} uye (aligned+paiky-mkn)', flush=True)

t0 = time.time()
oof = np.zeros(n)
for tr, va in folds:
    m = LogisticRegression(max_iter=2000, C=1.0).fit(LO[tr], y[tr])
    oof[va] = m.predict_proba(LO[va])[:, 1]
m = LogisticRegression(max_iter=2000, C=1.0).fit(LO, y)
pred = m.predict_proba(LT)[:, 1]
print(f'exp_a101: {roc_auc_score(y, oof):.5f} ({(time.time()-t0)/60:.1f} dk)', flush=True)
np.save(f'{CACHE}/stacko_exp_a101.npy', oof.astype('float64'))
np.save(f'{CACHE}/stackt_exp_a101.npy', pred.astype('float64'))
save_submission('exp_a101', test_id, pred)
print('DONE', flush=True)