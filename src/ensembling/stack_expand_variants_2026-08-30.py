"""
Genisletilmis (98 uyeli hizali) matris uzerinde meta varyantlar:
  exp_l1, exp_elasticnet, exp_xgb, exp_w66 (kendi w66 OOF'unu uye olarak ekle)
Ciktilari submission olarak yazar.
"""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, save_submission, CACHE
import importlib.util
_spec = importlib.util.spec_from_file_location('se', os.path.join(os.path.dirname(__file__), 'stack_expand_2026-08-30.py'))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
load_group_dariush = _se.load_group_dariush
load_group_rayk = _se.load_group_rayk
load_group_najiblends = _se.load_group_najiblends

SMOOTH = 1e-6
V = sys.argv[1]

names, O, T, y, n = load_all()
ntest = T.shape[0]
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))
test_id = pd.read_csv('data/test.csv')['id'].values

LO = logit_ = None
from scipy.special import logit
LO = logit(np.clip(O, SMOOTH, 1 - SMOOTH))
LT = logit(np.clip(T, SMOOTH, 1 - SMOOTH))

for gn, gO, gT in [load_group_dariush(), load_group_rayk(), load_group_najiblends()]:
    LO = np.hstack([LO, gO])
    LT = np.hstack([LT, gT])

extra_name = []
if V == 'exp_w66':
    o66 = np.log((0.66 * 1e-0))  # placeholder
    w_oof = 0.66 * np.load(f'{CACHE}/gbdt_abd_origfeat_oof.npy') + 0.34 * np.load(f'{CACHE}/nn_missingaug_featfull_kfold_oof.npy')
    w_tst = 0.66 * np.load(f'{CACHE}/gbdt_abd_origfeat_test_pred.npy') + 0.34 * np.load(f'{CACHE}/nn_missingaug_featfull_kfold_test_pred.npy')
    LO = np.hstack([LO, logit(np.clip(w_oof, SMOOTH, 1 - SMOOTH))[:, None]])
    LT = np.hstack([LT, logit(np.clip(w_tst, SMOOTH, 1 - SMOOTH))[:, None]])
    extra_name = ['w66']

print(f'[matris] {LO.shape[1]} uye', flush=True)
t0 = time.time()


def honest_lr(fitfun):
    oof = np.zeros(n)
    for tr, va in folds:
        m = fitfun(LO[tr], y[tr])
        oof[va] = m.predict_proba(LO[va])[:, 1]
    m = fitfun(LO, y)
    return oof, m.predict_proba(LT)[:, 1]


if V == 'exp_l1':
    oof, pred = honest_lr(lambda X, yy: LogisticRegression(penalty='l1', solver='saga', C=2.0, max_iter=1500, tol=1e-3).fit(X, yy))
elif V == 'exp_elasticnet':
    oof, pred = honest_lr(lambda X, yy: LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=2.0, max_iter=1500, tol=1e-3).fit(X, yy))
elif V == 'exp_xgb':
    import xgboost as xgb
    def f(X, yy):
        return xgb.XGBClassifier(max_depth=3, n_estimators=700, learning_rate=0.01, subsample=0.7,
                                 colsample_bytree=0.7, tree_method='hist', device='cuda',
                                 reg_lambda=5.0, reg_alpha=1.0, random_state=42, verbosity=0).fit(X, yy)
    oof, pred = honest_lr(f)
elif V == 'exp_w66':
    oof, pred = honest_lr(lambda X, yy: LogisticRegression(max_iter=2000, C=1.0).fit(X, yy))
else:
    print('bilinmiyor'); sys.exit(1)

auc = roc_auc_score(y, oof)
print(f'{V}: OOF AUC={auc:.5f}  ({(time.time()-t0)/60:.1f} dk)', flush=True)
np.save(f'{CACHE}/stacko_{V}.npy', oof.astype('float64'))
np.save(f'{CACHE}/stackt_{V}.npy', pred.astype('float64'))
save_submission(f'stack_{V}', test_id, pred)
print('done', flush=True)