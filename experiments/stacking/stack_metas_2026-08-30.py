"""
Meta-of-metas + C sweep. Amac: exp_xgb (0.97003) uzerine LR ile birlestirilecek
en iyi 2. seviye. Ayrica 98 matriste LR icin C taramasi.
"""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, save_submission, CACHE

SMOOTH = 1e-6
train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values.astype(np.uint8)
n = len(y)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))


def load(q):
    return np.load(f'{CACHE}/stacko_{q}.npy'), np.load(f'{CACHE}/stackt_{q}.npy')


def lr_honest(X, XT):
    oof = np.zeros(n)
    for tr, va in folds:
        m = LogisticRegression(max_iter=1500, C=1.0).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    m = LogisticRegression(max_iter=1500, C=1.0).fit(X, y)
    return oof, m.predict_proba(XT)[:, 1]


# --- 98 matrisi tekrar kur ---
import importlib.util
_spec = importlib.util.spec_from_file_location('se', os.path.join(os.path.dirname(__file__), 'stack_expand_2026-08-30.py'))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
names, O, T, _, _ = load_all()
LO = logit(np.clip(O, SMOOTH, 1 - SMOOTH))
LT = logit(np.clip(T, SMOOTH, 1 - SMOOTH))
for gn, gO, gT in [_se.load_group_dariush(), _se.load_group_rayk(), _se.load_group_najiblends()]:
    LO = np.hstack([LO, gO]); LT = np.hstack([LT, gT])
for Cv in [0.3, 3.0, 10.0]:
    m = LogisticRegression(max_iter=1500, C=Cv)
    oof = np.zeros(n)
    for tr, va in folds:
        mm = m.fit(LO[tr], y[tr]); oof[va] = mm.predict_proba(LO[va])[:, 1]
    print(f'LR C={Cv} (98 matris): {roc_auc_score(y, oof):.5f}', flush=True)

# --- exp_cat ve exp_ax OOF'larini kaydet (daha onceden npy yazilmamisti) ---
import catboost as cb
t0 = time.time()
oof = np.zeros(n)
mdl = cb.CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=5,
                            l2_leaf_reg=5.0, random_seed=42, eval_metric='AUC',
                            od_type='Iter', od_wait=100, verbose=0, task_type='CPU', loss_function='Logloss')
for tr, va in folds:
    mdl.fit(LO[tr], y[tr]); oof[va] = mdl.predict_proba(LO[va])[:, 1]
mdl.fit(LO, y)
pred = mdl.predict_proba(LT)[:, 1]
np.save(f'{CACHE}/stacko_exp_cat.npy', oof.astype('float64'))
np.save(f'{CACHE}/stackt_exp_cat.npy', pred.astype('float64'))
print(f'exp_cat: {roc_auc_score(y, oof):.5f} ({(time.time()-t0)/60:.1f} dk)', flush=True)

o_al, t_al = load('exp_aligned'); o_xg, t_xg = load('exp_xgb')
oof_ax = expit(np.mean([logit(np.clip(o_al, 1e-6, 1 - 1e-6)), logit(np.clip(o_xg, 1e-6, 1 - 1e-6))], axis=0))
pred_ax = expit(np.mean([logit(np.clip(t_al, 1e-6, 1 - 1e-6)), logit(np.clip(t_xg, 1e-6, 1 - 1e-6))], axis=0))
np.save(f'{CACHE}/stacko_exp_ax.npy', oof_ax.astype('float64'))
np.save(f'{CACHE}/stackt_exp_ax.npy', pred_ax.astype('float64'))
print(f'exp_ax: {roc_auc_score(y, oof_ax):.5f}', flush=True)

# --- meta-of-metas ---
mets = ['exp_aligned', 'exp_xgb', 'exp_cat', 'exp_all']
X = np.column_stack([logit(np.clip(load(q)[0], SMOOTH, 1 - SMOOTH)) for q in mets])
XT = np.column_stack([logit(np.clip(load(q)[1], SMOOTH, 1 - SMOOTH)) for q in mets])
oof, pred = lr_honest(X, XT)
print(f'meta-LR of {mets}: {roc_auc_score(y, oof):.5f}', flush=True)

# includes exp_ax oof too
mets2 = ['exp_aligned', 'exp_xgb', 'exp_cat', 'exp_all', 'exp_ax']
X2 = np.column_stack([logit(np.clip(load(q)[0], SMOOTH, 1 - SMOOTH)) for q in mets2])
XT2 = np.column_stack([logit(np.clip(load(q)[1], SMOOTH, 1 - SMOOTH)) for q in mets2])
oof2, pred2 = lr_honest(X2, XT2)
print(f'meta-LR of {mets2}: {roc_auc_score(y, oof2):.5f}', flush=True)

test_id = pd.read_csv(f'{DATA}/test.csv')['id'].values
for nm, pp in [('exp_meta5', pred), ('exp_meta6', pred2)]:
    save_submission(nm, test_id, pp)
print('DONE', flush=True)