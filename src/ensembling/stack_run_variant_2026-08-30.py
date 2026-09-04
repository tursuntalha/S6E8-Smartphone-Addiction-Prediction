"""
Tek varyant stack calistirici.
Kullanim: python scripts/stack_run_variant_2026-08-30.py <variant>
Her varyant: durust 5-fold CV ile OOF AUC, test tahmini, submission CSV.
Checkpoint: nn_cache/stack_checkpoint.json  (her varyant bitince guncellenir)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, get_transforms, save_submission, CACHE

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, NN_CACHE as CACHE, SUB
VARIANT = sys.argv[1]
CKPT = f'{CACHE}/stack_checkpoint.json'

names, O, T, y, n = load_all()
ntest = T.shape[0]
LO, LT, RO, RT, ZO, ZT = get_transforms(names, O, T, n, ntest)
test_id = pd.read_csv(f'{DATA}/test.csv')['id'].values
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))


def honest_clf(fitter, X, XT):
    oof = np.zeros(n)
    for tr, va in folds:
        mm = fitter(X[tr], y[tr])
        oof[va] = mm.predict_proba(X[va])[:, 1]
    mm = fitter(X, y)
    return oof, np.clip(mm.predict_proba(XT)[:, 1], 1e-8, 1 - 1e-8)


def honest_reg(fitter, X, XT):
    oof = np.zeros(n)
    for tr, va in folds:
        mm = fitter(X[tr], y[tr])
        oof[va] = np.clip(mm.predict(X[va]), 1e-8, 1 - 1e-8)
    mm = fitter(X, y)
    return oof, np.clip(mm.predict(XT), 1e-8, 1 - 1e-8)


# ================== varyantlar ==================
def v_logit_lr():
    return honest_clf(lambda Xtr, ytr: LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr), LO, LT)


def v_logit_lr_C03():
    return honest_clf(lambda Xtr, ytr: LogisticRegression(max_iter=2000, C=0.3).fit(Xtr, ytr), LO, LT)


def v_elasticnet():
    return honest_clf(lambda Xtr, ytr: LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                                          C=2.0, max_iter=1200, tol=1e-3).fit(Xtr, ytr), LO, LT)


def v_logit_lr_l1():
    return honest_clf(lambda Xtr, ytr: LogisticRegression(penalty='l1', solver='saga', C=2.0,
                                                          max_iter=1200, tol=1e-3).fit(Xtr, ytr), LO, LT)


def v_ridge():
    return honest_reg(lambda Xtr, ytr: Ridge(alpha=3.0, solver='lsqr').fit(Xtr, ytr), LO, LT)


def v_xgb():
    import xgboost as xgb
    def fit(Xtr, ytr):
        return xgb.XGBClassifier(max_depth=3, n_estimators=900, learning_rate=0.01,
                                 subsample=0.7, colsample_bytree=0.7, tree_method='hist',
                                 device='cuda', objective='binary:logistic',
                                 reg_lambda=5.0, reg_alpha=1.0, random_state=42, verbosity=0).fit(Xtr, ytr)
    return honest_clf(fit, LO, LT)


def v_mlp_z():
    def build(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        m = MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=150, alpha=1e-3,
                          learning_rate_init=2e-3, batch_size=4096, early_stopping=False,
                          n_iter_no_change=20, random_state=42).fit(sc.transform(Xtr), ytr)
        return sc, m
    oof = np.zeros(n)
    for tr, va in folds:
        sc, mm = build(LO[tr], y[tr])
        oof[va] = mm.predict_proba(sc.transform(LO[va]))[:, 1]
    sc, mm = build(LO, y)
    pred = mm.predict_proba(sc.transform(LT))[:, 1]
    return oof, np.clip(pred, 1e-8, 1 - 1e-8)


def v_nnls():
    oof = np.zeros(n)
    XA = np.column_stack([np.ones(n), O])
    XT = np.column_stack([np.ones(ntest), T])
    for tr, va in folds:
        w, _ = nnls(XA[tr], y[tr])
        oof[va] = np.clip(XA[va] @ w, 1e-8, 1 - 1e-8)
    w, _ = nnls(XA, y)
    pred = np.clip(XT @ w, 1e-8, 1 - 1e-8)
    return oof, pred


def v_rank_lr():
    return honest_clf(lambda Xtr, ytr: LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr), ZO, ZT)


def v_rf_z():
    def fit(Xtr, ytr):
        return RandomForestClassifier(n_estimators=250, max_depth=4, min_samples_leaf=300,
                                      n_jobs=-1, random_state=42).fit(Xtr, ytr)
    return honest_clf(fit, ZO, ZT)


VARIANTS = {
    'logit_lr': v_logit_lr,
    'logit_lr_C03': v_logit_lr_C03,
    'elasticnet': v_elasticnet,
    'logit_lr_l1': v_logit_lr_l1,
    'ridge': v_ridge,
    'xgb': v_xgb,
    'mlp_z': v_mlp_z,
    'nnls': v_nnls,
    'rank_lr': v_rank_lr,
    'rf_z': v_rf_z,
}

if VARIANT not in VARIANTS:
    print(f'Bilinmeyen varyant: {VARIANT}')
    print('Mevcut:', sorted(VARIANTS))
    sys.exit(1)

t0 = time.time()
print(f'=== Varyant: {VARIANT} ===', flush=True)
oof, pred = VARIANTS[VARIANT]()
auc = roc_auc_score(y, oof)
print(f'{VARIANT} OOF AUC = {auc:.5f}  ({(time.time()-t0)/60:.1f} dk)', flush=True)

np.save(f'{CACHE}/stacko_{VARIANT}.npy', oof.astype('float64'))
np.save(f'{CACHE}/stackt_{VARIANT}.npy', pred.astype('float64'))
save_submission(f'stack_{VARIANT}', test_id, pred)

ck = {}
if os.path.exists(CKPT):
    with open(CKPT) as f:
        ck = json.load(f)
ck[VARIANT] = {'oof_auc': auc, 'oof': f'{CACHE}/stacko_{VARIANT}.npy',
               'test': f'{CACHE}/stackt_{VARIANT}.npy',
               'sub': f'{SUB}/2026-08-30/stack_{VARIANT}_2026-08-30.csv',
               'seconds': int(time.time() - t0)}
with open(CKPT, 'w') as f:
    json.dump(ck, f, indent=2)
print(f'checkpoint guncellendi. Toplam gecen: {(time.time()-t0)/60:.1f} dk')