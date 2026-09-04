"""
Gece oturumu - Ana stacking framework.
OOF kutuphanesi (74 model) + bizim 2 OOF (gbdt_abd_origfeat, nn_missingaug_featfull) = 76 sütun.

Tum varyantlar DURUST (honest) nested-CV ile degerlendiriliyor:
  her variant icin: skf(5,42) icinde meta-model fold egitim verisinde fit, val'da tahmin.

Cikti:
  - nn_cache/stack_variants_oof.npy   (her variant icin OOF prob)
  - nn_cache/stack_variants_test.npy  (her variant icin test prob)
  - sub/2026-08-30/<variant>.csv      (her variant icin submission)
  - konsol: OOF AUC tablosu
"""
import numpy as np
import pandas as pd
import os, glob, json, time
from scipy.special import logit, expit
from scipy.stats import rankdata
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

t0 = time.time()
CACHE = 'nn_cache'
L = 'data/oof_library/oof'
SMOOTH = 1e-6

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
y = train['addicted_label'].values
test_id = test['id'].values
n = len(y)

names_lib = sorted(os.path.basename(p)[4:-4] for p in glob.glob(f'{L}/oof_*.npy'))
O_lib = np.column_stack([np.load(f'{L}/oof_{nmb}.npy') for nmb in names_lib]).astype('float64')
T_lib = np.column_stack([np.load(f'{L}/test_{nmb}.npy') for nmb in names_lib]).astype('float64')

ours_names = ['gbdt_abd_origfeat', 'nn_missingaug_featfull']
O_ours = np.column_stack([
    np.load(f'{CACHE}/gbdt_abd_origfeat_oof.npy'),
    np.load(f'{CACHE}/nn_missingaug_featfull_kfold_oof.npy'),
]).astype('float64')
T_ours = np.column_stack([
    np.load(f'{CACHE}/gbdt_abd_origfeat_test_pred.npy'),
    np.load(f'{CACHE}/nn_missingaug_featfull_kfold_test_pred.npy'),
]).astype('float64')

names = names_lib + ours_names
O = np.hstack([O_lib, O_ours])
T = np.hstack([T_lib, T_ours])
print(f'[{(time.time()-t0):.0f}s] {O.shape[1]} model yuklendi: O={O.shape} T={T.shape}')

# ---- uzay donusumleri ----
LO = logit(np.clip(O, SMOOTH, 1 - SMOOTH))          # logit uzayi
LT = logit(np.clip(T, SMOOTH, 1 - SMOOTH))
RO = rankdata(O, axis=1) / n                          # [0,1] rank uzayi
RT = rankdata(T, axis=1) / (T.shape[0])
ZO = (RO - RO.mean(axis=1, keepdims=True)) / (RO.std(axis=1, keepdims=True) + 1e-12)
ZT = (RT - RT.mean(axis=1, keepdims=True)) / (RT.std(axis=1, keepdims=True) + 1e-12)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))

# ============ meta-model tanimlari ============
# Her meta: (ad, fit_predict_oof(X, y, folds) -> oof prob, fit_predict_test(X, y, XT) -> test prob)
def make_logit_lr(max_iter=3000, C=1.0, penalty='l2', l1_ratio=None):
    def fit_oof(X, y, folds):
        out = np.zeros(len(y))
        for tr, va in folds:
            m = LogisticRegression(max_iter=max_iter, C=C, penalty=penalty,
                                   l1_ratio=l1_ratio, solver='saga' if penalty in ('l1', 'elasticnet') else 'lbfgs')
            m.fit(X[tr], y[tr])
            out[va] = m.predict_proba(X[va])[:, 1]
        return out
    def fit_test(X, y, XT):
        m = LogisticRegression(max_iter=max_iter, C=C, penalty=penalty,
                               l1_ratio=l1_ratio, solver='saga' if penalty in ('l1', 'elasticnet') else 'lbfgs')
        m.fit(X, y)
        return m.predict_proba(XT)[:, 1]
    return fit_oof, fit_test

def make_ridge(alpha=1.0):
    def fit_oof(X, y, folds):
        out = np.zeros(len(y))
        for tr, va in folds:
            m = Ridge(alpha=alpha, solver='lsqr')
            m.fit(X[tr], y[tr])
            out[va] = np.clip(m.predict(X[va]), 1e-8, 1 - 1e-8)
        return out
    def fit_test(X, y, XT):
        m = Ridge(alpha=alpha, solver='lsqr')
        m.fit(X, y)
        return np.clip(m.predict(XT), 1e-8, 1 - 1e-8)
    return fit_oof, fit_test

def make_xgb_meta(max_depth=3, n_est=800, lr=0.01, subsample=0.7, colsample=0.7):
    import xgboost as xgb
    def fit_oof(X, y, folds):
        out = np.zeros(len(y))
        for tr, va in folds:
            m = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_est, learning_rate=lr,
                                  subsample=subsample, colsample_bytree=colsample,
                                  tree_method='hist', device='cuda',
                                  objective='binary:logistic', eval_metric='auc',
                                  reg_lambda=5.0, reg_alpha=1.0, random_state=42, verbosity=0)
            m.fit(X[tr], y[tr])
            out[va] = m.predict_proba(X[va])[:, 1]
        return out
    def fit_test(X, y, XT):
        m = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_est, learning_rate=lr,
                              subsample=subsample, colsample_bytree=colsample,
                              tree_method='hist', device='cuda',
                              objective='binary:logistic', eval_metric='auc',
                              reg_lambda=5.0, reg_alpha=1.0, random_state=42, verbosity=0)
        m.fit(X, y)
        return m.predict_proba(XT)[:, 1]
    return fit_oof, fit_test

def make_mlp(hidden=(48, 24), max_iter=400):
    def fit_oof(X, y, folds):
        out = np.zeros(len(y))
        for tr, va in folds:
            m = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, alpha=1e-3,
                              learning_rate_init=1e-3, early_stopping=True,
                              n_iter_no_change=20, random_state=42)
            m.fit(X[tr], y[tr])
            out[va] = np.clip(m.predict_proba(X[va])[:, 1], 1e-8, 1 - 1e-8)
        return out
    def fit_test(X, y, XT):
        m = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, alpha=1e-3,
                          learning_rate_init=1e-3, early_stopping=True,
                          n_iter_no_change=20, random_state=42)
        m.fit(X, y)
        return np.clip(m.predict_proba(XT)[:, 1], 1e-8, 1 - 1e-8)
    return fit_oof, fit_test

def make_rf_meta(n_est=500, depth=4, min_samples_leaf=200):
    def fit_oof(X, y, folds):
        out = np.zeros(len(y))
        for tr, va in folds:
            m = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                       min_samples_leaf=min_samples_leaf,
                                       n_jobs=-1, random_state=42)
            m.fit(X[tr], y[tr])
            out[va] = m.predict_proba(X[va])[:, 1]
        return out
    def fit_test(X, y, XT):
        m = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                   min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=42)
        m.fit(X, y)
        return m.predict_proba(XT)[:, 1]
    return fit_oof, fit_test

def make_nnls():
    def fit_oof(X, y, folds):
        out = np.zeros(len(y))
        for tr, va in folds:
            w, _ = nnls(X[tr], y[tr])
            out[va] = np.clip(X[va] @ w, 1e-8, 1 - 1e-8)
        return out
    def fit_test(X, y, XT):
        w, _ = nnls(X, y)
        return np.clip(XT @ w, 1e-8, 1 - 1e-8)
    return fit_oof, fit_test

# ============================================================
variants = {}
results = {}

def evaluate(name, X, X_test, fit_oof, fit_test):
    ts = time.time()
    oof = fit_oof(X, y, folds)
    auc = roc_auc_score(y, oof)
    tpred = fit_test(X, y, X_test)
    variants[name] = {'oof': oof, 'test': tpred}
    results[name] = auc
    print(f'  {name:28s} OOF AUC={auc:.5f}  ({time.time()-ts:.0f}s)')
    return auc

print(f'\n--- Metamodel varyantlari (durust CV) ---')

# 1) Logit-uzayi LR (tam 76) - ana / calibration referansi
evaluate('full_logit_lr', LO, LT, *make_logit_lr())

# 2) Rank-uzayi LR
evaluate('full_rank_lr', ZO, ZT, *make_logit_lr())

# 3) Logit LR + l2 zayif (daha saglam)
evaluate('full_logit_lr_C03', LO, LT, *make_logit_lr(C=0.3))

# 4) Elasticnet meta
evaluate('elasticnet_lr', LO, LT, *make_logit_lr(penalty='elasticnet', l1_ratio=0.3, C=1.0))

# 5) Ridge meta (logit)
evaluate('ridge_logit', LO, LT, *make_ridge(alpha=5.0))

# 6) XGB tree meta (logit)
evaluate('xgb_meta', LO, LT, *make_xgb_meta())

# 7) MLP meta
evaluate('mlp_meta', ZO, ZT, *make_mlp())

# 8) RandomForest meta
evaluate('rf_meta', ZO, ZT, *make_rf_meta())

# 9) NNLS (logit uzayinda) - ama logit negatif degerleri NNLS icin capir; prob uzayinda yap
evaluate('nnls', np.column_stack([np.ones(n), O]), np.column_stack([np.ones(T.shape[0]), T]), *make_nnls())

np.save(f'{CACHE}/stack_variants_oof.npy',
        np.column_stack([v['oof'] for v in variants.values()]).astype('float64'))
np.save(f'{CACHE}/stack_variants_test.npy',
        np.column_stack([v['test'] for v in variants.values()]).astype('float64'))
with open(f'{CACHE}/stack_variants_ref.json', 'w') as f:
    json.dump({'names': list(variants.keys()), 'oof_auc': results}, f, indent=2)

print(f'\n--- SONUC TABLOSU (toplam {(time.time()-t0)/60:.1f} dk) ---')
for name, auc in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f'  {name:28s} OOF AUC={auc:.5f}')

# submission dosyalarini yaz (id sirasi test.csv)
os.makedirs('sub/2026-08-30', exist_ok=True)
for name, v in variants.items():
    p = np.clip(v['test'], SMOOTH, 1 - SMOOTH)
    p = p / p.max()
    out = pd.DataFrame({'id': test_id, 'addicted_label': p})
    out.to_csv(f'sub/2026-08-30/stack_{name}_2026-08-30.csv', index=False)
    print(f'Saved: sub/2026-08-30/stack_{name}_2026-08-30.csv')