"""
Son round: meta blendler + ek varyantlar + submission tablosu + checkpoint guncelleme.

Varyantlar:
  exp_l1_fast   : liblinear L1  (98 uye)
  exp_cat       : CatBoost meta (98 uye)
  exp_ax        : logit-ortalama(exp_aligned, exp_xgb)
  exp_axa       : logit-ortalama(exp_aligned, exp_xgb, exp_all)
Her varyant: icten OOF AUC + submission CSV.
Ek: tum aday subs'larin test pred korelasyon matrisi.
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, SUB

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, save_submission, CACHE

SMOOTH = 1e-6
skf = StratifiedKFold(5, shuffle=True, random_state=42)

# ---------- temel: y, folds, yardimcilar, results ----------
train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values.astype(np.uint8)
n = len(y)
folds = list(skf.split(np.zeros(n), y))
results = {}


def add_sub(name, oof, pred):
    test_id = pd.read_csv(f'{DATA}/test.csv')['id'].values
    auc = roc_auc_score(y, oof)
    save_submission(name, test_id, pred)
    print(f'{name}: OOF AUC={auc:.5f}', flush=True)
    return {'oof_auc': auc, 'sub': f'{SUB}/2026-08-30/{name}_2026-08-30.csv'}


def logit_avg(*preds):
    return expit(np.mean([logit(np.clip(p, 1e-6, 1 - 1e-6)) for p in preds], axis=0))


# ---------- 98 uyeli matris ----------
import importlib.util
_spec = importlib.util.spec_from_file_location('se', os.path.join(os.path.dirname(__file__), 'stack_expand_2026-08-30.py'))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)

names, O, T, y0, n0 = load_all()
LO = logit(np.clip(O, SMOOTH, 1 - SMOOTH))
LT = logit(np.clip(T, SMOOTH, 1 - SMOOTH))
for gn, gO, gT in [_se.load_group_dariush(), _se.load_group_rayk(), _se.load_group_najiblends()]:
    LO = np.hstack([LO, gO])
    LT = np.hstack([LT, gT])
print(f'[matris] {LO.shape[1]} uye', flush=True)

# ---------- exp_l1_fast (liblinear L1) ----------
t0 = time.time()
oof = np.zeros(n)
for tr, va in folds:
    m = LogisticRegression(penalty='l1', solver='liblinear', C=2.0, max_iter=500, tol=1e-3).fit(LO[tr], y[tr])
    oof[va] = m.predict_proba(LO[va])[:, 1]
m = LogisticRegression(penalty='l1', solver='liblinear', C=2.0, max_iter=500, tol=1e-3).fit(LO, y)
pred = m.predict_proba(LT)[:, 1]
results['exp_l1_fast'] = add_sub('exp_l1_fast', oof, pred)
print(f'exp_l1_fast done ({(time.time()-t0)/60:.1f} dk)', flush=True)

# ---------- exp_cat (CatBoost meta) ----------
import catboost as cb
t0 = time.time()
oof = np.zeros(n)
mdl = cb.CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=5,
                            l2_leaf_reg=5.0, random_seed=42, eval_metric='AUC',
                            od_type='Iter', od_wait=100, verbose=0, task_type='CPU', loss_function='Logloss')
oof = np.zeros(n)
for tr, va in folds:
    mdl.fit(LO[tr], y[tr])
    oof[va] = mdl.predict_proba(LO[va])[:, 1]
mdl.fit(LO, y)
pred = mdl.predict_proba(LT)[:, 1]
results['exp_cat'] = add_sub('exp_cat', oof, pred)
print(f'exp_cat done ({(time.time()-t0)/60:.1f} dk)', flush=True)

# ---------- kayitli OOF/test ----------
def load_pt(q):
    return np.load(f'{CACHE}/stacko_{q}.npy'), np.load(f'{CACHE}/stackt_{q}.npy')

# 1) exp_aligned + exp_xgb blend (icten, OOF tarafli)
o_al, t_al = load_pt('exp_aligned')
o_xg, t_xg = load_pt('exp_xgb')
oof_ax = logit_avg(o_al, o_xg)
pred_ax = logit_avg(t_al, t_xg)
results['exp_ax'] = add_sub('exp_ax', oof_ax, pred_ax)

# 2) + exp_all eklenmis 3'lü
o_all, t_all = load_pt('exp_all')
oof_axa = logit_avg(o_al, o_xg, o_all)
pred_axa = logit_avg(t_al, t_xg, t_all)
results['exp_axa'] = add_sub('exp_axa', oof_axa, pred_axa)

# 3) exp_aligned + exp_all 2'li
oof_aa = logit_avg(o_al, o_all)
pred_aa = logit_avg(t_al, t_all)
results['exp_aa'] = add_sub('exp_aa', oof_aa, pred_aa)

print(json.dumps(results, indent=2))
print('DONE', flush=True)