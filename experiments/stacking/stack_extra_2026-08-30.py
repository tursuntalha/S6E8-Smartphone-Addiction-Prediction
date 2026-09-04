"""
Ek stacking deneyleri (temel LR'nin 0.96969'unu asmayi/cesitlilik hedefliyor):
  A) meta LR: [76 logit + 9 raw sayisal sutun]
  B) ridge-coef oncelikli subset'ler (top-10/20/30/40) -> durust LR
  C) top-solo rank ortalama (k = 6/10)
  D) varyant toplulugu: logit_lr + elasticnet + l1 + xgb + nnls rank-avg
  E) eksiklik-rejimine gore ayri LR meta
Cikti: nn_cache/stack_extra_*.npy + sub/2026-08-30/stack_extra_*.csv + checkpoint
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, SUB

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, get_transforms, save_submission, CACHE

names, O, T, y, n = load_all()
ntest = T.shape[0]
LO, LT, RO, RT, ZO, ZT = get_transforms(names, O, T, n, ntest)
test_id = pd.read_csv(f'{DATA}/test.csv')['id'].values
train = pd.read_csv(f'{DATA}/train.csv')
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))

RAW9 = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        'work_study_hours', 'sleep_hours', 'notifications_per_day',
        'app_opens_per_day', 'weekend_screen_time']
raw_tr = train[RAW9].astype(float).values
raw_te = pd.read_csv(f'{DATA}/test.csv')[RAW9].astype(float).values
imp_tr = np.isnan(raw_tr).sum(axis=1)
imp_te = np.isnan(raw_te).sum(axis=1)
raw_tr = np.nan_to_num(raw_tr, nan=-999.0)
raw_te = np.nan_to_num(raw_te, nan=-999.0)

# standardize raw sutunlar (LR icin)
raw_mean = raw_tr.mean(axis=0); raw_std = raw_tr.std(axis=0) + 1e-9
raw_tr_s = (raw_tr - raw_mean) / raw_std
raw_te_s = (raw_te - raw_mean) / raw_std


def honest_lr(Xa, XT):
    oof = np.zeros(n)
    for tr, va in folds:
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xa[tr], y[tr])
        oof[va] = m.predict_proba(Xa[va])[:, 1]
    m = LogisticRegression(max_iter=2000, C=1.0).fit(Xa, y)
    return oof, np.clip(m.predict_proba(XT)[:, 1], 1e-8, 1 - 1e-8)


def finish(name, oof, pred):
    auc = roc_auc_score(y, oof)
    print(f'{name:36s} OOF AUC = {auc:.5f}  ({time.time()-t0:.0f}s)', flush=True)
    np.save(f'{CACHE}/stacko_{name}.npy', oof.astype('float64'))
    np.save(f'{CACHE}/stackt_{name}.npy', pred.astype('float64'))
    save_submission(f'stack_{name}', test_id, pred)
    with open(f'{CACHE}/stack_checkpoint.json') as f:
        ck = json.load(f)
    ck[name] = {'oof_auc': auc, 'sub': f'{SUB}/2026-08-30/stack_{name}_2026-08-30.csv'}
    with open(f'{CACHE}/stack_checkpoint.json', 'w') as f:
        json.dump(ck, f, indent=2)
    return auc


t0 = time.time()

# ---- A) meta LR + raw ----
XA = np.hstack([LO, raw_tr_s]); XT = np.hstack([LT, raw_te_s])
oof, pred = honest_lr(XA, XT)
finish('extra_lr_raw9', oof, pred)

# ---- A2) + eksiklik sayisi ----
XA = np.hstack([LO, raw_tr_s, imp_tr[:, None]]); XT = np.hstack([LT, raw_te_s, imp_te[:, None]])
oof, pred = honest_lr(XA, XT)
finish('extra_lr_raw9_imp', oof, pred)

# ---- B) ridge-coef oncelikli subset ----
ridge = Ridge(alpha=5.0, solver='lsqr').fit(LO, y)
w = np.abs(ridge.coef_)
order = np.argsort(w)[::-1]
for k in (10, 20, 30, 40):
    sel = order[:k]
    oof, pred = honest_lr(LO[:, sel], LT[:, sel])
    finish(f'extra_lr_subset{k}', oof, pred)

# ---- C) top-solo rank ortalama ----
solos = [roc_auc_score(y, O[:, i]) for i in range(O.shape[1])]
top_idx = np.argsort(solos)[::-1]
for k in (6, 12):
    sel = top_idx[:k]
    oof = np.mean([rankdata(O[:, i]) for i in sel], axis=0)
    pred = np.mean([rankdata(T[:, i]) for i in sel], axis=0)
    finish(f'extra_topk_avg{k}', oof, pred)

# ---- D) varyant toplulugu (5 meta varyant rank-avg) ----
vm = ['logit_lr', 'elasticnet', 'logit_lr_l1', 'xgb', 'nnls']
oofs = [np.load(f'{CACHE}/stacko_{v}.npy') for v in vm]
preds = [np.load(f'{CACHE}/stackt_{v}.npy') for v in vm]
oof = np.mean([rankdata(o) for o in oofs], axis=0)
pred = np.mean([rankdata(p) for p in preds], axis=0)
finish('extra_ens5', oof, pred)

# ---- E) eksiklik-rejimi meta ----
# rejim: 0 / 1 / 2 / 3+ eksik
groups_tr = np.select([imp_tr == 0, imp_tr == 1, imp_tr == 2], [0, 1, 2], default=3)
groups_te = np.select([imp_te == 0, imp_te == 1, imp_te == 2], [0, 1, 2], default=3)
oof = np.zeros(n); pred = np.zeros(ntest)
for g in range(4):
    mtr = groups_tr == g
    mte = groups_te == g
    if mtr.sum() < 2000:
        sub_folds = [(np.ones(mtr.sum(), dtype=bool), np.arange(mtr.sum()))]  # fallback
        print(f'  rejim {g}: az ornek ({mtr.sum()}), esit agirlik blend kullanildi')
        oof[mtr] = np.mean([rankdata(O[mtr, i]) for i in top_idx[:12]], axis=0)
        pred[mte] = np.mean([rankdata(T[mte, i]) for i in top_idx[:12]], axis=0)
        continue
    idx = np.where(mtr)[0]
    mapping = {old: new for new, old in enumerate(idx)}
    r_folds = []
    for tr, va in folds:
        trm = [mapping[i] for i in tr if i in mapping]
        vam = [mapping[i] for i in va if i in mapping]
        if len(trm) > 200 and len(vam) > 100:
            r_folds.append((np.array(trm), np.array(vam)))
    if not r_folds:
        print(f'  rejim {g}: fold ayristi, parametrik meta atlaniyor')
        oof[mtr] = 0.5
        pred[mte] = 0.5
        continue
    for tr, va in r_folds:
        m = LogisticRegression(max_iter=1500, C=1.0).fit(LO[idx][tr], y[idx][tr])
        oof[idx[va]] = m.predict_proba(LO[idx][va])[:, 1]
    m = LogisticRegression(max_iter=1500, C=1.0).fit(LO[idx], y[idx])
    pred[mte] = m.predict_proba(LT[mte])[:, 1]
    print(f'  rejim {g}: n={mtr.sum()}, islendi', flush=True)
finish('extra_regime_lr', oof, pred)

print(f'\nToplam: {(time.time()-t0)/60:.1f} dk')