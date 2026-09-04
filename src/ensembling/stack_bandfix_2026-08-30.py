"""
Rayk tarzi zayif-band duzeltmesi: exp_xgb global stack uzerine band-local FM karistiricasi.

Her band (3-6h, 6-7.8h) icin ic-durust CV ile bir karistirici LR fit edilir:
  logit(p_mix) = b0 + b1*logit(p_global) + b2*z(band_score)
Band ici AUC: sadece-global vs karisik. Kazanc pozitifse test'e uygula.
Cikti: exp_xgb_band  (band duzeltmeli submission)
Ayrica genel OOF AUC'yu (band icleri karisik + disardakiler global) raporla.
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
from config import DATA

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, save_submission, CACHE

SMOOTH = 1e-6
R = f'{DATA}/extra_oof/rayk/unzipped'

train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values.astype(np.uint8)
n = len(y)
test = pd.read_csv(f'{DATA}/test.csv')
ntest = len(test)
test_id = test['id'].values
assert np.all(train['id'].values == np.arange(n)), 'train id==rowoid'
assert np.all(test_id == 691369 + np.arange(ntest))

g_oof = np.load(f'{CACHE}/stacko_exp_xgb.npy')
g_tst = np.load(f'{CACHE}/stackt_exp_xgb.npy')

# n_missing (train + test): data kolonlarindaki NaN sayisi
feat_cols = [c for c in train.columns if c not in ('id', 'addicted_label')]
nm_tr = train[feat_cols].isna().sum(axis=1).values.astype(int)
nm_te = test[feat_cols].isna().sum(axis=1).values.astype(int)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))
fold_id = np.zeros(n, int)
for fi, (_, va) in enumerate(folds):
    fold_id[va] = fi

BANDS = [('band_3_6h', 'bandoof_bandfm2.npy', 'bandtest_bandfm2.npy'),
         ('band_6_78h', 'bandoof_band_mid.npy', 'bandtest_band_mid.npy')]

# once global OOF usaritini band icin santralize et
mix_oof = g_oof.copy()
mix_tst = g_tst.copy()


def band_mix(ids_tr, s_tr, s_te, g_tr, g_te, skip_mm4):
    """per-band karistirici; icten CV ile degerlendir. Loji uzayinda LR."""
    z_tr = (s_tr - s_tr.mean()) / (s_tr.std() + 1e-12)
    z_te = (s_te - s_tr.mean()) / (s_tr.std() + 1e-12)
    X = np.column_stack([logit(np.clip(g_tr, SMOOTH, 1 - SMOOTH)), z_tr])
    XT = np.column_stack([logit(np.clip(g_te, SMOOTH, 1 - SMOOTH)), z_te])
    yb = y[ids_tr]
    keep = np.ones(len(ids_tr), bool)
    if skip_mm4:
        keep = nm_tr[ids_tr] < 4
    ids_k = ids_tr[keep]
    Xk, yk = X[keep], yb[keep]
    foldk = fold_id[ids_k]

    oob = np.zeros(len(yb))
    idx_k = np.where(keep)[0]
    for fi in range(5):
        tr = (foldk != fi); va = (foldk == fi)
        m = LogisticRegression(max_iter=1000, C=1.0).fit(Xk[tr], yk[tr])
        oob[idx_k[va]] = m.predict_proba(Xk[va])[:, 1]

    auc_global_inband = roc_auc_score(yb, g_tr)
    auc_mix_inband = roc_auc_score(yb, np.where(keep, oob, g_tr))
    m_full = LogisticRegression(max_iter=1000, C=1.0).fit(Xk, yk)
    p_te = logit(np.clip(g_te, SMOOTH, 1 - SMOOTH))
    if skip_mm4:
        keep_te = nm_te[ids_te - 691369] < 4
    else:
        keep_te = np.ones(len(s_te), bool)
    if np.any(keep_te):
        p_te[keep_te] = m_full.predict_proba(XT[keep_te])[:, 1]
    oob_full = np.where(keep, oob, g_tr)
    return auc_global_inband, auc_mix_inband, expit(p_te), oob_full, len(ids_tr)


for band, oofn, tstn in BANDS:
    d = pd.read_parquet(f'{R}/{band}.parquet')
    mtr = d['split'].values == 'train'
    ids_tr = d.loc[mtr, 'id'].values.astype(int)
    ids_te = d.loc[~mtr, 'id'].values.astype(int)
    s_tr = np.load(f'{R}/{oofn}').astype('float64')
    s_te = np.load(f'{R}/{tstn}').astype('float64')
    g_tr = g_oof[ids_tr]
    g_te = g_tst[ids_te - 691369]
    a_g, a_m, p_te, oob, nrows = band_mix(ids_tr, s_tr, s_te, g_tr, g_te, skip_mm4=True)
    print(f'{band}: {nrows} satir  global={a_g:.5f}  mixed={a_m:.5f}  delta={(a_m-a_g):+.5f}', flush=True)
    if a_m > a_g:
        print(f'  -> band duzeltmesi UYGULANIYOR', flush=True)
        mix_oof[ids_tr] = oob
        mix_tst[ids_te - 691369] = p_te

auc_all = roc_auc_score(y, mix_oof)
print(f'exp_xgb global butun: {roc_auc_score(y, g_oof):.5f}   band-duzeltmeli tahmini: {auc_all:.5f}', flush=True)
save_submission('exp_xgb_band', test_id, mix_tst)
print('DONE', flush=True)