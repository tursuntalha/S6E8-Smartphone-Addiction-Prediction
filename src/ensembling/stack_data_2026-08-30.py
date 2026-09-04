"""
Ortak veri yukleyici: OOF kutuphanesi(74) + bizim 2 OOF = 76 model.
Transform uzanlari once hesaplanip cache'lenir (nn_cache/stack_*.npy).
"""
import numpy as np
import pandas as pd
import os, glob, time
from scipy.special import logit
from scipy.stats import rankdata

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA, NN_CACHE as CACHE, SUB
L = f'{DATA}/oof_library/oof'
SMOOTH = 1e-6


def load_all():
    t0 = time.time()
    train = pd.read_csv(f'{DATA}/train.csv')
    y = train['addicted_label'].values.astype(np.uint8)
    n = len(y)
    names_lib = sorted(os.path.basename(p)[4:-4] for p in glob.glob(f'{L}/oof_*.npy'))
    O_lib = np.column_stack([np.load(f'{L}/oof_{nm}.npy') for nm in names_lib]).astype('float64')
    T_lib = np.column_stack([np.load(f'{L}/test_{nm}.npy') for nm in names_lib]).astype('float64')
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
    print(f'[load] {len(names)} model, O={O.shape} T={T.shape}  ({time.time()-t0:.0f}s)', flush=True)
    return names, O, T, y, n


def get_transforms(names, O, T, n, ntest):
    """logit / ranked / standardize uzaylarini onbellege al."""
    cache_o = f'{CACHE}/stack_TR_O.npz'
    cache_t = f'{CACHE}/stack_TR_T.npz'
    if os.path.exists(cache_o) and os.path.exists(cache_t):
        dO = np.load(cache_o)
        dT = np.load(cache_t)
        print('[cache] transformlar okundu', flush=True)
        return dO['logit'], dT['logit'], dO['rank'], dT['rank'], dO['zrank'], dT['zrank']

    lo = logit(np.clip(O, SMOOTH, 1 - SMOOTH))
    lt = logit(np.clip(T, SMOOTH, 1 - SMOOTH))
    ro = rankdata(O, axis=1) / n
    rt = rankdata(T, axis=1) / ntest
    zo = (ro - ro.mean(axis=1, keepdims=True)) / (ro.std(axis=1, keepdims=True) + 1e-12)
    zt = (rt - rt.mean(axis=1, keepdims=True)) / (rt.std(axis=1, keepdims=True) + 1e-12)
    np.savez(cache_o, logit=lo, rank=ro, zrank=zo, names=names)
    np.savez(cache_t, logit=lt, rank=rt, zrank=zt)
    print('[cache] transformlar yazildi', flush=True)
    return lo, lt, ro, rt, zo, zt


def save_submission(name, test_id, pred):
    import pandas as pd
    os.makedirs(f'{SUB}/2026-08-30', exist_ok=True)
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    p = p / p.max()
    path = f'{SUB}/2026-08-30/{name}_2026-08-30.csv'
    pd.DataFrame({'id': test_id, 'addicted_label': p}).to_csv(path, index=False)
    print(f'Saved: {path}', flush=True)
    return path