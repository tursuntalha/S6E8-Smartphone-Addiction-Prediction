"""Final 10 tablosu: korrelasyon + dosya sagligi (dogru listeyle)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import sys, os
sys.path.insert(0, os.getcwd())
from src.config import NN_CACHE, SUB

SUBS = {
    'exp_meta6': f'{SUB}/2026-08-30/exp_meta6_2026-08-30.csv',
    'exp_xgb': f'{SUB}/2026-08-30/stack_exp_xgb_2026-08-30.csv',
    'exp_ax': f'{SUB}/2026-08-30/exp_ax_2026-08-30.csv',
    'exp_all': f'{SUB}/2026-08-30/stack_exp_all_2026-08-30.csv',
    'exp_aligned': f'{SUB}/2026-08-30/stack_exp_aligned_2026-08-30.csv',
    'exp_cat': f'{SUB}/2026-08-30/exp_cat_2026-08-30.csv',
    'w66_prod': f'{SUB}/2026-08-29/blend_gbdt_origfeat_nn_featfull_w66_2026-08-29.csv',
    'stack_logit_lr': f'{SUB}/2026-08-30/stack_logit_lr_2026-08-30.csv',
    'stack_nnls': f'{SUB}/2026-08-30/stack_nnls_2026-08-30.csv',
    'extra_lr_raw9': f'{SUB}/2026-08-30/stack_extra_lr_raw9_2026-08-30.csv',
}
OOF = {
    'exp_meta6': 0.97004, 'exp_xgb': 0.97003, 'exp_ax': 0.97002, 'exp_all': 0.96999,
    'exp_aligned': 0.96995, 'exp_cat': 0.96990, 'w66_prod': float('nan'),
    'stack_logit_lr': 0.96969, 'stack_nnls': 0.96943, 'extra_lr_raw9': 0.96968,
}
LB = {'w66_prod': 0.97035}

preds = {}
for k, p in SUBS.items():
    d = pd.read_csv(p).sort_values('id').reset_index(drop=True)
    v = d['addicted_label'].values.astype('float64')
    assert len(v) == 296302 and np.isfinite(v).all() and v.min() > 1e-12 and v.max() <= 1.0 + 1e-9 and v.std() > 1e-9
    preds[k] = rankdata(v) / len(v)
    print(f'{k:16s} min={v.min():.6f} max={v.max():.6f} OOF={OOF[k]:.5f} LB={LB.get(k,"-")}', flush=True)

names = list(SUBS)
R = np.zeros((len(names), len(names)))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        R[i, j] = np.corrcoef(preds[a], preds[b])[0, 1]
np.set_printoptions(precision=3, suppress=True)
print()
print('Spearman korrelasyon (test preds):')
print('   ' + ' '.join(f'{x[:10]:>11}' for x in names))
for i, a in enumerate(names):
    print(f'{a[:11]:>11} ' + ' '.join(f'{R[i,j]:11.3f}' for j in range(len(names))))
np.save(f'{NN_CACHE}/final10_corr.npy', R)
pd.DataFrame({'sub': names, 'path': [SUBS[k] for k in names], 'oof_auc': [OOF[k] for k in names], 'lb': [LB.get(k, np.nan) for k in names]}
            ).to_csv(f'{NN_CACHE}/final10_table.csv', index=False)
print('kaydedildi: nn_cache/final10_table.csv + final10_corr.npy', flush=True)