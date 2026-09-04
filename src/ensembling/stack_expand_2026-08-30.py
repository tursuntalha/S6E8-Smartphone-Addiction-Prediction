"""
Yeni OOF uyelerini stack'e ekler ve durust 5-fold LR CV ile olcer.

Gruplar:
  A) base76  (kutuphane 74 + bizim 2)
  B) +dariush(7, ayni frozen 5-fold), +rayk_fm(5, ayni fold, ham skor->z), +naji_blend(10, ayni fold)
  C) +paiky(11, fold semasi BILINMIYOR/10f -- karistirmayi dikkatli kabul et)

Member bazinda transform: [0,1] icindeyse logit, degilse z-score (ham skor).
Cikti: icten 5-fold CV OOF AUC + test tahminleri + submission CSV.
"""
import sys, os, time, glob
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, save_submission, CACHE

SMOOTH = 1e-6
ROOT = f'{DATA}/extra_oof'


def load_csv_sorted(path, col):
    d = pd.read_csv(path)
    d = d.sort_values('id').reset_index(drop=True)
    return d[col].astype('float64').values


def transform_member(v, is_prob):
    if is_prob:
        return logit(np.clip(v, SMOOTH, 1 - SMOOTH))
    m, s = v.mean(), v.std()
    return (v - m) / (s + 1e-12)


def load_group_dariush():
    names, O, T, zfit = [], [], [], []
    for c in 'abcdefg':
        names.append('dariush_' + c)
        o = np.load(f'{ROOT}/dariush/unzipped/oof_{c}.npy').astype('float64')
        t = np.load(f'{ROOT}/dariush/unzipped/test_{c}.npy').astype('float64')
        O.append(logit(np.clip(o, SMOOTH, 1 - SMOOTH)))
        T.append(logit(np.clip(t, SMOOTH, 1 - SMOOTH)))
    return names, np.column_stack(O), np.column_stack(T)


def load_group_rayk():
    names, O, T = [], [], []
    for c in ['fmplr', 'fmnum', 'fmdeep', 'fmwide', 'fmpure']:
        names.append('rayk_' + c)
        O.append(np.load(f'{ROOT}/rayk/unzipped/oof_{c}.npy').astype('float64'))
        T.append(np.load(f'{ROOT}/rayk/unzipped/test_{c}.npy').astype('float64'))
    O = np.column_stack(O)
    T = np.column_stack(T)
    mu, sd = O.mean(0), O.std(0)
    return names, (O - mu) / sd, (T - mu) / sd


def load_group_najiblends():
    names, O, T = [], [], []
    for b in ['07', '08', '09', '10', '12', '13', '14', '16', '18', '19']:
        names.append(f'najiblen_{b}')
        o = load_csv_sorted(f'{ROOT}/najiama/unzipped/{b}_blend_oof_predictions.csv', 'addicted_label')
        sp = f'{ROOT}/najiama/unzipped/{b}_blend_submission.csv'
        if not os.path.exists(sp):
            sp += '.csv'
        t = load_csv_sorted(sp, 'addicted_label')
        O.append(logit(np.clip(o, SMOOTH, 1 - SMOOTH)))
        T.append(logit(np.clip(t, SMOOTH, 1 - SMOOTH)))
    return names, np.column_stack(O), np.column_stack(T)


def load_group_paiky():
    o = pd.read_csv(f'{ROOT}/paiky/unzipped/oof_predictions.csv')
    t = pd.read_csv(f'{ROOT}/paiky/unzipped/test_predictions.csv')
    o = o.sort_values('id').reset_index(drop=True)
    t = t.sort_values('id').reset_index(drop=True)
    cols = [c for c in o.columns if c != 'id']
    for c in cols:
        assert c in t.columns
    names = ['paiky_' + c for c in cols]
    O = logit(np.clip(o[cols].values.astype('float64'), SMOOTH, 1 - SMOOTH))
    T = logit(np.clip(t[cols].values.astype('float64'), SMOOTH, 1 - SMOOTH))
    return names, O, T


def honest_lr(X, XT, y, n, folds):
    oof = np.zeros(n)
    for tr, va in folds:
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    m = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    return oof, m.predict_proba(XT)[:, 1]


def main():
    t0 = time.time()
    names, O, T, y, n = load_all()
    ntest = T.shape[0]
    print(f'[load] base={len(names)} {time.time()-t0:.0f}s', flush=True)
    test_id = pd.read_csv(f'{DATA}/test.csv')['id'].values
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n), y))

    LO = logit(np.clip(O, SMOOTH, 1 - SMOOTH))
    LT = logit(np.clip(T, SMOOTH, 1 - SMOOTH))

    reps = {
        'dariush': load_group_dariush(),
        'rayk_fm': load_group_rayk(),
        'najiblen': load_group_najiblends(),
        'paiky': load_group_paiky(),
    }
    new_total = 0
    for k, (gn, gO, gT) in reps.items():
        for i, g in enumerate(gn):
            auc_prob = roc_auc_score(y, gO[:, i]) if len(gO) else None
        new_total += gO.shape[1]
        print(f'[group] {k}: {gO.shape[1]} uye  (toplam snr~{sum(1 for _ in reps)})', flush=True)
    print(f'[ok] yeni uye toplam {new_total}', flush=True)
    # per-member rapor
    for k, (gn, gO, gT) in reps.items():
        for i, g in enumerate(gn):
            a = roc_auc_score(y, gO[:, i])
            corr = max(abs(np.corrcoef(gO[:, i], LO[:, j])[0, 1]) for j in range(LO.shape[1]))
            print(f'  {g:18s} AUC={a:.5f}  maxcorr-vs-base76={corr:.4f}', flush=True)

    combos = {
        'exp_aligned': ['dariush', 'rayk_fm', 'najiblen'],
        'exp_all': ['dariush', 'rayk_fm', 'najiblen', 'paiky'],
    }
    keep = {'exp_aligned': ['dariush', 'rayk_fm', 'najiblen']}
    for combo, grps in combos.items():
        X = LO.copy()
        XT_ = LT.copy()
        for g in grps:
            gn, gO, gT = reps[g]
            X = np.hstack([X, gO])
            XT_ = np.hstack([XT_, gT])
        oof, pred = honest_lr(X, XT_, y, n, folds)
        auc = roc_auc_score(y, oof)
        print(f'=== {combo}: {X.shape[1]} uye  OOF AUC={auc:.5f}  ({(time.time()-t0)/60:.1f} dk)', flush=True)
        np.save(f'{CACHE}/stacko_{combo}.npy', oof.astype('float64'))
        np.save(f'{CACHE}/stackt_{combo}.npy', pred.astype('float64'))
        save_submission(f'stack_{combo}', test_id, pred)


if __name__ == '__main__':
    main()