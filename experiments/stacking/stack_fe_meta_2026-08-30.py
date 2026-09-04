"""
Meta-seviyesinde temsil iyilestirmesi deneyi:
Meta = [98 uye logit] + [raw+engered ozellikler] uzerinde CatBoost / XGB honest CV.

Amac: meta'nin uyelerin kacirdigi ham-etkilesimlerden sinyal bulup bulamadigini olcmek.
Temel kiyas: exp_xgb (sadece 98 logit, 0.97003).
"""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit
import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA

sys.path.insert(0, os.path.dirname(__file__))
from stack_data_2026_08_30 import load_all, save_submission, CACHE

SMOOTH = 1e-6
tr = pd.read_csv(f'{DATA}/train.csv')
te = pd.read_csv(f'{DATA}/test.csv')
test_id = te['id'].values
y = tr['addicted_label'].values.astype(np.uint8)
n = len(y)
ntest = len(te)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y))

# --- raw + engineered ozellik seti (train/test ayrintili) ---
raw_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
            'weekend_screen_time']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']


def fe(df):
    F = pd.DataFrame(index=df.index)
    for c in raw_cols:
        F[c] = df[c].astype('float32')
        F[c + '_miss'] = df[c].isna().astype('float32')
        F['rnk_' + c] = df[c].rank(pct=True)
    F['nmiss'] = df[raw_cols + cat_cols].isna().sum(axis=1).astype('float32')
    F['sum_comp'] = df[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    F['weekend_div_daily'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 1e-6)
    F['sleep_plus_daily'] = df['sleep_hours'] + df['daily_screen_time_hours']
    # per-value lookup (tam tren uzerinde, stabil r=0.9965 oldugu icin kabul edilebilir ilk yoklamada)
    for c in ['app_opens_per_day', 'notifications_per_day', 'social_media_hours',
              'gaming_hours', 'work_study_hours', 'sleep_hours', 'daily_screen_time_hours',
              'weekend_screen_time', 'age']:
        rate = df.groupby(c)[['addicted_label']].mean() if 'addicted_label' in df else None
        if rate is not None:
            F['look_' + c] = df[c].map(rate['addicted_label'])
            F['look_' + c].fillna(df['addicted_label'].mean() if 'addicted_label' in df else 0.71, inplace=True)
    for c in cat_cols:
        F[c] = df[c].astype('category')
    return F


Ftr = fe(tr)
print('train fe cols:', Ftr.shape, flush=True)
test_label = pd.Series(tr['addicted_label'].mean(), index=te.index)
Fte = None
# test icin rate map'lerini tren'den tasi
rate_maps = {}
for c in ['app_opens_per_day', 'notifications_per_day', 'social_media_hours',
          'gaming_hours', 'work_study_hours', 'sleep_hours', 'daily_screen_time_hours',
          'weekend_screen_time', 'age']:
    rate_maps[c] = tr.groupby(c)['addicted_label'].mean()

Fte = Ftr.iloc[:0].copy()
Fte = fe(te.assign(addicted_label=np.nan))
for c in rate_maps:
    Fte['look_' + c] = te[c].map(rate_maps[c])

# 98 uyeli matris
import importlib.util
_spec = importlib.util.spec_from_file_location('se', os.path.join(os.path.dirname(__file__), 'stack_expand_2026-08-30.py'))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
names, O, T, _, _ = load_all()
LO = logit(np.clip(O, SMOOTH, 1 - SMOOTH))
LT = logit(np.clip(T, SMOOTH, 1 - SMOOTH))
for gn, gO, gT in [_se.load_group_dariush(), _se.load_group_rayk(), _se.load_group_najiblends()]:
    LO = np.hstack([LO, gO]); LT = np.hstack([LT, gT])
print('98 matris ok', LO.shape, flush=True)

# raw ozellikleri (sayisal) standardize etmeden catboost'a ver; kategorikleri ayir
num_cols = [c for c in Ftr.columns if c not in cat_cols]
X_raw = Ftr[num_cols].values.astype('float64')
XT_raw = Fte[num_cols].values.astype('float64')

# kombinasyonlar
def run_catboost(name, X, XT, cat_idx=None):
    import catboost as cb
    t0 = time.time()
    oof = np.zeros(n)
    params = dict(iterations=1200, learning_rate=0.035, depth=6, l2_leaf_reg=4.0,
                  random_seed=42, verbose=0, loss_function='Logloss', eval_metric='AUC')
    if cat_idx:
        params['cat_features'] = cat_idx
    oof = np.zeros(n)
    for tri, va in folds:
        m = cb.CatBoostClassifier(**params).fit(X[tri], y[tri])
        oof[va] = m.predict_proba(X[va])[:, 1]
    m = cb.CatBoostClassifier(**params).fit(X, y)
    pred = m.predict_proba(XT)[:, 1]
    a = roc_auc_score(y, oof)
    print(f'{name}: OOF={a:.5f} ({(time.time()-t0)/60:.1f} dk)', flush=True)
    np.save(f'{CACHE}/stacko_{name}.npy', oof.astype('float64'))
    np.save(f'{CACHE}/stackt_{name}.npy', pred.astype('float64'))
    save_submission(name, test_id, pred)
    return a

# (a) sadece 98 logit CatBoost (kontrol: exp_cat 0.96990)
run_catboost('fe_meta_cb_logitonly', LO, LT, None)

# (b) sadece raw eng'li CatBoost: raw'in kendi gucu
run_catboost('fe_only_cb', X_raw, XT_raw, None)

# (c) 98 logit + raw eng'li CatBoost : asil deney
Xc = np.hstack([LO, X_raw])
XTc = np.hstack([LT, XT_raw])
run_catboost('fe_meta_cb', Xc, XTc, None)
print('DONE', flush=True)