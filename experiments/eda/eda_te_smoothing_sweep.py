import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA
SEED = 42

train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values
prior = y.mean()


def oof_encode(col, y, folds, smooth):
    out = np.zeros(len(col))
    for tr, va in folds.split(np.zeros(len(col)), y):
        g = pd.DataFrame({'v': col[tr], 'y': y[tr]}).groupby('v')['y'].agg(['count', 'mean'])
        g['enc'] = (g['count'] * g['mean'] + smooth * prior) / (g['count'] + smooth)
        m = g['enc'].to_dict()
        out[va] = pd.Series(col[va]).map(m).fillna(prior).values
    return out


te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

print('TEK ÖZELLİK AUC (raw vs target-encoded):')
print(f'{"Sütun":28s} {"raw":>8s} {"encoded(s=20)":>14s} {"encoded(s=5)":>12s}')
for c in ['daily_screen_time_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
          'social_media_hours', 'gaming_hours', 'weekend_screen_time', 'work_study_hours']:
    raw = train[c].fillna(-999).values
    enc20 = oof_encode(train[c].astype(str).values, y, te_skf, 20)
    enc5 = oof_encode(train[c].astype(str).values, y, te_skf, 5)
    auc_raw = roc_auc_score(y, raw)
    auc20 = roc_auc_score(y, enc20)
    auc5 = roc_auc_score(y, enc5)
    print(f'{c:28s} {auc_raw:8.4f} {auc20:14.4f} {auc5:12.4f}')
