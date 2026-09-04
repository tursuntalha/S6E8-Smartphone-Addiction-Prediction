import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA
train = pd.read_csv(f'{DATA}/train.csv')
feats = [c for c in train.columns if c not in ['id', 'addicted_label']]

cat_cols = ['gender', 'stress_level', 'academic_work_impact']
for c in cat_cols:
    train[c] = train[c].astype('category')

results = {}
for target in ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'sleep_hours',
               'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time', 'work_study_hours', 'age']:
    others = [c for c in feats if c != target]
    sub = train[['addicted_label'] + others + [target]].dropna(subset=[target]).sample(200000, random_state=42)
    y = sub[target].values
    X = sub[others]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbosity=-1, random_state=42)
    m.fit(X_tr, y_tr)
    p = m.predict(X_va)
    r2 = 1 - mean_squared_error(y_va, p) / np.var(y_va)
    corr = pearsonr(y_va, p)[0]
    results[target] = (r2, corr)

print('{:28s} {:>8s} {:>10s}  Değerlendirme'.format('Sütun', 'R2', 'Pearson r'))
for k, (r2, c) in results.items():
    if r2 > 0.7:
        v = 'COK IYI impute edilir'
    elif r2 > 0.4:
        v = 'Iyi - kismen impute'
    elif r2 > 0.2:
        v = 'Zayif - dikkatli'
    else:
        v = 'Kotu - NaN bayragi daha iyi'
    print('{:28s} {:8.3f} {:10.3f}  {}'.format(k, r2, c, v))
