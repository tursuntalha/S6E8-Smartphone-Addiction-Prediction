import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA
train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values

print('=' * 60)
print('1) GRID ANALİZİ — değer adımları ve aralıklar')
print('=' * 60)
for c in ['daily_screen_time_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
          'social_media_hours', 'gaming_hours', 'work_study_hours', 'weekend_screen_time', 'age']:
    vals = np.sort(train[c].dropna().unique())
    diffs = np.diff(vals)
    print(f'{c:26s} n_unique={len(vals):5d} min={vals[0]:8.2f} max={vals[-1]:8.2f}')
    print(f'   min adım={np.min(diffs):.4f}  median adım={np.median(diffs):.4f}  max adım={np.max(diffs):.4f}')

print()
print('=' * 60)
print('2) P(bağımlı | değer) EĞRİLERİ — monotonluk / adım yapısı')
print('=' * 60)
for c in ['daily_screen_time_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
          'social_media_hours', 'gaming_hours', 'weekend_screen_time']:
    tmp = pd.DataFrame({'v': train[c], 'y': y}).dropna()
    tmp['bin'] = pd.qcut(tmp['v'], q=10, duplicates='drop')
    rates = tmp.groupby('bin', observed=True)['y'].mean()
    mono = np.sign(np.diff(rates.values))
    increasing = (mono >= 0).mean() * 100
    decreasing = (mono <= 0).mean() * 100
    dir_ = 'ARTAN' if increasing >= 80 else ('AZALAN' if decreasing >= 80 else 'KARIŞIK')
    print(f'{c:26s} bin ortalamaları: ' + ' '.join(f'{r:.2f}' for r in rates.values) + f'   -> {dir_}')
