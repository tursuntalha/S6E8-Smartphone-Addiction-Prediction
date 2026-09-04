import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA
train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')
feats = [c for c in train.columns if c not in ['id', 'addicted_label']]

print('=' * 60)
print('1) TRAIN vs TEST DAĞILIM KARŞILAŞTIRMASI (sayısal)')
print('=' * 60)
print(f'{"Sütun":26s} {"train_mean":>11s} {"test_mean":>11s} {"train_std":>10s} {"test_std":>10s} {"delta_mean":>10s}')
for c in feats:
    if train[c].dtype.kind in 'fi':
        tm, ts = train[c].mean(), train[c].std()
        em, es = test[c].mean(), test[c].std()
        print(f'{c:26s} {tm:11.3f} {em:11.3f} {ts:10.3f} {es:10.3f} {em - tm:10.3f}')

print()
print('=' * 60)
print('2) KATEGORİK DAĞILIMLAR (train vs test %)')
print('=' * 60)
for c in ['gender', 'stress_level', 'academic_work_impact']:
    t = train[c].value_counts(normalize=True) * 100
    e = test[c].value_counts(normalize=True) * 100
    print(f'--- {c} ---')
    for v in sorted(set(t.index) | set(e.index)):
        tv = t.get(v, 0)
        ev = e.get(v, 0)
        print(f'   {v:12s} train={tv:6.2f}%  test={ev:6.2f}%  fark={ev - tv:+6.2f}%')

print()
print('=' * 60)
print('3) DETERMİNİSTİK İLİŞKİ (sentetiklik göstergesi)')
print('=' * 60)
for name, df in [('train', train), ('test', test)]:
    sub = df.dropna(subset=['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']).sample(300000, random_state=1)
    resid = sub['daily_screen_time_hours'] - (sub['social_media_hours'] + sub['gaming_hours'] + sub['work_study_hours'])
    print(f'{name}: daily - toplam farkı -> mean={resid.mean():.3f} std={resid.std():.3f} min={resid.min():.3f} max={resid.max():.3f}')

print()
print('=' * 60)
print('4) YUVARLAMA / GRANÜLERLİK (sentetiklik göstergesi)')
print('=' * 60)
sub = train.sample(500000, random_state=1)
for c in ['daily_screen_time_hours', 'sleep_hours', 'social_media_hours', 'gaming_hours']:
    vals = sub[c].dropna()
    decimals = vals.mod(1)
    rounded_rate = (np.abs(vals - np.round(vals)) < 1e-9).mean()
    print(f'{c:26s} unique={vals.nunique():6d}  "tam sayı" oranı=%{rounded_rate:.1f}')

print()
print('=' * 60)
print('5) KORELASYON YAPISI (sentetiklik: çok yüksek/çok düşük korelasyon)')
print('=' * 60)
num = [c for c in feats if train[c].dtype.kind in 'fi']
corr = train[num].corr()
strong = []
for i in range(len(num)):
    for j in range(i + 1, len(num)):
        r = corr.iloc[i, j]
        if abs(r) > 0.3:
            strong.append((num[i], num[j], r))
strong.sort(key=lambda x: -abs(x[2]))
print('|r| > 0.3 olan çiftler:')
for a, b, r in strong:
    print(f'   {a:26s} ~ {b:26s}  r={r:+.3f}')
print(f'\nToplam çift sayısı: {len(num)*(len(num)-1)//2}, güçlü ilişki: {len(strong)}')

print()
print('=' * 60)
print('6) TAM TEKRAR (duplicate) SATIR')
print('=' * 60)
print('train dup:', train.duplicated(subset=feats).sum(), '| test dup:', test.duplicated(subset=feats).sum())

print()
print('=' * 60)
print('7) PSI (Population Stability Index) — dağılım kayması')
print('=' * 60)


def psi(a, b, bins=20):
    a = a.dropna(); b = b.dropna()
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
    return np.sum((pa - pb) * np.log(pa / pb))


for c in feats:
    if train[c].dtype.kind in 'fi':
        print(f'{c:26s} PSI={psi(train[c], test[c]):.3f}')
