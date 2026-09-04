import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from config import DATA
train = pd.read_csv(f'{DATA}/train.csv')
y = train['addicted_label'].values
prior = y.mean()
SMOOTH = 20.0

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

print('=' * 60)
print('1) NOTIFICATIONS "LOOKUP TABLE" — jeneratörün kodu')
print('=' * 60)
tmp = pd.DataFrame({'v': train['notifications_per_day'], 'y': y}).dropna()
g = tmp.groupby('v')['y'].agg(['count', 'mean'])
sample = g[(g['count'] > 50) & (g['count'] < 500)].sort_values('v')
print('integer değerlerin bazıları ve P(bağımlı|değer):')
for v, row in sample.sample(min(12, len(sample)), random_state=42).sort_index().iterrows():
    print(f'   notifications={v:5.0f}  P(addicted)={row["mean"]:.3f}  (n={int(row["count"])})')
print(f'   -> P değere göre rastgele dalgalanıyor (monotonik DEĞİL)')

print()
print('=' * 60)
print('2) TOPLAMSALLIK TESTİ — logistic vs LightGBM')
print('=' * 60)
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X_enc = pd.DataFrame(index=train.index)
for c in all_cats:
    col_tr = train[c].astype(str).values
    oof = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + SMOOTH * prior) / (gk['count'] + SMOOTH)
        oof[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    X_enc[c] = oof

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, Xf in [('TE sadece (additive testi)', X_enc)]:
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(Xf, y):
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(Xf.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = lr.predict_proba(Xf.iloc[va_idx])[:, 1]
    print(f'Logistic + {name}: CV AUC = {roc_auc_score(y, oof):.5f}')
    print(f'   (LightGBM raw+TE referansı: 0.96726)')
    print(f'   -> Aradaki fark etkileşim (interaction) bilgisinin ölçüsü')

print()
print('=' * 60)
print('3) İKİLİ ETKİLEŞİM — sosyal X bildirim matrisi')
print('=' * 60)
t2 = pd.DataFrame({'soc': pd.qcut(train['social_media_hours'], 3, labels=False, duplicates='drop'),
                   'not': pd.qcut(train['notifications_per_day'], 3, labels=False, duplicates='drop'),
                   'y': y}).dropna()
piv = t2.pivot_table(index='soc', columns='not', values='y', aggfunc='mean')
piv.columns = ['not_low', 'not_mid', 'not_high']
piv.index = ['soc_low', 'soc_mid', 'soc_high']
print(piv.round(3).to_string())
print('  -> Toplamsal ise satır/kolon etkileri ayrışır; çarpımsal ise köşe çok yüksek olur')
