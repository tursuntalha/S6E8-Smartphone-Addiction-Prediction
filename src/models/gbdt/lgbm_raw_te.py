import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = 'data'
SEED = 42
RUN_NAME = 'lgbm_raw_te'
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

for frame in (train, test):
    frame['sum_components'] = frame[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame['ratio_weekend_daily'] = frame['weekend_screen_time'] / frame['daily_screen_time_hours'].replace(0, np.nan)

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

# RAW numerik (NaN -> -999 sentinel), kategorik -> sayısal
raw_tr = train[all_cats].copy()
raw_te = test[all_cats].copy()
for c in all_cats:
    for frame in (raw_tr, raw_te):
        frame[c] = frame[c].astype('object').where(frame[c].notna(), np.nan)
        frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# ENCODED (10-fold OOF target encoding, m-estimate smoothing=20)
enc_tr = pd.DataFrame(index=train.index)
enc_te = pd.DataFrame(index=test.index)
for c in all_cats:
    col_tr = train[c].astype(str).values
    col_te = test[c].astype(str).values
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + 3.0 * prior) / (g['count'] + 3.0)
    enc_te[c] = pd.Series(col_te).map(g['enc'].to_dict()).fillna(prior).values
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + 3.0 * prior) / (gk['count'] + 3.0)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    enc_tr[c] = oof_enc

# collinearity kontrolü
cmat = enc_tr[cont_cols].corr()
print('Encoded feature korelasyonları (|r|>0.8):')
pairs = []
for i in range(len(cmat)):
    for j in range(i + 1, len(cmat)):
        if abs(cmat.iloc[i, j]) > 0.8:
            pairs.append((cmat.index[i], cmat.columns[j], round(cmat.iloc[i, j], 2)))
print('  ', len(pairs), 'cift (encoded), ornek:', pairs[:5])

# RAW+ENCODED birleşik (28 özellik: 14 raw + 14 te_)
Xc = pd.concat([raw_tr, enc_tr.add_prefix('te_')], axis=1)
Xc_test = pd.concat([raw_te, enc_te.add_prefix('te_')], axis=1)
print(f'Birleşik özellik sayısı: {Xc.shape[1]} | encoding: {time.time()-t0:.0f}s')

with open('sub/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(Xc))
test_pred = np.zeros(len(Xc_test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xc, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(Xc.iloc[tr_idx], y[tr_idx], eval_set=[(Xc.iloc[va_idx], y[va_idx])],
              eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = model.predict_proba(Xc.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(Xc_test)[:, 1] / skf.n_splits
    print(f'Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.5f}')

print(f'\nCV OOF AUC (raw+encoded): {roc_auc_score(y, oof):.5f}')
imp = pd.Series(model.feature_importances_, index=Xc.columns).sort_values(ascending=False)
print('Top 10 önem:')
print(imp.head(10).round(0).to_string())

sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
sub_path = f'sub/{RUN_NAME}_2026-08-11.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
print(f'Elapsed: {time.time()-t0:.0f}s')
