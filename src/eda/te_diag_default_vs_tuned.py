import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time

DATA = 'data'
SEED = 42
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

for frame in (train, test):
    for c in all_cats:
        frame[c] = frame[c].astype('object').where(frame[c].notna(), 'missing').astype(str)

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

X_enc = pd.DataFrame(index=train.index)
X_test_enc = pd.DataFrame(index=test.index)

for c in all_cats:
    col_tr = train[c].values
    col_te = test[c].values
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + 20.0 * prior) / (g['count'] + 20.0)
    map_te = g['enc'].to_dict()
    X_test_enc[c] = pd.Series(col_te).map(map_te).fillna(prior).values
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + 20.0 * prior) / (gk['count'] + 20.0)
        mapk = gk['enc'].to_dict()
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(mapk).fillna(prior).values
    X_enc[c] = oof_enc

feats = all_cats
print(f'Encoding: {time.time()-t0:.0f}s')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

param_sets = {
    'default': dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, n_estimators=3000, verbosity=-1, random_state=SEED),
    'tuned': dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED,
                  **json.load(open('sub/best_params_lgbm.json'))),
}

for pname, params in param_sets.items():
    oof = np.zeros(len(X_enc))
    test_pred = np.zeros(len(X_test_enc))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_enc, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_enc.iloc[tr_idx], y[tr_idx], eval_set=[(X_enc.iloc[va_idx], y[va_idx])],
                  eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = model.predict_proba(X_enc.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test_enc)[:, 1] / skf.n_splits
    train_auc = roc_auc_score(y[tr_idx], model.predict_proba(X_enc.iloc[tr_idx])[:, 1])
    print(f'\n[{pname}] CV OOF AUC: {roc_auc_score(y, oof):.5f} | son fold train AUC: {train_auc:.5f}')
    if pname == 'default':
        np.save('sub/te_test_pred_default.npy', test_pred)
        np.save('sub/te_oof_default.npy', oof)

print(f'Elapsed: {time.time()-t0:.0f}s')
