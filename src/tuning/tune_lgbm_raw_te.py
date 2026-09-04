import pandas as pd
import numpy as np
import json
import time
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

DATA = 'data'
SEED = 42
N_TRIALS = 40
t0 = time.time()

train = pd.read_csv(f'{DATA}/train.csv')

for frame in (train,):
    frame['sum_components'] = frame[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame['ratio_weekend_daily'] = frame['weekend_screen_time'] / frame['daily_screen_time_hours'].replace(0, np.nan)

cont_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time', 'sum_components', 'ratio_weekend_daily']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
all_cats = cont_cols + cat_cols

# RAW numerik (NaN -> -999 sentinel)
raw = train[all_cats].copy()
for c in all_cats:
    raw[c] = raw[c].astype('object').where(raw[c].notna(), np.nan)
    raw[c] = pd.to_numeric(raw[c], errors='coerce').fillna(-999)

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# 10-fold OOF target encoding (aynı raw_te.py mantığı)
enc = pd.DataFrame(index=train.index)
for c in all_cats:
    col_tr = train[c].astype(str).values
    g = pd.DataFrame({'v': col_tr, 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
    g['enc'] = (g['count'] * g['mean'] + 20.0 * prior) / (g['count'] + 20.0)
    oof_enc = np.zeros(len(train))
    for tr_idx, va_idx in te_skf.split(train, y):
        gk = pd.DataFrame({'v': col_tr[tr_idx], 'y': y[tr_idx]}).groupby('v')['y'].agg(['count', 'mean'])
        gk['enc'] = (gk['count'] * gk['mean'] + 20.0 * prior) / (gk['count'] + 20.0)
        oof_enc[va_idx] = pd.Series(col_tr[va_idx]).map(gk['enc'].to_dict()).fillna(prior).values
    enc[c] = oof_enc

X = pd.concat([raw, enc.add_prefix('te_')], axis=1)
print(f'Özellik sayısı: {X.shape[1]} | encoding: {time.time()-t0:.0f}s')

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'random_state': SEED,
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])


study = optuna.create_study(direction='maximize', study_name='lgbm_raw_te_tune',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f'\nBest val AUC (holdout): {study.best_value:.5f}')
print('Best params:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')

with open('sub/best_params_lgbm_raw_te.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print('Saved: sub/best_params_lgbm_raw_te.json')
print(f'Elapsed: {time.time()-t0:.0f}s')
