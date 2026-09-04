import pandas as pd
import numpy as np
import json
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.getcwd())
from src.config import DATA, CONFIGS
SEED = 42
N_TRIALS = 40

train = pd.read_csv(f'{DATA}/train.csv')

for frame_ in (train,):
    frame_['sum_components'] = frame_[['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1, min_count=1)
    frame_['ratio_weekend_daily'] = frame_['weekend_screen_time'] / frame_['daily_screen_time_hours'].replace(0, np.nan)

feats = [c for c in train.columns if c not in ['id', 'addicted_label']]
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
for c in cat_cols:
    train[c] = train[c].astype('category')

X = train[feats]
y = train['addicted_label'].values

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


study = optuna.create_study(direction='maximize', study_name='lgbm_tune',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f'\nBest val AUC: {study.best_value:.5f}')
print('Best params:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')

with open(f'{CONFIGS}/best_params_lgbm.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print(f'Saved: {CONFIGS}/best_params_lgbm.json')
