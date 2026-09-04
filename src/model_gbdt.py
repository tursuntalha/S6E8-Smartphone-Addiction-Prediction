"""GBDT ensemble: LightGBM + XGBoost + CatBoost, each on the same 5-fold split, each
using its own independently-Optuna-tuned hyperparameters (src/features.load_best_params,
searched by experiments/tuning/).
"""
from dataclasses import dataclass, field

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


@dataclass
class GBDTResult:
    oof_lgb: np.ndarray
    oof_xgb: np.ndarray
    oof_cat: np.ndarray
    pred_lgb: np.ndarray
    pred_xgb: np.ndarray
    pred_cat: np.ndarray
    feature_importance: np.ndarray = field(repr=False)


def train_gbdt_ensemble(X, y, X_test, tuned_lgb, tuned_xgb, tuned_cat,
                         seed=42, n_splits=5, device='cuda', fast=False):
    """Trains LightGBM, XGBoost and CatBoost with matching 5-fold splits.

    `device`: 'cuda' uses GPU XGBoost/CatBoost (what this project was developed with);
    anything else falls back to CPU. `fast=True` shrinks n_estimators/early-stopping for
    a quick smoke test — not for a real run.
    """
    Xv = X.values if hasattr(X, 'values') else X
    Xtestv = X_test.values if hasattr(X_test, 'values') else X_test

    n_estimators = 60 if fast else 5000
    early_stopping = 5 if fast else 100

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_idx = list(skf.split(Xv, y))

    params_lgb = dict(objective='binary', metric='auc', n_estimators=n_estimators,
                       verbosity=-1, random_state=seed, **tuned_lgb)
    xgb_device = 'cuda' if device == 'cuda' else 'cpu'
    params_xgb = dict(objective='binary:logistic', eval_metric='auc', n_estimators=n_estimators,
                       tree_method='hist', device=xgb_device, random_state=seed, **tuned_xgb)
    cat_task = 'GPU' if device == 'cuda' else 'CPU'

    oof_lgb, pred_lgb = np.zeros(len(Xv)), np.zeros(len(Xtestv))
    importances = np.zeros(Xv.shape[1])
    for fold, (tr, va) in enumerate(fold_idx):
        m = lgb.LGBMClassifier(**params_lgb)
        m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        oof_lgb[va] = m.predict_proba(Xv[va])[:, 1]
        pred_lgb += m.predict_proba(Xtestv)[:, 1] / len(fold_idx)
        importances += m.booster_.feature_importance(importance_type='gain') / len(fold_idx)
    print(f'LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.5f}')

    # No early stopping here, matching the original tuning/training scripts: XGBoost
    # trains the full n_estimators every fold (unlike LightGBM/CatBoost above).
    oof_xgb, pred_xgb = np.zeros(len(Xv)), np.zeros(len(Xtestv))
    for fold, (tr, va) in enumerate(fold_idx):
        m = xgb.XGBClassifier(**params_xgb)
        m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], verbose=False)
        oof_xgb[va] = m.predict_proba(Xv[va])[:, 1]
        pred_xgb += m.predict_proba(Xtestv)[:, 1] / len(fold_idx)
    print(f'XGBoost OOF AUC: {roc_auc_score(y, oof_xgb):.5f}')

    oof_cat, pred_cat = np.zeros(len(Xv)), np.zeros(len(Xtestv))
    for fold, (tr, va) in enumerate(fold_idx):
        m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', iterations=n_estimators,
                                task_type=cat_task, devices='0' if cat_task == 'GPU' else None,
                                random_seed=seed, verbose=0, **tuned_cat)
        m.fit(Xv[tr], y[tr], eval_set=Pool(Xv[va], y[va]), early_stopping_rounds=early_stopping)
        oof_cat[va] = m.predict_proba(Xv[va])[:, 1]
        pred_cat += m.predict_proba(Xtestv)[:, 1] / len(fold_idx)
    print(f'CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.5f}')

    return GBDTResult(oof_lgb, oof_xgb, oof_cat, pred_lgb, pred_xgb, pred_cat, importances)
