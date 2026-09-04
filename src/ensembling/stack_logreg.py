import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import sys, os
sys.path.insert(0, os.getcwd())
from config import SUB

SEED = 42

y = np.load(f'{SUB}/blend_y.npy')
oof_lgb = np.load(f'{SUB}/oof_lgb.npy'); pred_lgb = np.load(f'{SUB}/pred_lgb.npy')
oof_xgb = np.load(f'{SUB}/oof_xgb.npy'); pred_xgb = np.load(f'{SUB}/pred_xgb.npy')
oof_cat = np.load(f'{SUB}/oof_cat.npy'); pred_cat = np.load(f'{SUB}/pred_cat.npy')
test_id = np.load(f'{SUB}/blend_test_id.npy')

# Rank-average referans (blend_lgb_xgb_cat.py ile ayni deger cikmali)
oof_rank = (rankdata(oof_lgb) + rankdata(oof_xgb) + rankdata(oof_cat)) / 3
print(f'Rank-average OOF AUC (referans): {roc_auc_score(y, oof_rank):.5f}')

# Meta-ozellikler: 3 modelin OOF/test olasiliklari (rank-normalize edilmis, ayni olcek)
Xm = np.column_stack([rankdata(oof_lgb), rankdata(oof_xgb), rankdata(oof_cat)]) / len(y)
Xm_test = np.column_stack([
    rankdata(pred_lgb) / len(pred_lgb),
    rankdata(pred_xgb) / len(pred_xgb),
    rankdata(pred_cat) / len(pred_cat),
])

# Meta-model icin AYNI 5-fold split (base modellerle ayni seed) -> gercek OOF stacking,
# meta-modelin kendi egitim satirlarina bakmadan tahmin uretmesini saglar.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
meta_oof = np.zeros(len(y))
meta_test = np.zeros(len(test_id))
coefs = []
for tr_idx, va_idx in skf.split(Xm, y):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xm[tr_idx], y[tr_idx])
    meta_oof[va_idx] = lr.predict_proba(Xm[va_idx])[:, 1]
    meta_test += lr.predict_proba(Xm_test)[:, 1] / skf.n_splits
    coefs.append(lr.coef_[0])

stack_auc = roc_auc_score(y, meta_oof)
print(f'Stacking (logistic, OOF) AUC: {stack_auc:.5f}')
print(f'Karşılaştırma -> rank-average: {roc_auc_score(y, oof_rank):.5f} | stacking: {stack_auc:.5f}')
print(f'Ortalama katsayılar (lgb, xgb, cat): {np.mean(coefs, axis=0).round(3)}')

sub = pd.DataFrame({'id': test_id, 'addicted_label': meta_test})
sub_path = f'{SUB}/lgbm_xgb_cat_stack_2026-08-12.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
