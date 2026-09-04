import pandas as pd
import numpy as np
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = 'data'
SEED = 42
RUN_NAME = 'lgbm_raw_te_nativecat'
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

# RAW sayisal sutunlar: -999 sentinel (eskisi gibi)
raw_num_tr = train[cont_cols].copy()
raw_num_te = test[cont_cols].copy()
for c in cont_cols:
    for frame in (raw_num_tr, raw_num_te):
        frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(-999)

# RAW kategorik sutunlar: 2026-08-12 DUZELTME - eskiden pd.to_numeric ile hepsi -999'a
# sabitleniyordu (bilgi kaybi, importance=0 idi). Simdi LightGBM native category dtype
# kullaniliyor - train/test kategorileri hizalanip birlestiriliyor (unseen category riski yok).
raw_cat_tr = train[cat_cols].copy()
raw_cat_te = test[cat_cols].copy()
for c in cat_cols:
    raw_cat_tr[c] = raw_cat_tr[c].astype('category')
    raw_cat_te[c] = raw_cat_te[c].astype('category')
    cats = raw_cat_tr[c].cat.categories.union(raw_cat_te[c].cat.categories)
    raw_cat_tr[c] = raw_cat_tr[c].cat.set_categories(cats)
    raw_cat_te[c] = raw_cat_te[c].cat.set_categories(cats)

y = train['addicted_label'].values
prior = y.mean()
te_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

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

Xc = pd.concat([raw_num_tr, raw_cat_tr, enc_tr.add_prefix('te_')], axis=1)
Xc_test = pd.concat([raw_num_te, raw_cat_te, enc_te.add_prefix('te_')], axis=1)
print(f'Özellik sayısı: {Xc.shape[1]} | encoding: {time.time()-t0:.0f}s')

with open('sub/best_params_lgbm.json') as f:
    tuned = json.load(f)
params = dict(objective='binary', metric='auc', n_estimators=5000, verbosity=-1, random_state=SEED, **tuned)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(Xc))
test_pred = np.zeros(len(Xc_test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xc, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(Xc.iloc[tr_idx], y[tr_idx], eval_set=[(Xc.iloc[va_idx], y[va_idx])],
              eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=cat_cols)
    oof[va_idx] = model.predict_proba(Xc.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(Xc_test)[:, 1] / skf.n_splits
    print(f'Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.5f}')

cv_auc = roc_auc_score(y, oof)
print(f'\nCV OOF AUC (raw+TE+native-cat): {cv_auc:.5f}')
print(f'Karşılaştırma -> baseline LGB (v3, SMOOTH=3): 0.96738 | +native-cat: {cv_auc:.5f}')
imp = pd.Series(model.feature_importances_, index=Xc.columns).sort_values(ascending=False)
print('\nHam kategorik sütunların önemi (eskiden 0 idi):')
print(imp[cat_cols].sort_values(ascending=False).to_string())

sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_pred})
sub_path = f'sub/{RUN_NAME}_2026-08-12.csv'
sub.to_csv(sub_path, index=False)
print(f'\nSaved: {sub_path}')
print(f'Elapsed: {time.time()-t0:.0f}s')
