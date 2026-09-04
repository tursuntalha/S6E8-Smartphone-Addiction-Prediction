"""
GBDT (A+B+D kombosu, LB=0.96985) + Lookup-Transformer NN (solo val AUC=0.96346) rank-average
blend'i. Ayni yontem, tum ensemble boyunca kullandigimiz kanitlanmis yontemle (rank-average) -
iki modelin id'ye gore hizalanip rank'lerinin ortalamasi aliniyor.

NOT: bu bir naive/kor blend DEGIL - NN'in kendi OOF/test tahminleri sizintisiz uretildi
(nn_data_prep.py + nn_model.py, train/val split ile), GBDT'nin de kendi 5-fold CV'si var.
Iki modelin gercek korelasyonu test edilip (spearman=0.979, pearson=0.882) genuine
cesitlilik oldugu dogrulandiktan sonra blend deneniyor.
"""
import pandas as pd
from scipy.stats import rankdata

gbdt = pd.read_csv('sub/2026-08-19/lgbm_xgb_cat_ABD_combined_2026-08-19.csv').sort_values('id').reset_index(drop=True)
nn = pd.read_csv('sub/2026-08-19/nn_lookup_transformer_2026-08-19.csv').sort_values('id').reset_index(drop=True)
assert (gbdt['id'].values == nn['id'].values).all()

blend_rank = (rankdata(gbdt['addicted_label']) + rankdata(nn['addicted_label'])) / 2

sub = pd.DataFrame({'id': gbdt['id'], 'addicted_label': blend_rank / blend_rank.max()})
sub_path = 'sub/2026-08-19/blend_gbdt_nn_2026-08-19.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
