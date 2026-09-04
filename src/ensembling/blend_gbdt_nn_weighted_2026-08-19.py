"""
GBDT (A+B+D, LB=0.96985) + Lookup-Transformer NN (LB=0.96518) AGIRLIKLI rank-average blend'i.
Esit-agirlikli blend (0.96900) mevcut en iyiyi gecemedi - zayif modelin (NN) esit agirligi
gucu asagi cekiyor. Bu sefer GBDT'ye cok daha fazla agirlik veriliyor (0.85/0.15).
"""
import pandas as pd
from scipy.stats import rankdata

W_GBDT = 0.85
W_NN = 0.15

gbdt = pd.read_csv('sub/2026-08-19/lgbm_xgb_cat_ABD_combined_2026-08-19.csv').sort_values('id').reset_index(drop=True)
nn = pd.read_csv('sub/2026-08-19/nn_lookup_transformer_2026-08-19.csv').sort_values('id').reset_index(drop=True)
assert (gbdt['id'].values == nn['id'].values).all()

blend_rank = W_GBDT * rankdata(gbdt['addicted_label']) + W_NN * rankdata(nn['addicted_label'])

sub = pd.DataFrame({'id': gbdt['id'], 'addicted_label': blend_rank / blend_rank.max()})
sub_path = f'sub/2026-08-19/blend_gbdt_nn_w{int(W_GBDT*100)}_{int(W_NN*100)}_2026-08-19.csv'
sub.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
