"""
Yapay sinir agi pipeline'i - BOLUM 3: Submission.
nn_data_prep.py'nin uretttigi test array'lerini + nn_model.py'nin egittigi checkpoint'i
okuyup test tahmini uretir, sub/ altina submission csv'si yazar.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, predict_probabilities, device

import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR, SUB
SUB_PATH = f'{SUB}/2026-08-19/nn_lookup_transformer_v2_2026-08-19.csv'

data = np.load(f'{CACHE_DIR}/prepped.npz')

model = LookupTransformerNet(
    lookup_vocab_sizes=data['cont_vocab_sizes'],
    plr_only_count=data['plr_scaled_tr'].shape[1],
    cat_vocab_sizes=data['cat_vocab_sizes'],
).to(device)
model.load_state_dict(torch.load(f'{CACHE_DIR}/best_model.pt', map_location=device))

test_ds = LookupDataset(
    data['cont_idx_te'], data['cont_scaled_te'], data['cont_missing_te'],
    data['plr_scaled_te'], data['plr_missing_te'], data['cat_idx_te'],
)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

test_preds = predict_probabilities(model, test_loader)

sub = pd.DataFrame({'id': data['test_id'], 'addicted_label': test_preds})
sub.to_csv(SUB_PATH, index=False)
print(f'Saved: {SUB_PATH}')
