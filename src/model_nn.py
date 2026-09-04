"""The Lookup-Transformer: a small TransformerEncoder over per-column tokens, trained
with missingness augmentation (the single largest solo-model gain found in this
project — see the main README's "Key findings").

Two kinds of continuous-column tokens:
  - "lookup" columns (the 9 raw usage columns): exact-value embedding + PLR
    (periodic-linear / learned-Fourier) embedding, summed
  - "plr_only" columns (every derived ratio/decimal-lattice/ORIG-CDF feature): PLR only
    — these are far too high-cardinality for an exact-value lookup table to be
    meaningful
Categorical columns get a plain lookup embedding. A learned CLS token is prepended and
the pooled representation goes through a small residual-MLP head.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


class LookupDataset(Dataset):
    def __init__(self, lookup_idx, lookup_scaled, lookup_missing,
                 plr_only_scaled, plr_only_missing, cat_idx, y=None):
        self.lookup_idx = torch.tensor(lookup_idx, dtype=torch.long)
        self.lookup_scaled = torch.tensor(lookup_scaled, dtype=torch.float32)
        self.lookup_missing = torch.tensor(lookup_missing, dtype=torch.float32)
        self.plr_only_scaled = torch.tensor(plr_only_scaled, dtype=torch.float32)
        self.plr_only_missing = torch.tensor(plr_only_missing, dtype=torch.float32)
        self.cat_idx = torch.tensor(cat_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.lookup_idx)

    def __getitem__(self, i):
        item = (self.lookup_idx[i], self.lookup_scaled[i], self.lookup_missing[i],
                self.plr_only_scaled[i], self.plr_only_missing[i], self.cat_idx[i])
        return item + (self.y[i],) if self.y is not None else item


class PLREmbedding(nn.Module):
    """Periodic-Linear-Ratio embedding: a small learned-frequency Fourier feature map
    followed by a linear projection, used for every continuous token."""

    def __init__(self, n_frequencies, d_token):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 0.1)
        self.linear = nn.Linear(2 * n_frequencies, d_token)

    def forward(self, x):
        v = x.unsqueeze(-1) * self.frequencies.unsqueeze(0) * 2 * math.pi
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        return torch.relu(self.linear(v))


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, features):
        residual = features
        features = self.linear1(features)
        features = self.activation(features)
        features = self.dropout(features)
        features = self.linear2(features)
        features = features + residual
        features = self.batch_norm(features)
        return self.activation(features)


class LookupTransformerNet(nn.Module):
    def __init__(self, lookup_vocab_sizes, plr_only_count, cat_vocab_sizes,
                 d_token=64, n_heads=8, n_layers=2, n_freq=8, dropout=0.1, n_head_blocks=2):
        super().__init__()
        self.cont_lookup = nn.ModuleList([nn.Embedding(int(v), d_token) for v in lookup_vocab_sizes])
        self.cont_plr = nn.ModuleList([PLREmbedding(n_freq, d_token) for _ in lookup_vocab_sizes])
        self.plr_only = nn.ModuleList([PLREmbedding(n_freq, d_token) for _ in range(plr_only_count)])
        self.cat_lookup = nn.ModuleList([nn.Embedding(int(v), d_token) for v in cat_vocab_sizes])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        head_layers = [nn.Linear(d_token, d_token), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_head_blocks):
            head_layers.append(ResidualBlock(d_token, dropout))
        head_layers.append(nn.Linear(d_token, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, lookup_idx, lookup_scaled, lookup_missing, plr_only_scaled, plr_only_missing, cat_idx):
        batch = cat_idx.shape[0]
        tokens = []
        for j, (lookup, plr) in enumerate(zip(self.cont_lookup, self.cont_plr)):
            not_missing = (1.0 - lookup_missing[:, j]).unsqueeze(-1)
            tokens.append(lookup(lookup_idx[:, j]) + plr(lookup_scaled[:, j]) * not_missing)
        for j, plr in enumerate(self.plr_only):
            not_missing = (1.0 - plr_only_missing[:, j]).unsqueeze(-1)
            tokens.append(plr(plr_only_scaled[:, j]) * not_missing)
        for j, emb in enumerate(self.cat_lookup):
            tokens.append(emb(cat_idx[:, j]))

        cls = self.cls_token.expand(batch, -1, -1)
        seq = torch.cat([cls, torch.stack(tokens, dim=1)], dim=1)
        out = self.transformer(seq)
        return self.head(out[:, 0, :]).squeeze(1)


def augment_missingness(lookup_idx, lookup_missing, plr_missing, mask_prob):
    """Randomly hides additional (already-observed) values during training, on top of
    whatever's genuinely missing. Motivated by GBDT's native NaN-handling being far more
    robust to missing data than the NN was without this (full-data gap +0.0025 AUC,
    4+-missing gap +0.0068) — exposing the NN to more/varied missingness combinations
    during training closes most of that gap. This was the single largest solo-NN
    improvement found in the project (see the main README)."""
    if mask_prob <= 0:
        return lookup_idx, lookup_missing, plr_missing
    dev = lookup_idx.device

    cont_rand = torch.rand(lookup_missing.shape, device=dev)
    extra_cont = (cont_rand < mask_prob) & (lookup_missing < 0.5)
    lookup_idx = lookup_idx.clone()
    lookup_idx[extra_cont] = 0
    lookup_missing = lookup_missing.clone()
    lookup_missing[extra_cont] = 1.0

    plr_rand = torch.rand(plr_missing.shape, device=dev)
    extra_plr = (plr_rand < mask_prob) & (plr_missing < 0.5)
    plr_missing = plr_missing.clone()
    plr_missing[extra_plr] = 1.0

    return lookup_idx, lookup_missing, plr_missing


def predict_probabilities(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            if len(batch) == 7:
                batch = batch[:6]
            logits = model(*batch)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(model, train_loader, val_loader, y_val, device, epochs=40, base_lr=1e-3,
                    patience=8, missing_aug_prob=0.40, verbose=True):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    bce = nn.BCEWithLogitsLoss()
    best_auc, best_state, no_improve = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        for lookup_idx, lookup_scaled, lookup_missing, plr_scaled, plr_missing, cat_idx, yb in train_loader:
            lookup_idx, lookup_scaled = lookup_idx.to(device), lookup_scaled.to(device)
            lookup_missing = lookup_missing.to(device)
            plr_scaled, plr_missing = plr_scaled.to(device), plr_missing.to(device)
            cat_idx, yb = cat_idx.to(device), yb.to(device)

            if missing_aug_prob > 0:
                lookup_idx, lookup_missing, plr_missing = augment_missingness(
                    lookup_idx, lookup_missing, plr_missing, missing_aug_prob)

            opt.zero_grad()
            logits = model(lookup_idx, lookup_scaled, lookup_missing, plr_scaled, plr_missing, cat_idx)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()

        val_preds = predict_probabilities(model, val_loader, device)
        val_auc = roc_auc_score(y_val, val_preds)
        sched.step(val_auc)
        if verbose:
            print(f'  epoch {epoch + 1}/{epochs}  val AUC: {val_auc:.5f}  lr={opt.param_groups[0]["lr"]:.2e}')

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print('  early stopping.')
                break

    model.load_state_dict(best_state)
    return model, best_auc


def train_nn_kfold(nn_data, device='cuda', n_splits=5, mask_prob=0.40, epochs=40, patience=8,
                    batch_size=2048, fast=False, verbose=True):
    """Trains one Lookup-Transformer per fold of `nn_data` (src.features.build_nn_arrays'
    output; folds come from `nn_data['fold_id']`, built with the same seed/split as the
    GBDT ensemble so the two OOF vectors line up row-for-row). Returns (oof, test_pred).
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    y = nn_data['y']
    n_train = len(y)
    n_test = len(nn_data['test_id'])
    if fast:
        epochs, patience = 2, 2

    oof = np.zeros(n_train, dtype=np.float64)
    test_pred_sum = np.zeros(n_test, dtype=np.float64)

    test_ds = LookupDataset(
        nn_data['cont_idx_te'], nn_data['cont_scaled_te'], nn_data['cont_missing_te'],
        nn_data['plr_scaled_te'], nn_data['plr_missing_te'], nn_data['cat_idx_te'],
    )
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    for fold in range(n_splits):
        tr_idx = np.where(nn_data['fold_id'] != fold)[0]
        va_idx = np.where(nn_data['fold_id'] == fold)[0]
        if verbose:
            print(f'\n--- NN fold {fold + 1}/{n_splits} (train={len(tr_idx)}  val={len(va_idx)}) ---')

        train_ds = LookupDataset(
            nn_data['cont_idx_tr'][tr_idx], nn_data['cont_scaled_tr'][tr_idx], nn_data['cont_missing_tr'][tr_idx],
            nn_data['plr_scaled_tr'][tr_idx], nn_data['plr_missing_tr'][tr_idx], nn_data['cat_idx_tr'][tr_idx],
            y[tr_idx],
        )
        val_ds = LookupDataset(
            nn_data['cont_idx_tr'][va_idx], nn_data['cont_scaled_tr'][va_idx], nn_data['cont_missing_tr'][va_idx],
            nn_data['plr_scaled_tr'][va_idx], nn_data['plr_missing_tr'][va_idx], nn_data['cat_idx_tr'][va_idx],
            y[va_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

        model = LookupTransformerNet(
            lookup_vocab_sizes=nn_data['cont_vocab_sizes'],
            plr_only_count=nn_data['plr_scaled_tr'].shape[1],
            cat_vocab_sizes=nn_data['cat_vocab_sizes'],
        )
        model, best_auc = train_one_fold(model, train_loader, val_loader, y[va_idx], device,
                                          epochs=epochs, patience=patience,
                                          missing_aug_prob=mask_prob, verbose=verbose)
        oof[va_idx] = predict_probabilities(model, val_loader, device)
        test_pred_sum += predict_probabilities(model, test_loader, device)
        print(f'--- NN fold {fold + 1} done: val AUC={best_auc:.5f} ---')

    test_pred = test_pred_sum / n_splits
    print(f'\nNN {n_splits}-fold OOF AUC: {roc_auc_score(y, oof):.5f}')
    return oof, test_pred
