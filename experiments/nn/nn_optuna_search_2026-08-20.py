"""
Lookup-Transformer NN icin ILK KEZ sistematik hiperparametre taramasi (Optuna). GBDT gun-1'de
40 Optuna denemesi gormustu, NN hic gormedi - butun mimari/lr/dropout/batch_size elle
secilmisti. Motivasyon: referans notebook'ta AYNI mimari fikri (Lookup-Transformer)
tamerlanomralinov'un elinde solo LB=0.97041, 74-model notebook'unun kendi foldunda retrain'i
OOF=0.96853 veriyor - bizim NN'imiz sadece OOF=0.96576. Bu ~0.003-0.005 AUC'lik acik,
bugun denenen tum feature-engineering denemelerinden (hepsi <0.0003) kat kat buyuk.

Hiz icin: nn_data_prep.py'nin tek 90/10 split'i uzerinde arama (nn_cache/prepped.npz),
sinirli epoch (18) + patience (4). En iyi konfigurasyon bulununca k-fold ile tam dogrulanir
(ayri script).

Cikti: nn_cache/optuna_study.db (SQLite, ilerlemeyi kalici tutar), nn_cache/optuna_best_params.json
"""
import json
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, train_model, device

SEED = 42
import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR
N_TRIALS = 20
SEARCH_EPOCHS = 18
SEARCH_PATIENCE = 4

np.random.seed(SEED)
torch.manual_seed(SEED)
print('device:', device)

data = np.load(f'{CACHE_DIR}/prepped.npz')
tr_idx, va_idx = data['tr_idx'], data['va_idx']
y = data['y']


def objective(trial):
    d_token = trial.suggest_categorical('d_token', [32, 64, 96, 128, 160])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_freq = trial.suggest_categorical('n_freq', [4, 8, 16])
    dropout = trial.suggest_float('dropout', 0.05, 0.35)
    n_head_blocks = trial.suggest_int('n_head_blocks', 1, 3)
    base_lr = trial.suggest_float('base_lr', 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
    warmup_epochs = trial.suggest_categorical('warmup_epochs', [0, 2, 4])

    torch.manual_seed(SEED)
    train_ds = LookupDataset(
        data['cont_idx_tr'][tr_idx], data['cont_scaled_tr'][tr_idx], data['cont_missing_tr'][tr_idx],
        data['plr_scaled_tr'][tr_idx], data['plr_missing_tr'][tr_idx], data['cat_idx_tr'][tr_idx], y[tr_idx],
    )
    val_ds = LookupDataset(
        data['cont_idx_tr'][va_idx], data['cont_scaled_tr'][va_idx], data['cont_missing_tr'][va_idx],
        data['plr_scaled_tr'][va_idx], data['plr_missing_tr'][va_idx], data['cat_idx_tr'][va_idx], y[va_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = LookupTransformerNet(
        lookup_vocab_sizes=data['cont_vocab_sizes'],
        plr_only_count=data['plr_scaled_tr'].shape[1],
        cat_vocab_sizes=data['cat_vocab_sizes'],
        d_token=d_token, n_heads=8, n_layers=n_layers, n_freq=n_freq,
        dropout=dropout, n_head_blocks=n_head_blocks,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    bce = torch.nn.BCEWithLogitsLoss()
    best_auc, no_improve = 0.0, 0

    from sklearn.metrics import roc_auc_score
    from nn_common import predict_probabilities

    for epoch in range(SEARCH_EPOCHS):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for g in opt.param_groups:
                g['lr'] = warmup_lr

        model.train()
        for lookup_idx, lookup_scaled, lookup_missing, plr_scaled, plr_missing, cat_idx, yb in train_loader:
            lookup_idx, lookup_scaled = lookup_idx.to(device), lookup_scaled.to(device)
            lookup_missing = lookup_missing.to(device)
            plr_scaled, plr_missing = plr_scaled.to(device), plr_missing.to(device)
            cat_idx, yb = cat_idx.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(lookup_idx, lookup_scaled, lookup_missing, plr_scaled, plr_missing, cat_idx)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()

        val_preds = predict_probabilities(model, val_loader)
        val_auc = roc_auc_score(y[va_idx], val_preds)
        if not (warmup_epochs > 0 and epoch < warmup_epochs):
            sched.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= SEARCH_PATIENCE:
                break

        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_auc


if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        storage=f'sqlite:///{CACHE_DIR}/optuna_study.db',
        study_name='nn_lookup_transformer_2026-08-20',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=6),
    )
    print(f'Referans (mevcut production, kfold): OOF AUC=0.96576')
    print(f'Referans (mevcut production, 90/10 split): val AUC=0.96507')
    study.optimize(objective, n_trials=N_TRIALS)

    print('\n' + '=' * 50)
    print(f'En iyi val AUC: {study.best_value:.5f}')
    print(f'En iyi parametreler: {json.dumps(study.best_params, indent=2)}')

    with open(f'{CACHE_DIR}/optuna_best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f'Saved: {CACHE_DIR}/optuna_best_params.json')

    print('\nTum denemeler (deger sirali):')
    trials_df = study.trials_dataframe().sort_values('value', ascending=False)
    print(trials_df[['number', 'value', 'state']].head(10).to_string(index=False))
