"""
TabM (resmi 'tabm' pip paketi, Yandex Research) - GBDT/NN ile AYNI 5-fold OOF uzerinde
egitim. Girdi: nn_cache/tabm_prepped.npz (ABD 84 ozellik, standardize edilmis, GBDT ile
birebir ayni StratifiedKFold(n_splits=5, seed=42) fold atamasi).
TabM mimarisi: k=32 (varsayilan) paylasimli-agirlikli MLP ensemble'i (BatchEnsemble tarzi,
d_block=512, n_blocks=3, dropout=0.1) - tek egitimde k MLP'nin etkisini veriyor, GBDT
(agac) ve Lookup-Transformer'imizden (attention) MEKANIK OLARAK farkli bir ucuncu aile.

Cikti:
  nn_cache/tabm_model_fold{K}.pt
  nn_cache/tabm_oof.npy, nn_cache/tabm_test_pred.npy
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import tabm

SEED = 42
import sys, os
sys.path.insert(0, os.getcwd())
from config import NN_CACHE as CACHE_DIR
N_FOLDS = 5
SMOKE_TEST = '--smoke' in sys.argv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(SEED)
torch.manual_seed(SEED)
print('device:', device, 'smoke_test:', SMOKE_TEST)


def make_model(n_features):
    return tabm.TabM.make(n_num_features=n_features, cat_cardinalities=None, d_out=1).to(device)


def train_one_fold(X_tr, y_tr, X_va, y_va, epochs, patience=5, verbose=True):
    model = make_model(X_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    bce = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    X_va_t = torch.tensor(X_va, device=device)

    best_auc, best_state, no_improve = 0.0, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(x_num=xb).squeeze(-1)  # (B, k)
            k = logits.shape[1]
            loss = bce(logits, yb.unsqueeze(1).expand(-1, k))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(x_num=X_va_t).squeeze(-1)  # (B_va, k)
            va_pred = torch.sigmoid(va_logits).mean(dim=1).cpu().numpy()
        va_auc = roc_auc_score(y_va, va_pred)
        sched.step(va_auc)
        if verbose:
            print(f'  Epoch {epoch+1}/{epochs}  val AUC: {va_auc:.5f}  lr={opt.param_groups[0]["lr"]:.2e}')

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print('  Early stopping.')
                break

    model.load_state_dict(best_state)
    return model, best_auc


if __name__ == '__main__':
    t_start = time.time()
    data = np.load(f'{CACHE_DIR}/tabm_prepped.npz')
    X, X_test, y, fold_id = data['X'], data['X_test'], data['y'], data['fold_id']
    n_train, n_test = len(y), X_test.shape[0]

    if SMOKE_TEST:
        print('=== SMOKE TEST: fold 0, 3 epoch ===')
        tr_idx = np.where(fold_id != 0)[0]
        va_idx = np.where(fold_id == 0)[0]
        t1 = time.time()
        model, auc = train_one_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], epochs=3, patience=99)
        print(f'Smoke test tamam: {time.time()-t1:.0f}s, val AUC(3 epoch)={auc:.5f}')
        sys.exit(0)

    oof = np.zeros(n_train, dtype=np.float64)
    test_pred_sum = np.zeros(n_test, dtype=np.float64)
    fold_times = []

    for fold in range(N_FOLDS):
        t_fold = time.time()
        tr_idx = np.where(fold_id != fold)[0]
        va_idx = np.where(fold_id == fold)[0]
        print(f'\n===== Fold {fold+1}/{N_FOLDS}  (train={len(tr_idx)}  val={len(va_idx)}) '
              f'=====  toplam gecen: {(t_fold - t_start)/60:.1f} dk')

        model, best_auc = train_one_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], epochs=40, patience=6)

        model.eval()
        with torch.no_grad():
            va_logits = model(x_num=torch.tensor(X[va_idx], device=device)).squeeze(-1)
            oof[va_idx] = torch.sigmoid(va_logits).mean(dim=1).cpu().numpy()
            te_logits = model(x_num=torch.tensor(X_test, device=device)).squeeze(-1)
            test_pred_sum += torch.sigmoid(te_logits).mean(dim=1).cpu().numpy()

        torch.save(model.state_dict(), f'{CACHE_DIR}/tabm_model_fold{fold}.pt')
        np.save(f'{CACHE_DIR}/tabm_oof_partial.npy', oof)

        dt = time.time() - t_fold
        fold_times.append(dt)
        avg_fold_time = sum(fold_times) / len(fold_times)
        remaining = avg_fold_time * (N_FOLDS - fold - 1)
        print(f'--- Fold {fold+1} bitti: val AUC={best_auc:.5f}  sure={dt/60:.1f} dk  '
              f'ortalama fold suresi={avg_fold_time/60:.1f} dk  tahmini kalan sure={remaining/60:.1f} dk ---')

    test_pred = test_pred_sum / N_FOLDS
    oof_auc = roc_auc_score(y, oof)
    total_min = (time.time() - t_start) / 60
    print(f'\n===== TAMAMLANDI: 5-fold TabM OOF AUC = {oof_auc:.5f}  toplam sure={total_min:.1f} dk =====')

    np.save(f'{CACHE_DIR}/tabm_oof.npy', oof)
    np.save(f'{CACHE_DIR}/tabm_test_pred.npy', test_pred)
    print(f'Saved: {CACHE_DIR}/tabm_oof.npy, {CACHE_DIR}/tabm_test_pred.npy')
