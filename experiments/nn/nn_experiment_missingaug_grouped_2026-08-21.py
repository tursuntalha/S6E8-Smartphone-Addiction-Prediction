"""
NN deneyi: GRUP-BAZLI (kaskad) missingness-augmentation. Hucre-bazli versiyonun
(mask_prob=0.40, val AUC=0.96648, +0.00141) blend'e katkisi sinirdaydi (+0.00014) cunku
GBDT korelasyonu artmisti (0.9806) - NN GBDT'ye daha "benzer" hale gelmisti, cesitlilik
degil. Hipotez: hucre-bazli maskeleme, turetilmis oran sutunlarini ham sutundan BAGIMSIZ
maskeliyordu - model ham sutun eksikken turetilmis orandan gercek degeri "kacak" olarak
cikarabiliyordu (gercek eksiklikte bu olmaz - biri eksikse turevi de otomatik NaN olur).
Grup-bazli versiyon (augment_missingness_grouped) bu kacagi kapatiyor - PLR_DEPENDENCIES
kurallarina gore turetilmis sutunlarin missing bayragi ham sutunlardan KASKAD olarak
hesaplaniyor, ayrica bagimsiz maskelenmiyor. Kaskad etkisi nedeniyle ayni mask_prob daha
fazla hucreyi etkiliyor olabilir - daha genis bir mask_prob araligi taraniyor.
Mimari/diger hiperparametreler PRODUCTION (nn_model.py) ile AYNI, tek 90/10 split.
Referans (augmentation yok): val AUC=0.96507. Hucre-bazli en iyi: 0.96648 (mask_prob=0.40).
"""
import numpy as np
from torch.utils.data import DataLoader

from nn_common import LookupTransformerNet, LookupDataset, train_model, device

CACHE_DIR = 'nn_cache'
MASK_PROBS = [0.10, 0.20, 0.30, 0.40]
REFERENCE_AUC = 0.96507
CELL_BASED_BEST = 0.96648  # mask_prob=0.40, hucre-bazli

data = np.load(f'{CACHE_DIR}/prepped.npz')
tr_idx, va_idx = data['tr_idx'], data['va_idx']
y = data['y']

train_ds = LookupDataset(
    data['cont_idx_tr'][tr_idx], data['cont_scaled_tr'][tr_idx], data['cont_missing_tr'][tr_idx],
    data['plr_scaled_tr'][tr_idx], data['plr_missing_tr'][tr_idx], data['cat_idx_tr'][tr_idx], y[tr_idx],
)
val_ds = LookupDataset(
    data['cont_idx_tr'][va_idx], data['cont_scaled_tr'][va_idx], data['cont_missing_tr'][va_idx],
    data['plr_scaled_tr'][va_idx], data['plr_missing_tr'][va_idx], data['cat_idx_tr'][va_idx], y[va_idx],
)

results = {}
for mask_prob in MASK_PROBS:
    print(f'\n===== [GRUPLU] mask_prob={mask_prob} =====')
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = LookupTransformerNet(
        lookup_vocab_sizes=data['cont_vocab_sizes'],
        plr_only_count=data['plr_scaled_tr'].shape[1],
        cat_vocab_sizes=data['cat_vocab_sizes'],
    ).to(device)

    model, best_auc = train_model(model, train_loader, val_loader, y[va_idx], epochs=40,
                                   patience=8, missing_aug_prob=mask_prob, missing_aug_grouped=True)
    results[mask_prob] = best_auc
    print(f'[GRUPLU mask_prob={mask_prob}] En iyi val AUC: {best_auc:.5f}  (referans: {REFERENCE_AUC:.5f}, '
          f'delta={best_auc - REFERENCE_AUC:+.5f})')

print('\n===== OZET (gruplu maskeleme) =====')
print(f'Referans (augmentation yok): {REFERENCE_AUC:.5f}')
print(f'Hucre-bazli en iyi (mask_prob=0.40): {CELL_BASED_BEST:.5f}  (delta={CELL_BASED_BEST-REFERENCE_AUC:+.5f})')
for mp, auc in results.items():
    print(f'[GRUPLU] mask_prob={mp}: {auc:.5f}  (delta vs referans={auc - REFERENCE_AUC:+.5f}, '
          f'delta vs hucre-bazli={auc - CELL_BASED_BEST:+.5f})')
