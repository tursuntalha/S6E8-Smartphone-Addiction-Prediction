"""
Lookup-Transformer deneyleri icin PAYLASILAN bilesenler (nn_model.py'nin production
checkpoint'ini BOZMAMAK icin ayri tutuldu). 3 izole hiperparametre/mimari deneyinde
(nn_experiment_features.py, nn_experiment_capacity.py, nn_experiment_labelsmooth.py)
kullaniliyor.

LookupTransformerNet burada iki tur surekli sutunu destekliyor:
  - "lookup" sutunlar: exact-deger embedding + PLR (ham sutunlar, jenerator lookup-table
    arketipine sahip)
  - "plr_only" sutunlar: sadece PLR (turetilmis oran/fark feature'lari - kardinalite cok
    yuksek/seyrek, exact-deger lookup tablosu anlamsiz)
Baseline deneylerinde (capacity, labelsmooth) plr_only sutun sayisi 0.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        if self.y is not None:
            return item + (self.y[i],)
        return item


class PLREmbedding(nn.Module):
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
    """Egitim sirasinda rastgele EK deger maskeleme (zaten eksik olmayan hucreleri de
    eksik gibi gostermek). GBDT'nin eksik-veriye NN'den daha dayanikli olmasi (tam veri
    +0.0025, 4+ eksik +0.0068 fark, 2026-08-20 bulgusu) hipoteziyle motive: NN'i egitim
    sirasinda daha cok/cesitli eksik-kombinasyonuna maruz birakmak. Sadece cont-lookup ve
    plr-only akislarina uygulanir (kategorik zaten kendi 'missing' kategorisine sahip).
    lookup_idx guncellenir (0=missing index'ine), missing bayraklari 1.0'a cekilir - bu
    forward()'daki not_missing carpanini otomatik olarak dogru sekilde sifirlar.
    """
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


# CONT_COLS/PLR_ONLY_COLS sirasi nn_data_prep.py ile AYNI olmak zorunda (index eslemesi buna gore).
CONT_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
             'work_study_hours', 'sleep_hours', 'notifications_per_day',
             'app_opens_per_day', 'weekend_screen_time']
PLR_ONLY_COLS = ['sum_components', 'ratio_weekend_daily', 'ratio_screen_sleep', 'ratio_social_daily',
                  'ratio_gaming_daily', 'ratio_work_daily', 'diff_weekend_daily', 'max_activity3']
_CONT_IDX = {name: i for i, name in enumerate(CONT_COLS)}
_PLR_IDX = {name: i for i, name in enumerate(PLR_ONLY_COLS)}

# Her turetilmis (plr-only) sutunun hangi ham sutunlara bagimli oldugu + gercek pandas
# semantiginin NaN kurali: 'any' = bagimli sutunlardan HERHANGI biri eksikse NaN (bolme/fark
# islemleri, nn_data_prep.py'deki ratio_*/diff_* formulleri), 'all' = TUMU eksikse NaN
# (sum_components, .sum(min_count=1) skipna davranisi geregi tek sutun eksik olsa da toplanir).
PLR_DEPENDENCIES = {
    'ratio_weekend_daily': (['weekend_screen_time', 'daily_screen_time_hours'], 'any'),
    'ratio_screen_sleep':  (['daily_screen_time_hours', 'sleep_hours'], 'any'),
    'ratio_social_daily':  (['social_media_hours', 'daily_screen_time_hours'], 'any'),
    'ratio_gaming_daily':  (['gaming_hours', 'daily_screen_time_hours'], 'any'),
    'ratio_work_daily':    (['work_study_hours', 'daily_screen_time_hours'], 'any'),
    'diff_weekend_daily':  (['weekend_screen_time', 'daily_screen_time_hours'], 'any'),
    'max_activity3':       (['social_media_hours', 'gaming_hours', 'work_study_hours'], 'any'),
    'sum_components':      (['social_media_hours', 'gaming_hours', 'work_study_hours'], 'all'),
}


def augment_missingness_grouped(lookup_idx, lookup_missing, plr_missing, mask_prob):
    """augment_missingness()'in GRUP-BAZLI versiyonu (2026-08-21, korelasyon-kirma denemesi).
    Sorun: hucre-bazli bagimsiz maskeleme, gercek eksiklik yapisina aykiri bir ornuntu
    ogretiyor - gercekte 'daily_screen_time_hours' eksikse ondan tureyen TUM oranlar da
    (ratio_social_daily vb.) otomatik NaN olur (pandas bolme/fark islemi geregi), ama bizim
    eski augment_missingness() bunlari BIRBIRINDEN BAGIMSIZ rastgele maskeliyordu - model
    ham sutun maskeliyken turetilmis oranindan gercek degeri 'kacak' olarak cikarabiliyordu.
    Bu fonksiyon: ham sutunlari eskisi gibi hucre-bazli maskeler, SONRA turetilmis sutunlarin
    missing bayragini PLR_DEPENDENCIES'teki gercek bagimlilik kuraliyla (any/all) KASKAD
    olarak hesaplar - turetilmis sutunlara ayrica bagimsiz rastgele maskeleme UYGULANMAZ.
    Amac: NN'i GBDT'den daha farkli bir cozum yoluna zorlamak (blend cesitliligi), sadece
    NN kalitesini artirmak degil (eski versiyonun blend'e katkisi sinirda kalmisti - GBDT
    korelasyonu artmisti, 0.9806).
    """
    if mask_prob <= 0:
        return lookup_idx, lookup_missing, plr_missing
    dev = lookup_idx.device

    cont_rand = torch.rand(lookup_missing.shape, device=dev)
    extra_cont = (cont_rand < mask_prob) & (lookup_missing < 0.5)
    lookup_idx = lookup_idx.clone()
    lookup_idx[extra_cont] = 0
    lookup_missing = lookup_missing.clone()
    lookup_missing[extra_cont] = 1.0

    plr_missing = plr_missing.clone()
    for plr_name, (deps, rule) in PLR_DEPENDENCIES.items():
        k = _PLR_IDX[plr_name]
        dep_idxs = [_CONT_IDX[d] for d in deps]
        dep_missing = lookup_missing[:, dep_idxs]
        if rule == 'any':
            propagate = dep_missing.max(dim=1).values > 0.5
        else:
            propagate = dep_missing.min(dim=1).values > 0.5
        plr_missing[:, k] = torch.maximum(plr_missing[:, k], propagate.float())

    return lookup_idx, lookup_missing, plr_missing


def predict_probabilities(model, loader):
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


def train_model(model, train_loader, val_loader, y_val, epochs=25, base_lr=1e-3,
                 warmup_epochs=0, label_smoothing=0.0, patience=5, verbose=True,
                 missing_aug_prob=0.0, missing_aug_grouped=False):
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    bce = nn.BCEWithLogitsLoss()
    best_auc, best_state, no_improve = 0.0, None, 0

    for epoch in range(epochs):
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

            if missing_aug_prob > 0:
                aug_fn = augment_missingness_grouped if missing_aug_grouped else augment_missingness
                lookup_idx, lookup_missing, plr_missing = aug_fn(
                    lookup_idx, lookup_missing, plr_missing, missing_aug_prob)

            opt.zero_grad()
            logits = model(lookup_idx, lookup_scaled, lookup_missing, plr_scaled, plr_missing, cat_idx)
            if label_smoothing > 0:
                yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing
                loss = bce(logits, yb_smooth)
            else:
                loss = bce(logits, yb)
            loss.backward()
            opt.step()

        val_preds = predict_probabilities(model, val_loader)
        val_auc = roc_auc_score(y_val, val_preds)
        if not (warmup_epochs > 0 and epoch < warmup_epochs):
            sched.step(val_auc)
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}  val AUC: {val_auc:.5f}  lr={opt.param_groups[0]["lr"]:.2e}')

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print('Early stopping.')
                break

    model.load_state_dict(best_state)
    return model, best_auc
