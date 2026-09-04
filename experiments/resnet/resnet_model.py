"""
Acik madde #3: ResNet-tarzi tabular NN - attention'siz, sadece residual MLP bloklari.
GBDT'nin 113-ozellik seti (nn_cache/resnet_prepped.npz) uzerinde calisir - dogrudan
StandardScaler'li vektor girisi (Lookup-Transformer'in aksine exact-deger lookup embedding
YOK, tum ozellikler zaten TE/imputed/orig-cdf sayisal degerler). Mimari cesitliligi (GBDT
agac-ensemble vs. bu duz feed-forward residual mimari) hedefleniyor, feature cesitliligi
degil - ayni bilgiye farkli bir cozum yolundan bakmak.
"""
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class ResNetTabular(nn.Module):
    def __init__(self, n_features, hidden_dim=256, n_blocks=5, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(1)
