# Experiments

Scripts here were tried and did **not** make it into the final pipeline (`src/`) —
either the result was neutral/negative, or a later script in `src/` superseded them.
They're kept so the research trail is reproducible. Full narrative and numbers for each
are in [`../docs/gunluk.md`](../docs/gunluk.md).

- **`gbdt_fe/`** — feature-engineering variants tested against the GBDT baseline and
  rejected or found neutral: pair-lattice / dense pair-lattice target encoding (CV
  looked good, leaderboard reversed — sparse-cell overfitting), decimal-lattice
  features, frequency encoding, native-categorical CatBoost (clear regression),
  generator hard-rule flags, categorical-triple joint encoding, imputation variants,
  linear-score features, and earlier raw+TE iterations superseded by later ones.
- **`nn/`** — neural-net variants that underperformed or were superseded: capacity
  scaling, label smoothing, Optuna hyperparameter search (never closed the gap to
  external reference implementations), grouped/cascaded missingness augmentation
  (hypothesis about derived-column leakage was disproved), and the pre-"featfull"
  data prep / training scripts.
- **`resnet/`** — an attention-free residual-MLP tabular network, built specifically to
  get a model less correlated with the GBDT ensemble than the Transformer NN. It
  achieved lower correlation as intended, but solo quality was too low to earn any
  weight in the blend.
- **`tabm/`** — TabM (batch-ensemble tabular model). Underperformed on this hardware
  (GPU power-capped laptop), not pursued further.
- **`early_k1/`** — day 1-2 exploration: alternative model families (Random Forest,
  Extra Trees, HistGB), multi-seed bagging, focal loss, class-imbalance handling,
  pseudo-labeling.
