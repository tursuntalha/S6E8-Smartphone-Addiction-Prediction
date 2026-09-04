# Predicting Smartphone Addiction — Kaggle Playground Series S6E8

Binary classification (`addicted_label`, metric: **AUC**) on a synthetically generated
dataset of ~1M rows. Best confirmed public leaderboard score: **AUC = 0.97035**.

## Approach

The pipeline combines a tuned GBDT ensemble with a custom transformer-style neural
network, blended and later stacked with community out-of-fold (OOF) predictions.

1. **Data analysis & generator reverse-engineering** (`src/eda/`) — the competition
   data is synthetic, derived from a small (~7500-row) real dataset. Recovered the
   generator's hard decision rule (`daily_screen_time_hours > 8 OR social_media_hours > 4
   → label = 1`, and the symmetric certain-negative rule) and confirmed train/test come
   from the same distribution (PSI = 0), which makes cross-validation a reliable proxy
   for the leaderboard here.
2. **Feature engineering** (`src/features/`) — target encoding (raw value + smoothed
   TE) on every continuous column, ratio/difference features between usage columns, and
   "ORIG-CDF" features that place each row against the empirical distribution of the
   small original (pre-synthesis) dataset.
3. **GBDT ensemble** (`src/models/gbdt/`) — LightGBM + XGBoost + CatBoost, each
   independently Optuna-tuned, combined by rank averaging.
4. **Neural network** (`src/models/nn/`) — a "Lookup-Transformer": per-column exact-value
   lookup embeddings + learned periodic-linear (Fourier) trend features feed a small
   TransformerEncoder. Trained with **missingness augmentation** (randomly masking
   additional values during training) — the single biggest solo-model gain in the
   project (OOF AUC +0.0014), since the generator produces rows with anywhere from 0 to
   6+ missing fields and the model needs to be robust across that whole range. This
   technique was written up as a public Kaggle Discussion post with a companion
   notebook.
5. **Blending & stacking** (`src/ensembling/`) — GBDT/NN blend weight found by scanning
   the out-of-fold AUC surface (no extra leaderboard submissions spent); the final push
   extended this into a second-level stack over ~100 OOF members pooled from several
   community-shared OOF libraries.

See [`experiments/README.md`](experiments/README.md) for what was tried and rejected
along the way, with the measured result for each.

## Key findings

- **Missingness-augmentation training for tabular NNs** transfers a known
  computer-vision-style regularization idea to tabular data with structural (not
  random) missingness, and produced the largest single improvement in the project.
- **Small-OOF-delta calibration**: deltas in the 0.00005–0.00017 AUC range were
  repeatedly dismissed as noise early on, but were confirmed as real, leaderboard-moving
  signal four separate times once tested. Conversely, gains from adding many
  sparse/high-cardinality joint-encoding features looked real in CV (Δ > 0.0003) but
  reversed sign on the leaderboard — the one clear CV/LB contradiction found in the
  project, caused by fold-specific overfitting on sparse cells.
- **"Improving the NN" and "improving the blend" are different objectives**: every time
  the NN was made more GBDT-like (by handing it GBDT's engineered features) it got
  better on its own but contributed less to the blend, because it became more
  correlated with GBDT. This pattern repeated four times.
- Generator hard-rule reverse engineering, exhaustive pairwise-feature scans, and a
  from-scratch ResNet-style tabular model (built to lower correlation with GBDT) were
  all tried; results and reasoning are kept in `experiments/`.

## Repository structure

```
src/                    Production pipeline, organized by stage
  eda/                  Generator/data analysis (reused throughout the project)
  tuning/               Optuna hyperparameter searches (LightGBM/XGBoost/CatBoost)
  features/             Feature engineering (target encoding, ORIG-CDF, NN data prep)
  models/gbdt/           LightGBM+XGBoost+CatBoost training & blending
  models/nn/             Lookup-Transformer training
  ensembling/            GBDT×NN blends and the final OOF-library stack

experiments/            Tried-and-rejected or superseded directions, kept for the
                         research trail (each subfolder = one line of investigation
                         that did not make it into the final pipeline)
  gbdt_fe/               Rejected/neutral feature-engineering variants (pair-lattice
                         target encoding, decimal-lattice, frequency encoding, ...)
  nn/                    Rejected/superseded NN variants (grouped missingness-aug,
                         hyperparameter tuning, label smoothing, ...)
  resnet/                Attention-free tabular ResNet, built for architecture
                         diversity — net negative, closed
  tabm/                  TabM batch-ensemble — closed (GPU power-limited hardware)
  early_k1/              Early (day 1-2) model-family and augmentation exploration

notebooks/              Four notebooks walking through EDA, feature engineering, the
                         full GBDT pipeline, and ensembling — see notebooks/README.md
configs/                Best hyperparameters found for each GBDT model (JSON)
config.py               Single source of truth for the data/cache/submission/config
                         directory names every script imports (see below)

data/, sub/, nn_cache/  Not tracked in git — raw data, submissions, and cached
                         OOF/model artifacts. Not present in a fresh checkout except
                         data/.placeholder; scripts create sub/ and nn_cache/ as needed.
```

## Path configuration

Every script imports its directory names (`DATA`, `NN_CACHE`, `SUB`, `CONFIGS`) from
[`config.py`](config.py) at the repo root instead of hardcoding them, e.g.:

```python
from config import DATA, SUB
train = pd.read_csv(f'{DATA}/train.csv')
```

To point the pipeline at different locations (e.g. an external drive for `data/`),
edit `config.py` once rather than each script.

## Reproducing

Requires the competition data from Kaggle
([`playground-series-s6e8`](https://www.kaggle.com/competitions/playground-series-s6e8)):
place `train.csv`, `test.csv`, `sample_submission.csv` under `data/` (replacing the
`data/.placeholder` file that keeps the empty folder in git). Scripts are run from the
repository root, e.g.:

```
pip install -r requirements.txt
python src/models/gbdt/lgbm_xgb_cat_ABD_combined_2026-08-19.py
python src/features/lgbm_orig_features_lb_2026-08-21.py
python src/models/nn/nn_train_kfold_missingaug_featfull_2026-08-29.py
python src/ensembling/blend_gbdt_origfeat_nn_featfull_2026-08-29.py
```
