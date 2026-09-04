# Predicting Smartphone Addiction — Kaggle Playground Series S6E8

Kaggle competition: [Playground Series S6E8 — Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)

Binary classification (`addicted_label`, metric: **AUC**) on a synthetically generated
dataset of ~1M rows.

| Leaderboard | AUC |
|---|---|
| Public | 0.97035 |
| Private | 0.97082 |

## Approach

The pipeline combines a tuned GBDT ensemble with a custom transformer-style neural
network, blended and later stacked with community out-of-fold (OOF) predictions.

1. **Data analysis & generator reverse-engineering** (`notebooks/01_eda.ipynb`,
   `experiments/eda/`) — the competition data is synthetic, derived from a small
   (~7500-row) real dataset. Recovered the generator's hard decision rule
   (`daily_screen_time_hours > 8 OR social_media_hours > 4 → label = 1`, and the
   symmetric certain-negative rule) and confirmed train/test come from the same
   distribution (PSI = 0), which makes cross-validation a reliable proxy for the
   leaderboard here.
2. **Feature engineering** (`src/features.py`) — target encoding (raw value + smoothed
   TE) on every continuous column, ratio/difference features between usage columns, and
   "ORIG-CDF" features that place each row against the empirical distribution of the
   small original (pre-synthesis) dataset.
3. **GBDT ensemble** (`src/model_gbdt.py`) — LightGBM + XGBoost + CatBoost, each
   independently Optuna-tuned (`experiments/tuning/`), combined by rank averaging.
4. **Neural network** (`src/model_nn.py`) — a "Lookup-Transformer": per-column exact-value
   lookup embeddings + learned periodic-linear (Fourier) trend features feed a small
   TransformerEncoder. Trained with **missingness augmentation** (randomly masking
   additional values during training) — the single biggest solo-model gain in the
   project (OOF AUC +0.0014), since the generator produces rows with anywhere from 0 to
   6+ missing fields and the model needs to be robust across that whole range. This
   technique was written up as a public Kaggle Discussion post with a companion
   notebook.
5. **Blending** (`src/main.py`, using `src/utils.py`'s weight search) — GBDT/NN blend
   weight found by scanning the out-of-fold AUC surface (no extra leaderboard
   submissions spent). A further, more involved push stacked this blend with ~100 OOF
   members pooled from several community-shared OOF libraries — see
   [`experiments/README.md`](experiments/README.md#stacking--the-community-oof-library-leaderboard-push)
   and `notebooks/04_ensembling.ipynb` for the method (that final stack depended on
   external datasets and rapidly-iterated scripts, so it lives in `experiments/stacking/`
   rather than the maintained `src/` pipeline).

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
src/                    The pipeline: a small number of clean, runnable modules
  config.py              Path constants (DATA/NN_CACHE/SUB/CONFIGS)
  utils.py                Shared helpers: target/frequency encoding, ORIG-CDF reference
                          fitting, rank-average blending, OOF weight search
  features.py             Feature engineering for both the GBDT and NN views of the data
  model_gbdt.py            LightGBM + XGBoost + CatBoost training
  model_nn.py              The Lookup-Transformer NN (model, dataset, training loop)
  main.py                  Entry point: runs every stage end to end, writes a submission
                          (`python src/main.py`, or `--fast` for a smoke test)

experiments/            Everything that isn't the maintained pipeline: diagnostic/setup
                         scripts (eda/, tuning/), the final community-OOF stacking push
                         (stacking/), and every rejected or superseded modeling idea
                         (gbdt_fe/, nn/, resnet/, tabm/, early_k1/) — see
                         experiments/README.md for what each one is and what it returned

notebooks/              Four notebooks walking through EDA, feature engineering, the
                         full GBDT pipeline, and ensembling — see notebooks/README.md
configs/                Best hyperparameters found for each GBDT model (JSON)

data/, sub/, nn_cache/  Not tracked in git — raw data, submissions, and cached
                         OOF/model artifacts. Not present in a fresh checkout except
                         data/.placeholder; the pipeline creates sub/ and nn_cache/
                         as needed.
```

## Path configuration

Every module imports its directory names (`DATA`, `NN_CACHE`, `SUB`, `CONFIGS`) from
[`src/config.py`](src/config.py) instead of hardcoding them, e.g.:

```python
from src.config import DATA, SUB
train = pd.read_csv(f'{DATA}/train.csv')
```

To point the pipeline at different locations (e.g. an external drive for `data/`),
edit `src/config.py` once rather than each module.

## Reproducing

Requires the competition data from Kaggle
([`playground-series-s6e8`](https://www.kaggle.com/competitions/playground-series-s6e8)):
place `train.csv`, `test.csv`, `sample_submission.csv`, and the small original dataset
under `data/` (replacing the `data/.placeholder` file that keeps the empty folder in
git). Run from the repository root:

```
pip install -r requirements.txt
python src/main.py            # full run — needs a GPU for a reasonable runtime
python src/main.py --fast     # tiny smoke-test config, verifies the pipeline runs
```

`main.py` runs every stage (feature engineering → GBDT → NN → blend) and writes
`sub/submission.csv`. To reproduce a specific historical result or hyperparameter
search instead of the current `src/` pipeline, see `experiments/README.md`.
