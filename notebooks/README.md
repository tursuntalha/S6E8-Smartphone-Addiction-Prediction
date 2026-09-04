# Notebooks

Four notebooks, each demonstrating one stage of the pipeline documented in the main
README and implemented in `src/`. They're written to be read and run independently —
some duplication of code between them is intentional.

Requires the competition data under `data/` (see the main README's "Reproducing"
section) to actually run.

- **`01_eda.ipynb`** — data exploration: column meanings, target distribution, missing
  values, numeric summaries, dependent-vs-independent group comparisons, plus a quick
  LightGBM baseline and a look at missing-value handling strategies.
- **`02_feature_engineering.ipynb`** — walks through the three feature families used in
  the real pipeline (target encoding, ratio/sum features, ORIG-CDF features) and
  measures each one's effect with a quick 3-fold check.
- **`03_full_pipeline.ipynb`** — a complete, self-contained run of the GBDT side of the
  pipeline: ORIG-CDF + derived features → imputation → multi-seed LightGBM/XGBoost/
  CatBoost training → rank-average blend → submission file. This is the closest thing
  to a single "run everything" notebook (the neural network is `src/model_nn.py`, run
  via `src/main.py`; the further community-OOF stack is `experiments/stacking/`).
- **`04_ensembling.ipynb`** — combining already-trained model outputs: rank averaging,
  OOF-weight search, and a logistic-regression stack. Trains two small, deliberately
  different models on the spot (LightGBM + Logistic Regression) so the notebook is
  runnable without any cached prediction files, then applies the same combination
  methods used for the real leaderboard push.

**On the numbers:** notebooks `02` and `04` use small, single-seed, few-fold setups for
speed and clarity, and will report different (generally lower) AUC than the production
numbers in the main README, which came from tuned, multi-seed, 5-fold pipelines and —
for the smallest effects — direct leaderboard verification. That's expected: these
notebooks exist to make each *mechanism* inspectable, not to reproduce the leaderboard
score.
