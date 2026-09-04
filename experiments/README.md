# Experiments

Everything here was tried and did **not** make it into the final pipeline (`src/`) —
either it was neutral/negative, or a later script in `src/` superseded it. Each entry
below states what was tried and what it returned, so the outcome is visible without
having to read the script.

## `gbdt_fe/` — GBDT feature-engineering variants

- **`lgbm_xgb_cat_catboostnative_2026-08-19.py`** — CatBoost native categorical handling
  (raw strings + `cat_features`) instead of manual target encoding. **LB -0.00074**,
  clear regression.
- **`lgbm_pairlattice_dense_2026-08-20.py`**, **`lgbm_pairlattice36_2026-08-20.py`**,
  **`lgbm_xgb_cat_ABD_pairlattice_2026-08-20.py`** — joint target encoding over 36
  pairs of continuous columns. **CV +0.00027 but LB -0.00018** — the project's one
  clear CV/LB contradiction: sparse-cell (2.7-18 rows/cell) OOF encoding memorized
  CV-fold-specific patterns that didn't transfer to the test set.
- **`lgbm_xgb_cat_ABD_pruned68_2026-08-20.py`** — pruned the 84-feature set down to 68.
  CV-neutral (0.96985 → 0.96986).
- **`lgbm_xgb_cat_generatorregion_2026-08-19.py`** — explicit flag for the generator's
  recovered hard decision rule. **LB -0.00006** — the trees already capture this split
  on their own.
- **`lgbm_xgb_cat_decimallattice_2026-08-19.py`** (category D) and
  **`lgbm_xgb_cat_imputeaugment_2026-08-19.py`** (category B) — tested individually,
  each was noise-level in CV alone. **Combined**, they cleared the threshold
  (CV +0.00036) and were confirmed on the leaderboard (+0.00015) — that combined
  version lives in `src/models/gbdt/lgbm_xgb_cat_ABD_combined_2026-08-19.py`. Kept here
  as the original isolated (negative) tests.
- **`lgbm_xgb_cat_freqenc_2026-08-19.py`** — frequency encoding. No measurable gain
  over the existing target encoding.
- **`lgbm_fe_categories.py`**, **`lgbm_fe_categories_v2.py`** — tested four candidate
  ratio/derived feature categories (A/B/C/D) one at a time. A and B were kept (folded
  directly into `src/models/gbdt/blend_lgb_xgb_cat_AB.py`); C ("awake-hours budget")
  and D ("inverse-direction ratio", in this earlier form) were rejected at the time.
- **`lgbm_fe_bundle_check.py`** — sanity-checked candidate feature bundles together
  before promoting any of them individually.
- **`lgbm_fe_cleanresid.py`**, **`lgbm_fe_constraintimpute.py`**,
  **`lgbm_fe_dominant_stressdev.py`**, **`lgbm_fe_jointte.py`** (categorical-pair joint
  target encoding — the categorical analogue of the pair-lattice test above) — each
  tested individually, none produced a net AUC contribution.
- **`lgbm_impute.py`** — model-based imputation instead of native NaN handling. CV
  regression (0.96134 vs. the 0.96349 raw-NaN baseline).
- **`lgbm_daily_semantic.py`** — semantic (domain-specific) NaN handling for
  daily-usage columns. CV regression (0.96206 vs. 0.96349).
- **`lgbm_ismissing.py`** — added explicit missingness-flag columns. No contribution —
  the data is close to MCAR, so flags are redundant with LightGBM's native NaN
  handling.
- **`lgbm_te10.py`** — target-encoding-only (dropping the raw value). Worse than the
  raw+TE combination (0.96073 vs. 0.96726) — raw+TE together was the actual early
  breakthrough, either alone was much weaker.
- **`lgbm_raw_te_inter.py`** — raw+TE plus pairwise interaction terms. No gain
  (0.96698 vs. 0.96726).
- **`lgbm_raw_te_nativecat.py`** — raw+TE with native (dtype-level) categoricals.
  Regression vs. numeric-encoded categoricals.
- **`lgbm_raw_te_ratios.py`** — an early ratio-feature pass on raw+TE, superseded by
  the category-A/B ratios that made it to production.
- **`lgbm_sum_feature.py`** — the original isolated test of a single
  sum-of-components feature (later folded into the production baseline as
  `sum_components`).
- **`lgbm_linearscore_2026-08-20.py`** — a single linear-combination "risk score"
  feature. No net contribution on top of the tree ensemble.
- **`lgbm_categorical.py`** — an early native-categorical attempt on the LightGBM
  side only. Superseded once target encoding proved stronger.
- **`blend_lgb_xgb_cat_ratios.py`** — an early ratio-feature blend variant, superseded
  by `src/models/gbdt/blend_lgb_xgb_cat_AB.py`.
- **`blend_featuresubset_diversity.py`** — blended models trained on different feature
  subsets for diversity. No gain over the standard full-feature blend.

## `nn/` — neural-network variants

- **`nn_experiment_features.py`** — the original test expanding the NN's inputs from
  9+3 raw columns to +8 derived ratio/diff features. **+0.00193 solo AUC**, the one
  clear net win among the isolated NN experiments — merged directly into production
  (`src/features/nn_data_prep.py`, `src/models/nn/nn_model.py`). Kept here as the
  historical isolated test.
- **`nn_experiment_capacity.py`** — larger Transformer (d_token 64→128, 2→3 layers) +
  LR warmup. **-0.00085**, didn't converge in the 25-epoch budget.
- **`nn_experiment_labelsmooth.py`** — label smoothing (eps=0.02). +0.00019 in
  isolation, but **-0.00033 combined** with the feature-expansion win above — the two
  effects didn't stack, rejected.
- **`nn_optuna_search_2026-08-20.py`** — 20-trial Optuna hyperparameter search. Best
  val AUC only +0.0005 versus baseline, stayed in a narrow 0.961-0.966 band — did not
  close the ~0.003-0.005 AUC gap to external reference NN implementations.
- **`nn_train_kfold_tuned_2026-08-20.py`** — full 5-fold retrain with the best Optuna
  config. OOF 0.96569, slightly **worse** than the untuned NN's 0.96576 — this closed
  the "NN hyperparameter tuning" direction entirely.
- **`nn_experiment_missingaug_2026-08-21.py`**, **`nn_experiment_missingaug_round2_
  2026-08-21.py`** — mask-probability sweep for missingness augmentation (0.05→0.50).
  The winning setting (mask_prob=0.40) was promoted directly into `src/features/
  nn_common.py` and the production training scripts; these are the sweep scripts.
- **`nn_experiment_missingaug_grouped_2026-08-21.py`**,
  **`nn_train_kfold_missingaug_grouped_2026-08-21.py`**,
  **`blend_gbdt_nn_missingaug_grouped_2026-08-21.py`** — a cascaded (pandas-NaN-
  propagation-aware) masking variant, testing whether derived-column "leakage" was
  why the GBDT/NN correlation was so high. Hypothesis **disproved**: correlation
  barely moved (0.9808 vs. 0.9806), and solo NN quality was worse (OOF 0.96684 vs.
  0.96717 for the cell-based production version); blend OOF also worse (0.96902 vs.
  0.96909).
- **`nn_data_prep_kfold.py`** — the pre-"featfull" NN feature-prep script (8 derived
  columns). Superseded by `src/features/nn_data_prep_kfold_featfull_2026-08-29.py`
  (62 derived columns, +0.00074 solo NN AUC, +0.00010 LB).

## `resnet/` — attention-free tabular ResNet (architecture diversity)

Built to get a model with lower correlation to the GBDT ensemble than the Transformer
NN, using the exact same GBDT 113-feature set, no attention — just residual MLP
blocks.

- **v1** (`resnet_data_prep_2026-08-29.py`, `resnet_model.py`,
  `resnet_train_kfold_2026-08-29.py`; hidden=256, 5 blocks): solo OOF **0.96452**.
  Correlation with GBDT genuinely dropped as intended (0.9791 vs. the Transformer
  NN's 0.9814) — the diversity hypothesis worked directionally. But solo quality was
  too low to earn blend weight: 2-way blend weight search gave it ~0 weight, and using
  it anyway would have cost -0.00025 vs. the production blend.
- **v2** (`resnet_train_kfold_v2_2026-08-29.py`; hidden=384, 7 blocks, more dropout):
  solo OOF **0.96380** — worse than v1, capacity increase didn't help.
- **`blend_gbdt_origfeat_resnet_2026-08-29.py`** — 3-way blend (GBDT + NN + ResNet):
  optimal ResNet weight came out to **0**.
- **Verdict**: architecture diversity is not sufficient on its own — a model needs to
  clear a minimum solo-quality bar before its "differentness" is worth anything to a
  blend.

## `tabm/` — TabM batch-ensemble

**`tabm_data_prep_2026-08-20.py`**, **`tabm_train_kfold_2026-08-20.py`** — TabM
(k=32 batch-ensemble). Fold-0 validation AUC **0.96595**, not competitive with the
GBDT/NN combination, and the k=32 batch-ensemble ran inefficiently on this hardware
(GPU power-capped laptop). Abandoned after one fold.

## `early_k1/` — early (day 1-2) exploration

Before the raw+target-encoding GBDT approach became the clear baseline:
**`model_family_battery_k1.py`** (Random Forest / Extra Trees / HistGB alternatives to
LightGBM), **`blend_multiseed.py`** / **`blend_multiseed_K1.py`** (multi-seed
bagging), **`focal_loss_k1.py`** (focal loss for the class imbalance),
**`imbalance_test_k1.py`** (explicit imbalance handling), **`capacity_sweep_k1.py`**
(model capacity sweep), **`pseudo_label_k1.py`** (pseudo-labeling on the test set).
None beat the GBDT + target-encoding baseline that emerged the same week; kept for the
record of what was ruled out early rather than re-explored later.
