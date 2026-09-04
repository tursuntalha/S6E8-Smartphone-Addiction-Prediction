# Experiments

`src/` is kept to a small, clean, runnable pipeline (`config.py`, `utils.py`,
`features.py`, `model_gbdt.py`, `model_nn.py`, `main.py`) implementing the approach
described in the main README. Everything else — analysis/diagnostic scripts, one-time
hyperparameter search, and every rejected or superseded modeling idea — lives here
instead, organized by what kind of thing it is:

- **`eda/`** and **`tuning/`** aren't "rejected" — they're one-off setup/diagnostic work
  (data analysis, Optuna hyperparameter search) that doesn't need to run every time the
  pipeline does. Their outputs are already where they need to be: EDA's findings are in
  `notebooks/01_eda.ipynb` and the main README, and tuning's outputs are the JSON files
  in `configs/`.
- **`stacking/`** is the final leaderboard push: a further ensembling step, on top of
  the GBDT+NN blend `src/main.py` produces, that stacks in several community-shared OOF
  (out-of-fold) prediction libraries. It's kept separate from `src/` because it depends
  on external datasets not included in this repo and was iterated on rapidly (many
  near-duplicate scripts) rather than kept clean — see its own section below.
- **`gbdt_fe/`**, **`nn/`**, **`resnet/`**, **`tabm/`**, **`early_k1/`** are genuinely
  rejected or superseded modeling ideas. Each entry below states what was tried and what
  it returned, so the outcome is visible without having to read the script.

## `eda/` — data analysis and generator reverse-engineering

Diagnostic scripts behind the findings in `notebooks/01_eda.ipynb` and the main
README's "Approach": **`eda_generator_grid_monotonic.py`** and
**`eda_generator_lookup_interaction.py`** (reverse-engineering the synthetic
generator's per-column value patterns and its hard decision rule),
**`eda_dist_shift_psi.py`** (train/test population-stability check, PSI = 0),
**`eda_imputability_r2.py`** (how predictable each column is from the others — informed
the model-based imputation feature), and **`eda_te_smoothing_sweep.py`** /
**`te_diag_default_vs_tuned.py`** / **`te_smoothing_model_sweep.py`** /
**`te_smoothing_sweep_v2.py`** (target-encoding smoothing parameter sweeps that settled
on `SMOOTH=3.0`, used in `src/features.py`).

## `tuning/` — hyperparameter search

**`tune_lgbm.py`**, **`tune_xgb.py`**, **`tune_cat.py`** — independent Optuna searches
for LightGBM/XGBoost/CatBoost on the raw+TE feature set; their output is what's cached
in `configs/best_params_{lgbm,xgb,cat}.json` and loaded by `src/features.load_best_params()`.
**`tune_lgbm_raw_te.py`** is an earlier LightGBM-only search superseded by `tune_lgbm.py`.
Independently tuning XGBoost/CatBoost (rather than reusing LightGBM's params) was worth
about +0.00017 on the leaderboard.

## `stacking/` — the community-OOF-library leaderboard push

Starting from the GBDT+NN blend (`src/main.py`, OOF≈0.9691, LB=0.97035), the final
session pooled ~100 out-of-fold prediction vectors — the project's own GBDT and NN, plus
several other competitors' publicly-shared OOF libraries — and searched several 2nd-level
combination strategies on top, all validated with honest 5-fold meta-CV (never fit on
the same rows a member's own OOF came from):

- **`blend_gbdt_nn_*.py`**, **`blend_4model_stack_k1.py`** — earlier (day 1-3) 2-model
  and 4-model blend/stack iterations, superseded by the ones below.
- **`stack_data_2026-08-30.py`** (+ the accidental duplicate `stack_data_2026_08_30.py`)
  — the shared OOF-library loader (74 external members + this project's own 2).
- **`stack_explore_2026-08-30.py`**, **`stack_expand_2026-08-30.py`**,
  **`stack_expand_variants_2026-08-30.py`**, **`stack_variants_2026-08-30.py`**,
  **`stack_run_variant_2026-08-30.py`**, **`stack_mkn_2026-08-30.py`**,
  **`stack_impute_variants_2026-08-30.py`**, **`stack_fe_meta_2026-08-30.py`**,
  **`stack_metas_2026-08-30.py`**, **`stack_bandfix_2026-08-30.py`** — the search over
  2nd-level combination strategies (logistic-regression / XGBoost / CatBoost meta-models,
  band-local corrections, different OOF-member subsets). Best result: a 2nd-level
  logistic-regression stack over 5 different 1st-level combinations, meta-CV OOF≈0.97004.
- **`stack_logreg.py`**, **`meta_stack_test.py`** — the underlying meta-model fitting
  helpers used by the scripts above.
- **`stack_extra_2026-08-30.py`** — has a known bug in its test-set prediction path
  (produces a constant column); its OOF numbers are fine but its submission output is
  not, kept as-is for the record rather than silently fixed.
- **`stack_finale_2026-08-30.py`**, **`final_verify_2026-08-30.py`** — assembled and
  sanity-checked the final 10 candidate submissions from the above.

This step was never confirmed on the leaderboard — the competition ended before a
result came back. `src/main.py` reproduces the confirmed-best step before it
(OOF≈0.9691, LB=0.97035).

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
  (CV +0.00036) and were confirmed on the leaderboard (+0.00015) — both are permanently
  part of `src/features.py`'s feature set now. Kept here as the original isolated
  (negative) tests.
- **`lgbm_xgb_cat_freqenc_2026-08-19.py`** — frequency encoding. No measurable gain
  over the existing target encoding.
- **`lgbm_fe_categories.py`**, **`lgbm_fe_categories_v2.py`** — tested four candidate
  ratio/derived feature categories (A/B/C/D) one at a time. A and B were kept (now part
  of `src/features.py`'s feature set); C ("awake-hours budget") and D ("inverse-direction
  ratio", in this earlier form) were rejected at the time.
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
  by the feature set now in `src/features.py`.
- **`blend_featuresubset_diversity.py`** — blended models trained on different feature
  subsets for diversity. No gain over the standard full-feature blend.

## `nn/` — neural-network variants

- **`nn_experiment_features.py`** — the original test expanding the NN's inputs from
  9+3 raw columns to +8 derived ratio/diff features. **+0.00193 solo AUC**, the one
  clear net win among the isolated NN experiments — merged directly into production
  (`src/features.py`'s `build_nn_arrays`, `src/model_nn.py`). Kept here as the
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
  The winning setting (mask_prob=0.40) is now the default in `src/model_nn.py`'s
  `train_nn_kfold`; these are the sweep scripts.
- **`nn_experiment_missingaug_grouped_2026-08-21.py`**,
  **`nn_train_kfold_missingaug_grouped_2026-08-21.py`**,
  **`blend_gbdt_nn_missingaug_grouped_2026-08-21.py`** — a cascaded (pandas-NaN-
  propagation-aware) masking variant, testing whether derived-column "leakage" was
  why the GBDT/NN correlation was so high. Hypothesis **disproved**: correlation
  barely moved (0.9808 vs. 0.9806), and solo NN quality was worse (OOF 0.96684 vs.
  0.96717 for the cell-based production version); blend OOF also worse (0.96902 vs.
  0.96909).
- **`nn_data_prep_kfold.py`** — the pre-"featfull" NN feature-prep script (8 derived
  columns). Superseded by the 62-derived-column version now in `src/features.py`'s
  `build_nn_arrays` (+0.00074 solo NN AUC, +0.00010 LB over the 8-column version).

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
