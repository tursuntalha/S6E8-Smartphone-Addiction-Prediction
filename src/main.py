"""Entry point: runs the full pipeline end to end and writes a submission file.

    python src/main.py            # full production-scale run (needs a GPU for a
                                   # reasonable runtime; takes hours on ~1M rows)
    python src/main.py --fast     # tiny smoke-test config (few estimators/epochs) —
                                   # verifies the pipeline runs, not a real result

Stages: load data -> GBDT feature matrix -> train GBDT ensemble -> NN tensors ->
train NN -> search the GBDT/NN blend weight on out-of-fold AUC -> write submission.
See the main README's "Approach" section for what each stage does and why.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import SUB
from src import features, model_gbdt, model_nn, utils


def main(fast=False):
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}' + ('  (--fast smoke test)' if fast else ''))

    train, test, orig = features.load_raw_data()
    y = train['addicted_label'].values
    print(f'train: {train.shape}  test: {test.shape}  ({time.time() - t0:.0f}s)')

    orig_clean, orig_y = features.clean_orig_reference(train, orig)
    refs, global_rate = utils.fit_orig_cdf_refs(orig_clean, orig_y)
    print(f'ORIG reference fit: {len(orig)} -> {len(orig_clean)} rows after dedup  '
          f'({time.time() - t0:.0f}s)')

    impute_n_estimators = 20 if fast else 300
    X_train, X_test, feature_names = features.build_gbdt_feature_matrix(
        train, test, refs, global_rate, impute_n_estimators=impute_n_estimators)
    print(f'GBDT feature matrix: {X_train.shape}  ({time.time() - t0:.0f}s)')

    tuned_lgb, tuned_xgb, tuned_cat = features.load_best_params()
    gbdt = model_gbdt.train_gbdt_ensemble(
        X_train, y, X_test, tuned_lgb, tuned_xgb, tuned_cat, device=device, fast=fast)
    gbdt_oof = utils.rank_average(gbdt.oof_lgb, gbdt.oof_xgb, gbdt.oof_cat)
    gbdt_test = utils.rank_average(gbdt.pred_lgb, gbdt.pred_xgb, gbdt.pred_cat)
    print(f'GBDT blend OOF AUC: {roc_auc_score(y, gbdt_oof):.5f}  ({time.time() - t0:.0f}s)')

    # NN gets the same derived columns (already added to train/test in place by
    # build_gbdt_feature_matrix) plus the same ORIG-CDF references, as its own
    # PLR-only token group.
    nn_data = features.build_nn_arrays(train, test, refs, global_rate)
    print(f'NN tensors built  ({time.time() - t0:.0f}s)')
    nn_oof, nn_test = model_nn.train_nn_kfold(nn_data, device=device, fast=fast, verbose=not fast)
    print(f'NN done  ({time.time() - t0:.0f}s)')

    weight, blend_auc = utils.search_blend_weight(gbdt_oof, nn_oof, y)
    print(f'\nBest GBDT weight: {weight:.3f}  blended OOF AUC: {blend_auc:.5f}  '
          f'(GBDT solo: {roc_auc_score(y, gbdt_oof):.5f}, NN solo: {roc_auc_score(y, nn_oof):.5f})')

    final_test = utils.apply_blend_weight(gbdt_test, nn_test, weight)
    os.makedirs(SUB, exist_ok=True)
    sub_path = f'{SUB}/submission.csv'
    pd.DataFrame({'id': nn_data['test_id'], 'addicted_label': final_test}).to_csv(sub_path, index=False)
    print(f'\nSaved: {sub_path}  (total: {(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', action='store_true',
                         help='tiny smoke-test config instead of the full production run')
    args = parser.parse_args()
    main(fast=args.fast)
