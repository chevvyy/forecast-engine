from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from forecast.evaluation.splits import holdout_cutoff, walk_forward

pytestmark = pytest.mark.leakage


def test_purge_embargo_holdout(fs):
    meta = fs.df
    emb, hold = timedelta(minutes=450), timedelta(days=3)
    folds = list(walk_forward(meta, datetime(2024, 1, 9, tzinfo=UTC), timedelta(days=2),
                              timedelta(days=2), emb, hold, min_train_rows=100))
    assert len(folds) >= 2
    ls = meta["label_start"].to_numpy().astype("datetime64[us]")
    le = meta["label_end"].to_numpy().astype("datetime64[us]")
    cut = np.datetime64(holdout_cutoff(meta, hold).replace(tzinfo=None), "us")
    for f in folds:
        vs = np.datetime64(f.val_start.replace(tzinfo=None), "us")
        assert (le[f.train_idx] <= vs - np.timedelta64(450, "m")).all()   # purge + gap
        assert (ls[f.val_idx] >= vs).all()
        assert len(np.intersect1d(f.train_idx, f.val_idx)) == 0
        assert (ls[f.train_idx] < cut).all() and (le[f.val_idx] <= cut).all()  # holdout untouched
        # no train label interval overlaps any val label interval
        assert le[f.train_idx].max() <= ls[f.val_idx].min()
