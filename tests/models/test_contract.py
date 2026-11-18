import numpy as np
import pytest

from forecast.models.base import PickleMixin, Task
from forecast.models.baselines import baselines_for
from forecast.models.gbdt import LGBMQuantile


@pytest.mark.parametrize("task_name", ["ret_15", "rv_15"])
def test_contract(fs, tmp_path, task_name):
    kind, h = task_name.split("_")
    task = Task(task_name, kind, int(h), int(h), fs.label_cols[task_name], (0.1, 0.25, 0.5, 0.75, 0.9))
    d = fs.df.drop_nulls(task.label_col)
    X, y = d.select(fs.feature_cols), d[task.label_col].to_numpy()
    n = len(y)
    a, b = slice(0, int(n * 0.6)), slice(int(n * 0.6), int(n * 0.8))
    for m in [*baselines_for(task), LGBMQuantile(num_boost_round=50, early_stopping=10, n_jobs=2)]:
        m.fit(X[a], y[a], X[b], y[b], task)
        q = m.predict(X[int(n * 0.8):])
        assert q.shape == (n - int(n * 0.8), 5)
        assert np.all(np.diff(q, axis=1) >= 0), f"{m.name}: quantiles not monotone"
        if kind == "rv":
            assert (q > 0).all()
        p = tmp_path / f"{m.name}.pkl"
        m.save(p)
        assert np.allclose(PickleMixin.load(p).predict(X[:50]), m.predict(X[:50]))
