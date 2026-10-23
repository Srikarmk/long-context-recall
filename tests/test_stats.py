import numpy as np
import pandas as pd

from lcrecall.stats import above_zero, below_zero, excludes_zero, item_ci, mean_ci, paired_ci


def frame(vals):
    return pd.DataFrame([{"id": f"i{i}", "v": float(x)} for i, x in enumerate(vals)])


def test_mean_ci_constant_and_seeded():
    assert mean_ci([1, 1, 1]) == {"mean": 1.0, "lo": 1.0, "hi": 1.0, "n": 3}
    x = np.random.default_rng(1).integers(0, 2, 60)
    a = mean_ci(x)
    assert a["lo"] <= a["mean"] <= a["hi"] and mean_ci(x) == a


def test_item_rows_are_averaged_before_resampling():
    df = pd.DataFrame([{"id": "x", "v": 1.0}, {"id": "x", "v": 0.0}, {"id": "y", "v": 1.0}])
    assert item_ci(df, "v")["n"] == 2 and item_ci(df, "v")["mean"] == 0.75


def test_paired_ci_uses_common_items_only():
    a = frame([1, 1, 0, 1] * 10)
    b = frame([0, 1, 0, 0] * 10)
    d = paired_ci(a, "v", b)
    assert abs(d["mean"] - 0.5) < 1e-12 and d["lo"] > 0 and above_zero(d) and excludes_zero(d)
    assert not above_zero(paired_ci(a, "v", a))
    assert paired_ci(a, "v", b.iloc[:10])["n"] == 10


def test_nan_values_are_dropped_and_signs():
    r = mean_ci([1.0, float("nan"), 0.0])
    assert r["n"] == 2 and r["mean"] == 0.5
    assert below_zero({"n": 5, "lo": -0.3, "hi": -0.1}) and not below_zero({"n": 5, "lo": -0.3, "hi": 0.1})
