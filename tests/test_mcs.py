import numpy as np
import pandas as pd

from oilfc.eval.mcs import model_confidence_set


def _losses(n=600, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, n)
    return pd.DataFrame(
        {"a": base, "b": base + rng.normal(0, 0.1, n), "bad": base + 1.0 + rng.normal(0, 0.1, n)}
    )


def test_a_clearly_worse_model_is_excluded_and_equal_models_are_kept():
    out = model_confidence_set(_losses(), size=0.10, block_size=5, reps=500)
    assert out.loc["a", "in_set"] and out.loc["b", "in_set"]
    assert not out.loc["bad", "in_set"]
    assert out.loc["bad", "mcs_pvalue"] < 0.10


def test_result_has_one_row_per_model():
    out = model_confidence_set(_losses(), block_size=5, reps=200)
    assert list(out.index) == ["a", "b", "bad"]
    assert set(out.columns) == {"mcs_pvalue", "in_set"}
