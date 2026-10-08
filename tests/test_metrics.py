import numpy as np
import pytest

from haiku_decides.metrics import (
    Prices,
    accuracy,
    bootstrap_ci,
    brier,
    cache_read_ratio,
    ece,
    flip_rate,
    mean_absolute_error,
    mean_tv_distance,
    overlaps,
    percentile,
    usd_per_1k,
)
from haiku_decides.schema import Usage


def test_accuracy():
    assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == 0.75


def test_mean_absolute_error():
    assert mean_absolute_error([0, 4, 2], [1, 4, 0]) == 1.0


def test_ece_hand_computed():
    assert ece([0.9, 0.9, 0.6, 0.6], [True, True, True, False]) == pytest.approx(0.1)


def test_ece_confidence_one_goes_to_last_bin():
    assert ece([1.0, 1.0], [True, True]) == 0.0


def test_brier_hand_computed():
    probs = [{"a": 0.8, "b": 0.2}, {"a": 0.5, "b": 0.5}]
    assert brier(probs, ["a", "b"]) == pytest.approx(0.29)


def test_flip_rate():
    assert flip_rate({"i1": ["a"] * 5, "i2": ["a", "b", "a", "a", "a"]}) == 0.5


def test_mean_tv_distance():
    assert mean_tv_distance({"i1": [{"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}]}) == pytest.approx(0.5)


def test_percentile():
    assert percentile([100.0, 200.0, 300.0], 50) == 200.0


def test_usd_per_1k():
    prices = Prices(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125)
    usages = [Usage(input_tokens=500, output_tokens=50)] * 2000
    assert usd_per_1k(usages, prices) == pytest.approx(0.075)


def test_cache_read_ratio():
    assert cache_read_ratio([Usage(100, 5, 300, 0)]) == 0.75
    assert cache_read_ratio([]) == 0.0


def test_bootstrap_ci_is_deterministic_and_tight_on_constant_data():
    correct = np.ones(50)
    assert bootstrap_ci(lambda idx: float(correct[idx].mean()), 50) == (1.0, 1.0)
    data = np.arange(50) % 2
    stat = lambda idx: float(data[idx].mean())
    assert bootstrap_ci(stat, 50, seed=7) == bootstrap_ci(stat, 50, seed=7)
    lo, hi = bootstrap_ci(stat, 50, seed=7)
    assert lo < 0.5 < hi


def test_overlaps():
    assert overlaps((0.1, 0.3), (0.3, 0.5))
    assert not overlaps((0.1, 0.2), (0.3, 0.5))


def test_ece_keeps_0_9_and_1_0_in_separate_bins():
    conf = [0.9] * 50 + [1.0] * 50
    correct = [True] * 50 + [True] * 40 + [False] * 10
    assert ece(conf, correct) == pytest.approx(0.15)
