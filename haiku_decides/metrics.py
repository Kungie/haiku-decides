"""Pure metric functions. No I/O, no network."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from .schema import Usage


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


def accuracy(golds: Sequence, answers: Sequence) -> float:
    return sum(g == a for g, a in zip(golds, answers)) / len(golds)


def mean_absolute_error(golds: Sequence[int], answers: Sequence[int]) -> float:
    return sum(abs(g - a) for g, a in zip(golds, answers)) / len(golds)


def ece(confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error over equal-width bins of the chosen answer's probability."""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(corrects, dtype=float)
    # round before flooring so 0.7 * 10 == 7.000000000000001 and 0.3 * 10 land in the intended bin
    bins = np.minimum(np.floor(np.round(conf * n_bins, 9)).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def brier(probabilities: Sequence[dict[str, float]], gold_keys: Sequence[str]) -> float:
    scores = [
        sum((p - (1.0 if key == gold else 0.0)) ** 2 for key, p in probs.items())
        for probs, gold in zip(probabilities, gold_keys)
    ]
    return sum(scores) / len(scores)


def flip_rate(answers_by_item: dict[str, list]) -> float:
    """Share of items whose answer is not identical across all orderings."""
    if not answers_by_item:
        return 0.0
    flipped = sum(len(set(answers)) > 1 for answers in answers_by_item.values())
    return flipped / len(answers_by_item)


def _tv(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in p.keys() | q.keys())


def mean_tv_distance(probs_by_item: dict[str, list[dict[str, float]]]) -> float:
    """Mean pairwise total variation distance between orderings, averaged over items."""
    per_item = []
    for dists in probs_by_item.values():
        pairs = list(combinations(dists, 2))
        if pairs:
            per_item.append(sum(_tv(p, q) for p, q in pairs) / len(pairs))
    return sum(per_item) / len(per_item) if per_item else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def usd_per_1k(usages: Sequence[Usage], prices: Prices) -> float:
    if not usages:
        return 0.0
    total = sum(
        u.input_tokens * prices.input
        + u.output_tokens * prices.output
        + u.cache_read_input_tokens * prices.cache_read
        + u.cache_creation_input_tokens * prices.cache_write
        for u in usages
    ) / 1_000_000
    return total / len(usages) * 1000


def cache_read_ratio(usages: Sequence[Usage]) -> float:
    read = sum(u.cache_read_input_tokens for u in usages)
    denominator = read + sum(u.input_tokens + u.cache_creation_input_tokens for u in usages)
    return read / denominator if denominator else 0.0


def bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    n_items: int,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap over items. `stat` receives an array of resampled item indices."""
    rng = np.random.default_rng(seed)
    stats = [stat(rng.integers(0, n_items, n_items)) for _ in range(n_resamples)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]
