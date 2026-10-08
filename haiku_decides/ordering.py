from __future__ import annotations

import random


def permuted(keys: list[str], seed: str) -> list[str]:
    """A seeded permutation of `keys`; the same seed always gives the same order."""
    return random.Random(seed).sample(keys, len(keys))


def permute_criteria(criteria: dict[str, str], seed: str) -> dict[str, str]:
    return {key: criteria[key] for key in permuted(list(criteria), seed)}
