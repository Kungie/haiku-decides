from __future__ import annotations

import random


def permuted(keys: list[str], seed: str) -> list[str]:
    """A seeded permutation of `keys`; the same seed always gives the same order."""
    return random.Random(seed).sample(keys, len(keys))


def reorderings(keys: list[str], seed: str, count: int) -> list[list[str]]:
    """`count` seeded orderings of `keys`, distinct from each other and from the given order.

    Falls back to repeats only when fewer than `count` such orderings exist.
    """
    rng = random.Random(seed)
    found: list[list[str]] = []
    for _ in range(1000):
        candidate = rng.sample(keys, len(keys))
        if candidate != keys and candidate not in found:
            found.append(candidate)
            if len(found) == count:
                return found
    while len(found) < count:
        found.append(rng.sample(keys, len(keys)))
    return found


def permute_criteria(criteria: dict[str, str], seed: str) -> dict[str, str]:
    return {key: criteria[key] for key in permuted(list(criteria), seed)}
