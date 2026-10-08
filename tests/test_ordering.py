from haiku_decides.ordering import permute_criteria, permuted


def test_permuted_is_deterministic_permutation():
    keys = [f"k{i}" for i in range(77)]
    assert permuted(keys, "s") == permuted(keys, "s")
    assert sorted(permuted(keys, "s")) == sorted(keys)
    assert permuted(keys, "s") != keys
    assert permuted(keys, "s") != permuted(keys, "t")


def test_permute_criteria_keeps_descriptions():
    c = {"a": "A", "b": "B", "c": "C"}
    p = permute_criteria(c, "x")
    assert p == c and list(p) == permuted(list(c), "x")
