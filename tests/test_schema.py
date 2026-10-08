import pytest

from haiku_decides.schema import Question, Usage, option_keys, validate_question


def test_choice_needs_2_to_255_options():
    with pytest.raises(ValueError):
        validate_question(Question("choice", "q", {"a": "x"}))
    validate_question(Question("choice", "q", {"a": "x", "b": "y"}))
    with pytest.raises(ValueError):
        validate_question(Question("choice", "q", {str(i): "x" for i in range(256)}))


def test_score_needs_2_to_10_levels():
    with pytest.raises(ValueError):
        validate_question(Question("score", "q", ["only"]))
    validate_question(Question("score", "q", ["lo", "hi"]))
    with pytest.raises(ValueError):
        validate_question(Question("score", "q", [str(i) for i in range(11)]))


def test_noul_criteria_optional_with_fixed_keys():
    validate_question(Question("noul", "q"))
    validate_question(Question("noul", "q", {"true": "y", "false": "n"}))
    with pytest.raises(ValueError):
        validate_question(Question("noul", "q", {"yes": "y"}))


def test_option_keys():
    assert option_keys(Question("choice", "q", {"b": "x", "a": "y"})) == ["b", "a"]
    assert option_keys(Question("score", "q", ["l", "m", "h"])) == ["0", "1", "2"]
    assert option_keys(Question("noul", "q")) == ["true", "false"]


def test_usage_adds_fieldwise():
    assert Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40) == Usage(11, 22, 33, 44)
