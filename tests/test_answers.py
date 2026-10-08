from haiku_decides.backends._answers import choice_answer, noul_answer, score_answer
from haiku_decides.schema import Question

SCORE = Question("score", "q", ["a", "b", "c", "d", "e"])
CHOICE = Question("choice", "q", {"A": "x", "B": "y"})


def test_score_without_probabilities_is_an_error_not_level_zero():
    assert score_answer(SCORE, 3.6, {}, 0.8).status == "error"


def test_score_with_label_keyed_probabilities_is_an_error():
    assert score_answer(SCORE, 3.6, {"a": 0.1, "d": 0.9}, 0.8).status == "error"


def test_choice_probabilities_must_sit_on_the_known_keys():
    assert choice_answer(CHOICE, "A", {"x": 0.9, "y": 0.1}, 0.9).status == "error"
    assert choice_answer(CHOICE, "A", {"A": 0.98, "B": 0.01}, 0.9).status == "ok"


def test_noul_probability_must_be_a_number_between_zero_and_one():
    for bad in (1.7, float("nan"), True, -0.1, "0.5", None):
        assert noul_answer(bad).status == "error", bad
    assert noul_answer(0.5).status == "ok"
