import pytest

from haiku_decides.aggregate import Sample, aggregate
from haiku_decides.schema import Question, Usage

CHOICE = Question("choice", "q", {"a": "A", "b": "B", "c": "C"})


def ok(key):
    return Sample("ok", key, Usage(10, 2))


def test_choice_vote_shares():
    a = aggregate(CHOICE, [ok("a")] * 7 + [ok("b")] * 3, with_probabilities=True, latency_ms=5.0)
    assert (a.status, a.answer, a.confidence, a.n_ok) == ("ok", "a", 0.7, 10)
    assert a.probabilities == {"a": 0.7, "b": 0.3, "c": 0.0}
    assert a.usage == Usage(100, 20) and a.latency_ms == 5.0


def test_tie_breaks_to_first_option():
    a = aggregate(CHOICE, [ok("b")] * 5 + [ok("a")] * 5, with_probabilities=True, latency_ms=0)
    assert a.answer == "a"


def test_noul_half_is_true_and_has_no_confidence():
    q = Question("noul", "q")
    a = aggregate(q, [ok("true")] * 5 + [ok("false")] * 5, with_probabilities=True, latency_ms=0)
    assert a.answer is True and a.confidence is None
    assert a.probabilities == {"true": 0.5, "false": 0.5}


def test_score_expected_value():
    q = Question("score", "q", ["lo", "mid", "hi"])
    a = aggregate(q, [ok("1")] * 7 + [ok("2")] * 3, with_probabilities=True, latency_ms=0)
    assert a.answer == 1 and a.confidence == 0.7
    assert a.expected_score == pytest.approx(1.3)


def test_partial_success_rule():
    err, ref = Sample("error", error="x"), Sample("refused")
    half = aggregate(CHOICE, [ok("a")] * 5 + [err] * 5, with_probabilities=True, latency_ms=0)
    assert (half.status, half.n_ok, half.probabilities["a"]) == ("ok", 5, 1.0)
    assert half.samples == ["a"] * 5 + [None] * 5
    assert aggregate(CHOICE, [ok("a")] * 4 + [err] * 6, with_probabilities=True, latency_ms=0).status == "error"
    mostly_refused = aggregate(CHOICE, [ok("a")] * 4 + [err] * 2 + [ref] * 4, with_probabilities=True, latency_ms=0)
    assert mostly_refused.status == "refused" and mostly_refused.answer is None


def test_single_mode_has_no_probabilities():
    a = aggregate(CHOICE, [ok("b")], with_probabilities=False, latency_ms=0)
    assert a.answer == "b" and a.probabilities is None and a.confidence is None
