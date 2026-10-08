"""Shared mapping from a native decision API's answer to the common Answer type."""

from __future__ import annotations

import math

from ..schema import Answer, Question, option_keys

# native APIs round their probabilities, so the mass on known keys may be a little off 1
MASS_TOLERANCE = 0.05


def noul_answer(p_true: object) -> Answer:
    if isinstance(p_true, bool) or not isinstance(p_true, (int, float)) or math.isnan(p_true) or not 0 <= p_true <= 1:
        return Answer("error", error="yes/no probability is not a number between 0 and 1")
    return Answer("ok", answer=p_true >= 0.5, probabilities={"true": p_true, "false": 1 - p_true})


def _known_mass(question: Question, probabilities: dict[str, float]) -> dict[str, float] | None:
    """Probabilities on the question's own keys, or None when the response does not carry them.

    Zero-filling an unrecognised distribution would turn a schema mismatch into a
    confident wrong answer, so anything that does not sum to about 1 is rejected.
    """
    probs = {key: probabilities.get(key, 0.0) for key in option_keys(question)}
    try:
        total = float(sum(probs.values()))
    except TypeError:
        return None
    return probs if abs(total - 1) <= MASS_TOLERANCE else None


def choice_answer(question: Question, choice: str, probabilities: dict[str, float], confidence: float) -> Answer:
    keys = option_keys(question)
    if choice not in keys:
        return Answer("error", error=f"choice outside the option set: {choice!r}")
    probs = _known_mass(question, probabilities)
    if probs is None:
        return Answer("error", error="probabilities do not cover the question's options")
    return Answer("ok", answer=choice, probabilities=probs, confidence=confidence)


def score_answer(question: Question, score: float, probabilities: dict[str, float], confidence: float) -> Answer:
    keys = option_keys(question)
    probs = _known_mass(question, probabilities)
    if probs is None:
        return Answer("error", error="probabilities do not cover the question's levels")
    # max() keeps the first maximum, so ties go to the lower level
    top = max(keys, key=lambda key: probs[key])
    return Answer("ok", answer=int(top), probabilities=probs, confidence=confidence, expected_score=score)


def finish(
    answers: dict[str, Answer], latency_ms: float, input_tokens: int, output_tokens: int
) -> dict[str, Answer]:
    """Stamp latency on every answer and book the request's usage on the first one."""
    for i, answer in enumerate(answers.values()):
        answer.latency_ms = latency_ms
        if i == 0:
            answer.usage.input_tokens = input_tokens
            answer.usage.output_tokens = output_tokens
    return answers


def all_failed(questions: dict[str, Question], error: str | None, latency_ms: float) -> dict[str, Answer]:
    return {name: Answer("error", error=error, latency_ms=latency_ms) for name in questions}
