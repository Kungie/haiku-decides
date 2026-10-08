"""Shared mapping from a native decision API's answer to the common Answer type."""

from __future__ import annotations

from ..schema import Answer, Question, option_keys


def noul_answer(p_true: float) -> Answer:
    return Answer("ok", answer=p_true >= 0.5, probabilities={"true": p_true, "false": 1 - p_true})


def choice_answer(question: Question, choice: str, probabilities: dict[str, float], confidence: float) -> Answer:
    keys = option_keys(question)
    if choice not in keys:
        return Answer("error", error=f"choice outside the option set: {choice!r}")
    return Answer(
        "ok",
        answer=choice,
        probabilities={key: probabilities.get(key, 0.0) for key in keys},
        confidence=confidence,
    )


def score_answer(question: Question, score: float, probabilities: dict[str, float], confidence: float) -> Answer:
    keys = option_keys(question)
    probs = {key: probabilities.get(key, 0.0) for key in keys}
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
