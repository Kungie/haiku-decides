"""Turn N sampled answers into one decision with vote-share probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Answer, Question, Status, Usage, option_keys


@dataclass
class Sample:
    status: Status
    # always one of option_keys(question) when status is "ok"
    answer: str | None = None
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


def aggregate(question: Question, samples: list[Sample], *, with_probabilities: bool, latency_ms: float) -> Answer:
    keys = option_keys(question)
    usage = sum((s.usage for s in samples), Usage())
    raw = [s.answer if s.status == "ok" else None for s in samples]
    oks = [s for s in samples if s.status == "ok"]
    n_ok = len(oks)

    # a decision needs at least half of its samples
    if not samples or n_ok * 2 < len(samples):
        refused = sum(s.status == "refused" for s in samples)
        errors = sum(s.status == "error" for s in samples)
        first_error = next((s.error for s in samples if s.error), None)
        return Answer(
            "refused" if refused > errors else "error",
            n_ok=n_ok, samples=raw, usage=usage, latency_ms=latency_ms, error=first_error,
        )

    shares = {key: sum(s.answer == key for s in oks) / n_ok for key in keys}
    # max() keeps the first maximum, so ties go to the earliest option
    top = max(keys, key=lambda key: shares[key])

    if question.type == "noul":
        answer: str | bool | int = shares["true"] >= 0.5
    elif question.type == "score":
        answer = int(top)
    else:
        answer = top

    result = Answer("ok", answer=answer, n_ok=n_ok, samples=raw, usage=usage, latency_ms=latency_ms)
    if with_probabilities:
        result.probabilities = shares
        if question.type != "noul":
            result.confidence = shares[top]
        if question.type == "score":
            result.expected_score = sum(int(key) * p for key, p in shares.items())
    return result
