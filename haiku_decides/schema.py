from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QuestionType = Literal["choice", "noul", "score"]
Status = Literal["ok", "refused", "error"]

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


@dataclass(frozen=True)
class Question:
    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )


@dataclass
class Answer:
    status: Status
    # choice: option key, noul: bool, score: level index
    answer: str | bool | int | None = None
    # keyed by option_keys(question)
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    expected_score: float | None = None
    n_ok: int | None = None
    samples: list[str | None] | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    error: str | None = None


def validate_question(q: Question) -> None:
    if q.type == "choice":
        if not isinstance(q.criteria, dict) or not 2 <= len(q.criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 2 to {MAX_CHOICE_OPTIONS} options")
    elif q.type == "score":
        if not isinstance(q.criteria, list) or not 2 <= len(q.criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score needs 2 to {MAX_SCORE_LEVELS} levels")
    elif q.type == "noul":
        if q.criteria is not None and (not isinstance(q.criteria, dict) or set(q.criteria) != {"true", "false"}):
            raise ValueError('noul criteria must have exactly the keys "true" and "false"')
    else:
        raise ValueError(f"unknown question type: {q.type!r}")


def option_keys(q: Question) -> list[str]:
    if q.type == "choice":
        return list(q.criteria)
    if q.type == "score":
        return [str(i) for i in range(len(q.criteria))]
    return ["true", "false"]
