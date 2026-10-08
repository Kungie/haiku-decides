"""OpenAI's Decisions API (public beta), passed straight through.

Request and response shapes follow OpenAI's Decisions guide: questions are a
list, yes/no questions are called "predicate", and probabilities come back as
lists of {value, probability}.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx

from ..schema import Answer, Question
from ._answers import all_failed, choice_answer, finish, noul_answer, score_answer
from ._http import post_json

MODEL = "gpt-6-luna"


def _question(name: str, q: Question) -> dict:
    if q.type == "choice":
        return {
            "name": name,
            "type": "choice",
            "instructions": q.instructions,
            "choices": [{"value": key, "description": text} for key, text in q.criteria.items()],
        }
    if q.type == "score":
        return {
            "name": name,
            "type": "score",
            "instructions": q.instructions,
            "levels": [{"label": level, "description": level} for level in q.criteria],
        }
    instructions = q.instructions
    if q.criteria:
        # predicate questions have no criteria field, so the meanings ride along in the instructions
        instructions += f" Answer true if: {q.criteria['true']}. Answer false if: {q.criteria['false']}."
    return {"name": name, "type": "predicate", "instructions": instructions}


def build_payload(state: str | dict | list, questions: dict[str, Question], model: str = MODEL) -> dict:
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return {"model": model, "input": text, "questions": [_question(name, q) for name, q in questions.items()]}


def _map(question: Question, raw: dict | None) -> Answer:
    if raw is None:
        return Answer("error", error="no answer returned for this question")
    if raw.get("type") == "refusal":
        return Answer("refused")
    try:
        if question.type == "noul":
            return noul_answer(float(raw["probability"]))
        probabilities = {str(p["value"]): p["probability"] for p in raw.get("probabilities") or []}
        if question.type == "choice":
            return choice_answer(question, raw["choice"], probabilities, raw["confidence"])
        return score_answer(question, raw["score"], probabilities, raw["confidence"])
    except (KeyError, TypeError, ValueError) as e:
        return Answer("error", error=f"malformed answer: {e!r}")


def parse_response(body: dict, questions: dict[str, Question], latency_ms: float) -> dict[str, Answer]:
    by_name = {raw.get("name"): raw for raw in body.get("answers") or []}
    answers = {name: _map(q, by_name.get(name)) for name, q in questions.items()}
    # the guide documents no usage object; read it if the API sends one
    usage = body.get("usage") or {}
    return finish(answers, latency_ms, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


class OpenAIDecisionsBackend:
    name = "openai"

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com",
        path: str = "/v1/decisions",
        model: str = MODEL,
        max_attempts: int = 3,
        sleep=asyncio.sleep,
    ):
        self.http = http
        self.url = base_url.rstrip("/") + path
        self.api_key = api_key
        self.model = model
        self.max_attempts = max_attempts
        self.sleep = sleep

    @classmethod
    def from_env(cls, http: httpx.AsyncClient) -> OpenAIDecisionsBackend:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        return cls(
            http,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com",
        )

    async def decide(
        self, state: str | dict | list, questions: dict[str, Question], *, seed: str = "0"
    ) -> dict[str, Answer]:
        started = time.perf_counter()
        body, error = await post_json(
            self.http,
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=build_payload(state, questions, self.model),
            max_attempts=self.max_attempts,
            sleep=self.sleep,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if body is None:
            return all_failed(questions, error, latency_ms)
        return parse_response(body, questions, latency_ms)
