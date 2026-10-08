"""TypeSafe AI's Jev: a native decision model, passed straight through.

The request and response shapes follow AIMLAPI's Jev reference. Hosts differ in
endpoint path and model id, so those come from the environment.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx

from ..schema import Answer, Question
from ._answers import all_failed, choice_answer, finish, noul_answer, score_answer
from ._http import post_json


def build_payload(state: str | dict | list, questions: dict[str, Question], model: str) -> dict:
    payload_questions = {}
    for name, q in questions.items():
        entry: dict = {"type": q.type, "instructions": q.instructions}
        if q.criteria is not None:
            entry["criteria"] = q.criteria
        payload_questions[name] = entry
    return {"model": model, "state": state, "questions": payload_questions}


def _map(question: Question, raw: dict | None) -> Answer:
    if raw is None:
        return Answer("error", error="no answer returned for this question")
    try:
        if question.type == "noul":
            return noul_answer(float(raw["noul"]))
        if question.type == "choice":
            return choice_answer(question, raw["choice"], raw.get("probabilities") or {}, raw["confidence"])
        return score_answer(question, raw["score"], raw.get("probabilities") or {}, raw["confidence"])
    except (KeyError, TypeError, ValueError) as e:
        return Answer("error", error=f"malformed answer: {e!r}")


def parse_response(body: dict, questions: dict[str, Question], latency_ms: float) -> dict[str, Answer]:
    raw_answers = body.get("answers") or {}
    answers = {name: _map(q, raw_answers.get(name)) for name, q in questions.items()}
    usage = body.get("usage") or {}
    return finish(answers, latency_ms, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


class JevBackend:
    name = "jev"

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        base_url: str,
        api_key: str,
        model: str,
        path: str = "/v1/decisions",
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
    def from_env(cls, http: httpx.AsyncClient) -> JevBackend:
        for name in ("JEV_API_KEY", "JEV_BASE_URL", "JEV_MODEL"):
            if not os.environ.get(name):
                raise ValueError(f"{name} is not set")
        model = os.environ["JEV_MODEL"]
        if "latest" in model:
            raise ValueError("JEV_MODEL must pin a version, not a 'latest' alias")
        return cls(
            http,
            base_url=os.environ["JEV_BASE_URL"],
            api_key=os.environ["JEV_API_KEY"],
            model=model,
            path=os.environ.get("JEV_PATH") or "/v1/decisions",
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
