"""Claude Haiku 5.5 emulating a decision model in three modes.

single    one call, answer only
sampled   N identical calls, vote shares become probabilities
shuffled  N calls with the options listed in a different order each time
"""

from __future__ import annotations

import asyncio
import html
import json
import time
from typing import Literal

import anthropic

from ..aggregate import Sample, aggregate
from ..ordering import permuted
from ..schema import Answer, Question, Usage, option_keys, validate_question

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 256
SYSTEM_PROMPT = (
    "You answer a closed question about the state the user provides. "
    "Read the question and its options, then read the state. "
    "Reply with the single best answer and nothing else. "
    "Base the answer only on the state."
)

Mode = Literal["single", "sampled", "shuffled"]


def _attr(text: object) -> str:
    return html.escape(str(text), quote=True)


def _text(text: object) -> str:
    # quotes are harmless in element text; escaping them would only garble what the model reads
    return html.escape(str(text), quote=False)


def render_question(question: Question, order: list[str]) -> str:
    lines = [
        f'<question type="{question.type}">',
        f"<instructions>{_text(question.instructions)}</instructions>",
    ]
    if question.type == "choice":
        lines.append("<options>")
        lines += [f'<option key="{_attr(key)}">{_text(question.criteria[key])}</option>' for key in order]
        lines.append("</options>")
    elif question.type == "score":
        # levels are ordered, so they are never shuffled
        lines.append("<levels>")
        lines += [f'<level index="{i}">{_text(level)}</level>' for i, level in enumerate(question.criteria)]
        lines.append("</levels>")
    elif question.criteria:
        lines.append(f"<true>{_text(question.criteria['true'])}</true>")
        lines.append(f"<false>{_text(question.criteria['false'])}</false>")
    lines.append("</question>")
    return "\n".join(lines)


def render_state(state: str | dict | list) -> str:
    body = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return f"<state>\n{body}\n</state>"


def answer_schema(question: Question) -> dict:
    if question.type == "noul":
        answer: dict = {"type": "boolean"}
    else:
        # the enum stays sorted so the schema is identical on every call;
        # only the option list in the prompt is ever shuffled
        answer = {"type": "string", "enum": sorted(option_keys(question))}
    return {
        "type": "object",
        "properties": {"answer": answer},
        "required": ["answer"],
        "additionalProperties": False,
    }


def build_request(
    state: str | dict | list,
    question: Question,
    order: list[str],
    *,
    effort: str = "medium",
    cache: bool,
    cache_prefix: bool = False,
) -> dict:
    """`cache` marks the question block. `cache_prefix` marks the system prompt instead, which
    caches the constant part before it (the output schema and the system prompt) when the
    question block itself changes on every call."""
    question_block: dict = {"type": "text", "text": render_question(question, order)}
    if cache:
        question_block["cache_control"] = {"type": "ephemeral"}
    system: str | list = SYSTEM_PROMPT
    if cache_prefix:
        system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "thinking": {"type": "disabled"},
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": answer_schema(question)},
        },
        "messages": [
            {"role": "user", "content": [question_block, {"type": "text", "text": render_state(state)}]}
        ],
    }


def _usage(response) -> Usage:
    u = response.usage
    return Usage(
        getattr(u, "input_tokens", 0) or 0,
        getattr(u, "output_tokens", 0) or 0,
        getattr(u, "cache_read_input_tokens", 0) or 0,
        getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def parse_sample(response, question: Question) -> Sample:
    usage = _usage(response)
    if response.stop_reason == "refusal":
        return Sample("refused", usage=usage)
    if response.stop_reason == "max_tokens":
        return Sample("error", usage=usage, error="output truncated at max_tokens")
    text = next((block.text for block in response.content if block.type == "text"), "")
    try:
        value = json.loads(text)["answer"]
    except (ValueError, KeyError, TypeError):
        return Sample("error", usage=usage, error=f"unparseable output: {text[:80]!r}")
    key = str(value).lower() if isinstance(value, bool) else str(value)
    if key not in option_keys(question):
        return Sample("error", usage=usage, error=f"answer outside the option set: {key!r}")
    return Sample("ok", key, usage)


class HaikuBackend:
    def __init__(
        self,
        client,
        mode: Mode,
        n: int = 10,
        effort: str = "medium",
        *,
        cache: bool = True,
        warm_first: bool = False,
    ):
        """`cache` marks the question block for prompt caching (never in shuffled mode).

        `warm_first` finishes one sample before starting the rest, so they read the cache
        the first one wrote. Parallel requests cannot read each other's cache writes.
        """
        self.cache = cache
        self.warm_first = warm_first
        self.client = client
        self.mode = mode
        self.n = 1 if mode == "single" else n
        self.effort = effort
        self.name = f"haiku-{mode}"

    async def decide(
        self, state: str | dict | list, questions: dict[str, Question], *, seed: str = "0"
    ) -> dict[str, Answer]:
        answers = await asyncio.gather(*(self._decide_one(state, q, seed) for q in questions.values()))
        return dict(zip(questions, answers))

    async def _decide_one(self, state, question: Question, seed: str) -> Answer:
        validate_question(question)
        if self.mode == "shuffled" and question.type != "choice":
            raise ValueError("shuffled mode only applies to choice questions")
        keys = option_keys(question)
        if self.mode == "shuffled":
            orders = [permuted(keys, f"{seed}:{i}") for i in range(self.n)]
        else:
            orders = [keys] * self.n
        # a shuffled question block differs on every call, so there is nothing to cache
        cache = self.cache and self.mode != "shuffled"
        started = time.perf_counter()
        first = []
        if self.warm_first and cache and len(orders) > 1:
            first = [await self._sample(state, question, orders[0], cache)]
            orders = orders[1:]
        samples = first + list(
            await asyncio.gather(*(self._sample(state, question, order, cache) for order in orders))
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return aggregate(question, list(samples), with_probabilities=self.mode != "single", latency_ms=latency_ms)

    async def _sample(self, state, question: Question, order: list[str], cache: bool) -> Sample:
        # when the question block cannot be cached, the constant prefix before it still can
        request = build_request(
            state, question, order, effort=self.effort, cache=cache, cache_prefix=not cache
        )
        try:
            response = await self.client.messages.create(**request)
        except anthropic.APIError as e:
            # the type and status only: an API error message can quote the request
            status = getattr(e, "status_code", None)
            return Sample("error", error=type(e).__name__ + (f" {status}" if status else ""))
        return parse_sample(response, question)
