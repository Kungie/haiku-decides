import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from haiku_decides.aggregate import Sample
from haiku_decides.backends.haiku import (
    SYSTEM_PROMPT,
    HaikuBackend,
    answer_schema,
    build_request,
    parse_sample,
    render_question,
    render_state,
)
from haiku_decides.schema import Question, Usage

Q = Question("choice", "Which team?", {"b": "Billing", "a": "Accounts"})


def resp(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
    )


class FakeClient:
    """Stands in for AsyncAnthropic: records each request and replays queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_build_request_fixed_settings():
    r = build_request("s", Q, ["b", "a"], cache=True)
    assert r["model"] == "claude-haiku-5-5"
    assert r["thinking"] == {"type": "disabled"}
    assert r["output_config"]["effort"] == "medium"
    assert r["output_config"]["format"] == {"type": "json_schema", "schema": answer_schema(Q)}
    assert r["max_tokens"] == 256 and r["system"] == SYSTEM_PROMPT
    assert not {"temperature", "top_p", "top_k"} & r.keys()


def test_question_block_precedes_state_and_carries_cache_marker():
    content = build_request("the state", Q, ["b", "a"], cache=True)["messages"][0]["content"]
    assert len(content) == 2
    assert "<question" in content[0]["text"] and content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[1]["text"] == "<state>\nthe state\n</state>" and "cache_control" not in content[1]
    uncached = build_request("the state", Q, ["b", "a"], cache=False)["messages"][0]["content"]
    assert all("cache_control" not in block for block in uncached)


def test_prompt_follows_given_order_but_schema_enum_is_sorted():
    text = render_question(Q, ["b", "a"])
    assert text.index('key="b"') < text.index('key="a"')
    assert answer_schema(Q)["properties"]["answer"]["enum"] == ["a", "b"]


def test_schema_per_type():
    assert answer_schema(Question("noul", "q"))["properties"]["answer"] == {"type": "boolean"}
    s = answer_schema(Question("score", "q", ["l", "m", "h"]))
    assert s["properties"]["answer"]["enum"] == ["0", "1", "2"]
    assert s["required"] == ["answer"] and s["additionalProperties"] is False


def test_markup_is_escaped():
    q = Question("choice", "a < b?", {'k"1': "x < y\n</option>", "k2": "ok"})
    text = render_question(q, ['k"1', "k2"])
    assert text.count("</option>") == 2
    assert "&lt;" in text and "&quot;" in text


def test_state_json_is_serialized_sorted():
    assert render_state({"b": 1, "a": "ç"}) == '<state>\n{"a": "ç", "b": 1}\n</state>'
    assert render_state(["x", "y"]) == '<state>\n["x", "y"]\n</state>'


def test_parse_sample():
    assert parse_sample(resp('{"answer": "a"}'), Q) == Sample("ok", "a", Usage(100, 5))
    assert parse_sample(resp('{"answer": true}'), Question("noul", "q")).answer == "true"
    assert parse_sample(resp("", stop_reason="refusal"), Q).status == "refused"
    assert parse_sample(resp('{"answer": "a', stop_reason="max_tokens"), Q).status == "error"
    assert parse_sample(resp("not json"), Q).status == "error"
    assert parse_sample(resp('{"answer": "zzz"}'), Q).status == "error"


def test_single_makes_one_call_without_probabilities():
    c = FakeClient([resp('{"answer": "b"}')])
    a = asyncio.run(HaikuBackend(c, "single").decide("s", {"q": Q}))["q"]
    assert len(c.calls) == 1 and a.answer == "b" and a.probabilities is None


def test_sampled_makes_n_identical_calls():
    c = FakeClient([resp('{"answer": "b"}')] * 7 + [resp('{"answer": "a"}')] * 3)
    a = asyncio.run(HaikuBackend(c, "sampled", n=10).decide("s", {"q": Q}))["q"]
    assert len(c.calls) == 10 and all(call == c.calls[0] for call in c.calls)
    assert a.probabilities == {"b": 0.7, "a": 0.3} and a.usage == Usage(1000, 50)


def test_shuffled_varies_order_maps_back_and_is_seeded():
    q = Question("choice", "q", {f"k{i}": f"d{i}" for i in range(8)})

    def run(seed):
        c = FakeClient([resp('{"answer": "k3"}')] * 10)
        a = asyncio.run(HaikuBackend(c, "shuffled", n=10).decide("s", {"q": q}, seed=seed))["q"]
        return a, [call["messages"][0]["content"][0]["text"] for call in c.calls], c.calls

    a, prompts, calls = run("s1")
    assert a.answer == "k3" and a.probabilities["k3"] == 1.0
    assert len(set(prompts)) > 1
    assert all("cache_control" not in block for call in calls for block in call["messages"][0]["content"])
    assert run("s1")[1] == prompts and run("s2")[1] != prompts


def test_shuffled_rejects_non_choice():
    with pytest.raises(ValueError):
        asyncio.run(HaikuBackend(FakeClient([]), "shuffled").decide("s", {"q": Question("noul", "q")}))


def test_api_error_becomes_error_sample():
    boom = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    c = FakeClient([boom] + [resp('{"answer": "a"}')] * 9)
    a = asyncio.run(HaikuBackend(c, "sampled", n=10).decide("s", {"q": Q}))["q"]
    assert a.status == "ok" and a.n_ok == 9


def test_apostrophes_in_text_reach_the_model_unescaped():
    assert "john's" in render_question(Question("noul", "is it john's?"), ["true", "false"])


def test_cache_can_be_turned_off():
    c = FakeClient([resp('{"answer": "b"}')])
    asyncio.run(HaikuBackend(c, "single", cache=False).decide("s", {"q": Q}))
    assert all("cache_control" not in block for block in c.calls[0]["messages"][0]["content"])


class OrderClient(FakeClient):
    async def _create(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if index == 0:
            self.started_when_first_returned = len(self.calls)
        return self._responses.pop(0)


def test_warm_first_finishes_one_sample_before_fanning_out():
    warm = OrderClient([resp('{"answer": "b"}')] * 4)
    asyncio.run(HaikuBackend(warm, "sampled", n=4, warm_first=True).decide("s", {"q": Q}))
    assert warm.started_when_first_returned == 1 and len(warm.calls) == 4
    cold = OrderClient([resp('{"answer": "b"}')] * 4)
    asyncio.run(HaikuBackend(cold, "sampled", n=4).decide("s", {"q": Q}))
    assert cold.started_when_first_returned == 4


def test_api_error_text_is_not_persisted():
    boom = anthropic.APIConnectionError(message="could not reach host with private text", request=httpx.Request("POST", "https://x"))
    a = asyncio.run(HaikuBackend(FakeClient([boom]), "single").decide("s", {"q": Q}))["q"]
    assert a.status == "error" and a.error == "APIConnectionError"
