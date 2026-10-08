import asyncio

import httpx
import pytest

from conftest import DOC_QUESTIONS as QS
from haiku_decides.backends.openai_decisions import OpenAIDecisionsBackend, build_payload, parse_response
from haiku_decides.schema import Question, Usage


def test_build_payload_matches_guide():
    assert build_payload("s", QS) == {
        "model": "gpt-6-luna",
        "input": "s",
        "questions": [
            {"name": "is_urgent", "type": "predicate", "instructions": "Does this convey urgency?"},
            {
                "name": "department",
                "type": "choice",
                "instructions": "Which team should handle this?",
                "choices": [
                    {"value": "billing", "description": "Payments, invoicing, refunds"},
                    {"value": "technical", "description": "Bugs, outages, integrations"},
                    {"value": "sales", "description": "Pricing, upgrades, new accounts"},
                ],
            },
            {
                "name": "frustration",
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "levels": [
                    {"label": "Calm", "description": "Calm"},
                    {"label": "Frustrated", "description": "Frustrated"},
                    {"label": "Very angry", "description": "Very angry"},
                ],
            },
        ],
    }


def test_noul_criteria_are_folded_into_instructions():
    q = {"spam": Question("noul", "Is this spam?", {"true": "an ad", "false": "a normal message"})}
    assert (
        build_payload("s", q)["questions"][0]["instructions"]
        == "Is this spam? Answer true if: an ad. Answer false if: a normal message."
    )


def test_json_state_is_serialized_sorted():
    assert build_payload({"b": 1, "a": 2}, QS)["input"] == '{"a": 2, "b": 1}'


def test_parse_response():
    body = {
        "answers": [
            {"type": "predicate", "name": "is_urgent", "probability": 0.96},
            {
                "type": "choice",
                "name": "department",
                "choice": "billing",
                "confidence": 0.93,
                "probabilities": [
                    {"value": "billing", "probability": 0.95},
                    {"value": "technical", "probability": 0.05},
                ],
            },
            {
                "type": "score",
                "name": "frustration",
                "score": 1.3,
                "confidence": 0.55,
                "probabilities": [
                    {"value": 0, "label": "Calm", "probability": 0.0},
                    {"value": 1, "label": "Frustrated", "probability": 0.7},
                    {"value": 2, "label": "Very angry", "probability": 0.3},
                ],
            },
        ]
    }
    out = parse_response(body, QS, latency_ms=9.0)
    assert out["is_urgent"].answer is True
    assert out["is_urgent"].probabilities == pytest.approx({"true": 0.96, "false": 0.04})
    assert (out["department"].answer, out["department"].confidence) == ("billing", 0.93)
    assert out["department"].probabilities == {"billing": 0.95, "technical": 0.05, "sales": 0.0}
    f = out["frustration"]
    assert (f.answer, f.expected_score, f.confidence) == (1, 1.3, 0.55)
    assert f.probabilities == {"0": 0.0, "1": 0.7, "2": 0.3}
    assert all(a.usage == Usage() and a.latency_ms == 9.0 for a in out.values())


def test_refusal_missing_and_unknown_choice():
    body = {
        "answers": [
            {"type": "refusal", "name": "is_urgent"},
            {"type": "choice", "name": "department", "choice": "legal", "confidence": 1, "probabilities": []},
        ]
    }
    out = parse_response(body, QS, latency_ms=0)
    assert out["is_urgent"].status == "refused"
    assert out["department"].status == "error" and out["frustration"].status == "error"


def test_usage_is_read_when_present():
    body = {
        "answers": [{"type": "predicate", "name": "is_urgent", "probability": 0.5}],
        "usage": {"input_tokens": 40, "output_tokens": 0},
    }
    assert parse_response(body, {"is_urgent": QS["is_urgent"]}, 0)["is_urgent"].usage == Usage(40, 0)


def test_auth_header_path_and_retry(no_sleep):
    seen = []
    ok = {"answers": [{"type": "predicate", "name": "is_urgent", "probability": 0.9}]}
    replies = iter([httpx.Response(500), httpx.Response(200, json=ok)])

    def handler(request):
        seen.append(request)
        return next(replies)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    b = OpenAIDecisionsBackend(http, api_key="k", sleep=no_sleep)
    out = asyncio.run(b.decide("s", {"is_urgent": QS["is_urgent"]}))
    assert len(seen) == 2 and out["is_urgent"].status == "ok"
    assert seen[0].headers["authorization"] == "Bearer k"
    assert str(seen[0].url) == "https://api.openai.com/v1/decisions"


def test_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIDecisionsBackend.from_env(httpx.AsyncClient())
