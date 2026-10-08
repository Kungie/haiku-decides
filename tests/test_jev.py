import asyncio

import httpx
import pytest

from conftest import DOC_QUESTIONS as QS
from haiku_decides.backends.jev import JevBackend, build_payload, parse_response
from haiku_decides.schema import Usage

DOC_RESPONSE = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.96},
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.97,
            "probabilities": {"billing": 0.98, "technical": 0.02, "sales": 0},
        },
        "frustration": {
            "type": "score",
            "score": 1.3,
            "confidence": 0.55,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0, "1": 0.7, "2": 0.3},
        },
    },
    "usage": {"input_tokens": 403, "output_tokens": 73},
}


def backend(handler, no_sleep):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JevBackend(http, base_url="https://jev.test", api_key="k", model="typesafe/jev-1.13", sleep=no_sleep)


def recording(seen, replies):
    replies = iter(replies)

    def handler(request):
        seen.append(request)
        return next(replies)

    return handler


def test_build_payload_matches_documented_shape():
    p = build_payload({"k": "v"}, QS, "typesafe/jev-1.13")
    assert p["model"] == "typesafe/jev-1.13" and p["state"] == {"k": "v"}
    assert p["questions"]["is_urgent"] == {"type": "noul", "instructions": "Does this convey urgency?"}
    assert p["questions"]["department"]["criteria"] == QS["department"].criteria
    assert p["questions"]["frustration"]["criteria"] == ["Calm", "Frustrated", "Very angry"]


def test_parse_documented_response():
    out = parse_response(DOC_RESPONSE, QS, latency_ms=12.0)
    d, u, f = out["department"], out["is_urgent"], out["frustration"]
    assert (d.answer, d.confidence) == ("billing", 0.97)
    assert d.probabilities == {"billing": 0.98, "technical": 0.02, "sales": 0}
    assert u.answer is True and u.confidence is None
    assert u.probabilities == pytest.approx({"true": 0.96, "false": 0.04})
    assert (f.answer, f.expected_score, f.confidence) == (1, 1.3, 0.55)
    assert sum((a.usage for a in out.values()), Usage()) == Usage(403, 73)
    assert all(a.latency_ms == 12.0 for a in out.values())


def test_malformed_answers_become_errors():
    body = {
        "answers": {"department": {"type": "choice", "choice": "legal", "confidence": 1, "probabilities": {}}},
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    out = parse_response(body, QS, latency_ms=0)
    assert out["department"].status == "error" and out["is_urgent"].status == "error"


def test_missing_probability_keys_are_zero_filled():
    body = {
        "answers": {
            "department": {"type": "choice", "choice": "billing", "confidence": 1.0, "probabilities": {"billing": 1.0}}
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    out = parse_response(body, {"department": QS["department"]}, 0)
    assert out["department"].probabilities == {"billing": 1.0, "technical": 0.0, "sales": 0.0}


def test_auth_header_and_path(no_sleep):
    seen = []
    b = backend(recording(seen, [httpx.Response(200, json=DOC_RESPONSE)]), no_sleep)
    asyncio.run(b.decide("s", QS))
    assert seen[0].headers["authorization"] == "Bearer k" and seen[0].url.path == "/v1/decisions"


def test_retries_429_then_succeeds(no_sleep):
    seen = []
    b = backend(recording(seen, [httpx.Response(429), httpx.Response(200, json=DOC_RESPONSE)]), no_sleep)
    out = asyncio.run(b.decide("s", QS))
    assert len(seen) == 2 and out["department"].status == "ok"


def test_gives_up_after_three_5xx(no_sleep):
    seen = []
    b = backend(recording(seen, [httpx.Response(500)] * 3), no_sleep)
    out = asyncio.run(b.decide("s", QS))
    assert len(seen) == 3 and all(a.status == "error" for a in out.values())


def test_400_is_not_retried(no_sleep):
    seen = []
    b = backend(recording(seen, [httpx.Response(400)]), no_sleep)
    out = asyncio.run(b.decide("s", QS))
    assert len(seen) == 1 and out["department"].status == "error"


def test_from_env_validates(monkeypatch):
    for k in ("JEV_API_KEY", "JEV_BASE_URL", "JEV_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        JevBackend.from_env(httpx.AsyncClient())
    monkeypatch.setenv("JEV_API_KEY", "k")
    monkeypatch.setenv("JEV_BASE_URL", "https://x")
    monkeypatch.setenv("JEV_MODEL", "typesafe/jev-latest")
    with pytest.raises(ValueError, match="latest"):
        JevBackend.from_env(httpx.AsyncClient())
