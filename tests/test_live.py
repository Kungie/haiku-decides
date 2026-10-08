"""Live smoke tests. They spend real money, so they only run with HAIKU_DECIDES_LIVE=1."""

import asyncio
import os

import httpx
import pytest
from anthropic import AsyncAnthropic

from conftest import DOC_QUESTIONS as QS
from conftest import DOC_STATE as STATE
from haiku_decides.backends.haiku import HaikuBackend
from haiku_decides.backends.jev import JevBackend
from haiku_decides.backends.openai_decisions import OpenAIDecisionsBackend
from haiku_decides.cli import load_env_file

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("HAIKU_DECIDES_LIVE") != "1", reason="set HAIKU_DECIDES_LIVE=1 to run"),
]


@pytest.fixture(autouse=True)
def _env():
    load_env_file()


def check_native(out):
    assert all(a.status == "ok" for a in out.values()), {k: a.error for k, a in out.items()}
    assert out["department"].answer in QS["department"].criteria
    assert abs(sum(out["department"].probabilities.values()) - 1) < 0.02
    assert isinstance(out["is_urgent"].answer, bool) and out["frustration"].answer in (0, 1, 2)


def test_jev_live():
    async def go():
        async with httpx.AsyncClient(timeout=60) as http:
            return await JevBackend.from_env(http).decide(STATE, QS)

    check_native(asyncio.run(go()))


def test_openai_live():
    async def go():
        async with httpx.AsyncClient(timeout=60) as http:
            return await OpenAIDecisionsBackend.from_env(http).decide(STATE, QS)

    check_native(asyncio.run(go()))


def test_haiku_sampled_live():
    async def go():
        async with AsyncAnthropic(max_retries=4) as client:
            return await HaikuBackend(client, "sampled", n=3).decide(STATE, QS)

    out = asyncio.run(go())
    assert all(a.status == "ok" and a.n_ok == 3 for a in out.values()), {k: a.error for k, a in out.items()}
    assert out["department"].answer in QS["department"].criteria
