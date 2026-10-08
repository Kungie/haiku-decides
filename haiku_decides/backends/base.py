from __future__ import annotations

from typing import Protocol

from ..schema import Answer, Question


class Backend(Protocol):
    name: str

    async def decide(
        self, state: str | dict | list, questions: dict[str, Question], *, seed: str = "0"
    ) -> dict[str, Answer]: ...
