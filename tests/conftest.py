import pytest

from haiku_decides.schema import Question


@pytest.fixture
def no_sleep():
    async def _no_sleep(_seconds):
        return None

    return _no_sleep


# The three-question example from the Jev and OpenAI Decisions documentation.
DOC_QUESTIONS = {
    "is_urgent": Question("noul", "Does this convey urgency?"),
    "department": Question(
        "choice",
        "Which team should handle this?",
        {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    ),
    "frustration": Question("score", "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
}
DOC_STATE = "Help! My payments have been failing for 3 days and nobody answers support."
