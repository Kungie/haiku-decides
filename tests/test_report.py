import json
from pathlib import Path

import pytest

from haiku_decides.datasets import Item
from haiku_decides.metrics import Prices
from haiku_decides.report import build_scorecard, load_prices, render_chart, render_markdown
from haiku_decides.runner import result_path
from haiku_decides.schema import Question

PR = {
    "claude-haiku-5-5": Prices(0.10, 0.50, 0.01, 0.125),
    "jev": Prices(0.042, 0.0, 0.0, 0.0),
    "openai-decisions": Prices(0.10, 0.0, 0.0, 0.0),
}
CHOICE = Question("choice", "q", {"a": "A", "b": "B"})
ITEMS = [Item(f"d:{i}", "s", CHOICE, "a") for i in range(4)]
USAGE = {"input_tokens": 500, "output_tokens": 50, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
NO_USAGE = dict.fromkeys(USAGE, 0)


def rec(item_id, pid=0, status="ok", answer="a", probs=None, conf=None, latency=100.0, usage=USAGE):
    return {
        "item_id": item_id,
        "permutation_id": pid,
        "status": status,
        "answer": answer,
        "probabilities": probs,
        "confidence": conf,
        "expected_score": None,
        "n_ok": None,
        "samples": None,
        "usage": usage,
        "latency_ms": latency,
        "error": None,
        "meta": {"n": 10},
    }


def write(root, system, pass_name, dataset, records):
    path = result_path(root, system, pass_name, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_comparison_set_is_intersection_of_ok_items(tmp_path):
    write(tmp_path, "jev", "main", "d", [rec(f"d:{i}", probs={"a": 1.0, "b": 0.0}, conf=1.0) for i in range(4)])
    write(
        tmp_path,
        "haiku-sampled",
        "main",
        "d",
        [rec("d:0", status="refused", answer=None)]
        + [rec(f"d:{i}", answer="b", probs={"a": 0.3, "b": 0.7}, conf=0.7) for i in (1, 2, 3)],
    )
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.comparison_size["d"] == 3
    assert card.cells[("d", "jev", "accuracy")].value == 1.0
    assert card.cells[("d", "haiku-sampled", "accuracy")].value == 0.0
    assert card.cells[("d", "haiku-sampled", "coverage")].value == 0.75
    assert card.cells[("d", "haiku-sampled", "refusal_rate")].value == 0.25
    assert card.cells[("d", "jev", "accuracy")].ci == (1.0, 1.0)
    assert card.verdicts[("d", "jev", "haiku-sampled")] == "jev higher"
    assert ("d", "haiku-single", "accuracy") not in card.cells


def test_notes_for_modes_and_types(tmp_path):
    noul_items = [Item(f"n:{i}", "s", Question("noul", "q"), True) for i in range(2)]
    write(tmp_path, "haiku-single", "main", "n", [rec(f"n:{i}", answer=True) for i in range(2)])
    card = build_scorecard(tmp_path, {"n": noul_items}, PR)
    assert card.cells[("n", "haiku-single", "ece")].note == "none"
    assert card.cells[("n", "haiku-single", "mae")].note == "n/a"
    assert card.cells[("n", "haiku-single", "flip_rate")].note == "n/a"


def test_empty_comparison_set_does_not_crash(tmp_path):
    write(tmp_path, "jev", "main", "d", [rec("d:0", status="error", answer=None)])
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.comparison_size["d"] == 0 and card.cells[("d", "jev", "accuracy")].note == "n/a"


def test_order_latency_and_cost_cells(tmp_path):
    write(tmp_path, "jev", "main", "d", [rec(f"d:{i}", probs={"a": 1.0, "b": 0.0}, conf=1.0) for i in range(4)])
    write(
        tmp_path,
        "jev",
        "order",
        "d",
        [rec("d:0", pid=p, answer="a", probs={"a": 1.0, "b": 0.0}) for p in range(1, 6)]
        + [rec("d:1", pid=p, answer="a" if p < 5 else "b", probs={"a": 1.0, "b": 0.0}) for p in range(1, 6)],
    )
    write(tmp_path, "jev", "latency", "d", [rec(f"d:{i}", latency=ms) for i, ms in enumerate([100.0, 200.0, 300.0])])
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.cells[("d", "jev", "flip_rate")].value == 0.5
    assert card.cells[("d", "jev", "tv_shift")].value == 0.0
    assert card.cells[("d", "jev", "p50_ms")].value == 200.0
    assert card.cells[("d", "jev", "usd_per_1k")].value == pytest.approx(0.021)


def test_markdown_and_chart(tmp_path):
    for system in ("jev", "haiku-sampled"):
        write(tmp_path, system, "main", "d", [rec(f"d:{i}", probs={"a": 1.0, "b": 0.0}, conf=1.0) for i in range(4)])
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    md = render_markdown(card, location="Istanbul", date="2026-10-08", prices_as_of="2026-10-08")
    assert "| jev |" in md and "| haiku-sampled |" in md
    assert "no difference" in md and "Istanbul" in md and "N=10" in md
    assert "1.000 [1.00, 1.00]" in md
    render_chart(card, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0


def test_zero_usage_makes_cost_not_applicable(tmp_path):
    write(tmp_path, "openai", "main", "d", [rec(f"d:{i}", usage=NO_USAGE) for i in range(4)])
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.cells[("d", "openai", "usd_per_1k")].note == "n/a"
    assert card.cells[("d", "openai", "accuracy")].value == 1.0


def test_load_prices():
    as_of, prices = load_prices(Path("prices.toml"))
    assert as_of == "2026-10-08" and prices["claude-haiku-5-5"] == Prices(0.10, 0.50, 0.01, 0.125)
