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


def rec(item_id, pid=0, status="ok", answer="a", probs=None, conf=None, latency=100.0, usage=USAGE,
        started_at="2026-10-07T10:00:00+00:00"):
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
        "meta": {"n": 10, "started_at": started_at},
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
    assert "not distinguishable" in md and "Istanbul" in md and "N=10" in md
    assert "1.000 [1.000, 1.000]" in md
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


def test_empty_scorecard_writes_no_chart(tmp_path):
    from haiku_decides.report import Scorecard

    assert render_chart(Scorecard(), tmp_path / "c.png") is False
    assert not (tmp_path / "c.png").exists()


def test_verdict_uses_the_paired_difference_not_interval_overlap(tmp_path):
    from haiku_decides.metrics import overlaps

    items = [Item(f"d:{i}", "s", CHOICE, "a") for i in range(200)]
    # jev is wrong on 20 items; haiku is wrong on 10 of those same items and never worse
    write(tmp_path, "jev", "main", "d", [rec(f"d:{i}", answer="b" if i < 20 else "a") for i in range(200)])
    write(tmp_path, "haiku-single", "main", "d", [rec(f"d:{i}", answer="b" if i < 10 else "a") for i in range(200)])
    card = build_scorecard(tmp_path, {"d": items}, PR)
    jev, haiku = card.cells[("d", "jev", "accuracy")], card.cells[("d", "haiku-single", "accuracy")]
    assert overlaps(jev.ci, haiku.ci)
    assert card.verdicts[("d", "jev", "haiku-single")] == "haiku-single higher"


def test_system_with_no_ok_rows_does_not_blank_the_others(tmp_path):
    write(tmp_path, "jev", "main", "d", [rec(f"d:{i}") for i in range(4)])
    write(tmp_path, "openai", "main", "d", [rec(f"d:{i}", status="error", answer=None) for i in range(4)])
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.comparison_size["d"] == 4
    assert card.cells[("d", "jev", "accuracy")].value == 1.0
    assert card.cells[("d", "openai", "accuracy")].note == "n/a"
    assert card.cells[("d", "openai", "coverage")].value == 0.0


def test_flip_rate_only_counts_items_with_every_ordering(tmp_path):
    write(tmp_path, "jev", "main", "d", [rec(f"d:{i}") for i in range(4)])
    write(
        tmp_path, "jev", "order", "d",
        [rec("d:0", pid=p, probs={"a": 1.0, "b": 0.0}) for p in range(1, 6)]
        + [rec("d:1", pid=1, answer="a"), rec("d:1", pid=2, answer="b")],
    )
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.cells[("d", "jev", "flip_rate")].value == 0.0


def test_cost_is_per_answered_decision(tmp_path):
    write(
        tmp_path, "jev", "main", "d",
        [rec("d:0"), rec("d:1")] + [rec(f"d:{i}", status="error", answer=None, usage=NO_USAGE) for i in (2, 3)],
    )
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.cells[("d", "jev", "usd_per_1k")].value == pytest.approx(0.021)


def test_repeat_flip_is_the_same_order_baseline(tmp_path):
    write(tmp_path, "jev", "main", "d", [rec(f"d:{i}") for i in range(4)])
    write(tmp_path, "jev", "latency", "d", [rec("d:0", answer="a"), rec("d:1", answer="b")])
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    assert card.cells[("d", "jev", "repeat_flip")].value == 0.5


def test_footer_uses_the_measurement_date(tmp_path):
    write(tmp_path, "jev", "main", "d", [rec(f"d:{i}") for i in range(4)])
    write(tmp_path, "jev", "latency", "d", [rec("d:0")])
    card = build_scorecard(tmp_path, {"d": ITEMS}, PR)
    md = render_markdown(card, location="Istanbul", date="2026-12-31", prices_as_of="2026-10-08")
    assert "on 2026-10-07" in md and "2026-12-31" not in md
