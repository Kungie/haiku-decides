import asyncio
import json

from haiku_decides.datasets import Item
from haiku_decides.runner import applicable, load_records, planned_keys, run_pass
from haiku_decides.schema import Answer, Question, Usage


class FakeBackend:
    def __init__(self, name, answer="a", fail_on=()):
        self.name = name
        self.answer = answer
        self.fail_on = set(fail_on)
        self.calls = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def decide(self, state, questions, *, seed="0"):
        self.calls.append((state, questions, seed))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        if seed.rsplit(":", 1)[0] in self.fail_on:
            raise RuntimeError("boom")
        return {name: Answer("ok", self.answer, usage=Usage(10, 1)) for name in questions}


def choice_items(n):
    q = Question("choice", "q", {"a": "A", "b": "B", "c": "C"})
    return [Item(f"d:{i}", f"SECRET-{i}", q, "a") for i in range(n)]


def run(backend, items, pass_name, path, **kw):
    return asyncio.run(run_pass(backend, items, pass_name, path, decision_concurrency=4, meta={"n": 10}, **kw))


def test_main_pass_writes_one_record_per_item_without_state_text(tmp_path):
    p = tmp_path / "out.jsonl"
    s = run(FakeBackend("jev"), choice_items(3), "main", p)
    recs = load_records(p)
    assert (s.done, s.skipped, s.errors) == (3, 0, 0)
    assert set(recs) == {("d:0", 0), ("d:1", 0), ("d:2", 0)}
    assert recs[("d:0", 0)]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert recs[("d:0", 0)]["meta"] == {"n": 10}
    assert "SECRET" not in p.read_text()


def test_resume_skips_done_and_retries_errors(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"item_id": "d:0", "permutation_id": 0, "status": "ok"})
        + "\n"
        + json.dumps({"item_id": "d:1", "permutation_id": 0, "status": "error"})
        + "\n"
    )
    b = FakeBackend("jev")
    s = run(b, choice_items(3), "main", p)
    assert (s.done, s.skipped) == (2, 1)
    assert sorted(c[2] for c in b.calls) == ["d:1:0", "d:2:0"]
    assert load_records(p)[("d:1", 0)]["status"] == "ok"


def test_truncated_last_line_is_ignored(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(json.dumps({"item_id": "d:0", "permutation_id": 0, "status": "ok"}) + '\n{"item_id": "d:1", "perm')
    assert set(load_records(p)) == {("d:0", 0)}
    assert run(FakeBackend("jev"), choice_items(2), "main", p).done == 1
    assert set(load_records(p)) == {("d:0", 0), ("d:1", 0)}


def test_planned_keys_per_pass():
    items = choice_items(500)
    assert len(planned_keys(items, "main")) == 500
    order = planned_keys(items, "order")
    assert len(order) == 1000 and {pid for _, pid in order} == {1, 2, 3, 4, 5}
    assert {it.item_id for it, _ in order} == {f"d:{i}" for i in range(200)}
    assert len(planned_keys(items, "latency")) == 100
    assert len(planned_keys(items, "order", limit=5)) == 25


def test_order_pass_permutes_criteria_except_for_shuffled(tmp_path):
    plain, shuffled = FakeBackend("haiku-sampled"), FakeBackend("haiku-shuffled")
    run(plain, choice_items(40), "order", tmp_path / "a.jsonl")
    run(shuffled, choice_items(40), "order", tmp_path / "b.jsonl")
    orders = lambda b: {tuple(call[1]["q"].criteria) for call in b.calls}
    assert len(orders(plain)) > 1 and orders(shuffled) == {("a", "b", "c")}


def test_latency_pass_is_sequential(tmp_path):
    b = FakeBackend("jev")
    run(b, choice_items(150), "latency", tmp_path / "l.jsonl")
    assert len(b.calls) == 100 and b.max_in_flight == 1


def test_applicable():
    assert not applicable("haiku-shuffled", "noul", "main")
    assert not applicable("jev", "score", "order")
    assert applicable("haiku-shuffled", "choice", "order")
    assert applicable("haiku-single", "score", "latency")


def test_backend_exception_is_recorded_and_run_continues(tmp_path):
    p = tmp_path / "out.jsonl"
    s = run(FakeBackend("jev", fail_on={"d:1"}), choice_items(3), "main", p)
    assert (s.done, s.errors) == (3, 1)
    assert load_records(p)[("d:1", 0)]["status"] == "error"
