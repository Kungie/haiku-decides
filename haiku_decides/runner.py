"""Run a backend over a dataset and append one JSON line per decision.

Three passes:
  main     every item once
  order    choice items under several option orderings (order sensitivity)
  latency  a small sequential pass, so concurrency does not distort timings
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .backends.base import Backend
from .datasets import Item
from .ordering import permute_criteria
from .schema import Answer, QuestionType

PASSES = ("main", "order", "latency")
ORDER_ITEMS = 200
ORDER_PERMUTATIONS = 5
LATENCY_ITEMS = 100
SHUFFLED = "haiku-shuffled"
QUESTION_NAME = "q"


@dataclass
class RunSummary:
    done: int
    skipped: int
    errors: int


def result_path(root: Path, system: str, pass_name: str, dataset: str) -> Path:
    return Path(root) / system / pass_name / f"{dataset}.jsonl"


def load_records(path: Path) -> dict[tuple[str, int], dict]:
    """Last record wins per (item_id, permutation_id); undecodable lines are skipped."""
    records: dict[tuple[str, int], dict] = {}
    if not Path(path).exists():
        return records
    for line in Path(path).read_text().splitlines():
        try:
            record = json.loads(line)
            records[(record["item_id"], record["permutation_id"])] = record
        except (ValueError, KeyError, TypeError):
            # a run killed mid-write leaves a partial last line
            continue
    return records


def planned_keys(items: list[Item], pass_name: str, limit: int | None = None) -> list[tuple[Item, int]]:
    if limit is not None:
        items = items[:limit]
    if pass_name == "order":
        return [(item, pid) for item in items[:ORDER_ITEMS] for pid in range(1, ORDER_PERMUTATIONS + 1)]
    if pass_name == "latency":
        return [(item, 0) for item in items[:LATENCY_ITEMS]]
    return [(item, 0) for item in items]


def applicable(system: str, question_type: QuestionType, pass_name: str) -> bool:
    if system == SHUFFLED and question_type != "choice":
        return False
    if pass_name == "order" and question_type != "choice":
        return False
    return True


def _end_with_newline(path: Path) -> None:
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as f:
            f.write("\n")


async def run_pass(
    backend: Backend,
    items: list[Item],
    pass_name: str,
    out_path: Path,
    *,
    decision_concurrency: int,
    meta: dict,
    limit: int | None = None,
) -> RunSummary:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_records(out_path)
    plan = planned_keys(items, pass_name, limit)
    todo = [
        (item, pid)
        for item, pid in plan
        if existing.get((item.item_id, pid), {}).get("status") not in ("ok", "refused")
    ]
    summary = RunSummary(done=0, skipped=len(plan) - len(todo), errors=0)
    semaphore = asyncio.Semaphore(1 if pass_name == "latency" else max(1, decision_concurrency))
    _end_with_newline(out_path)

    with out_path.open("a") as out:

        async def decide_one(item: Item, pid: int) -> None:
            seed = f"{item.item_id}:{pid}"
            question = item.question
            # the shuffled mode permutes options itself, so it gets the original order
            if pass_name == "order" and backend.name != SHUFFLED:
                question = replace(question, criteria=permute_criteria(question.criteria, seed))
            async with semaphore:
                try:
                    answers = await backend.decide(item.state, {QUESTION_NAME: question}, seed=seed)
                    answer = answers[QUESTION_NAME]
                except Exception as e:  # one bad decision must not end the run
                    answer = Answer("error", error=repr(e))
            # the record carries model output only, never dataset text
            record = {"item_id": item.item_id, "permutation_id": pid, **asdict(answer), "meta": meta}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            summary.done += 1
            summary.errors += answer.status == "error"

        await asyncio.gather(*(decide_one(item, pid) for item, pid in todo))
    return summary
