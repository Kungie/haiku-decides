"""Command line for haiku-decides: select-ids, run, report."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

import anthropic
import httpx

from .backends.base import Backend
from .backends.haiku import MODEL as HAIKU_MODEL
from .backends.haiku import HaikuBackend
from .backends.jev import JevBackend
from .backends.openai_decisions import OpenAIDecisionsBackend
from .datasets import DATASETS, default_load, load_items, select_ids
from .report import build_scorecard, load_prices, render_chart, render_markdown
from .runner import PASSES, applicable, result_path, run_pass

PRICES_PATH = Path(__file__).resolve().parent.parent / "prices.toml"
IDS_DIR = Path("data/ids")
HAIKU_MODES = ("single", "sampled", "shuffled")
SAMPLED_SYSTEMS = ("haiku-sampled", "haiku-shuffled")
HTTP_TIMEOUT_SECONDS = 60


def load_env_file(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into the environment without overriding what is already set."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def make_backends(backend: str, mode: str, n: int) -> list[Backend]:
    backends: list[Backend] = []
    if backend in ("jev", "openai", "all"):
        http = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        if backend in ("jev", "all"):
            backends.append(JevBackend.from_env(http))
        if backend in ("openai", "all"):
            backends.append(OpenAIDecisionsBackend.from_env(http))
    if backend in ("haiku", "all"):
        try:
            client = anthropic.AsyncAnthropic(max_retries=4)
        except anthropic.AnthropicError as e:
            raise ValueError(f"ANTHROPIC_API_KEY is not set ({e})") from e
        modes = HAIKU_MODES if mode == "all" else (mode,)
        backends += [HaikuBackend(client, m, n=n) for m in modes]
    return backends


def _select_ids(args) -> int:
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = IDS_DIR / f"{name}.json"
        if path.exists():
            print(f"{name}: {path} exists, left as is")
            continue
        spec = DATASETS[name]
        loaded = default_load(spec)
        ids = select_ids(len(loaded.rows))
        # convert one row now so a field-name mismatch fails here, not mid-run
        spec.to_item(name, ids[0], loaded.rows[ids[0]], loaded.label_names)
        path.write_text(json.dumps(ids) + "\n")
        print(f"{name}: wrote {len(ids)} row indices to {path}")
    return 0


async def _run(args) -> int:
    backends = make_backends(args.backend, args.mode, args.n)
    passes = PASSES if args.pass_name == "all" else (args.pass_name,)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    for name in names:
        items = load_items(name)
        question_type = DATASETS[name].question_type
        for backend in backends:
            sampled = backend.name in SAMPLED_SYSTEMS
            is_haiku = isinstance(backend, HaikuBackend)
            meta = {
                "model": HAIKU_MODEL if is_haiku else backend.model,
                "effort": backend.effort if is_haiku else None,
                "n": backend.n if sampled else None,
                "started_at": started_at,
            }
            # --concurrency caps API requests; a sampled decision fans out into n of them
            decision_concurrency = max(1, args.concurrency // args.n) if sampled else args.concurrency
            for pass_name in passes:
                if not applicable(backend.name, question_type, pass_name):
                    continue
                summary = await run_pass(
                    backend,
                    items,
                    pass_name,
                    result_path(Path(args.results), backend.name, pass_name, name),
                    decision_concurrency=decision_concurrency,
                    meta=meta,
                    limit=args.limit,
                )
                print(
                    f"{backend.name:15} {pass_name:8} {name:10} "
                    f"done={summary.done} skipped={summary.skipped} errors={summary.errors}",
                    flush=True,
                )
    return 0


def _report(args) -> int:
    # the report reads cached datasets only; it never reaches for the network
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    as_of, prices = load_prices(PRICES_PATH)
    items = {name: load_items(name) for name in DATASETS}
    card = build_scorecard(Path(args.results), items, prices)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_markdown(card, location=args.location, date=datetime.date.today().isoformat(), prices_as_of=as_of)
    )
    drew_chart = render_chart(card, Path(args.chart))
    print(f"wrote {out}" + (f" and {args.chart}" if drew_chart else " (no results yet, so no chart)"))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haiku-decides", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dataset_choices = [*DATASETS, "all"]

    select = sub.add_parser("select-ids", help="pick the benchmark items from each dataset")
    select.add_argument("--dataset", choices=dataset_choices, default="all")

    run = sub.add_parser("run", help="run backends over the datasets")
    run.add_argument("--backend", choices=["jev", "openai", "haiku", "all"], default="all")
    run.add_argument("--mode", choices=[*HAIKU_MODES, "all"], default="all")
    run.add_argument("--pass", dest="pass_name", choices=[*PASSES, "all"], default="all")
    run.add_argument("--dataset", choices=dataset_choices, default="all")
    run.add_argument("--n", type=int, default=10, help="samples per decision in the sampled modes")
    run.add_argument("--concurrency", type=int, default=16, help="maximum concurrent API requests")
    run.add_argument("--limit", type=int, default=None, help="only the first N items of each dataset")
    run.add_argument("--results", default="results")

    report = sub.add_parser("report", help="build the scorecard from results on disk")
    report.add_argument("--results", default="results")
    report.add_argument("--out", default="results/scorecard.md")
    report.add_argument("--chart", default="results/scorecard.png")
    report.add_argument("--location", required=True, help="where latency was measured from")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = _parser().parse_args(argv)
    try:
        if args.command == "select-ids":
            return _select_ids(args)
        if args.command == "run":
            return asyncio.run(_run(args))
        return _report(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"haiku-decides: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
