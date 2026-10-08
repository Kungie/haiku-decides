"""Build the scorecard from result files on disk. Makes no network calls."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics
from .datasets import Item
from .metrics import Prices, bootstrap_ci, overlaps
from .runner import applicable, load_records, result_path
from .schema import Usage

SYSTEMS = ("jev", "openai", "haiku-single", "haiku-sampled", "haiku-shuffled")
REFERENCES = ("jev", "openai")
PRICE_KEY = {
    "jev": "jev",
    "openai": "openai-decisions",
    "haiku-single": "claude-haiku-5-5",
    "haiku-sampled": "claude-haiku-5-5",
    "haiku-shuffled": "claude-haiku-5-5",
}
METRICS = (
    "coverage", "refusal_rate", "accuracy", "mae", "ece", "brier",
    "flip_rate", "tv_shift", "p50_ms", "p95_ms", "usd_per_1k", "cache_read_ratio",
)
LABELS = {
    "coverage": "Coverage", "refusal_rate": "Refusals", "accuracy": "Accuracy", "mae": "MAE",
    "ece": "ECE", "brier": "Brier", "flip_rate": "Flip rate", "tv_shift": "TV shift",
    "p50_ms": "p50 ms", "p95_ms": "p95 ms", "usd_per_1k": "$ / 1k", "cache_read_ratio": "Cache read",
}
NOT_APPLICABLE = "n/a"  # does not apply, or no data
NOT_PRODUCED = "none"  # this mode produces no probabilities


@dataclass(frozen=True)
class Cell:
    value: float | None = None
    ci: tuple[float, float] | None = None
    note: str | None = None


@dataclass
class Scorecard:
    cells: dict[tuple[str, str, str], Cell] = field(default_factory=dict)  # (dataset, system, metric)
    comparison_size: dict[str, int] = field(default_factory=dict)
    verdicts: dict[tuple[str, str, str], str] = field(default_factory=dict)  # (dataset, reference, haiku system)
    n: int | None = None


def load_prices(path: Path) -> tuple[str, dict[str, Prices]]:
    data = tomllib.loads(Path(path).read_text())
    prices = {
        name: Prices(row["input"], row["output"], row["cache_read"], row["cache_write"])
        for name, row in data.items()
        if isinstance(row, dict)
    }
    return data["as_of"], prices


def _key(value) -> str:
    """The probability key for an answer or gold label: 'true'/'false', a level index, or an option key."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def _with_ci(per_item: np.ndarray) -> Cell:
    return Cell(float(per_item.mean()), bootstrap_ci(lambda idx: float(per_item[idx].mean()), len(per_item)))


def _quality_cells(system: str, question_type: str, records: list[dict], golds: list) -> dict[str, Cell]:
    """Accuracy, MAE, ECE and Brier over the comparison set."""
    na = Cell(note=NOT_APPLICABLE)
    if not records:
        return {"accuracy": na, "mae": na, "ece": na, "brier": na}
    answers = [r["answer"] for r in records]
    correct = np.array([a == g for a, g in zip(answers, golds)], dtype=float)
    cells = {"accuracy": _with_ci(correct), "mae": na}
    if question_type == "score":
        cells["mae"] = _with_ci(np.array([abs(a - g) for a, g in zip(answers, golds)], dtype=float))

    if system == "haiku-single":
        cells["ece"] = cells["brier"] = Cell(note=NOT_PRODUCED)
    elif any(not r.get("probabilities") for r in records):
        cells["ece"] = cells["brier"] = na
    else:
        probs = [r["probabilities"] for r in records]
        # confidence here is the probability of the chosen answer, not the API's own "confidence" field
        chosen = np.array([p.get(_key(a), 0.0) for p, a in zip(probs, answers)])
        cells["ece"] = Cell(
            metrics.ece(chosen, correct),
            bootstrap_ci(lambda idx: metrics.ece(chosen[idx], correct[idx]), len(records)),
        )
        per_item_brier = np.array([metrics.brier([p], [_key(g)]) for p, g in zip(probs, golds)])
        cells["brier"] = _with_ci(per_item_brier)
    return cells


def _order_cells(system: str, order_records: dict) -> dict[str, Cell]:
    answers: dict[str, list] = {}
    probs: dict[str, list] = {}
    for (item_id, _pid), r in order_records.items():
        if r.get("status") != "ok":
            continue
        answers.setdefault(item_id, []).append(r["answer"])
        if r.get("probabilities"):
            probs.setdefault(item_id, []).append(r["probabilities"])
    answers = {k: v for k, v in answers.items() if len(v) > 1}
    if not answers:
        return {"flip_rate": Cell(note=NOT_APPLICABLE), "tv_shift": Cell(note=NOT_APPLICABLE)}
    if system == "haiku-single":
        tv = Cell(note=NOT_PRODUCED)
    else:
        probs = {k: v for k, v in probs.items() if len(v) > 1}
        tv = Cell(metrics.mean_tv_distance(probs)) if probs else Cell(note=NOT_APPLICABLE)
    return {"flip_rate": Cell(metrics.flip_rate(answers)), "tv_shift": tv}


def build_scorecard(
    results_root: Path, items_by_dataset: dict[str, list[Item]], prices: dict[str, Prices]
) -> Scorecard:
    card = Scorecard()
    na = Cell(note=NOT_APPLICABLE)
    for dataset, items in items_by_dataset.items():
        mains = {}
        for system in SYSTEMS:
            path = result_path(results_root, system, "main", dataset)
            if path.exists():
                mains[system] = load_records(path)
        if not items or not mains:
            card.comparison_size[dataset] = 0
            continue
        question_type = items[0].question.type
        included = [s for s in mains if applicable(s, question_type, "main")]
        # the comparison set: items every included system answered
        common = [
            item for item in items
            if included and all(mains[s].get((item.item_id, 0), {}).get("status") == "ok" for s in included)
        ]
        card.comparison_size[dataset] = len(common)
        golds = [item.gold for item in common]

        for system, records in mains.items():
            if system not in included:
                for metric in METRICS:
                    card.cells[(dataset, system, metric)] = na
                continue
            if card.n is None:
                card.n = next((r["meta"]["n"] for r in records.values() if (r.get("meta") or {}).get("n")), None)
            cells = dict.fromkeys(METRICS, na)
            statuses = [records.get((item.item_id, 0), {}).get("status") for item in items]
            cells["coverage"] = Cell(statuses.count("ok") / len(items))
            cells["refusal_rate"] = Cell(statuses.count("refused") / len(items))
            cells.update(
                _quality_cells(system, question_type, [records[(item.item_id, 0)] for item in common], golds)
            )

            order_path = result_path(results_root, system, "order", dataset)
            if question_type == "choice" and order_path.exists():
                cells.update(_order_cells(system, load_records(order_path)))

            latency_path = result_path(results_root, system, "latency", dataset)
            if latency_path.exists():
                timings = [r["latency_ms"] for r in load_records(latency_path).values() if r.get("status") == "ok"]
                if timings:
                    cells["p50_ms"] = Cell(metrics.percentile(timings, 50))
                    cells["p95_ms"] = Cell(metrics.percentile(timings, 95))

            usages = [Usage(**r["usage"]) for r in records.values() if r.get("usage")]
            total = sum(usages, Usage())
            # an API that reports no token usage cannot be priced from its responses
            if total != Usage() and PRICE_KEY[system] in prices:
                cells["usd_per_1k"] = Cell(metrics.usd_per_1k(usages, prices[PRICE_KEY[system]]))
                cells["cache_read_ratio"] = Cell(metrics.cache_read_ratio(usages))

            for metric, cell in cells.items():
                card.cells[(dataset, system, metric)] = cell

        for reference in REFERENCES:
            ref = card.cells.get((dataset, reference, "accuracy"))
            if ref is None or ref.ci is None:
                continue
            for system in included:
                cell = card.cells[(dataset, system, "accuracy")]
                if not system.startswith("haiku") or cell.ci is None:
                    continue
                if overlaps(ref.ci, cell.ci):
                    verdict = "no difference"
                else:
                    verdict = f"{reference if ref.value > cell.value else system} higher"
                card.verdicts[(dataset, reference, system)] = verdict
    return card


def _format(metric: str, cell: Cell) -> str:
    if cell.value is None:
        return cell.note or NOT_APPLICABLE
    if cell.ci is not None:
        return f"{cell.value:.3f} [{cell.ci[0]:.2f}, {cell.ci[1]:.2f}]"
    if metric in ("p50_ms", "p95_ms"):
        return f"{cell.value:.0f}"
    if metric == "usd_per_1k":
        return f"{cell.value:.4f}"
    return f"{cell.value:.3f}"


def _datasets(card: Scorecard) -> list[str]:
    return list(dict.fromkeys(dataset for dataset, _, _ in card.cells))


def _systems(card: Scorecard, dataset: str) -> list[str]:
    return [s for s in SYSTEMS if (dataset, s, "accuracy") in card.cells]


def render_markdown(card: Scorecard, *, location: str, date: str, prices_as_of: str) -> str:
    lines = ["# Scorecard", ""]
    for dataset in _datasets(card):
        systems = _systems(card, dataset)
        # drop columns that say nothing for this dataset
        columns = [m for m in METRICS if any(card.cells[(dataset, s, m)].value is not None for s in systems)]
        lines += [f"## {dataset}", "", f"Items every system answered: {card.comparison_size[dataset]}", ""]
        lines.append("| System | " + " | ".join(LABELS[m] for m in columns) + " |")
        lines.append("|---|" + "---|" * len(columns))
        for system in systems:
            row = " | ".join(_format(m, card.cells[(dataset, system, m)]) for m in columns)
            lines.append(f"| {system} | {row} |")
        lines.append("")
        for (d, reference, system), verdict in card.verdicts.items():
            if d == dataset:
                lines.append(f"- Accuracy, {reference} vs {system}: {verdict}")
        lines.append("")
    n = card.n if card.n is not None else NOT_APPLICABLE
    lines += [
        "Brackets are 95% bootstrap intervals. `none`: the mode produces no probabilities. `n/a`: not applicable.",
        "",
        f"Measured from {location} on {date}. N={n} samples per decision in the sampled modes. "
        f"Prices as of {prices_as_of}.",
        "",
    ]
    return "\n".join(lines)


# Chart tokens. Series colors follow the system, never its rank, and each system
# also gets its own marker shape so identity never rests on color alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"
SERIES = {
    "jev": ("#2a78d6", "o"),
    "openai": ("#eb6834", "s"),
    "haiku-single": ("#1baf7a", "^"),
    "haiku-sampled": ("#eda100", "D"),
    "haiku-shuffled": ("#e87ba4", "v"),
}
PANELS = (("accuracy", "Accuracy (higher is better)"), ("ece", "Calibration error (lower is better)"))


def render_chart(card: Scorecard, path: Path) -> bool:
    """Write the chart and return True, or return False when there is nothing to plot."""
    datasets = _datasets(card)
    if not datasets:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 1.6 + 0.9 * max(len(datasets), 1)), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    step = 0.15
    for ax, (metric, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for row, dataset in enumerate(datasets):
            for slot, system in enumerate(SYSTEMS):
                cell = card.cells.get((dataset, system, metric))
                if cell is None or cell.value is None:
                    continue
                color, marker = SERIES[system]
                y = row + (slot - (len(SYSTEMS) - 1) / 2) * step
                xerr = None
                if cell.ci is not None:
                    xerr = [[max(cell.value - cell.ci[0], 0.0)], [max(cell.ci[1] - cell.value, 0.0)]]
                ax.errorbar(
                    cell.value, y, xerr=xerr, fmt=marker, color=color, ecolor=color, elinewidth=2,
                    markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=0,
                    label=system,
                )
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_yticks(range(len(datasets)), datasets)
        ax.set_ylim(len(datasets) - 0.5, -0.5)
        ax.tick_params(colors=INK_SECONDARY, length=0)
        ax.grid(axis="x", color=AXIS, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)

    # every system that appears anywhere, in fixed order
    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    ordered = [s for s in SYSTEMS if s in handles]
    if ordered:
        fig.legend(
            [handles[s] for s in ordered], ordered, loc="upper center", ncol=len(ordered),
            frameon=False, labelcolor=INK_SECONDARY, fontsize=9,
        )
    fig.text(0.01, 0.01, "Whiskers are 95% bootstrap intervals.", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True
