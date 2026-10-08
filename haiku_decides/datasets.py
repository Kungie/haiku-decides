"""The five benchmark datasets and the seeded selection of items from each.

Only row indices are stored in this repository. Dataset text is downloaded from
Hugging Face at run time and is never written to results.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .schema import Question, QuestionType

SEED = 20261008
SAMPLE_SIZE = 500


@dataclass(frozen=True)
class Item:
    item_id: str  # f"{dataset}:{row_index}"
    state: str
    question: Question
    gold: str | bool | int


@dataclass(frozen=True)
class LoadedSplit:
    rows: list[dict]
    label_names: list[str] | None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    revision: str  # pinned commit, so row indices keep pointing at the same rows
    split: str
    question_type: QuestionType
    license: str  # as declared on the Hugging Face dataset card
    to_item: Callable[[str, int, dict, list[str] | None], Item]


AG_NEWS_TOPICS = {
    "world": "World news and international affairs",
    "sports": "Sports",
    "business": "Business and finance",
    "sci_tech": "Science and technology",
}
SMS_SPAM_CRITERIA = {
    "true": "Unsolicited promotional, scam, or phishing message",
    "false": "Ordinary personal or transactional message",
}
SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def _banking77(name: str, index: int, row: dict, label_names: list[str] | None) -> Item:
    criteria = {label: label.replace("_", " ") for label in label_names}
    question = Question("choice", "Which intent does this customer message express?", criteria)
    return Item(f"{name}:{index}", row["text"], question, label_names[row["label"]])


def _ag_news(name: str, index: int, row: dict, label_names: list[str] | None) -> Item:
    question = Question("choice", "Which topic is this news article about?", AG_NEWS_TOPICS)
    return Item(f"{name}:{index}", row["text"], question, list(AG_NEWS_TOPICS)[row["label"]])


def _boolq(name: str, index: int, row: dict, label_names: list[str] | None) -> Item:
    return Item(f"{name}:{index}", row["passage"], Question("noul", row["question"]), bool(row["answer"]))


def _sms_spam(name: str, index: int, row: dict, label_names: list[str] | None) -> Item:
    question = Question("noul", "Is this message spam?", SMS_SPAM_CRITERIA)
    return Item(f"{name}:{index}", row["sms"], question, row["label"] == 1)


def _sst5(name: str, index: int, row: dict, label_names: list[str] | None) -> Item:
    question = Question("score", "How positive is the sentiment of this sentence?", SST5_LEVELS)
    return Item(f"{name}:{index}", row["text"], question, int(row["label"]))


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec("banking77", "legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec", "test", "choice", "cc-by-4.0", _banking77),
        DatasetSpec("ag_news", "fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "test", "choice", "unknown", _ag_news),
        DatasetSpec("boolq", "google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5", "validation", "noul", "cc-by-sa-3.0", _boolq),
        DatasetSpec("sms_spam", "ucirvine/sms_spam", "cae486f927c250fe1d4a5b55f11357964ed1646c", "train", "noul", "unknown", _sms_spam),
        DatasetSpec("sst5", "SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3", "test", "score", "not stated", _sst5),
    )
}


def select_ids(n_rows: int, k: int = SAMPLE_SIZE, seed: int = SEED) -> list[int]:
    """Seeded sample of row indices. Order is kept: passes take subsets from the front."""
    return random.Random(seed).sample(range(n_rows), min(k, n_rows))


def default_load(spec: DatasetSpec) -> LoadedSplit:
    import datasets as hf_datasets

    split = hf_datasets.load_dataset(spec.hf_id, split=spec.split, revision=spec.revision)
    label = split.features.get("label")
    label_names = list(label.names) if isinstance(label, hf_datasets.ClassLabel) else None
    return LoadedSplit(list(split), label_names)


def load_items(
    name: str,
    *,
    ids_dir: Path = Path("data/ids"),
    load: Callable[[DatasetSpec], LoadedSplit] = default_load,
) -> list[Item]:
    spec = DATASETS[name]
    ids_path = Path(ids_dir) / f"{name}.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"{ids_path} not found; run `haiku-decides select-ids` first")
    ids = json.loads(ids_path.read_text())
    loaded = load(spec)
    return [spec.to_item(name, i, loaded.rows[i], loaded.label_names) for i in ids]
