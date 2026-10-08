import os

from haiku_decides import cli
from haiku_decides.cli import load_env_file, make_backends
from haiku_decides.datasets import DATASETS, Item
from haiku_decides.runner import RunSummary
from haiku_decides.schema import Question


import pytest


@pytest.fixture(autouse=True)
def _away_from_the_real_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nA_KEY=from_file\nB_KEY = spaced \n\n")
    monkeypatch.setenv("A_KEY", "from_env")
    monkeypatch.delenv("B_KEY", raising=False)
    load_env_file(f)
    assert os.environ["A_KEY"] == "from_env" and os.environ["B_KEY"] == "spaced"
    load_env_file(tmp_path / "missing")


def test_make_backends_names(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert [b.name for b in make_backends("haiku", "all", 10)] == ["haiku-single", "haiku-sampled", "haiku-shuffled"]
    assert [b.name for b in make_backends("haiku", "sampled", 10)] == ["haiku-sampled"]


def test_run_skips_inapplicable_and_sets_concurrency(tmp_path, monkeypatch):
    calls, settings = [], {}

    async def fake_run_pass(backend, items, pass_name, out_path, *, decision_concurrency, meta, limit=None):
        calls.append((backend.name, pass_name, out_path.stem, decision_concurrency, limit))
        settings[(backend.name, pass_name)] = (backend.cache, backend.warm_first)
        return RunSummary(0, 0, 0)

    def fake_load_items(name, **kw):
        question_type = DATASETS[name].question_type
        criteria = {"a": "A", "b": "B"} if question_type == "choice" else None
        return [Item(f"{name}:0", "s", Question(question_type, "q", criteria), "a")]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(cli, "run_pass", fake_run_pass)
    monkeypatch.setattr(cli, "load_items", fake_load_items)
    argv = ["run", "--backend", "haiku", "--concurrency", "20", "--limit", "5", "--results", str(tmp_path)]
    assert cli.main(argv) == 0
    assert ("haiku-shuffled", "main", "boolq", 2, 5) not in calls
    assert ("haiku-shuffled", "main", "banking77", 2, 5) in calls
    assert ("haiku-single", "main", "sst5", 20, 5) in calls
    assert not [c for c in calls if c[1] == "order" and c[2] in ("boolq", "sms_spam", "sst5")]
    # reordered prompts are unique, so the order pass must not pay for cache writes it cannot read back
    assert settings[("haiku-single", "order")] == (False, False)
    assert settings[("haiku-sampled", "order")] == (True, True)
    assert settings[("haiku-sampled", "main")] == (True, False)
    assert settings[("haiku-single", "main")] == (True, False)


def test_missing_key_exits_2_and_names_variable(monkeypatch, capsys, tmp_path):
    for k in ("JEV_API_KEY", "JEV_BASE_URL", "JEV_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "--backend", "jev"]) == 2
    assert "JEV_API_KEY" in capsys.readouterr().err


def test_report_needs_no_keys(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "JEV_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cli, "load_items", lambda name, **kw: [])
    monkeypatch.setattr(cli, "PRICES_PATH", cli.PRICES_PATH.resolve())
    monkeypatch.chdir(tmp_path)
    out, chart = tmp_path / "s.md", tmp_path / "s.png"
    argv = ["report", "--results", str(tmp_path), "--out", str(out), "--chart", str(chart), "--location", "X"]
    assert cli.main(argv) == 0
    assert out.exists()


def test_haiku_needs_credentials_up_front(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_backends("haiku", "single", 10)
