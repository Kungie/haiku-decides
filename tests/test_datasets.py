import json

import pytest

from haiku_decides.datasets import LoadedSplit, load_items, select_ids


def fake(rows, names=None):
    return lambda spec: LoadedSplit(rows, names)


def write_ids(tmp_path, name, ids):
    (tmp_path / f"{name}.json").write_text(json.dumps(ids))


def test_select_ids_is_seeded_sample():
    ids = select_ids(3080)
    assert ids == select_ids(3080) and len(ids) == 500 and len(set(ids)) == 500
    assert max(ids) < 3080 and ids != sorted(ids)
    assert sorted(select_ids(300)) == list(range(300))


def test_banking77_item(tmp_path):
    write_ids(tmp_path, "banking77", [0])
    names = ["activate_my_card", "card_not_working", "lost_card"]
    it = load_items("banking77", ids_dir=tmp_path, load=fake([{"text": "card broke", "label": 1}], names))[0]
    assert (it.item_id, it.state, it.gold) == ("banking77:0", "card broke", "card_not_working")
    assert it.question.type == "choice"
    assert it.question.criteria == {
        "activate_my_card": "activate my card",
        "card_not_working": "card not working",
        "lost_card": "lost card",
    }


def test_ag_news_item(tmp_path):
    write_ids(tmp_path, "ag_news", [0])
    it = load_items("ag_news", ids_dir=tmp_path, load=fake([{"text": "Stocks fell", "label": 2}]))[0]
    assert it.gold == "business" and list(it.question.criteria) == ["world", "sports", "business", "sci_tech"]


def test_boolq_item_uses_row_question(tmp_path):
    write_ids(tmp_path, "boolq", [0])
    row = {"question": "is it blue", "passage": "The sky is blue.", "answer": True}
    it = load_items("boolq", ids_dir=tmp_path, load=fake([row]))[0]
    assert (it.state, it.question.instructions, it.gold) == ("The sky is blue.", "is it blue", True)
    assert it.question.type == "noul" and it.question.criteria is None


def test_sms_spam_item(tmp_path):
    write_ids(tmp_path, "sms_spam", [0])
    it = load_items("sms_spam", ids_dir=tmp_path, load=fake([{"sms": "WIN a prize", "label": 1}]))[0]
    assert it.gold is True and set(it.question.criteria) == {"true", "false"}


def test_sst5_item(tmp_path):
    write_ids(tmp_path, "sst5", [0])
    it = load_items("sst5", ids_dir=tmp_path, load=fake([{"text": "great film", "label": 4}]))[0]
    assert it.gold == 4 and it.question.type == "score" and len(it.question.criteria) == 5


def test_items_follow_committed_id_order(tmp_path):
    write_ids(tmp_path, "ag_news", [2, 0])
    rows = [{"text": f"t{i}", "label": 0} for i in range(3)]
    items = load_items("ag_news", ids_dir=tmp_path, load=fake(rows))
    assert [i.item_id for i in items] == ["ag_news:2", "ag_news:0"]


def test_missing_ids_file_points_to_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="select-ids"):
        load_items("ag_news", ids_dir=tmp_path, load=fake([]))
