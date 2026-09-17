import json

from gasp.cli import main
from gasp.io import read_items, write_records


def test_read_items_jsonl(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps({"context": "c", "answer": "a", "query": "q"}) + "\n", encoding="utf-8")
    items = read_items(str(p))
    assert items[0]["answer"] == "a"


def test_read_items_requires_fields(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"context": "c"}) + "\n", encoding="utf-8")
    try:
        read_items(str(p))
        assert False, "should have raised"
    except ValueError:
        pass


def test_write_and_roundtrip_csv(tmp_path):
    recs = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "out.csv"
    write_records(recs, str(p))
    assert "a,b" in p.read_text(encoding="utf-8")


def test_cli_eval(tmp_path, capsys):
    # lower sensitivity should read as more suspicious; label 1 = unsupported
    p = tmp_path / "res.jsonl"
    rows = [{"label": 1, "sensitivity": 0.1}, {"label": 1, "sensitivity": 0.2},
            {"label": 0, "sensitivity": 0.8}, {"label": 0, "sensitivity": 0.9}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rc = main(["eval", "--input", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "roc_auc" in out and "1.0000" in out
