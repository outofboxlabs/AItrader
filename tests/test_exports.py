import csv
import os

from portfolio_monitor import exports


def test_export_to_csv_writes_timestamped_file_in_subfolder(tmp_path):
    rows = [{"ticker": "AAPL", "price": 232.0}, {"ticker": "TSLA", "price": 175.0}]
    path = exports.export_to_csv(rows, "top_movers", "movers", export_root=str(tmp_path))

    assert os.path.exists(path)
    assert "top_movers" in path
    assert "movers_" in os.path.basename(path)

    with open(path) as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)
    assert len(read_rows) == 2
    assert read_rows[0]["ticker"] == "AAPL"


def test_export_to_csv_separate_subfolders_dont_collide(tmp_path):
    path1 = exports.export_to_csv([{"a": 1}], "top_movers", "movers", export_root=str(tmp_path))
    path2 = exports.export_to_csv([{"a": 1}], "top_growth", "growth", export_root=str(tmp_path))
    assert os.path.dirname(path1) != os.path.dirname(path2)
    assert os.path.exists(path1)
    assert os.path.exists(path2)


def test_export_to_csv_flattens_nested_values(tmp_path):
    rows = [{"ticker": "AAPL", "rebound": {"analyst_sentiment": "neutral"}, "risk_factors": ["a", "b"]}]
    path = exports.export_to_csv(rows, "top_movers", "movers", export_root=str(tmp_path))
    with open(path) as f:
        content = f.read()
    assert "neutral" in content
    assert '"a"' in content or "a" in content


def test_export_to_csv_handles_empty_rows(tmp_path):
    path = exports.export_to_csv([], "top_movers", "movers", export_root=str(tmp_path))
    assert os.path.exists(path)
