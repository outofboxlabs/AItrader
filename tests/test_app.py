import io
import json
from datetime import date

import pytest

import app as app_mod
import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.config, "POSITIONS_PATH", str(tmp_path / "positions.json"))
    monkeypatch.setattr(app_mod.config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(app_mod.config, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(app_mod.config, "EXPORTS_DIR", str(tmp_path / "exports"))
    app_mod.app.config.update(TESTING=True)
    with app_mod.app.test_client() as c:
        yield c


# --- positions ---------------------------------------------------------


def test_get_positions_empty_when_file_missing(client):
    res = client.get("/api/positions")
    assert res.status_code == 200
    assert res.get_json() == []


def test_save_and_get_positions_roundtrip(client):
    payload = [{"asset_type": "shares", "ticker": "AAPL", "entry_price": 200.0, "contracts": 10, "entry_date": "2025-01-01"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    assert res.get_json()["count"] == 1
    assert client.get("/api/positions").get_json()[0]["ticker"] == "AAPL"


def test_save_positions_rejects_invalid_row(client):
    payload = [{"asset_type": "not-a-real-type", "ticker": "AAPL"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 400


def test_save_positions_backs_up_previous_file(client, tmp_path):
    positions_path = tmp_path / "positions.json"
    positions_path.write_text('[{"asset_type": "shares", "ticker": "OLD", "entry_price": 1, "contracts": 1, "entry_date": "2025-01-01"}]')
    payload = [{"asset_type": "shares", "ticker": "NEW", "entry_price": 2, "contracts": 2, "entry_date": "2025-01-02"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    assert (tmp_path / "positions.json.bak").exists()
    assert "OLD" in (tmp_path / "positions.json.bak").read_text()


def test_parse_screenshot_requires_image(client):
    res = client.post("/api/parse-screenshot", data={"provider": "anthropic"})
    assert res.status_code == 400


def test_parse_screenshot_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    data = {"image": (io.BytesIO(b"fake"), "screenshot.png"), "provider": "anthropic"}
    res = client.post("/api/parse-screenshot", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "No saved API key" in res.get_json()["error"]


def test_parse_screenshot_returns_extracted_positions(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(
        app_mod.vision,
        "extract_positions_from_image",
        lambda image_bytes, media_type, provider, model, api_key=None: [{"ticker": "AAPL"}],
    )
    data = {"image": (io.BytesIO(b"fake"), "screenshot.png"), "provider": "anthropic"}
    res = client.post("/api/parse-screenshot", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["positions"] == [{"ticker": "AAPL"}]


# --- run-analysis --------------------------------------------------------


def test_run_analysis_returns_pipeline_result(client, monkeypatch, tmp_path):
    (tmp_path / "positions.json").write_text(
        '[{"asset_type": "shares", "ticker": "AAPL", "entry_price": 200.0, "contracts": 5, "entry_date": "2025-01-01"}]'
    )
    monkeypatch.setattr(
        app_mod.pipeline,
        "run_full_analysis",
        lambda **kw: {"asof_date": "2025-05-01", "valuations": [], "allocation": {"total_value": 0}},
    )
    res = client.post("/api/run-analysis", data=json.dumps({"asof": "2025-05-01", "skip_news": True}), content_type="application/json")
    assert res.status_code == 200
    assert res.get_json()["asof_date"] == "2025-05-01"


def test_run_analysis_degrades_when_no_api_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    captured = {}

    def fake_run(**kw):
        captured.update(kw)
        return {"ok": True}

    monkeypatch.setattr(app_mod.pipeline, "run_full_analysis", fake_run)
    res = client.post("/api/run-analysis", data=json.dumps({}), content_type="application/json")
    assert res.status_code == 200
    assert captured["skip_news"] is True  # no key found -> degrades rather than failing


def test_run_analysis_handles_pipeline_failure(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(app_mod.pipeline, "run_full_analysis", boom)
    res = client.post("/api/run-analysis", data=json.dumps({"skip_news": True}), content_type="application/json")
    assert res.status_code == 502
    assert "error" in res.get_json()


# --- movers --------------------------------------------------------------


def test_get_movers_empty_when_no_scan_yet(client):
    res = client.get("/api/movers")
    assert res.status_code == 200
    assert res.get_json() == {"asof_date": None, "movers": []}


def test_run_movers_now_returns_results(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(
        app_mod.movers,
        "run_movers_scan",
        lambda conn, asof_date, provider, model, api_key=None, macro_score=None: [
            {"ticker": "ACME", "pct_change": -45.0, "rebound": {"analyst_sentiment": "neutral"}}
        ],
    )
    res = client.post("/api/movers/run", data=json.dumps({"provider": "anthropic"}), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["movers"][0]["ticker"] == "ACME"


def test_run_movers_now_rejects_unknown_provider(client):
    res = client.post("/api/movers/run", data=json.dumps({"provider": "not-real"}), content_type="application/json")
    assert res.status_code == 400


def test_get_movers_after_run_reads_from_db(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")

    def fake_scan(conn, asof_date, provider, model, api_key=None, macro_score=None):
        from portfolio_monitor import db as db_mod

        db_mod.save_market_movers(conn, asof_date.isoformat(), [{"ticker": "ACME", "name": "Acme", "pct_change": -45.0, "price": 10.0, "volume": 1, "market_cap": 1}])
        return [{"ticker": "ACME", "pct_change": -45.0, "rebound": {}}]

    monkeypatch.setattr(app_mod.movers, "run_movers_scan", fake_scan)
    client.post("/api/movers/run", data=json.dumps({"provider": "anthropic"}), content_type="application/json")

    res = client.get("/api/movers")
    data = res.get_json()
    assert data["asof_date"] == date.today().isoformat()
    assert data["movers"][0]["ticker"] == "ACME"


# --- scheduler -------------------------------------------------------------


def test_scheduler_status_not_running_by_default(client):
    app_mod.scheduler.stop_scheduler()
    res = client.get("/api/scheduler/status")
    assert res.get_json() == {"running": False, "next_run": None}


# --- settings --------------------------------------------------------------


def test_has_key_reports_false_when_none_saved(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    res = client.get("/api/settings/has-key?provider=anthropic")
    assert res.get_json() == {"has_key": False}


def test_save_api_key_and_then_has_key(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CREDENTIALS_PATH", str(tmp_path / ".credentials.json"))
    res = client.post("/api/settings/api-key", data=json.dumps({"provider": "anthropic", "api_key": "sk-abc"}), content_type="application/json")
    assert res.status_code == 200

    res = client.get("/api/settings/has-key?provider=anthropic")
    assert res.get_json() == {"has_key": True}


def test_save_api_key_rejects_unknown_provider(client):
    res = client.post("/api/settings/api-key", data=json.dumps({"provider": "nope", "api_key": "x"}), content_type="application/json")
    assert res.status_code == 400


def test_get_models_requires_saved_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    res = client.get("/api/models?provider=anthropic")
    assert res.status_code == 400


def test_get_models_returns_list(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(app_mod.news, "list_models", lambda provider, api_key=None: ["model-a", "model-b"])
    res = client.get("/api/models?provider=anthropic")
    assert res.status_code == 200
    assert res.get_json()["models"] == ["model-a", "model-b"]


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Portfolio Dashboard" in res.data
    assert b"{{" not in res.data  # no leftover unrendered Jinja


# --- growth screener -----------------------------------------------------


def test_get_growth_empty_when_no_scan_yet(client):
    res = client.get("/api/growth")
    assert res.status_code == 200
    assert res.get_json() == {"asof_date": None, "candidates": []}


def test_run_growth_now_returns_and_persists_results(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        app_mod.growth_screener,
        "find_growth_candidates",
        lambda **kw: [
            {
                "ticker": "STRONG",
                "name": "Strong Co",
                "price": 100.0,
                "target_mean": 170.0,
                "target_upside_pct": 70.0,
                "analyst_ratings": {"strongBuy": 9, "buy": 1},
                "pct_from_52w_high": -16.7,
                "pct_from_52w_low": 66.7,
                "market_cap": 5e9,
            }
        ],
    )
    res = client.post("/api/growth/run")
    assert res.status_code == 200
    data = res.get_json()
    assert data["candidates"][0]["ticker"] == "STRONG"
    assert data["asof_date"] == date.today().isoformat()

    # Persisted -- a fresh GET reads it back from the db.
    res2 = client.get("/api/growth")
    data2 = res2.get_json()
    assert data2["candidates"][0]["ticker"] == "STRONG"
    assert data2["candidates"][0]["analyst_ratings"]["strongBuy"] == 9

    # And a CSV landed in the configured exports dir under top_growth/.
    growth_export_dir = tmp_path / "exports" / "top_growth"
    assert growth_export_dir.exists()
    assert len(list(growth_export_dir.glob("*.csv"))) == 1


def test_run_growth_now_handles_screener_failure(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("yahoo screener down")

    monkeypatch.setattr(app_mod.growth_screener, "find_growth_candidates", boom)
    res = client.post("/api/growth/run")
    assert res.status_code == 502
    assert "error" in res.get_json()


def test_run_movers_now_writes_csv_export(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(
        app_mod.movers,
        "run_movers_scan",
        lambda conn, asof_date, provider, model, api_key=None, macro_score=None: [{"ticker": "ACME", "pct_change": -45.0}],
    )
    res = client.post("/api/movers/run", data=json.dumps({"provider": "anthropic"}), content_type="application/json")
    assert res.status_code == 200

    movers_export_dir = tmp_path / "exports" / "top_movers"
    assert movers_export_dir.exists()
    assert len(list(movers_export_dir.glob("*.csv"))) == 1
