import json
import io

import pytest

import web_ui


@pytest.fixture
def client(tmp_path, monkeypatch):
    positions_path = tmp_path / "positions.json"
    monkeypatch.setattr(web_ui.config, "POSITIONS_PATH", str(positions_path))
    web_ui.app.config.update(TESTING=True)
    with web_ui.app.test_client() as c:
        yield c


def test_get_positions_empty_when_file_missing(client):
    res = client.get("/api/positions")
    assert res.status_code == 200
    assert res.get_json() == []


def test_save_and_get_positions_roundtrip(client):
    payload = [
        {
            "asset_type": "shares",
            "ticker": "AAPL",
            "entry_price": 200.0,
            "contracts": 10,
            "entry_date": "2025-01-01",
        }
    ]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    assert res.get_json()["count"] == 1

    res = client.get("/api/positions")
    saved = res.get_json()
    assert saved[0]["ticker"] == "AAPL"


def test_save_positions_rejects_invalid_row(client):
    payload = [{"asset_type": "not-a-real-type", "ticker": "AAPL"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_save_positions_backs_up_previous_file(client, tmp_path):
    positions_path = tmp_path / "positions.json"
    positions_path.write_text('[{"asset_type": "shares", "ticker": "OLD", "entry_price": 1, "contracts": 1, "entry_date": "2025-01-01"}]')

    payload = [{"asset_type": "shares", "ticker": "NEW", "entry_price": 2, "contracts": 2, "entry_date": "2025-01-02"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200

    backup_path = tmp_path / "positions.json.bak"
    assert backup_path.exists()
    assert "OLD" in backup_path.read_text()


def test_parse_screenshot_requires_image(client):
    res = client.post("/api/parse-screenshot", data={"provider": "anthropic"})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_parse_screenshot_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(web_ui.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    data = {"image": (io.BytesIO(b"fake-image-bytes"), "screenshot.png"), "provider": "anthropic"}
    res = client.post("/api/parse-screenshot", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "No saved API key" in res.get_json()["error"]


def test_parse_screenshot_returns_extracted_positions(client, monkeypatch):
    monkeypatch.setattr(web_ui.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(
        web_ui.vision,
        "extract_positions_from_image",
        lambda image_bytes, media_type, provider, model, api_key=None: [{"ticker": "AAPL", "asset_type": "shares"}],
    )
    data = {"image": (io.BytesIO(b"fake-image-bytes"), "screenshot.png"), "provider": "anthropic"}
    res = client.post("/api/parse-screenshot", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["positions"] == [{"ticker": "AAPL", "asset_type": "shares"}]


def test_parse_screenshot_handles_extraction_failure(client, monkeypatch):
    monkeypatch.setattr(web_ui.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")

    def boom(image_bytes, media_type, provider, model, api_key=None):
        raise RuntimeError("bad image")

    monkeypatch.setattr(web_ui.vision, "extract_positions_from_image", boom)
    data = {"image": (io.BytesIO(b"fake-image-bytes"), "screenshot.png"), "provider": "anthropic"}
    res = client.post("/api/parse-screenshot", data=data, content_type="multipart/form-data")
    assert res.status_code == 502
    assert "error" in res.get_json()


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Positions Editor" in res.data
