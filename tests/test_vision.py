import pytest

from portfolio_monitor import vision


def test_parse_json_array_plain():
    text = '[{"asset_type": "shares", "ticker": "AAPL"}]'
    result = vision._parse_json_array(text)
    assert result == [{"asset_type": "shares", "ticker": "AAPL"}]


def test_parse_json_array_embedded_in_text():
    text = 'Here you go:\n[{"ticker": "MSFT"}]\nLet me know if you need anything else.'
    result = vision._parse_json_array(text)
    assert result == [{"ticker": "MSFT"}]


def test_parse_json_array_wrapped_in_object_with_positions_key():
    text = '{"positions": [{"ticker": "TSLA"}]}'
    result = vision._parse_json_array(text)
    assert result == [{"ticker": "TSLA"}]


def test_parse_json_array_returns_empty_on_garbage():
    assert vision._parse_json_array("not json at all") == []


def test_extract_positions_from_image_dispatches_to_claude(monkeypatch):
    calls = []
    monkeypatch.setattr(
        vision,
        "_extract_with_claude",
        lambda image_bytes, media_type, model, api_key: calls.append(("claude", model)) or [{"ticker": "AAPL"}],
    )
    result = vision.extract_positions_from_image(b"fake", "image/png", "anthropic", "claude-opus-5")
    assert calls == [("claude", "claude-opus-5")]
    assert result == [{"ticker": "AAPL"}]


def test_extract_positions_from_image_dispatches_to_openai(monkeypatch):
    calls = []
    monkeypatch.setattr(
        vision,
        "_extract_with_openai",
        lambda image_bytes, media_type, model, api_key: calls.append(("openai", model)) or [{"ticker": "MSFT"}],
    )
    result = vision.extract_positions_from_image(b"fake", "image/png", "openai", "gpt-4o")
    assert calls == [("openai", "gpt-4o")]
    assert result == [{"ticker": "MSFT"}]


def test_extract_positions_from_image_dispatches_to_gemini(monkeypatch):
    calls = []
    monkeypatch.setattr(
        vision,
        "_extract_with_gemini",
        lambda image_bytes, media_type, model, api_key: calls.append(("gemini", model)) or [{"ticker": "TSLA"}],
    )
    result = vision.extract_positions_from_image(b"fake", "image/png", "gemini", "gemini-2.0-flash")
    assert calls == [("gemini", "gemini-2.0-flash")]
    assert result == [{"ticker": "TSLA"}]


def test_extract_positions_from_image_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        vision.extract_positions_from_image(b"fake", "image/png", "not-a-real-provider", "model")
