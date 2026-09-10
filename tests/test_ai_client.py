import pytest

from portfolio_monitor import ai_client


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


def test_call_provider_claude(monkeypatch):
    import anthropic

    class FakeMessages:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs

            class Resp:
                content = [_FakeTextBlock('{"a": 1}')]

            return Resp()

    class FakeClient:
        def __init__(self, api_key=None):
            self.messages = FakeMessages()

    captured = {}
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)

    text = ai_client.call_provider("anthropic", "sys", "user msg", "claude-haiku-4-5", api_key="sk-test")
    assert text == '{"a": 1}'
    assert captured["kwargs"]["system"] == "sys"
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "user msg"}]


def test_call_provider_openai(monkeypatch):
    import openai

    class FakeMessage:
        content = '{"b": 2}'

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs

            class Resp:
                choices = [FakeChoice()]

            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, api_key=None):
            self.chat = FakeChat()

    captured = {}
    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    text = ai_client.call_provider("openai", "sys", "user msg", "gpt-4o-mini", api_key="sk-test")
    assert text == '{"b": 2}'
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}


def test_call_provider_gemini(monkeypatch):
    import google.genai as genai

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["kwargs"] = kwargs

            class Resp:
                text = '{"c": 3}'

            return Resp()

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    captured = {}
    monkeypatch.setattr(genai, "Client", FakeClient)

    text = ai_client.call_provider("gemini", "sys", "user msg", "gemini-2.0-flash", api_key="key")
    assert text == '{"c": 3}'
    assert "sys" in captured["kwargs"]["contents"]
    assert "user msg" in captured["kwargs"]["contents"]


def test_call_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        ai_client.call_provider("not-a-real-provider", "sys", "msg", "model")
