import config
from portfolio_monitor import credentials


def _use_tmp_credentials_file(monkeypatch, tmp_path):
    path = str(tmp_path / ".credentials.json")
    monkeypatch.setattr(config, "CREDENTIALS_PATH", path)
    return path


def test_resolve_api_key_prefers_explicit_cli_value(monkeypatch, tmp_path):
    _use_tmp_credentials_file(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    result = credentials.resolve_api_key("anthropic", cli_value="cli-key")
    assert result == "cli-key"


def test_resolve_api_key_falls_back_to_env_var(monkeypatch, tmp_path):
    _use_tmp_credentials_file(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    result = credentials.resolve_api_key("anthropic")
    assert result == "env-key"


def test_resolve_api_key_falls_back_to_saved_file(monkeypatch, tmp_path):
    _use_tmp_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials.save_key("openai", "saved-key")
    result = credentials.resolve_api_key("openai")
    assert result == "saved-key"


def test_resolve_api_key_returns_none_when_nothing_found_and_not_interactive(monkeypatch, tmp_path):
    _use_tmp_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = credentials.resolve_api_key("gemini", interactive=False)
    assert result is None


def test_resolve_api_key_interactive_prompts_and_saves_on_yes(monkeypatch, tmp_path):
    creds_path = _use_tmp_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "getpass", lambda prompt: "typed-key")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    result = credentials.resolve_api_key("gemini", interactive=True)

    assert result == "typed-key"
    assert credentials.get_saved_key("gemini") == "typed-key"
    import os

    assert os.path.exists(creds_path)


def test_resolve_api_key_interactive_does_not_save_on_no(monkeypatch, tmp_path):
    _use_tmp_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "getpass", lambda prompt: "typed-key")
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    result = credentials.resolve_api_key("gemini", interactive=True)

    assert result == "typed-key"
    assert credentials.get_saved_key("gemini") is None


def test_resolve_api_key_interactive_empty_input_returns_none(monkeypatch, tmp_path):
    _use_tmp_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "getpass", lambda prompt: "")

    result = credentials.resolve_api_key("gemini", interactive=True)
    assert result is None


def test_save_key_merges_with_existing_providers(monkeypatch, tmp_path):
    _use_tmp_credentials_file(monkeypatch, tmp_path)
    credentials.save_key("anthropic", "key-a")
    credentials.save_key("openai", "key-b")
    assert credentials.get_saved_key("anthropic") == "key-a"
    assert credentials.get_saved_key("openai") == "key-b"
