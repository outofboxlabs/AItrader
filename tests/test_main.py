import main as main_mod


def test_choose_model_flag_defaults_false():
    args = main_mod.parse_args(["--asof", "2025-05-01"])
    assert args.choose_model is False


def test_choose_model_flag_can_be_set():
    args = main_mod.parse_args(["--choose-model"])
    assert args.choose_model is True


def test_prompt_for_model_returns_selected_choice(monkeypatch):
    monkeypatch.setattr(main_mod.news, "list_models", lambda provider, api_key=None: ["model-a", "model-b"])
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    result = main_mod.prompt_for_model("anthropic", "sk-test")
    assert result == "model-b"


def test_prompt_for_model_empty_input_keeps_configured_default(monkeypatch):
    monkeypatch.setattr(main_mod.news, "list_models", lambda provider, api_key=None: ["model-a"])
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    result = main_mod.prompt_for_model("anthropic", "sk-test")
    assert result is None


def test_prompt_for_model_out_of_range_keeps_configured_default(monkeypatch):
    monkeypatch.setattr(main_mod.news, "list_models", lambda provider, api_key=None: ["model-a"])
    monkeypatch.setattr("builtins.input", lambda prompt: "99")
    result = main_mod.prompt_for_model("anthropic", "sk-test")
    assert result is None


def test_prompt_for_model_non_numeric_input_keeps_configured_default(monkeypatch):
    monkeypatch.setattr(main_mod.news, "list_models", lambda provider, api_key=None: ["model-a"])
    monkeypatch.setattr("builtins.input", lambda prompt: "banana")
    result = main_mod.prompt_for_model("anthropic", "sk-test")
    assert result is None


def test_prompt_for_model_falls_back_when_listing_fails(monkeypatch):
    def boom(provider, api_key=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(main_mod.news, "list_models", boom)
    result = main_mod.prompt_for_model("anthropic", "sk-test")
    assert result is None


def test_prompt_for_model_falls_back_when_list_is_empty(monkeypatch):
    monkeypatch.setattr(main_mod.news, "list_models", lambda provider, api_key=None: [])
    result = main_mod.prompt_for_model("anthropic", "sk-test")
    assert result is None


def test_interactive_flag_defaults_false():
    args = main_mod.parse_args(["--asof", "2025-05-01"])
    assert args.interactive is False


def test_news_provider_accepts_gemini():
    args = main_mod.parse_args(["--news-provider", "gemini"])
    assert args.news_provider == "gemini"


def test_prompt_for_provider_returns_selected_choice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    result = main_mod.prompt_for_provider("anthropic")
    assert result == "openai"


def test_prompt_for_provider_empty_input_keeps_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    result = main_mod.prompt_for_provider("gemini")
    assert result == "gemini"


def test_prompt_for_provider_invalid_input_keeps_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "99")
    result = main_mod.prompt_for_provider("anthropic")
    assert result == "anthropic"
