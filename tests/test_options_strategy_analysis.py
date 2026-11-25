import pytest

from portfolio_monitor import options_strategy_analysis as osa


def test_strategy_labels_has_all_13_in_order():
    assert len(osa.STRATEGY_LABELS) == 13
    assert list(osa.STRATEGY_LABELS) == [
        "long_call",
        "covered_call",
        "long_put",
        "cash_secured_put",
        "call_debit_spread",
        "call_credit_spread",
        "put_debit_spread",
        "put_credit_spread",
        "long_straddle",
        "long_strangle",
        "long_call_calendar",
        "long_put_calendar",
        "short_put_calendar",
    ]


def test_every_strategy_has_a_system_prompt():
    assert set(osa.STRATEGY_SYSTEM_PROMPTS) == set(osa.STRATEGY_LABELS)
    for prompt in osa.STRATEGY_SYSTEM_PROMPTS.values():
        assert "favorable|neutral|unfavorable" in prompt


def test_parse_strategy_json_parses_clean_response():
    result = osa._parse_strategy_json(
        '{"verdict": "favorable", "rationale": "IV is cheap.", "suggested_structure": "30 DTE, ATM"}'
    )
    assert result == {"verdict": "favorable", "rationale": "IV is cheap.", "suggested_structure": "30 DTE, ATM"}


def test_parse_strategy_json_extracts_embedded_object():
    result = osa._parse_strategy_json('Sure, here is my take:\n{"verdict": "unfavorable", "rationale": "Too rich."}')
    assert result["verdict"] == "unfavorable"
    assert result["rationale"] == "Too rich."
    assert result["suggested_structure"] is None  # defaulted


def test_parse_strategy_json_degrades_to_raw_text_rationale():
    result = osa._parse_strategy_json("not valid json at all")
    assert result["rationale"] == "not valid json at all"
    assert result["verdict"] == "neutral"  # defaulted


def _context():
    return {
        "current_price": 100.0,
        "near_expiry_dte": 30,
        "far_expiry_dte": 75,
        "atm_iv_pct": 40.0,
        "skew_pct": 5.0,
        "term_structure_pct": -8.0,
        "hv_20_pct": 25.0,
        "hv_60_pct": 28.0,
        "days_to_earnings": 10,
        "technical": {"price_vs_sma_20_pct": 3.2, "rsi_14": 62.0, "macd_histogram": 0.45},
    }


def test_context_lines_includes_all_real_numbers_given():
    lines = osa._context_lines("ACME", _context())
    text = " ".join(lines)
    assert "ACME" in text
    assert "100.0" in text
    assert "40.0%" in text
    assert "-8.0" in text or "term structure" in text.lower()
    assert "inverted" in text.lower()  # negative term structure
    assert "10 day(s)" in text
    assert "RSI(14): 62.0" in text


def test_context_lines_handles_missing_data_gracefully():
    lines = osa._context_lines("ACME", {"current_price": 50.0})
    text = " ".join(lines)
    assert "not available" in text.lower()


def test_run_strategy_panel_runs_all_13_by_default(monkeypatch):
    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        return '{"verdict": "neutral", "rationale": "ok", "suggested_structure": "n/a"}'

    monkeypatch.setattr(osa.ai_client, "call_provider", fake_call_provider)

    results = osa.run_strategy_panel("ACME", _context(), "anthropic", "claude-haiku-4-5", api_key="sk-test")
    assert len(results) == 13
    assert [r["strategy"] for r in results] == list(osa.STRATEGY_LABELS)  # canonical order preserved
    assert all(r["verdict"] == "neutral" and r["error"] is None for r in results)
    assert all(r["category"] in {"Single-Leg Strategies", "Vertical Spreads", "Straddles and Strangles", "Calendar Spreads"} for r in results)


def test_run_strategy_panel_isolates_one_strategy_failure(monkeypatch):
    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        if "LONG STRADDLE" in system_prompt:
            raise RuntimeError("rate limited")
        return '{"verdict": "favorable", "rationale": "ok"}'

    monkeypatch.setattr(osa.ai_client, "call_provider", fake_call_provider)

    results = osa.run_strategy_panel("ACME", _context(), "anthropic", "claude-haiku-4-5", api_key="sk-test")
    by_key = {r["strategy"]: r for r in results}
    assert by_key["long_straddle"]["error"] == "rate limited"
    assert by_key["long_straddle"]["verdict"] is None
    assert by_key["long_call"]["verdict"] == "favorable"  # the rest succeeded


def test_run_strategy_panel_respects_explicit_subset(monkeypatch):
    calls = []

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        calls.append(system_prompt)
        return '{"verdict": "favorable", "rationale": "ok"}'

    monkeypatch.setattr(osa.ai_client, "call_provider", fake_call_provider)

    results = osa.run_strategy_panel(
        "ACME", _context(), "anthropic", "claude-haiku-4-5", api_key="sk-test", strategies=["long_call", "covered_call"]
    )
    assert len(results) == 2
    assert len(calls) == 2
    assert {r["strategy"] for r in results} == {"long_call", "covered_call"}


def test_run_strategy_panel_shares_one_context_across_all_strategies(monkeypatch):
    """Unlike run_expert_panel's per-persona narrow slices, every
    strategy here sees the identical user_message -- only the system
    prompt differs."""
    user_messages = []

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        user_messages.append(user_message)
        return '{"verdict": "neutral", "rationale": "ok"}'

    monkeypatch.setattr(osa.ai_client, "call_provider", fake_call_provider)

    osa.run_strategy_panel("ACME", _context(), "anthropic", "claude-haiku-4-5", api_key="sk-test")
    assert len(set(user_messages)) == 1
