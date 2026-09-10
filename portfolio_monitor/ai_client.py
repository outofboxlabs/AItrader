"""Shared low-level client for calling whichever AI provider (Anthropic,
OpenAI, or Gemini) is configured. Both news.py (headline analysis) and
movers.py (rebound analysis) call through here so the three provider
integrations live in exactly one place, each returning raw text -- the
caller does its own domain-specific JSON parsing and fallback shaping.
"""

from __future__ import annotations

from typing import Optional

_PROVIDERS = ("anthropic", "openai", "gemini")


def call_provider(
    provider: str,
    system_prompt: str,
    user_message: str,
    model: str,
    api_key: Optional[str] = None,
    max_tokens: int = 800,
) -> str:
    """Call `provider` with a system+user message pair. Raises on failure
    (missing key, network, rate limit) -- the caller decides how to
    degrade rather than crashing the whole run over one item."""
    if provider == "anthropic":
        return _call_claude(system_prompt, user_message, model, api_key, max_tokens)
    if provider == "openai":
        return _call_openai(system_prompt, user_message, model, api_key, max_tokens)
    if provider == "gemini":
        return _call_gemini(system_prompt, user_message, model, api_key)
    raise ValueError(f"Unknown provider: {provider!r} (expected one of {_PROVIDERS})")


def _call_claude(system_prompt: str, user_message: str, model: str, api_key: Optional[str], max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def _call_openai(system_prompt: str, user_message: str, model: str, api_key: Optional[str], max_tokens: int) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_gemini(system_prompt: str, user_message: str, model: str, api_key: Optional[str]) -> str:
    """UNVERIFIED against a live Gemini doc source in this session -- the
    SDK surface used here (genai.Client, models.generate_content) was
    checked against the actually-installed google-genai package instead
    (see news.py's analyze_headlines_with_gemini for details)."""
    from google import genai

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    response = client.models.generate_content(model=model, contents=f"{system_prompt}\n\n{user_message}")
    return response.text or ""
