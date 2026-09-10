"""Local API-key storage so a provider only needs to be entered once.

Lookup order: an explicit value the caller already has (e.g. a CLI flag)
-> the provider's standard environment variable -> a local, gitignored
credentials file (config.CREDENTIALS_PATH) -> an interactive prompt that
offers to save the entered key to that file for next time.
"""

from __future__ import annotations

import json
import os
import stat
from getpass import getpass
from typing import Optional

import config

ENV_VAR_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

DISPLAY_NAME_BY_PROVIDER = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI (GPT)",
    "gemini": "Google (Gemini)",
}


def _load_saved() -> dict:
    if not os.path.exists(config.CREDENTIALS_PATH):
        return {}
    try:
        with open(config.CREDENTIALS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(creds: dict) -> None:
    with open(config.CREDENTIALS_PATH, "w") as f:
        json.dump(creds, f, indent=2)
    try:
        os.chmod(config.CREDENTIALS_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600: owner read/write only
    except OSError:
        pass  # best-effort; not every platform/filesystem supports chmod


def get_saved_key(provider: str) -> Optional[str]:
    return _load_saved().get(provider)


def save_key(provider: str, api_key: str) -> None:
    creds = _load_saved()
    creds[provider] = api_key
    _save(creds)


def resolve_api_key(
    provider: str,
    cli_value: Optional[str] = None,
    interactive: bool = False,
) -> Optional[str]:
    """Resolve an API key for `provider`. Returns None if nothing is found
    and `interactive` is False (the caller decides how to degrade) rather
    than raising -- a missing key is an expected, recoverable state here."""
    if cli_value:
        return cli_value

    env_var = ENV_VAR_BY_PROVIDER.get(provider)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]

    saved = get_saved_key(provider)
    if saved:
        return saved

    if not interactive:
        return None

    display_name = DISPLAY_NAME_BY_PROVIDER.get(provider, provider)
    entered = getpass(f"No saved API key found for {display_name}. Enter it now (input hidden): ").strip()
    if not entered:
        return None

    choice = input("Save this key locally for future runs? [Y/n]: ").strip().lower()
    if choice in ("", "y", "yes"):
        save_key(provider, entered)
        print(f"Saved to {config.CREDENTIALS_PATH} (gitignored -- never commit this file).")

    return entered
