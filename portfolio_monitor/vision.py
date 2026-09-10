"""Extract draft position entries from a screenshot of a brokerage app,
using whichever AI provider is configured. This never writes to
positions.json itself -- it only returns candidate rows for a human to
review, edit, and save (see web_ui.py), since misreading a number here
touches real position data.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Optional

VISION_SYSTEM_PROMPT = """You are extracting stock and options positions \
from a screenshot of a brokerage or trading app. Output ONLY a JSON array \
(no other text) where each element follows exactly this shape:

{
  "asset_type": "option" or "shares",
  "ticker": "AAPL",
  "option_type": "call" or "put" (ONLY include this field if asset_type is "option"),
  "strike": a number (ONLY if asset_type is "option"),
  "expiry": "YYYY-MM-DD" (ONLY if asset_type is "option"),
  "entry_price": a number -- the per-share average cost / cost basis shown, not the current price,
  "contracts": a number -- option contract count, or share count for "shares",
  "entry_date": "YYYY-MM-DD" if a purchase/open date is visible, otherwise null,
  "target_price": null,
  "stop_price": null
}

Rules:
- Only include a field if you can actually read it in the image. If a
  value is unclear, blurry, or not shown, use null for that field rather
  than guessing -- these numbers represent real money.
- entry_price is PER SHARE (the average cost / cost basis), never the
  total position value.
- One JSON object per distinct position (each option contract line, each
  stock holding line).
- Do not include any explanation, markdown formatting, or text outside
  the JSON array itself.
"""

EXTRACTION_INSTRUCTION = (
    "Extract every position visible in this screenshot as a JSON array "
    "following exactly the schema and rules in your instructions."
)

_PROVIDERS = ("anthropic", "openai", "gemini")


def extract_positions_from_image(
    image_bytes: bytes, media_type: str, provider: str, model: str, api_key: Optional[str] = None
) -> list[dict]:
    if provider == "anthropic":
        return _extract_with_claude(image_bytes, media_type, model, api_key)
    if provider == "openai":
        return _extract_with_openai(image_bytes, media_type, model, api_key)
    if provider == "gemini":
        return _extract_with_gemini(image_bytes, media_type, model, api_key)
    raise ValueError(f"Unknown provider: {provider!r} (expected one of {_PROVIDERS})")


def _extract_with_claude(image_bytes: bytes, media_type: str, model: str, api_key: Optional[str]) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=VISION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": EXTRACTION_INSTRUCTION},
                ],
            }
        ],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return _parse_json_array(text)


def _extract_with_openai(image_bytes: bytes, media_type: str, model: str, api_key: Optional[str]) -> list[dict]:
    import openai

    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=2000,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                ],
            },
        ],
    )
    text = response.choices[0].message.content or ""
    return _parse_json_array(text)


def _extract_with_gemini(image_bytes: bytes, media_type: str, model: str, api_key: Optional[str]) -> list[dict]:
    """UNVERIFIED: no live Gemini reference was available in this session
    to confirm the `google-genai` SDK surface -- if this errors, check
    Google's current docs for the right way to pass inline image bytes."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[
            VISION_SYSTEM_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=media_type),
            EXTRACTION_INSTRUCTION,
        ],
    )
    text = response.text or ""
    return _parse_json_array(text)


def _parse_json_array(text: str) -> list[dict]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("positions"), list):
            return parsed["positions"]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []
