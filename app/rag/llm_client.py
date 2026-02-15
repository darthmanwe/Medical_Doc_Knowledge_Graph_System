"""Anthropic Claude LLM wrapper with retry / back-off.

Provides two high-level methods:
  • generate()           — free-form text generation
  • extract_structured() — tool_use API for structured JSON extraction
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def verify_api_key() -> bool:
    """Lightweight check that the API key is valid."""
    try:
        client = _get_client()
        client.messages.create(
            model=settings.llm_model,
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True
    except Exception as exc:
        logger.error("Anthropic API key verification failed: %s", exc)
        return False


# ── Core generation with retry ─────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIConnectionError)
    ),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def generate(
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Send a chat completion request and return the assistant text."""
    client = _get_client()
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens or settings.llm_max_tokens,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


# ── Structured extraction via tool_use ─────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIConnectionError)
    ),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def extract_structured(
    system: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_choice: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Call Claude with tool_use and return the parsed tool input dict.

    *tools* follows the Anthropic tool schema:
        [{"name": "extract_entities", "description": "...", "input_schema": {...}}]
    """
    client = _get_client()

    create_kwargs: dict[str, Any] = dict(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=0.0,
        system=system,
        messages=messages,
        tools=tools,
    )
    if tool_choice:
        create_kwargs["tool_choice"] = tool_choice

    resp = client.messages.create(**create_kwargs)

    for block in resp.content:
        if block.type == "tool_use":
            return block.input  # already a dict

    logger.warning("No tool_use block in response; falling back to text parse.")
    text = resp.content[0].text if resp.content else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}
