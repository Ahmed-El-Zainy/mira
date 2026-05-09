"""utils/llm_client.py – returns the right AsyncOpenAI-compatible client.

Ollama exposes an OpenAI-compatible REST API at /v1, so we simply point
the openai SDK's base_url there. No extra library needed.

Supported providers:
  - openai  : OpenAI API (GPT-4o, GPT-4, etc.)
  - ollama  : Local Ollama server (qwen3.5, gemma4, mistral, etc.)

Recommended Ollama models (from best to lightest for this use case):
  - qwen3.5:latest   (6.6 GB) – best reasoning + JSON, handles <think> tags
  - qwen3.5:4b       (3.4 GB) – lighter, still solid JSON output
  - gemma4:e2b       (7.2 GB) – good alternative
  AVOID: ministral-3:3b, lfm2.5-thinking:1.2b, granite4:3b (too weak for planning)
  AVOID: qwen2.5-coder:* (code-focused, weak at financial reasoning)
"""
from __future__ import annotations

import re

from openai import AsyncOpenAI

from utils.config import get_settings


def get_llm_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI-compatible client for the configured provider."""
    s = get_settings()
    if s.llm_provider == "ollama":
        return AsyncOpenAI(
            base_url=f"{s.ollama_base_url}/v1",
            api_key="ollama",   # required by SDK but ignored by Ollama
        )
    return AsyncOpenAI(api_key=s.openai_api_key)


def get_model_name() -> str:
    """Return the model name string for the configured provider."""
    s = get_settings()
    return s.ollama_model if s.llm_provider == "ollama" else s.llm_model


def supports_json_mode() -> bool:
    """
    Returns True only for OpenAI models that support response_format=json_object.
    Ollama models vary – we disable JSON mode and rely on prompt instructions instead.
    """
    return get_settings().llm_provider == "openai"


def clean_llm_response(raw: str) -> str:
    """
    Normalise raw LLM output to a plain JSON string.

    Handles:
      1. <think>…</think> blocks – qwen3 emits chain-of-thought before JSON
      2. Markdown fences – ```json … ``` or ``` … ```
      3. Leading/trailing whitespace
    """
    # Strip <think>...</think> reasoning traces (qwen3.x models)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Strip markdown fences
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    return raw
