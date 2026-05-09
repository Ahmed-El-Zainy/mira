"""utils/llm_client.py – returns the right AsyncOpenAI-compatible client.

Ollama exposes an OpenAI-compatible REST API at /v1, so we simply point
the openai SDK's base_url there. No extra library needed.

Supported providers:
  - openai  : OpenAI API (GPT-4o, GPT-4, etc.)
  - ollama  : Local Ollama server (qwen3.5, gemma4, mistral, etc.)

Recommended Ollama models (from best to lightest for this use case):
  - qwen3.5:latest   (6.6 GB) – best reasoning + JSON, handles <think> tags
  - lfm2.5-thinking:1.2b (1.2 GB) – extremely light, focuses on reasoning
  - qwen3.5:4b       (3.4 GB) – lighter, still solid JSON output
  - gemma4:e2b       (7.2 GB) – good alternative
  AVOID: ministral-3:3b, granite4:3b (too weak for planning)
  AVOID: qwen2.5-coder:* (code-focused, weak at financial reasoning)
"""

from __future__ import annotations

import json
import logging
import re

import httpx
from openai import AsyncOpenAI

from utils.config import get_settings

logger = logging.getLogger("mira")


def get_llm_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI-compatible client for the configured provider."""
    s = get_settings()
    if s.llm_provider == "ollama":
        return AsyncOpenAI(
            base_url=f"{s.ollama_base_url}/v1",
            api_key="ollama",  # required by SDK but ignored by Ollama
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


async def ensure_model_ready() -> bool:
    """
    Ensures the configured LLM model is pulled and ready.
    If using Ollama and the model is missing, it attempts to pull it.
    """
    s = get_settings()
    if s.llm_provider != "ollama":
        return True

    model = s.ollama_model
    base_url = s.ollama_base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # 1. Check if model exists
            resp = await client.post(f"{base_url}/api/show", json={"name": model})
            if resp.status_code == 200:
                logger.info(f"Ollama model {model} is ready.")
                return True

            # 2. If not, trigger a pull
            logger.info(
                f"Ollama model {model} not found. Pulling now (this may take a few minutes)..."
            )
            async with client.stream(
                "POST", f"{base_url}/api/pull", json={"name": model}, timeout=None
            ) as stream:
                async for line in stream.aiter_lines():
                    if not line:
                        continue
                    body = json.loads(line)
                    status = body.get("status")
                    if status == "success":
                        logger.info(f"Ollama model {model} pulled successfully.")
                        return True
            return False
        except Exception as exc:
            logger.error(f"Failed to ensure Ollama model readiness: {str(exc)}")
            return False


if __name__ == "__main__":
    import asyncio

    async def main():
        client = get_llm_client()
        model = get_model_name()
        print(f"LLM client: {client}, model: {model}")
        print(f"Supports JSON mode: {supports_json_mode()}")
        print("Ensuring model ready...")
        await ensure_model_ready()

    asyncio.run(main())
