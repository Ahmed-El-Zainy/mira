"""utils/llm_client.py – returns the right AsyncOpenAI-compatible client.

Ollama and HuggingFace expose OpenAI-compatible REST APIs, so we simply point
the openai SDK's base_url there. No extra library needed.

Supported providers:
  - hf      : HuggingFace Inference API (via router)
  - openai  : OpenAI API (GPT-4o, GPT-4, etc.)
  - ollama  : Local Ollama server (qwen3.5, gemma4, mistral, etc.)
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
    elif s.llm_provider == "hf":
        token = s.huggingface_token or s.hf_token
        return AsyncOpenAI(
            base_url=s.hf_base_url,
            api_key=token,
        )
    return AsyncOpenAI(api_key=s.openai_api_key)


def get_model_name() -> str:
    """Return the model name string for the configured provider."""
    s = get_settings()
    if s.llm_provider == "ollama":
        return s.ollama_model
    elif s.llm_provider == "hf":
        return s.llm_hf_model_id
    return s.llm_model


def supports_json_mode() -> bool:
    """
    Returns True only for OpenAI models that support response_format=json_object.
    Ollama and HF models vary – we disable JSON mode and rely on prompt instructions instead.
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
    Ensures the configured LLM model is ready.
    If using Ollama and the model is missing, it attempts to pull it.
    If using HF, it performs a simple handshake.
    """
    s = get_settings()
    if s.llm_provider == "openai":
        return True

    if s.llm_provider == "hf":
        client = get_llm_client()
        try:
            # Check if we can reach the router and if the model is available
            # HF router usually allows listing models
            await client.models.list()
            logger.info(f"HuggingFace provider is ready with model {s.llm_hf_model_id}")
            return True
        except Exception as exc:
            logger.error(f"HuggingFace handshake failed: {str(exc)}")
            return False

    # Ollama logic
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
