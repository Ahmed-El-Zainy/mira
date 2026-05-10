"""utils/llm_client.py – AsyncOpenAI-compatible client with automatic fallback.

The exposed `FallbackLLMClient` is a drop-in replacement for `AsyncOpenAI`
that, on every `chat.completions.create` call:

  1. Tries the primary provider configured by `LLM_PROVIDER`.
  2. If the call raises a recognised "primary is unhealthy" error (network
     blip, HTTP 4xx/5xx, payment required, quota exceeded, rate limit,
     timeout, …) it is automatically retried against the provider configured
     by `LLM_FALLBACK_PROVIDER` — defaults to `ollama`, which is local and
     free.
  3. A small circuit breaker remembers consecutive primary failures and skips
     the primary entirely for a short cool-down so we don't waste seconds on
     a known-down provider.

Set `LLM_FALLBACK_PROVIDER=none` (or leave empty) to disable the fallback.

Supported providers: `hf`, `openai`, `ollama` — all reached via the same
OpenAI-compatible `chat.completions` endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx
from openai import AsyncOpenAI

from utils.config import get_settings

logger = logging.getLogger("mira")


# Tuning knobs for the circuit breaker (seconds + count).
_PRIMARY_COOLDOWN_SECONDS = 300        # 5 min
_PRIMARY_FAILURE_THRESHOLD = 3         # open the circuit after N failures in a row


# ╭──────────────────────────────────────────────────────────────────────────╮
# │  per-provider client + model lookup                                       │
# ╰──────────────────────────────────────────────────────────────────────────╯
def _build_raw_client(provider: str) -> AsyncOpenAI | None:
    """Construct a raw AsyncOpenAI client for *provider*. None on misconfig."""
    s = get_settings()
    p = (provider or "").lower().strip()

    if p == "ollama":
        return AsyncOpenAI(
            base_url=f"{s.ollama_base_url}/v1",
            api_key="ollama",  # ignored by Ollama but required by SDK
        )
    if p == "hf":
        token = s.huggingface_token or s.hf_token
        if not token:
            logger.warning("HF provider requested but no HUGGINGFACE_TOKEN set.")
            return None
        return AsyncOpenAI(base_url=s.hf_base_url, api_key=token)
    if p == "openai":
        if not s.openai_api_key:
            logger.warning("OpenAI provider requested but no OPENAI_API_KEY set.")
            return None
        return AsyncOpenAI(api_key=s.openai_api_key)
    return None


def _model_for_provider(provider: str) -> str | None:
    s = get_settings()
    p = (provider or "").lower().strip()
    if p == "ollama":
        return s.ollama_model
    if p == "hf":
        return s.llm_hf_model_id
    if p == "openai":
        return s.llm_model
    return None


# ╭──────────────────────────────────────────────────────────────────────────╮
# │  error classification                                                     │
# ╰──────────────────────────────────────────────────────────────────────────╯
def _classify_primary_error(exc: BaseException) -> tuple[bool, str]:
    """Return (should_failover, human_label).

    We *always* failover on transient/HTTP/network errors. We do NOT failover
    on `asyncio.CancelledError` / `KeyboardInterrupt` / `SystemExit` so that
    user-initiated cancellation is honoured.
    """
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        return False, type(exc).__name__

    # Try to extract an HTTP status code if this is an OpenAI SDK error.
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )

    name = type(exc).__name__
    label = name
    if status == 401:
        label = "401 Unauthorized — bad/expired token"
    elif status == 402:
        label = "402 Payment Required — credits/quota exhausted"
    elif status == 403:
        label = "403 Forbidden — model access denied or quota exceeded"
    elif status == 404:
        label = "404 Not Found — model does not exist on this provider"
    elif status == 429:
        label = "429 Rate Limited"
    elif status and 500 <= status < 600:
        label = f"{status} Server Error"
    elif "Timeout" in name or "Connect" in name:
        label = f"{name} (network/timeout)"
    elif "RateLimit" in name:
        label = "Rate limit"
    elif "Permission" in name or "Authentication" in name:
        label = f"{name} (auth/quota)"

    return True, label


# ╭──────────────────────────────────────────────────────────────────────────╮
# │  fallback wrapper                                                         │
# ╰──────────────────────────────────────────────────────────────────────────╯
class _FallbackCompletions:
    """Drop-in replacement for `client.chat.completions` with provider fallback."""

    def __init__(
        self,
        primary: AsyncOpenAI,
        primary_model: str,
        primary_provider: str,
        fallback: AsyncOpenAI | None,
        fallback_model: str | None,
        fallback_provider: str | None,
    ) -> None:
        self._primary = primary
        self._primary_model = primary_model
        self._primary_provider = primary_provider
        self._fallback = fallback
        self._fallback_model = fallback_model
        self._fallback_provider = fallback_provider

        # Circuit-breaker state for the primary.
        self._primary_consecutive_failures = 0
        self._primary_cooldown_until = 0.0

    # ── public ----------------------------------------------------------
    async def create(self, **kwargs: Any):
        """Call primary; on failure transparently retry against fallback."""
        if self._primary_skipped_now():
            # Circuit is open — skip primary entirely until cooldown expires.
            if self._fallback is not None and self._fallback_model is not None:
                logger.info(
                    "Primary (%s) in cool-down for %.0fs more — using fallback (%s) directly.",
                    self._primary_provider,
                    self._primary_cooldown_until - time.time(),
                    self._fallback_provider,
                )
                return await self._call_fallback(kwargs)

        try:
            resp = await self._primary.chat.completions.create(**kwargs)
        except BaseException as exc:
            should_failover, label = _classify_primary_error(exc)
            if not should_failover:
                raise

            self._record_primary_failure()

            if self._fallback is None or self._fallback_model is None:
                logger.error(
                    "Primary LLM (%s/%s) failed [%s] and no fallback configured.",
                    self._primary_provider, self._primary_model, label,
                )
                raise

            logger.warning(
                "Primary LLM (%s/%s) failed [%s] — failing over to %s/%s.",
                self._primary_provider, self._primary_model, label,
                self._fallback_provider, self._fallback_model,
            )
            return await self._call_fallback(kwargs)

        # Success on primary → reset the circuit breaker.
        self._reset_primary_state()
        # Tag the response so the agent logger can attribute tokens correctly.
        try:
            resp._mira_provider = self._primary_provider  # type: ignore[attr-defined]
            resp._mira_model = self._primary_model  # type: ignore[attr-defined]
        except Exception:
            pass
        return resp

    # ── helpers ---------------------------------------------------------
    async def _call_fallback(self, kwargs: dict) -> Any:
        """Invoke the fallback provider with the right model name."""
        fb_kwargs = dict(kwargs)
        fb_kwargs["model"] = self._fallback_model
        # JSON mode handling: only OpenAI reliably honours `response_format`.
        if self._fallback_provider != "openai":
            fb_kwargs.pop("response_format", None)
        try:
            resp = await self._fallback.chat.completions.create(**fb_kwargs)  # type: ignore[union-attr]
        except BaseException as fb_exc:
            logger.error(
                "Fallback LLM (%s/%s) ALSO failed: %s — re-raising.",
                self._fallback_provider, self._fallback_model, fb_exc,
            )
            raise
        try:
            resp._mira_provider = self._fallback_provider  # type: ignore[attr-defined]
            resp._mira_model = self._fallback_model  # type: ignore[attr-defined]
            resp._mira_failover = True  # type: ignore[attr-defined]
        except Exception:
            pass
        logger.info(
            "Fallback (%s/%s) answered successfully.",
            self._fallback_provider, self._fallback_model,
        )
        return resp

    def _primary_skipped_now(self) -> bool:
        return time.time() < self._primary_cooldown_until

    def _record_primary_failure(self) -> None:
        self._primary_consecutive_failures += 1
        if self._primary_consecutive_failures >= _PRIMARY_FAILURE_THRESHOLD:
            self._primary_cooldown_until = time.time() + _PRIMARY_COOLDOWN_SECONDS
            logger.warning(
                "Circuit breaker OPEN for primary (%s) — pausing it for %ds "
                "after %d consecutive failures. Calls go straight to %s.",
                self._primary_provider,
                _PRIMARY_COOLDOWN_SECONDS,
                self._primary_consecutive_failures,
                self._fallback_provider,
            )

    def _reset_primary_state(self) -> None:
        if self._primary_consecutive_failures or self._primary_cooldown_until:
            logger.info("Primary (%s) is healthy again — circuit breaker CLOSED.",
                        self._primary_provider)
        self._primary_consecutive_failures = 0
        self._primary_cooldown_until = 0.0


class _FallbackChat:
    def __init__(self, completions: _FallbackCompletions) -> None:
        self.completions = completions


class FallbackLLMClient:
    """`AsyncOpenAI`-shaped facade with transparent provider failover.

    Unknown attribute access (e.g. `.models`) is proxied to the primary so
    existing call sites like `client.models.list()` keep working.
    """

    def __init__(
        self,
        primary: AsyncOpenAI,
        primary_model: str,
        primary_provider: str,
        fallback: AsyncOpenAI | None,
        fallback_model: str | None,
        fallback_provider: str | None,
    ) -> None:
        self._primary = primary
        self.primary_model = primary_model
        self.primary_provider = primary_provider
        self.fallback_model = fallback_model
        self.fallback_provider = fallback_provider
        self.chat = _FallbackChat(
            _FallbackCompletions(
                primary, primary_model, primary_provider,
                fallback, fallback_model, fallback_provider,
            )
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


# ╭──────────────────────────────────────────────────────────────────────────╮
# │  public API                                                               │
# ╰──────────────────────────────────────────────────────────────────────────╯
def get_llm_client():
    """Return a chat client. Wraps primary in `FallbackLLMClient` when a
    different fallback provider is configured."""
    s = get_settings()
    primary_provider = s.llm_provider
    primary = _build_raw_client(primary_provider)
    if primary is None:
        # Misconfigured primary — try to satisfy with the fallback alone.
        fb = (s.llm_fallback_provider or "").lower().strip()
        if fb and fb not in ("none", primary_provider.lower()):
            fb_client = _build_raw_client(fb)
            if fb_client is not None:
                logger.warning(
                    "Primary provider '%s' could not be initialised — "
                    "using fallback '%s' as the primary instead.",
                    primary_provider, fb,
                )
                return fb_client
        return AsyncOpenAI(api_key=s.openai_api_key or "missing")

    primary_model = _model_for_provider(primary_provider)

    fb_provider = (s.llm_fallback_provider or "").lower().strip()
    if not fb_provider or fb_provider in ("none", primary_provider.lower()):
        return primary

    fallback = _build_raw_client(fb_provider)
    fallback_model = _model_for_provider(fb_provider)
    if fallback is None or fallback_model is None:
        logger.info(
            "Fallback provider '%s' not usable — proceeding without fallback.",
            fb_provider,
        )
        return primary

    logger.info(
        "LLM client initialised: primary=%s/%s · fallback=%s/%s "
        "(circuit breaker: %d failures → %ds cooldown)",
        primary_provider, primary_model, fb_provider, fallback_model,
        _PRIMARY_FAILURE_THRESHOLD, _PRIMARY_COOLDOWN_SECONDS,
    )
    return FallbackLLMClient(
        primary=primary,
        primary_model=primary_model,
        primary_provider=primary_provider,
        fallback=fallback,
        fallback_model=fallback_model,
        fallback_provider=fb_provider,
    )


def get_model_name() -> str:
    """Return the model string for the *primary* provider (caller-facing)."""
    s = get_settings()
    return _model_for_provider(s.llm_provider) or "gpt-4o"


def supports_json_mode() -> bool:
    """Returns True only for OpenAI primaries that support `response_format=json_object`."""
    return get_settings().llm_provider == "openai"


def clean_llm_response(raw: str) -> str:
    """Normalise raw LLM output to a plain JSON string.

    Handles:
      1. <think>…</think> blocks – qwen3 emits chain-of-thought before JSON
      2. Markdown fences – ```json … ``` or ``` … ```
      3. Leading/trailing whitespace
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    return raw


async def ensure_model_ready() -> bool:
    """Ensure the configured PRIMARY LLM model is reachable.

    If using Ollama and the model is missing, attempt to pull it. For HF/OpenAI,
    perform a lightweight handshake. Returns False on any failure — callers can
    still proceed because the FallbackLLMClient will route around it at runtime.
    """
    s = get_settings()
    primary = s.llm_provider.lower().strip()

    if primary == "openai":
        return True

    if primary == "hf":
        client = _build_raw_client("hf")
        if client is None:
            return False
        try:
            await client.models.list()
            logger.info("HuggingFace primary ready: %s", s.llm_hf_model_id)
            return True
        except Exception as exc:
            logger.error("HuggingFace handshake failed: %s", exc)
            return False

    # Ollama: check then optionally pull
    model = s.ollama_model
    base_url = s.ollama_base_url
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{base_url}/api/show", json={"name": model})
            if resp.status_code == 200:
                logger.info("Ollama model %s is ready.", model)
                return True
            logger.info(
                "Ollama model %s not found. Pulling now (this may take a few minutes)...",
                model,
            )
            async with client.stream(
                "POST", f"{base_url}/api/pull", json={"name": model}, timeout=None
            ) as stream:
                async for line in stream.aiter_lines():
                    if not line:
                        continue
                    body = json.loads(line)
                    if body.get("status") == "success":
                        logger.info("Ollama model %s pulled successfully.", model)
                        return True
            return False
        except Exception as exc:
            logger.error("Failed to ensure Ollama model readiness: %s", exc)
            return False


async def ensure_fallback_ready() -> bool:
    """Best-effort warmup of the fallback provider so the first failover is fast."""
    s = get_settings()
    fb = (s.llm_fallback_provider or "").lower().strip()
    if not fb or fb in ("none", s.llm_provider.lower()):
        return True
    if fb != "ollama":
        return True

    model = s.ollama_model
    base_url = s.ollama_base_url
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{base_url}/api/show", json={"name": model})
            if resp.status_code == 200:
                logger.info("Fallback (Ollama) ready: %s", model)
                return True
            logger.warning(
                "Fallback (Ollama) model '%s' not present locally — "
                "pull it with: `ollama pull %s`",
                model, model,
            )
            return False
    except Exception as exc:
        logger.warning("Could not reach Ollama at %s: %s", base_url, exc)
        return False


if __name__ == "__main__":
    async def _main():
        client = get_llm_client()
        model = get_model_name()
        print(f"LLM client: {client}, model: {model}")
        print(f"Supports JSON mode: {supports_json_mode()}")
        await ensure_model_ready()
        await ensure_fallback_ready()

    asyncio.run(_main())
