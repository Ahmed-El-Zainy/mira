"""agent/executor.py – dispatches tool calls and records timing/logs.

Optimisations vs. the original implementation:
  * Independent tools (`market_data`, `news_sentiment`, `hf_sentiment`) run
    concurrently via `asyncio.gather`. `peer_correlation` is scheduled in a
    second wave because it depends on the sector returned by `market_data`.
  * Each tool keeps its own try/except so one failure cannot abort the rest.
  * Uses `asyncio.to_thread` instead of the deprecated `get_event_loop().run_in_executor`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from storage.redis_client import RedisClient
from tools.market_data import MarketDataTool
from tools.news_sentiment import NewsSentimentTool
from tools.peer_correlation import PeerCorrelationTool
from tools.hf_sentiment import HuggingFaceSentimentTool
from utils.config import get_settings
from utils.agent_logger import AgentLogger

logger = logging.getLogger("mira.executor")
_settings = get_settings()

# Tools that can always run concurrently with no input dependency on other tools.
_INDEPENDENT_TOOLS = {"market_data", "news_sentiment", "hf_sentiment"}
_DEPENDENT_TOOLS = {"peer_correlation"}


class Executor:
    def __init__(self) -> None:
        self._market = MarketDataTool()
        self._news = NewsSentimentTool(_settings.news_api_key)
        self._peers = PeerCorrelationTool()
        self._hf = HuggingFaceSentimentTool(_settings.huggingface_token, _settings.hf_model_id)
        self._logger = AgentLogger()
        self._redis = RedisClient()

    async def execute_plan(
        self,
        plan: dict,
        job_id: str,
        existing_results: dict | None = None,
    ) -> dict:
        """Run all steps in *plan*, parallelising where dependencies allow."""
        ticker = plan.get("ticker", "").upper()
        company = plan.get("company", ticker)
        steps = plan.get("steps", [])
        results: dict = existing_results or {}

        status = self._redis.get_job_status(job_id) or {}
        call_count: int = int(status.get("tool_calls_used", 0) or 0)
        budget: int = _settings.max_tool_calls_per_job

        # ── split steps into waves so independent tools fan out ──────────────
        independent: list[dict] = []
        dependent: list[dict] = []
        unknown: list[dict] = []
        for step in steps:
            tool_name = step.get("tool", "")
            if tool_name in _INDEPENDENT_TOOLS:
                independent.append(step)
            elif tool_name in _DEPENDENT_TOOLS:
                dependent.append(step)
            else:
                unknown.append(step)

        # ── wave 1: independent tools fan out ────────────────────────────────
        wave1, call_count = self._take_within_budget(independent, call_count, budget)
        if wave1:
            await self._run_wave(wave1, ticker, company, results, job_id)

        # ── wave 2: dependent tools (need market_data sector etc.) ───────────
        wave2, call_count = self._take_within_budget(dependent, call_count, budget)
        if wave2:
            await self._run_wave(wave2, ticker, company, results, job_id)

        # ── wave 3: unknown / future tools — run sequentially with a warning ─
        for step in unknown:
            if call_count >= budget:
                break
            await self._run_one(step, ticker, company, results, job_id)
            call_count += 1

        self._redis.update_job_status(job_id, {"tool_calls_used": call_count})
        if call_count >= budget and (independent or dependent or unknown):
            self._redis.update_job_status(job_id, {"budget_exceeded": True})

        return results

    # ── internal helpers ─────────────────────────────────────────────────────
    def _take_within_budget(
        self, steps: list[dict], used: int, budget: int
    ) -> tuple[list[dict], int]:
        """Return (steps_to_run, new_used_count) limited by remaining budget."""
        remaining = max(0, budget - used)
        take = steps[:remaining]
        return take, used + len(take)

    async def _run_wave(
        self,
        steps: list[dict],
        ticker: str,
        company: str,
        results: dict,
        job_id: str,
    ) -> None:
        """Run all steps in a wave concurrently."""
        coros = [self._run_one(step, ticker, company, results, job_id) for step in steps]
        await asyncio.gather(*coros, return_exceptions=False)

    async def _run_one(
        self,
        step: dict,
        ticker: str,
        company: str,
        results: dict,
        job_id: str,
    ) -> None:
        tool_name: str = step.get("tool", "")
        runner: Callable[[], Awaitable[Any]] | None = self._dispatch(tool_name, ticker, company, results)

        t0 = time.perf_counter()
        success = True
        output: Any = None
        timeout = max(5, _settings.tool_timeout_seconds)
        try:
            if runner is None:
                output = {"warning": f"Unknown tool '{tool_name}' – skipped."}
            else:
                # Hard per-tool timeout so a single hung dependency (e.g. a
                # cold FinBERT download or a paywalled HF endpoint) cannot
                # block the whole wave.
                output = await asyncio.wait_for(runner(), timeout=timeout)
            results[tool_name] = output
        except asyncio.TimeoutError:
            success = False
            output = f"timed out after {timeout}s"
            results[tool_name] = {"error": output}
            logger.warning("Tool %s timed out (%ds) for job %s.", tool_name, timeout, job_id)
        except Exception as exc:
            success = False
            output = str(exc)
            results[tool_name] = {"error": output}
            logger.warning("Tool %s failed for job %s: %s", tool_name, job_id, exc)

        latency = (time.perf_counter() - t0) * 1_000
        self._logger.log_tool_invocation(
            job_id=job_id,
            tool_name=tool_name,
            inputs={"ticker": ticker, "company": company},
            outputs=output,
            latency_ms=round(latency, 2),
            success=success,
        )

    def _dispatch(
        self, tool_name: str, ticker: str, company: str, results: dict
    ) -> Callable[[], Awaitable[Any]] | None:
        """Map tool name → bound coroutine factory."""
        if tool_name == "market_data":
            return lambda: self._market.execute(ticker)
        if tool_name == "news_sentiment":
            return lambda: self._news.execute(company, ticker)
        if tool_name == "peer_correlation":
            sector = results.get("market_data", {}).get("sector")
            return lambda: self._peers.execute(ticker, sector=sector)
        if tool_name == "hf_sentiment":
            async def _wrap() -> dict:
                score = await self._hf.get_sentiment(company)
                return {"score": score}
            return _wrap
        return None
