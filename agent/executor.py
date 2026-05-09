"""agent/executor.py – dispatches tool calls and records timing/logs."""
from __future__ import annotations

import time
from typing import Any

from storage.redis_client import RedisClient
from tools.market_data import MarketDataTool
from tools.news_sentiment import NewsSentimentTool
from tools.peer_correlation import PeerCorrelationTool
from utils.config import get_settings
from utils.agent_logger import AgentLogger

_settings = get_settings()


class Executor:
    def __init__(self) -> None:
        self._market   = MarketDataTool()
        self._news     = NewsSentimentTool(_settings.news_api_key)
        self._peers    = PeerCorrelationTool()
        self._logger   = AgentLogger()
        self._redis    = RedisClient()

    async def execute_plan(
        self,
        plan: dict,
        job_id: str,
        existing_results: dict | None = None,
    ) -> dict:
        """Run each step in *plan*, enforcing the per-job tool-call budget."""
        ticker   = plan.get("ticker", "").upper()
        company  = plan.get("company", ticker)
        steps    = plan.get("steps", [])
        results  = existing_results or {}

        # Retrieve current call count from Redis
        status = self._redis.get_job_status(job_id) or {}
        call_count: int = status.get("tool_calls_used", 0)
        budget: int     = _settings.max_tool_calls_per_job

        for step in steps:
            if call_count >= budget:
                self._redis.update_job_status(
                    job_id,
                    {"budget_exceeded": True, "tool_calls_used": call_count},
                )
                break

            tool_name: str = step.get("tool", "")
            t0 = time.perf_counter()
            success = True
            output: Any = None

            try:
                if tool_name == "market_data":
                    output = await self._market.execute(ticker)
                    results["market_data"] = output

                elif tool_name == "news_sentiment":
                    output = await self._news.execute(company, ticker)
                    results["news_sentiment"] = output

                elif tool_name == "peer_correlation":
                    sector = results.get("market_data", {}).get("sector")
                    output = await self._peers.execute(ticker, sector=sector)
                    results["peer_correlation"] = output

                else:
                    output = {"warning": f"Unknown tool '{tool_name}' – skipped."}
                    results[tool_name] = output

            except Exception as exc:
                success = False
                output  = str(exc)
                results[tool_name] = {"error": output}

            latency = (time.perf_counter() - t0) * 1_000
            call_count += 1

            self._logger.log_tool_invocation(
                job_id    = job_id,
                tool_name = tool_name,
                inputs    = {"ticker": ticker, "company": company},
                outputs   = output,
                latency_ms= round(latency, 2),
                success   = success,
            )
            self._redis.update_job_status(job_id, {"tool_calls_used": call_count})

        return results
