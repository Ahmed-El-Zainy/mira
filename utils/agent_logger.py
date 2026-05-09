"""utils/agent_logger.py – structured logging + token/cost tracking."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.redis_client import RedisClient
from utils.config import get_settings

logger = logging.getLogger("mira")


class AgentLogger:
    # Cost per 1K tokens (USD) – OpenAI models only
    _COST_TABLE: dict[str, dict[str, float]] = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
    }

    def __init__(self) -> None:
        self._redis = RedisClient()

    # ── tool invocation ───────────────────────────────────────────────────────
    def log_tool_invocation(
        self,
        job_id: str,
        tool_name: str,
        inputs: dict,
        outputs: Any,
        latency_ms: float,
        success: bool,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "event_type": "tool_invocation",
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs if success else None,
            "error": outputs if not success else None,
            "latency_ms": latency_ms,
            "success": success,
        }
        self._redis.add_job_log(job_id, entry)
        logger.info(json.dumps(entry))

    # ── token usage ───────────────────────────────────────────────────────────
    def log_token_usage(
        self,
        job_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        s = get_settings()

        # Local Ollama models have zero API cost
        if s.llm_provider == "ollama":
            estimated_cost = 0.0
        else:
            table = self._COST_TABLE.get(model, self._COST_TABLE["gpt-4o"])
            estimated_cost = (prompt_tokens / 1_000) * table["prompt"] + (
                completion_tokens / 1_000
            ) * table["completion"]

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "event_type": "token_usage",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
            "provider": s.llm_provider,
            "estimated_cost_usd": round(estimated_cost, 6),
        }
        self._redis.add_job_log(job_id, entry)
        self._redis.update_job_token_usage(
            job_id, prompt_tokens, completion_tokens, estimated_cost
        )
        logger.info(json.dumps(entry))
