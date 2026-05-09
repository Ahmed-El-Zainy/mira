"""agent/core.py – main orchestrator: plan → execute → reflect → synthesise."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agent.executor import Executor
from agent.planner import Planner
from agent.reflection import ReflectionModule
from storage.redis_client import RedisClient
from utils.config import get_settings
from utils.llm_client import clean_llm_response, get_llm_client, get_model_name, supports_json_mode
from utils.logging import AgentLogger

_settings = get_settings()

_SYNTHESIS_SYSTEM = """
You are M.I.R.A., an expert AI financial analyst. Using ONLY the structured data
provided, produce a JSON investment analysis report.

Return ONLY valid JSON matching this exact schema. No markdown fences, no explanation:
{
  "company_ticker":    "STRING",
  "company_name":      "STRING",
  "analysis_summary":  "STRING – 3-5 sentence concise synthesis",
  "sentiment_score":   FLOAT (-1.0 to 1.0),
  "market_snapshot": {
    "price":               NUMBER | null,
    "daily_change_pct":    NUMBER | null,
    "market_cap":          NUMBER | null,
    "pe_ratio":            NUMBER | null,
    "52w_high":            NUMBER | null,
    "52w_low":             NUMBER | null,
    "quarterly_revenues":  [NUMBER, NUMBER]
  },
  "correlation_analysis": {
    "market_correlation":  NUMBER | null,
    "sector_etf":          "STRING",
    "sector_correlation":  NUMBER | null,
    "peer_correlations":   { "TICKER": NUMBER },
    "all_correlations":    { "TICKER": NUMBER }
  },
  "key_findings":     ["STRING", "STRING", "STRING"],
  "tools_used":       ["STRING"],
  "citation_sources": ["URL"],
  "generated_at":     "ISO8601"
}

Rules:
  • sentiment_score must equal the news_sentiment.sentiment_score value.
  • market_snapshot must be populated from market_data tool results.
  • key_findings must be exactly 3 actionable, specific insights.
  • Do NOT invent data that is missing from the provided tool results.
""".strip()


class AgentCore:
    def __init__(self) -> None:
        self._planner    = Planner()
        self._executor   = Executor()
        self._reflection = ReflectionModule()
        self._redis      = RedisClient()
        self._logger     = AgentLogger()
        self._client     = get_llm_client()
        self._model      = get_model_name()

    # ── public entry point ────────────────────────────────────────────────────
    async def run_analysis(self, job_id: str, query: str, tag: str = "") -> None:
        try:
            await self._run(job_id, query, tag)
        except Exception as exc:
            self._redis.update_job_status(
                job_id,
                {"status": "failed", "error": str(exc), "progress": 0},
            )

    # ── private pipeline ──────────────────────────────────────────────────────
    async def _run(self, job_id: str, query: str, tag: str) -> None:
        # 1. Plan
        self._redis.update_job_status(job_id, {"status": "planning", "progress": 10})
        plan = await self._planner.create_initial_plan(query)

        ticker  = plan.get("ticker", "UNKNOWN").upper()
        company = plan.get("company", ticker)

        # 2. Execute initial plan
        self._redis.update_job_status(job_id, {"status": "executing", "progress": 30})
        results = await self._executor.execute_plan(plan, job_id)

        # Check for hard errors from market_data (unknown / delisted ticker)
        mkt = results.get("market_data", {})
        if "error" in mkt:
            raise ValueError(mkt["error"])

        # 3. Reflect
        self._redis.update_job_status(job_id, {"status": "reflecting", "progress": 65})
        reflection = await self._reflection.evaluate_results(results, query)

        if reflection.get("needs_more_research"):
            self._redis.update_job_status(job_id, {"status": "re-planning", "progress": 70})
            extra_plan = await self._planner.create_reflection_plan(reflection, query)
            results    = await self._executor.execute_plan(extra_plan, job_id, results)
            self._redis.update_job_status(job_id, {"progress": 85})

        # 4. Synthesise final report
        self._redis.update_job_status(job_id, {"status": "synthesising", "progress": 90})
        report = await self._synthesise(ticker, company, results, query)

        if tag:
            report["tag"] = tag
        report["reflection"] = {
            "triggers_fired": reflection.get("triggers_fired", []),
            "reasoning":      reflection.get("reasoning", ""),
        }

        # 5. Persist & complete
        self._redis.store_job_result(job_id, report)
        self._redis.update_job_status(job_id, {"status": "completed", "progress": 100})

    # ── synthesis via LLM ─────────────────────────────────────────────────────
    async def _synthesise(
        self,
        ticker: str,
        company: str,
        results: dict,
        query: str,
    ) -> dict[str, Any]:
        context = {
            "ticker":       ticker,
            "company":      company,
            "query":        query,
            "data":         results,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user",   "content": json.dumps(context)},
            ],
        )
        if supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)

        usage = resp.usage
        if usage:
            self._logger.log_token_usage(
                job_id=ticker,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                model=self._model,
            )

        raw    = clean_llm_response(resp.choices[0].message.content)
        report = json.loads(raw)

        # Ensure citation_sources populated from news data
        articles      = results.get("news_sentiment", {}).get("articles", [])
        existing_urls = set(report.get("citation_sources") or [])
        for a in articles:
            url = a.get("url")
            if url and url not in existing_urls:
                report.setdefault("citation_sources", []).append(url)
                existing_urls.add(url)

        report["tools_used"] = list(results.keys())
        return report
