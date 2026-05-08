"""agent/planner.py – LLM-powered planning with JSON output."""
from __future__ import annotations

import json

from openai import AsyncOpenAI

from utils.config import get_settings

_settings = get_settings()

_INITIAL_SYSTEM = """
You are M.I.R.A.'s research planning assistant. Given a company analysis query, produce
a JSON plan specifying tools to use IN ORDER. Think step-by-step (chain-of-thought) before
emitting JSON.

Available tools:
  - market_data        : current price, fundamentals, revenue history
  - news_sentiment     : recent news articles + FinBERT sentiment scores
  - peer_correlation   : Pearson correlations vs S&P 500, sector ETF, peers

Respond ONLY with a valid JSON object:
{
  "ticker":  "TSLA",
  "company": "Tesla, Inc.",
  "steps": [
    {"tool": "market_data",      "reason": "…"},
    {"tool": "news_sentiment",   "reason": "…"},
    {"tool": "peer_correlation", "reason": "…"}
  ]
}
""".strip()

_REFLECTION_SYSTEM = """
You are M.I.R.A.'s adaptive research planner. Based on the reflection findings, produce
an additional research plan. Return ONLY a JSON plan (same schema as initial plan).
""".strip()


class Planner:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=_settings.openai_api_key)

    async def create_initial_plan(self, query: str) -> dict:
        resp = await self._client.chat.completions.create(
            model=_settings.llm_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",  "content": _INITIAL_SYSTEM},
                {"role": "user",    "content": query},
            ],
        )
        return json.loads(resp.choices[0].message.content)

    async def create_reflection_plan(self, reflection_result: dict, original_query: str) -> dict:
        prompt = (
            f"Original query: {original_query}\n\n"
            f"Reflection findings:\n{json.dumps(reflection_result, indent=2)}\n\n"
            "Produce an additional research plan addressing the gaps above."
        )
        resp = await self._client.chat.completions.create(
            model=_settings.llm_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _REFLECTION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content)
