"""tools/news_sentiment.py – fetches news via NewsAPI and scores via FinBERT.

Trade-off documented in README:
  • FinBERT (local)  – no API cost, domain-tuned for finance, ~200 ms/article,
    but requires ~500 MB model download on first run and GPU is optional.
  • LLM-based        – zero extra setup, GPT-4 is highly context-aware, but
    adds latency & cost per article. FinBERT is the better default here.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from functools import lru_cache
from typing import Any

import requests

_FINBERT_MODEL = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def _get_pipeline():
    """Lazy-load FinBERT so it doesn't block startup."""
    from transformers import pipeline  # type: ignore

    return pipeline("sentiment-analysis", model=_FINBERT_MODEL, truncation=True)


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


class NewsSentimentTool:
    name = "news_sentiment"

    def __init__(self, news_api_key: str) -> None:
        self._key = news_api_key

    async def execute(self, company_name: str, ticker: str) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_and_score, company_name, ticker
        )

    def _fetch_and_score(self, company_name: str, ticker: str) -> dict[str, Any]:
        articles = self._fetch_articles(company_name, ticker)

        if not articles:
            return {
                "articles": [],
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_score": 0.0,
                "article_count": 0,
                "oldest_article_hours": None,
            }

        try:
            nlp = _get_pipeline()
        except Exception as e:
            # Handle NumPy 2.x or model loading issues gracefully
            logging.getLogger("mira").warning(f"Local FinBERT failed: {str(e)}. Falling back to neutral scores.")
            return {
                "articles": articles,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": len(articles)},
                "sentiment_score": 0.0,
                "article_count": len(articles),
                "error": str(e)
            }

        results = []
        dist: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        oldest_hours: float | None = None

        for art in articles:
            text = f"{art.get('title', '')}. {art.get('description') or ''}"
            text = text[:512]  # FinBERT max token safety
            scored = nlp(text)[0]
            label: str = scored["label"].lower()
            dist[label] = dist.get(label, 0) + 1

            # Age in hours
            from datetime import datetime, timezone

            published = art.get("publishedAt")
            age_h: float | None = None
            if published:
                try:
                    pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
                    if oldest_hours is None or age_h > oldest_hours:
                        oldest_hours = age_h
                except Exception:
                    pass

            results.append(
                {
                    "id": _article_id(art.get("url", "")),
                    "title": art.get("title"),
                    "url": art.get("url"),
                    "published_at": published,
                    "age_hours": round(age_h, 1) if age_h is not None else None,
                    "sentiment": label,
                    "confidence": round(scored["score"], 4),
                }
            )

        total = len(results)
        score = (dist["positive"] - dist["negative"]) / total if total else 0.0

        return {
            "articles": results,
            "sentiment_distribution": dist,
            "sentiment_score": round(score, 4),
            "article_count": total,
            "oldest_article_hours": round(oldest_hours, 1) if oldest_hours else None,
        }

    def _fetch_articles(self, company_name: str, ticker: str) -> list[dict]:
        if not self._key:
            return []
        query = f'"{company_name}" OR "{ticker}"'
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={requests.utils.quote(query)}"
            "&language=en&sortBy=publishedAt&pageSize=5"
        )
        try:
            resp = requests.get(
                url, headers={"Authorization": f"Bearer {self._key}"}, timeout=10
            )
            resp.raise_for_status()
            return resp.json().get("articles", [])
        except Exception:
            return []


if __name__ == "__main__":
    news = NewsSentiment()
    articles = news._fetch_articles("", "")
    print(articles)
