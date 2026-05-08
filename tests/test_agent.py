"""tests/test_agent.py – documented test cases per assessment Section 3.D.

Test Cases:
  1. AAPL analysis should always include self-correlation = 1.0
  2. Unknown ticker should return a graceful error (not a hallucinated report)
  3. A query about a delisted ticker should fail gracefully
"""
from __future__ import annotations

import pytest
import pytest_asyncio

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_redis():
    """Return a RedisClient (will skip if Redis unavailable)."""
    from storage.redis_client import RedisClient
    client = RedisClient()
    if not client.ping():
        pytest.skip("Redis not available – skipping integration test.")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Test Case 1: AAPL self-correlation = 1.0
# Expected: correlation_analysis.all_correlations["AAPL"] == 1.0
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_aapl_self_correlation_is_one():
    """AAPL analysis must include correlation with itself = 1.0."""
    from agent.core import AgentCore
    import uuid

    redis  = _make_redis()
    agent  = AgentCore()
    job_id = str(uuid.uuid4())

    redis.set_job_status(job_id, {"job_id": job_id, "status": "queued", "progress": 0, "tool_calls_used": 0})
    await agent.run_analysis(job_id, "Analyse Apple Inc. (AAPL)")

    status = redis.get_job_status(job_id)
    assert status["status"] == "completed", f"Expected completed, got {status['status']}: {status.get('error')}"

    result = redis.get_job_result(job_id)
    assert result is not None, "No result stored."

    corr = result.get("correlation_analysis", {}).get("all_correlations", {})
    assert "AAPL" in corr, "AAPL key missing from all_correlations."
    assert corr["AAPL"] == 1.0, f"Expected 1.0, got {corr['AAPL']}"


# ─────────────────────────────────────────────────────────────────────────────
# Test Case 2: Unknown ticker → graceful error (no hallucinated report)
# Expected: status == "failed", error contains "not found" or "invalid"
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_unknown_ticker_graceful_error():
    """Unknown ticker should fail gracefully, not produce a hallucinated report."""
    from agent.core import AgentCore
    import uuid

    redis  = _make_redis()
    agent  = AgentCore()
    job_id = str(uuid.uuid4())

    redis.set_job_status(job_id, {"job_id": job_id, "status": "queued", "progress": 0, "tool_calls_used": 0})
    await agent.run_analysis(job_id, "Analyse ZZZZZ99_UNKNOWN_TICKER_XYZ")

    status = redis.get_job_status(job_id)
    assert status["status"] == "failed", f"Expected 'failed', got '{status['status']}'"

    error_msg = (status.get("error") or "").lower()
    assert any(kw in error_msg for kw in ["not found", "invalid", "no tradable", "unknown"]), (
        f"Error message does not indicate ticker not found: '{status.get('error')}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test Case 3: Delisted ticker → graceful failure
# Expected: status == "failed", error contains "not found", "delisted", or "no data"
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_delisted_ticker_fails_gracefully():
    """A query about a delisted company (Toys R Us – TOY) must fail gracefully."""
    from agent.core import AgentCore
    import uuid

    redis  = _make_redis()
    agent  = AgentCore()
    job_id = str(uuid.uuid4())

    redis.set_job_status(job_id, {"job_id": job_id, "status": "queued", "progress": 0, "tool_calls_used": 0})
    await agent.run_analysis(job_id, "Analyse Toys R Us (TOY)")

    status = redis.get_job_status(job_id)
    assert status["status"] == "failed", f"Expected 'failed', got '{status['status']}'"

    error_msg = (status.get("error") or "").lower()
    assert any(kw in error_msg for kw in ["not found", "delisted", "no tradable", "no data", "invalid"]), (
        f"Error does not mention ticker unavailability: '{status.get('error')}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: MarketDataTool raises ValueError for bad ticker (no Redis needed)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_market_data_tool_raises_on_bad_ticker():
    """MarketDataTool.execute must raise ValueError for unknown tickers."""
    from tools.market_data import MarketDataTool

    tool = MarketDataTool()
    with pytest.raises(ValueError, match="not found"):
        await tool.execute("ZZZZZ99_INVALID")
