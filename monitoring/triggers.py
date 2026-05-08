"""monitoring/triggers.py – evaluates trigger conditions for persistent monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yfinance as yf

from utils.config import get_settings

_settings = get_settings()


@dataclass
class TriggerResult:
    triggered: bool
    reason: str = ""
    triggers: list[str] = field(default_factory=list)
    new_state: dict = field(default_factory=dict)


class TriggerEvaluator:
    def evaluate(self, ticker: str, state: dict | None) -> TriggerResult:
        """Return TriggerResult indicating whether M.I.R.A. should run."""
        if state is None:
            return TriggerResult(triggered=True, reason="First run – no baseline.", triggers=["FIRST_RUN"])

        stock   = yf.Ticker(ticker)
        history = stock.history(period="35d")

        if history.empty:
            return TriggerResult(triggered=False, reason="No market data available.")

        closes  = history["Close"].values
        volumes = history["Volume"].values

        # ── price deviation trigger ───────────────────────────────────────────
        baseline_price = state.get("baseline_price") or float(closes[:-1].mean())
        std_price      = float(closes[:-1].std()) if len(closes) > 2 else 1.0
        latest_close   = float(closes[-1])
        price_dev      = abs(latest_close - baseline_price) / (std_price + 1e-9)

        triggers: list[str] = []
        reasons:  list[str] = []

        if price_dev > _settings.price_deviation_threshold:
            triggers.append("PRICE_DEVIATION")
            reasons.append(f"Price deviated {price_dev:.2f}σ from 30-day mean.")

        # ── volume spike trigger ──────────────────────────────────────────────
        baseline_vol = state.get("baseline_volume") or float(volumes[:-1].mean())
        latest_vol   = float(volumes[-1])
        if baseline_vol and latest_vol > _settings.volume_spike_threshold * baseline_vol:
            triggers.append("VOLUME_SPIKE")
            reasons.append(f"Volume {latest_vol:,.0f} > {_settings.volume_spike_threshold}× baseline {baseline_vol:,.0f}.")

        # ── new articles trigger (placeholder – article IDs from last run) ────
        # In production this would compare current article IDs vs last_article_ids.
        # Here we mark as triggered if the state has never seen articles.
        if not state.get("last_article_ids"):
            triggers.append("NEW_ARTICLES")
            reasons.append("No articles tracked yet – seeding baseline.")

        fired = bool(triggers)
        new_state = {
            "baseline_price":    float(np.mean(closes[-30:])),
            "baseline_volume":   float(np.mean(volumes[-30:])),
        }

        return TriggerResult(
            triggered=fired,
            reason="; ".join(reasons) if reasons else "No triggers fired.",
            triggers=triggers,
            new_state=new_state,
        )
