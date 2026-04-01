"""
Bonsai Fund — Risk Engine
Kelly criterion sizing, signal grading, and drawdown circuit breaker.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
def _cfg(key, default=None):
    """Lazy config loader — reads YAML, no circular import."""
    import os, yaml
    from pathlib import Path
    HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    SKILL_DIR = HERMES_HOME / "skills" / "bonsai-fund"
    try:
        raw = yaml.safe_load(open(SKILL_DIR / "config.yaml")) or {}
        return raw.get("bonsai_fund", {}).get(key, default)
    except:
        return default




@dataclass
class RiskLimits:
    max_position_pct:  float = 0.05
    max_concentration:  float = 0.20
    max_drawdown_pct:  float = 0.25
    daily_loss_limit:  float = 0.05
    min_edge_to_trade:  float = 0.03
    min_confidence:     float = 0.55
    min_vote_margin:    int   = 2
    circuit_breaker:    bool  = False


class RiskEngine:
    """
    Kelly-based risk management with circuit breaker.

    Kelly: f* = (b × p - q) / b
      b = (100 / yes_price_cents) - 1   (net decimal odds)
      p = estimated true probability
      q = 1 - p
    """

    def __init__(self, portfolio, limits: RiskLimits = None):
        self.portfolio = portfolio
        self.limits = limits or RiskLimits()
        self.peak_bankroll = portfolio.starting_capital
        self.daily_pnl = 0.0
        self.daily_start = portfolio.bankroll

    def update_peak(self):
        if self.portfolio.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.portfolio.bankroll

    @property
    def current_drawdown(self) -> float:
        return max(0.0, (self.peak_bankroll - self.portfolio.bankroll) / self.peak_bankroll)

    @property
    def is_circuit_breaker(self) -> bool:
        return self.limits.circuit_breaker or self.current_drawdown >= self.limits.max_drawdown_pct

    def update_daily_pnl(self, realized_pnl: float):
        self.daily_pnl += realized_pnl

    def check_circuit_breaker(self) -> tuple[bool, str]:
        if self.limits.circuit_breaker:
            return True, "MANUAL_CIRCUIT_BREAKER"
        dd = self.current_drawdown
        if dd >= self.limits.max_drawdown_pct:
            self.limits.circuit_breaker = True
            return True, f"DRAWDOWN {dd*100:.1f}% >= {self.limits.max_drawdown_pct*100:.0f}%"
        daily_loss = max(0.0, (self.daily_start - self.portfolio.bankroll) / self.daily_start)
        if daily_loss >= self.limits.daily_loss_limit:
            return True, f"DAILY_LOSS {daily_loss*100:.1f}% >= {self.limits.daily_loss_limit*100:.0f}%"
        return False, ""

    def compute_kelly_dollars(self, yes_price_cents: int, estimated_prob: float) -> float:
        if self.is_circuit_breaker:
            return 0.0
        market_prob = yes_price_cents / 100.0
        edge = estimated_prob - market_prob
        if edge < self.limits.min_edge_to_trade:
            return 0.0
        b = (100.0 / yes_price_cents) - 1.0
        p = estimated_prob
        q = 1.0 - p
        if b <= 0 or p <= 0 or p >= 1:
            return 0.0
        f_star = (b * p - q) / b
        if f_star <= 0:
            return 0.0
        max_dollar = self.portfolio.bankroll * self.limits.max_position_pct
        return round(min(f_star * self.portfolio.bankroll, max_dollar), 2)

    def compute_contracts(self, yes_price_cents: int, estimated_prob: float) -> int:
        kelly_dollars = self.compute_kelly_dollars(yes_price_cents, estimated_prob)
        if kelly_dollars <= 0:
            return 0
        contracts = kelly_dollars / (yes_price_cents / 100.0)
        return max(1, int(contracts))

    def assess_signal(
        self,
        yes_votes: int, no_votes: int, pass_votes: int,
        avg_confidence: float, avg_edge: float,
        market_prob: float, ticker: str
    ) -> tuple[bool, str, str]:
        """
        Returns (approved, grade, reason).
        """
        if self.is_circuit_breaker:
            return False, "CIRCUIT_BREAKER", "Circuit breaker tripped"

        total = yes_votes + no_votes + pass_votes
        margin = abs(yes_votes - no_votes)

        if margin < self.limits.min_vote_margin:
            return False, "NO_SIGNAL", f"Margin {margin} < {self.limits.min_vote_margin}"

        if avg_confidence < self.limits.min_confidence:
            return False, "LOW_CONFIDENCE", f"Conf {avg_confidence:.2f} < {self.limits.min_confidence}"

        direction = "BUY" if yes_votes > no_votes else "SELL"
        edge_dir = avg_edge if direction == "BUY" else -avg_edge

        if abs(edge_dir) < self.limits.min_edge_to_trade:
            return False, "LOW_EDGE", f"Edge {abs(edge_dir)*100:.1f}% < {self.limits.min_edge_to_trade*100:.0f}%"

        ci = avg_confidence * (margin / 7.0)

        if direction == "BUY":
            if ci >= 0.40 and margin >= 5:   grade = "STRONG_BUY"
            elif ci >= 0.20 and margin >= 3: grade = "BUY"
            elif edge_dir >= 0.05:           grade = "WEAK_BUY"
            else:                             grade = "PASS"
        else:
            if ci >= 0.40 and margin >= 5:   grade = "STRONG_SELL"
            elif ci >= 0.20 and margin >= 3: grade = "SELL"
            elif edge_dir >= 0.05:           grade = "WEAK_SELL"
            else:                             grade = "PASS"

        return True, grade, f"{direction} approved"

    def status(self) -> dict:
        self.update_peak()
        return {
            "bankroll":       round(self.portfolio.bankroll, 4),
            "peak":           round(self.peak_bankroll, 4),
            "drawdown_pct":   round(self.current_drawdown * 100, 2),
            "circuit_breaker": self.is_circuit_breaker,
            "daily_pnl":      round(self.daily_pnl, 4),
            "daily_start":    round(self.daily_start, 4),
            "limits": {
                "max_position_pct":  self.limits.max_position_pct,
                "max_drawdown_pct":  self.limits.max_drawdown_pct,
                "min_edge":          self.limits.min_edge_to_trade,
                "min_confidence":    self.limits.min_confidence,
                "min_vote_margin":   self.limits.min_vote_margin,
            },
            "kelly_max_dollar": round(self.portfolio.bankroll * self.limits.max_position_pct, 2),
        }
