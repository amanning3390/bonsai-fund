"""
Bonsai Fund — Risk Engine + Staged Drawdown Monitor
Kelly criterion sizing, signal grading, and 5-tier staged drawdown system.

Drawdown severity tiers:
  GREEN    (0-10%):   Normal operation — full Kelly sizing
  YELLOW   (10-15%):  Reduce to 50% size, raise confidence threshold +5pp
  ORANGE   (15-20%):  Reduce to 25% size, raise thresholds +10pp, begin analysis
  RED      (20-25%):  Reduce to 10% size, thresholds +15pp, run simulation
  CRITICAL (25%+):    HALT all trading, trigger full evolution cycle
  FROZEN   (35%+):    Full reset — manual review, retrain from best Gen
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional

def _cfg(key, default=None):
    import os, yaml
    from pathlib import Path
    HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    SKILL_DIR = HERMES_HOME / "skills" / "bonsai-fund"
    try:
        raw = yaml.safe_load(open(SKILL_DIR / "config.yaml")) or {}
        return raw.get("bonsai_fund", {}).get(key, default)
    except:
        return default


# ---------------------------------------------------------------------------
# Drawdown stages
# ---------------------------------------------------------------------------

class DrawdownStage(IntEnum):
    GREEN    = 0   # 0-10%  — normal trading
    YELLOW   = 1   # 10-15% — reduced sizing, elevated filters
    ORANGE   = 2   # 15-20% — heavy reduction, analysis mode
    RED      = 3   # 20-25% — near-halt, simulation required
    CRITICAL = 4   # 25-35% — full halt, evolution cycle
    FROZEN   = 5   # 35%+   — full reset + manual review


STAGE_META = {
    DrawdownStage.GREEN: {
        "name": "GREEN",
        "emoji": "🟢",
        "size_multiplier": 1.0,
        "conf_boost": 0.0,
        "margin_boost": 0,
        "description": "Normal operation",
        "can_trade": True,
        "training_required": False,
    },
    DrawdownStage.YELLOW: {
        "name": "YELLOW",
        "emoji": "🟡",
        "size_multiplier": 0.5,
        "conf_boost": 0.05,
        "margin_boost": 1,
        "description": "Reduce sizing, raise filters",
        "can_trade": True,
        "training_required": False,
    },
    DrawdownStage.ORANGE: {
        "name": "ORANGE",
        "emoji": "🟠",
        "size_multiplier": 0.25,
        "conf_boost": 0.10,
        "margin_boost": 2,
        "description": "Heavy reduction, begin analysis",
        "can_trade": True,
        "training_required": True,   # run analysis in background
        "training_action": "analyze",
    },
    DrawdownStage.RED: {
        "name": "RED",
        "emoji": "🔴",
        "size_multiplier": 0.10,
        "conf_boost": 0.15,
        "margin_boost": 2,
        "description": "Near-halt, simulation required",
        "can_trade": True,          # only highest conviction
        "training_required": True,
        "training_action": "simulate",
    },
    DrawdownStage.CRITICAL: {
        "name": "CRITICAL",
        "emoji": "🚨",
        "size_multiplier": 0.0,
        "conf_boost": 0.0,
        "margin_boost": 0,
        "description": "HALT — full evolution cycle",
        "can_trade": False,
        "training_required": True,
        "training_action": "evolve",
    },
    DrawdownStage.FROZEN: {
        "name": "FROZEN",
        "emoji": "❄️",
        "size_multiplier": 0.0,
        "conf_boost": 0.0,
        "margin_boost": 0,
        "description": "FULL RESET — manual review required",
        "can_trade": False,
        "training_required": True,
        "training_action": "deep_reset",
    },
}


# ---------------------------------------------------------------------------
# Staged Drawdown Monitor
# ---------------------------------------------------------------------------

@dataclass
class DrawdownEvent:
    stage: DrawdownStage
    drawdown_pct: float
    timestamp: str
    trades_in_stage: int = 0
    improvement_triggered: bool = False
    improvement_completed: bool = False
    improvement_result: str = ""


class StagedDrawdownMonitor:
    """
    Monitors drawdown in 6 stages. Each stage has progressively:
    - Smaller position sizes
    - Higher confidence/vote margin requirements
    - Mandatory training actions before resuming

    The key insight: a 10% drawdown means the swarm is wrong about something.
    Rather than just stopping, the system learns WHY and fixes it before resuming.
    """

    def __init__(self, portfolio, limits: RiskLimits = None):
        self.portfolio = portfolio
        self.limits = limits or RiskLimits()
        self.peak = portfolio.starting_capital
        self.stage = DrawdownStage.GREEN
        self.stage_entered_at: str = datetime.now(timezone.utc).isoformat()
        self.trades_in_current_stage: int = 0
        self.stage_history: list[DrawdownEvent] = []
        self._last_improvement_action_at: str = ""
        self.simulations_run: int = 0
        self.evolutions_run: int = 0

    def update_peak(self):
        if self.portfolio.bankroll > self.peak:
            self.peak = self.portfolio.bankroll

    @property
    def drawdown_pct(self) -> float:
        if self.peak <= 0:
            return 0.0
        return max(0.0, (self.peak - self.portfolio.bankroll) / self.peak)

    def current_stage(self) -> DrawdownStage:
        dd = self.drawdown_pct
        if dd >= 0.35: return DrawdownStage.FROZEN
        if dd >= 0.25: return DrawdownStage.CRITICAL
        if dd >= 0.20: return DrawdownStage.RED
        if dd >= 0.15: return DrawdownStage.ORANGE
        if dd >= 0.10: return DrawdownStage.YELLOW
        return DrawdownStage.GREEN

    def check(self) -> tuple[DrawdownStage, DrawdownStage, bool]:
        """
        Check drawdown and return (old_stage, new_stage, stage_changed).
        Call this at the start of every scan cycle.
        """
        self.update_peak()
        old = self.stage
        new = self.current_stage()
        changed = (old != new)

        if changed:
            self._enter_stage(new)

        return old, new, changed

    def _enter_stage(self, new: DrawdownStage):
        meta = STAGE_META[new]
        self.stage = new
        self.stage_entered_at = datetime.now(timezone.utc).isoformat()
        self.trades_in_current_stage = 0

        event = DrawdownEvent(
            stage=new,
            drawdown_pct=self.drawdown_pct,
            timestamp=self.stage_entered_at,
            trades_in_stage=0,
            improvement_triggered=meta["training_required"],
        )
        self.stage_history.append(event)

    def record_trade(self):
        """Call after each trade — counts trades in current stage."""
        self.trades_in_current_stage += 1
        if self.stage_history:
            self.stage_history[-1].trades_in_stage += 1

    def mark_improvement_done(self, result: str = ""):
        """Mark that the required training/improvement for current stage completed."""
        if self.stage_history:
            self.stage_history[-1].improvement_completed = True
            self.stage_history[-1].improvement_result = result
        self._last_improvement_action_at = datetime.now(timezone.utc).isoformat()

    def effective_limits(self) -> dict:
        """
        Return the effective risk limits adjusted for current drawdown stage.
        Used by RiskEngine to size positions correctly.
        """
        meta = STAGE_META[self.stage]
        base = self.limits
        return {
            "max_position_pct": base.max_position_pct * meta["size_multiplier"],
            "min_confidence": base.min_confidence + meta["conf_boost"],
            "min_vote_margin": base.min_vote_margin + meta["margin_boost"],
            "min_edge": base.min_edge_to_trade,
            "can_trade": meta["can_trade"],
            "stage": meta["name"],
            "stage_emoji": meta["emoji"],
            "description": meta["description"],
            "drawdown_pct": round(self.drawdown_pct * 100, 2),
            "training_required": meta["training_required"],
            "training_action": meta.get("training_action", ""),
        }

    def can_continue(self) -> tuple[bool, str]:
        """
        Returns (can_trade, reason). Checks both stage constraints
        and whether required improvement has completed.
        """
        meta = STAGE_META[self.stage]
        if not meta["can_trade"]:
            return False, f"{meta['emoji']} {meta['name']}: trading halted"

        if meta["training_required"] and self.stage_history:
            latest = self.stage_history[-1]
            if not latest.improvement_completed:
                return False, (
                    f"{meta['emoji']} {meta['name']}: "
                    f"{meta['training_action']} required before resuming. "
                    f"Trades this stage: {latest.trades_in_stage}"
                )

        return True, f"{meta['emoji']} {meta['name']}: clear to trade"

    def status(self) -> dict:
        self.update_peak()
        eff = self.effective_limits()
        return {
            "stage": eff["stage"],
            "stage_emoji": eff["stage_emoji"],
            "drawdown_pct": eff["drawdown_pct"],
            "peak": round(self.peak, 4),
            "bankroll": round(self.portfolio.bankroll, 4),
            "can_trade": eff["can_trade"],
            "training_required": eff["training_required"],
            "training_action": eff["training_action"],
            "trades_in_stage": self.trades_in_current_stage,
            "stage_entered_at": self.stage_entered_at,
            "simulations_run": self.simulations_run,
            "evolutions_run": self.evolutions_run,
            "effective_limits": {
                "max_position_pct": round(eff["max_position_pct"] * 100, 1),
                "min_confidence": round(eff["min_confidence"] * 100, 1),
                "min_vote_margin": eff["min_vote_margin"],
            },
            "stage_history": [
                {"stage": str(e.stage.name), "emoji": STAGE_META[e.stage]["emoji"],
                 "dd_pct": round(e.drawdown_pct * 100, 1),
                 "trades": e.trades_in_stage,
                 "improvement_done": e.improvement_completed,
                 "improvement_result": e.improvement_result[:60] if e.improvement_result else ""}
                for e in self.stage_history[-5:]
            ],
        }


# ---------------------------------------------------------------------------
# RiskEngine — updated to use staged drawdown
# ---------------------------------------------------------------------------

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

    # Staged drawdown thresholds (override max_drawdown_pct for multi-stage)
    dd_yellow_pct: float = 0.10
    dd_orange_pct: float = 0.15
    dd_red_pct:     float = 0.20
    dd_critical_pct: float = 0.25
    dd_frozen_pct:  float = 0.35


class RiskEngine:
    """
    Kelly-based risk management with STAGED drawdown circuit breaker.
    StagedDrawdownMonitor replaces the binary breaker with 6 severity tiers.
    """

    def __init__(self, portfolio, limits: RiskLimits = None):
        self.portfolio = portfolio
        self.limits = limits or RiskLimits()
        self.drawdown_monitor = StagedDrawdownMonitor(portfolio, self.limits)
        self.daily_pnl = 0.0
        self.daily_start = portfolio.bankroll

    def update_peak(self):
        self.drawdown_monitor.update_peak()

    @property
    def current_drawdown(self) -> float:
        return self.drawdown_monitor.drawdown_pct

    @property
    def is_circuit_breaker(self) -> bool:
        # Legacy compatibility — returns True only at CRITICAL or FROZEN
        return self.drawdown_monitor.stage >= DrawdownStage.CRITICAL

    @property
    def stage(self) -> DrawdownStage:
        return self.drawdown_monitor.stage

    def check_circuit_breaker(self) -> tuple[bool, str]:
        """
        Returns (tripped, reason).
        Also triggers training actions based on stage transitions.
        """
        old, new, changed = self.drawdown_monitor.check()

        if changed and new >= DrawdownStage.ORANGE:
            meta = STAGE_META[new]
            return True, (
                f"{meta['emoji']} DRAWDOWN STAGE: {meta['name']} "
                f"({self.drawdown_monitor.drawdown_pct*100:.1f}%) — {meta['description']}"
            )

        # Daily loss check
        daily_loss = max(0.0, (self.daily_start - self.portfolio.bankroll) / self.daily_start)
        if daily_loss >= self.limits.daily_loss_limit:
            return True, f"DAILY_LOSS {daily_loss*100:.1f}% >= {self.limits.daily_loss_limit*100:.0f}%"

        can_trade, reason = self.drawdown_monitor.can_continue()
        if not can_trade:
            return True, reason

        return False, ""

    def compute_kelly_dollars(self, yes_price_cents: int, estimated_prob: float) -> float:
        eff = self.drawdown_monitor.effective_limits()
        if not eff["can_trade"]:
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

        base_max = self.portfolio.bankroll * self.limits.max_position_pct
        stage_max = self.portfolio.bankroll * eff["max_position_pct"]
        kelly_dollars = min(f_star * self.portfolio.bankroll, base_max, stage_max)
        return round(kelly_dollars, 2)

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
        Uses stage-adjusted limits from StagedDrawdownMonitor.
        """
        can_trade, reason = self.drawdown_monitor.can_continue()
        if not can_trade:
            return False, "STAGE_HALT", reason

        eff = self.drawdown_monitor.effective_limits()

        total = yes_votes + no_votes + pass_votes
        margin = abs(yes_votes - no_votes)

        # Stage-adjusted thresholds
        min_conf = eff["min_confidence"]
        min_margin = eff["min_vote_margin"]

        if margin < min_margin:
            return False, "NO_SIGNAL", f"Margin {margin} < {min_margin} (stage: {eff['stage']})"

        if avg_confidence < min_conf:
            return False, "LOW_CONFIDENCE", (
                f"Conf {avg_confidence:.2f} < {min_conf:.2f} "
                f"(+{eff['min_confidence']*100:.0f}pp stage boost)"
            )

        direction = "BUY" if yes_votes > no_votes else "SELL"
        edge_dir = avg_edge if direction == "BUY" else -avg_edge

        if abs(edge_dir) < self.limits.min_edge_to_trade:
            return False, "LOW_EDGE", f"Edge {abs(edge_dir)*100:.1f}% < {self.limits.min_edge_to_trade*100:.0f}%"

        ci = avg_confidence * (margin / 7.0)

        if direction == "BUY":
            if ci >= 0.40 and margin >= 5:   grade = "STRONG_BUY"
            elif ci >= 0.20 and margin >= 3:  grade = "BUY"
            elif edge_dir >= 0.05:            grade = "WEAK_BUY"
            else:                              grade = "PASS"
        else:
            if ci >= 0.40 and margin >= 5:   grade = "STRONG_SELL"
            elif ci >= 0.20 and margin >= 3:  grade = "SELL"
            elif edge_dir >= 0.05:            grade = "WEAK_SELL"
            else:                              grade = "PASS"

        stage_note = f" [stage: {eff['stage_emoji']}{eff['stage']}]"
        return True, grade, f"{direction} approved{stage_note}"

    def status(self) -> dict:
        self.update_peak()
        dm_status = self.drawdown_monitor.status()
        return {
            "bankroll":       round(self.portfolio.bankroll, 4),
            "peak":           round(self.drawdown_monitor.peak, 4),
            "drawdown_pct":   dm_status["drawdown_pct"],
            "stage":          dm_status["stage"],
            "stage_emoji":    dm_status["stage_emoji"],
            "circuit_breaker": self.is_circuit_breaker,
            "can_trade":      dm_status["can_trade"],
            "daily_pnl":      round(self.daily_pnl, 4),
            "daily_start":    round(self.daily_start, 4),
            "effective_limits": dm_status["effective_limits"],
            "limits": {
                "max_position_pct":  self.limits.max_position_pct,
                "max_drawdown_pct":  self.limits.max_drawdown_pct,
                "min_edge":          self.limits.min_edge_to_trade,
                "min_confidence":    self.limits.min_confidence,
                "min_vote_margin":   self.limits.min_vote_margin,
            },
            "kelly_max_dollar": round(self.portfolio.bankroll * self.limits.max_position_pct, 2),
            "simulations_run": dm_status["simulations_run"],
            "evolutions_run":  dm_status["evolutions_run"],
            "stage_history":   dm_status["stage_history"],
        }

    def update_daily_pnl(self, realized_pnl: float):
        self.daily_pnl += realized_pnl
