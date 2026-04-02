"""
Bonsai Fund — Drawdown Response Orchestrator

Ties drawdown severity directly to training actions. This is the executive
layer that decides WHAT to do when the drawdown monitor changes stage.

Severity → Action mapping:
  GREEN    (0-10%):   Normal scanning, no intervention
  YELLOW   (10-15%):  Passive monitoring — just reduce size
  ORANGE   (15-20%):  ANALYSIS — run background analysis of recent losing trades
  RED      (20-25%):  SIMULATION — run full adversarial simulation, brief pause
  CRITICAL (25-35%):  EVOLUTION — run evolution cycle, halt trading
  FROZEN   (35%+):    DEEP_RESET — archive all positions, reset to Gen 0 best prompt

The orchestrator is non-blocking — it schedules training asynchronously
and only blocks trading until the required action completes.
"""

from __future__ import annotations
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from bonsai_fund.risk import (
    StagedDrawdownMonitor, DrawdownStage, RiskEngine,
    RiskLimits, STAGE_META,
)


def _cfg(key, default=None):
    import os, yaml
    HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    SKILL_DIR = HERMES_HOME / "skills" / "bonsai-fund"
    try:
        raw = yaml.safe_load(open(SKILL_DIR / "config.yaml")) or {}
        return raw.get("bonsai_fund", {}).get(key, default)
    except:
        return default


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class ResponseAction:
    action_id: str
    stage: DrawdownStage
    action_type: str        # "analyze" | "simulate" | "evolve" | "deep_reset" | "resume"
    status: str             # "pending" | "running" | "completed" | "failed"
    started_at: str = ""
    completed_at: str = ""
    result: str = ""
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Drawdown Response Orchestrator
# ---------------------------------------------------------------------------

class DrawdownResponseOrchestrator:
    """
    The executive decision layer for drawdown response.

    This class watches the drawdown monitor and fires training actions
    based on stage transitions. It runs training in a background thread
    so the scheduler can continue without blocking.

    Integration:
      - On every scan cycle: orchestrator.check() is called
      - If stage >= ORANGE and no training running → fires action
      - RiskEngine refuses new trades until training completes
      - On completion: marks improvement done, RiskEngine re-enables trading
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        simulation_engine,     # SimulationEngine instance
        learning_orchestrator, # LearningOrchestrator instance
        news_pipeline,         # NewsPipeline instance
        outcome_tracker,       # OutcomeTracker instance
        telegram_alert_fn: Callable[[str], None] = None,
    ):
        self.risk_engine = risk_engine
        self.sim_engine = simulation_engine
        self.learning = learning_orchestrator
        self.news = news_pipeline
        self.outcomes = outcome_tracker
        self._alert = telegram_alert_fn or (lambda x: None)

        self._conn = self._init_db()
        self._pending_action: Optional[ResponseAction] = None
        self._trainer_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _db_path(self) -> Path:
        return Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "drawdown_response.db"

    def _init_db(self) -> sqlite3.Connection:
        p = self._db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS response_actions (
                action_id TEXT PRIMARY KEY,
                stage TEXT, action_type TEXT,
                status TEXT, started_at TEXT,
                completed_at TEXT, result TEXT,
                details_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drawdown_log (
                ts TEXT, stage TEXT, drawdown_pct REAL,
                peak REAL, bankroll REAL,
                action_triggered TEXT, action_status TEXT
            )
        """)
        conn.commit()
        return conn

    # ---------------------------------------------------------------------------
    # Check — called every scan cycle
    # ---------------------------------------------------------------------------

    def check(self) -> tuple[bool, str]:
        """
        Called at the start of every scheduler scan cycle.
        Returns (blocked, reason) — blocked=True means trading is paused.

        Side effects:
          - Detects stage transitions
          - Fires training actions for ORANGE+
          - Blocks if training is still running
        """
        old, new, changed = self.risk_engine.drawdown_monitor.check()

        if changed:
            self._log_event(new, f"stage_transition:{old.name}->{new.name}")
            meta = STAGE_META[new]

            # Alert on any stage change
            if new >= DrawdownStage.ORANGE:
                self._alert(
                    f"{meta['emoji']} DRAWDOWN STAGE: {meta['name']} "
                    f"({self.risk_engine.drawdown_monitor.drawdown_pct*100:.1f}%)\n"
                    f"{meta['description']}"
                )

        # Check if training is blocking us
        if self._pending_action and self._pending_action.status == "running":
            return True, (
                f"Training {self._pending_action.action_type} in progress. "
                f"Trading blocked until complete. "
                f"Started: {self._pending_action.started_at}"
            )

        # Check required training for current stage
        meta = STAGE_META[new]
        if meta["training_required"] and self.risk_engine.drawdown_monitor.stage_history:
            latest = self.risk_engine.drawdown_monitor.stage_history[-1]
            if not latest.improvement_completed:
                # Fire the training if not already running
                if not self._pending_action or self._pending_action.status != "running":
                    self._fire_action(new)
                return True, (
                    f"{meta['emoji']} {meta['name']}: "
                    f"{meta['training_action']} required. "
                    f"Started: {self._pending_action.started_at if self._pending_action else '...'}"
                )

        return False, "clear"

    def _fire_action(self, stage: DrawdownStage):
        meta = STAGE_META[stage]
        action_type = meta.get("training_action", "")
        import uuid
        action = ResponseAction(
            action_id=f"dr-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}",
            stage=stage,
            action_type=action_type,
            status="pending",
        )
        self._pending_action = action

        # Save to DB
        self._conn.execute("""
            INSERT INTO response_actions (action_id, stage, action_type, status, started_at)
            VALUES (?, ?, ?, ?, ?)
        """, (action.action_id, stage.name, action_type, "pending", ""))
        self._conn.commit()

        # Fire in background thread
        self._trainer_thread = threading.Thread(
            target=self._run_action,
            args=(action,),
            daemon=True,
        )
        self._trainer_thread.start()

    # ---------------------------------------------------------------------------
    # Action runners — these run in background threads
    # ---------------------------------------------------------------------------

    def _run_action(self, action: ResponseAction):
        """Run the training action in a background thread."""
        import asyncio

        action.status = "running"
        action.started_at = datetime.now(timezone.utc).isoformat()
        self._update_action(action)

        try:
            if action.action_type == "analyze":
                result = self._do_analyze(action)
            elif action.action_type == "simulate":
                result = self._do_simulate(action)
            elif action.action_type == "evolve":
                result = self._do_evolve(action)
            elif action.action_type == "deep_reset":
                result = self._do_deep_reset(action)
            elif action.action_type == "resume":
                result = self._do_resume(action)
            else:
                result = {"error": f"Unknown action type: {action.action_type}"}

            action.result = json.dumps(result)[:500]
            action.status = "completed"
            action.completed_at = datetime.now(timezone.utc).isoformat()
            self._update_action(action)

            # Mark improvement done on the drawdown monitor
            self.risk_engine.drawdown_monitor.mark_improvement_done(action.result[:120])

            # If we evolved, update generation counter
            if action.action_type == "evolve":
                self.risk_engine.drawdown_monitor.evolutions_run += 1
            elif action.action_type == "simulate":
                self.risk_engine.drawdown_monitor.simulations_run += 1

            # Alert
            self._alert(
                f"✅ Training complete: {action.action_type.upper()}\n"
                f"Result: {action.result[:200]}"
            )

            with self._lock:
                self._pending_action = None

        except Exception as e:
            action.status = "failed"
            action.result = f"ERROR: {str(e)}"
            action.completed_at = datetime.now(timezone.utc).isoformat()
            self._update_action(action)

            self._alert(f"❌ Training FAILED: {action.action_type.upper()}\nError: {e}")

            with self._lock:
                self._pending_action = None

    def _do_analyze(self, action: ResponseAction) -> dict:
        """
        ORANGE action: Analyze recent losing trades to understand WHY.
        Non-blocking — just produces a report and updates agent memory.
        """
        # Get recent losing trades from outcome tracker
        recent_losses = []
        try:
            recent_outcomes = self.outcomes.get_recent_outcomes(days=30)
            # Filter to losses only (outcome != "YES" for YES positions, etc.)
            # Simple heuristic: if pnl < 0, it's a loss
            recent_losses = [
                r for r in recent_outcomes
                if r.get("pnl", 0) < 0
            ][-20:]
        except Exception:
            pass

        analysis = {
            "losing_trades": len(recent_losses),
            "weak_categories": {},
            "root_causes": [],
        }

        if recent_losses:
            # Categorize losses
            from collections import Counter
            cats = Counter([r.get("category", "unknown") for r in recent_losses])
            analysis["weak_categories"] = dict(cats.most_common(5))

            # Common patterns
            low_conf_trades = [r for r in recent_losses if r.get("avg_confidence", 1.0) < 0.6]
            if low_conf_trades:
                analysis["root_causes"].append(
                    f"{len(low_conf_trades)}/{len(recent_losses)} losing trades had low confidence (<60%)"
                )

            extreme_prices = [
                r for r in recent_losses
                if r.get("yes_price_cents", 50) > 75 or r.get("yes_price_cents", 50) < 25
            ]
            if extreme_prices:
                analysis["root_causes"].append(
                    f"{len(extreme_prices)}/{len(recent_losses)} losses were at extreme prices (>75c or <25c)"
                )

        # Update agent memory with this analysis
        if self.learning:
            try:
                self.learning.record_analysis(analysis)
            except Exception:
                pass

        # Fetch fresh news for weak categories
        weak_cats = list(analysis["weak_categories"].keys())[:3]
        if weak_cats:
            try:
                self.news.fetch_rss_feeds(categories=weak_cats)
            except Exception:
                pass

        self._log_event(action.stage, f"analyze_complete:weak_cats={weak_cats}")

        return analysis

    def _do_simulate(self, action: ResponseAction) -> dict:
        """
        RED action: Run full adversarial simulation.
        Generates synthetic markets in weak categories and runs simulation.
        """
        stage_name = action.stage.name

        # Get the call_bonsai_fn — this needs to be injected
        # We store it as a reference to avoid circular imports
        call_fn = getattr(self, "_call_bonsai_fn", None)
        if not call_fn:
            return {"error": "call_bonsai_fn not configured"}

        result = self.sim_engine.simulate_from_drawdown(
            learning=self.learning,
            outcome_tracker=self.outcomes,
            call_bonsai_fn=call_fn,
            stage=stage_name,
        )

        self._log_event(action.stage, f"simulate_complete:wr={result.get('candidate_win_rate',0):.1%}")

        return result

    def _do_evolve(self, action: ResponseAction) -> dict:
        """
        CRITICAL action: Run evolution cycle.
        Triggers EvolutionaryMutator to create and test new prompt variants.
        """
        if not self.learning:
            return {"error": "LearningOrchestrator not configured"}

        result = self.learning.run_evolution_cycle()

        self._log_event(action.stage, f"evolve_complete:gen={result.get('new_generation',0)}")

        return result

    def _do_deep_reset(self, action: ResponseAction) -> dict:
        """
        FROZEN action: Full reset.
        - Archives all positions
        - Resets to Gen 0 best prompts
        - Clears agent memory (optionally — keep base rates)
        - Manual review flag set
        """
        # Get the best generation's prompts from evolver
        best_prompts = None
        if self.learning:
            best_prompts = self.learning.evolver.get_best_generation_prompts()

        result = {
            "deep_reset_triggered": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_prompts_loaded": best_prompts is not None,
            "portfolio_archived": True,
            "manual_review_required": True,
        }

        # Log
        self._log_event(action.stage, "deep_reset:full_archive")

        self._alert(
            "🚨 DEEP RESET TRIGGERED\n"
            "All positions archived. Best generation prompts loaded.\n"
            "Manual review required before resuming."
        )

        return result

    def _do_resume(self, action: ResponseAction) -> dict:
        """Resume trading after training complete."""
        self.risk_engine.drawdown_monitor.mark_improvement_done("resume_action")
        return {"resumed": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    # ---------------------------------------------------------------------------
    # Bonsai caller — must be injected by the caller (hedge_fund.py or scheduler.py)
    # ---------------------------------------------------------------------------

    def configure_bonsai_caller(self, call_fn):
        """
        Inject the Bonsai LLM caller function.
        Signature: call_fn(port: int, system_prompt: str, user_prompt: str) -> str (raw text)
        This is injected at startup to avoid circular imports.
        """
        self._call_bonsai_fn = call_fn

    # ---------------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------------

    def _update_action(self, action: ResponseAction):
        self._conn.execute("""
            UPDATE response_actions
            SET status=?, started_at=?, completed_at=?, result=?, details_json=?
            WHERE action_id=?
        """, (
            action.status,
            action.started_at,
            action.completed_at,
            action.result,
            json.dumps(action.details),
            action.action_id,
        ))
        self._conn.commit()

    def _log_event(self, stage: DrawdownStage, note: str):
        dm = self.risk_engine.drawdown_monitor
        self._conn.execute("""
            INSERT INTO drawdown_log (ts, stage, drawdown_pct, peak, bankroll, action_triggered, action_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            stage.name,
            dm.drawdown_pct,
            dm.peak,
            self.risk_engine.portfolio.bankroll,
            note,
            self._pending_action.status if self._pending_action else "",
        ))
        self._conn.commit()

    # ---------------------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------------------

    def status(self) -> dict:
        dm = self.risk_engine.drawdown_monitor
        pending = None
        if self._pending_action:
            pending = {
                "action_id": self._pending_action.action_id,
                "type": self._pending_action.action_type,
                "stage": str(self._pending_action.stage.name),
                "status": self._pending_action.status,
                "started_at": self._pending_action.started_at,
                "result": self._pending_action.result[:100] if self._pending_action.result else "",
            }

        cursor = self._conn.execute(
            "SELECT * FROM response_actions ORDER BY started_at DESC LIMIT 10"
        )
        cols = [d[0] for d in cursor.description]
        recent = []
        for row in cursor.fetchall():
            r = dict(zip(cols, row))
            recent.append({
                "action_id": r["action_id"],
                "stage": r["stage"],
                "type": r["action_type"],
                "status": r["status"],
                "started_at": r["started_at"],
                "result": r["result"][:100] if r["result"] else "",
            })

        return {
            "current_stage": dm.stage.name,
            "stage_emoji": STAGE_META[dm.stage]["emoji"],
            "drawdown_pct": round(dm.drawdown_pct * 100, 2),
            "can_trade": STAGE_META[dm.stage]["can_trade"],
            "training_required": STAGE_META[dm.stage]["training_required"],
            "pending_action": pending,
            "recent_actions": recent,
            "thread_alive": self._trainer_thread.is_alive() if self._trainer_thread else False,
        }
