"""
Bonsai Fund — Self-Learning: Learning Orchestrator

The recursive self-learning orchestrator. Runs after every scan cycle
and after every market resolution. Ties together:
  OutcomeTracker → AgentMemory → MarketClassifier → EvolutionaryMutator

The learning loop:
  SHORT (every 30 min): Record votes, update agent memory weights
  MEDIUM (after resolution): Update base rates, refine category affinities
  LONG (every 100+ new trades): Run evolution cycle, create new prompt variants
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bonsai_fund.self_learning.outcome_tracker import OutcomeTracker, TradeOutcome
from bonsai_fund.self_learning.agent_memory import AgentMemory
from bonsai_fund.self_learning.market_classifier import MarketClassifier
from bonsai_fund.self_learning.evolver import EvolutionaryMutator, EvolvedAgent


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


class LearningOrchestrator:
    """
    The meta-cognitive layer that orchestrates all three learning time scales.

    After each scan:
      → Record agent votes in AgentMemory (for context-sensitive weighting)

    After each market resolution:
      → OutcomeTracker records the result
      → MarketClassifier updates base rates + agent affinities
      → AgentMemory records the resolution, updates per-agent weights
      → EvolutionaryMutator checks if it's time to evolve

    After every 100+ new trades:
      → EvolutionaryMutator runs a full evolution cycle
      → Weak agents are replaced with mutated variants of strong ones
      → New prompts are tested in the next scan cycle
    """

    def __init__(self):
        self.outcome_tracker = OutcomeTracker()
        self.agent_memory = AgentMemory()
        self.market_classifier = MarketClassifier()
        self.evolver = EvolutionaryMutator()
        self._initialized = False
        self._init_evolver()

    def _init_evolver(self):
        """Seed Generation 0 on first run."""
        if self._initialized:
            return
        # Check if Generation 0 exists
        prompts = self.evolver.get_current_prompts()
        if not any(prompts.values()):
            self.evolver.seed_generation_0()
        self._initialized = True

    # ---------------------------------------------------------------------------
    # Post-scan learning — SHORT-TERM
    # ---------------------------------------------------------------------------

    def record_scan_votes(
        self,
        markets: list,
        all_votes: list[list],  # [[votes for market 0], [votes for market 1], ...]
        portfolio
    ):
        """
        After a scan, record all agent votes in memory.
        This enables context-sensitive weighting on the NEXT scan.
        """
        for market, votes in zip(markets, all_votes):
            if not votes:
                continue
            category_name, base_rate, confidence = self.market_classifier.get_base_rate(market)
            for vote in votes:
                self.agent_memory.record_vote(
                    vote=vote,
                    market_prob=market.implied_prob_yes,
                    market_category=category_name,
                )

    # ---------------------------------------------------------------------------
    # Post-resolution learning — MEDIUM-TERM
    # ---------------------------------------------------------------------------

    def process_resolution(
        self,
        ticker: str,
        resolution: str,  # YES or NO
        market,
    ):
        """
        After a market resolves, update all learning systems.
        Called by the scheduler when it detects a resolved market.
        """
        # 1. Update outcome tracker
        outcome = self.outcome_tracker.resolve_market(
            ticker=ticker,
            resolution=resolution,
            yes_fill=market.yes_price_cents,
            no_fill=market.no_price_cents,
            contracts=1,
        )

        # 2. Update agent memory with the resolution
        self.agent_memory.record_resolution(ticker, resolution)

        # 3. Update market classifier + agent affinities
        agent_scores = self.outcome_tracker.get_agent_scores()
        self.market_classifier.update_from_outcome(
            market=market,
            resolution=resolution,
            agent_scores=agent_scores,
        )

        # 4. Update evolver performance
        self._update_evolver_performance()

        # 5. Check if we should run an evolution cycle
        total_trades = self.outcome_tracker.get_total_edge_analysis()["total_trades"]
        if self.evolver.should_evolve(total_trades):
            result = self.run_evolution_cycle()
            return result

        return None

    def _update_evolver_performance(self):
        """Sync evolver with latest agent scores."""
        agent_scores = self.outcome_tracker.get_agent_scores()
        current_prompts = self.evolver.get_current_prompts()
        for score in agent_scores:
            aid = score["agent_id"]
            phash = None
            # Find the hash for this agent's current active prompt
            # (simplified: use agent_id as key)
            current_prompts_hash = self.evolver._compute_hash(
                current_prompts.get(aid, "") + ""
            )
            self.evolver.update_performance(
                agent_id=aid,
                prompt_hash=f"agent_{aid}_gen0",
                trade_count=score["total_votes"],
                win_rate=score["win_rate"],
                avg_edge=score.get("avg_edge_raw", 0.0),
                sharpe_approx=score.get("sharpe_approx", 0.0),
            )

    # ---------------------------------------------------------------------------
    # LONG-TERM — Evolutionary cycle
    # ---------------------------------------------------------------------------

    def run_evolution_cycle(self) -> dict:
        """
        Trigger a full evolutionary cycle.
        Called automatically every 100+ trades, or manually.
        """
        agent_scores = self.outcome_tracker.get_agent_scores()
        result = self.evolver.run_evolution_cycle(agent_scores)

        if result.get("status") == "success":
            # Apply new prompts immediately
            new_prompts = self.evolver.get_current_prompts()
            # Log the evolution event
            self._log_learning_event("EVOLUTION", result)

        return result

    # ---------------------------------------------------------------------------
    # Context-sensitive voting
    # ---------------------------------------------------------------------------

    def get_weighted_votes(self, votes, market, days_to_event: int) -> list[tuple]:
        """
        Return agent votes with context-sensitive weights applied.
        For use in the hedge_fund.py signal aggregation.
        """
        return self.agent_memory.get_weighted_votes(
            votes=votes,
            market_prob=market.implied_prob_yes,
            market_category=self.market_classifier.classify(market).name,
            days_to_event=days_to_event,
        )

    # ---------------------------------------------------------------------------
    # Status and reporting
    # ---------------------------------------------------------------------------

    def get_learning_status(self) -> dict:
        """Return a comprehensive status of all learning systems."""
        total = self.outcome_tracker.get_total_edge_analysis()
        agent_scores = self.outcome_tracker.get_agent_scores()
        cat_rates = self.outcome_tracker.get_category_base_rates()
        agent_summary = self.agent_memory.get_agent_summary()
        evo_log = self.evolver.get_evolution_log(limit=5)
        current_prompts = self.evolver.get_current_prompts()
        all_time_best = self.evolver.get_all_time_best()

        return {
            "learning_active": True,
            "total_trades_recorded": total["total_trades"],
            "system_edge_captured": total["avg_edge"],
            "agent_scores": agent_scores,
            "category_base_rates": cat_rates,
            "agent_memory_profiles": agent_summary,
            "evolver_generation": self.evolver._generation,
            "evolution_log": evo_log,
            "current_prompts_active": {str(k): v[:60] + "..." for k, v in current_prompts.items()},
            "all_time_best_agents": [
                {
                    "agent_id": a.agent_id,
                    "win_rate": round(a.win_rate, 4),
                    "avg_edge": round(a.avg_edge, 4),
                    "fitness": round(a.fitness, 4),
                    "trade_count": a.trade_count,
                    "lineage_length": len(a.lineage),
                }
                for a in all_time_best[:5]
            ],
        }

    def format_learning_report(self) -> str:
        """Format a human-readable learning status report."""
        status = self.get_learning_status()

        lines = [
            "🧠 BONSAI SWARM — RECURSIVE LEARNING STATUS",
            "=" * 48,
            f"Evolution generation:  Gen {status['evolver_generation']}",
            f"Total trades learned: {status['total_trades_recorded']}",
            f"Avg edge captured:    {status['system_edge_captured']:+.4f}",
            "",
            "AGENT PERFORMANCE (win rate × edge)",
            "-" * 48,
        ]

        for score in sorted(status["agent_scores"], key=lambda x: x["agent_id"]):
            wr = score.get("win_rate", 0) * 100
            ae = score.get("avg_edge_raw", 0)
            tot = score.get("total_votes", 0)
            lines.append(
                f"  [{score['agent_id']}] {score['agent_name']:20s} "
                f"WR={wr:5.1f}%  edge={ae:+.3f}  n={tot:4d}"
            )

        lines += ["", "CATEGORY BASE RATES", "-" * 48]
        for cat, data in status.get("category_base_rates", {}).items():
            br = data.get("base_rate", 0.5) * 100
            n = data.get("trades", 0)
            lines.append(f"  {cat:30s} base={br:5.1f}%  n={n:3d}")

        if status.get("evolution_log"):
            lines += ["", "RECENT EVOLUTION EVENTS", "-" * 48]
            for entry in status["evolution_log"][:3]:
                try:
                    details = json.loads(entry.get("details", "{}"))
                    evt = details.get("event", entry.get("event", ""))
                    gen = entry.get("generation", "?")
                    lines.append(f"  Gen {gen}: {evt}")
                except:
                    lines.append(f"  {entry.get('event', '?')}: {entry.get('details', '')[:60]}")

        lines += ["", "ALL-TIME BEST VARIANTS", "-" * 48]
        for a in status.get("all_time_best_agents", []):
            lines.append(
                f"  Agent {a['agent_id']}: WR={a['win_rate']*100:.1f}%  "
                f"edge={a['avg_edge']:+.3f}  fitness={a['fitness']:.3f}  "
                f"n={a['trade_count']}  lineage={a['lineage_length']}"
            )

        lines.append("")
        return "\n".join(lines)

    def _log_learning_event(self, event_type: str, data: dict):
        """Log a learning event to a JSON file."""
        log_path = Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "learning_events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "data": data,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ---------------------------------------------------------------------------
    # Prompt retrieval (used by hedge_fund at runtime)
    # ---------------------------------------------------------------------------

    def get_prompt_for_agent(self, agent_id: int) -> str:
        """Get the current evolved prompt for an agent, or fall back to original."""
        prompts = self.evolver.get_current_prompts()
        return prompts.get(agent_id, "")

    def get_all_prompts(self) -> dict[int, str]:
        """Get all current agent prompts (evolved or original)."""
        return self.evolver.get_current_prompts()
