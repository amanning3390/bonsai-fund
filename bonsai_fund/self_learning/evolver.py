"""
Bonsai Fund — Self-Learning: Evolutionary Mutator

The LONG-TERM recursive learning loop. After 100+ trades, the system
evaluates whether the current swarm is still optimal and, if not, evolves
new system prompts and strategy weights.

How evolution works:

  GENERATION 0: Hand-crafted 7 agents (FastIntuit, DeepAnalyst, etc.)
       ↓ 100+ trades later
  GENERATION 1: Mutate system prompts based on which thinking styles
                 performed best. Clone top performers. Discard worst.

The mutator does NOT blindly rewrite prompts. It makes targeted mutations
guided by performance data:
  - If BayesianUpdater consistently outperforms on CPI → strengthen its
    Bayesian framing, weaken its system-1 tendencies
  - If Contrarian wins on Geopolitics but loses on Sports → restrict
    Contrarian to high-volume extreme markets only
  - If FinalVote vetoes too much → reduce veto threshold slightly

Mutations are STOCHASTIC but GUIDED — performance data shapes the mutation space.
"""

from __future__ import annotations
import json
import sqlite3
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bonsai_fund.agent import AGENT_NAMES, AGENT_SPECIALTIES, SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES


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
# Mutation operators — targeted prompt modifications
# ---------------------------------------------------------------------------

MUTATION_OPERATORS = {

    "strengthen_bayesian": [
        "Add a sentence about sequential evidence updating.",
        "Include a step about prior probability before updating.",
        "Add a sentence about likelihood ratios.",
        "Strengthen the instruction to always state prior before posterior.",
    ],

    "strengthen_contrarian": [
        "Add: 'High-volume consensus at extremes is usually wrong.'",
        "Add: 'When YES>80c with high volume, instinct is to fade the crowd.'",
        "Add: 'When volume is low, go with the crowd.'",
        "Strengthen: always ask 'why is the crowd wrong?' before agreeing.",
    ],

    "strengthen_macro": [
        "Add: 'Start by identifying the macro regime before analyzing the event.'",
        "Add: 'Cross-reference with BTC, rates, and credit spreads.'",
        "Add: 'If macro contradicts the market direction, bet the macro direction.'",
        "Strengthen: regime analysis comes before event analysis.",
    ],

    "weaken_veto": [
        "Reduce veto threshold: only veto when ALL 6 agents agree with conf<0.5.",
        "Allow passage if at least 2 agents have conf>0.7 regardless of others.",
        "Raise confidence threshold for veto from 0.6 to 0.5.",
    ],

    "strengthen_forensic": [
        "Add: 'Look for the outlier. The thing no one is pricing.'",
        "Add: 'Is this market truly binary or does it have gray zones?'",
        "Add: 'What hidden assumption is the market making?'",
        "Add: 'Is there a liquidity trap or structural flaw?'",
    ],

    "add_calibration_note": [
        "Important: calibrate your confidence against your historical accuracy.",
        "Reminder: if your win rate on similar trades is 55%, your confidence should reflect that.",
        "Note: avoid overconfidence. Your gut calibration matters.",
    ],

    "strengthen_probabilistic": [
        "Add: 'Always decompose into conditional probabilities before estimating.'",
        "Add: 'P(outcome) = P(C1) × P(C2|C1) × ... is your core method.'",
        "Add: 'State your probability tree explicitly before voting.'",
    ],

    "add_time_horizon_sensitivity": [
        "Add: For events <14 days out, be more aggressive. Short-term catalysts dominate.",
        "Add: For events >60 days out, reduce confidence — many things can change.",
        "Add: Time decay matters. Near-term events are more predictable.",
    ],
}


@dataclass
class EvolvedAgent:
    """A system prompt variant with a performance lineage."""
    agent_id: int
    generation: int
    mutation_type: str          # which mutation operator produced this
    system_prompt: str
    few_shot_example: str
    lineage: list[dict]         # [{generation, mutation, parent_hash, perf}]
    parent_hash: str
    hash: str

    # Performance (filled in after evaluation)
    trade_count: int = 0
    win_rate: float = 0.0
    avg_edge: float = 0.0
    sharpe_approx: float = 0.0
    fitness: float = 0.0       # composite score used for selection

    status: str = "active"      # active / champion / discarded

    def compute_fitness(self) -> float:
        """
        Composite fitness: balance win rate, edge magnitude, and sample size.
        Prefer agents with high win rate AND positive edge.
        Penalize low sample sizes.
        """
        sample_penalty = max(0, 1.0 - (self.trade_count / 50)) * 0.3  # up to -0.3 for n<50
        win_rate_bonus = self.win_rate - 0.5  # baseline at 50%
        edge_bonus = self.avg_edge * 2.0
        self.fitness = max(0, win_rate_bonus + edge_bonus - sample_penalty)
        return self.fitness


class EvolutionaryMutator:
    """
    Evolves agent system prompts over generations.

    The cycle:
    1. EVALUATE: After 100+ new trades, score all active prompt variants
    2. SELECT: Keep top performers (champions), discard bottom performers
    3. MUTATE: Create new variants from champions using guided mutation operators
    4. HERITAGE: New variants carry lineage — you can trace which mutations
                 produced which performance improvements

    Key insight: mutations are NOT random. They are guided by performance data.
    If BayesianUpdater outperforms DeepAnalyst on CPI markets, the mutator
    MUTATES BayesianUpdater's prompts to be even MORE Bayesian, and creates
    a variant of DeepAnalyst that borrows some Bayesian structure.
    """

    def __init__(self):
        self._conn = self._init_db()
        self._generation = 0
        self._champions: list[EvolvedAgent] = []
        self._load_state()

    def _db_path(self) -> Path:
        return Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "evolver.db"

    def _init_db(self) -> sqlite3.Connection:
        p = self._db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolved_agents (
                agent_id INTEGER,
                generation INTEGER,
                mutation_type TEXT,
                system_prompt TEXT,
                few_shot_example TEXT,
                lineage_json TEXT,
                parent_hash TEXT,
                hash TEXT,
                trade_count INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_edge REAL DEFAULT 0.0,
                sharpe_approx REAL DEFAULT 0.0,
                fitness REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                created_at TEXT,
                PRIMARY KEY (hash)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER,
                event TEXT,
                details TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()
        return conn

    def _load_state(self):
        """Load current generation and champions from DB."""
        cursor = self._conn.execute(
            "SELECT value FROM evolution_state WHERE key='generation'"
        )
        row = cursor.fetchone()
        self._generation = int(row[0]) if row else 0

        cursor = self._conn.execute(
            "SELECT * FROM evolved_agents WHERE status='champion'"
        )
        cols = [d[0] for d in cursor.description]
        for row in cursor.fetchall():
            r = dict(zip(cols, row))
            self._champions.append(self._row_to_ea(r))

    def _row_to_ea(self, r: dict) -> EvolvedAgent:
        return EvolvedAgent(
            agent_id=r["agent_id"],
            generation=r["generation"],
            mutation_type=r["mutation_type"],
            system_prompt=r["system_prompt"],
            few_shot_example=r["few_shot_example"],
            lineage=json.loads(r["lineage_json"] or "[]"),
            parent_hash=r["parent_hash"],
            hash=r["hash"],
            trade_count=r["trade_count"] or 0,
            win_rate=r["win_rate"] or 0.0,
            avg_edge=r["avg_edge"] or 0.0,
            sharpe_approx=r["sharpe_approx"] or 0.0,
            fitness=r["fitness"] or 0.0,
            status=r["status"],
        )

    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def _log(self, event: str, details: str = ""):
        self._conn.execute("""
            INSERT INTO evolution_log (generation, event, details, timestamp)
            VALUES (?, ?, ?, ?)
        """, (self._generation, event, details, datetime.now(timezone.utc).isoformat()))
        self._conn.commit()

    # ---------------------------------------------------------------------------
    # Initialization — seed Generation 0
    # ---------------------------------------------------------------------------

    def seed_generation_0(self) -> int:
        """
        Seed the database with Generation 0 agents (the hand-crafted originals).
        Call this once on first run. Returns number of agents seeded.
        """
        seeded = 0
        for aid in range(7):
            prompt = SYSTEM_PROMPTS[aid]
            few_shot = FEW_SHOT_EXAMPLES.get(aid, "")
            h = self._compute_hash(prompt + few_shot)

            self._conn.execute("""
                INSERT OR IGNORE INTO evolved_agents
                (agent_id, generation, mutation_type, system_prompt, few_shot_example,
                 lineage_json, parent_hash, hash, trade_count, win_rate, avg_edge,
                 fitness, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                aid, 0, "seed",
                prompt, few_shot, "[]", "none", h,
                0, 0.0, 0.0, 0.0, "champion",
                datetime.now(timezone.utc).isoformat(),
            ))
            seeded += 1

        self._conn.execute(
            "INSERT OR REPLACE INTO evolution_state (key, value) VALUES ('generation', '0')"
        )
        self._conn.commit()
        self._generation = 0
        self._log("SEEDED", f"Generation 0 seeded with {seeded} original agents")
        return seeded

    # ---------------------------------------------------------------------------
    # Performance evaluation
    # ---------------------------------------------------------------------------

    def update_performance(self, agent_id: int, prompt_hash: str,
                          trade_count: int, win_rate: float,
                          avg_edge: float, sharpe_approx: float):
        """Update performance stats for an evolved agent after new trades resolve."""
        fitness = 0.0
        if trade_count > 0:
            sample_penalty = max(0, 1.0 - (trade_count / 50)) * 0.3
            fitness = max(0, (win_rate - 0.5) + (avg_edge * 2.0) - sample_penalty)

        self._conn.execute("""
            UPDATE evolved_agents SET
                trade_count=?, win_rate=?, avg_edge=?,
                sharpe_approx=?, fitness=?
            WHERE hash=?
        """, (trade_count, win_rate, avg_edge, sharpe_approx, fitness, prompt_hash))
        self._conn.commit()

    def evaluate_generation(self, agent_scores: list[dict]) -> dict:
        """
        Evaluate all active agents after a sufficient sample of trades.
        Returns dict: {agent_id: {old_hash, new_hash, status, mutation_type}}
        """
        if len(agent_scores) < 10:
            return {"status": "insufficient_data", "message": f"{len(agent_scores)} trades — need 10+"}

        results = {}
        for score in agent_scores:
            aid = score["agent_id"]
            cursor = self._conn.execute(
                "SELECT hash, fitness, win_rate, avg_edge, trade_count "
                "FROM evolved_agents WHERE agent_id=? AND status='active' ORDER BY generation DESC"
                , (aid,))
            rows = cursor.fetchall()
            if not rows:
                continue

            ea = self._row_to_ea(dict(zip(["hash","fitness","win_rate","avg_edge","trade_count"], rows[0])))
            ea.trade_count = score.get("total_calls", ea.trade_count)
            ea.win_rate = score.get("win_rate", ea.win_rate)
            ea.avg_edge = score.get("avg_edge", ea.avg_edge)
            ea.fitness = ea.compute_fitness()
            results[aid] = ea

        return results

    # ---------------------------------------------------------------------------
    # Selection + mutation
    # ---------------------------------------------------------------------------

    def select_and_mutate(self, evaluated: dict[str, EvolvedAgent],
                          target_mutations_per_agent: int = 2) -> list[EvolvedAgent]:
        """
        Select champions and create mutated variants for the next generation.

        Strategy:
        - Top 2 agents by fitness → KEEP as champions (carry to next gen)
        - Top agent → CLONE with 1-2 mutations (guided by its strengths)
        - Bottom agent → REJECT, replace with mutated variant of top performer
        - Agents with <50 trades → NO mutations yet (insufficient sample)

        Mutations are GUIDED:
        - If agent has high win_rate in Bayesian tasks → apply strengthen_bayesian
        - If agent has high edge in Contrarian tasks → apply strengthen_contrarian
        """
        if not evaluated:
            return []

        sorted_agents = sorted(evaluated.values(), key=lambda x: x.fitness, reverse=True)
        champions = sorted_agents[:2]  # top 2
        losers = sorted_agents[-2:] if len(sorted_agents) >= 4 else []
        self._champions = champions

        new_variants = []

        # Log selection
        self._log("SELECTION", json.dumps({
            "generation": self._generation,
            "champions": [(a.agent_id, a.fitness, a.win_rate) for a in champions],
            "losers": [(a.agent_id, a.fitness) for a in losers],
        }))

        # Create mutated variants from champions
        for champ in champions:
            for i in range(target_mutations_per_agent):
                variant = self._mutate(champ)
                if variant:
                    new_variants.append(variant)

        # Create variants of the TOP performer only
        top = champions[0] if champions else None
        if top and target_mutations_per_agent > len(new_variants):
            extra = self._mutate(top)
            if extra:
                new_variants.append(extra)

        return new_variants

    def _mutate(self, parent: EvolvedAgent) -> Optional[EvolvedAgent]:
        """Create a single mutated variant from a parent agent."""

        # Determine which mutation operator to use, guided by performance
        mutation_type = self._select_mutation_type(parent)

        # Get mutation options for this agent type
        agent_type = AGENT_SPECIALTIES[parent.agent_id]
        ops_key = self._mutation_for_specialty(agent_type, parent)

        ops = MUTATION_OPERATORS.get(ops_key, MUTATION_OPERATORS["add_calibration_note"])
        mutation_text = random.choice(ops)

        # Apply mutation
        new_prompt = parent.system_prompt
        if random.random() < 0.5:
            # Append to end of system prompt
            new_prompt = new_prompt.rstrip() + "\n" + mutation_text + "\n"
        else:
            # Insert after the first sentence of the method section
            lines = new_prompt.split("\n")
            for i, line in enumerate(lines):
                if "Your method" in line or "Your technique" in line or "Method:" in line:
                    lines.insert(i + 1, mutation_text)
                    break
            new_prompt = "\n".join(lines)

        new_few_shot = parent.few_shot_example
        # Optionally mutate the few-shot example too
        if random.random() < 0.3 and new_few_shot:
            new_few_shot = self._mutate_few_shot(new_few_shot, ops_key)

        new_hash = self._compute_hash(new_prompt + new_few_shot)

        # Check we haven't already created this variant
        cursor = self._conn.execute(
            "SELECT hash FROM evolved_agents WHERE hash=?", (new_hash,)
        )
        if cursor.fetchone():
            return None  # already exists

        lineage = list(parent.lineage) + [{
            "generation": parent.generation,
            "mutation_type": mutation_type,
            "parent_hash": parent.hash,
            "perf": parent.fitness,
        }]

        ea = EvolvedAgent(
            agent_id=parent.agent_id,
            generation=self._generation + 1,
            mutation_type=mutation_type,
            system_prompt=new_prompt,
            few_shot_example=new_few_shot,
            lineage=lineage,
            parent_hash=parent.hash,
            hash=new_hash,
            status="active",
        )

        # Save to DB
        self._conn.execute("""
            INSERT INTO evolved_agents
            (agent_id, generation, mutation_type, system_prompt, few_shot_example,
             lineage_json, parent_hash, hash, trade_count, win_rate, avg_edge,
             fitness, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ea.agent_id, ea.generation, ea.mutation_type, ea.system_prompt,
            ea.few_shot_example, json.dumps(ea.lineage), ea.parent_hash, ea.hash,
            0, 0.0, 0.0, 0.0, "active",
            datetime.now(timezone.utc).isoformat(),
        ))

        # Mark old champion as "heritage" (kept but not active in new gen)
        self._conn.execute(
            "UPDATE evolved_agents SET status='heritage' WHERE hash=?",
            (parent.hash,)
        )
        self._conn.commit()

        self._log("MUTATION", json.dumps({
            "parent_hash": parent.hash,
            "new_hash": new_hash,
            "mutation_type": mutation_type,
            "agent_id": parent.agent_id,
        }))

        return ea

    def _select_mutation_type(self, parent: EvolvedAgent) -> str:
        """Select mutation operator guided by parent performance."""
        # If agent has high win rate → strengthen what it's already good at
        if parent.win_rate > 0.58:
            specialty = AGENT_SPECIALTIES[parent.agent_id]
            mutation_map = {
                "system1_gut": "add_time_horizon_sensitivity",
                "system2_probabilistic": "strengthen_probabilistic",
                "bayesian_sequential": "strengthen_bayesian",
                "contrarian_crowd": "strengthen_contrarian",
                "macro_topdown": "strengthen_macro",
                "forensic_anomaly": "strengthen_forensic",
                "synthesis_veto": "weaken_veto",
            }
            return mutation_map.get(specialty, "add_calibration_note")

        # If agent has poor calibration → add calibration note
        if parent.trade_count >= 20 and parent.win_rate < 0.48:
            return "add_calibration_note"

        # Otherwise random
        return random.choice(list(MUTATION_OPERATORS.keys()))

    def _mutation_for_specialty(self, specialty: str, parent: EvolvedAgent) -> str:
        """Map agent specialty to the best mutation operator."""
        mutation_map = {
            "system1_gut": "strengthen_probabilistic",    # add structure to gut
            "system2_probabilistic": "strengthen_probabilistic",
            "bayesian_sequential": "strengthen_bayesian",
            "contrarian_crowd": "strengthen_contrarian",
            "macro_topdown": "strengthen_macro",
            "forensic_anomaly": "strengthen_forensic",
            "synthesis_veto": "weaken_veto",
        }
        return mutation_map.get(specialty, "add_calibration_note")

    def _mutate_few_shot(self, few_shot: str, ops_key: str) -> str:
        """Optionally mutate a few-shot example."""
        lines = few_shot.strip().split("\n")
        if len(lines) <= 2:
            return few_shot

        # Swap order of examples, or add a twist
        if random.random() < 0.5 and len(lines) >= 4:
            # Swap first two example blocks
            lines = lines[2:] + lines[:2]

        return "\n".join(lines)

    # ---------------------------------------------------------------------------
    # Evolution cycle driver
    # ---------------------------------------------------------------------------

    def should_evolve(self, total_trades: int) -> bool:
        """
        Trigger evolution when:
        - We have 100+ new trades since last evolution
        - AND at least 10 trades per agent in the current generation
        """
        cursor = self._conn.execute(
            "SELECT value FROM evolution_state WHERE key='last_evolution_trades'"
        )
        row = cursor.fetchone()
        last_trades = int(row[0]) if row else 0

        cursor = self._conn.execute(
            "SELECT SUM(trade_count) FROM evolved_agents WHERE status IN ('active','champion')"
        )
        row = cursor.fetchone()
        total_active_trades = row[0] or 0

        return (total_trades - last_trades >= 100) and (total_active_trades >= 50)

    def run_evolution_cycle(self, agent_scores: list[dict]) -> dict:
        """
        Full evolution cycle:
        1. Evaluate current generation
        2. Select champions + discard losers
        3. Create mutated variants
        4. Log results
        """
        self._generation += 1

        evaluated = self.evaluate_generation(agent_scores)
        if not evaluated or len(evaluated) < 2:
            self._generation -= 1
            return {"status": "skipped", "reason": "insufficient agents evaluated"}

        new_variants = self.select_and_mutate(evaluated, target_mutations_per_agent=2)

        self._conn.execute(
            "INSERT OR REPLACE INTO evolution_state (key, value) VALUES ('generation', ?)",
            (str(self._generation),)
        )

        cursor = self._conn.execute(
            "SELECT SUM(trade_count) FROM evolved_agents WHERE resolution='PENDING' OR 1=1"
        )

        self._log("EVOLUTION_CYCLE", json.dumps({
            "generation": self._generation,
            "variants_created": len(new_variants),
            "champions": [(a.agent_id, a.hash) for a in self._champions],
        }))

        return {
            "status": "success",
            "generation": self._generation,
            "variants_created": len(new_variants),
            "champions": [{"id": a.agent_id, "hash": a.hash, "fitness": a.fitness}
                          for a in self._champions],
            "new_variants": [{"id": v.agent_id, "hash": v.hash, "mutation": v.mutation_type}
                             for v in new_variants],
        }

    def get_current_prompts(self) -> dict[int, str]:
        """Get the current system prompts for all 7 agents."""
        prompts = {}
        for aid in range(7):
            cursor = self._conn.execute(
                "SELECT system_prompt FROM evolved_agents "
                "WHERE agent_id=? AND status IN ('active','champion','heritage') "
                "ORDER BY generation DESC LIMIT 1",
                (aid,)
            )
            row = cursor.fetchone()
            if row:
                prompts[aid] = row[0]
            else:
                prompts[aid] = SYSTEM_PROMPTS.get(aid, "")
        return prompts

    def get_evolution_log(self, limit: int = 20) -> list[dict]:
        """Return recent evolution log entries."""
        cursor = self._conn.execute(
            "SELECT * FROM evolution_log ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in cursor.fetchall()]

    def get_all_time_best(self) -> list[EvolvedAgent]:
        """Return all-time best performing agent variants."""
        cursor = self._conn.execute(
            "SELECT * FROM evolved_agents WHERE trade_count >= 10 "
            "ORDER BY fitness DESC LIMIT 10"
        )
        cols = [d[0] for d in cursor.description]
        return [self._row_to_ea(dict(zip(cols, r))) for r in cursor.fetchall()]
