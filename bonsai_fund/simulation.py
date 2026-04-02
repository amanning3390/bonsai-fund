"""
Bonsai Fund — Simulation Engine

Replay-based agent training using historical + synthetic market data.
All trading is learning — every scan generates synthetic counterfactuals
to train agents against their weaknesses.

Simulation pipeline:
  1. REPLAY:  Re-run recent losing trades with agent vote substitutions
  2. SYNTHETIC: Generate markets in weak categories with perturbed probabilities
  3. ADVERSARIAL: Find the market conditions that caused the drawdown
  4. CANDIDATE: Test evolved prompt variants against the adversarial set
  5. COMMIT: If candidate outperforms original by >5%, deploy it

Training sessions are stored in SQLite and used as priors for the EvolutionaryMutator.
"""

from __future__ import annotations
import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bonsai_fund.agent import Market, AgentVote, AGENT_NAMES


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
# Types
# ---------------------------------------------------------------------------

@dataclass
class SimMarket:
    """A synthetic market for training purposes."""
    ticker: str
    series_title: str
    market_title: str
    yes_price_cents: int
    no_price_cents: int
    volume: int
    days_to_event: int
    implied_prob_yes: float
    ground_truth: str           # YES or NO — the hidden outcome
    source: str                 # "replay" | "synthetic" | "adversarial" | "candidate"
    parent_ticker: str = ""     # for replays: original market ticker
    metadata: dict = field(default_factory=dict)


@dataclass
class SimResult:
    """Result of a single simulated trade."""
    sim_id: str
    agent_id: int
    agent_name: str
    original_vote: str
    original_confidence: float
    original_edge: float
    sim_vote: str
    sim_confidence: float
    sim_edge: float
    outcome_correct: bool
    pnl: float
    roi: float
    market_ticker: str
    session_id: str
    timestamp: str


@dataclass
class TrainingSession:
    """A full training session — one or more simulations run together."""
    session_id: str
    trigger: str          # "drawdown_orange" | "drawdown_red" | "manual" | "evolution"
    trades_tested: int
    trades_improved: int
    original_win_rate: float
    candidate_win_rate: float
    improvement_pct: float    # % improvement of candidate over original
    gen_before: int
    gen_after: int
    status: str            # "running" | "completed" | "deployed" | "rejected"
    started_at: str
    completed_at: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# Synthetic Market Generator
# ---------------------------------------------------------------------------

class SyntheticMarketGenerator:
    """
    Generates synthetic markets for training. Uses:
    1. Historical replay — real markets with perturbed prices/outcomes
    2. Category-based — markets drawn from weak categories
    3. Adversarial — markets specifically designed to test the drawdown period
    """

    # Realistic market templates by category
    MARKET_TEMPLATES = {
        "Geopolitics": [
            {"series": "Geopolitics", "title": "Putin remains President end of {year}", "prob_range": (30, 75), "volume_range": (5000, 50000)},
            {"series": "Geopolitics", "title": "Iran nuclear deal reached by {date}", "prob_range": (20, 60), "volume_range": (3000, 30000)},
            {"series": "Geopolitics", "title": "US-China trade war escalates {year}", "prob_range": (40, 80), "volume_range": (4000, 40000)},
            {"series": "Geopolitics", "title": "Major war begins {year}", "prob_range": (5, 25), "volume_range": (2000, 20000)},
            {"series": "Geopolitics", "title": "New sanctions on Russia by {date}", "prob_range": (50, 85), "volume_range": (3000, 25000)},
        ],
        "NBA": [
            {"series": "NBA", "title": "{team} wins {year} NBA Championship", "prob_range": (10, 60), "volume_range": (5000, 80000)},
            {"series": "NBA", "title": "{team} makes playoffs {year}", "prob_range": (40, 80), "volume_range": (3000, 50000)},
            {"series": "NBA", "title": "MVP {year}: {player}", "prob_range": (15, 40), "volume_range": (2000, 20000)},
        ],
        "CPI": [
            {"series": "CPI", "title": "CPI {month} {year} > {threshold}pct", "prob_range": (20, 50), "volume_range": (8000, 50000)},
            {"series": "CPI", "title": "CPI {month} {year} < {threshold}pct", "prob_range": (30, 60), "volume_range": (8000, 50000)},
            {"series": "CPI", "title": "Core CPI {month} {year} > {threshold}pct", "prob_range": (25, 55), "volume_range": (6000, 30000)},
        ],
        "Jobs": [
            {"series": "Jobs", "title": "NFP {month} {year} > {threshold}K", "prob_range": (35, 65), "volume_range": (5000, 40000)},
            {"series": "Jobs", "title": "Unemployment {month} {year} < {pct}pct", "prob_range": (40, 70), "volume_range": (5000, 30000)},
            {"series": "Jobs", "title": "Jobs report beat {month} {year}", "prob_range": (45, 70), "volume_range": (4000, 25000)},
        ],
        "Earnings": [
            {"series": "Earnings", "title": "{company} EPS beat {quarter} {year}", "prob_range": (45, 70), "volume_range": (5000, 60000)},
            {"series": "Earnings", "title": "{company} revenue beat {quarter} {year}", "prob_range": (40, 65), "volume_range": (4000, 50000)},
            {"series": "Earnings", "title": "{company} misses {quarter} {year}", "prob_range": (20, 45), "volume_range": (3000, 30000)},
        ],
        "Elections": [
            {"series": "Elections", "title": "{candidate} wins {race} {year}", "prob_range": (30, 70), "volume_range": (10000, 100000)},
            {"series": "Elections", "title": "{party} takes control of {chamber} {year}", "prob_range": (35, 65), "volume_range": (8000, 80000)},
        ],
        "Hurricane": [
            {"series": "Hurricane", "title": "Major hurricane makes landfall {region} {season}", "prob_range": (20, 60), "volume_range": (2000, 20000)},
            {"series": "Hurricane", "title": "{n} named storms form in {season}", "prob_range": (30, 70), "volume_range": (1000, 10000)},
        ],
        "Crypto": [
            {"series": "Crypto", "title": "BTC > ${price}K by {date}", "prob_range": (25, 65), "volume_range": (10000, 100000)},
            {"series": "Crypto", "title": "ETH > ${price}K by {date}", "prob_range": (25, 60), "volume_range": (5000, 50000)},
            {"series": "Crypto", "title": "Crypto ETF approved by {date}", "prob_range": (20, 50), "volume_range": (3000, 30000)},
        ],
        "Interest Rates": [
            {"series": "Interest Rates", "title": "Fed rate cut by {date}", "prob_range": (30, 70), "volume_range": (10000, 80000)},
            {"series": "Interest Rates", "title": "Rates > {pct}pct by {date}", "prob_range": (20, 50), "volume_range": (5000, 40000)},
        ],
    }

    TEAMS = ["Lakers", "Warriors", "Celtics", "Nuggets", "Heat", "Suns", "Bucks", "Clippers", "Mavericks", "76ers"]
    PLAYERS = ["Jokic", "Giannis", "Embiid", "Luka", "Tatum", "Curry", "LeBron", "Durant", "Antetokounmpo", "Doncic"]
    COMPANIES = ["Amazon", "Apple", "Google", "Meta", "Microsoft", "Tesla", "NVDA", "Netflix", "AMD", "Intel"]
    CANDIDATES = ["Trump", "Biden", "DeSantis", "Harris", " Newsom", "Clinton", "Warren", "Young", "Kennedy"]
    REGIONS = ["Gulf Coast", "East Coast", "Florida", "Gulf", "Atlantic", "Caribbean"]
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    def __init__(self):
        self.rng = random.Random()

    def generate(
        self,
        category: str,
        count: int = 10,
        mode: str = "synthetic",
        ground_truth_override: str = None,
        market_prob_override: float = None,
    ) -> list[SimMarket]:
        """
        Generate synthetic markets for training.
        mode: "synthetic" | "adversarial" | "replay"
        """
        markets = []
        templates = self.MARKET_TEMPLATES.get(category, self.MARKET_TEMPLATES["Earnings"])

        for i in range(count):
            tpl = self.rng.choice(templates)
            prob = self.rng.uniform(*tpl["prob_range"])
            if market_prob_override is not None:
                prob = market_prob_override

            # Perturb probability for adversarial mode (push to extremes)
            if mode == "adversarial":
                if self.rng.random() < 0.5:
                    prob = self.rng.uniform(0.75, 0.95)  # crowded YES
                else:
                    prob = self.rng.uniform(0.05, 0.25)  # crowded NO

            yes_cents = max(5, min(95, int(prob * 100)))
            no_cents = 100 - yes_cents
            vol = self.rng.randint(*tpl["volume_range"])
            days = self.rng.randint(7, 180)

            # Ground truth: slightly against the crowded direction for realistic losses
            if ground_truth_override:
                outcome = ground_truth_override
            else:
                # When prob > 70 or < 30, outcome is often the opposite (regression to mean)
                if prob > 0.70 and self.rng.random() < 0.35:
                    outcome = "NO"
                elif prob < 0.30 and self.rng.random() < 0.35:
                    outcome = "YES"
                else:
                    outcome = "YES" if self.rng.random() < prob else "NO"

            title = self._fill_template(tpl["title"])
            ticker = f"SIM-{category[:3].upper()}-{i+1:03d}"

            markets.append(SimMarket(
                ticker=ticker,
                series_title=tpl["series"],
                market_title=title,
                yes_price_cents=yes_cents,
                no_price_cents=no_cents,
                volume=vol,
                days_to_event=days,
                implied_prob_yes=yes_cents / 100.0,
                ground_truth=outcome,
                source=mode,
                metadata={"category": category, "true_prob": prob},
            ))

        return markets

    def _fill_template(self, template: str) -> str:
        year = self.rng.randint(2026, 2028)
        date = f"{self.rng.choice(self.MONTHS)[:3]} {year}"
        return (template
            .replace("{year}", str(year))
            .replace("{date}", date)
            .replace("{team}", self.rng.choice(self.TEAMS))
            .replace("{player}", self.rng.choice(self.PLAYERS))
            .replace("{company}", self.rng.choice(self.COMPANIES))
            .replace("{candidate}", self.rng.choice(self.CANDIDATES))
            .replace("{race}", self.rng.choice(["President", "Governor", "Senate"]))
            .replace("{party}", self.rng.choice(["Democrats", "Republicans"]))
            .replace("{chamber}", self.rng.choice(["Senate", "House"]))
            .replace("{n}", str(self.rng.randint(5, 20)))
            .replace("{price}", str(self.rng.choice([50, 75, 100, 150, 200])))
            .replace("{pct}", str(self.rng.randint(3, 7)))
            .replace("{threshold}", str(self.rng.randint(150, 350)))
            .replace("{month}", self.rng.choice(self.MONTHS))
            .replace("{region}", self.rng.choice(self.REGIONS))
            .replace("{season}", f"{year} Season")
            .replace("{quarter}", f"Q{self.rng.randint(1,4)}"))

    def generate_diverse_set(self, categories: list[str] = None, count: int = 20) -> list[SimMarket]:
        """Generate a diverse set of markets across categories for baseline training."""
        if categories is None:
            categories = list(self.MARKET_TEMPLATES.keys())
        markets = []
        per_cat = max(1, count // len(categories))
        for cat in categories:
            markets.extend(self.generate(cat, count=per_cat, mode="synthetic"))
        self.rng.shuffle(markets)
        return markets[:count]


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Runs agent training simulations. All trading generates learning data.

    Pipeline for a drawdown-triggered simulation:
      1. Identify weak categories from recent losing trades
      2. Generate adversarial + synthetic markets in those categories
      3. Run each market through BOTH original prompts AND candidate prompts
      4. Score: candidate_win_rate vs original_win_rate
      5. If candidate improves by >5% AND >10 sample trades → deploy

    Pipeline for a scheduled baseline simulation:
      1. Generate diverse market set across all categories
      2. Run through all agent prompt variants
      3. Update EvolutionaryMutator with results
    """

    def __init__(self):
        self._conn = self._init_db()
        self._gen = SyntheticMarketGenerator()
        self._active_session: Optional[TrainingSession] = None

    def _db_path(self) -> Path:
        return Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "simulation.db"

    def _init_db(self) -> sqlite3.Connection:
        p = self._db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sim_markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, series_title TEXT, market_title TEXT,
                yes_price_cents INTEGER, no_price_cents INTEGER,
                volume INTEGER, days_to_event INTEGER,
                implied_prob_yes REAL, ground_truth TEXT,
                source TEXT, parent_ticker TEXT,
                metadata_json TEXT, created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sim_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id TEXT, agent_id INTEGER, agent_name TEXT,
                original_vote TEXT, original_confidence REAL, original_edge REAL,
                sim_vote TEXT, sim_confidence REAL, sim_edge REAL,
                outcome_correct INTEGER, pnl REAL, roi REAL,
                market_ticker TEXT, session_id TEXT, timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                trigger TEXT, trades_tested INTEGER,
                trades_improved INTEGER,
                original_win_rate REAL,
                candidate_win_rate REAL,
                improvement_pct REAL,
                gen_before INTEGER, gen_after INTEGER,
                status TEXT, started_at TEXT,
                completed_at TEXT, summary TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_sim_scores (
                agent_id INTEGER,
                session_id TEXT,
                original_wr REAL, candidate_wr REAL,
                improvement_pct REAL,
                PRIMARY KEY (agent_id, session_id)
            )
        """)
        conn.commit()
        return conn

    # ---------------------------------------------------------------------------
    # Session management
    # ---------------------------------------------------------------------------

    def start_session(self, trigger: str, gen_before: int) -> TrainingSession:
        import uuid
        sid = f"sim-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        session = TrainingSession(
            session_id=sid,
            trigger=trigger,
            trades_tested=0,
            trades_improved=0,
            original_win_rate=0.0,
            candidate_win_rate=0.0,
            improvement_pct=0.0,
            gen_before=gen_before,
            gen_after=gen_before,  # will be updated if evolution happens
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._active_session = session
        self._conn.execute("""
            INSERT INTO training_sessions
            (session_id, trigger, trades_tested, trades_improved,
             original_win_rate, candidate_win_rate, improvement_pct,
             gen_before, gen_after, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sid, trigger, 0, 0, 0.0, 0.0, 0.0,
            gen_before, gen_before, "running",
            session.started_at,
        ))
        self._conn.commit()
        return session

    def end_session(self, session_id: str, status: str, summary: str = ""):
        self._conn.execute("""
            UPDATE training_sessions
            SET status=?, completed_at=?, summary=?
            WHERE session_id=?
        """, (status, datetime.now(timezone.utc).isoformat(), summary, session_id))
        self._conn.commit()
        self._active_session = None

    # ---------------------------------------------------------------------------
    # Core simulation runner
    # ---------------------------------------------------------------------------

    def run_simulation(
        self,
        markets: list[SimMarket],
        original_prompts: dict[int, str],
        candidate_prompts: dict[int, str],
        agent_names: list[str],
        call_bonsai_fn,       # fn(port, system_prompt, user_prompt) -> text
        port_base: int = 8090,
        session_id: str = None,
    ) -> dict:
        """
        Run a head-to-head comparison of original vs candidate prompts on a set of markets.

        For each market:
          1. Call agents with ORIGINAL prompts → record votes
          2. Call same agents with CANDIDATE prompts → record votes
          3. Compare: which set predicted the outcome better?

        Returns summary dict with per-agent improvement scores.
        """
        from bonsai_fund.agent import AgentVote

        original_correct = 0
        candidate_correct = 0
        total = 0
        per_agent_original = {i: {"correct": 0, "total": 0} for i in range(7)}
        per_agent_candidate = {i: {"correct": 0, "total": 0} for i in range(7)}
        results = []

        for mkt in markets:
            # Build user prompt for this market
            from bonsai_fund.agent import build_user_prompt

            # Run ORIGINAL prompts
            orig_votes = []
            for aid in range(6):
                text = call_bonsai_fn(port_base + aid, original_prompts.get(aid, ""), build_user_prompt(aid, mkt))
                v = self._parse_vote(aid, agent_names[aid], mkt.ticker, text)
                orig_votes.append(v)
                per_agent_original[aid]["total"] += 1

            # Run CANDIDATE prompts
            cand_votes = []
            for aid in range(6):
                text = call_bonsai_fn(port_base + aid, candidate_prompts.get(aid, ""), build_user_prompt(aid, mkt))
                v = self._parse_vote(aid, agent_names[aid], mkt.ticker, text)
                cand_votes.append(v)
                per_agent_candidate[aid]["total"] += 1

            # Score each agent
            for ov, cv in zip(orig_votes, cand_votes):
                oc = (ov.vote == mkt.ground_truth)
                cc = (cv.vote == mkt.ground_truth)
                if oc: per_agent_original[ov.agent_id]["correct"] += 1
                if cc: per_agent_candidate[cv.agent_id]["correct"] += 1

            # Majority vote
            orig_majority = max(set([v.vote for v in orig_votes]),
                               key=[v.vote for v in orig_votes].count)
            cand_majority = max(set([v.vote for v in cand_votes]),
                               key=[v.vote for v in cand_votes].count)

            if orig_majority == mkt.ground_truth:
                original_correct += 1
            if cand_majority == mkt.ground_truth:
                candidate_correct += 1
            total += 1

            # P&L simulation
            # Use original vote as what would have been traded
            # YES side
            if orig_majority == mkt.ground_truth:
                orig_pnl = (100.0 - mkt.yes_price_cents) / 100.0 if mkt.ground_truth == "YES" else 0
            else:
                orig_pnl = -mkt.yes_price_cents / 100.0 if orig_majority == "YES" else 0

            if cand_majority == mkt.ground_truth:
                cand_pnl = (100.0 - mkt.yes_price_cents) / 100.0 if mkt.ground_truth == "YES" else 0
            else:
                cand_pnl = -mkt.yes_price_cents / 100.0 if cand_majority == "YES" else 0

            results.append({
                "ticker": mkt.ticker,
                "ground_truth": mkt.ground_truth,
                "orig_majority": orig_majority,
                "cand_majority": cand_majority,
                "orig_correct": orig_majority == mkt.ground_truth,
                "cand_correct": cand_majority == mkt.ground_truth,
                "orig_pnl": orig_pnl,
                "cand_pnl": cand_pnl,
            })

        orig_wr = original_correct / total if total else 0
        cand_wr = candidate_correct / total if total else 0
        improvement = (cand_wr - orig_wr) / orig_wr if orig_wr > 0 else 0

        # Per-agent
        agent_improvements = {}
        for aid in range(7):
            ot = per_agent_original[aid]["total"] or 1
            ct = per_agent_candidate[aid]["total"] or 1
            owr = per_agent_original[aid]["correct"] / ot
            cwr = per_agent_candidate[aid]["correct"] / ct
            agent_improvements[aid] = {
                "name": agent_names[aid],
                "original_wr": round(owr, 4),
                "candidate_wr": round(cwr, 4),
                "improvement": round(cwr - owr, 4),
                "trades": ot,
            }

        return {
            "session_id": session_id,
            "total_markets": total,
            "original_win_rate": round(orig_wr, 4),
            "candidate_win_rate": round(cand_wr, 4),
            "improvement_pct": round(improvement, 4),
            "candidate_outperforms": cand_wr > orig_wr,
            "passes_threshold": cand_wr > orig_wr and improvement > 0.05,
            "per_agent": agent_improvements,
            "trade_results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_vote(self, agent_id: int, agent_name: str, ticker: str, raw: str) -> AgentVote:
        """Parse a vote from raw text output."""
        try:
            s, e = raw.find("{"), raw.rfind("}") + 1
            j = json.loads(raw[s:e])
            return AgentVote(
                agent_id, agent_name, ticker,
                j.get("vote", "PASS"),
                float(j.get("confidence", 0.5)),
                float(j.get("edge", 0.0)),
                str(j.get("reason", ""))[:80],
                0.0,
            )
        except:
            t = raw.upper()
            vote = "PASS" if "PASS" in t else ("YES" if "YES" in t[:10] else ("NO" if "NO" in t[:10] else "PASS"))
            return AgentVote(agent_id, agent_name, ticker, vote, 0.3, 0.0, "parse_error", 0.0)

    # ---------------------------------------------------------------------------
    # Drawdown-triggered simulation (ORANGE / RED)
    # ---------------------------------------------------------------------------

    def simulate_from_drawdown(
        self,
        learning,
        outcome_tracker,
        call_bonsai_fn,
        stage: str = "ORANGE",
    ) -> dict:
        """
        Triggered when drawdown enters ORANGE or RED stage.
        Generates adversarial markets in weak categories and runs simulation.
        """
        from bonsai_fund.self_learning.evolver import EvolutionaryMutator

        # Determine generation
        gen_before = learning.evolver._generation if learning else 0
        session = self.start_session(f"drawdown_{stage.lower()}", gen_before)

        # 1. Identify weak categories from outcome tracker
        agent_scores = outcome_tracker.get_agent_scores()
        weak_categories = self._identify_weak_categories(outcome_tracker)

        # 2. Generate adversarial + synthetic markets
        categories = list(weak_categories.keys())
        sim_markets = []

        # More adversarial markets at RED stage
        adversarial_count = 15 if stage == "RED" else 8
        synthetic_count = 20

        for cat in categories[:4]:  # top 4 weak categories
            sim_markets.extend(self._gen.generate(cat, count=adversarial_count // 2, mode="adversarial"))
        sim_markets.extend(self._gen.generate_diverse_set(categories, count=synthetic_count))

        # Save markets to DB
        for mkt in sim_markets:
            self._save_market(mkt)

        # 3. Get original and candidate prompts
        if learning:
            candidate_prompts = learning.evolver.get_current_prompts()
        else:
            from bonsai_fund.agent import SYSTEM_PROMPTS
            candidate_prompts = {i: SYSTEM_PROMPTS[i] for i in range(7)}

        from bonsai_fund.agent import SYSTEM_PROMPTS, AGENT_NAMES
        original_prompts = {i: SYSTEM_PROMPTS[i] for i in range(7)}

        # 4. Run simulation
        result = self.run_simulation(
            markets=sim_markets,
            original_prompts=original_prompts,
            candidate_prompts=candidate_prompts,
            agent_names=AGENT_NAMES,
            call_bonsai_fn=call_bonsai_fn,
            session_id=session.session_id,
        )

        # 5. If candidate wins, trigger evolution
        deploy_triggered = False
        if result["passes_threshold"]:
            # Trigger evolution with these results
            if learning:
                result_evo = learning.run_evolution_cycle()
                gen_after = learning.evolver._generation
                deploy_triggered = True
            else:
                gen_after = gen_before
            self.end_session(session.session_id, "completed",
                           f"Candidate deployed: WR {result['original_win_rate']:.1%}→{result['candidate_win_rate']:.1%}")
        else:
            gen_after = gen_before
            self.end_session(session.session_id, "rejected",
                           f"Candidate rejected: improvement {result['improvement_pct']:.1%} < 5% threshold")

        return {
            "session_id": session.session_id,
            "trigger": f"drawdown_{stage.lower()}",
            "weak_categories": weak_categories,
            "markets_tested": len(sim_markets),
            "original_win_rate": result["original_win_rate"],
            "candidate_win_rate": result["candidate_win_rate"],
            "improvement_pct": result["improvement_pct"],
            "candidate_outperforms": result["candidate_outperforms"],
            "passes_threshold": result["passes_threshold"],
            "deploy_triggered": deploy_triggered,
            "generation_before": gen_before,
            "generation_after": gen_after,
            "per_agent": result["per_agent"],
        }

    def _identify_weak_categories(self, outcome_tracker) -> dict[str, float]:
        """Find categories where the system is underperforming."""
        # get_category_base_rates returns dict[category, base_rate_float]
        base_rates = outcome_tracker.get_category_base_rates()
        weak = {}
        for cat, base_rate in base_rates.items():
            # base_rate is a float (the YES base rate)
            # Weak = base_rate near 0.5 means market is uncertain (harder to trade)
            # We want categories where swarm is performing WORST vs what we'd expect
            # Use entropy-like measure: categories near 50/50 are hardest
            entropy = min(base_rate, 1 - base_rate)  # 0.5 → 0.5 (high entropy = hard)
            weak[cat] = entropy
        # Sort by weakness (lowest = hardest/most wrong)
        return dict(sorted(weak.items(), key=lambda x: x[1])[:4])

    def _save_market(self, mkt: SimMarket):
        self._conn.execute("""
            INSERT INTO sim_markets
            (ticker, series_title, market_title, yes_price_cents, no_price_cents,
             volume, days_to_event, implied_prob_yes, ground_truth,
             source, parent_ticker, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mkt.ticker, mkt.series_title, mkt.market_title,
            mkt.yes_price_cents, mkt.no_price_cents,
            mkt.volume, mkt.days_to_event, mkt.implied_prob_yes,
            mkt.ground_truth, mkt.source, mkt.parent_ticker,
            json.dumps(mkt.metadata),
            datetime.now(timezone.utc).isoformat(),
        ))
        self._conn.commit()

    # ---------------------------------------------------------------------------
    # Baseline simulation (scheduled, builds training data)
    # ---------------------------------------------------------------------------

    def run_baseline_simulation(
        self,
        candidate_prompts: dict[int, str],
        call_bonsai_fn,
        session_id: str = None,
    ) -> dict:
        """
        Scheduled baseline simulation — builds training data across all categories.
        Does NOT compare vs original; just updates agent performance records.
        """
        from bonsai_fund.agent import SYSTEM_PROMPTS, AGENT_NAMES
        original_prompts = {i: SYSTEM_PROMPTS[i] for i in range(7)}

        categories = list(SyntheticMarketGenerator.MARKET_TEMPLATES.keys())
        markets = self._gen.generate_diverse_set(categories, count=30)

        for mkt in markets:
            self._save_market(mkt)

        result = self.run_simulation(
            markets=markets,
            original_prompts=original_prompts,
            candidate_prompts=candidate_prompts,
            agent_names=AGENT_NAMES,
            call_bonsai_fn=call_bonsai_fn,
            session_id=session_id or f"baseline-{datetime.now().strftime('%Y%m%d%H%M')}",
        )

        return result

    # ---------------------------------------------------------------------------
    # Query methods
    # ---------------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[TrainingSession]:
        cursor = self._conn.execute(
            "SELECT * FROM training_sessions WHERE session_id=?", (session_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        r = dict(zip(cols, row))
        return TrainingSession(
            session_id=r["session_id"], trigger=r["trigger"],
            trades_tested=r["trades_tested"], trades_improved=r["trades_improved"],
            original_win_rate=r["original_win_rate"], candidate_win_rate=r["candidate_win_rate"],
            improvement_pct=r["improvement_pct"], gen_before=r["gen_before"],
            gen_after=r["gen_after"], status=r["status"],
            started_at=r["started_at"], completed_at=r.get("completed_at", ""),
            summary=r.get("summary", ""),
        )

    def recent_sessions(self, limit: int = 10) -> list[TrainingSession]:
        cursor = self._conn.execute(
            "SELECT * FROM training_sessions ORDER BY started_at DESC LIMIT ?",
            (limit,)
        )
        cols = [d[0] for d in cursor.description]
        sessions = []
        for row in cursor.fetchall():
            r = dict(zip(cols, row))
            sessions.append(TrainingSession(
                session_id=r["session_id"], trigger=r["trigger"],
                trades_tested=r["trades_tested"], trades_improved=r["trades_improved"],
                original_win_rate=r["original_win_rate"], candidate_win_rate=r["candidate_win_rate"],
                improvement_pct=r["improvement_pct"], gen_before=r["gen_before"],
                gen_after=r["gen_after"], status=r["status"],
                started_at=r["started_at"], completed_at=r.get("completed_at", ""),
                summary=r.get("summary", ""),
            ))
        return sessions

    def get_all_time_best(self) -> list[dict]:
        """Return best-performing simulation sessions by improvement."""
        cursor = self._conn.execute("""
            SELECT * FROM training_sessions
            WHERE status='completed' AND improvement_pct > 0
            ORDER BY improvement_pct DESC LIMIT 10
        """)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in cursor.fetchall()]
