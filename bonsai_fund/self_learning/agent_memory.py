"""
Bonsai Fund — Self-Learning: Agent Memory

Each agent's voting history is tracked and used to compute adaptive weights.
When a market in a specific category or price range comes up,
the system uses historical performance to WEIGHT that agent's vote more heavily.

This is the MEDIUM-TERM learning loop — pattern recognition across trade histories.
"""

from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bonsai_fund.agent import AgentVote


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


@dataclass
class AgentPerformance:
    """Snapshot of a single agent's historical performance."""
    agent_id: int
    agent_name: str
    total_calls: int = 0
    correct_calls: int = 0
    avg_confidence: float = 0.0
    avg_edge: float = 0.0
    sharpe_approx: float = 0.0
    calibration_error: float = 0.0

    # Category-specific performance
    category_scores: dict[str, dict] = field(default_factory=dict)
    # Price-range performance: 0-20c, 20-40c, 40-60c, 60-80c, 80-100c
    price_bin_scores: dict[str, dict] = field(default_factory=dict)
    # Time-horizon performance: short (<14d), medium (14-60d), long (>60d)
    horizon_scores: dict[str, dict] = field(default_factory=dict)

    # Computed weight (0.0 to 2.0, default 1.0)
    base_weight: float = 1.0
    category_weight: float = 1.0
    price_weight: float = 1.0
    horizon_weight: float = 1.0

    @property
    def win_rate(self) -> float:
        return self.correct_calls / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def effective_weight(self) -> float:
        return self.base_weight * self.category_weight * self.price_weight * self.horizon_weight

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "total_calls": self.total_calls,
            "correct_calls": self.correct_calls,
            "win_rate": round(self.win_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_edge": round(self.avg_edge, 4),
            "calibration_error": round(self.calibration_error, 4),
            "sharpe_approx": round(self.sharpe_approx, 4),
            "effective_weight": round(self.effective_weight, 4),
            "category_scores": self.category_scores,
            "price_bin_scores": self.price_bin_scores,
            "horizon_scores": self.horizon_scores,
        }


class AgentMemory:
    """
    Tracks every agent vote and builds adaptive performance profiles.

    When analyzing a new market, the memory system looks at:
    1. Category match: Has this agent been accurate on Geopolitics? Sports?
    2. Price-range match: Does the agent perform better on 20c markets vs 80c?
    3. Time-horizon match: Is the agent better on short-term or long-term events?

    The agent's vote is then WEIGHTED accordingly in the final aggregation.

    Learning compounds: more trades = more precise weights.
    """

    def __init__(self):
        self._conn = self._init_db()
        self._performance_cache: dict[int, AgentPerformance] = {}
        self._base_weights: dict[int, float] = {}
        self._last_recalibrate = datetime.min

    def _db_path(self) -> Path:
        return Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "agent_memory.db"

    def _init_db(self) -> sqlite3.Connection:
        p = self._db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_vote_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER, agent_name TEXT, ticker TEXT,
                vote TEXT, confidence REAL, edge_raw REAL,
                market_prob REAL, resolution TEXT,
                correct INTEGER, edge_realized REAL,
                market_category TEXT, price_bin TEXT,
                horizon TEXT, timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_weights (
                agent_id INTEGER PRIMARY KEY,
                agent_name TEXT,
                base_weight REAL DEFAULT 1.0,
                last_updated TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_votes_agent
                ON agent_vote_history(agent_id, correct, market_category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_votes_ticker
                ON agent_vote_history(ticker)
        """)
        conn.commit()
        return conn

    # ---------------------------------------------------------------------------
    # Recording
    # ---------------------------------------------------------------------------

    def record_vote(
        self,
        vote: AgentVote,
        market_prob: float,
        market_category: str,
        resolution: str = "PENDING",  # PENDING until market resolves
        correct: int = 0,
        edge_realized: float = 0.0,
    ):
        """Record a single agent vote in the history."""
        price_bin = self._price_bin(market_prob)
        horizon = self._horizon_bin(0)  # days_to_event not available here

        self._conn.execute("""
            INSERT INTO agent_vote_history (
                agent_id, agent_name, ticker, vote, confidence, edge_raw,
                market_prob, resolution, correct, edge_realized,
                market_category, price_bin, horizon, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vote.agent_id, vote.agent_name, vote.ticker,
            vote.vote, vote.confidence, vote.edge,
            market_prob, resolution, correct, edge_realized,
            market_category, price_bin, horizon,
            datetime.now(timezone.utc).isoformat(),
        ))
        self._conn.commit()

    def record_resolution(self, ticker: str, resolution: str):
        """
        Mark all pending votes for a ticker as resolved.
        Called when a market closes — this is what triggers learning.
        """
        # Find all pending votes for this ticker
        cursor = self._conn.execute(
            "SELECT id, agent_id, agent_name, vote, confidence, edge_raw, market_prob, market_category "
            "FROM agent_vote_history WHERE ticker=? AND resolution='PENDING'",
            (ticker,)
        )
        for row in cursor.fetchall():
            vid, aid, aname, vote, conf, edge_raw, prob, cat = row
            correct = 1 if (vote == resolution) else 0
            edge_realized = abs(edge_raw) if (vote == resolution) else -abs(edge_raw)

            self._conn.execute("""
                UPDATE agent_vote_history
                SET resolution=?, correct=?, edge_realized=?
                WHERE id=?
            """, (resolution, correct, edge_realized, vid))

        self._conn.commit()
        self._invalidate_cache()

    # ---------------------------------------------------------------------------
    # Performance computation
    # ---------------------------------------------------------------------------

    def get_performance(self, agent_id: int) -> AgentPerformance:
        """Get or compute performance profile for an agent."""
        if agent_id in self._performance_cache:
            return self._performance_cache[agent_id]

        perf = self._compute_performance(agent_id)
        self._performance_cache[agent_id] = perf
        return perf

    def _compute_performance(self, agent_id: int) -> AgentPerformance:
        """Compute all performance metrics from vote history."""
        cursor = self._conn.execute(
            "SELECT agent_name FROM agent_vote_history WHERE agent_id=? LIMIT 1",
            (agent_id,)
        )
        row = cursor.fetchone()
        agent_name = row[0] if row else f"Agent_{agent_id}"

        # Overall stats
        cursor = self._conn.execute("""
            SELECT
                COUNT(*),
                SUM(correct),
                AVG(confidence),
                AVG(edge_raw),
                AVG(ABS(edge_realized)) * SIGN(SUM(edge_realized)) as signed_edge
            FROM agent_vote_history
            WHERE agent_id=? AND resolution != 'PENDING'
        """, (agent_id,))
        row = cursor.fetchone()
        total, correct, avg_conf, avg_edge, signed_edge = row
        total = total or 0; correct = correct or 0

        # Category breakdown
        cat_scores = {}
        cursor = self._conn.execute("""
            SELECT market_category, COUNT(*), SUM(correct), AVG(confidence)
            FROM agent_vote_history
            WHERE agent_id=? AND resolution != 'PENDING' AND market_category != 'unknown'
            GROUP BY market_category
        """, (agent_id,))
        for cat, tot, cor, avg_c in cursor.fetchall():
            cat_scores[cat] = {"total": tot, "correct": cor, "win_rate": round(cor/tot, 4) if tot else 0, "avg_confidence": round(avg_c, 4) if avg_c else 0}

        # Price bin breakdown
        price_scores = {}
        cursor = self._conn.execute("""
            SELECT price_bin, COUNT(*), SUM(correct), AVG(confidence)
            FROM agent_vote_history
            WHERE agent_id=? AND resolution != 'PENDING'
            GROUP BY price_bin
        """, (agent_id,))
        for pb, tot, cor, avg_c in cursor.fetchall():
            price_scores[pb] = {"total": tot, "correct": cor, "win_rate": round(cor/tot, 4) if tot else 0, "avg_confidence": round(avg_c, 4) if avg_c else 0}

        # Horizon breakdown
        horizon_scores = {}
        cursor = self._conn.execute("""
            SELECT horizon, COUNT(*), SUM(correct), AVG(confidence)
            FROM agent_vote_history
            WHERE agent_id=? AND resolution != 'PENDING'
            GROUP BY horizon
        """, (agent_id,))
        for hz, tot, cor, avg_c in cursor.fetchall():
            horizon_scores[hz] = {"total": tot, "correct": cor, "win_rate": round(cor/tot, 4) if tot else 0, "avg_confidence": round(avg_c, 4) if avg_c else 0}

        # Base weight from stored weights
        base_weight = self._base_weights.get(agent_id, 1.0)

        perf = AgentPerformance(
            agent_id=agent_id,
            agent_name=agent_name,
            total_calls=total,
            correct_calls=correct,
            avg_confidence=avg_conf or 0.0,
            avg_edge=avg_edge or 0.0,
            category_scores=cat_scores,
            price_bin_scores=price_scores,
            horizon_scores=horizon_scores,
            base_weight=base_weight,
        )

        # Calibration error
        if total > 0 and perf.avg_confidence > 0:
            perf.calibration_error = round(abs(perf.avg_confidence - perf.win_rate), 4)

        # Sharpe approx
        if total > 0 and abs(signed_edge or 0) > 0:
            perf.sharpe_approx = round(correct * abs(avg_edge or 0) / total, 4)

        return perf

    def recalculate_all_performance(self):
        """Recalculate all agent performance profiles from full history."""
        cursor = self._conn.execute(
            "SELECT DISTINCT agent_id FROM agent_vote_history WHERE resolution != 'PENDING'"
        )
        for (aid,) in cursor.fetchall():
            self._performance_cache.pop(aid, None)
        self._last_recalibrate = datetime.now(timezone.utc)

    def _invalidate_cache(self):
        self._performance_cache.clear()

    # ---------------------------------------------------------------------------
    # Context-sensitive weights
    # ---------------------------------------------------------------------------

    def get_weighted_votes(
        self,
        votes: list[AgentVote],
        market_prob: float,
        market_category: str,
        days_to_event: int,
    ) -> list[tuple[AgentVote, float]]:
        """
        Return votes with context-sensitive weights.
        For each vote, look up how well this agent has performed
        in similar markets (same category, price range, time horizon).

        Agents that have a strong track record in similar situations get UP-WEIGHTED.
        Agents that have a poor track record in similar situations get DOWN-WEIGHTED.
        """
        price_bin = self._price_bin(market_prob)
        horizon = self._horizon_bin(days_to_event)

        weighted = []
        for vote in votes:
            perf = self.get_performance(vote.agent_id)

            # Category weight: has this agent been accurate in this category?
            cat_weight = 1.0
            if market_category in perf.category_scores:
                cat_wr = perf.category_scores[market_category]["win_rate"]
                cat_count = perf.category_scores[market_category]["total"]
                if cat_count >= 3:  # need minimum sample
                    cat_weight = 0.5 + cat_wr * 1.0  # 0.5 at 0% win_rate → 1.5 at 100%

            # Price bin weight
            price_weight = 1.0
            if price_bin in perf.price_bin_scores:
                pb_wr = perf.price_bin_scores[price_bin]["win_rate"]
                pb_count = perf.price_bin_scores[price_bin]["total"]
                if pb_count >= 3:
                    price_weight = 0.5 + pb_wr * 1.0

            # Horizon weight
            horizon_weight = 1.0
            if horizon in perf.horizon_scores:
                hz_wr = perf.horizon_scores[horizon]["win_rate"]
                hz_count = perf.horizon_scores[horizon]["total"]
                if hz_count >= 3:
                    horizon_weight = 0.5 + hz_wr * 1.0

            # Effective weight = multiplicative boost/reduction
            eff = perf.base_weight * cat_weight * price_weight * horizon_weight
            weighted.append((vote, eff))

        return weighted

    def get_agent_summary(self) -> list[dict]:
        """Return performance summary for all agents."""
        cursor = self._conn.execute(
            "SELECT DISTINCT agent_id FROM agent_vote_history ORDER BY agent_id"
        )
        summaries = []
        for (aid,) in cursor.fetchall():
            perf = self.get_performance(aid)
            summaries.append(perf.to_dict())
        return summaries

    # ---------------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------------

    @staticmethod
    def _price_bin(prob: float) -> str:
        """Bucket market probability into price bins."""
        if prob <= 0.20: return "0-20c"
        elif prob <= 0.40: return "20-40c"
        elif prob <= 0.60: return "40-60c"
        elif prob <= 0.80: return "60-80c"
        else: return "80-100c"

    @staticmethod
    def _horizon_bin(days: int) -> str:
        if days <= 14: return "short"
        elif days <= 60: return "medium"
        else: return "long"

    @staticmethod
    def _sign(x: float) -> float:
        return 1.0 if x >= 0 else -1.0
