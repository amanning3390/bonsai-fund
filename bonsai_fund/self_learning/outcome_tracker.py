"""
Bonsai Fund — Self-Learning: Outcome Tracker

Records every trade outcome and feeds results back into the learning pipeline.
After market resolution, the outcome is tagged with the agent votes that produced it,
then used to update agent weights, refine edge estimates, and detect systematic biases.

This is the SHORT-TERM learning loop — single-trade feedback.
"""

from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, field, asdict
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
class TradeOutcome:
    """
    The result of a single trade — used to update agent weights.
    Stores the market context and the agent votes that generated the signal.
    """
    ticker: str
    resolution: str            # YES / NO (what actually happened)
    market_prob_before: float  # market-implied probability before the trade
    yes_fill_price: float     # cents paid for YES contracts
    no_fill_price: float      # cents paid for NO contracts
    side: str                 # YES / NO (what we traded)
    contracts: int
    realized_pnl: float       # $ profit/loss
    roi_pct: float            # return on capital deployed
    days_to_resolution: int

    # Agent votes that generated this signal
    agent_votes: list[dict]   # [{agent_id, agent_name, vote, confidence, edge}]

    # Context
    market_category: str = "unknown"
    market_title: str = ""
    event_date: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Derived after recording
    vote_correct: bool = field(default=False)
    edge_realized: float = field(default=0.0)


class OutcomeTracker:
    """
    Records trade outcomes and computes agent-level accuracy scores.

    After a market resolves, call record_resolution(ticker).
    The tracker will:
    1. Find all positions for that ticker
    2. Determine if the trade won or lost
    3. Attribute the outcome to each agent vote that contributed
    4. Update rolling accuracy stats for each agent
    5. Feed into the medium-term MarketClassifier for category patterns
    """

    def __init__(self):
        self._conn = self._init_db()

    def _db_path(self) -> Path:
        return Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "outcomes.db"

    def _init_db(self) -> sqlite3.Connection:
        p = self._db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, resolution TEXT, market_prob_before REAL,
                yes_fill_price REAL, no_fill_price REAL, side TEXT,
                contracts INTEGER, realized_pnl REAL, roi_pct REAL,
                days_to_resolution INTEGER, market_category TEXT,
                market_title TEXT, event_date TEXT, timestamp TEXT,
                vote_correct INTEGER, edge_realized REAL,
                agent_votes_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_scores (
                agent_id INTEGER PRIMARY KEY,
                agent_name TEXT,
                total_votes INTEGER DEFAULT 0,
                correct_votes INTEGER DEFAULT 0,
                total_edge_sum REAL DEFAULT 0.0,
                total_edge_correct REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                avg_edge_raw REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                calibration_error REAL DEFAULT 0.0,
                sharpe_approx REAL DEFAULT 0.0,
                last_updated TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS category_base_rates (
                category TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                yes_wins INTEGER DEFAULT 0,
                no_wins INTEGER DEFAULT 0,
                base_rate REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.0,
                updated_at TEXT
            )
        """)
        conn.commit()
        return conn

    def record_trade(
        self,
        ticker: str,
        side: str,
        yes_fill: float,
        no_fill: float,
        contracts: int,
        market_prob: float,
        market_title: str,
        days_to_resolution: int,
        agent_votes: list[AgentVote],
        market_category: str = "unknown",
    ) -> int:
        """Record a new open trade (before resolution)."""
        return self._conn.execute("""
            INSERT INTO outcomes (
                ticker, side, yes_fill_price, no_fill_price, contracts,
                market_prob_before, market_title, market_category,
                days_to_resolution, timestamp, resolution,
                realized_pnl, roi_pct, vote_correct, edge_realized
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, side, yes_fill, no_fill, contracts,
            market_prob, market_title, market_category,
            days_to_resolution, datetime.now(timezone.utc).isoformat(),
            "PENDING", 0.0, 0.0, 0, 0.0,
            json.dumps([{
                "agent_id": v.agent_id,
                "agent_name": v.agent_name,
                "vote": v.vote,
                "confidence": v.confidence,
                "edge": v.edge,
            } for v in agent_votes])
        )).lastrowid

    def resolve_market(
        self,
        ticker: str,
        resolution: str,   # YES or NO
        yes_fill: float,
        no_fill: float,
        contracts: int,
    ) -> Optional[dict]:
        """
        Mark a trade as resolved. Call this when a market closes.
        Returns the outcome dict with win/loss data.
        """
        # Find the open position record
        cursor = self._conn.execute(
            "SELECT id, side, market_prob_before, roi_pct, agent_votes_json "
            "FROM outcomes WHERE ticker=? AND resolution='PENDING' LIMIT 1",
            (ticker,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        outcome_id, side, market_prob, _, agent_votes_json = row
        agent_votes = json.loads(agent_votes_json or "[]")

        # Calculate P&L
        if side == "YES":
            if resolution == "YES":
                realized_pnl = contracts * (yes_fill - yes_fill)  # closed at market
                # Actually we need current market price — approximate with NO side
                # For a proper P&L, use the market's closing price
                realized_pnl = contracts * (100.0 - yes_fill) / 100.0  # full value on YES win
            else:
                realized_pnl = -contracts * yes_fill / 100.0  # lost the premium
        else:  # NO side
            if resolution == "NO":
                realized_pnl = contracts * (100.0 - no_fill) / 100.0
            else:
                realized_pnl = -contracts * no_fill / 100.0

        roi_pct = (realized_pnl / (contracts * (yes_fill if side == "YES" else no_fill) / 100.0)) * 100.0
        vote_correct = 1 if (side == resolution) else 0
        edge_realized = market_prob - (1.0 if resolution == "YES" else 0.0)

        self._conn.execute("""
            UPDATE outcomes SET
                resolution=?, realized_pnl=?, roi_pct=?,
                vote_correct=?, edge_realized=?
            WHERE id=?
        """, (resolution, realized_pnl, roi_pct, vote_correct, edge_realized, outcome_id))

        # Update per-agent scores
        self._update_agent_scores(agent_votes, vote_correct, side, resolution, market_prob)

        # Update category base rates
        category = self._get_category(ticker)
        self._update_category_base_rate(category, resolution)

        self._conn.commit()
        return {
            "ticker": ticker, "resolution": resolution,
            "realized_pnl": realized_pnl, "roi_pct": roi_pct,
            "vote_correct": vote_correct, "side": side,
        }

    def _get_category(self, ticker: str) -> str:
        cursor = self._conn.execute(
            "SELECT market_category FROM outcomes WHERE ticker=? LIMIT 1", (ticker,)
        )
        row = cursor.fetchone()
        return row[0] if row else "unknown"

    def _update_agent_scores(
        self,
        agent_votes: list[dict],
        correct: int,
        side: str,
        resolution: str,
        market_prob: float,
    ):
        """Update rolling per-agent accuracy stats."""
        for av in agent_votes:
            aid = av["agent_id"]
            aname = av["agent_name"]
            vote = av["vote"]
            conf = av["confidence"]
            edge_raw = av["edge"]

            # Was this agent's vote direction correct?
            vote_correct = 1 if (vote == resolution) else 0

            # Marginal edge contribution
            edge_if_correct = abs(edge_raw) if vote == resolution else 0.0

            self._conn.execute("""
                INSERT INTO agent_scores (
                    agent_id, agent_name, total_votes, correct_votes,
                    total_edge_sum, total_edge_correct, last_updated
                ) VALUES (?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    total_votes = total_votes + 1,
                    correct_votes = correct_votes + excluded.correct_votes,
                    total_edge_sum = total_edge_sum + excluded.total_edge_sum,
                    total_edge_correct = total_edge_correct + excluded.total_edge_correct,
                    last_updated = excluded.last_updated
            """, (aid, aname, vote_correct, abs(edge_raw), edge_if_correct,
                  datetime.now(timezone.utc).isoformat()))

    def _update_category_base_rate(self, category: str, resolution: str):
        """Update base win rate for a market category."""
        self._conn.execute("""
            INSERT INTO category_base_rates (category, total_trades, yes_wins, no_wins, updated_at)
            VALUES (?, 1,
                CASE WHEN ?='YES' THEN 1 ELSE 0 END,
                CASE WHEN ?='NO' THEN 1 ELSE 0 END,
                ?)
            ON CONFLICT(category) DO UPDATE SET
                total_trades = total_trades + 1,
                yes_wins = yes_wins + CASE WHEN excluded.yes_wins=1 THEN 1 ELSE 0 END,
                no_wins = no_wins + CASE WHEN excluded.no_wins=1 THEN 1 ELSE 0 END,
                base_rate = CAST(yes_wins AS REAL) / total_trades,
                updated_at = excluded.updated_at
        """, (category, resolution, resolution, datetime.now(timezone.utc).isoformat()))

    def get_agent_scores(self) -> list[dict]:
        """Return per-agent performance scores."""
        cursor = self._conn.execute("SELECT * FROM agent_scores ORDER BY agent_id")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        scores = []
        for row in rows:
            r = dict(zip(cols, row))
            total = r["total_votes"] or 1
            correct = r["correct_votes"] or 0
            r["win_rate"] = round(correct / total, 4)
            r["avg_confidence"] = round(r["avg_confidence"] / total, 4) if r["avg_confidence"] else 0
            r["avg_edge_raw"] = round(r["avg_edge_raw"] / total, 4) if r["total_edge_sum"] else 0
            # Calibration error: |predicted_prob - actual_win_rate|
            pred_prob = r["avg_confidence"]
            r["calibration_error"] = round(abs(pred_prob - r["win_rate"]), 4)
            # Sharpe-like: edge_correct / edge_sum (signal quality ratio)
            if r["total_edge_sum"] and r["total_edge_sum"] > 0:
                r["sharpe_approx"] = round(r["total_edge_correct"] / r["total_edge_sum"], 4)
            else:
                r["sharpe_approx"] = 0.0
            scores.append(r)
        return scores

    def get_category_base_rates(self) -> dict[str, float]:
        """Return current base rates by category."""
        cursor = self._conn.execute("SELECT category, base_rate, total_trades, confidence FROM category_base_rates")
        return {r[0]: {"base_rate": r[1], "trades": r[2], "confidence": r[3]}
                for r in cursor.fetchall()}

    def get_recent_outcomes(self, days: int = 30) -> list[dict]:
        """Get recent resolved outcomes."""
        cutoff = (datetime.now(timezone.utc) - datetime.timedelta(days=days)).isoformat()
        cursor = self._conn.execute(
            "SELECT ticker, resolution, side, realized_pnl, roi_pct, vote_correct, market_category "
            "FROM outcomes WHERE timestamp >= ? AND resolution != 'PENDING' ORDER BY timestamp DESC",
            (cutoff,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in cursor.fetchall()]

    def get_total_edge_analysis(self) -> dict:
        """Aggregate analysis: how much edge has the swarm been capturing?"""
        cursor = self._conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(vote_correct) as correct,
                SUM(realized_pnl) as total_pnl,
                AVG(roi_pct) as avg_roi,
                AVG(edge_realized) as avg_edge
            FROM outcomes WHERE resolution != 'PENDING'
        """)
        row = cursor.fetchone()
        return {
            "total_trades": row[0] or 0,
            "correct": row[1] or 0,
            "total_pnl": round(row[2] or 0.0, 4),
            "avg_roi": round(row[3] or 0.0, 2),
            "avg_edge": round(row[4] or 0.0, 4),
        }
