"""
Bonsai Fund — Self-Learning: Market Classifier

Classifies incoming markets into categories and maintains category-level base rates.
When the swarm encounters a market in an unfamiliar category, the classifier
uses analogical reasoning from similar known categories to bootstrap a base rate.

This feeds the MEDIUM-TERM learning loop — after 10-50 trades in a category,
the classifier knows which agents perform best on which types of markets.
"""

from __future__ import annotations
import json
import sqlite3
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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


@dataclass
class MarketCategory:
    """A market category with learned base rates and agent affinities."""
    name: str
    base_rate: float          # Historical YES win rate
    total_trades: int = 0
    yes_wins: int = 0
    no_wins: int = 0
    avg_volume: int = 0
    typical_days_range: tuple[int, int] = (1, 90)
    description: str = ""
    keywords: list[str] = field(default_factory=list)

    # Which agents perform best in this category?
    agent_affinity: dict[int, float] = field(default_factory=dict)
    # Which price ranges are most common?
    price_distribution: dict[str, int] = field(default_factory=dict)
    # Confidence of estimate
    confidence: float = 0.0   # 0.0 = no data, 1.0 = very confident

    @property
    def win_rate(self) -> float:
        return self.yes_wins / self.total_trades if self.total_trades > 0 else 0.5


# Seed categories with initial priors (before any trades)
SEED_CATEGORIES: dict[str, dict] = {
    "Geopolitics": {
        "base_rate": 0.38,
        "description": "Wars, diplomacy, political events between nations",
        "keywords": ["war", "military", "sanction", "nuclear", "treaty", "putin", "iran", "china", "russia", "election", "president"],
        "days_range": (30, 365),
        "confidence": 0.5,
    },
    "NBA": {
        "base_rate": 0.50,
        "description": "NBA basketball — playoffs, championships, MVP, trades",
        "keywords": ["nba", "lakers", "warriors", "playoffs", "finals", "mvp", "basketball", "champion"],
        "days_range": (30, 240),
        "confidence": 0.5,
    },
    "NFL": {
        "base_rate": 0.50,
        "description": "NFL football — Super Bowl, playoffs, MVP, season wins",
        "keywords": ["nfl", "super bowl", "playoffs", "afc", "nfc", "mvp", "football", "season wins"],
        "days_range": (30, 240),
        "confidence": 0.5,
    },
    "Earnings": {
        "base_rate": 0.52,
        "description": "Company earnings reports — EPS beat/miss, revenue",
        "keywords": ["eps", "earnings", "revenue", "quarterly", "q1", "q2", "q3", "q4", "amazon", "apple", "google", "meta"],
        "days_range": (1, 60),
        "confidence": 0.5,
    },
    "Jobs": {
        "base_rate": 0.50,
        "description": "US jobs reports — NFP, unemployment, payrolls",
        "keywords": ["jobs", "payroll", "employment", "unemployment", "nfp", "labor"],
        "days_range": (1, 35),
        "confidence": 0.5,
    },
    "CPI": {
        "base_rate": 0.40,
        "description": "Consumer price index — inflation readings",
        "keywords": ["cpi", "inflation", "pce", "price index", "core cpi", "headline cpi"],
        "days_range": (1, 35),
        "confidence": 0.5,
    },
    "Crypto": {
        "base_rate": 0.45,
        "description": "Cryptocurrency prices and events",
        "keywords": ["btc", "bitcoin", "eth", "ethereum", "crypto", "coinbase", "sec"],
        "days_range": (30, 365),
        "confidence": 0.5,
    },
    "Hurricane": {
        "base_rate": 0.40,
        "description": "Hurricane and tropical storm landfalls",
        "keywords": ["hurricane", "tropical", "storm", "gulf coast", "atlantic", "landfall"],
        "days_range": (60, 180),
        "confidence": 0.5,
    },
    "Climate": {
        "base_rate": 0.50,
        "description": "Climate events — El Nino/La Nina, temperature records",
        "keywords": ["el nino", "la nina", "temperature", "climate", "weather", "anomaly"],
        "days_range": (14, 180),
        "confidence": 0.5,
    },
    "Elections": {
        "base_rate": 0.48,
        "description": "Political elections — presidential, congressional",
        "keywords": ["election", "president", "congress", "senate", "house", "governor", "vote"],
        "days_range": (14, 365),
        "confidence": 0.5,
    },
    "Interest Rates": {
        "base_rate": 0.45,
        "description": "Fed rate decisions and monetary policy",
        "keywords": ["fed", "rate", "federal reserve", "powell", "fomc", "cut", "hike", "interest"],
        "days_range": (1, 180),
        "confidence": 0.5,
    },
    "Economics": {
        "base_rate": 0.48,
        "description": "General macroeconomic indicators and forecasts",
        "keywords": ["gdp", "recession", "pce", "retail sales", "housing starts", "manufacturing"],
        "days_range": (1, 180),
        "confidence": 0.5,
    },
}


class MarketClassifier:
    """
    Classifies markets and maintains category-level learning.

    For any incoming market, the classifier:
    1. Identifies the best-matching category via keyword + text analysis
    2. Returns the historical base rate for that category (calibrated with seed priors)
    3. Tracks which agents perform best in each category
    4. Uses analogical reasoning when a category is new (borrows from similar categories)
    """

    def __init__(self):
        self._conn = self._init_db()
        self._categories: dict[str, MarketCategory] = {}
        self._load_categories()

    def _db_path(self) -> Path:
        return Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "market_classifier.db"

    def _init_db(self) -> sqlite3.Connection:
        p = self._db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                name TEXT PRIMARY KEY,
                base_rate REAL,
                total_trades INTEGER DEFAULT 0,
                yes_wins INTEGER DEFAULT 0,
                no_wins INTEGER DEFAULT 0,
                avg_volume INTEGER DEFAULT 0,
                description TEXT,
                keywords_json TEXT,
                agent_affinity_json TEXT,
                confidence REAL DEFAULT 0.0,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, market_title TEXT, series_title TEXT,
                classified_category TEXT, keywords_matched TEXT,
                base_rate_used REAL, confidence REAL, timestamp TEXT
            )
        """)
        conn.commit()
        return conn

    def _load_categories(self):
        """Load categories from DB, falling back to seed data."""
        self._categories = {}
        # Load from DB
        cursor = self._conn.execute("SELECT * FROM categories")
        cols = [d[0] for d in cursor.description]
        for row in cursor.fetchall():
            r = dict(zip(cols, row))
            self._categories[r["name"]] = MarketCategory(
                name=r["name"],
                base_rate=r["base_rate"] or 0.5,
                total_trades=r["total_trades"] or 0,
                yes_wins=r["yes_wins"] or 0,
                no_wins=r["no_wins"] or 0,
                avg_volume=r["avg_volume"] or 0,
                description=r.get("description", ""),
                keywords=json.loads(r.get("keywords_json", "[]")),
                agent_affinity=json.loads(r.get("agent_affinity_json", "{}")),
                confidence=r.get("confidence", 0.0),
            )
        # Fill in missing seed categories
        for name, data in SEED_CATEGORIES.items():
            if name not in self._categories:
                self._categories[name] = MarketCategory(
                    name=name,
                    base_rate=data["base_rate"],
                    description=data["description"],
                    keywords=data["keywords"],
                    confidence=data["confidence"],
                )
        return self._categories

    def classify(self, market) -> MarketCategory:
        """
        Classify a market by ticker + title + series_title.
        Returns the best-matching MarketCategory with an updated base rate.
        """
        text = f"{market.ticker} {market.market_title} {market.series_title}".lower()

        best_cat = None
        best_score = 0

        for name, cat in self._categories.items():
            score = 0
            for kw in cat.keywords:
                if kw.lower() in text:
                    score += 1
            # Also boost by confidence — prefer known categories
            score += cat.confidence * 0.5
            if score > best_score:
                best_score = score
                best_cat = cat

        if best_cat is None or best_score == 0:
            # Unknown market — use broad "Economics" category
            best_cat = self._categories.get("Economics", MarketCategory(
                name="Economics",
                base_rate=0.48,
                description="General economic events",
                confidence=0.1,
            ))

        # Record classification
        keywords_matched = [kw for kw in best_cat.keywords if kw.lower() in text]
        self._conn.execute("""
            INSERT INTO classification_history
            (ticker, market_title, series_title, classified_category, keywords_matched,
             base_rate_used, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market.ticker, market.market_title, market.series_title,
            best_cat.name, ",".join(keywords_matched),
            best_cat.base_rate, best_cat.confidence,
            datetime.now(timezone.utc).isoformat(),
        ))
        self._conn.commit()

        return best_cat

    def update_from_outcome(
        self,
        market,
        resolution: str,  # YES / NO
        agent_scores: list[dict],
    ):
        """Update category base rate and agent affinities after a market resolves."""
        cat = self.classify(market)

        yes_win = 1 if resolution == "YES" else 0
        total = cat.total_trades + 1
        yes_wins = cat.yes_wins + yes_win
        new_base_rate = yes_wins / total

        # Update agent affinity: which agents voted correctly?
        agent_affinity = dict(cat.agent_affinity)
        for score in agent_scores:
            # If agent voted correctly in this category, boost their affinity
            aid = score["agent_id"]
            current = agent_affinity.get(str(aid), 1.0)
            vote_correct = score.get("vote_correct", 0)
            if vote_correct:
                agent_affinity[str(aid)] = min(2.0, current * 1.05)  # 5% boost, cap at 2.0
            else:
                agent_affinity[str(aid)] = max(0.5, current * 0.95)  # 5% reduction, floor at 0.5

        # New confidence: sqrt(n) convergence
        new_confidence = min(1.0, (total ** 0.5) / 10.0)  # 100 trades → 1.0 confidence

        self._conn.execute("""
            INSERT INTO categories
            (name, base_rate, total_trades, yes_wins, no_wins, avg_volume,
             description, keywords_json, agent_affinity_json, confidence, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                base_rate = excluded.base_rate,
                total_trades = excluded.total_trades,
                yes_wins = excluded.yes_wins,
                no_wins = excluded.no_wins,
                agent_affinity_json = excluded.agent_affinity_json,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
        """, (
            cat.name, new_base_rate, total, yes_wins, total - yes_wins, market.volume,
            cat.description, json.dumps(cat.keywords),
            json.dumps(agent_affinity), new_confidence,
            datetime.now(timezone.utc).isoformat(),
        ))
        self._conn.commit()

        # Reload
        self._load_categories()

    def get_base_rate(self, market) -> tuple[str, float, float]:
        """
        Get the best base rate for a market.
        Returns (category_name, base_rate, confidence).
        """
        cat = self.classify(market)
        return cat.name, cat.base_rate, cat.confidence

    def get_agent_affinities(self, category_name: str) -> dict[int, float]:
        """Return agent weight recommendations for a category."""
        cat = self._categories.get(category_name)
        if cat:
            return {int(k): v for k, v in cat.agent_affinity.items()}
        return {}

    def get_all_categories(self) -> dict[str, MarketCategory]:
        return self._categories

    def analogy_reasoning(self, source_category: str, target_text: str) -> float:
        """
        When a category has low confidence, borrow its base rate from
        the most similar high-confidence category.

        E.g., "NFT market" has no data, but is similar to "Crypto"
        → use Crypto's base rate as the prior.
        """
        if source_category not in self._categories:
            return 0.5
        cat = self._categories[source_category]
        if cat.confidence >= 0.5:
            return cat.base_rate

        # Find most similar category by keyword overlap
        target_lower = target_text.lower()
        best_similar = None
        best_overlap = 0
        for name, c in self._categories.items():
            if name == source_category or c.confidence < 0.5:
                continue
            overlap = sum(1 for kw in c.keywords if kw.lower() in target_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_similar = c

        if best_similar and best_overlap > 0:
            # Blend: low_confidence prior + similar_category base rate
            blend = cat.confidence * cat.base_rate + best_similar.confidence * best_similar.base_rate
            blend /= (cat.confidence + best_similar.confidence)
            return round(blend, 4)

        return cat.base_rate
