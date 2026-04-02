"""
Bonsai Fund — News & Event Pipeline

Monitors RSS feeds and web sources for market-relevant news,
extracts key facts, and injects them into agent prompts before each scan.

Key sources:
  - RSS feeds for prediction-market-relevant news categories
  - Web scraping for Kalshi market descriptions (market context)
  - Calendar events (FOMC, NFP, CPI release dates) as context

Injection: Before each scan, news is summarized and prepended to agent prompts.
The news context is stored in SQLite so the system "remembers" what it knew when.
"""

from __future__ import annotations
import json
import sqlite3
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from bonsai_fund.agent import Market


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
class NewsItem:
    title: str
    url: str
    source: str
    published_at: str
    summary: str
    category: str          # "geopolitics" | "earnings" | "macro" | "sports" | "weather" | etc.
    relevance_score: float  # 0.0 - 1.0
    ticker_hints: list[str] = field(default_factory=list)  # potentially affected tickers
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_hash: str = ""


@dataclass
class MarketContext:
    """News context for a specific market, assembled before the scan."""
    ticker: str
    headlines: list[str]       # recent headlines relevant to this market
    calendar_events: list[str] # upcoming economic releases
    sentiment_tags: list[str]  # "bullish", "risk-off", "inflation-fear", etc.
    composite_summary: str     # the full context string injected into prompts


# ---------------------------------------------------------------------------
# RSS Feed Registry
# ---------------------------------------------------------------------------

RSS_FEEDS = {
    "general": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.ft.com/?format=rss",
    ],
    "geopolitics": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    ],
    "economics": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
        "https://www.ft.com/?format=rss&tab=markets",
    ],
    "earnings": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ],
    "sports": [
        "https://www.espn.com/espn/rss/news",
    ],
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    ],
    "weather": [
        "https://www.nhc.noxml.us/rss/AlternateAtlasFeeds.xml",
    ],
}


# ---------------------------------------------------------------------------
# News Pipeline
# ---------------------------------------------------------------------------

class NewsPipeline:
    """
    Ingests news from RSS feeds, filters for market relevance,
    and assembles context for injection into agent prompts.

    All news is cached in SQLite with content hashing to avoid duplicates.
    News older than 48 hours is archived, not deleted (for backtesting).
    """

    def __init__(self):
        self._conn = self._init_db()
        self._last_fetch: dict[str, str] = {}
        self._session_cache: dict[str, MarketContext] = {}

    def _db_path(self) -> Path:
        return Path(_cfg(
            "data_dir",
            str(Path.home() / ".hermes" / "bonsai_fund_data")
        )) / "news_pipeline.db"

    def _init_db(self) -> sqlite3.Connection:
        p = self._db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, url TEXT UNIQUE, source TEXT,
                published_at TEXT, summary TEXT, category TEXT,
                relevance_score REAL, ticker_hints_json TEXT,
                ingested_at TEXT, content_hash TEXT,
                archived INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT, event_date TEXT,
                source TEXT, ticker_hint TEXT,
                ingested_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_context_cache (
                ticker TEXT PRIMARY KEY,
                composite_summary TEXT,
                headlines_json TEXT,
                sentiment_tags_json TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_category
                ON news_items(category, published_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ingested
                ON news_items(ingested_at)
        """)
        conn.commit()
        return conn

    # ---------------------------------------------------------------------------
    # Fetching
    # ---------------------------------------------------------------------------

    def fetch_rss_feeds(self, categories: list[str] = None) -> list[NewsItem]:
        """Fetch and ingest news from RSS feeds."""
        if categories is None:
            categories = list(RSS_FEEDS.keys())

        all_items = []
        for cat in categories:
            feeds = RSS_FEEDS.get(cat, [])
            for feed_url in feeds:
                try:
                    items = self._fetch_single_feed(feed_url, cat)
                    all_items.extend(items)
                except Exception as e:
                    pass  # silently skip failed feeds

        # Deduplicate by URL
        seen, unique = set(), []
        for item in all_items:
            if item.url not in seen:
                seen.add(item.url)
                unique.append(item)
        return unique

    def _fetch_single_feed(self, url: str, category: str) -> list[NewsItem]:
        """Fetch a single RSS feed."""
        import urllib.request
        import xml.etree.ElementTree as ET

        req = urllib.request.Request(url, headers={"User-Agent": "BonsaiFund/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()

        root = ET.fromstring(raw)
        items = []
        for entry in root.findall(".//item"):
            try:
                title = (entry.findtext("title") or "").strip()
                link = (entry.findtext("link") or "").strip()
                pub = (entry.findtext("pubDate") or "").strip()
                desc = (entry.findtext("description") or "").strip()

                if not title or not link:
                    continue

                # Clean HTML from description
                desc = desc[:300] if desc else ""
                h = hashlib.md5((title + link).encode()).hexdigest()

                # Try to extract date
                try:
                    from email.utils import parsedate_to_datetime
                    pub_dt = parsedate_to_datetime(pub)
                    pub_str = pub_dt.isoformat()
                except:
                    pub_str = datetime.now(timezone.utc).isoformat()

                # Relevance scoring
                score = self._score_relevance(title, desc, category)

                # Ticker hints
                tickers = self._extract_ticker_hints(title, desc)

                item = NewsItem(
                    title=title[:200],
                    url=link,
                    source=url.split("/")[2],
                    published_at=pub_str,
                    summary=desc,
                    category=category,
                    relevance_score=score,
                    ticker_hints=tickers,
                    content_hash=h,
                )
                items.append(item)

                # Save to DB
                self._save_item(item)

            except Exception:
                pass

        return items

    def _score_relevance(self, title: str, desc: str, category: str) -> float:
        """Score how relevant a news item is for trading."""
        text = (title + " " + desc).lower()
        score = 0.5  # baseline

        # Boost for prediction-market-relevant keywords
        relevant_keywords = {
            "geopolitics": ["war", "military", "sanction", "nuclear", "russia", "china", "iran", "putin", "election", "vote"],
            "economics": ["fed", "rate", "inflation", "cpi", "gdp", "jobs", "payroll", "recession", "unemployment"],
            "earnings": ["earnings", "eps", "revenue", "quarterly", "profit", "beat", "miss"],
            "sports": ["nba", "nfl", "super bowl", "playoffs", "championship", "finals"],
            "crypto": ["bitcoin", "ethereum", "sec", "etf", "crypto", "coinbase", "binance"],
            "weather": ["hurricane", "storm", "tropical", "landfall", "climate", "el nino"],
        }

        for kw in relevant_keywords.get(category, []):
            if kw in text:
                score += 0.1

        return min(1.0, score)

    def _extract_ticker_hints(self, title: str, desc: str) -> list[str]:
        """Try to extract market tickers mentioned in the headline."""
        import re
        text = title + " " + desc
        # Look for common ticker patterns
        tickers = re.findall(r'\b[A-Z]{1,5}(?:-[A-Z0-9]{1,6})\b', text)
        # Filter to likely prediction market tickers (Kalshi-style)
        valid = []
        for t in tickers:
            t = t.upper()
            if len(t) >= 4 and not t.startswith("HTTP"):
                valid.append(t)
        return list(set(valid))[:3]

    def _save_item(self, item: NewsItem):
        self._conn.execute("""
            INSERT OR IGNORE INTO news_items
            (title, url, source, published_at, summary, category,
             relevance_score, ticker_hints_json, ingested_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.title, item.url, item.source, item.published_at,
            item.summary, item.category, item.relevance_score,
            json.dumps(item.ticker_hints), item.ingested_at, item.content_hash,
        ))
        self._conn.commit()

    # ---------------------------------------------------------------------------
    # Context assembly
    # ---------------------------------------------------------------------------

    def get_context_for_market(self, market: Market) -> MarketContext:
        """
        Assemble news context for a specific market before the scan.
        Uses cached context if fresh (< 1 hour).
        """
        # Check cache
        if market.ticker in self._session_cache:
            return self._session_cache[market.ticker]

        # Check DB cache
        cursor = self._conn.execute(
            "SELECT * FROM market_context_cache WHERE ticker=?",
            (market.ticker,)
        )
        row = cursor.fetchone()
        if row:
            cols = [d[0] for d in cursor.description]
            r = dict(zip(cols, row))
            cached_at = datetime.fromisoformat(r["updated_at"])
            if (datetime.now(timezone.utc) - cached_at).total_seconds() < 3600:
                ctx = MarketContext(
                    ticker=market.ticker,
                    headlines=json.loads(r["headlines_json"]),
                    calendar_events=[],
                    sentiment_tags=json.loads(r["sentiment_tags_json"]),
                    composite_summary=r["composite_summary"],
                )
                self._session_cache[market.ticker] = ctx
                return ctx

        # Build fresh context
        headlines = self._get_recent_headlines(market)
        sentiment_tags = self._score_sentiment(market, headlines)
        composite = self._build_composite(market, headlines, sentiment_tags)

        ctx = MarketContext(
            ticker=market.ticker,
            headlines=headlines,
            calendar_events=self._get_upcoming_events(market),
            sentiment_tags=sentiment_tags,
            composite_summary=composite,
        )

        # Cache it
        self._conn.execute("""
            INSERT OR REPLACE INTO market_context_cache
            (ticker, composite_summary, headlines_json, sentiment_tags_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            market.ticker, composite, json.dumps(headlines),
            json.dumps(sentiment_tags), datetime.now(timezone.utc).isoformat(),
        ))
        self._conn.commit()

        self._session_cache[market.ticker] = ctx
        return ctx

    def _get_recent_headlines(self, market: Market, hours: int = 24) -> list[str]:
        """Get recent headlines relevant to a market."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        # Determine relevant categories
        categories = self._market_categories(market)

        headlines = []
        for cat in categories:
            cursor = self._conn.execute("""
                SELECT title, relevance_score FROM news_items
                WHERE category=? AND ingested_at >= ? AND archived=0
                ORDER BY relevance_score DESC, ingested_at DESC LIMIT 5
            """, (cat, cutoff))
            for title, score in cursor.fetchall():
                if score and score >= 0.4:
                    headlines.append(f"[{score:.0%}] {title}")

        # Also look for ticker-specific news
        cursor = self._conn.execute("""
            SELECT title, relevance_score FROM news_items
            WHERE ticker_hints_json LIKE ? AND ingested_at >= ?
            ORDER BY relevance_score DESC LIMIT 5
        """, (f"%{market.ticker}%", cutoff))
        for title, score in cursor.fetchall():
            headlines.append(f"[{score:.0%}] {title}")

        return headlines[:8]

    def _market_categories(self, market: Market) -> list[str]:
        """Map a market to relevant news categories."""
        text = (market.market_title + " " + market.series_title).lower()
        cats = []
        if any(k in text for k in ["war", "military", "sanction", "russia", "iran", "putin", "election"]):
            cats.append("geopolitics")
        if any(k in text for k in ["cpi", "inflation", "jobs", "payroll", "fed", "rate", "gdp", "recession"]):
            cats.append("economics")
        if any(k in text for k in ["earnings", "eps", "revenue", "quarterly"]):
            cats.append("earnings")
        if any(k in text for k in ["nba", "nfl", "super bowl", "playoffs", "championship"]):
            cats.append("sports")
        if any(k in text for k in ["bitcoin", "ethereum", "crypto", "sec", "coinbase"]):
            cats.append("crypto")
        if any(k in text for k in ["hurricane", "storm", "climate", "weather"]):
            cats.append("weather")
        return cats if cats else ["general"]

    def _score_sentiment(self, market: Market, headlines: list[str]) -> list[str]:
        """Score the sentiment of recent headlines for this market."""
        text = " ".join(headlines).lower()
        tags = []
        if any(k in text for k in ["war", "escalation", "sanction", "military", "attack"]):
            tags.append("risk-off")
        if any(k in text for k in ["fed", "rate hike", "inflation", "hot cpi"]):
            tags.append("inflation-fear")
        if any(k in text for k in ["fed", "rate cut", "easing", "soft landing"]):
            tags.append("risk-on")
        if any(k in text for k in ["beat", "surge", "rally", "strong", "boom"]):
            tags.append("bullish")
        if any(k in text for k in ["miss", "plunge", "crash", "weak", "bust"]):
            tags.append("bearish")
        return tags

    def _get_upcoming_events(self, market: Market) -> list[str]:
        """Get upcoming economic calendar events relevant to this market."""
        # Check calendar events table
        cursor = self._conn.execute("""
            SELECT event_name, event_date FROM calendar_events
            WHERE event_date >= ?
            ORDER BY event_date LIMIT 5
        """, (datetime.now(timezone.utc).date().isoformat(),))
        return [f"{date}: {name}" for name, date in cursor.fetchall()]

    def _build_composite(self, market: Market, headlines: list[str], sentiment_tags: list[str]) -> str:
        """Build the composite context string injected into agent prompts."""
        parts = []

        if headlines:
            parts.append(f"RECENT NEWS FOR {market.ticker}:")
            for h in headlines[:5]:
                parts.append(f"  • {h}")
            parts.append("")

        if sentiment_tags:
            parts.append(f"Sentiment tags: {', '.join(sentiment_tags)}")
            parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    # ---------------------------------------------------------------------------
    # News context injection for prompts
    # ---------------------------------------------------------------------------

    def build_scanner_context(self, markets: list[Market]) -> dict[str, MarketContext]:
        """
        Build context for all markets in a scan.
        Returns dict of ticker -> MarketContext.
        """
        return {m.ticker: self.get_context_for_market(m) for m in markets}

    def prepend_to_prompt(self, system_prompt: str, market: Market) -> str:
        """
        Prepend news context to an agent's system prompt before the scan.
        This is the key integration point with hedge_fund.py.
        """
        ctx = self.get_context_for_market(market)
        if not ctx.composite_summary:
            return system_prompt

        return (
            f"[MARKET CONTEXT — READ BEFORE ANALYZING]\n"
            f"{ctx.composite_summary}\n"
            f"{'='*50}\n\n"
            f"{system_prompt}"
        )

    # ---------------------------------------------------------------------------
    # Calendar events
    # ---------------------------------------------------------------------------

    def add_calendar_event(self, name: str, date: str, source: str = "", ticker_hint: str = ""):
        """Add an upcoming economic calendar event."""
        self._conn.execute("""
            INSERT INTO calendar_events (event_name, event_date, source, ticker_hint, ingested_at)
            VALUES (?, ?, ?, ?, ?)
        """, (name, date, source, ticker_hint, datetime.now(timezone.utc).isoformat()))
        self._conn.commit()

    def seed_economic_calendar(self):
        """Seed the calendar with known economic release dates."""
        import datetime as dt
        now = datetime.now(timezone.utc)
        year = now.year

        # FOMC meeting dates (approximate)
        fomc_dates = [
            f"{year}-01-29", f"{year}-01-30",
            f"{year}-03-18", f"{year}-03-19",
            f"{year}-05-06", f"{year}-05-07",
            f"{year}-06-17", f"{year}-06-18",
            f"{year}-07-29", f"{year}-07-30",
            f"{year}-09-16", f"{year}-09-17",
            f"{year}-11-04", f"{year}-11-05",
            f"{year}-12-16", f"{year}-12-17",
        ]
        for d in fomc_dates:
            try:
                dt.date.fromisoformat(d)
                self.add_calendar_event("FOMC Meeting", d, "Bonsai Fund Seed", "KXFed-??")
            except:
                pass

        # NFP (first Friday of month) — approximate
        for month in range(1, 13):
            nfp_date = f"{year}-{month:02d}-07"  # rough approximation
            self.add_calendar_event(f"NFP Report", nfp_date, "Bonsai Fund Seed", "KXJOBS-??")

        # CPI (typically mid-month)
        for month in range(1, 13):
            cpi_date = f"{year}-{month:02d}-15"
            self.add_calendar_event(f"CPI Release", cpi_date, "Bonsai Fund Seed", "KXCPICPI-??")

        self._conn.commit()

    # ---------------------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------------------

    def status(self) -> dict:
        cursor = self._conn.execute("""
            SELECT category, COUNT(*) as cnt FROM news_items
            WHERE archived=0 GROUP BY category
        """)
        by_cat = {cat: cnt for cat, cnt in cursor.fetchall()}

        cursor = self._conn.execute("SELECT COUNT(*) FROM news_items WHERE archived=0")
        total = cursor.fetchone()[0]

        cursor = self._conn.execute(
            "SELECT ingested_at FROM news_items ORDER BY ingested_at DESC LIMIT 1"
        )
        last_fetch = cursor.fetchone()[0] if cursor.fetchone() else "never"

        return {
            "total_items": total,
            "by_category": by_cat,
            "last_fetch": last_fetch,
            "contexts_cached": len(self._session_cache),
        }
