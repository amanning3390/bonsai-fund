"""
Bonsai Fund — Portfolio
Position tracking, realized/unrealized P&L, equity curve.
"""

from __future__ import annotations
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _cfg(key, default):
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
class Position:
    ticker: str
    side: str           # "YES" or "NO"
    contracts: int
    avg_fill_price: float   # cents per contract
    market_price: float     # current mark-to-market price (cents)
    realized_pnl: float = 0.0
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    closed_at: Optional[str] = None

    @property
    def cost(self) -> float:
        return self.contracts * self.avg_fill_price / 100.0

    @property
    def market_value(self) -> float:
        return self.contracts * self.market_price / 100.0

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "YES":
            return (self.market_price - self.avg_fill_price) * self.contracts / 100.0
        return (self.avg_fill_price - self.market_price) * self.contracts / 100.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class Portfolio:
    def __init__(self, starting_capital: float = None):
        self.starting_capital = starting_capital or _cfg("bankroll", 50.0)
        self.positions: dict[str, Position] = {}
        self.closed: list[Position] = []
        self._conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        db_path = _cfg("db_path", str(Path.home() / ".hermes" / "bonsai_fund_data" / "bonsai_portfolio.db"))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT, side TEXT, contracts INTEGER, avg_fill_price REAL,
                market_price REAL DEFAULT 0, realized_pnl REAL DEFAULT 0,
                opened_at TEXT, closed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp TEXT PRIMARY KEY, bankroll REAL, open_pnl REAL,
                closed_pnl REAL, position_count INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY, ticker TEXT, side TEXT, contracts INTEGER,
                price REAL, pnl REAL, closed_at TEXT, voted_by TEXT
            )
        """)
        conn.commit()
        return conn

    @property
    def open_positions(self) -> dict[str, Position]:
        return {k: v for k, v in self.positions.items() if v.closed_at is None}

    @property
    def bankroll(self) -> float:
        closed_pnl = sum(p.realized_pnl for p in self.closed)
        open_pnl   = sum(p.unrealized_pnl for p in self.open_positions.values())
        return self.starting_capital + closed_pnl + open_pnl

    @property
    def total_pnl(self) -> float:
        return self.bankroll - self.starting_capital

    @property
    def total_return_pct(self) -> float:
        return (self.total_pnl / self.starting_capital) * 100

    def open_position(self, ticker: str, side: str, contracts: int,
                     fill_price: float, voted_by: str = "") -> Position:
        key = f"{ticker}:{side}"
        if key in self.positions and self.positions[key].closed_at is None:
            pos = self.positions[key]
            total_cost = pos.avg_fill_price * pos.contracts + fill_price * contracts
            pos.contracts += contracts
            pos.avg_fill_price = total_cost / pos.contracts
            self._log_trade(ticker, side, contracts, fill_price, 0.0, voted_by)
            return pos
        pos = Position(ticker=ticker, side=side, contracts=contracts,
                       avg_fill_price=fill_price, market_price=fill_price)
        self.positions[key] = pos
        self._log_trade(ticker, side, contracts, fill_price, 0.0, voted_by)
        return pos

    def close_position(self, ticker: str, side: str, contracts: int,
                       fill_price: float, pnl: float) -> None:
        key = f"{ticker}:{side}"
        if key not in self.positions:
            return
        pos = self.positions[key]
        pos.contracts -= contracts
        pos.realized_pnl += pnl
        self._log_trade(ticker, "CLOSE", contracts, fill_price, pnl, "")
        if pos.contracts <= 0:
            pos.closed_at = datetime.now(timezone.utc).isoformat()
            self.closed.append(pos)
            del self.positions[key]

    def mark_to_market(self, ticker: str, side: str, current_price: float) -> None:
        key = f"{ticker}:{side}"
        if key in self.positions:
            self.positions[key].market_price = current_price

    def get_position(self, ticker: str, side: str) -> Optional[Position]:
        return self.positions.get(f"{ticker}:{side}")

    def _log_trade(self, ticker: str, side: str, contracts: int,
                    price: float, pnl: float, voted_by: str):
        self._conn.execute("""
            INSERT INTO trades (ticker, side, contracts, price, pnl, closed_at, voted_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, side, contracts, price, pnl,
              datetime.now(timezone.utc).isoformat() if side == "CLOSE" else None, voted_by))
        self._conn.commit()

    def record_equity(self):
        open_pnl   = sum(p.unrealized_pnl for p in self.open_positions.values())
        closed_pnl = sum(p.realized_pnl for p in self.closed)
        self._conn.execute("""
            INSERT OR REPLACE INTO equity_curve
            (timestamp, bankroll, open_pnl, closed_pnl, position_count)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), round(self.bankroll, 4),
              round(open_pnl, 4), round(closed_pnl, 4), len(self.open_positions)))
        self._conn.commit()

    def win_rate(self) -> float:
        closed = [p for p in self.closed if p.realized_pnl != 0]
        if not closed:
            return 0.0
        wins = sum(1 for p in closed if p.realized_pnl > 0)
        return wins / len(closed)

    def sharpe(self) -> float:
        cursor = self._conn.execute(
            "SELECT bankroll FROM equity_curve ORDER BY timestamp")
        rows = cursor.fetchall()
        if len(rows) < 3:
            return 0.0
        rets = [(rows[i][0] - rows[i-1][0]) / rows[i-1][0]
                for i in range(1, len(rows)) if rows[i-1][0] != 0]
        if not rets:
            return 0.0
        import statistics
        mean_ret = statistics.mean(rets)
        std_ret  = statistics.stdev(rets) if len(rets) > 1 else 0.0
        if std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * (365 ** 0.5)

    def summary(self) -> dict:
        return {
            "bankroll": round(self.bankroll, 4),
            "starting_capital": self.starting_capital,
            "total_pnl": round(self.total_pnl, 4),
            "total_return_pct": round(self.total_return_pct, 2),
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed),
            "win_rate": round(self.win_rate() * 100, 1),
            "sharpe_approx": round(self.sharpe(), 2),
            "positions": [
                {
                    "ticker": p.ticker, "side": p.side,
                    "contracts": p.contracts, "avg_price": p.avg_fill_price,
                    "market_price": p.market_price,
                    "unrealized_pnl": round(p.unrealized_pnl, 4),
                    "realized_pnl": round(p.realized_pnl, 4),
                }
                for p in self.open_positions.values()
            ],
        }

    def reset_to_cash(self):
        for key, pos in list(self.positions.items()):
            self.close_position(pos.ticker, pos.side, pos.contracts,
                               pos.market_price, pos.unrealized_pnl)
