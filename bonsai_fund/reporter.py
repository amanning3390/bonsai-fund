"""
Bonsai Fund — Reporter
Board Report generator and Telegram formatting.
"""

from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
def _cfg(key, default=None):
    """Lazy config loader — reads YAML, no circular import."""
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
class BoardReport:
    period_start: str
    period_end: str
    bankroll: float
    starting_capital: float
    total_pnl: float
    total_return_pct: float
    open_positions: int
    closed_positions: int
    win_rate: float
    sharpe_approx: float
    drawdown_pct: float
    peak_bankroll: float
    trades: list
    new_signals: list
    top_position: Optional[dict]
    alerts: list
    board_notes: str


class Reporter:
    def __init__(self, portfolio, risk_engine):
        self.portfolio = portfolio
        self.risk = risk_engine

    def generate_board_report(self, days: int = 7) -> BoardReport:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)
        self.risk.update_peak()
        summary = self.portfolio.summary()

        db_path = _cfg("db_path", str(Path.home() / ".hermes" / "bonsai_fund_data" / "bonsai_portfolio.db"))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT ticker, side, contracts, price, pnl, closed_at, voted_by "
            "FROM trades WHERE closed_at IS NOT NULL AND closed_at >= ? ORDER BY closed_at",
            (start.isoformat(),))
        trade_rows = cursor.fetchall()

        wins = sum(1 for r in trade_rows if r[4] and r[4] > 0)
        losses = sum(1 for r in trade_rows if r[4] and r[4] < 0)

        open_pos = sorted(summary.get("positions", []),
                          key=lambda x: abs(x.get("unrealized_pnl", 0)), reverse=True)
        top_position = open_pos[0] if open_pos else None

        new_signals = []
        try:
            votes_path = _cfg("votes_db_path", str(Path.home() / ".hermes" / "bonsai_fund_data" / "bonsai_votes.db"))
            Path(votes_path).parent.mkdir(parents=True, exist_ok=True)
            vc = sqlite3.connect(votes_path)
            for row in vc.execute(
                "SELECT ticker, COUNT(*) FROM votes WHERE timestamp >= ? GROUP BY ticker ORDER BY COUNT(*) DESC LIMIT 5",
                (start.isoformat(),)):
                new_signals.append({"ticker": row[0], "votes": row[1]})
            vc.close()
        except:
            pass

        alerts = self._generate_alerts(summary)
        board_notes = self._generate_board_notes(summary, wins, losses)

        return BoardReport(
            period_start=start.strftime("%Y-%m-%d"),
            period_end=now.strftime("%Y-%m-%d"),
            bankroll=summary["bankroll"],
            starting_capital=summary["starting_capital"],
            total_pnl=summary["total_pnl"],
            total_return_pct=summary["total_return_pct"],
            open_positions=summary["open_positions"],
            closed_positions=summary["closed_positions"],
            win_rate=summary["win_rate"],
            sharpe_approx=summary["sharpe_approx"],
            drawdown_pct=round(self.risk.current_drawdown * 100, 2),
            peak_bankroll=self.risk.peak_bankroll,
            trades=[{"ticker": r[0], "side": r[1], "contracts": r[2],
                     "price": r[3], "pnl": r[4], "closed_at": r[5]}
                    for r in trade_rows],
            new_signals=new_signals,
            top_position=top_position,
            alerts=alerts,
            board_notes=board_notes,
        )

    def _generate_alerts(self, summary: dict) -> list:
        alerts = []
        if self.risk.is_circuit_breaker:
            alerts.append(("CIRCUIT_BREAKER",
                         f"Drawdown {self.risk.current_drawdown*100:.1f}% — trading halted"))
        elif self.risk.current_drawdown > 0.10:
            alerts.append(("DRAWDOWN_ALERT",
                         f"Current drawdown {self.risk.current_drawdown*100:.1f}% from peak"))
        if summary["open_positions"] == 0 and summary["closed_positions"] == 0:
            alerts.append(("NO_ACTIVITY", "No trades placed this period"))
        return alerts

    def _generate_board_notes(self, summary: dict, wins: int, losses: int) -> str:
        pnl = summary["total_pnl"]
        ret = summary["total_return_pct"]
        notes = []
        if pnl > 0:
            notes.append(f"The fund is up ${pnl:.2f} ({ret:.1f}%) since inception.")
        elif pnl < 0:
            notes.append(f"The fund is down ${abs(pnl):.2f} ({ret:.1f}%). The swarm continues to learn.")
        else:
            notes.append("The fund is flat — building the track record before scaling.")
        if wins + losses > 0:
            wr = wins / (wins + losses) * 100
            notes.append(f"Closed {wins + losses} trades ({wins}W/{losses}L, {wr:.0f}% win rate).")
        notes.append("The Bonsai Swarm (7 specialized agents, local inference) operates fully autonomously.")
        notes.append(f"Risk rules: Kelly sizing, 5% max position, 25% drawdown circuit breaker.")
        return " ".join(notes)

    def format_telegram_board_report(self, report: BoardReport) -> str:
        bank_emoji = "🟢" if report.total_pnl >= 0 else "🔴"
        dd_emoji = "🟢" if report.drawdown_pct < 10 else "🟡" if report.drawdown_pct < 20 else "🔴"

        lines = [
            "📊 BOARD REPORT — Bonsai Hedge Fund",
            f"Period: {report.period_start} → {report.period_end}",
            "=" * 42,
            "💰 PERFORMANCE",
            f"  Bankroll:      ${report.bankroll:.4f}",
            f"  Starting:      ${report.starting_capital:.2f}",
            f"  P&L:           {bank_emoji} ${report.total_pnl:+.4f} ({report.total_return_pct:+.2f}%)",
            f"  Peak:          ${report.peak_bankroll:.4f}",
            f"  Drawdown:      {dd_emoji} {report.drawdown_pct:.2f}%",
            "",
            "📈 TRADING",
            f"  Open:          {report.open_positions} positions",
            f"  Closed:        {report.closed_positions} trades",
            f"  Win Rate:      {report.win_rate:.1f}%",
            f"  Sharpe (approx): {report.sharpe_approx:.2f}",
            "",
        ]

        if report.top_position:
            p = report.top_position
            lines += [
                "📌 TOP POSITION",
                f"  {p['ticker']} — {p['side']} x{p['contracts']}",
                f"  Entry: {p['avg_price']:.0f}c  Mkt: {p['market_price']:.0f}c",
                f"  Unrealized: {p['unrealized_pnl']:+.4f}",
                "",
            ]

        if report.trades:
            lines.append("📋 RECENT TRADES:")
            for t in report.trades[-5:]:
                pnl_str = f"+${t['pnl']:.2f}" if t['pnl'] >= 0 else f"-${abs(t['pnl']):.2f}"
                lines.append(f"  {t['ticker'][:35]:35s} {t['side']:4s} x{t['contracts']} @ {t['price']:.0f}c  {pnl_str}")
            lines.append("")

        if report.new_signals:
            lines.append("🔮 SCANNED MARKETS:")
            for s in report.new_signals:
                lines.append(f"  {s['ticker'][:40]} — {s['votes']} votes")
            lines.append("")

        for alert_cat, alert_text in report.alerts:
            emoji = {"CIRCUIT_BREAKER": "🔴", "DRAWDOWN_ALERT": "🟡", "NO_ACTIVITY": "⚪"}.get(alert_cat, "ℹ️")
            lines.append(f"{emoji} {alert_cat}: {alert_text}")

        lines += [
            "",
            "=" * 42,
            f"BOARD NOTES:\n{report.board_notes}",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "Bonsai Hedge Fund — Fully Autonomous",
        ]
        return "\n".join(lines)

    def save_report_json(self, report: BoardReport, path: Path = None):
        path = path or _cfg("report_path", str(Path.home() / ".hermes" / "bonsai_fund_data" / "bonsai_board_report_latest.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "report_type": "board_report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {"start": report.period_start, "end": report.period_end},
            "performance": {
                "bankroll": report.bankroll,
                "starting_capital": report.starting_capital,
                "total_pnl": report.total_pnl,
                "total_return_pct": report.total_return_pct,
                "peak_bankroll": report.peak_bankroll,
                "drawdown_pct": report.drawdown_pct,
            },
            "trading": {
                "open_positions": report.open_positions,
                "closed_positions": report.closed_positions,
                "win_rate": report.win_rate,
                "sharpe_approx": report.sharpe_approx,
            },
            "top_position": report.top_position,
            "trades": report.trades,
            "alerts": [{"category": a, "message": m} for a, m in report.alerts],
            "board_notes": report.board_notes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
