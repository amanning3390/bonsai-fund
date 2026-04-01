#!/usr/bin/env python3
"""
Bonsai Fund — Autonomous Scheduler
Runs the fund continuously: periodic scans + Sunday 8am Board Reports.
Designed to be called from cron OR run as a daemon.

Usage:
    python3 bonsai_fund/scheduler.py once        # single scan cycle
    python3 bonsai_fund/scheduler.py board-report # send Board Report now
    python3 bonsai_fund/scheduler.py status     # scheduler state
    python3 bonsai_fund/scheduler.py run --daemon # background daemon
"""

from __future__ import annotations
import json, os, sys, signal, time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from dataclasses import dataclass, asdict

# Resolve skill directory
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

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

from bonsai_fund.portfolio import Portfolio
from bonsai_fund.risk import RiskEngine, RiskLimits
from bonsai_fund.reporter import Reporter
from bonsai_fund.hedge_fund import (
    fetch_markets, sample_markets, collect_votes, analyze,
    send_telegram, print_signal,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class State:
    started_at: str
    last_scan_at: str
    last_board_report_at: str
    total_scans: int
    total_trades: int
    cb_tripped_at: str | None

    @staticmethod
    def _state_path() -> Path:
        return Path(_cfg("state_file", str(Path.home() / ".hermes" / "bonsai_fund_data" / "bonsai_scheduler_state.json")))

    @staticmethod
    def _data_dir() -> Path:
        return Path(_cfg("data_dir", str(Path.home() / ".hermes" / "bonsai_fund_data")))

    @staticmethod
    def _log_dir() -> Path:
        return Path(_cfg("log_dir", str(Path.home() / ".hermes" / "bonsai_fund_data" / "logs")))

    @staticmethod
    def load() -> "State":
        sp = State._state_path()
        if sp.exists():
            try:
                return State(**json.loads(sp.read_text()))
            except:
                pass
        return State(
            started_at=datetime.now(timezone.utc).isoformat(),
            last_scan_at="", last_board_report_at="",
            total_scans=0, total_trades=0, cb_tripped_at=None,
        )

    def save(self):
        self._data_dir().mkdir(parents=True, exist_ok=True)
        self._state_path().write_text(json.dumps(asdict(self), indent=2))

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        print(line)
        log_dir = self._log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "bonsai.log"
        log_file.write_text((log_file.read_text() if log_file.exists() else "") + line + "\n")

    def should_scan(self) -> bool:
        if not self.last_scan_at:
            return True
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(self.last_scan_at)).total_seconds() / 60
        return elapsed >= _cfg("scan_interval_min", 30)

    def is_board_report_due(self) -> bool:
        now = datetime.now(timezone.utc)
        if now.weekday() != _cfg("board_report_day", 6):
            return False
        h, m = now.hour, now.minute
        return (h == _cfg("board_report_hour", 8) - 1 and 50 <= m <= 59) or (h == _cfg("board_report_hour", 8) and m <= 10)


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------

def run_scan(state: State):
    portfolio = Portfolio()
    risk      = RiskEngine(portfolio, RiskLimits())
    reporter  = Reporter(portfolio, risk)

    tripped, reason = risk.check_circuit_breaker()
    if tripped:
        state.log(f"CIRCUIT_BREAKER: {reason}")
        if not state.cb_tripped_at:
            state.cb_tripped_at = datetime.now(timezone.utc).isoformat()
            state.save()
            send_telegram(f"🚨 CIRCUIT BREAKER: {reason}\nBonsai Fund halted.")
        return []

    markets = fetch_markets(limit=40)
    if len(markets) <= 1:
        markets = sample_markets()

    state.log(f"Scanning {len(markets)} markets...")
    signals, approved = [], 0

    for mkt in markets:
        votes = collect_votes(mkt)
        sig   = analyze(mkt, votes, portfolio, risk)
        signals.append(sig)
        if sig.approved:
            approved += 1
            side = sig.majority
            price = sig.yes_cents if side == "YES" else sig.no_cents
            portfolio.open_position(mkt.ticker, side, sig.contracts, price, voted_by="bonsai_swarm")
            state.total_trades += 1
            if _cfg("alerts_enabled", False):
                edge_pct = sig.avg_edge * 100
                emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "WEAK_BUY": "🟡"}.get(sig.grade, "⚪")
                msg = (f"{emoji} Bonsai Signal\n{sig.ticker}: {side} x{sig.contracts} @ {price:.0f}¢\n"
                       f"Edge: {edge_pct:+.0%}  Conf: {sig.avg_conf:.0%}  Grade: {sig.grade}")
                send_telegram(msg)

    portfolio.record_equity()
    state.last_scan_at = datetime.now(timezone.utc).isoformat()
    state.total_scans += 1
    state.save()

    st = risk.status()
    state.log(
        f"Scan #{state.total_scans}: {len(markets)} markets, {approved} approved. "
        f"Bankroll=${st['bankroll']:.4f}, DD={st['drawdown_pct']:.1f}%"
    )
    return signals


def run_board_report(state: State):
    portfolio = Portfolio()
    risk      = RiskEngine(portfolio, RiskLimits())
    reporter  = Reporter(portfolio, risk)
    report    = reporter.generate_board_report(days=7)
    text      = reporter.format_telegram_board_report(report)
    reporter.save_report_json(report)
    send_telegram(text)
    state.last_board_report_at = datetime.now(timezone.utc).isoformat()
    state.save()
    state.log("Board Report sent.")


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def daemon_loop(state: State, stop: Event):
    scan_int = _cfg("scan_interval_min", 30)
    state.log(f"Daemon started PID={os.getpid()} interval={scan_int}min")
    data_dir = Path(_cfg("data_dir", str(Path.home() / ".hermes" / "bonsai_fund_data")))
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "bonsai_scheduler_state.json").write_text(json.dumps(asdict(state)))

    while not stop.is_set():
        try:
            if state.is_board_report_due():
                state.log("Triggering Sunday 8am Board Report")
                run_board_report(state)
            elif state.should_scan():
                run_scan(state)
        except Exception as e:
            state.log(f"ERROR: {e}")
            import traceback; state.log(traceback.format_exc())
        stop.wait(timeout=300)

    state.log("Daemon stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bonsai Fund Scheduler")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("once")           # single scan
    sub.add_parser("board-report")  # Board Report now
    sub.add_parser("status")

    run_p = sub.add_parser("run")
    run_p.add_argument("--daemon", action="store_true")

    args = parser.parse_args()
    state = State.load()

    if args.cmd == "once":
        state.log("=== Manual scan ===")
        run_scan(state)

    elif args.cmd == "board-report":
        state.log("=== Manual Board Report ===")
        run_board_report(state)

    elif args.cmd == "status":
        print(f"""
BONSAI SCHEDULER STATUS
=======================
Started:         {state.started_at}
Last scan:       {state.last_scan_at or 'never'}
Last report:     {state.last_board_report_at or 'never'}
Total scans:     {state.total_scans}
Total trades:    {state.total_trades}
CB tripped:      {state.cb_tripped_at or 'never'}
""")
        p = Portfolio(); r = RiskEngine(p, RiskLimits()); st = r.status()
        print(f"FUND: Bankroll=${st['bankroll']:.4f} DD={st['drawdown_pct']:.1f}% CB={'TRIPPED' if st['circuit_breaker'] else 'clear'}")

    elif args.cmd == "run":
        if args.daemon:
            stop = Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stop.set())
            daemon_loop(state, stop)
        else:
            state.log("=== Scheduler one-shot ===")
            run_scan(state)

    else:
        parser.print_help()
