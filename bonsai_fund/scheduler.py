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
from bonsai_fund.self_learning.orchestrator import LearningOrchestrator
from bonsai_fund.self_learning.outcome_tracker import OutcomeTracker
from bonsai_fund.drawdown_response import DrawdownResponseOrchestrator
from bonsai_fund.simulation import SimulationEngine
from bonsai_fund.news_pipeline import NewsPipeline


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
    # New: staged drawdown fields
    last_stage: str = "GREEN"
    simulations_run: int = 0
    evolutions_run: int = 0
    last_learning_at: str = ""

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
            last_stage="GREEN", simulations_run=0, evolutions_run=0, last_learning_at="",
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

def run_scan(state: State, learning=None):
    """
    Run a full market scan cycle with full recursive self-learning stack:
      - NewsPipeline: fetch + cache relevant headlines before scanning
      - StagedDrawdownMonitor: 6-tier drawdown response
      - DrawdownResponseOrchestrator: fires ANALYSIS / SIMULATION / EVOLUTION
      - SimulationEngine: adversarial replay + synthetic market generation
      - LearningOrchestrator: context-sensitive weights + evolutionary mutation

    All trading generates learning data. All learning improves trading.
    """
    from bonsai_fund.self_learning.orchestrator import LearningOrchestrator
    learning = learning or LearningOrchestrator()

    portfolio = Portfolio()
    risk      = RiskEngine(portfolio, RiskLimits())
    reporter  = Reporter(portfolio, risk)

    # --- NEW: Wire the full stack ---
    news = NewsPipeline()
    sim  = SimulationEngine()
    outcomes = learning.outcome_tracker  # OutcomeTracker lives on LearningOrchestrator

    # Drawdown response orchestrator — watches the monitor and fires training
    def _alert(msg: str):
        send_telegram(msg)

    orchestrator = DrawdownResponseOrchestrator(
        risk_engine=risk,
        simulation_engine=sim,
        learning_orchestrator=learning,
        news_pipeline=news,
        outcome_tracker=outcomes,
        telegram_alert_fn=_alert,
    )

    # Inject Bonsai caller so orchestrator can run simulations in background
    from bonsai_fund.hedge_fund import call_bonsai
    orchestrator.configure_bonsai_caller(call_bonsai)

    # --- STEP 1: Check drawdown stage (before scanning) ---
    old_stage, new_stage, changed = risk.drawdown_monitor.check()
    if changed:
        state.last_stage = new_stage.name
        state.save()
        meta = {
            "GREEN": "🟢 GREEN — normal trading",
            "YELLOW": "🟡 YELLOW — reduced sizing",
            "ORANGE": "🟠 ORANGE — analysis triggered",
            "RED": "🔴 RED — simulation required",
            "CRITICAL": "🚨 CRITICAL — evolution cycle",
            "FROZEN": "❄️ FROZEN — full reset",
        }.get(new_stage.name, new_stage.name)
        state.log(f"DRAWDOWN STAGE: {meta} ({risk.drawdown_monitor.drawdown_pct*100:.1f}%)")

    # Ask orchestrator if we're blocked
    blocked, block_reason = orchestrator.check()
    if blocked:
        state.log(f"BLOCKED: {block_reason}")
        if "CIRCUIT BREAKER" not in block_reason and new_stage.name != "GREEN":
            # Don't return empty — still record equity
            portfolio.record_equity()
            return []

    # --- STEP 2: Fetch news for weak categories (medium-term learning) ---
    if orchestrator._pending_action and orchestrator._pending_action.action_type in ("analyze", "simulate"):
        # We're in a training phase — fetch news for weak categories
        try:
            analysis = outcomes.get_recent_losses(limit=5)
            weak_cats = list({r.get("category", "general") for r in analysis})
            if weak_cats:
                news.fetch_rss_feeds(categories=weak_cats[:3])
        except Exception:
            pass

    # --- STEP 3: Fetch + prep markets ---
    markets = fetch_markets(limit=40)
    if len(markets) <= 1:
        markets = sample_markets()

    # --- STEP 4: Pre-build news context for all markets ---
    market_contexts = {}
    for mkt in markets:
        ctx = news.get_context_for_market(mkt)
        market_contexts[mkt.ticker] = ctx

    state.log(f"Scanning {len(markets)} markets | Stage: {state.last_stage} | "
              f"Gen {learning.evolver._generation} | "
              f"{outcomes.get_total_edge_analysis()['total_trades']} trades in memory")

    signals, approved, all_votes = [], 0, []

    for mkt in markets:
        # STEP 5: Get context-enhanced votes
        votes = collect_votes(mkt, learning=learning)
        all_votes.append(votes)

        # Apply context-sensitive weights
        weighted = learning.get_weighted_votes(votes, mkt, mkt.days_to_event)
        sig = analyze(mkt, votes, portfolio, risk, weighted_votes=weighted)

        signals.append(sig)

        # STEP 6: Execute (paper or live)
        if sig.approved:
            approved += 1
            side = sig.majority
            price = sig.yes_cents if side == "YES" else sig.no_cents
            portfolio.open_position(mkt.ticker, side, sig.contracts, price, voted_by="bonsai_swarm")
            state.total_trades += 1
            risk.drawdown_monitor.record_trade()  # track trades per stage

            if _cfg("alerts_enabled", False):
                edge_pct = sig.avg_edge * 100
                emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "WEAK_BUY": "🟡"}.get(sig.grade, "⚪")
                msg = (f"{emoji} Bonsai Signal\n{sig.ticker}: {side} x{sig.contracts} @ {price:.0f}¢\n"
                       f"Edge: {edge_pct:+.0%}  Conf: {sig.avg_conf:.0%}  Grade: {sig.grade}")
                send_telegram(msg)

    # STEP 7: SHORT-TERM LEARNING — record votes for context-sensitive weighting
    learning.record_scan_votes(markets, all_votes, portfolio)

    # STEP 8: Update state from orchestrator
    if orchestrator._pending_action:
        at = orchestrator._pending_action
        if at.action_type == "simulate":
            state.simulations_run += 1
        elif at.action_type == "evolve":
            state.evolutions_run += 1
    state.last_learning_at = datetime.now(timezone.utc).isoformat()

    portfolio.record_equity()
    state.last_scan_at = datetime.now(timezone.utc).isoformat()
    state.total_scans += 1
    state.save()

    # STEP 9: Periodic baseline simulation (every 50 scans)
    if state.total_scans % 50 == 0 and learning:
        state.log("Triggering periodic baseline simulation...")
        try:
            result = sim.run_baseline_simulation(
                candidate_prompts=learning.evolver.get_current_prompts(),
                call_bonsai_fn=call_bonsai,
            )
            state.log(f"Baseline sim: WR={result['candidate_win_rate']:.1%} | "
                      f"Gen {learning.evolver._generation}")
        except Exception as e:
            state.log(f"Baseline sim error: {e}")

    st = risk.status()
    state.log(
        f"Scan #{state.total_scans}: {len(markets)} markets, {approved} approved. "
        f"Bankroll=${st['bankroll']:.4f}, DD={st['drawdown_pct']:.1f}% [{st['stage']}]"
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

def daemon_loop(state: State, stop: Event, learning=None):
    from bonsai_fund.self_learning.orchestrator import LearningOrchestrator
    learning = learning or LearningOrchestrator()
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

    res_p = sub.add_parser("resolve")
    res_p.add_argument("ticker")
    res_p.add_argument("resolution")

    learn_p = sub.add_parser("learn")
    learn_p.add_argument("--status", action="store_true")
    learn_p.add_argument("--evolve", action="store_true")

    sim_p = sub.add_parser("simulate")
    sim_p.add_argument("--stage", default="ORANGE", choices=["ORANGE", "RED", "CRITICAL"])
    sim_p.add_argument("--force", action="store_true", help="Force even if not in that stage")

    args = parser.parse_args()
    state = State.load()

    # Learning orchestrator is shared across all commands
    learning = None
    if getattr(args, "cmd", None) in ("once", "run", "resolve", "learn"):
        from bonsai_fund.self_learning.orchestrator import LearningOrchestrator
        learning = LearningOrchestrator()

    if args.cmd == "once":
        state.log("=== Manual scan ===")
        run_scan(state, learning=learning)

    elif args.cmd == "board-report":
        state.log("=== Manual Board Report ===")
        run_board_report(state)

    elif args.cmd == "status":
        p = Portfolio(); r = RiskEngine(p, RiskLimits()); st = r.status()
        dm_st = r.drawdown_monitor.status()
        learning = LearningOrchestrator()
        lo_st = learning.get_learning_status()
        print(f"""
BONSAI SCHEDULER STATUS
=======================
Started:         {state.started_at}
Last scan:       {state.last_scan_at or 'never'}
Last report:     {state.last_board_report_at or 'never'}
Total scans:     {state.total_scans}
Total trades:    {state.total_trades}
CB tripped:      {state.cb_tripped_at or 'never'}
Simulations:     {state.simulations_run}
Evolutions:      {state.evolutions_run}
Last learning:   {state.last_learning_at or 'never'}
Last stage:      {state.last_stage}

FUND STATUS
-----------
Bankroll:        ${st['bankroll']:.4f}
Peak:            ${st['peak']:.4f}
Drawdown:        {st['drawdown_pct']:.2f}%
Stage:           {st['stage_emoji']} {st['stage']}
Can trade:       {'YES' if st['can_trade'] else 'NO'}
  eff. pos max: {st['effective_limits']['max_position_pct']:.1f}% of bankroll
  min conf:     {st['effective_limits']['min_confidence']:.0f}%
  min margin:   {st['effective_limits']['min_vote_margin']} votes
Simulations:    {dm_st['simulations_run']}
Evolutions:     {dm_st['evolutions_run']}

LEARNING STATUS
---------------
Generation:      {lo_st['evolver_generation']}
Total trades:    {lo_st['total_trades_recorded']}
Edge captured:   {lo_st['system_edge_captured']:+.4f}
Active:         {lo_st['learning_active']}
""")

    elif args.cmd == "resolve":
        ticker = args.ticker.upper()
        resolution = args.resolution.upper()
        markets = sample_markets()
        mkt = next((m for m in markets if m.ticker == ticker), None)
        if not mkt:
            print(f"Unknown ticker: {ticker}")
            sys.exit(1)
        result = learning.process_resolution(ticker, resolution, mkt)
        if result:
            print(f"🧬 Evolution triggered: {result}")
        else:
            print(f"✅ Resolution recorded: {ticker} = {resolution}")

    elif args.cmd == "learn":
        if getattr(args, "status", False):
            print(learning.format_learning_report())
        elif getattr(args, "evolve", False):
            result = learning.run_evolution_cycle()
            print(f"🧬 Evolution: {result}")
        else:
            print(learning.format_learning_report())

    elif args.cmd == "simulate":
        from bonsai_fund.hedge_fund import call_bonsai
        learning = LearningOrchestrator()
        outcomes = learning.outcome_tracker
        sim = SimulationEngine()
        portfolio = Portfolio()
        risk = RiskEngine(portfolio, RiskLimits())

        # Wire orchestrator just to get the Bonsai caller
        orch = DrawdownResponseOrchestrator(
            risk_engine=risk,
            simulation_engine=sim,
            learning_orchestrator=learning,
            news_pipeline=NewsPipeline(),
            outcome_tracker=outcomes,
        )
        orch.configure_bonsai_caller(call_bonsai)

        stage = getattr(args, "stage", "ORANGE")
        print(f"Running {stage} simulation...")
        result = sim.simulate_from_drawdown(
            learning=learning,
            outcome_tracker=outcomes,
            call_bonsai_fn=call_bonsai,
            stage=stage,
        )
        print(f"\n{'='*60}")
        print(f"  SIMULATION RESULT — {stage}")
        print(f"{'='*60}")
        print(f"  Weak categories: {result.get('weak_categories', {})}")
        print(f"  Markets tested: {result.get('markets_tested', 0)}")
        print(f"  Original win rate:  {result.get('original_win_rate', 0):.1%}")
        print(f"  Candidate win rate: {result.get('candidate_win_rate', 0):.1%}")
        print(f"  Improvement:        {result.get('improvement_pct', 0):+.1%}")
        print(f"  Passes threshold:   {'YES' if result.get('passes_threshold') else 'NO'}")
        print(f"  Deploy triggered:   {'YES' if result.get('deploy_triggered') else 'NO'}")
        print(f"  Generation:        {result.get('generation_before', 0)} → {result.get('generation_after', 0)}")
        print(f"\n  Per-agent:")
        for aid, data in result.get("per_agent", {}).items():
            print(f"    [{aid}] {data.get('name', ''):20s} "
                  f"WR: {data.get('original_wr', 0):.0%} → {data.get('candidate_wr', 0):.0%} "
                  f"({data.get('improvement', 0):+.0%})")
        print()

    elif args.cmd == "run":
        if args.daemon:
            stop = Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stop.set())
            daemon_loop(state, stop, learning=learning)
        else:
            state.log("=== Scheduler one-shot ===")
            run_scan(state, learning=learning)

    else:
        parser.print_help()
