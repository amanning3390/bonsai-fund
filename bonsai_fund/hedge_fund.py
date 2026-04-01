#!/usr/bin/env python3
"""
Bonsai Fund — Hedge Fund Orchestrator
Main entry point: scan → vote → size → execute (paper or live).
"""

from __future__ import annotations
import json, os, sys, time, argparse
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from bonsai_fund.agent import (
    Market, AgentVote, AGENT_NAMES,
    SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES, build_user_prompt,
    PORT_BASE, MODEL,
)
from bonsai_fund.portfolio import Portfolio
from bonsai_fund.risk import RiskEngine, RiskLimits
from bonsai_fund.reporter import Reporter
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


PAPER_MODE = _cfg("paper_mode", True)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def call_bonsai(port: int, system: str, user: str) -> dict:
    import urllib.request
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "top_k": 20,
            "top_p": 0.9,
            "max_tokens": 80,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"text": json.loads(resp.read())["choices"][0]["message"]["content"],
                    "elapsed": time.time() - t0}
    except Exception as e:
        return {"text": "", "elapsed": time.time() - t0, "error": str(e)}


def parse_vote(agent_id: int, ticker: str, raw: str, elapsed: float) -> AgentVote:
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        j = json.loads(raw[s:e])
        reason = str(j.get("reason", "")).replace("\n", " ").replace("\\n", " ").strip()[:80]
        return AgentVote(agent_id, AGENT_NAMES[agent_id], ticker,
                         j.get("vote", "PASS"), float(j.get("confidence", 0.5)),
                         float(j.get("edge", 0.0)), reason, elapsed * 1000)
    except:
        t = raw.upper()
        vote = "PASS" if "PASS" in t else ("YES" if "YES" in t[:10] else ("NO" if "NO" in t[:10] else "PASS"))
        return AgentVote(agent_id, AGENT_NAMES[agent_id], ticker, vote, 0.3, 0.0, "parse_error", elapsed * 1000)


def true_edge_yes(vote: str, raw_edge: float, market_prob: float) -> float:
    """Convert vote-specific edge to YES-directional edge."""
    return raw_edge if vote == "YES" else -raw_edge


def collect_votes(market: Market) -> list[AgentVote]:
    votes, primary = [], []

    def final_context(pv):
        return "\n".join(
            f"- {v.agent_name}: vote={v.vote}, conf={v.confidence:.2f}, edge={v.edge:+.2f} | {v.reason}"
            for v in pv
        )

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {
            ex.submit(call_bonsai, PORT_BASE + i, SYSTEM_PROMPTS[i], build_user_prompt(i, market)): i
            for i in range(6)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            v = parse_vote(i, market.ticker, r.get("text", ""), r.get("elapsed", 0))
            votes.append(v); primary.append(v)

    # FinalVote (agent 6) — gets full context of agents 0-5
    ctx = final_context(primary)
    sys6 = SYSTEM_PROMPTS[6].replace(
        "Other votes received:\n- FastIntuit: YES, conf=0.80\n...",
        f"Other votes received:\n{ctx}"
    )
    r6 = call_bonsai(PORT_BASE + 6, sys6, build_user_prompt(6, market))
    votes.append(parse_vote(6, market.ticker, r6.get("text", ""), r6.get("elapsed", 0)))
    return votes


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    ticker: str
    market_title: str
    yes_cents: int
    no_cents: int
    volume: int
    days_out: int
    market_prob: float
    est_prob: float
    avg_edge: float
    yes_votes: int
    no_votes: int
    pass_votes: int
    margin: int
    avg_conf: float
    ci: float          # confidence index = avg_conf × (margin/7)
    majority: str
    approved: bool
    grade: str
    reason: str
    contracts: int
    per_agent: list
    wall_time: float


def analyze(market: Market, votes: list[AgentVote],
            portfolio: Portfolio, risk: RiskEngine) -> Signal:
    yes_v = [v for v in votes if v.vote == "YES"]
    no_v  = [v for v in votes if v.vote == "NO"]

    avg_conf = sum(v.confidence for v in votes) / len(votes) if votes else 0.5
    yd_edges = [true_edge_yes(v.vote, v.edge, market.implied_prob_yes) * v.confidence for v in votes]
    avg_edge = sum(yd_edges) / len(yd_edges) if yd_edges else 0.0
    est_prob = min(0.99, max(0.01, market.implied_prob_yes + avg_edge))
    margin = abs(len(yes_v) - len(no_v))
    ci = avg_conf * (margin / 7.0)

    approved, grade, reason = risk.assess_signal(
        len(yes_v), len(no_v), len(votes) - len(yes_v) - len(no_v),
        avg_conf, avg_edge, market.implied_prob_yes, market.ticker
    )
    contracts = risk.compute_contracts(market.yes_price_cents, est_prob) if approved else 0

    return Signal(
        ticker=market.ticker,
        market_title=market.market_title,
        yes_cents=market.yes_price_cents,
        no_cents=market.no_price_cents,
        volume=market.volume,
        days_out=market.days_to_event,
        market_prob=market.implied_prob_yes,
        est_prob=est_prob,
        avg_edge=avg_edge,
        yes_votes=len(yes_v), no_votes=len(no_v),
        pass_votes=len(votes) - len(yes_v) - len(no_v),
        margin=margin, avg_conf=avg_conf, ci=ci,
        majority="YES" if len(yes_v) > len(no_v) else "NO" if len(no_v) > len(yes_v) else "PASS",
        approved=approved, grade=grade, reason=reason, contracts=contracts,
        per_agent=[{
            "id": v.agent_id, "name": v.agent_name, "vote": v.vote,
            "conf": v.confidence, "edge_raw": v.edge,
            "edge_yes": true_edge_yes(v.vote, v.edge, market.implied_prob_yes),
            "reason": v.reason,
        } for v in votes],
        wall_time=sum(v.latency_ms for v in votes) / 1000,
    )


# ---------------------------------------------------------------------------
# Market fetching
# ---------------------------------------------------------------------------

def fetch_markets(limit: int = 50) -> list[Market]:
    import urllib.request
    key_id, key_path = "", ""
    env_file = Path("/home/am/kalshi-mev-engine/.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "KALSHI_API_KEY_ID" in line: key_id = line.split("=")[1].strip()
            if "KALSHI_PRIVATE_KEY_PATH" in line:
                key_path = line.split("=", 1)[1].strip().replace("~", str(Path.home()))

    if not key_id or not key_path:
        return sample_markets()

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    import base64

    ts = str(int(time.time() * 1000))
    path = f"/trade-api/v2/markets?status=active&limit={limit}"
    with open(key_path) as f:
        pk = serialization.load_pem_private_key(f.read().encode(), password=None, backend=default_backend())
    sig = base64.b64encode(
        pk.sign((ts + "GET" + path).encode(), padding.PKCS1v15(), hashes.SHA256())
    ).decode()

    req = urllib.request.Request(
        f"https://api.elections.kalshi.com{path}",
        headers={"KALSHI-ACCESS-KEY": key_id,
                 "KALSHI-ACCESS-SIGNATURE": sig,
                 "KALSHI-ACCESS-TIMESTAMP": ts}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            markets = []
            for m in data.get("markets", []):
                try:
                    markets.append(Market(
                        ticker=m["ticker"],
                        series_title=m.get("series_title", ""),
                        market_title=m.get("title", ""),
                        yes_price_cents=int(float(m.get("yes_price", 0.5)) * 100),
                        no_price_cents=int(float(m.get("no_price", 0.5)) * 100),
                        volume=int(m.get("volume", 0)),
                        days_to_event=max(1, (
                            datetime.fromisoformat(m["close_date"].replace("Z", "+00:00"))
                            - datetime.now(timezone.utc)
                        ).days),
                    ))
                except:
                    pass
            return markets
    except Exception:
        return sample_markets()


def sample_markets() -> list[Market]:
    return [
        Market("KXPUTIN-26", "Geopolitics", "Putin President End 2026", 75, 25, 20000, 270),
        Market("KXNBA-26", "NBA 2026", "NBA Champion 2026", 55, 45, 8000, 120),
        Market("KXCPICPI-26MAR", "CPI", "CPI > 4pct March 2026", 28, 72, 12000, 14),
        Market("KXNFL-26", "NFL 2026", "Super Bowl Winner 2026", 40, 60, 15000, 240),
        Market("KXIRAN-26", "Geopolitics", "Iran Nuclear Deal 2026", 35, 65, 8000, 300),
        Market("KXWEATHER-ATL-26", "Hurricane", "Major Hurricane Gulf Coast 2026", 60, 40, 5000, 180),
        Market("KXELNINO-26", "Climate", "El Nino April 2026", 45, 55, 3000, 21),
        Market("KXJOBS-26APR", "Jobs", "Jobs Beat April 2026", 55, 45, 6000, 7),
        Market("KXAMAZON-26", "Earnings", "Amazon EPS Beat Q1 2026", 58, 42, 9000, 45),
        Market("KXBTC-100K-26", "Crypto", "BTC > 100k by Dec 2026", 45, 55, 30000, 270),
    ]


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(msg: str):
    token = os.getenv(_cfg("telegram_bot_token_env", "MVE_TELEGRAM_BOT_TOKEN"))
    if not token:
        return
    import urllib.request
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": _cfg("telegram_chat_id", "6953738253"), "text": msg, "parse_mode": "HTML"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

GRADE_EMOJI = {
    "STRONG_BUY": "🟢🟢", "BUY": "🟢", "WEAK_BUY": "🟡",
    "STRONG_SELL": "🔴🔴", "SELL": "🔴", "WEAK_SELL": "🟠",
    "CIRCUIT_BREAKER": "🚨", "NO_SIGNAL": "⚪", "LOW_EDGE": "⚪",
    "LOW_CONFIDENCE": "⚪", "PASS": "⚪",
}


def print_signal(sig: Signal):
    emoji = GRADE_EMOJI.get(sig.grade, "⚪")
    edge_pct = sig.avg_edge * 100
    side = sig.majority
    price = sig.yes_cents if side == "YES" else sig.no_cents

    print(f"\n{emoji} {sig.ticker}")
    print(f"  {sig.market_title[:60]}")
    print(f"  YES={sig.yes_cents}¢ NO={sig.no_cents}¢ | vol={sig.volume:,} | {sig.days_out}d out")
    print(f"  Market: {sig.market_prob:.0%}  Est: {sig.est_prob:.0%}  Edge: {edge_pct:+.0%}")
    print(f"  Votes: {sig.yes_votes}Y/{sig.no_votes}N/{sig.pass_votes}P (margin={sig.margin})")
    print(f"  Conf: {sig.avg_conf:.0%}  CI: {sig.ci:.3f}  Grade: {sig.grade}")
    if sig.approved:
        print(f"  ✅ APPROVED — {side} x{sig.contracts} @ {price:.0f}¢")
    else:
        print(f"  ⛔ REJECTED — {sig.reason}")
    for a in sig.per_agent:
        print(f"    [{a['id']}] {a['name']:18s} {a['vote']:4s}  conf={a['conf']:.2f}  edge_y={a['edge_yes']:+.2f}  {a['reason'][:45]}")


def print_status(portfolio: Portfolio, risk: RiskEngine):
    st = risk.status()
    sm = portfolio.summary()
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
        vram = r.stdout.strip().replace(", ", " MiB / ") + " MiB"
    except:
        vram = "unknown"
    print(f"""
╔══════════════════════════════════════════════╗
║        BONSAI HEDGE FUND — STATUS           ║
╠══════════════════════════════════════════════╣
║  Bankroll:       ${st['bankroll']:.4f}
║  Starting:       ${sm['starting_capital']:.2f}
║  P&L:            ${sm['total_pnl']:+.4f} ({sm['total_return_pct']:+.2f}%)
║  Peak:           ${st['peak']:.4f}
║  Drawdown:       {st['drawdown_pct']:.2f}%
║  Circuit Break:  {'TRIPPED' if st['circuit_breaker'] else 'CLEAR'}
╠══════════════════════════════════════════════╣
║  Open Positions: {sm['open_positions']}
║  Closed Trades:  {sm['closed_positions']}
║  Win Rate:       {sm['win_rate']:.1f}%
║  Sharpe (approx):{sm['sharpe_approx']:.2f}
╠══════════════════════════════════════════════╣
║  Bonsai instances: 7 @ ports {PORT_BASE}-{PORT_BASE+6}
║  VRAM:           {vram}
║  Mode:           {'PAPER' if PAPER_MODE else 'LIVE'}
╚══════════════════════════════════════════════╝
""")
    for p in sm.get("positions", []):
        print(f"  {p['ticker']}: {p['side']} x{p['contracts']} @ {p['avg_price']:.0f}c → {p['market_price']:.0f}c  PnL: {p['unrealized_pnl']:+.4f}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_scan(portfolio: Portfolio, risk: RiskEngine, reporter: Reporter, use_api: bool = True):
    print(f"\n{'='*60}\n  BONSAI FUND — MARKET SCAN  [{datetime.now(timezone.utc).strftime('%H:%M UTC')}]\n{'='*60}")
    markets = fetch_markets(limit=30) if use_api else sample_markets()
    if not markets or len(markets) <= 1:
        print("API unavailable — using sample markets"); markets = sample_markets()
    print(f"\nScanning {len(markets)} markets with 7-agent swarm...\n")

    approved, all_sigs = [], []

    for mkt in markets:
        votes = collect_votes(mkt)
        sig = analyze(mkt, votes, portfolio, risk)
        all_sigs.append(sig)
        print_signal(sig)
        if sig.approved:
            approved.append((mkt, sig))
            if PAPER_MODE:
                side, price = sig.majority, (sig.yes_cents if sig.majority == "YES" else sig.no_cents)
                portfolio.open_position(mkt.ticker, side, sig.contracts, price, voted_by="bonsai_swarm")
                print(f"  📝 PAPER: Opened {side} x{sig.contracts} @ {price:.0f}¢")

    portfolio.record_equity()
    print(f"\n{'='*60}\n  SCAN SUMMARY  [{datetime.now(timezone.utc).strftime('%H:%M UTC')}]\n{'='*60}")
    print(f"  Markets: {len(markets)}  Approved: {len(approved)}  Bankroll: ${portfolio.bankroll:.4f}")
    if approved:
        print(f"\n  APPROVED:")
        for mkt, sig in approved:
            side = sig.majority; price = sig.yes_cents if side == "YES" else sig.no_cents
            print(f"    {mkt.ticker:40s} {side} x{sig.contracts} @ {price}¢  edge={sig.avg_edge:+.0%}")
    if approved and _cfg("alerts_enabled", False) and not PAPER_MODE:
        lines = [f"✅ Bonsai Signal — {len(approved)} approved"]
        for mkt, sig in approved:
            side = sig.majority; price = sig.yes_cents if side == "YES" else sig.no_cents
            lines.append(f"• {mkt.ticker}: {side} x{sig.contracts} @ {price}¢")
        send_telegram("\n".join(lines))
    return all_sigs


def cmd_board_report(portfolio: Portfolio, risk: RiskEngine, reporter: Reporter):
    report = reporter.generate_board_report(days=7)
    text = reporter.format_telegram_board_report(report)
    reporter.save_report_json(report)
    send_telegram(text)
    print(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bonsai Hedge Fund")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status")
    sub.add_parser("scan")
    board = sub.add_parser("board-report")
    cb    = sub.add_parser("circuit-break")
    rst   = sub.add_parser("reset-portfolio")
    vote  = sub.add_parser("vote"); vote.add_argument("ticker")
    args = parser.parse_args()

    portfolio = Portfolio()
    risk      = RiskEngine(portfolio, RiskLimits())
    reporter  = Reporter(portfolio, risk)

    if args.cmd == "status":
        print_status(portfolio, risk)
    elif args.cmd == "scan":
        cmd_scan(portfolio, risk, reporter, use_api=False)
    elif args.cmd == "board-report":
        cmd_board_report(portfolio, risk, reporter)
    elif args.cmd == "circuit-break":
        risk.limits.circuit_breaker = False
        print("✅ Circuit breaker reset.")
    elif args.cmd == "reset-portfolio":
        portfolio.reset_to_cash()
        print("✅ All positions liquidated at market price.")
    elif args.cmd == "vote":
        markets = {m.ticker: m for m in sample_markets()}
        if args.ticker not in markets:
            print(f"Unknown ticker: {args.ticker}"); return
        votes = collect_votes(markets[args.ticker])
        sig   = analyze(markets[args.ticker], votes, portfolio, risk)
        print_signal(sig)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
