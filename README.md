# Bonsai Fund

> An open-source quantitative hedge fund powered by a swarm of 7 specialized LLM agents,
> with **recursive self-learning** that compounds knowledge across three time scales.
> Dedicated to the Hermes Agent community — built by The Firm, shared with the world.

```ascii
┌─────────────────────────────────────────────────────────────────┐
│                    BONSAI HEDGE FUND                             │
│                   Recursive Self-Learning                        │
│                                                             │
│  SHORT-TERM (every scan)         MEDIUM-TERM (resolution)     │
│  ┌──────────────────┐            ┌──────────────────────┐     │
│  │  AgentMemory     │            │  OutcomeTracker      │     │
│  │  Vote history    │──────────▶ │  Category base rates │     │
│  │  Context weights │  tag by   │  Agent affinities    │     │
│  │  (category,      │  category │                      │     │
│  │   price, horizon) │            │  MarketClassifier    │     │
│  └──────────────────┘            │  (analogical priors) │     │
│            │                       └──────────┬─────────┘     │
│            ▼                                   │               │
│  ┌──────────────────┐                          │               │
│  │ Weighted votes   │                          │               │
│  │ boost strong     │◀─────────────────────────┘               │
│  │ agents per       │   LONG-TERM (every 100+ trades)          │
│  │ context          │   ┌──────────────────────────────────┐  │
│  └──────────────────┘   │  EvolutionaryMutator              │  │
│                          │  • Score all prompt variants     │  │
│                          │  • Keep top 2 as champions        │  │
│                          │  • Mutate: strengthen_bayesian,   │  │
│                          │    strengthen_contrarian, etc.    │  │
│                          │  • Lineage tracking (hash chain) │  │
│                          └──────────────┬───────────────────┘  │
│                                         ▼                       │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│   │ Fast │ │ Deep │ │Baye- │ │Con-  │ │Macro │ │Foren-│     │
│   │Intuit│ │Analyst│ │sian │ │trarian│ │Integr│ │sic  │     │
│   │  0   │ │  1   │ │  2   │ │  3   │ │  4   │ │  5   │     │
│   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘     │
│      └────────┴────────┴────────┴────────┴────────┘           │
│                           ↓                                    │
│                    ┌──────────────┐                             │
│                    │  FinalVote   │                             │
│                    │  (Synthesize)│                             │
│                    └──────┬───────┘                             │
│                           ↓                                      │
│              ┌────────────┼────────────┐                        │
│              ↓            ↓            ↓                        │
│         Portfolio    RiskEngine    Reporter                      │
│         (SQLite)     (Kelly+CB)   (Telegram)                  │
└─────────────────────────────────────────────────────────────────┘
```

## What it does

Bonsai Fund runs a **swarm of 7 specialized LLM agents** as a collective intelligence to
trade prediction markets (Kalshi). Each agent has a distinct cognitive style.

**Recursive self-learning**: The swarm gets smarter over time across three time scales:

- **Short-term** (every scan): Context-sensitive weighting — agents that have been accurate
  in similar markets (same category, price range, time horizon) get up to 2x their vote weight
- **Medium-term** (every resolution): Category base rates refine, agent affinities update,
  and the market classifier uses analogical reasoning to bootstrap new categories
- **Long-term** (every 100+ trades): The EvolutionaryMutator runs a full generation cycle —
  scoring, selecting champions, and guided-mutating system prompts to create smarter variants

**The swarm learns like a fund manager**: Every trade is a lesson. The edge compounds.

| # | Agent | Style | Best At |
|---|-------|-------|---------|
| 0 | **FastIntuit** | System-1 gut | Quick pattern recognition |
| 1 | **DeepAnalyst** | System-2 slow | Probability decomposition |
| 2 | **BayesianUpdater** | Bayesian | Sequential evidence updating |
| 3 | **Contrarian** | Crowd fade | Fat-tail extremes |
| 4 | **MacroIntegrator** | Top-down | Cross-market regime |
| 5 | **ForensicReader** | Anomaly | Hidden assumptions |
| 6 | **FinalVote** | Synthesis | Herding detection + veto |

All agents run **locally** — no API key, no cloud dependency, no per-token cost.

## Features

- **7 cognitive agents** with distinct thinking styles, all running on Bonsai-8B GGUF locally
- **Kelly criterion** sizing — mathematically optimal position sizing for repeated bets
- **Circuit breaker** — 25% drawdown halt, manual reset required
- **SQLite portfolio** — position tracking, realized/unrealized P&L, equity curve
- **Board Reports** — Sunday 8am Telegram digest for stakeholders
- **Strategy plugins** — drop in new signal generators without touching core code
- **Hermes-native** — `/bonsai` slash commands, cron integration, skill install

## Quick Start

```bash
# 1. Install as a Hermes skill
hermes skills install bonsai-fund

# 2. Install llama-server (pre-built binary)
python3 ~/.hermes/skills/bonsai-fund/bonsai_fund/launch.py install-llama

# 3. Download Bonsai-8B model (~1.2 GB)
python3 ~/.hermes/skills/bonsai-fund/bonsai_fund/launch.py download-model

# 4. Launch 7 Bonsai instances
python3 ~/.hermes/skills/bonsai-fund/bonsai_fund/launch.py start --count 7

# 5. Run a scan
python3 ~/.hermes/skills/bonsai-fund/bonsai_fund/hedge_fund.py scan

# 6. Check status
python3 ~/.hermes/skills/bonsai-fund/bonsai_fund/hedge_fund.py status
```

## Architecture

```
bonsai_fund/
├── agent.py          — 7 system prompts + few-shot examples
├── portfolio.py      — SQLite position tracking, P&L, equity curve
├── risk.py           — Kelly sizing, signal grading, circuit breaker
├── reporter.py       — Board Report generator, Telegram HTML format
├── hedge_fund.py     — Main orchestrator: scan → vote → size → execute
├── scheduler.py      — Autonomous loop (30-min scans, Sunday 8am Board Report)
└── launch.py         — Bonsai instance lifecycle manager
```

## Risk Rules

| Parameter | Value |
|-----------|-------|
| Max position | 5% of bankroll |
| Min edge | 3% |
| Min confidence | 55% |
| Min vote margin | 2 votes |
| Circuit breaker | 25% drawdown |
| Leverage | None (cash-only) |

## Hermes Integration

```bash
# Schedule 30-minute market scans
/hermes cron add "Bonsai Scan" --schedule "every 30m" \
  --prompt "cd ~/.hermes/skills/bonsai-fund && python3 bonsai_fund/scheduler.py once"

# Schedule Sunday 8am Board Report
/hermes cron add "Bonsai Board Report" --schedule "0 8 * * 0" \
  --prompt "cd ~/.hermes/skills/bonsai-fund && python3 bonsai_fund/scheduler.py board-report"

# Direct commands
/bonsai scan          # Run market scan now
/bonsai status        # Portfolio + risk status
/bonsai board-report  # Send Board Report to Telegram
/bonsai circuit-break # Reset circuit breaker
```

## Strategy Plugins

Drop in new signal generators:

```python
# examples/strategies/momentum.py
from bonsai_fund.agent import Market

def generate_signal(market: Market) -> dict:
    return {
        "vote": "YES",
        "confidence": 0.72,
        "edge": 0.08,
        "reason": "high-volume momentum breakout",
    }
```

Register in `hedge_fund.py` to include in the voting swarm.

## Requirements

- Python 3.10+
- llama.cpp (`llama-server`) in PATH or `~/bin/llama-server`
- Bonsai-8B GGUF model (~1.2 GB)
- 24GB VRAM for 7 instances (RTX 3090 / RTX 4090)
- Hermes CLI (for skill install + cron)

## Installation

```bash
# One-line install via Hermes
hermes skills install bonsai-fund

# Or manual
git clone https://github.com/amanning3390/bonsai-fund.git ~/.hermes/skills/bonsai-fund
```

## License

MIT — use it, fork it, improve it, ship it.

---

*Built with love for the Hermes Agent community.*
*The swarm learns. The fund persists. The edge compounds.*
