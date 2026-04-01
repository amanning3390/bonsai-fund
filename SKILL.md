---
name: bonsai-fund
description: >
  Open-source quantitative hedge fund powered by a swarm of specialized LLM agents.
  Runs a 7-agent cognitive team locally (Bonsai-8B GGUF via llama.cpp) to analyze
  prediction markets (Kalshi), generate Kelly-sized signals, and manage a live portfolio
  with circuit breakers. Designed for Hermes — runs autonomously with Board Reports
  to Telegram. Install: hermes skills install bonsai-fund
version: 1.0.0
category: trading
tags: [prediction-markets, llm-swarm, kalshi, quantitative-trading, autonomous-agent, bonsai-8b]
requires: ["llama-cpp"]
suggests: ["kalshi-api", "telegram-bot"]
platforms: [linux, macos]
license: MIT
author: The Firm
homepage: https://github.com/the-firm/bonsai-fund
issues: https://github.com/the-firm/bonsai-fund/issues
---

# Bonsai Fund — LLM-Powered Quantitative Hedge Fund

A fully autonomous, open-source quant fund that runs a **swarm of 7 specialized LLM agents**
as a collective intelligence to trade prediction markets (Kalshi). Every component is
self-contained and runs locally — no cloud LLM API required.

## Quick Start

```bash
# 1. Install the model (one-time)
python3 bonsai_fund/launch.py download-model

# 2. Launch 7 Bonsai instances (background)
python3 bonsai_fund/launch.py start --count 7

# 3. Run a market scan
python3 bonsai_fund/hedge_fund.py scan

# 4. Check status
python3 bonsai_fund/hedge_fund.py status
```

## Architecture

```
bonsai_fund/
├── agent.py          — 7 agent identities (system prompts + few-shot examples)
├── portfolio.py      — Position tracking, P&L, equity curve (SQLite)
├── risk.py           — Kelly sizing, drawdown circuit breaker, signal grading
├── reporter.py       — Board Report generator (Telegram HTML format)
├── hedge_fund.py     — Main orchestrator (scan → vote → size → execute)
├── scheduler.py      — Autonomous cron loop (30min scans, Sunday 8am Board Report)
├── launch.py         — Bonsai instance lifecycle manager (start/stop/restart/status)
└── examples/         — Custom agent templates, strategy plugins
```

## The 7-Agent Trading Team

| # | Agent | Cognitive Style | Core Method |
|---|-------|----------------|-------------|
| 0 | **FastIntuit** | System-1 gut | Recognition-primed — "gut fires before cortex" |
| 1 | **DeepAnalyst** | System-2 slow | Probability tree decomposition |
| 2 | **BayesianUpdater** | Bayesian | Sequential evidence → prior/posterior updating |
| 3 | **Contrarian** | Crowd inversion | Fade consensus at extremes (Elliott Wave logic) |
| 4 | **MacroIntegrator** | Top-down | Regime → sector → event cascade |
| 5 | **ForensicReader** | Anomaly hunter | Hidden assumptions, false binary, liquidity traps |
| 6 | **FinalVote** | Synthesis + veto | Full context read, herding detection, veto power |

## Risk Rules

| Parameter | Value |
|-----------|-------|
| Max position size | 5% of bankroll |
| Min edge to trade | 3% |
| Min vote margin | 2 votes |
| Min avg confidence | 55% |
| Circuit breaker | 25% drawdown from peak |
| Daily loss pause | 5% bankroll |
| Leverage | None (cash-only) |

## Commands

```bash
# Status
python3 bonsai_fund/hedge_fund.py status

# Scan and paper trade (default — always paper)
python3 bonsai_fund/hedge_fund.py scan

# Vote on a specific ticker
python3 bonsai_fund/hedge_fund.py vote KXNBA-26

# Reset circuit breaker
python3 bonsai_fund/hedge_fund.py circuit-break

# Liquidation at market
python3 bonsai_fund/hedge_fund.py reset-portfolio

# Autonomous scheduler
python3 bonsai_fund/scheduler.py once           # single cycle
python3 bonsai_fund/scheduler.py board-report   # send Board Report now
python3 bonsai_fund/scheduler.py status         # scheduler state

# Launch manager
python3 bonsai_fund/launch.py start --count 7
python3 bonsai_fund/launch.py status
python3 bonsai_fund/launch.py stop
python3 bonsai_fund/launch.py restart --count 7
python3 bonsai_fund/launch.py restart --count 7 --force  # kill + restart
```

## Configuration

Edit `~/.hermes/skills/bonsai-fund/config.yaml`:

```yaml
bonsai_fund:
  data_dir: ~/.hermes/bonsai_fund_data
  bankroll: 50.00
  paper_mode: true                    # set false to enable live trading

  # Bonsai model
  model_path: ~/.local/share/llama.cpp/bonzai/8b-v1-q1_0.gguf
  model_name: Bonsai-8B.gguf
  port_base: 8090
  instances: 7                       # VRAM-limited (RTX 3090 maxes at 7)

  # Risk
  max_position_pct: 0.05
  min_edge: 0.03
  min_confidence: 0.55
  min_vote_margin: 2
  max_drawdown_pct: 0.25

  # Scheduling
  scan_interval_min: 30
  board_report_utc_hour: 8
  board_report_utc_day: 6           # 0=Monday, 6=Sunday

  # Notifications
  telegram_bot_token: ""             # optional — Board Report + alerts
  telegram_chat_id: "6953738253"
  alerts_enabled: false             # true = send Telegram alerts on signals

kalshi:
  api_url: https://api.elections.kalshi.com
  key_id: ""                        # from ~/.env or config
  private_key_path: ~/.ssh/kalshi.pem
  markets_limit: 50
```

## Customizing Agents

Each agent is defined by a system prompt + few-shot examples in `agent.py`.
To create a new agent or modify an existing one:

```python
# In agent.py — extend SYSTEM_PROMPTS and FEW_SHOT_EXAMPLES

SYSTEM_PROMPTS[7] = """You are MomentumTrader — a momentum-first analyst.

You only trade with the trend. You wait for the market to prove direction,
then pile in. You exit when momentum fades.

Your method: momentum confirmation.
1. Is the price moving? (volume > baseline?)
2. What direction? (7-day slope positive/negative?)
3. Is the move accelerating or decelerating?
4. Entry: with momentum on high-volume days only.
"""

FEW_SHOT_EXAMPLES[7] = """
Q: Will TSLA hit 300? YES=55c, NO=45c, volume=3x average.
Your momentum read: Volume surging, price breaking out above 200 DMA.
Momentum is BUILDING in the YES direction.
Decision: YES, conf=0.72, edge=+0.17

Now analyze:
"""
```

Then add the agent ID to the vote collection loop in `hedge_fund.py`:
```python
# Change range(6) to range(8) to include agent 7
for agent_id in range(8):   # was range(6)
    ...
```

## Adding a New Strategy

Strategies are pluggable signal generators. To add one:

1. Create `examples/strategies/your_strategy.py`:

```python
"""
Your custom strategy plugin.
Must implement: generate_signal(market: Market) -> dict
"""
from agent import Market

def generate_signal(market: Market) -> dict:
    # Your logic here
    return {
        "strategy": "your_strategy",
        "vote": "YES",        # or "NO" or "PASS"
        "confidence": 0.65,
        "edge": 0.07,         # positive = YES underpriced
        "reason": "momentum breakout confirmed",
    }
```

2. Register it in `hedge_fund.py` by adding to the `analyze()` function:
```python
from examples.strategies.your_strategy import generate_signal as your_sig
signals.append(your_sig(market))
```

## Understanding the Signal Grades

| Grade | Meaning | Action |
|-------|---------|--------|
| STRONG_BUY | CI≥0.40, margin≥5 | Full Kelly size |
| BUY | CI≥0.20, margin≥3 | Standard Kelly |
| WEAK_BUY | CI≥0.10, edge≥5% | Half Kelly |
| PASS | Below thresholds | No trade |
| WEAK_SELL | edge≥5%, majority=NO | Half Kelly on NO |
| SELL | CI≥0.20, majority=NO | Standard Kelly on NO |
| STRONG_SELL | CI≥0.40, margin≥5 | Full Kelly on NO |

## Kelly Sizing Formula

```
b = (100 / yes_price_cents) − 1      # net decimal odds
p = estimated_true_probability       # swarm's inferred probability
q = 1 − p
f* = (b × p − q) / b                 # Kelly fraction
position_dollars = f* × bankroll    # capped at max_position_pct
contracts = position_dollars / (yes_price_cents / 100)
```

## Circuit Breaker Protocol

When drawdown reaches 25%:
1. All positions are marked for review
2. No new trades are opened
3. Telegram alert is sent (if configured)
4. Manual reset required: `hedge_fund.py circuit-break`

## Data Storage

| Data | Location |
|------|----------|
| Portfolio / positions | `~/.hermes/bonsai_fund_data/bonsai_portfolio.db` |
| Equity curve | Same SQLite file |
| Vote history | `~/.hermes/bonsai_fund_data/bonsai_votes.db` |
| Scheduler state | `~/.hermes/bonsai_fund_data/bonsai_scheduler_state.json` |
| Logs | `~/.hermes/bonsai_fund_data/logs/` |
| Board Reports | `~/.hermes/bonsai_fund_data/bonsai_board_report_latest.json` |

## Troubleshooting

**`llama-server: command not found`**
Install PrismML llama.cpp: `bash bonsai_fund/scripts/install_llama.sh`

**`HTTP 400 Bad Request` on market fetch**
The Kalshi API requires RSA-signed requests. Ensure your `.env` has:
```
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=~/.ssh/kalshi.pem
```

**`VRAM exhausted` when starting instances**
Reduce count: `launch.py start --count 6`. 7 is the max on RTX 3090 24GB.

**All agents return PASS votes**
The model may not be generating valid JSON. Check:
- Is Bonsai-8B loaded correctly? Run `launch.py status`
- Try a direct test: `curl http://127.0.0.1:8090/v1/models`

**Scheduler not running**
Use cron-mode: `python3 scheduler.py once` (cron calls this every 30 min).
The `--daemon` flag is for background systemd services only.

## Hermes Integration

```bash
# Install as a Hermes skill
hermes skills install bonsai-fund

# Run via Hermes
/hermes bonsai scan
/hermes bonsai status
/hermes bonsai board-report

# Add to cron via Hermes
/hermes cron add "Bonsai Scan" "every 30m" \
  --prompt "Run: cd ~/.hermes/skills/bonsai-fund && python3 bonsai_fund/scheduler.py once"

/hermes cron add "Bonsai Board Report" "0 8 * * 0" \
  --prompt "Run: cd ~/.hermes/skills/bonsai-fund && python3 bonsai_fund/scheduler.py board-report"
```

## Architecture Decisions

- **Why 7 agents?** Odd number prevents ties. 7 is the minimum for meaningful cognitive diversity
  (System-1/System-2/Bayesian/Contrarian/Macro/Forensic/Synthesis).
- **Why local GGUF?** Eliminates API latency, cost, and dependency. A single RTX 3090
  runs 7 parallel instances at 161 tokens/second aggregate.
- **Why paper first?** The swarm needs a live calibration period. Paper trading builds the
  track record before capital is at risk.
- **Why SQLite?** Zero-dependency, durable, queryable. Perfect for a single-server system.
- **Why Kelly?** Mathematically optimal for infinite repetitions. The swarm estimates `p`
  (true probability), and Kelly sizes accordingly. Full Kelly is halved in practice.
