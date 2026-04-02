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

## Recursive Self-Learning System

Bonsai Fund implements a **three-time-scale recursive learning loop** that compounds knowledge
across trades, just as capital compounds across returns.

### Architecture

```
bonsai_fund/self_learning/
├── outcome_tracker.py   — SHORT-TERM: records every trade outcome
├── agent_memory.py       — MEDIUM-TERM: per-agent, per-category weights
├── market_classifier.py  — MEDIUM-TERM: category base rates + analogical priors
├── evolver.py           — LONG-TERM: evolutionary prompt mutation
└── orchestrator.py      — Ties all three loops together
```

### The Three Learning Loops

#### SHORT-TERM (every scan cycle)

Every time the swarm votes on a market, those votes are stored in `AgentMemory` tagged with:
- Market category (Geopolitics, NBA, CPI, etc.)
- Price bin (0-20c, 20-40c, ..., 80-100c)
- Time horizon (short <14d, medium 14-60d, long >60d)

On the **next** scan, when a similar market comes up, agents that have historically
performed well in that category/price/horizon get **up-weighted** by up to 2x.
Agents with poor track records in that context get **down-weighted** to 0.5x.

```
Example: A CPI market at 28c, 14 days out
  → FastIntuit has been right on CPI 12/15 times  → weight = 1.3x
  → DeepAnalyst has been wrong on CPI 4/8 times   → weight = 0.7x
  → BayesianUpdater has been right on CPI 9/10 times → weight = 1.4x
Weighted confidence: (0.75×1.3 + 0.70×0.7 + 0.65×1.4) / (1.3+0.7+1.4) = ...
```

#### MEDIUM-TERM (after every market resolution)

When a market resolves (YES or NO), `OutcomeTracker` records the result and:
1. Updates each agent's accuracy score for that category/price/horizon
2. Updates the category's base rate (e.g., Geopolitics YES base rate = 38%)
3. Refines `MarketClassifier` agent affinities — which agents should be trusted more
   in each category going forward

#### LONG-TERM (every 100+ new trades)

`EvolutionaryMutator` runs a full evolution cycle:

1. **Evaluate**: Score all active agent prompt variants by win rate + edge + sample size
2. **Select**: Keep top 2 performers as champions; discard worst performers
3. **Mutate**: Create 2-3 new prompt variants from the champions using **guided mutations**

**Mutation operators** (guided by performance data):
| Operator | When Applied |
|----------|-------------|
| `strengthen_bayesian` | BayesianUpdater wins on CPI/Economics markets |
| `strengthen_contrarian` | Contrarian wins on high-volume Geopolitics |
| `strengthen_macro` | MacroIntegrator correctly called regime shifts |
| `strengthen_forensic` | ForensicReader found hidden anomalies in Earnings |
| `weaken_veto` | FinalVote is vetoing too many valid signals |
| `add_calibration_note` | Agent has poor calibration (predicted conf ≠ actual win rate) |
| `add_time_horizon_sensitivity` | Agent performs differently on short vs long markets |

**Lineage tracking**: Every evolved prompt variant carries its full lineage:
```
Generation 2 agent 2 (hash=a7f3c1)
  parent: Generation 1 agent 2 (hash=b2e8d4) — strengthen_bayesian
    parent: Generation 0 agent 2 (hash=seed) — original BayesianUpdater
```

### Commands

```bash
# Run a scan with self-learning enabled
python3 bonsai_fund/hedge_fund.py scan --learn

# Check learning status
python3 bonsai_fund/hedge_fund.py learn --status
python3 bonsai_fund/scheduler.py learn --status

# Force an evolution cycle (normally triggers automatically at 100+ trades)
python3 bonsai_fund/hedge_fund.py learn --evolve

# Record a market resolution (feeds into learning)
python3 bonsai_fund/hedge_fund.py resolve KXNBA-26 YES
python3 bonsai_fund/scheduler.py resolve KXCPICPI-26MAR NO
```

### Example: How Learning Compounds

```
TRADE 1:   FastIntuit → YES on Geopolitics @ 75c → WRONG (Putin event)
           AgentMemory: FastIntuit/Geopolitics -1 correct
           
TRADE 47:  FastIntuit votes YES on Geopolitics again
           Context weight for Geopolitics = 0.9x (penalized from Trade 1)
           Swarm adjusts: if other agents agree strongly, signal still fires
           
TRADE 112: BayesianUpdater outperforms FastIntuit on Geopolitics 8/10 vs 4/8
           → BayesianUpdater gets 1.3x weight on Geopolitics
           → FastIntuit gets 0.8x weight on Geopolitics
           
TRADE 200+: Evolution triggers. FastIntuit's prompt gets a mutation:
           "Important: your gut calibration on Geopolitics has been poor.
            Your confidence on Geopolitics should be reduced by 10-15%."
           
TRADE 350: New FastIntuit variant outperforms original: 62% vs 55% on Geopolitics
           → New prompt becomes the champion
           → Original is archived in lineage
```

### Data Stored by the Learning System

| Data | Location |
|------|----------|
| Trade outcomes + per-agent attribution | `bonsai_fund_data/outcomes.db` |
| Agent vote history + context tags | `bonsai_fund_data/agent_memory.db` |
| Category base rates + agent affinities | `bonsai_fund_data/market_classifier.db` |
| Evolved prompt variants + lineage | `bonsai_fund_data/evolver.db` |
| Learning events log | `bonsai_fund_data/learning_events.jsonl` |

### Seed Categories

The classifier starts with informed priors (refined automatically after first trades):

| Category | Base Rate | Description |
|----------|-----------|-------------|
| Geopolitics | 38% YES | Wars, sanctions, diplomatic events |
| NBA / NFL | 50% | Sports championships |
| Earnings | 52% | EPS beat/miss |
| Jobs | 50% | NFP, unemployment |
| CPI | 40% | Inflation readings |
| Hurricane | 40% | Landfall events |
| Elections | 48% | Political outcomes |
| Interest Rates | 45% | Fed decisions |
| Crypto | 45% | BTC, ETH events |

## Staged Drawdown System

Replaces the binary circuit breaker with 6 severity tiers. Each tier progressively:
- Reduces position sizing
- Raises vote confidence thresholds
- Mandates a training/improvement action before resuming

| Stage | DD Range | Position Size | Min Conf | Min Margin | Training Required |
|-------|----------|--------------|----------|------------|-------------------|
| 🟢 GREEN | 0-10% | 100% (5%) | 55% | 2 | None |
| 🟡 YELLOW | 10-15% | 50% (2.5%) | 60% (+5pp) | 3 | None |
| 🟠 ORANGE | 15-20% | 25% (1.25%) | 65% (+10pp) | 4 | ANALYSIS |
| 🔴 RED | 20-25% | 10% (0.5%) | 70% (+15pp) | 4 | SIMULATION |
| 🚨 CRITICAL | 25-35% | 0% (halt) | — | — | EVOLUTION |
| ❄️ FROZEN | 35%+ | 0% (halt) | — | — | DEEP_RESET |

**How it works:**
1. `StagedDrawdownMonitor` tracks peak and current bankroll continuously
2. On every scan, `check()` compares drawdown against tier boundaries
3. If tier requires training (ORANGE+), `DrawdownResponseOrchestrator` fires the appropriate action
4. Trading is blocked until the training completes and `mark_improvement_done()` is called
5. After training, stage limits relax but don't fully reset — the system stays cautious

**The insight:** A 15% drawdown means the swarm is wrong about something specific.
Rather than just stopping, the system learns WHY and fixes it before resuming.

## Simulation Engine (`simulation.py`)

Replay-based agent training using historical + synthetic market data.
**All trading generates learning data. All learning improves trading.**

### Simulation Modes

| Mode | When | Purpose |
|------|------|---------|
| `replay` | Manual | Re-run recent losing trades with agent substitutions |
| `synthetic` | Baseline (every 50 scans) | Diverse markets across all categories |
| `adversarial` | Drawdown ORANGE+ | Markets in weak categories at extreme prices |
| `candidate` | Evolution | Test evolved prompt variants against originals |

### Synthetic Market Generator

Generates realistic markets using templates per category:
- **Geopolitics:** Putin, sanctions, Iran nuclear, war escalation
- **NBA/NFL:** Championship winners, playoff teams, MVP
- **CPI/Jobs:** Inflation readings, NFP, unemployment
- **Earnings:** EPS beat/miss for major companies
- **Elections:** Candidate wins, party control
- **Crypto:** BTC/ETH price targets, ETF decisions
- **Hurricane:** Landfall, named storm counts

Ground truth is set to be slightly against crowded directions (regression to mean),
creating realistic losses for the swarm to learn from.

### Simulation Pipeline

```
1. Generate synthetic markets (adversarial + diverse)
2. Run agents with ORIGINAL prompts → record votes
3. Run agents with CANDIDATE prompts (evolved) → record votes
4. Score: candidate_win_rate vs original_win_rate
5. If candidate wins by >5% on 10+ trades → DEPLOY
   Else → keep original, log the failure
```

## News Pipeline (`news_pipeline.py`)

Monitors RSS feeds for market-relevant news and injects context into agent prompts
before each scan. **The swarm doesn't trade blind.**

### Sources

| Category | Feeds |
|----------|-------|
| Geopolitics | BBC World, NYT Politics |
| Economics | BBC Business, NYT Economy, FT |
| Earnings | BBC Business |
| Sports | ESPN |
| Crypto | CoinTelegraph, Decrypt |
| Weather | NOAA RSS |

### Context Injection

Before each market scan:
1. Fetch latest headlines for that market's categories
2. Score relevance (keyword matching + ticker hints)
3. Assemble `MarketContext` with sentiment tags (risk-off, inflation-fear, etc.)
4. Prepend to agent system prompts: `bonsai_fund/news_pipeline.py` wraps prompts

```python
# In the scan loop:
from bonsai_fund.news_pipeline import NewsPipeline
news = NewsPipeline()
for mkt in markets:
    ctx = news.get_context_for_market(mkt)
    enhanced_prompt = news.prepend_to_prompt(base_system_prompt, mkt)
    # ... vote with enhanced prompt
```

### Economic Calendar

Seed dates for known events: FOMC meetings, NFP (first Friday), CPI releases.
Calendar events appear in the composite context injected into agent prompts.

## Drawdown Response Orchestrator (`drawdown_response.py`)

The executive layer that maps drawdown stage → training action.
Runs training in background threads so the scheduler never blocks.

### Severity → Action Mapping

| Stage | Training Action | What Happens |
|-------|----------------|--------------|
| 🟢 GREEN | None | Normal scanning |
| 🟡 YELLOW | None | Passive monitoring, reduced sizing |
| 🟠 ORANGE | `analyze` | Background analysis of recent losing trades; identifies weak categories; fetches relevant news |
| 🔴 RED | `simulate` | Full adversarial simulation against weak categories; triggers evolution if candidate wins |
| 🚨 CRITICAL | `evolve` | Full evolution cycle; all trading halted |
| ❄️ FROZEN | `deep_reset` | Archive positions; load Gen 0 best prompts; manual review required |

### Non-Blocking Design

```
Scheduler scan cycle:
  1. orchestrator.check() → blocked? (training running?)
  2. If training needed → fire in background thread
  3. Continue scan (trading blocked by RiskEngine)
  4. On completion → mark_improvement_done() → RiskEngine re-enables trading
```

### Database Tables

| Table | Purpose |
|-------|---------|
| `response_actions` | Every training action fired |
| `drawdown_log` | Stage transitions with timestamps |

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
