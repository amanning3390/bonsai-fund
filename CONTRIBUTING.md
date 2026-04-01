# Contributing to Bonsai Fund

Thank you for contributing to the Hermes community's open-source quant fund.

## Ways to Contribute

1. **New agent strategies** — Add cognitive styles to `agent.py` and `FEW_SHOT_EXAMPLES`
2. **Strategy plugins** — Drop new generators into `examples/strategies/`
3. **Risk rules** — Improve Kelly estimation, drawdown logic, signal grading
4. **Market integrations** — Connect new prediction markets (Polymarket, Betfair, etc.)
5. **UI / reporting** — Better Telegram formatting, dashboards, charts
6. **Performance** — Optimize the 7-agent parallel voting pipeline

## Getting Started

```bash
git clone https://github.com/amanning3390/bonsai-fund.git
cd bonsai-fund

# Run tests
python3 -c "from bonsai_fund import *; print('OK')"

# Run full smoke suite
python3 bonsai_fund/hedge_fund.py status
```

## Pull Request Guidelines

- All CI checks must pass (imports, portfolio ops, signal analysis, Board Report)
- New agents require: system prompt, few-shot example, entry in `AGENT_NAMES`
- New strategies require: `generate_signal(market) -> dict` function + tests
- Risk changes require: updated `RiskLimits` defaults + comments

## Project Structure

```
bonsai_fund/
├── agent.py         ← 7 cognitive agents (edit here to add/modify agents)
├── portfolio.py    ← SQLite tracking (do not modify schema in PRs)
├── risk.py         ← Kelly + circuit breaker
├── reporter.py     ← Board Report
├── hedge_fund.py   ← Orchestrator
├── scheduler.py    ← Autonomous loop
└── launch.py       ← Model management
examples/
├── strategies/      ← Drop-in strategy plugins
└── hermes_cron_integration.py
```

## Code of Conduct

Be excellent to each other. The swarm competes against the market — not each other.
