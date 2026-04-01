"""
Bonsai Fund — Open-source LLM-powered quantitative hedge fund.
"""

__version__ = "1.0.0"
__author__  = "The Firm"
__license__ = "MIT"

# Core classes — no bonsai_fund imports here (avoids circular dep)
from bonsai_fund.portfolio import Portfolio, Position
from bonsai_fund.risk import RiskEngine, RiskLimits
from bonsai_fund.reporter import Reporter, BoardReport
from bonsai_fund.agent import (
    Market, AgentVote,
    AGENT_NAMES, AGENT_SPECIALTIES,
    SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES,
    PORT_BASE, MODEL,
)
