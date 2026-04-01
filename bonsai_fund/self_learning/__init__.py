"""
Bonsai Fund — Recursive Self-Learning System

The swarm learns from every outcome. After each market resolves,
the system recursively updates agent weights, refines base rates,
evolves system prompts, and compounds knowledge across three time scales:

  SHORT-TERM  (single trade):     OutcomeTracker records result, updates agent weights
  MEDIUM-TERM (10-50 trades):    MarketClassifier finds category patterns, refines base rates
  LONG-TERM   (100+ trades):     EvolutionaryMutator evolves system prompts, discovers new strategies

The learning is COMPOUNDING — just like capital. Each cycle's insight is stored
and becomes input to the next cycle.
"""

from __future__ import annotations

from bonsai_fund.self_learning.outcome_tracker import OutcomeTracker, TradeOutcome
from bonsai_fund.self_learning.agent_memory import AgentMemory, AgentPerformance
from bonsai_fund.self_learning.market_classifier import MarketClassifier, MarketCategory
from bonsai_fund.self_learning.evolver import EvolutionaryMutator, EvolvedAgent
from bonsai_fund.self_learning.orchestrator import LearningOrchestrator

__all__ = [
    "OutcomeTracker",
    "TradeOutcome",
    "AgentMemory",
    "AgentPerformance",
    "MarketClassifier",
    "MarketCategory",
    "EvolutionaryMutator",
    "EvolvedAgent",
    "LearningOrchestrator",
]
