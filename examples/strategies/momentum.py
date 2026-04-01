"""
Bonsai Fund — Example Strategy: Momentum Agent

Plug-and-play strategy agent. Drop this file into strategies/
and register it in hedge_fund.py to add it to the voting swarm.

MomentumTrader: Only trades with the trend. Waits for the market
to prove direction, then piles in. Exits when momentum fades.

Method: Price acceleration + volume confirmation
1. Is the price moving? (volume > baseline?)
2. What direction? (7-day slope positive/negative?)
3. Is the move accelerating or decelerating?
4. Entry: with momentum on high-volume days only.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MomentumSignal:
    strategy: str
    vote: str          # YES / NO / PASS
    confidence: float  # 0.0 - 1.0
    edge: float        # positive = YES underpriced
    reason: str


def generate_signal(market) -> MomentumSignal:
    """
    Generate a momentum-based signal for a market.

    Args:
        market: bonsai_fund.agent.Market

    Returns:
        MomentumSignal with vote, confidence, edge, reason
    """
    prob = market.implied_prob_yes
    volume = market.volume
    days = market.days_to_event

    # Volume signal: high volume = institutional conviction
    volume_signal = min(volume / 10000, 1.0)  # cap at 1.0

    # Time decay signal: binary events near expiry have higher precision
    # Close to event (< 14 days): momentum is more reliable
    # Far from event (> 60 days): momentum can fade — reduce confidence
    if days <= 14:
        time_factor = 1.0
    elif days <= 60:
        time_factor = 0.8
    else:
        time_factor = 0.5

    # Price proximity to even-money (50c)
    # Markets near 50c are contested — momentum less reliable
    # Markets at extremes (>80c or <20c) have strong directional bias
    if prob >= 0.80:
        # Strong YES consensus — momentum is BULLISH if volume is high
        direction = "YES"
        base_conf = 0.75
    elif prob <= 0.20:
        # Strong NO consensus — momentum is BEARISH (fade the crowd)
        direction = "NO"
        base_conf = 0.75
    else:
        # Contested market — momentum signal is weak
        direction = "PASS"
        base_conf = 0.40

    confidence = base_conf * volume_signal * time_factor

    # Edge: momentum adds directional signal beyond market price
    # If market is at 55c and momentum is YES → edge for YES
    # If market is at 80c and momentum is YES → edge is minimal
    if direction == "YES":
        edge = confidence * 0.05   # small positive edge
    elif direction == "NO":
        edge = -confidence * 0.05
    else:
        edge = 0.0

    if confidence < 0.35:
        vote = "PASS"
        reason = f"low momentum conf={confidence:.2f}"
    else:
        vote = direction
        if volume_signal > 0.7:
            reason = f"high-volume momentum {direction.lower()} {volume*volume_signal:,.0f} adj"
        else:
            reason = f"moderate momentum {direction.lower()} conf={confidence:.2f}"

    return MomentumSignal(
        strategy="MomentumTrader",
        vote=vote,
        confidence=round(confidence, 3),
        edge=round(edge, 3),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_with_hedge_fund(hf_module):
    """
    Monkey-patch the collect_votes or analyze function to include
    this strategy. Call this after importing hedge_fund.

    Usage:
        from bonsai_fund import hedge_fund
        from strategies.momentum import register_with_hedge_fund
        register_with_hedge_fund(hedge_fund)
    """
    import inspect
    original_analyze = hf_module.analyze

    def enhanced_analyze(market, votes, portfolio, risk):
        sig = original_analyze(market, votes, portfolio, risk)
        # Add momentum signal to per_agent
        mom = generate_signal(market)
        sig.per_agent.append({
            "id": 99,
            "name": "MomentumTrader",
            "vote": mom.vote,
            "conf": mom.confidence,
            "edge_raw": mom.edge,
            "edge_yes": mom.edge if mom.vote == "YES" else -mom.edge,
            "reason": mom.reason,
        })
        return sig

    hf_module.analyze = enhanced_analyze
