"""
Bonsai Fund — Example Strategy: Mean Reversion Agent

Contrarian strategy that fades markets priced far from fair value.
Works best on markets with high volume and extreme pricing.

Method: Fair-value estimation vs market pricing
1. Estimate true probability via base-rate analysis
2. Compare to market-implied probability
3. If deviation > threshold → fade the market
4. Confidence scales with deviation magnitude
"""

from __future__ import annotations
from dataclasses import dataclass


# Historical base rates by category (from historical Kalshi data)
BASE_RATES = {
    "Geopolitics": 0.40,
    "NBA": 0.50,
    "NFL": 0.50,
    "Earnings": 0.55,
    "Jobs": 0.50,
    "CPI": 0.35,
    "Crypto": 0.45,
    "Hurricane": 0.40,
    "Climate": 0.50,
    "Default": 0.50,
}


@dataclass
class MeanReversionSignal:
    strategy: str
    vote: str
    confidence: float
    edge: float
    reason: str


def estimate_fair_value(market) -> tuple[float, str]:
    """Estimate fair probability and category."""
    # Try to match category from series_title
    category = "Default"
    for cat in BASE_RATES:
        if cat.lower() in market.series_title.lower():
            category = cat
            break

    base_rate = BASE_RATES[category]

    # Adjust by time to event
    # Near-term events (< 14 days): market is fairly efficient
    # Medium-term (14-90 days): some mean reversion opportunity
    # Long-term (> 90 days): high uncertainty, base rate dominates
    if market.days_to_event <= 14:
        fair = market.implied_prob_yes * 0.7 + base_rate * 0.3
    elif market.days_to_event <= 90:
        fair = market.implied_prob_yes * 0.4 + base_rate * 0.6
    else:
        fair = market.implied_prob_yes * 0.1 + base_rate * 0.9

    return min(0.99, max(0.01, fair)), category


def generate_signal(market) -> MeanReversionSignal:
    fair_prob, category = estimate_fair_value(market)
    market_prob = market.implied_prob_yes

    deviation = market_prob - fair_prob  # positive = market overpriced YES

    # Only trade if deviation is meaningful
    if abs(deviation) < 0.05:
        return MeanReversionSignal(
            strategy="MeanReversion",
            vote="PASS",
            confidence=0.3,
            edge=0.0,
            reason=f"no deviation market={market_prob:.0%} fair={fair_prob:.0%}",
        )

    # Direction: fade the deviation
    if deviation > 0:
        vote = "NO"   # market too high on YES
        edge = deviation
    else:
        vote = "YES"  # market too low on YES
        edge = -deviation

    # Confidence scales with deviation magnitude
    # 5% deviation → conf=0.55, 20% deviation → conf=0.85
    confidence = min(0.9, 0.35 + abs(deviation) * 3.0)

    # Volume confidence: high volume = more institutional conviction
    # Low volume = can be retail noise
    volume_factor = min(market.volume / 20000, 1.0)
    confidence *= (0.5 + 0.5 * volume_factor)

    if abs(deviation) > 0.15:
        reason = f"strong {category} mean-reversion {vote} dev={deviation:+.0%}"
    else:
        reason = f"moderate {category} fade {vote} dev={deviation:+.0%}"

    return MeanReversionSignal(
        strategy="MeanReversion",
        vote=vote,
        confidence=round(confidence, 3),
        edge=round(edge, 3),
        reason=reason,
    )
