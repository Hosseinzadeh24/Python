"""Position sizing and risk limits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    fee_rate: float = 0.001
    slippage: float = 0.0005
    kelly_fraction: float = 0.25
    max_position_frac: float = 0.25
    max_daily_loss_frac: float = 0.03
    max_drawdown_kill: float = 0.15

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee_rate + self.slippage)


def expected_net_edge(p_up: float, stop_pct: float, take_pct: float, cost: float) -> float:
    """Expected net return per trade (fraction of notional) given a bracket order.

    p_up: probability of hitting take-profit before stop-loss (model's P(up)).
    """
    return p_up * take_pct - (1 - p_up) * stop_pct - cost


def position_fraction(p_up: float, stop_pct: float, take_pct: float, limits: RiskLimits) -> float:
    """Fractional-Kelly share of equity to allocate. Returns 0 when there is no positive edge."""
    if stop_pct <= 0 or take_pct <= 0:
        return 0.0
    b = take_pct / stop_pct                      # payoff odds
    kelly = p_up - (1 - p_up) / b                # full Kelly
    if kelly <= 1e-9:                            # no (or numerically zero) edge
        return 0.0
    return min(limits.kelly_fraction * kelly, limits.max_position_frac)
