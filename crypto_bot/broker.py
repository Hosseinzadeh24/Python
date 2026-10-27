"""Order execution.

PaperBroker: simulated fills with fees and slippage. Default and recommended first step.
LiveBroker:  real spot orders through ccxt. Refuses to start unless BOT_LIVE_CONFIRM is set
             to I_ACCEPT_RISK, so it cannot be enabled by accident.
Both expose the same methods: buy(), sell(), equity(), positions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .risk import RiskLimits


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    notional: float
    stop: float
    take: float
    entry_time: str
    entry_bar: int = 0


@dataclass
class PaperBroker:
    capital: float
    limits: RiskLimits
    state_path: Path
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    halted: bool = False
    day: str = ""
    day_start_equity: float = 0.0

    def __post_init__(self):
        self.state_path = Path(self.state_path)
        if self.state_path.exists():
            self.load()
        else:
            self.cash = self.capital
            self.save()

    # --- persistence ----------------------------------------------------------
    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "capital": self.capital, "cash": self.cash, "halted": self.halted,
            "day": self.day, "day_start_equity": self.day_start_equity,
            "positions": {k: vars(v) for k, v in self.positions.items()},
            "trades": self.trades[-500:],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)          # atomic write: safe if the process is killed

    def load(self) -> None:
        d = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.capital, self.cash, self.halted = d["capital"], d["cash"], d["halted"]
        self.day, self.day_start_equity = d.get("day", ""), d.get("day_start_equity", 0.0)
        self.positions = {k: Position(**v) for k, v in d["positions"].items()}
        self.trades = d.get("trades", [])

    # --- accounting -----------------------------------------------------------
    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.qty * prices[s] for s, p in self.positions.items() if s in prices)

    # --- orders ---------------------------------------------------------------
    def buy(self, symbol: str, notional: float, price: float, stop_pct: float, take_pct: float,
            when: str, bar: int = 0) -> Position | None:
        notional = min(notional, self.cash)
        if notional <= 0 or symbol in self.positions or self.halted:
            return None
        fill = price * (1 + self.limits.slippage)
        qty = notional * (1 - self.limits.fee_rate) / fill
        self.cash -= notional
        pos = Position(symbol, qty, fill, notional, fill * (1 - stop_pct), fill * (1 + take_pct), when, bar)
        self.positions[symbol] = pos
        self.trades.append({"time": when, "symbol": symbol, "side": "buy", "price": fill,
                            "qty": qty, "notional": notional, "reason": "entry"})
        return pos

    def sell(self, symbol: str, price: float, reason: str, when: str) -> dict | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        fill = price * (1 - self.limits.slippage)
        proceeds = pos.qty * fill * (1 - self.limits.fee_rate)
        self.cash += proceeds
        trade = {"time": when, "symbol": symbol, "side": "sell", "price": fill, "qty": pos.qty,
                 "pnl": proceeds - pos.notional, "return": proceeds / pos.notional - 1, "reason": reason}
        self.trades.append(trade)
        return trade

    def roll_day(self, equity: float, now: datetime) -> None:
        """Reset the daily loss reference at the start of each UTC day."""
        today = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.day_start_equity = today, equity


class LiveBroker:
    """Real spot trading through ccxt. Use only after paper results and with a small capital."""

    def __init__(self, exchange: str, limits: RiskLimits, state_path: Path, capital: float):
        if os.environ.get("BOT_LIVE_CONFIRM") != "I_ACCEPT_RISK":
            raise RuntimeError("Live trading disabled. Set BOT_LIVE_CONFIRM=I_ACCEPT_RISK to enable.")
        import ccxt  # noqa: WPS433 (lazy import)

        key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not key or not secret:
            raise RuntimeError("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET environment variables.")
        self.ex = getattr(ccxt, exchange)({"apiKey": key, "secret": secret, "enableRateLimit": True})
        self.limits = limits
        # local ledger mirrors what the bot did, so the same risk rules apply
        self.ledger = PaperBroker(capital, limits, state_path)

    @property
    def positions(self):
        return self.ledger.positions

    @property
    def trades(self):
        return self.ledger.trades

    @property
    def halted(self):
        return self.ledger.halted

    @property
    def day(self):
        return self.ledger.day

    def equity(self, prices):
        return self.ledger.equity(prices)

    def save(self):
        self.ledger.save()

    def roll_day(self, equity, now):
        self.ledger.roll_day(equity, now)

    def buy(self, symbol, notional, price, stop_pct, take_pct, when, bar=0):
        notional = min(notional, self.ledger.cash)
        if notional < 5:                           # most exchanges reject tiny orders
            return None
        amount = self.ex.amount_to_precision(symbol, notional / price)
        order = self.ex.create_market_buy_order(symbol, float(amount))
        filled = float(order.get("filled") or amount)
        avg = float(order.get("average") or price)
        fill_notional = filled * avg
        self.ledger.cash -= fill_notional
        pos = Position(symbol, filled * (1 - self.limits.fee_rate), avg, fill_notional,
                       avg * (1 - stop_pct), avg * (1 + take_pct), when, bar)
        self.ledger.positions[symbol] = pos
        self.ledger.trades.append({"time": when, "symbol": symbol, "side": "buy", "price": avg,
                                   "qty": pos.qty, "notional": fill_notional, "reason": "entry",
                                   "order_id": order.get("id")})
        return pos

    def sell(self, symbol, price, reason, when):
        pos = self.ledger.positions.get(symbol)
        if pos is None:
            return None
        amount = self.ex.amount_to_precision(symbol, pos.qty)
        order = self.ex.create_market_sell_order(symbol, float(amount))
        avg = float(order.get("average") or price)
        proceeds = float(order.get("filled") or pos.qty) * avg
        self.ledger.positions.pop(symbol)
        self.ledger.cash += proceeds
        trade = {"time": when, "symbol": symbol, "side": "sell", "price": avg, "qty": pos.qty,
                 "pnl": proceeds - pos.notional, "return": proceeds / pos.notional - 1,
                 "reason": reason, "order_id": order.get("id")}
        self.ledger.trades.append(trade)
        return trade
