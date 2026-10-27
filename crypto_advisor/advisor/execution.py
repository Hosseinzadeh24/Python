"""Order execution behind one interface.

WallexExecutor : real limit orders on Wallex (needs your API key).
SimExecutor    : simulated limit orders filled against REAL prices (no money, for testing).
The trader uses only these methods, so the same logic runs in both modes.
Balances are per asset (USDT, TMN, BTC, ...). Each market uses its own quote asset.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from .wallex import MarketMeta, OrderInfo, WallexClient, base_quote, fmt_fixed, parse_order, quote_of

PricesFn = Callable[[str], tuple[float, float]]        # symbol -> (bid, ask)


class ExecutionError(RuntimeError):
    pass


class SimExecutor:
    is_live = False

    def __init__(self, metas: dict[str, MarketMeta], prices: PricesFn,
                 start_balances: dict[str, float] | None = None, fee_rate: float = 0.002):
        self.metas = metas
        self.prices = prices
        self.fee = fee_rate
        self.balance: dict[str, float] = {k: float(v) for k, v in (start_balances or {}).items()}
        self.orders: dict[str, OrderInfo] = {}
        self.reserved: dict[str, float] = {}           # cid -> quote (BUY) or base (SELL) amount held back

    # ---- persistence (dry-run balances survive a restart together with the trader state)
    def save(self, path) -> None:
        data = {"balance": self.balance, "reserved": self.reserved,
                "orders": {k: asdict(v) for k, v in self.orders.items()}}
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(Path(path))

    def load(self, path) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.balance = {k: float(v) for k, v in data.get("balance", {}).items()}
        self.reserved = {k: float(v) for k, v in data.get("reserved", {}).items()}
        self.orders = {k: OrderInfo(**v) for k, v in data.get("orders", {}).items()}

    # ---- queries
    def meta(self, symbol: str) -> MarketMeta:
        if symbol not in self.metas:
            raise ExecutionError(f"unknown market {symbol}")
        return self.metas[symbol]

    def free_quote(self, symbol: str) -> float:
        q = quote_of(symbol)
        held = sum(v for k, v in self.reserved.items()
                   if self.orders[k].side == "BUY" and quote_of(self.orders[k].symbol) == q)
        return self.balance.get(q, 0.0) - held

    def free_base(self, symbol: str) -> float:
        base = base_quote(symbol)[0]
        held = sum(v for k, v in self.reserved.items()
                   if self.orders[k].side == "SELL" and base_quote(self.orders[k].symbol)[0] == base)
        return self.balance.get(base, 0.0) - held

    # ---- orders
    def place_limit(self, side: str, symbol: str, price: float, qty: float, client_id: str) -> OrderInfo:
        side = side.upper()
        if client_id in self.orders:
            raise ExecutionError("duplicate client id")
        info = OrderInfo(client_id, symbol, side, price, qty, 0.0, 0.0, "NEW", True)
        self.orders[client_id] = info
        if side == "BUY":
            need = qty * price
            if need > self.free_quote(symbol) + 1e-9:
                self.orders.pop(client_id)
                raise ExecutionError(f"not enough {quote_of(symbol)}: need {need:.2f}")
            self.reserved[client_id] = need
        else:
            if qty > self.free_base(symbol) + 1e-12:
                self.orders.pop(client_id)
                raise ExecutionError(f"not enough {base_quote(symbol)[0]} to sell {qty}")
            self.reserved[client_id] = qty
        self._try_fill(client_id, *self.prices(symbol))
        return self.orders[client_id]

    def _try_fill(self, cid: str, bid: float, ask: float) -> None:
        o = self.orders[cid]
        if not o.active:
            return
        if o.side == "BUY" and ask > 0 and ask <= o.price:
            px = min(o.price, ask)
        elif o.side == "SELL" and bid > 0 and bid >= o.price:
            px = max(o.price, bid)
        else:
            return
        base, q = base_quote(o.symbol)
        if o.side == "BUY":
            self.balance[q] = self.balance.get(q, 0.0) - o.qty * px
            self.balance[base] = self.balance.get(base, 0.0) + o.qty * (1 - self.fee)
        else:
            self.balance[base] = self.balance.get(base, 0.0) - o.qty
            self.balance[q] = self.balance.get(q, 0.0) + o.qty * px * (1 - self.fee)
        self.reserved.pop(cid, None)
        self.orders[cid] = replace(o, executed_qty=o.qty, avg_price=px, status="FILLED", active=False)

    def tick(self) -> None:
        """Called by the bot with fresh prices: fills resting orders the market has reached."""
        for cid, o in list(self.orders.items()):
            if o.active:
                self._try_fill(cid, *self.prices(o.symbol))

    def get(self, client_id: str) -> OrderInfo:
        if client_id not in self.orders:
            raise ExecutionError(f"unknown order {client_id}")
        return self.orders[client_id]

    def cancel(self, client_id: str) -> OrderInfo:
        o = self.get(client_id)
        if o.active:
            self.orders[client_id] = replace(o, status="CANCELED", active=False)
            self.reserved.pop(client_id, None)
        return self.orders[client_id]

    def open_order_ids(self) -> list[tuple[str, str]]:
        return [(cid, o.symbol) for cid, o in self.orders.items() if o.active]


class WallexExecutor:
    is_live = True

    def __init__(self, client: WallexClient):
        self.client = client
        self._metas: dict[str, MarketMeta] = {}

    def load_markets(self, metas: dict[str, MarketMeta]) -> None:
        self._metas = metas

    def meta(self, symbol: str) -> MarketMeta:
        if symbol not in self._metas:
            raise ExecutionError(f"unknown Wallex market {symbol}")
        return self._metas[symbol]

    def _free(self, asset: str) -> float:
        b = self.client.balances().get(asset, {})
        return max(0.0, float(b.get("value", 0) or 0) - float(b.get("locked", 0) or 0))

    def free_quote(self, symbol: str) -> float:
        return self._free(quote_of(symbol))

    def free_base(self, symbol: str) -> float:
        return self._free(base_quote(symbol)[0])

    def place_limit(self, side: str, symbol: str, price: float, qty: float, client_id: str) -> OrderInfo:
        m = self.meta(symbol)
        res = self.client.place_limit(symbol, side, fmt_fixed(price, m.price_dp),
                                      fmt_fixed(qty, m.qty_dp), client_id)
        return parse_order(res, client_id)

    def get(self, client_id: str, symbol: str = "") -> OrderInfo:
        return parse_order(self.client.order(client_id), client_id)

    def cancel(self, client_id: str) -> OrderInfo:
        self.client.cancel(client_id)
        time.sleep(0.3)
        return self.get(client_id)

    def open_order_ids(self) -> list[tuple[str, str]]:
        return [(str(o.get("clientOrderId", "")), str(o.get("symbol", ""))) for o in self.client.open_orders()]
