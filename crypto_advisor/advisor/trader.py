"""Automatic trading logic (buy, stop-loss, take-profit, time exit, kill switch).

Rules
- One position per market. Entry = LIMIT BUY slightly above the ask (fills quickly).
- After the buy fills, a resting LIMIT SELL is placed at entry * (1 + min_profit%) on Wallex.
- Stop-loss is checked by the app every few seconds on the live bid: entry * (1 - max_loss%).
  The app must be running for the stop-loss to act. Exit = LIMIT SELL slightly below bid.
- Money is kept per quote currency: USDT markets use capital, TMN markets use capital_tmn.
  Profit, loss, exposure and limits are never mixed between USDT and TMN.
- Halts: daily loss limit (resumes next day), total drawdown limit (manual resume), kill switch.
- Only orders placed by this app (client id prefix ADV) are ever cancelled.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Config
from .execution import ExecutionError
from .strategy import Advice
from .wallex import QUOTE_ASSETS, MarketMeta, OrderInfo, quote_of, round_down, round_up

PREFIX = "ADV"


@dataclass
class Position:
    symbol: str
    phase: str                         # entering | holding | exiting
    cid: str                           # order currently being worked
    entry_cid: str
    price: float                       # limit price of the current order
    qty_req: float
    t_open: str
    order_ts: float
    entry_price: float = 0.0           # average fill price of the buy
    entry_exec: float = 0.0            # base amount bought (before fee)
    qty: float = 0.0                   # base amount we hold
    stop: float = 0.0
    take: float = 0.0
    take_cid: str | None = None
    sold_qty: float = 0.0
    proceeds: float = 0.0              # quote received for sold_qty (before fee)
    booked: list[str] = field(default_factory=list)
    exit_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cid(kind: str) -> str:
    return f"{PREFIX}{kind}{secrets.token_hex(6)}"       # letters/digits only


def _zero() -> dict[str, float]:
    return {q: 0.0 for q in QUOTE_ASSETS}


def _per_quote(v) -> dict[str, float]:
    """Accepts the new dict form or an old single number (treated as USDT)."""
    if isinstance(v, dict):
        return {q: float(v.get(q, 0.0)) for q in QUOTE_ASSETS}
    out = _zero()
    out["USDT"] = float(v or 0.0)
    return out


class Trader:
    def __init__(self, cfg: Config, executor, state_path: Path, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.ex = executor
        self.path = state_path
        self.log = log
        self.positions: dict[str, Position] = {}
        self.trades: list[dict] = []
        self.day = _now().date().isoformat()
        self.realised_today = _zero()
        self.realised_total = _zero()
        self.halted = False
        self.halt_reason = ""
        self.halt_kind = ""           # "daily" (auto-resume next day) | "manual" | "drawdown"
        self._load()

    # ------------------------------------------------------------ persistence
    def _save(self) -> None:
        data = {"positions": {s: p.to_dict() for s, p in self.positions.items()}, "trades": self.trades,
                "day": self.day, "realised_today": self.realised_today, "realised_total": self.realised_total,
                "halted": self.halted, "halt_reason": self.halt_reason, "halt_kind": self.halt_kind}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.positions = {s: Position(**p) for s, p in d.get("positions", {}).items()}
        self.trades = d.get("trades", [])
        self.day = d.get("day", self.day)
        self.realised_today = _per_quote(d.get("realised_today", 0.0))
        self.realised_total = _per_quote(d.get("realised_total", 0.0))
        self.halted = d.get("halted", False)
        self.halt_reason = d.get("halt_reason", "")
        self.halt_kind = d.get("halt_kind", "")

    # ------------------------------------------------------------ risk limits
    def _roll_day(self) -> None:
        today = _now().date().isoformat()
        if today != self.day:
            self.day = today
            self.realised_today = _zero()
            if self.halted and self.halt_kind == "daily":
                self.halted, self.halt_reason, self.halt_kind = False, "", ""
                self.log("New day: daily-loss halt lifted.")
            self._save()

    def _check_limits(self) -> None:
        if self.halted:
            return
        for q in QUOTE_ASSETS:
            cap = self.cfg.capital_for(q)
            if cap > 0 and self.realised_total[q] <= -self.cfg.max_drawdown_kill * cap:
                self._halt(f"total {q} loss reached {self.cfg.max_drawdown_kill:.0%} of {q} capital", "drawdown")
                return
        for q in QUOTE_ASSETS:
            cap = self.cfg.capital_for(q)
            if cap > 0 and self.realised_today[q] <= -self.cfg.max_daily_loss_frac * cap:
                self._halt(f"daily {q} loss reached {self.cfg.max_daily_loss_frac:.0%} of {q} capital", "daily")
                return

    def _halt(self, reason: str, kind: str) -> None:
        self.halted, self.halt_reason, self.halt_kind = True, reason, kind
        self.log(f"TRADING HALTED: {reason}. New buys are blocked; open positions stay protected.")
        self._save()

    def resume(self) -> None:
        """Manual resume after a manual or drawdown halt (you must choose this)."""
        self.halted, self.halt_reason, self.halt_kind = False, "", ""
        self.log("Trading resumed by user.")
        self._save()

    def exposure(self, bids: dict[str, float], quote: str | None = None) -> float:
        """Value in use. With quote given, only markets quoted in that currency are counted."""
        total = 0.0
        for s, p in self.positions.items():
            if quote is not None and quote_of(s) != quote:
                continue
            if p.phase == "entering":
                total += p.qty_req * p.price
            else:
                px = bids.get(s) or p.entry_price or p.price
                total += max(0.0, p.qty - p.sold_qty) * px
        return total

    # ------------------------------------------------------------ entries
    def on_advice(self, adv: Advice, bid: float, ask: float, bids: dict[str, float]) -> list[str]:
        """Called once per analysis cycle. Opens a position on BUY advice, or exits on SELL advice."""
        ev: list[str] = []
        self._roll_day()
        pos = self.positions.get(adv.symbol)
        if pos is not None and pos.phase == "holding" and adv.action == "SELL":
            ev += self._start_exit(pos, bid, f"model/signal: {adv.reason}"[:120])
            return ev
        if pos is not None or adv.action != "BUY":
            return ev
        if self.halted:
            return [f"{adv.symbol}: BUY skipped, trading halted ({self.halt_reason})"]
        ev += self._open(adv, bid, ask, bids)
        return ev

    def _open(self, adv: Advice, bid: float, ask: float, bids: dict[str, float]) -> list[str]:
        sym = adv.symbol
        q = quote_of(sym)
        if ask <= 0:
            return [f"{sym}: no ask price, skipped"]
        if self.cfg.capital_for(q) <= 0:
            return [f"{sym}: BUY skipped, no {q} capital set (enter it in the window)"]
        try:
            meta = self.ex.meta(sym)
        except ExecutionError as exc:
            return [f"{sym}: {exc}"]
        scfg = self.cfg.scoped(q)
        budget = min(adv.amount_quote, scfg.capital * self.cfg.max_total_exposure - self.exposure(bids, q))
        budget = min(budget, self.ex.free_quote(sym) * 0.995)
        if budget < scfg.min_order_quote:
            return [f"{sym}: BUY skipped, budget {budget:.2f} {q} below minimum {scfg.min_order_quote:g} {q}"]
        price = round_up(ask * (1 + self.cfg.entry_buffer), meta.price_dp)
        qty = round_down(budget / price, meta.qty_dp)
        if qty < meta.min_qty or qty * price < max(meta.min_notional, scfg.min_order_quote):
            return [f"{sym}: BUY skipped, order too small for Wallex minimums"]
        cid = _cid("B")
        try:
            self.ex.place_limit("BUY", sym, price, qty, cid)
        except Exception as exc:                  # network/API errors are reported, not raised
            return [f"{sym}: BUY order failed: {exc}"]
        pos = Position(symbol=sym, phase="entering", cid=cid, entry_cid=cid, price=price, qty_req=qty,
                       t_open=_now().isoformat(timespec="seconds"), order_ts=time.time())
        self.positions[sym] = pos
        self._save()
        mode = "SIMULATED" if not self.ex.is_live else "LIVE"
        return [f"{sym}: {mode} BUY limit {qty} @ {price} (about {qty * price:,.2f} {q})"]

    # ------------------------------------------------------------ monitoring (every tick)
    def on_tick(self, sym: str, bid: float, ask: float) -> list[str]:
        pos = self.positions.get(sym)
        if pos is None:
            return []
        try:
            if pos.phase == "entering":
                return self._check_entry(pos, bid)
            if pos.phase == "holding":
                return self._check_holding(pos, bid)
            if pos.phase == "exiting":
                return self._check_exit(pos, bid)
        except Exception as exc:
            return [f"{sym}: order check failed, will retry: {exc}"]
        return []

    def _age(self, pos: Position) -> float:
        return time.time() - pos.order_ts

    def _cancel_safely(self, cid: str) -> OrderInfo:
        try:
            self.ex.cancel(cid)
        except Exception:
            pass
        return self.ex.get(cid)

    def _check_entry(self, pos: Position, bid: float) -> list[str]:
        o = self.ex.get(pos.cid)
        if not o.terminal:
            if self._age(pos) < self.cfg.order_timeout_s:
                return []
            o = self._cancel_safely(pos.cid)
        if o.executed_qty <= 0:
            del self.positions[pos.symbol]
            self._save()
            return [f"{pos.symbol}: BUY not filled, nothing bought"]
        meta = self.ex.meta(pos.symbol)
        held = min(o.executed_qty, self.ex.free_base(pos.symbol))
        entry_px = o.avg_price or pos.price
        pos.entry_price, pos.entry_exec = entry_px, o.executed_qty
        pos.qty = round_down(held, meta.qty_dp)
        pos.stop = entry_px * (1 - self.cfg.stop_frac)
        pos.take = entry_px * (1 + self.cfg.take_frac)
        pos.phase, pos.cid, pos.price = "holding", "", 0.0
        pos.sold_qty, pos.proceeds, pos.booked = 0.0, 0.0, []
        ev = [f"{pos.symbol}: BOUGHT {pos.qty} @ {entry_px:.8g}. Stop-loss {pos.stop:.8g} "
              f"(-{self.cfg.max_loss_pct:g}%), take-profit {pos.take:.8g} (+{self.cfg.min_profit_pct:g}%)"]
        ev += self._place_take(pos)
        self._save()
        return ev

    def _place_take(self, pos: Position) -> list[str]:
        meta = self.ex.meta(pos.symbol)
        qty = round_down(pos.qty - pos.sold_qty, meta.qty_dp)
        if qty < meta.min_qty:
            return []
        px = round_up(pos.take, meta.price_dp)
        cid = _cid("T")
        try:
            o = self.ex.place_limit("SELL", pos.symbol, px, qty, cid)
        except Exception as exc:
            return [f"{pos.symbol}: take-profit order not placed ({exc}); app will still sell at the target while running"]
        pos.take_cid = cid
        if o.filled:
            return self._after_fill_check(pos, cid)
        return [f"{pos.symbol}: take-profit SELL limit {qty} @ {px} placed on Wallex"]

    def _book_sell(self, pos: Position, o: OrderInfo) -> None:
        """Add a finished sell order to the position once (partial fills included)."""
        if o.client_id in pos.booked:
            return
        pos.booked.append(o.client_id)
        if o.executed_qty > 0:
            pos.sold_qty += o.executed_qty
            pos.proceeds += o.executed_qty * (o.avg_price or o.price)

    def _after_fill_check(self, pos: Position, cid: str) -> list[str]:
        o = self.ex.get(cid)
        if o.filled:
            self._book_sell(pos, o)
            return self._finish_if_done(pos, o.avg_price, "take-profit")
        return []

    def _check_holding(self, pos: Position, bid: float) -> list[str]:
        if pos.take_cid:
            t = self.ex.get(pos.take_cid)
            if t.filled:
                self._book_sell(pos, t)
                pos.take_cid = None
                done = self._finish_if_done(pos, t.avg_price, "take-profit (exchange order)")
                if done:
                    return done
                self._save()                              # leftover dust: keep monitoring
                return []
        reason = ""
        if bid > 0 and bid <= pos.stop:
            reason = f"stop-loss {pos.stop:.8g} hit (-{self.cfg.max_loss_pct:g}%)"
        elif bid > 0 and bid >= pos.take:
            reason = f"take-profit {pos.take:.8g} reached (+{self.cfg.min_profit_pct:g}%)"
        elif (time.time() - datetime.fromisoformat(pos.t_open).timestamp()) > self.cfg.max_hold_hours * 3600:
            reason = f"max holding time {self.cfg.max_hold_hours:g} h reached"
        if reason:
            return self._start_exit(pos, bid, reason)
        return []

    # ------------------------------------------------------------ exits
    def _start_exit(self, pos: Position, bid: float, reason: str) -> list[str]:
        if pos.take_cid:
            t = self._cancel_safely(pos.take_cid)
            self._book_sell(pos, t)
            pos.take_cid = None
            if t.filled:
                return self._finish_if_done(pos, t.avg_price, reason)
        pos.phase, pos.exit_reason = "exiting", reason
        self.log(f"{pos.symbol}: exit started ({reason})")
        ev = [f"{pos.symbol}: EXIT started: {reason}"]
        self._save()
        return ev + self._place_exit(pos, bid)

    def _place_exit(self, pos: Position, bid: float) -> list[str]:
        meta = self.ex.meta(pos.symbol)
        remaining = min(pos.qty - pos.sold_qty, self.ex.free_base(pos.symbol))
        qty = round_down(remaining, meta.qty_dp)
        if self._is_dust(meta, qty, bid):
            return self._finish_if_done(pos, bid, pos.exit_reason, force=True)
        px = round_down(bid * (1 - self.cfg.exit_buffer), meta.price_dp)
        cid = _cid("S")
        try:
            o = self.ex.place_limit("SELL", pos.symbol, px, qty, cid)
        except Exception as exc:
            return [f"{pos.symbol}: exit order failed, retrying: {exc}"]
        pos.cid, pos.price, pos.order_ts = cid, px, time.time()
        self._save()
        if o.filled:
            return self._check_exit(pos, bid)
        return [f"{pos.symbol}: exit SELL limit {qty} @ {px} placed"]

    def _is_dust(self, meta: MarketMeta, qty: float, price: float) -> bool:
        return qty <= 0 or qty < meta.min_qty or (price > 0 and qty * price < meta.min_notional)

    def _check_exit(self, pos: Position, bid: float) -> list[str]:
        o = self.ex.get(pos.cid)
        if not o.terminal:
            if self._age(pos) < self.cfg.order_timeout_s:
                return []
            o = self._cancel_safely(pos.cid)
        self._book_sell(pos, o)
        meta = self.ex.meta(pos.symbol)
        remaining = min(pos.qty - pos.sold_qty, self.ex.free_base(pos.symbol))
        if self._is_dust(meta, round_down(remaining, meta.qty_dp), bid):
            return self._finish_if_done(pos, o.avg_price or o.price, pos.exit_reason, force=True)
        return self._place_exit(pos, bid)

    def _finish_if_done(self, pos: Position, avg_px: float, reason: str, force: bool = False) -> list[str]:
        meta = self.ex.meta(pos.symbol)
        left = pos.qty - pos.sold_qty
        if not force and not self._is_dust(meta, round_down(left, meta.qty_dp), avg_px or pos.entry_price):
            return []
        q = quote_of(pos.symbol)
        sold_px = pos.proceeds / pos.sold_qty if pos.sold_qty else avg_px
        frac = min(1.0, pos.sold_qty / pos.qty) if pos.qty else 1.0
        cost = pos.entry_price * pos.entry_exec * frac            # quote paid for the part we sold
        received = pos.proceeds * (1 - self.cfg.fee_rate)          # quote received after sell fee
        pnl = received - cost
        pnl_pct = (pnl / cost * 100) if cost else 0.0
        self.realised_today[q] += pnl
        self.realised_total[q] += pnl
        self.trades.append({"symbol": pos.symbol, "quote": q, "opened": pos.t_open,
                            "closed": _now().isoformat(timespec="seconds"),
                            "entry": round(pos.entry_price, 10), "exit": round(sold_px, 10),
                            "qty": round(pos.sold_qty, 10), "pnl_quote": round(pnl, 6),
                            "pnl_pct": round(pnl_pct, 4), "reason": reason,
                            "mode": "live" if self.ex.is_live else "dry_run"})
        del self.positions[pos.symbol]
        self._save()
        self._check_limits()
        return [f"{pos.symbol}: CLOSED ({reason}). P&L {pnl:+,.2f} {q} ({pnl_pct:+.2f}%)"]

    # ------------------------------------------------------------ kill switch
    def kill_switch(self, sell_all: bool, bids: dict[str, float]) -> list[str]:
        ev = []
        self._halt("kill switch pressed by user", "manual")
        for cid, sym in self.ex.open_order_ids():
            if not cid.startswith(PREFIX):
                continue                                    # never touch your own orders
            try:
                self.ex.cancel(cid)
                ev.append(f"cancelled {cid} ({sym})")
            except Exception as exc:
                ev.append(f"could not cancel {cid}: {exc}")
        for pos in list(self.positions.values()):
            if pos.take_cid:
                pos.take_cid = None
            if pos.phase == "entering":
                self._roll_back_entry(pos)
        self._save()
        if sell_all:
            for pos in list(self.positions.values()):
                if pos.phase == "holding":
                    ev += self._start_exit(pos, bids.get(pos.symbol, 0.0), "kill switch: sell all")
        else:
            ev.append("positions kept; the app still applies stop-loss and take-profit while it runs")
        return ev

    def _roll_back_entry(self, pos: Position) -> None:
        try:
            o = self.ex.get(pos.cid)
        except Exception:
            return
        if o.executed_qty > 0:
            # bought partly before cancel: keep it as a holding, monitored like any other
            pos.entry_exec = o.executed_qty
            pos.entry_price = o.avg_price or pos.price
            pos.qty = round_down(min(o.executed_qty, self.ex.free_base(pos.symbol)),
                                 self.ex.meta(pos.symbol).qty_dp)
            pos.stop = pos.entry_price * (1 - self.cfg.stop_frac)
            pos.take = pos.entry_price * (1 + self.cfg.take_frac)
            pos.phase, pos.cid, pos.sold_qty, pos.proceeds, pos.booked = "holding", "", 0.0, 0.0, []
        else:
            del self.positions[pos.symbol]

    # ------------------------------------------------------------ status for GUI / logs
    def summary(self, bids: dict[str, float]) -> dict:
        return {"halted": self.halted, "halt_reason": self.halt_reason,
                "realised_today": {q: round(v, 4) for q, v in self.realised_today.items()},
                "realised_total": {q: round(v, 4) for q, v in self.realised_total.items()},
                "exposure": {q: round(self.exposure(bids, q), 4) for q in QUOTE_ASSETS},
                "open_positions": len(self.positions), "trades": len(self.trades)}
