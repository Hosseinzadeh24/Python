"""Ties everything together: real-time prices (polling), analysis every cycle, trading, alerts.

Modes (set in config.json or the GUI):
  suggest  : analysis and alerts only; no orders
  dry_run  : everything runs, orders are SIMULATED with real Wallex prices (no money)
  live     : real orders on Wallex with your API key
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .execution import ExecutionError, SimExecutor, WallexExecutor
from .market import MarketDataError, fetch_candles
from .notify import Notifier, format_alert
from .strategy import Advice, Holding, advise, default_params, limits_from, load_params
from .trader import PREFIX, Trader
from .wallex import WallexClient, parse_markets, parse_snapshot, quote_of

ALERT_KINDS_IMPORTANT = ("BUY_SIGNAL", "SELL_SIGNAL", "OPPORTUNITY")
PREFIX_ORDERS = PREFIX               # client-id prefix of every order this app places
KILL_FLAG = "KILL_SWITCH"            # file written by the kill switch (window or command line)


def write_kill_flag(state_dir: Path, why: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / KILL_FLAG).write_text(f"{datetime.now().isoformat(timespec='seconds')} {why}\n", encoding="utf-8")


def clear_kill_flag(state_dir: Path) -> None:
    (state_dir / KILL_FLAG).unlink(missing_ok=True)


class Bot:
    def __init__(self, cfg: Config, api_key: str | None = None, log=print, state_dir: Path | None = None):
        self.cfg = cfg
        self.log = log
        self.client = WallexClient(api_key)
        self.notifier = Notifier(cfg)
        self.state_dir = state_dir or cfg.out
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.metas: dict = {}
        self.snapshot: dict[str, dict] = {}
        self.last_advice: dict[str, Advice] = {}
        self.errors: dict[str, str] = {}
        self.last_cycle = 0.0
        self.executor = self._make_executor()
        self.sim_path = self.state_dir / f"sim_{cfg.trading_mode}.json"
        if isinstance(self.executor, SimExecutor):
            self.executor.load(self.sim_path)
        self.trader = Trader(cfg, self.executor, self.state_dir / f"trader_{cfg.trading_mode}.json", log=self.log)

    # ---------------------------------------------------------------- setup
    def _make_executor(self):
        if self.cfg.trading_mode == "live":
            if not self.client.api_key:
                raise ExecutionError("live mode needs your Wallex API key")
            return WallexExecutor(self.client)
        return SimExecutor({}, self.price_pair,
                           start_balances={"USDT": self.cfg.capital, "TMN": self.cfg.capital_tmn},
                           fee_rate=self.cfg.fee_rate)

    def check_key(self) -> str:
        """Test that the key works (read-only). Returns a short message."""
        if not self.client.api_key:
            return "no API key"
        bal = self.client.balances()
        return f"key works: {len(bal)} assets on the account"

    def refresh_markets(self) -> None:
        raw = self.client.markets()
        self.metas = parse_markets(raw)
        self.snapshot = parse_snapshot(raw)
        if isinstance(self.executor, WallexExecutor):
            self.executor.load_markets(self.metas)
        else:
            self.executor.metas = self.metas

    def price_pair(self, symbol: str) -> tuple[float, float]:
        s = self.snapshot.get(symbol)
        if not s:
            return 0.0, 0.0
        return s["bid"], s["ask"]

    def bids(self) -> dict[str, float]:
        return {s: v["bid"] for s, v in self.snapshot.items()}

    # ---------------------------------------------------------------- fast loop
    def tick(self) -> list[str]:
        """Every few seconds: fresh prices, then stop-loss / take-profit / fill checks."""
        events: list[str] = []
        if (self.state_dir / KILL_FLAG).exists() and not self.trader.halted:
            self.trader._halt("kill switch was pressed (command line or another window)", "manual")
            events.append("kill switch flag found: new buys stopped. Stop-loss and take-profit still active.")
        try:
            self.refresh_markets()
        except Exception as exc:
            return [f"price update failed (will retry): {exc}"]
        if isinstance(self.executor, SimExecutor):
            self.executor.tick()
        for sym in list(self.trader.positions):
            if sym in self.snapshot:
                bid, ask = self.price_pair(sym)
                events += self.trader.on_tick(sym, bid, ask)
        self._save_sim()
        return events

    def _save_sim(self) -> None:
        if isinstance(self.executor, SimExecutor):
            self.executor.save(self.sim_path)

    # ---------------------------------------------------------------- analysis cycle
    def analyse(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now().astimezone()
        cfg = self.cfg
        limits = limits_from(cfg)
        params = load_params(cfg)
        events: list[str] = []
        advices: list[Advice] = []
        bids = self.bids()
        for sym in cfg.symbols:
            try:
                if sym not in self.snapshot:
                    raise MarketDataError(f"{sym} is not listed on Wallex")
                df = fetch_candles(cfg, self.client, sym, cfg.train_bars + 100)
                self.errors.pop(sym, None)
            except Exception as exc:
                self.errors[sym] = str(exc)[:200]
                continue
            bid, ask = self.price_pair(sym)
            pos = self.trader.positions.get(sym)
            holding = None
            if pos is not None and pos.phase == "holding":
                holding = Holding(sym, pos.qty, pos.entry_price, pos.stop, pos.take)
            q = quote_of(sym)
            scfg = cfg.scoped(q)                     # this market's capital and minimum order
            exposure_left = scfg.capital * cfg.max_total_exposure - self.trader.exposure(bids, q)
            prm = params.get(sym, default_params(cfg))
            price = self.snapshot[sym]["last"] or ask
            adv = advise(sym, df, price, prm, scfg, limits, holding, exposure_left)
            self.last_advice[sym] = adv
            advices.append(adv)
            if cfg.trading_mode != "suggest":
                events += self.trader.on_advice(adv, bid, ask, bids)
            events += self._alerts(adv, now)
        self.notifier.save_state()
        self._save_sim()
        if advices:
            self.notifier.write_history(advices, now)
        self.last_cycle = time.time()
        return events

    def _alerts(self, adv: Advice, now: datetime) -> list[str]:
        out = []
        for kind in adv.alerts:
            if not self.notifier.should_alert(adv.symbol, kind, now):
                continue
            record = {"time": now.isoformat(timespec="seconds"), "kind": kind,
                      **{k: v for k, v in adv.__dict__.items() if k != "alerts"}}
            text = format_alert(adv, kind)
            self.notifier.alert(text, record, kind in ALERT_KINDS_IMPORTANT)
            self.notifier.mark(adv.symbol, kind, now)
            out.append(f"ALERT {kind} {adv.symbol}: P(up) {adv.prob_up:.0%}, edge {adv.net_edge:+.2%}")
        return out

    # ---------------------------------------------------------------- user actions
    def kill_switch(self, sell_all: bool) -> list[str]:
        write_kill_flag(self.state_dir, "window kill switch")
        return self.trader.kill_switch(sell_all, self.bids())

    def resume(self) -> None:
        clear_kill_flag(self.state_dir)
        self.trader.resume()
