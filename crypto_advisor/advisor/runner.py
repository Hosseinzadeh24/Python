"""Background worker used by the window (and testable without Tk).

Price checks every tick_seconds, analysis every cycle_minutes, messages go to a queue.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from datetime import datetime

from .bot import Bot
from .config import Config


class Runner:
    """Background thread: price checks every tick_seconds, analysis every cycle_minutes."""

    def __init__(self, cfg: Config, api_key: str | None, out: queue.Queue):
        self.cfg = cfg
        self.out = out
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.bot = Bot(cfg, api_key=api_key, log=lambda m: self.out.put(("log", m)))
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _emit_logs(self, events: list[str]) -> None:
        for e in events:
            self.out.put(("log", f"{datetime.now():%H:%M:%S}  {e}"))

    def _run(self) -> None:
        self.out.put(("log", f"Started in mode '{self.cfg.trading_mode}'. Max loss {self.cfg.max_loss_pct:g}%, "
                             f"min profit {self.cfg.min_profit_pct:g}%."))
        last_cycle = 0.0
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    self._emit_logs(self.bot.tick())
                    if time.time() - last_cycle >= self.cfg.cycle_minutes * 60 or last_cycle == 0.0:
                        self._emit_logs(self.bot.analyse())
                        last_cycle = time.time()
                    self.out.put(("snapshot", self._snapshot()))
            except Exception as exc:
                self.out.put(("log", f"error (will retry): {exc}"))
                self.out.put(("log_detail", traceback.format_exc()))
            self.stop_event.wait(self.cfg.tick_seconds)
        self.out.put(("log", "Stopped."))
        self.out.put(("stopped", None))

    def _snapshot(self) -> dict:
        bids = self.bot.bids()
        rows = []
        for sym in self.cfg.symbols:
            s = self.bot.snapshot.get(sym, {})
            adv = self.bot.last_advice.get(sym)
            pos = self.bot.trader.positions.get(sym)
            rows.append((sym, s.get("bid", 0), s.get("ask", 0), s.get("last", 0), s.get("change_24h", 0),
                         f"{adv.prob_up:.0%}" if adv else "-", adv.action if adv else "-",
                         (pos.phase if pos else "-")))
        return {"rows": rows, "summary": self.bot.trader.summary(bids), "errors": dict(self.bot.errors)}
