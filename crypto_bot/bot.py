"""The 24/7 trading loop. The same decision code runs in live, paper and replay modes."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .alerts import Notifier, Opportunity
from .backtest import TradeParams
from .config import Config
from .features import atr, build_features, forward_labels
from .models import Ensemble
from .risk import RiskLimits, expected_net_edge, position_fraction

log = logging.getLogger("crypto_bot")


class TradingBot:
    def __init__(self, cfg: Config, broker, notifier: Notifier, params: dict[str, TradeParams] | None = None):
        self.cfg = cfg
        self.broker = broker
        self.notifier = notifier
        self.limits = RiskLimits(cfg.fee_rate, cfg.slippage, cfg.kelly_fraction,
                                 cfg.max_position_frac, cfg.max_daily_loss_frac, cfg.max_drawdown_kill)
        self.params = params or {}
        self.models: dict[str, Ensemble] = {}
        self.trained_at: dict[str, int] = {}
        self.last_probs: dict[str, float] = {}
        self.bar_no = 0                       # monotonic counter of processed bars
        self.data_dir = Path(cfg.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model
    def _model_for(self, sym: str, df: pd.DataFrame) -> Ensemble:
        n = len(df)
        due = sym not in self.models or n - self.trained_at.get(sym, -10**9) >= self.cfg.retrain_every_bars
        if due:
            feats = build_features(df)
            _, y = forward_labels(df, self.cfg.horizon_bars, self.limits.round_trip_cost)
            ok = feats.notna().all(axis=1) & y.notna()
            ok.iloc[max(0, n - self.cfg.horizon_bars):] = False      # labels need future bars
            ok.iloc[: max(0, n - self.cfg.train_bars)] = False        # rolling window
            m = Ensemble(self.cfg.seed).fit(feats[ok], y[ok])
            self.models[sym] = m
            self.trained_at[sym] = n
            log.info("retrained %s on %d rows", sym, int(ok.sum()))
        return self.models[sym]

    # ------------------------------------------------------------------- step
    def step(self, frames: dict[str, pd.DataFrame], now: datetime | None = None) -> dict:
        """Process ONE newly closed bar for every symbol. `frames` must end at the last closed bar."""
        now = now or datetime.now(timezone.utc)
        self.bar_no += 1
        when = str(frames[self.cfg.symbols[0]].index[-1])
        prices = {s: float(frames[s]["close"].iloc[-1]) for s in self.cfg.symbols}
        equity = self.broker.equity(prices)
        self.broker.roll_day(equity, now)

        if not self.broker.halted and equity < self.cfg.capital * (1 - self.cfg.max_drawdown_kill):
            self.broker.halted = True
            log.error("max drawdown reached: trading halted and positions closed")
            for s in list(self.broker.positions):
                self.broker.sell(s, prices[s], "halt", when)
            equity = self.broker.equity(prices)
        halted_day = equity < self.broker.day_start_equity * (1 - self.cfg.max_daily_loss_frac)

        for sym in self.cfg.symbols:
            df = frames[sym]
            last = df.iloc[-1]
            bar_i = self.bar_no
            prm = self.params.get(sym, TradeParams())
            feats = build_features(df)
            if feats.iloc[-1].isna().any():
                continue                                    # not enough history yet
            p_up = float(self._model_for(sym, df).predict_proba(feats.iloc[[-1]]).iloc[0])
            self.last_probs[sym] = p_up
            pos = self.broker.positions.get(sym)

            if pos is not None:
                exit_px, reason = None, None
                if last["open"] <= pos.stop:
                    exit_px, reason = last["open"], "stop"
                elif last["low"] <= pos.stop:
                    exit_px, reason = pos.stop, "stop"
                elif last["open"] >= pos.take:
                    exit_px, reason = last["open"], "take"
                elif last["high"] >= pos.take:
                    exit_px, reason = pos.take, "take"
                elif self.broker.halted:
                    exit_px, reason = last["close"], "halt"
                elif bar_i - pos.entry_bar >= prm.max_hold_bars:
                    exit_px, reason = last["close"], "time"
                elif p_up < prm.p_exit:
                    exit_px, reason = last["close"], "signal"
                if exit_px is not None:
                    t = self.broker.sell(sym, exit_px, reason, when)
                    log.info("SELL %s %s pnl=%.2f (%+.2f%%)", sym, reason, t["pnl"], 100 * t["return"])
                continue

            a = float(atr(df).iloc[-1])
            c = float(last["close"])
            stop_pct, take_pct = prm.stop_mult * a / c, prm.take_mult * a / c
            edge = expected_net_edge(p_up, stop_pct, take_pct, self.limits.round_trip_cost)
            if p_up >= self.cfg.alert_min_prob and edge >= self.cfg.alert_min_net_edge:
                self.notifier.send(Opportunity(sym, when, c, p_up, edge,
                                               c * (1 + take_pct), c * (1 - stop_pct)))
            if (p_up >= prm.p_enter and edge > 0 and not halted_day and not self.broker.halted):
                frac = position_fraction(p_up, stop_pct, take_pct, self.limits)
                notional = equity * frac
                if frac > 0 and self.broker.buy(sym, notional, c, stop_pct, take_pct, when, bar_i):
                    log.info("BUY %s notional=%.2f P(up)=%.2f edge=%+.2f%%", sym, notional, p_up, 100 * edge)

        equity = self.broker.equity(prices)
        self.broker.save()
        status = self.status(prices, equity, when, now)
        (self.data_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
        return status

    def status(self, prices, equity, when, now) -> dict:
        cap = self.cfg.capital
        return {
            "updated_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "last_bar": when,
            "mode": self.cfg.mode,
            "halted": self.broker.halted,
            "equity": round(equity, 4),
            "capital": cap,
            "total_return": round(equity / cap - 1, 6),
            "daily_pnl": round(equity - self.broker.day_start_equity, 4),
            "cash": round(self.broker.cash, 4),
            "positions": {s: {"qty": p.qty, "entry": p.entry_price, "stop": p.stop, "take": p.take,
                              "price": prices.get(s), "entry_time": p.entry_time}
                          for s, p in self.broker.positions.items()},
            "prob_up": {s: round(v, 4) for s, v in self.last_probs.items()},
            "recent_trades": self.broker.trades[-20:],
            "recent_alerts": list(self.notifier.recent)[-10:],
        }


def run_forever(cfg: Config, bot: TradingBot, feed, stop_file: Path | None = None) -> None:
    """Poll the exchange; act once per newly closed candle. Restarts after transient errors."""
    stop_file = stop_file or Path(cfg.data_dir) / "STOP"
    last_bar = None
    backoff = 5
    while True:
        if stop_file.exists():
            log.warning("STOP file found: exiting loop")
            break
        try:
            frames = {s: feed.fetch(s, cfg.timeframe, cfg.train_bars + 200).iloc[:-1] for s in cfg.symbols}
            bar = min(df.index[-1] for df in frames.values())
            if bar != last_bar:
                frames = {s: df.loc[:bar] for s, df in frames.items()}
                bot.step(frames)
                last_bar = bar
            backoff = 5
        except Exception as exc:  # network / exchange errors must not kill the 24/7 loop
            log.exception("loop error: %s (retry in %ss)", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
            continue
        time.sleep(cfg.poll_seconds)


def replay(cfg: Config, bot: TradingBot, frames: dict[str, pd.DataFrame], start: int) -> list[dict]:
    """Feed historical bars one by one through the same step() code (end-to-end test)."""
    n = min(len(f) for f in frames.values())
    statuses = []
    for i in range(start, n):
        sl = {s: f.iloc[: i + 1] for s, f in frames.items()}
        statuses.append(bot.step(sl, now=sl[cfg.symbols[0]].index[-1].to_pydatetime()))
    return statuses
