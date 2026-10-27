"""Turns model output and market state into advice: BUY / SELL / HOLD / WAIT plus alerts.

This module never places orders. The trader (trader.py) decides what to do with advice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features import atr, build_features, forward_labels
from .models import Ensemble
from .risk import RiskLimits, expected_net_edge, position_fraction
from .simulator import TradeParams


@dataclass
class Holding:
    symbol: str
    qty: float
    entry_price: float
    stop_price: float | None = None
    take_price: float | None = None


@dataclass
class Advice:
    symbol: str
    time: str
    price: float
    action: str                      # BUY / SELL / HOLD / WAIT
    reason: str
    prob_up: float = 0.0
    net_edge: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    trend: str = "-"
    ret_1h: float = 0.0
    vol_z: float = 0.0
    buy_pressure: float | None = None
    amount_quote: float = 0.0        # USDT to spend (BUY) or receive (SELL)
    qty: float = 0.0
    stop_price: float | None = None
    take_price: float | None = None
    suggested_stop: float | None = None
    alerts: list[str] = field(default_factory=list)


def limits_from(cfg: Config) -> RiskLimits:
    return RiskLimits(cfg.fee_rate, cfg.slippage, cfg.kelly_fraction, cfg.max_position_frac,
                      cfg.max_daily_loss_frac, cfg.max_drawdown_kill)


def load_holdings(path: Path) -> list[Holding]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Holding(**h) for h in raw.get("holdings", []) if float(h.get("qty", 0)) > 0]


def load_params(cfg: Config) -> dict[str, TradeParams]:
    p = cfg.out / "advisor_params.json"
    if not p.exists():
        return {}
    return {s: TradeParams(**v) for s, v in json.loads(p.read_text(encoding="utf-8")).items()}


def default_params(cfg: Config) -> TradeParams:
    return TradeParams(p_enter=cfg.p_enter, p_exit=cfg.p_exit, max_hold_bars=cfg.max_hold_bars,
                       stop_pct=cfg.stop_frac, take_pct=cfg.take_frac)


def train_model(df: pd.DataFrame, cfg: Config, limits: RiskLimits) -> Ensemble:
    """Fit on the last `train_bars` candles. Labels only use bars that are already closed."""
    feats = build_features(df)
    _, y = forward_labels(df, cfg.horizon_bars, limits.round_trip_cost)
    ok = feats.notna().all(axis=1) & y.notna()
    n = len(df)
    ok.iloc[max(0, n - cfg.horizon_bars):] = False      # their label would need future candles
    ok.iloc[: max(0, n - cfg.train_bars)] = False        # rolling training window
    return Ensemble(cfg.seed).fit(feats[ok], y[ok])


def advise(symbol: str, df: pd.DataFrame, price: float, prm: TradeParams, cfg: Config,
           limits: RiskLimits, holding: Holding | None, exposure_left: float) -> Advice:
    """df: closed candles only. price: current market price (ticker)."""
    when = str(df.index[-1])
    feats = build_features(df)
    last = feats.iloc[-1]
    if last.isna().any():
        return Advice(symbol, when, price, "WAIT", "not enough history yet")

    model = train_model(df, cfg, limits)
    p = float(model.predict_proba(feats.iloc[[-1]]).iloc[0])
    a = float(atr(df).iloc[-1])
    # Your own percentages (max loss / min profit) replace the ATR-based levels.
    stop_pct, take_pct = cfg.stop_frac, cfg.take_frac
    edge = expected_net_edge(p, stop_pct, take_pct, limits.round_trip_cost)
    trend = "UP" if last["sma10_sma30"] > 0 else "DOWN"
    buy_pressure = float(last["taker_ratio_6"]) if "taker_ratio_6" in last and last["taker_ratio_6"] == last["taker_ratio_6"] else None
    adv = Advice(symbol, when, price, "WAIT", "", prob_up=p, net_edge=edge, atr=a,
                 rsi=float(last["rsi_14"]) * 100, trend=trend,
                 ret_1h=float(np.exp(last["ret_1"]) - 1), vol_z=float(last["vol_z"]),
                 buy_pressure=buy_pressure)
    bp_txt = f"buy pressure 6h {buy_pressure:.0%}" if buy_pressure is not None else "buy pressure n/a"

    if holding is not None:
        stop = holding.stop_price or holding.entry_price * (1 - stop_pct)
        take = holding.take_price or holding.entry_price * (1 + take_pct)
        trail = price * (1 - stop_pct)                      # hint only: a stop may only move UP
        adv.stop_price, adv.take_price = stop, take
        adv.qty = holding.qty
        adv.amount_quote = holding.qty * price
        if price <= stop:
            adv.action, adv.reason = "SELL", f"stop-loss level {stop:.6g} reached"
        elif price >= take:
            adv.action, adv.reason = "SELL", f"take-profit level {take:.6g} reached"
        elif p < prm.p_exit:
            adv.action, adv.reason = "SELL", f"model turned bearish: P(up)={p:.0%} < {prm.p_exit:.0%}"
        else:
            adv.action = "HOLD"
            adv.reason = f"trend intact, P(up)={p:.0%}, {bp_txt}"
            if trail > stop:
                adv.suggested_stop = trail
                adv.reason += f"; consider raising stop to {trail:.6g}"
        if adv.action == "SELL":
            adv.alerts.append("SELL_SIGNAL")
        return adv

    # not holding: is there an entry?
    if p >= prm.p_enter and edge > 0:
        frac = position_fraction(p, stop_pct, take_pct, limits)
        amount = min(cfg.capital * frac, max(exposure_left, 0.0))
        if amount < cfg.min_order_quote:
            adv.reason = ("exposure limit reached" if exposure_left < cfg.min_order_quote
                          else "position size below minimum order")
        else:
            adv.action = "BUY"
            adv.amount_quote = amount
            adv.qty = amount / price
            adv.stop_price = price * (1 - stop_pct)
            adv.take_price = price * (1 + take_pct)
            adv.reason = f"P(up)={p:.0%} >= {prm.p_enter:.0%}, net edge {edge:+.2%}, {bp_txt}"
            adv.alerts.append("BUY_SIGNAL")
    else:
        adv.reason = (f"P(up)={p:.0%} below entry level {prm.p_enter:.0%}" if p < prm.p_enter
                      else f"no positive expected edge ({edge:+.2%} after costs)")

    if p >= cfg.alert_min_prob and edge >= cfg.alert_min_net_edge:
        adv.alerts.append("OPPORTUNITY")
    if abs(adv.ret_1h) >= cfg.sharp_move_pct:
        adv.alerts.append("SHARP_MOVE")
    if adv.vol_z >= cfg.volume_spike_z:
        adv.alerts.append("UNUSUAL_VOLUME")
    return adv
