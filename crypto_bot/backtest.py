"""Bar-by-bar trade simulator with fees, slippage, bracket exits and risk limits.

Timing rules (to avoid look-ahead bias):
  * A signal is computed from bar i's CLOSE and the order is filled at bar i+1's OPEN.
  * Stop-loss / take-profit are checked against each bar's high/low.
  * If a bar touches both stop and take, the stop is assumed to hit first (conservative).
Spot long-only: the bot buys and sells the coin, it never shorts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import atr
from .risk import RiskLimits, expected_net_edge, position_fraction

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class TradeParams:
    p_enter: float = 0.60       # enter when model P(up) >= p_enter
    p_exit: float = 0.50        # exit when model P(up) falls below p_exit
    stop_mult: float = 2.0      # stop-loss distance in ATRs
    take_mult: float = 3.0      # take-profit distance in ATRs
    max_hold_bars: int = 24     # time stop


@dataclass
class SimResult:
    equity: pd.Series
    trades: pd.DataFrame
    in_position: pd.Series
    halted: bool


TRADE_COLS = ["entry_time", "exit_time", "entry_price", "exit_price", "qty",
              "pnl", "return", "bars_held", "reason"]


def simulate(ohlc: pd.DataFrame, prob: pd.Series, params: TradeParams, limits: RiskLimits,
             start_equity: float = 1000.0, min_net_edge: float = 0.0) -> SimResult:
    df = ohlc
    n = len(df)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    p = prob.reindex(df.index).to_numpy(float)
    a = atr(df, 14).to_numpy(float)
    days = np.asarray(df.index.floor("D")) if isinstance(df.index, pd.DatetimeIndex) else np.zeros(n)
    slip, fee = limits.slippage, limits.fee_rate
    cost = limits.round_trip_cost

    cash = start_equity
    qty = 0.0
    entry = dict(price=0.0, stop=0.0, take=0.0, i=0, time=None)
    pending = None                    # (notional, stop_pct, take_pct) scheduled for next open
    halted = False                    # max-drawdown kill switch (permanent)
    halted_day = False
    day_start_eq = start_equity
    prev_eq = start_equity
    eq = np.empty(n)
    in_pos = np.zeros(n, dtype=bool)
    trades = []

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty
        fill = raw_price * (1 - slip)
        proceeds = qty * fill * (1 - fee)
        cash += proceeds
        cost_basis = entry["notional"]
        pnl = proceeds - cost_basis
        trades.append([entry["time"], df.index[i], entry["price"], fill, qty, pnl,
                       pnl / cost_basis, i - entry["i"], reason])
        qty = 0.0

    for i in range(n):
        # new calendar day -> reset daily loss tracking
        if i == 0 or days[i] != days[i - 1]:
            day_start_eq = prev_eq
            halted_day = False

        # 1) fill a pending entry at this bar's open
        if pending is not None and not halted and qty == 0 and cash > 0:
            notional, stop_pct, take_pct = pending
            notional = min(notional, cash)
            fill = o[i] * (1 + slip)
            qty = (notional * (1 - fee)) / fill
            cash -= notional
            entry.update(price=fill, stop=fill * (1 - stop_pct), take=fill * (1 + take_pct),
                         i=i, time=df.index[i], notional=notional)
        pending = None

        # 2) manage an open position against this bar's range
        if qty > 0:
            exit_px, reason = None, None
            if o[i] <= entry["stop"]:
                exit_px, reason = o[i], "stop"
            elif l[i] <= entry["stop"]:
                exit_px, reason = entry["stop"], "stop"
            elif o[i] >= entry["take"]:
                exit_px, reason = o[i], "take"
            elif h[i] >= entry["take"]:
                exit_px, reason = entry["take"], "take"
            elif halted:
                exit_px, reason = c[i], "halt"
            elif i - entry["i"] >= params.max_hold_bars:
                exit_px, reason = c[i], "time"
            elif p[i] == p[i] and p[i] < params.p_exit:      # NaN-safe check
                exit_px, reason = c[i], "signal"
            if exit_px is not None:
                close_position(i, exit_px, reason)

        in_pos[i] = qty > 0
        eq[i] = cash + qty * c[i]

        # 3) daily loss limit and max-drawdown kill switch (evaluated at close)
        if eq[i] < day_start_eq * (1 - limits.max_daily_loss_frac):
            halted_day = True
        if eq[i] < start_equity * (1 - limits.max_drawdown_kill) and not halted:
            halted = True
            if qty > 0:
                close_position(i, c[i], "halt")
                in_pos[i] = False
                eq[i] = cash
        prev_eq = eq[i]

        # 4) entry signal at close -> order executes at next bar's open
        if (i < n - 1 and qty == 0 and not halted and not halted_day
                and p[i] == p[i] and p[i] >= params.p_enter and a[i] == a[i] and c[i] > 0):
            stop_pct = params.stop_mult * a[i] / c[i]
            take_pct = params.take_mult * a[i] / c[i]
            if expected_net_edge(p[i], stop_pct, take_pct, cost) > min_net_edge:
                frac = position_fraction(p[i], stop_pct, take_pct, limits)
                if frac > 0:
                    pending = (eq[i] * frac, stop_pct, take_pct)

    # close anything still open at the end of the sample
    if qty > 0:
        close_position(n - 1, c[-1], "end")
        eq[-1] = cash
        in_pos[-1] = False

    equity = pd.Series(eq, index=df.index, name="equity")
    trades_df = pd.DataFrame(trades, columns=TRADE_COLS)
    return SimResult(equity=equity, trades=trades_df,
                     in_position=pd.Series(in_pos, index=df.index), halted=halted)


def metrics(res: SimResult, start_equity: float) -> dict:
    eq = res.equity
    rets = eq.pct_change().dropna()
    dd = eq / eq.cummax() - 1
    t = res.trades
    wins = t[t["pnl"] > 0] if len(t) else t
    gross_win = wins["pnl"].sum() if len(t) else 0.0
    gross_loss = -t.loc[t["pnl"] <= 0, "pnl"].sum() if len(t) else 0.0
    sharpe = (rets.mean() / rets.std() * np.sqrt(HOURS_PER_YEAR)) if rets.std() > 0 else 0.0
    return {
        "start_equity": float(start_equity),
        "end_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / start_equity - 1),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "sharpe_annualised": float(sharpe),
        "trades": int(len(t)),
        "win_rate": float((t["pnl"] > 0).mean()) if len(t) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "exposure": float(res.in_position.mean()),
        "halted": bool(res.halted),
    }
