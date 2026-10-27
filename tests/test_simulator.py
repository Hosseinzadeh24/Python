import numpy as np
import pandas as pd

from crypto_bot.backtest import TradeParams, metrics, simulate
from crypto_bot.risk import RiskLimits, expected_net_edge, position_fraction

FREE = RiskLimits(fee_rate=0.0, slippage=0.0, kelly_fraction=1.0, max_position_frac=1.0,
                  max_daily_loss_frac=0.99, max_drawdown_kill=0.99)


def _frame(opens, highs, lows, closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": 1.0}, index=idx)


def _flat_atr_frame(n, price=100.0, rng_pct=0.02):
    # constant range so ATR is ~ rng_pct*price
    o = np.full(n, price)
    return _frame(o, o * (1 + rng_pct / 2), o * (1 - rng_pct / 2), o)


def test_entry_fills_next_open_and_take_profit_hits():
    n = 40
    df = _flat_atr_frame(n)
    df.iloc[10:, df.columns.get_loc("open")] = 100.0
    df.iloc[20:, df.columns.get_loc("high")] = 110.0  # big rally to trigger take
    prob = pd.Series(0.9, index=df.index)
    res = simulate(df, prob, TradeParams(p_enter=0.6, p_exit=0.1, stop_mult=2, take_mult=3,
                                         max_hold_bars=100), FREE, start_equity=1000)
    t = res.trades
    assert len(t) >= 1
    first = t.iloc[0]
    # entry is at the open of the bar AFTER the signal (look-ahead protection)
    assert first["entry_time"] == df.index[1]
    assert first["entry_price"] == 100.0
    assert first["reason"] == "take"
    assert first["return"] > 0


def test_stop_loss_limits_loss():
    n = 40
    df = _flat_atr_frame(n)
    df.iloc[5:, df.columns.get_loc("low")] = 90.0  # crash through stop
    prob = pd.Series(0.9, index=df.index)
    res = simulate(df, prob, TradeParams(p_enter=0.6, p_exit=0.1, stop_mult=2, take_mult=3,
                                         max_hold_bars=100), FREE, start_equity=1000)
    stops = res.trades[res.trades["reason"] == "stop"]
    assert len(stops) >= 1
    assert stops.iloc[0]["return"] < 0
    assert stops.iloc[0]["return"] > -0.1  # stop is ~2 ATR (~2%) below entry


def test_fees_and_slippage_reduce_result():
    n = 40
    df = _flat_atr_frame(n)
    prob = pd.Series(0.9, index=df.index)
    prm = TradeParams(p_enter=0.6, p_exit=0.1, stop_mult=2, take_mult=3, max_hold_bars=5)
    free = simulate(df, prob, prm, FREE, start_equity=1000)
    costly = simulate(df, prob, prm, RiskLimits(fee_rate=0.001, slippage=0.001, kelly_fraction=1.0,
                                                max_position_frac=1.0, max_daily_loss_frac=0.99,
                                                max_drawdown_kill=0.99), start_equity=1000)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_no_trades_when_probability_low():
    df = _flat_atr_frame(50)
    res = simulate(df, pd.Series(0.3, index=df.index), TradeParams(), FREE, start_equity=1000)
    assert len(res.trades) == 0
    assert metrics(res, 1000)["total_return"] == 0.0


def test_position_size_zero_without_positive_edge():
    lim = RiskLimits()
    assert position_fraction(0.3, 0.02, 0.03, lim) == 0.0       # negative Kelly
    assert position_fraction(0.4, 0.02, 0.03, lim) == 0.0       # exactly break-even Kelly
    assert position_fraction(0.9, 0.02, 0.03, lim) <= lim.max_position_frac
    assert expected_net_edge(0.5, 0.02, 0.02, 0.0) == 0.0


def test_daily_loss_limit_blocks_new_entries():
    n = 30
    df = _flat_atr_frame(n)
    # price falls steadily inside day 1 -> daily loss triggers -> no entry afterwards that day
    closes = np.linspace(100, 80, n)
    df = _frame(closes, closes * 1.001, closes * 0.999, closes)
    lim = RiskLimits(fee_rate=0, slippage=0, kelly_fraction=1, max_position_frac=1,
                     max_daily_loss_frac=0.05, max_drawdown_kill=0.99)
    res = simulate(df, pd.Series(0.9, index=df.index),
                   TradeParams(p_enter=0.6, p_exit=0.0, stop_mult=50, take_mult=60, max_hold_bars=999),
                   lim, start_equity=1000)
    day1 = res.trades[res.trades["entry_time"].dt.date == df.index[0].date()]
    assert len(day1) == 1                      # no second entry on the same day after the loss
    assert (res.trades["entry_time"].dt.date != df.index[0].date()).any()  # trading resumes next day
