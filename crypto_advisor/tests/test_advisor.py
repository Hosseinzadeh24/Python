import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as app  # noqa: E402
from advisor import market  # noqa: E402
from advisor.backtest import walk_forward  # noqa: E402
from advisor.config import Config  # noqa: E402
from advisor.notify import Notifier  # noqa: E402
from advisor.strategy import Holding, advise, default_params, limits_from, train_model  # noqa: E402
from advisor.synthetic import synthetic_ohlcv  # noqa: E402


def _cfg(tmp_path, **kw):
    base = dict(output_dir=str(tmp_path), train_bars=400, capital=1000.0)
    base.update(kw)
    return Config(**base)


def test_sell_when_price_breaks_stop_loss(tmp_path):
    cfg = _cfg(tmp_path)
    df = synthetic_ohlcv(n_bars=500, seed=1)
    close = float(df.close.iloc[-1])
    h = Holding("X", qty=2.0, entry_price=close, stop_price=close * 0.9, take_price=close * 1.5)
    adv = advise("X", df, price=close * 0.85, prm=default_params(cfg), cfg=cfg,
                 limits=limits_from(cfg), holding=h, exposure_left=500)
    assert adv.action == "SELL"
    assert "stop-loss" in adv.reason
    assert adv.qty == 2.0


def test_hold_when_inside_stop_and_target(tmp_path):
    cfg = _cfg(tmp_path)
    df = synthetic_ohlcv(n_bars=500, seed=2)
    close = float(df.close.iloc[-1])
    h = Holding("X", qty=1.0, entry_price=close, stop_price=close * 0.5, take_price=close * 2)
    prm = default_params(cfg)
    prm = type(prm)(**{**prm.__dict__, "p_exit": 0.0})      # model can never trigger an exit
    adv = advise("X", df, price=close, prm=prm, cfg=cfg, limits=limits_from(cfg),
                 holding=h, exposure_left=500)
    assert adv.action == "HOLD"


def test_no_buy_when_exposure_budget_exhausted(tmp_path):
    cfg = _cfg(tmp_path)
    df = synthetic_ohlcv(n_bars=500, seed=3)
    prm = default_params(cfg)
    prm = type(prm)(**{**prm.__dict__, "p_enter": 0.0})     # would otherwise be a BUY candidate
    adv = advise("X", df, price=float(df.close.iloc[-1]), prm=prm, cfg=cfg,
                 limits=limits_from(cfg), holding=None, exposure_left=0.0)
    assert adv.action != "BUY"


def test_quarter_hour_schedule():
    t = datetime(2026, 10, 27, 10, 7, 0)
    assert app.next_quarter_hour(t) == datetime(2026, 10, 27, 10, 15, 20)
    t = datetime(2026, 10, 27, 10, 59, 30)
    assert app.next_quarter_hour(t) == datetime(2026, 10, 27, 11, 0, 20)
    t = datetime(2026, 10, 27, 23, 45, 1)
    assert app.next_quarter_hour(t) == datetime(2026, 10, 28, 0, 0, 20)


def test_forming_candle_is_dropped(monkeypatch):
    now_ms = int(datetime.now().timestamp() * 1000)
    hour = 3_600_000
    rows = []
    for k in range(5):
        open_ms = now_ms - (6 - k) * hour
        close_ms = open_ms + hour - 1
        rows.append([open_ms, 100, 101, 99, 100.5, 10, close_ms, 1000, 5, 6, 600, 0])
    rows[-1][6] = now_ms + hour                               # last candle still forming
    monkeypatch.setattr(market, "_get", lambda urls, path, params: rows)
    df = market.binance_candles("BTCUSDT", "1h", 10, ["x"])
    assert len(df) == 4
    assert {"open", "high", "low", "close", "volume", "taker_buy"} <= set(df.columns)
    assert df["taker_buy"].iloc[0] == 6


def test_alert_deduplication(tmp_path):
    cfg = _cfg(tmp_path, alert_repeat_hours=4)
    n = Notifier(cfg)
    t0 = datetime(2026, 1, 1, 12, 0)
    assert n.should_alert("BTCUSDT", "BUY_SIGNAL", t0)
    n.mark("BTCUSDT", "BUY_SIGNAL", t0)
    assert not n.should_alert("BTCUSDT", "BUY_SIGNAL", t0 + timedelta(hours=1))
    assert n.should_alert("BTCUSDT", "BUY_SIGNAL", t0 + timedelta(hours=4, minutes=1))
    assert n.should_alert("ETHUSDT", "BUY_SIGNAL", t0)          # per coin


def test_walk_forward_runs_and_is_out_of_sample(tmp_path):
    cfg = _cfg(tmp_path, train_bars=24 * 30, test_bars=168, calibration_periods=2)
    n = cfg.train_bars + 2 * cfg.test_bars + 100
    frames = {"A": synthetic_ohlcv(n_bars=n, seed=1), "B": synthetic_ohlcv(n_bars=n, seed=2)}
    cfg.symbols = ["A", "B"]
    from advisor.backtest import param_grid
    grid = param_grid({"p_enter": [0.6], "p_exit": [0.5], "stop_mult": [2.0],
                       "take_mult": [3.0], "max_hold_bars": [24]})
    res = walk_forward(frames, cfg, periods=2, grid=grid, log=lambda *_: None)
    assert len(res["periods"]) == 2
    assert all(p["start"] < p["end"] for p in res["periods"])
    assert all((eq["bot"] >= 0).all() for eq in res["equity"])
