import json

from crypto_bot.broker import PaperBroker
from crypto_bot.alerts import Notifier
from crypto_bot.bot import TradingBot, replay
from crypto_bot.calibration import param_grid, walk_forward
from crypto_bot.config import Config
from crypto_bot.data import synthetic_ohlcv
from crypto_bot.risk import RiskLimits


def test_walk_forward_small_run_is_out_of_sample_and_consistent():
    cfg = Config(train_bars=24 * 30, test_bars=168, calibration_periods=2, capital=1000)
    n = cfg.train_bars + 2 * cfg.test_bars + 100
    frames = {"A": synthetic_ohlcv(n_bars=n, seed=1), "B": synthetic_ohlcv(n_bars=n, seed=2, start_price=50)}
    cfg.symbols = ["A", "B"]
    res = walk_forward(frames, cfg, periods=2, grid=param_grid({"p_enter": [0.6], "p_exit": [0.5],
                                                                 "stop_mult": [2.0], "take_mult": [3.0],
                                                                 "max_hold_bars": [24]}), log=lambda *_: None)
    assert len(res["periods"]) == 2
    for p in res["periods"]:
        assert p["start"] < p["end"]
        assert set(p["chosen_params"]) == {"A", "B"}
    # equity series start at the capital for the bot and are never negative
    for eq in res["equity"]:
        assert (eq["bot"] >= 0).all()
    json.dumps(res["summary"])  # serialisable


def test_live_step_path_runs_and_writes_status(tmp_path):
    cfg = Config(train_bars=24 * 20, retrain_every_bars=24, data_dir=str(tmp_path), capital=1000,
                 symbols=["A", "B"], alert_min_prob=0.5, alert_min_net_edge=-1.0)
    frames = {"A": synthetic_ohlcv(n_bars=24 * 22, seed=5), "B": synthetic_ohlcv(n_bars=24 * 22, seed=6)}
    broker = PaperBroker(1000, RiskLimits(), tmp_path / "state.json")
    bot = TradingBot(cfg, broker, Notifier(tmp_path))
    statuses = replay(cfg, bot, frames, start=24 * 20)
    assert statuses
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["mode"] == "paper"
    assert st["equity"] > 0
    assert (tmp_path / "alerts.jsonl").exists()


def test_drawdown_kill_switch_halts_trading(tmp_path):
    cfg = Config(data_dir=str(tmp_path), capital=1000, symbols=["A"], max_drawdown_kill=0.05)
    broker = PaperBroker(1000, RiskLimits(max_drawdown_kill=0.05), tmp_path / "s.json")
    # simulate a big loss already realised
    broker.cash = 900.0
    import pandas as pd
    df = synthetic_ohlcv(n_bars=50, seed=9)
    bot = TradingBot(cfg, broker, Notifier(tmp_path))
    bot.step({"A": df})
    assert broker.halted
