"""Tests for the trading logic. No internet and no real money: SimExecutor fills against controlled prices,
and the Wallex client is replaced by a fake that returns documented response shapes."""
import re
import queue
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advisor.bot import Bot                                     # noqa: E402
from advisor.config import Config, MAX_LOSS_CAP_PCT             # noqa: E402
from advisor.execution import SimExecutor, WallexExecutor       # noqa: E402
from advisor.market import wallex_candles                       # noqa: E402
from advisor.strategy import Advice                             # noqa: E402
from advisor.trader import PREFIX, Position, Trader             # noqa: E402
from advisor.wallex import fmt_fixed, parse_markets, parse_snapshot, quote_of, round_down, round_up  # noqa: E402

MARKETS = {"BTCUSDT": {"baseAsset": "BTC", "quoteAsset": "USDT", "tickSize": 2, "stepSize": 6,
                       "minQty": 1e-6, "minNotional": 10,
                       "stats": {"bidPrice": "99.9", "askPrice": "100", "lastPrice": "99.95", "24h_ch": 1.5}},
           "BTCTMN": {"baseAsset": "BTC", "quoteAsset": "TMN", "tickSize": 0, "stepSize": 6,
                      "minQty": 1e-6, "minNotional": 1000000,
                      "stats": {"bidPrice": "3000000000", "askPrice": "3000100000", "lastPrice": "3000050000",
                                "24h_ch": 0.5}}}


def buy_advice(price=100.0, amount=200.0):
    return Advice("BTCUSDT", "t", price, "BUY", "test", prob_up=0.8, net_edge=0.05, amount_quote=amount)


class Market:
    """Controllable prices shared by the simulator and the trader."""
    def __init__(self, bid=99.9, ask=100.0):
        self.bid, self.ask = bid, ask

    def __call__(self, symbol):
        return self.bid, self.ask

    def move(self, bid, ask=None):
        self.bid, self.ask = bid, (ask if ask is not None else bid + 0.1)


@pytest.fixture
def setup(tmp_path):
    def make(capital=1000.0, start_quote=None, **kw):
        cfg = Config(capital=capital, max_loss_pct=5.0, min_profit_pct=5.0, trading_mode="dry_run",
                     symbols=["BTCUSDT"], output_dir=str(tmp_path), order_timeout_s=45, **kw)
        cfg.validate()
        mkt = Market()
        ex = SimExecutor(parse_markets(MARKETS), mkt,
                         start_balances={"USDT": start_quote if start_quote is not None else capital, "TMN": 0.0},
                         fee_rate=cfg.fee_rate)
        logs = []
        tr = Trader(cfg, ex, tmp_path / "trader.json", log=logs.append)
        return cfg, mkt, ex, tr, logs
    return make


def open_position(mkt, tr, bid=99.9, ask=100.0, amount=200.0):
    mkt.move(bid, ask)
    ev = tr.on_advice(buy_advice(amount=amount), bid, ask, {"BTCUSDT": bid})
    tr.on_tick("BTCUSDT", bid, ask)                     # buy has filled -> take-profit placed
    return ev


# ---------------------------------------------------------------- rules and formats
def test_max_loss_is_hard_capped_at_15():
    with pytest.raises(ValueError):
        Config(max_loss_pct=MAX_LOSS_CAP_PCT + 0.1).validate()
    Config(max_loss_pct=MAX_LOSS_CAP_PCT).validate()
    with pytest.raises(ValueError):
        Config(max_loss_pct=0).validate()


def test_min_profit_must_beat_round_trip_cost():
    with pytest.raises(ValueError):
        Config(min_profit_pct=0.4).validate()


def test_client_ids_are_letters_digits_only(setup):
    _, mkt, ex, tr, _ = setup()
    open_position(mkt, tr)
    for pos in tr.positions.values():
        assert re.fullmatch(r"[A-Za-z0-9]+", pos.entry_cid) and pos.entry_cid.startswith(PREFIX)


def test_wallex_price_and_qty_strings():
    assert fmt_fixed(100.1, 2) == "100.10"
    assert fmt_fixed(0.0000019, 6) == "0.000001"            # never rounds up past what we can afford
    assert round_down(1.9999999, 6) == 1.999999
    assert round_up(100.001, 2) == 100.01


def test_snapshot_parsing_uses_bid_and_ask():
    snap = parse_snapshot(MARKETS)["BTCUSDT"]
    assert snap["bid"] == 99.9 and snap["ask"] == 100.0 and snap["last"] == 99.95


# ---------------------------------------------------------------- trading lifecycle
def test_take_profit_fills_and_books_gain(setup):
    cfg, mkt, ex, tr, _ = setup()
    open_position(mkt, tr)
    pos = tr.positions["BTCUSDT"]
    assert pos.phase == "holding" and pos.take_cid
    assert pos.stop == pytest.approx(100 * 0.95) and pos.take == pytest.approx(105.0)
    mkt.move(105.5, 105.6)
    ex.tick()                                           # market reaches the resting sell
    tr.on_tick("BTCUSDT", 105.5, 105.6)
    assert "BTCUSDT" not in tr.positions
    t = tr.trades[-1]
    assert t["reason"].startswith("take-profit") and t["pnl_quote"] > 0
    # filled at the 105.50 bid (better than the 105.00 limit): 5.5% gross minus about 0.4% fees
    assert t["pnl_pct"] == pytest.approx(5.5 - 2 * cfg.fee_rate * 100, abs=0.1)
    assert tr.realised_total["USDT"] == pytest.approx(t["pnl_quote"])
    assert tr.realised_total["TMN"] == 0


def test_stop_loss_exits_and_books_loss(setup):
    cfg, mkt, ex, tr, logs = setup()
    open_position(mkt, tr)
    mkt.move(94.5, 94.6)
    tr.on_tick("BTCUSDT", 94.5, 94.6)
    assert "BTCUSDT" not in tr.positions
    t = tr.trades[-1]
    assert t["reason"].startswith("stop-loss")
    assert t["pnl_pct"] < -5 and t["pnl_pct"] > -6.5          # -5% plus fees and slippage
    assert ex.free_base("BTCUSDT") < 1e-5                     # rounding dust below Wallex minimum
    assert not ex.open_order_ids()                            # take-profit was cancelled


def test_stop_only_triggers_at_the_stop_not_before(setup):
    _, mkt, ex, tr, _ = setup()
    open_position(mkt, tr)
    mkt.move(96.0, 96.1)
    tr.on_tick("BTCUSDT", 96.0, 96.1)
    assert tr.positions["BTCUSDT"].phase == "holding"


def test_time_stop(setup):
    _, mkt, ex, tr, _ = setup(max_hold_hours=0.0002)     # about 0.7 s
    open_position(mkt, tr)
    time.sleep(1.0)
    mkt.move(100.0, 100.1)
    tr.on_tick("BTCUSDT", 100.0, 100.1)
    assert "BTCUSDT" not in tr.positions and tr.trades[-1]["reason"].startswith("max holding time")


def test_daily_loss_halts_buys_then_new_day_resumes(setup):
    cfg, mkt, ex, tr, logs = setup(capital=200.0, max_daily_loss_frac=0.02)
    open_position(mkt, tr, amount=200.0)
    mkt.move(94.5, 94.6)
    tr.on_tick("BTCUSDT", 94.5, 94.6)                  # about -11 USDT > 2% of 200 = 4
    assert tr.halted and tr.halt_kind == "daily"
    mkt.move(50.0, 50.1)
    msg = tr.on_advice(buy_advice(price=50.0, amount=50.0), 50.0, 50.1, {})
    assert msg and "halted" in msg[0] and "BTCUSDT" not in tr.positions
    tr.day = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    tr._roll_day()
    assert not tr.halted


def test_drawdown_halt_needs_manual_resume(setup):
    cfg, mkt, ex, tr, _ = setup(capital=100.0, max_drawdown_kill=0.05)
    tr.realised_total = {"USDT": -6.0, "TMN": 0.0}
    tr._check_limits()
    assert tr.halted and tr.halt_kind == "drawdown"
    tr._roll_day()
    assert tr.halted                                      # not lifted automatically
    tr.resume()
    assert not tr.halted


def test_kill_switch_cancels_only_our_orders_and_sells_all(setup):
    cfg, mkt, ex, tr, logs = setup()
    open_position(mkt, tr)
    ex.place_limit("BUY", "BTCUSDT", 50.0, 0.5, "MANUAL1")          # your own resting order
    ex.place_limit("BUY", "BTCUSDT", 40.0, 0.5, PREFIX + "Bdeadbeef12")  # ours, resting
    mkt.move(100.0, 100.1)
    ev = tr.kill_switch(sell_all=True, bids={"BTCUSDT": 100.0})
    assert tr.halted and tr.halt_kind == "manual"
    active = dict(ex.open_order_ids())
    assert "MANUAL1" in active                                        # untouched
    assert PREFIX + "Bdeadbeef12" not in active                       # cancelled
    tr.on_tick("BTCUSDT", 100.0, 100.1)
    assert "BTCUSDT" not in tr.positions and ex.free_base("BTCUSDT") < 1e-5


def test_kill_switch_keep_positions_still_protected(setup):
    cfg, mkt, ex, tr, _ = setup()
    open_position(mkt, tr)
    tr.kill_switch(sell_all=False, bids={"BTCUSDT": 100.0})
    assert tr.positions["BTCUSDT"].phase == "holding" and tr.positions["BTCUSDT"].take_cid is None
    mkt.move(94.0, 94.1)
    tr.on_tick("BTCUSDT", 94.0, 94.1)                                  # stop still works
    assert "BTCUSDT" not in tr.positions


def test_insufficient_funds_is_reported_not_raised(setup):
    _, mkt, ex, tr, _ = setup(start_quote=5.0)            # less than the 10 USDT minimum order
    mkt.move(99.9, 100.0)
    ev = tr.on_advice(buy_advice(amount=200.0), 99.9, 100.0, {"BTCUSDT": 99.9})
    assert "BTCUSDT" not in tr.positions and ev and "skipped" in ev[0]


def test_exposure_cap_limits_buy_size(setup):
    cfg, mkt, ex, tr, _ = setup(max_total_exposure=0.1)
    mkt.move(99.9, 100.0)
    tr.on_advice(buy_advice(amount=500.0), 99.9, 100.0, {"BTCUSDT": 99.9})
    pos = tr.positions["BTCUSDT"]
    assert pos.qty_req * pos.price <= 100.0 + 0.2                     # 10% of 1000


def test_state_survives_restart(setup):
    cfg, mkt, ex, tr, _ = setup()
    open_position(mkt, tr)
    tr2 = Trader(cfg, ex, tr.path, log=lambda *_: None)
    assert tr2.positions["BTCUSDT"].phase == "holding"
    assert tr2.positions["BTCUSDT"].take_cid == tr.positions["BTCUSDT"].take_cid


# ---------------------------------------------------------------- Wallex executor with a fake client
class FakeWallex:
    api_key = "TESTKEY123"

    def __init__(self):
        self.placed = []
        self.open = {}

    def balances(self):
        return {"USDT": {"asset": "USDT", "value": "500", "locked": "100"},
                "BTC": {"asset": "BTC", "value": "0.2", "locked": "0"}}

    def place_limit(self, symbol, side, price, quantity, client_id):
        self.placed.append((symbol, side, price, quantity, client_id))
        return {"clientOrderId": client_id, "symbol": symbol, "side": side, "price": price,
                "origQty": quantity, "executedQty": "0", "status": "NEW", "active": True}

    def order(self, cid):
        return {"clientOrderId": cid, "symbol": "BTCUSDT", "side": "BUY", "price": "100.1",
                "origQty": "1.998002", "executedQty": "1.998002", "executedPrice": "100",
                "status": "FILLED", "active": False}

    def cancel(self, cid):
        return {"clientOrderId": cid}

    def open_orders(self, symbol=None):
        return [{"clientOrderId": "MANUAL9", "symbol": "ETHUSDT"}]


def test_wallex_executor_sends_documented_fields():
    fake = FakeWallex()
    ex = WallexExecutor(fake)
    ex.load_markets(parse_markets(MARKETS))
    o = ex.place_limit("BUY", "BTCUSDT", 100.1, 1.998002, "ADVBabc123")
    assert fake.placed[0] == ("BTCUSDT", "BUY", "100.10", "1.998002", "ADVBabc123")
    assert o.status == "NEW" and not o.terminal
    assert ex.free_quote("BTCUSDT") == pytest.approx(400.0) and ex.free_base("BTCUSDT") == pytest.approx(0.2)
    assert ex.free_quote("BTCTMN") == 0.0                               # no TMN in this test wallet
    assert ex.open_order_ids() == [("MANUAL9", "ETHUSDT")]
    assert ex.get("ADVBabc123").filled


# ---------------------------------------------------------------- Wallex candles and a full analysis cycle
def fake_history(n_bars=1500, seed=3):
    rng = np.random.default_rng(seed)
    end = (int(time.time()) // 3600) * 3600
    ts = np.arange(end - n_bars * 3600, end, 3600)          # all closed, last one ends before now
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n_bars)))
    return ts, close


class FakeHistoryClient(FakeWallex):
    def __init__(self):
        super().__init__()
        self.ts, self.close = fake_history()
        self.calls = 0

    def markets(self):
        return MARKETS

    def history(self, symbol, resolution, frm, to):
        self.calls += 1
        mask = (self.ts >= frm) & (self.ts < to)
        t = self.ts[mask]
        c = self.close[mask]
        return {"s": "ok", "t": t.tolist(), "o": [str(x) for x in c], "h": [str(x * 1.001) for x in c],
                "l": [str(x * 0.999) for x in c], "c": [str(x) for x in c], "v": ["10"] * len(c)}


def test_wallex_candles_paginate_and_drop_forming_bar():
    client = FakeHistoryClient()
    df = wallex_candles(client, "BTCUSDT", "1h", 1200)
    assert len(df) == 1200 and df.index.is_unique and df.index.is_monotonic_increasing
    assert df.index[-1] + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz="UTC")
    assert client.calls >= 2                                 # needed more than one request


def test_full_analysis_cycle_in_dry_run(tmp_path):
    cfg = Config(trading_mode="dry_run", symbols=["BTCUSDT"], output_dir=str(tmp_path),
                 train_bars=1000, capital=1000.0)
    bot = Bot(cfg, api_key=None, log=lambda *_: None, state_dir=tmp_path)
    bot.client = FakeHistoryClient()
    bot.refresh_markets()
    events = bot.analyse()
    assert "BTCUSDT" not in bot.errors, bot.errors
    adv = bot.last_advice["BTCUSDT"]
    assert adv.action in ("BUY", "WAIT", "HOLD", "SELL") and 0 <= adv.prob_up <= 1
    assert isinstance(bot.tick(), list)                      # price refresh + checks run without raising
    assert (tmp_path / "suggestions_history.csv").exists()


def test_suggest_mode_never_trades(tmp_path):
    cfg = Config(trading_mode="suggest", symbols=["BTCUSDT"], output_dir=str(tmp_path), train_bars=1000)
    bot = Bot(cfg, api_key=None, log=lambda *_: None, state_dir=tmp_path)
    bot.client = FakeHistoryClient()
    bot.refresh_markets()
    bot.analyse()
    assert bot.trader.positions == {} and bot.executor.orders == {}


def test_live_mode_requires_key(tmp_path):
    from advisor.execution import ExecutionError
    cfg = Config(trading_mode="live", output_dir=str(tmp_path))
    with pytest.raises(ExecutionError):
        Bot(cfg, api_key=None, log=lambda *_: None, state_dir=tmp_path)


def test_runner_thread_runs_and_stops(tmp_path):
    import queue as _q
    from advisor.runner import Runner
    cfg = Config(trading_mode="dry_run", symbols=["BTCUSDT"], output_dir=str(tmp_path),
                 train_bars=1000, capital=1000.0, tick_seconds=2)
    out = _q.Queue()
    r = Runner(cfg, api_key=None, out=out)
    r.bot.client = FakeHistoryClient()
    r.start()
    kinds = set()
    deadline = time.time() + 20
    while time.time() < deadline and "snapshot" not in kinds:
        try:
            kinds.add(out.get(timeout=0.5)[0])
        except _q.Empty:
            pass
    assert "snapshot" in kinds
    r.stop()
    r.thread.join(timeout=10)
    assert not r.thread.is_alive()


def test_kill_flag_from_command_line_halts_running_bot(tmp_path):
    from advisor.bot import KILL_FLAG, clear_kill_flag, write_kill_flag
    cfg = Config(trading_mode="dry_run", symbols=["BTCUSDT"], output_dir=str(tmp_path), train_bars=1000)
    bot = Bot(cfg, api_key=None, log=lambda *_: None, state_dir=tmp_path)
    bot.client = FakeHistoryClient()
    write_kill_flag(tmp_path, "test")
    assert (tmp_path / KILL_FLAG).exists()
    ev = bot.tick()
    assert bot.trader.halted and any("kill switch" in e for e in ev)
    bot.resume()
    assert not (tmp_path / KILL_FLAG).exists() and not bot.trader.halted
    clear_kill_flag(tmp_path)                       # idempotent


# ---------------------------------------------------------------- TMN markets and per-currency money
def test_market_names_split_into_base_and_quote():
    from advisor.wallex import base_quote
    assert base_quote("BTCUSDT") == ("BTC", "USDT")
    assert base_quote("USDTTMN") == ("USDT", "TMN")
    assert base_quote("BTCTMN") == ("BTC", "TMN")
    with pytest.raises(ValueError):
        base_quote("BTCEUR")


def test_config_refuses_tmn_market_without_tmn_capital():
    with pytest.raises(ValueError, match="capital_tmn"):
        Config(symbols=["BTCTMN"], capital_tmn=0).validate()
    Config(symbols=["BTCTMN"], capital_tmn=1e8).validate()


def test_tmn_round_trip_books_profit_in_toman_only(tmp_path):
    cfg = Config(capital=1000.0, capital_tmn=1e8, max_loss_pct=5.0, min_profit_pct=5.0,
                 trading_mode="dry_run", symbols=["BTCTMN"], output_dir=str(tmp_path))
    cfg.validate()
    mkt = Market(bid=3000000000.0, ask=3000100000.0)
    ex = SimExecutor(parse_markets(MARKETS), mkt,
                     start_balances={"USDT": 1000.0, "TMN": 1e8}, fee_rate=cfg.fee_rate)
    tr = Trader(cfg, ex, tmp_path / "t.json", log=lambda *_: None)
    adv = Advice("BTCTMN", "t", 3000050000.0, "BUY", "test", prob_up=0.8, net_edge=0.05, amount_quote=2e7)
    tr.on_advice(adv, 3000000000.0, 3000100000.0, {"BTCTMN": 3000000000.0})
    tr.on_tick("BTCTMN", 3000000000.0, 3000100000.0)
    pos = tr.positions["BTCTMN"]
    assert pos.phase == "holding"
    assert ex.free_quote("BTCTMN") == pytest.approx(1e8 - 2e7, rel=2e-3)   # filled at the ask, a bit under budget
    assert ex.balance["USDT"] == 1000.0
    mkt.move(pos.take * 1.01, pos.take * 1.01 + 1000)
    ex.tick()
    tr.on_tick("BTCTMN", mkt.bid, mkt.ask)
    assert "BTCTMN" not in tr.positions
    assert tr.realised_total["TMN"] > 0 and tr.realised_total["USDT"] == 0
    assert tr.trades[-1]["quote"] == "TMN"


def test_tmn_buy_skipped_when_no_tmn_capital(setup):
    cfg, mkt, ex, tr, _ = setup()
    msg = tr.on_advice(Advice("BTCTMN", "t", 3e9, "BUY", "x", amount_quote=2e7), 3e9, 3.0001e9, {})
    assert msg and "no TMN capital" in msg[0] and "BTCTMN" not in tr.positions


def test_limits_are_checked_per_currency(tmp_path):
    cfg = Config(capital=1000.0, capital_tmn=1e8, max_daily_loss_frac=0.03, max_drawdown_kill=0.15,
                 symbols=["BTCUSDT", "BTCTMN"], output_dir=str(tmp_path))
    ex = SimExecutor({}, lambda s: (0, 0), start_balances={"USDT": 1000.0, "TMN": 1e8})
    tr = Trader(cfg, ex, tmp_path / "t.json", log=lambda *_: None)
    tr.realised_today = {"USDT": 0.0, "TMN": -2e6}            # 2% of TMN capital: below the 3% limit
    tr._check_limits()
    assert not tr.halted
    tr.realised_today = {"USDT": 0.0, "TMN": -4e6}            # 4%: halts
    tr._check_limits()
    assert tr.halted and tr.halt_kind == "daily" and "TMN" in tr.halt_reason


def test_exposure_is_counted_per_currency(tmp_path):
    cfg = Config(capital=1000.0, capital_tmn=1e8, symbols=["BTCUSDT", "BTCTMN"], output_dir=str(tmp_path))
    ex = SimExecutor({}, lambda s: (0, 0), start_balances={})
    tr = Trader(cfg, ex, tmp_path / "t.json", log=lambda *_: None)
    tr.positions["BTCUSDT"] = Position("BTCUSDT", "holding", "", "", 0, 0, "", 0, entry_price=100, qty=2, stop=0, take=0)
    tr.positions["BTCTMN"] = Position("BTCTMN", "holding", "", "", 0, 0, "", 0, entry_price=3e9, qty=0.01, stop=0, take=0)
    bids = {"BTCUSDT": 100.0, "BTCTMN": 3e9}
    assert tr.exposure(bids, "USDT") == pytest.approx(200.0)
    assert tr.exposure(bids, "TMN") == pytest.approx(3e7)
