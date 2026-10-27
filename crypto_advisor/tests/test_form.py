import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.config import Config                                   # noqa: E402
from advisor.form import live_confirmed, parse_form                 # noqa: E402

GOOD = {"max_loss": "5", "min_profit": "6", "capital": "500", "exposure": "50",
        "symbols": "btcusdt, ETHUSDT", "mode": "dry_run"}


def test_good_form_builds_config():
    cfg, errs = parse_form(Config(), GOOD, has_key=False)
    assert errs == [] and cfg.max_loss_pct == 5 and cfg.min_profit_pct == 6
    assert cfg.symbols == ["BTCUSDT", "ETHUSDT"] and cfg.max_total_exposure == 0.5


def test_max_loss_above_15_is_refused():
    cfg, errs = parse_form(Config(), {**GOOD, "max_loss": "15.5"}, has_key=False)
    assert cfg is None and any("at most 15" in e for e in errs)


def test_max_loss_zero_or_text_refused():
    assert parse_form(Config(), {**GOOD, "max_loss": "0"}, False)[0] is None
    cfg, errs = parse_form(Config(), {**GOOD, "max_loss": "abc"}, False)
    assert cfg is None and any("enter a number" in e for e in errs)


def test_min_profit_must_cover_costs():
    cfg, errs = parse_form(Config(), {**GOOD, "min_profit": "0.3"}, False)
    assert cfg is None and errs


def test_percent_with_sign_and_comma_accepted():
    cfg, errs = parse_form(Config(), {**GOOD, "max_loss": "7,5%"}, False)
    assert errs == [] and cfg.max_loss_pct == 7.5


def test_live_needs_key():
    cfg, errs = parse_form(Config(), {**GOOD, "mode": "live"}, has_key=False)
    assert cfg is None and any("API key" in e for e in errs)
    cfg, errs = parse_form(Config(), {**GOOD, "mode": "live"}, has_key=True)
    assert errs == [] and cfg.trading_mode == "live"


def test_unknown_mode_refused():
    assert parse_form(Config(), {**GOOD, "mode": "turbo"}, False)[0] is None


def test_typed_confirmation():
    assert live_confirmed("I ACCEPT RISK")
    assert live_confirmed("  i accept   risk ")
    assert not live_confirmed("yes")
    assert not live_confirmed("")


def test_tmn_market_needs_tmn_capital_in_form():
    cfg, errs = parse_form(Config(), {**GOOD, "symbols": "BTCTMN", "capital_tmn": "0"}, has_key=False)
    assert cfg is None and errs
    cfg, errs = parse_form(Config(), {**GOOD, "symbols": "BTCTMN, BTCUSDT", "capital_tmn": "100000000"}, False)
    assert errs == [] and cfg.capital_tmn == 1e8 and cfg.symbols == ["BTCTMN", "BTCUSDT"]


def test_unsupported_market_refused():
    cfg, errs = parse_form(Config(), {**GOOD, "symbols": "BTCEUR"}, has_key=False)
    assert cfg is None and errs
