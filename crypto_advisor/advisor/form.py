"""Turns the text typed in the window into a validated Config. No GUI code here, so it can be tested."""
from __future__ import annotations

import dataclasses

from .config import MAX_LOSS_CAP_PCT, TRADING_MODES, Config

LIVE_PHRASE = "I ACCEPT RISK"


def _num(text: str, label: str, errors: list[str]) -> float | None:
    t = (text or "").strip().replace(",", ".").replace("%", "")
    try:
        return float(t)
    except ValueError:
        errors.append(f"{label}: enter a number")
        return None


def parse_form(base: Config, values: dict[str, str], has_key: bool) -> tuple[Config | None, list[str]]:
    """values keys: max_loss, min_profit, capital, exposure, symbols, mode.
    Percent fields are typed as plain numbers (5 means 5%). Returns (config or None, error messages)."""
    errors: list[str] = []
    cfg = dataclasses.replace(base, symbols=list(base.symbols),
                              target_weekly_returns=list(base.target_weekly_returns))

    loss = _num(values.get("max_loss", ""), "Max loss %", errors)
    if loss is not None:
        if not 0 < loss <= MAX_LOSS_CAP_PCT:
            errors.append(f"Max loss % must be greater than 0 and at most {MAX_LOSS_CAP_PCT:g}")
        cfg.max_loss_pct = loss
    profit = _num(values.get("min_profit", ""), "Min profit %", errors)
    if profit is not None:
        cfg.min_profit_pct = profit
    capital = _num(values.get("capital", ""), "Capital (USDT)", errors)
    if capital is not None:
        if capital <= 0:
            errors.append("Capital must be greater than 0")
        cfg.capital = capital
    exposure = _num(values.get("exposure", ""), "Max invested %", errors)
    if exposure is not None:
        if not 0 < exposure <= 100:
            errors.append("Max invested % must be between 1 and 100")
        cfg.max_total_exposure = exposure / 100.0

    syms = [s.strip().upper() for s in (values.get("symbols", "") or "").replace(";", ",").split(",") if s.strip()]
    if not syms:
        errors.append("Enter at least one market, for example BTCUSDT")
    cfg.symbols = syms

    mode = (values.get("mode") or "suggest").strip()
    if mode not in TRADING_MODES:
        errors.append(f"Mode must be one of {', '.join(TRADING_MODES)}")
    cfg.trading_mode = mode
    if mode == "live" and not has_key:
        errors.append("Live trading needs your Wallex API key (save it first)")

    if errors:
        return None, errors
    try:
        cfg.validate()
    except ValueError as exc:
        return None, [str(exc)]
    return cfg, []


def live_confirmed(typed: str) -> bool:
    return " ".join((typed or "").upper().split()) == LIVE_PHRASE
