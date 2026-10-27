"""Crypto advisor: analyses the market and suggests BUY / SELL actions and investment opportunities.

Usage (from this folder):
    python main.py              run once now
    python main.py --loop       run every 15 minutes until you press Ctrl+C
    python main.py --backtest   walk-forward test on real history (writes charts)
    python main.py --demo       run offline with synthetic data (to check the installation)

It never places orders. You decide and trade yourself.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from advisor.config import Config  # noqa: E402
from advisor.market import MarketDataError, demo_frame, fetch_candles, fetch_price  # noqa: E402
from advisor.notify import Notifier, format_alert  # noqa: E402
from advisor.strategy import advise, default_params, limits_from, load_holdings, load_params  # noqa: E402

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def run_once(cfg: Config, demo: bool = False) -> list:
    now = datetime.now().astimezone()
    limits = limits_from(cfg)
    holdings = load_holdings(ROOT / "portfolio.json")
    held = {h.symbol: h for h in holdings}
    params = load_params(cfg)
    symbols = list(dict.fromkeys(cfg.symbols + [h.symbol for h in holdings]))
    notifier = Notifier(cfg)

    frames, prices, errors = {}, {}, {}
    for i, s in enumerate(symbols):
        try:
            if demo:
                frames[s] = demo_frame(s, cfg.train_bars + 100, seed=cfg.seed + i)
                prices[s] = float(frames[s]["close"].iloc[-1])
            else:
                frames[s] = fetch_candles(s, cfg.timeframe, cfg.train_bars, cfg.binance_urls)
                prices[s] = fetch_price(s, cfg.binance_urls)
        except (MarketDataError, Exception) as exc:           # one coin failing must not stop the rest
            errors[s] = str(exc)[:200]

    held_value = sum(h.qty * prices[h.symbol] for h in holdings if h.symbol in prices)
    exposure_left = cfg.capital * cfg.max_total_exposure - held_value

    advices = []
    for s in symbols:
        if s not in frames:
            continue
        prm = params.get(s, default_params(cfg))
        adv = advise(s, frames[s], prices[s], prm, cfg, limits, held.get(s), exposure_left)
        if adv.action == "BUY":
            exposure_left -= adv.amount_quote                  # later coins get the remaining budget
        advices.append(adv)

    lines = [f"Crypto advisor run: {now:%Y-%m-%d %H:%M:%S %Z}   capital {cfg.capital:.0f} USDT   "
             f"held value {held_value:.2f} USDT", ""]
    pending_alerts = []          # (text, record, important) - printed after the summary table
    for adv in advices:
        for kind in adv.alerts:
            if notifier.should_alert(adv.symbol, kind, now):
                record = {"time": now.isoformat(timespec="seconds"), "kind": kind,
                          **{k: v for k, v in adv.__dict__.items() if k != "alerts"}}
                pending_alerts.append((format_alert(adv, kind), record,
                                       kind in ("BUY_SIGNAL", "SELL_SIGNAL", "OPPORTUNITY")))
                notifier.mark(adv.symbol, kind, now)
    notifier.save_state()

    for adv in advices:
        pos = f"{adv.buy_pressure:.0%}" if adv.buy_pressure is not None else "n/a"
        lines.append(f"{adv.symbol:<9} {adv.action:<5} price {adv.price:<12.6g} P(up) {adv.prob_up:5.0%}  "
                     f"edge {adv.net_edge:+6.2%}  RSI {adv.rsi:4.0f}  trend {adv.trend:<4} buy-pressure {pos}")
        if adv.action in ("BUY", "SELL"):
            lines.append(f"          -> {adv.action} {adv.qty:.6g} {adv.symbol[:-4]} (~{adv.amount_quote:.2f} USDT)"
                         + (f"  stop {adv.stop_price:.6g}  target {adv.take_price:.6g}"
                            if adv.action == "BUY" else f"  (stop {adv.stop_price:.6g}, target {adv.take_price:.6g})"))
        if adv.suggested_stop:
            lines.append(f"          -> consider moving stop to {adv.suggested_stop:.6g}")
        lines.append(f"          {adv.reason}")
    for s, e in errors.items():
        lines.append(f"{s:<9} ERROR: {e}")
    if not advices:
        lines.append("No data could be fetched. Check your internet connection.")
    text = "\n".join(lines)
    print(text, flush=True)
    for alert_text, record, important in pending_alerts:
        notifier.alert(alert_text, record, important)
    notifier.write_latest(text + "\n")
    notifier.write_history(advices, now)
    return advices


def next_quarter_hour(now: datetime) -> datetime:
    base = now.replace(minute=0, second=0, microsecond=0)
    nxt = base + timedelta(minutes=(now.minute // 15 + 1) * 15)
    return nxt + timedelta(seconds=20)                       # small delay: let the exchange finish the candle


def loop(cfg: Config, demo: bool) -> None:
    err_log = cfg.out / "errors.log"
    while True:
        try:
            run_once(cfg, demo)
        except Exception:                                    # keep running through unexpected errors
            msg = traceback.format_exc()
            print(msg, flush=True)
            err_log.parent.mkdir(parents=True, exist_ok=True)
            with err_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}\n{msg}\n")
        nxt = next_quarter_hour(datetime.now().astimezone())
        print(f"\nNext run at {nxt:%H:%M:%S}. Press Ctrl+C to stop.\n", flush=True)
        time.sleep(max(1.0, (nxt - datetime.now().astimezone()).total_seconds()))


def backtest(cfg: Config) -> None:
    from advisor.backtest import save_results, walk_forward

    weeks = cfg.calibration_periods
    bars = cfg.train_bars + weeks * cfg.test_bars + 100
    print(f"Downloading {bars} hourly candles per coin ...", flush=True)
    frames = {s: fetch_candles(s, cfg.timeframe, bars, cfg.binance_urls) for s in cfg.symbols}
    result = walk_forward(frames, cfg, periods=weeks)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = save_results(result, cfg.out / "backtest" / f"run_{stamp}")
    (cfg.out / "advisor_params.json").write_text(
        json.dumps(result["periods"][-1]["chosen_params"], indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Charts and CSVs: {out}")
    print("The latest chosen rules were saved to output/advisor_params.json and will be used by the advisor.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Crypto advisor (suggestions only)")
    ap.add_argument("--loop", action="store_true", help="run every 15 minutes")
    ap.add_argument("--backtest", action="store_true", help="walk-forward backtest with charts")
    ap.add_argument("--demo", action="store_true", help="offline synthetic data (testing only)")
    ap.add_argument("--config", default=None, help="path to config.json")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    try:
        if args.backtest:
            backtest(cfg)
        elif args.loop:
            loop(cfg, args.demo)
        else:
            run_once(cfg, args.demo)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
