"""Command line entry point.

  python -m crypto_bot calibrate --source synthetic            # demo on generated data
  python -m crypto_bot calibrate --source csv                  # data/candles/BTC_USDT.csv ...
  python -m crypto_bot calibrate --source ccxt                 # real public market data
  python -m crypto_bot replay --source synthetic               # run the live code path offline
  python -m crypto_bot run                                     # 24/7 loop (paper unless mode=live)
  python -m crypto_bot serve-status                            # HTTP status for the phone app
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .alerts import Notifier
from .backtest import TradeParams
from .calibration import save_results, walk_forward
from .config import Config
from .data import CcxtFeed, load_csv, synthetic_ohlcv


def _load_frames(cfg: Config, source: str, bars: int) -> dict:
    if source == "synthetic":
        n = bars or cfg.train_bars + cfg.calibration_periods * cfg.test_bars + 200
        return {s: synthetic_ohlcv(n_bars=n, seed=cfg.seed + i, start_price=100.0 * (i + 1))
                for i, s in enumerate(cfg.symbols)}
    if source == "csv":
        base = Path(cfg.data_dir) / "candles"
        return {s: load_csv(base / f"{s.replace('/', '_')}.csv") for s in cfg.symbols}
    if source == "ccxt":
        feed = CcxtFeed(cfg.exchange)
        need = bars or cfg.train_bars + cfg.calibration_periods * cfg.test_bars + 200
        return {s: feed.fetch(s, cfg.timeframe, need).iloc[:-1] for s in cfg.symbols}
    raise ValueError(source)


def cmd_calibrate(args, cfg: Config) -> None:
    frames = _load_frames(cfg, args.source, args.bars)
    result = walk_forward(frames, cfg, periods=args.periods)
    run_dir = Path(cfg.report_dir) / f"run_{result['created_utc'].replace(':', '').replace('-', '')}"
    out = save_results(result, run_dir)
    # the live bot uses the parameters chosen in the most recent (last) calibration week
    last = result["periods"][-1]["chosen_params"]
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.data_dir) / "params.json").write_text(json.dumps(last, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"reports written to {out}")


def _load_params(cfg: Config) -> dict[str, TradeParams]:
    p = Path(cfg.data_dir) / "params.json"
    if not p.exists():
        return {}
    return {s: TradeParams(**v) for s, v in json.loads(p.read_text(encoding="utf-8")).items()}


def _make_broker(cfg: Config):
    from .broker import LiveBroker, PaperBroker
    from .risk import RiskLimits

    limits = RiskLimits(cfg.fee_rate, cfg.slippage, cfg.kelly_fraction,
                        cfg.max_position_frac, cfg.max_daily_loss_frac, cfg.max_drawdown_kill)
    state = Path(cfg.data_dir) / "broker_state.json"
    if cfg.mode == "live":
        return LiveBroker(cfg.exchange, limits, state, cfg.capital)
    return PaperBroker(cfg.capital, limits, state)


def cmd_replay(args, cfg: Config) -> None:
    from .bot import TradingBot, replay

    frames = _load_frames(cfg, args.source, args.bars)
    cfg.data_dir = str(Path(cfg.data_dir) / "replay")
    cfg.mode = "paper"
    bot = TradingBot(cfg, _make_broker(cfg), Notifier(cfg.data_dir), _load_params(cfg))
    start = cfg.train_bars
    statuses = replay(cfg, bot, frames, start)
    last = statuses[-1]
    print(json.dumps({k: last[k] for k in ("last_bar", "equity", "total_return", "halted")}, indent=2))
    print(f"trades: {len(bot.broker.trades)}  alerts log: {bot.notifier.log_path}")


def cmd_run(args, cfg: Config) -> None:
    from .bot import TradingBot, run_forever

    if cfg.mode == "live" and os.environ.get("BOT_LIVE_CONFIRM") != "I_ACCEPT_RISK":
        raise SystemExit("mode=live requires BOT_LIVE_CONFIRM=I_ACCEPT_RISK")
    bot = TradingBot(cfg, _make_broker(cfg), Notifier(cfg.data_dir), _load_params(cfg))
    run_forever(cfg, bot, CcxtFeed(cfg.exchange))


def cmd_serve_status(args, cfg: Config) -> None:
    """Serve data/status.json over HTTP. Set BOT_STATUS_TOKEN; clients send header X-Token."""
    import http.server

    token = os.environ.get("BOT_STATUS_TOKEN")
    if not token:
        raise SystemExit("set BOT_STATUS_TOKEN first")
    path = Path(cfg.data_dir) / "status.json"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.headers.get("X-Token") != token:
                self.send_response(401)
                self.end_headers()
                return
            body = path.read_bytes() if path.exists() else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # keep logs quiet
            pass

    http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="crypto_bot")
    ap.add_argument("--config", default="config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="walk-forward calibration with weekly reports")
    c.add_argument("--source", choices=["synthetic", "csv", "ccxt"], default="synthetic")
    c.add_argument("--periods", type=int, default=None)
    c.add_argument("--bars", type=int, default=0)
    r = sub.add_parser("replay", help="run the live decision code over historical bars")
    r.add_argument("--source", choices=["synthetic", "csv", "ccxt"], default="synthetic")
    r.add_argument("--bars", type=int, default=0)
    sub.add_parser("run", help="24/7 trading loop")
    s = sub.add_parser("serve-status", help="HTTP status endpoint for the Android app")
    s.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    {"calibrate": cmd_calibrate, "replay": cmd_replay,
     "run": cmd_run, "serve-status": cmd_serve_status}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
