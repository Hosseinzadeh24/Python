"""Configuration with safe defaults (paper trading, strict risk limits)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # --- market -----------------------------------------------------------
    exchange: str = "binance"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe: str = "1h"               # one bar = 1 hour; 24/7 market
    # --- capital ------------------------------------------------------------
    capital: float = 1000.0             # quote currency (e.g. USDT) given to the bot
    mode: str = "paper"                 # "paper" (simulated) or "live" (real orders)
    # --- forecasting --------------------------------------------------------
    horizon_bars: int = 4               # predict direction over the next 4 bars
    train_bars: int = 24 * 60           # rolling training window: 60 days
    test_bars: int = 24 * 7             # evaluation period: 7 days
    retrain_every_bars: int = 24        # live: retrain once per day
    # --- costs --------------------------------------------------------------
    fee_rate: float = 0.001             # 0.1% per side (typical spot taker fee)
    slippage: float = 0.0005            # 0.05% per side
    # --- risk ---------------------------------------------------------------
    kelly_fraction: float = 0.25        # fractional Kelly (quarter Kelly)
    max_position_frac: float = 0.25     # max share of allocated capital per coin
    max_daily_loss_frac: float = 0.03   # no new entries after -3% in a day
    max_drawdown_kill: float = 0.15     # halt trading permanently at -15%
    # --- alerts -------------------------------------------------------------
    alert_min_prob: float = 0.70        # model confidence needed for an alert
    alert_min_net_edge: float = 0.03    # expected net gain (after costs) for an alert
    # --- calibration --------------------------------------------------------
    calibration_periods: int = 4
    target_weekly_returns: list[float] = field(default_factory=lambda: [0.10, 0.15])
    seed: int = 42
    # --- paths / runtime ----------------------------------------------------
    data_dir: str = "data"
    report_dir: str = "reports"
    poll_seconds: int = 60

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path is None or not Path(path).exists():
            return cfg
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for k, v in raw.items():
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError("mode must be 'paper' or 'live'")
        if not 0 < self.max_position_frac <= 1:
            raise ValueError("max_position_frac must be in (0, 1]")
        if not 0 < self.max_drawdown_kill < 1:
            raise ValueError("max_drawdown_kill must be in (0, 1)")
        if self.capital <= 0:
            raise ValueError("capital must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
