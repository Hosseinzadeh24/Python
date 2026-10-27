"""Settings. Everything has a default; override any of them in config.json."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent   # the crypto_advisor folder


@dataclass
class Config:
    # --- markets (Binance spot pairs, no API key needed for prices) -------------
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    timeframe: str = "1h"
    binance_urls: list[str] = field(default_factory=lambda: [
        "https://api.binance.com",
        "https://data-api.binance.vision",          # public market-data mirror, used as fallback
    ])
    # --- your money (the advisor only sizes its SUGGESTIONS against this) -------
    capital: float = 1000.0                          # in USDT
    max_total_exposure: float = 0.60                 # never suggest more than 60% of capital in total
    min_order_quote: float = 10.0                    # ignore suggestions smaller than 10 USDT
    # --- forecasting -------------------------------------------------------------
    horizon_bars: int = 4                            # direction over the next 4 hours
    train_bars: int = 1000                           # ~41 days of hourly candles
    test_bars: int = 168                             # backtest evaluation week (7 days)
    seed: int = 42
    # --- costs and risk ----------------------------------------------------------
    fee_rate: float = 0.001                          # 0.1% per side
    slippage: float = 0.0005                         # 0.05% per side
    kelly_fraction: float = 0.25
    max_position_frac: float = 0.25                  # max 25% of capital in one coin
    max_daily_loss_frac: float = 0.03
    max_drawdown_kill: float = 0.15
    # --- default trade rules (overridden by output/advisor_params.json after backtest) --
    p_enter: float = 0.60
    p_exit: float = 0.50
    stop_mult: float = 2.0                           # stop-loss = 2 x ATR
    take_mult: float = 3.0                           # take-profit = 3 x ATR
    max_hold_bars: int = 24
    # --- alerts ------------------------------------------------------------------
    alert_min_prob: float = 0.70                     # P(up) needed for an OPPORTUNITY alert
    alert_min_net_edge: float = 0.03                 # expected gain after costs needed for an alert
    alert_repeat_hours: float = 4.0                  # same alert is not repeated within this time
    volume_spike_z: float = 2.5                      # unusual volume alert threshold (z-score)
    sharp_move_pct: float = 0.02                     # 2% move in one hour triggers an alert
    beep: bool = True                                # Windows beep on new opportunities
    # --- backtest ----------------------------------------------------------------
    calibration_periods: int = 4                     # weeks of walk-forward testing
    target_weekly_returns: list[float] = field(default_factory=lambda: [0.10, 0.15])
    # --- output ------------------------------------------------------------------
    output_dir: str = "output"

    @property
    def out(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else PROJECT_DIR / p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        p = Path(path) if path else PROJECT_DIR / "config.json"
        if not p.exists():
            return cfg
        raw = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown keys in config.json: {sorted(unknown)}")
        for k, v in raw.items():
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 0 < self.max_total_exposure <= 1:
            raise ValueError("max_total_exposure must be in (0, 1]")
        if not 0 < self.max_position_frac <= 1:
            raise ValueError("max_position_frac must be in (0, 1]")
        if self.capital <= 0:
            raise ValueError("capital must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
