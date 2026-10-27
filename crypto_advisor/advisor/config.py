"""Settings. Every value has a default; override any of them in config.json (or in the GUI)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent   # the crypto_advisor folder
MAX_LOSS_CAP_PCT = 15.0                                # hard limit for the stop-loss, cannot be raised
TRADING_MODES = ("suggest", "dry_run", "live")


@dataclass
class Config:
    # --- exchange & data -----------------------------------------------------------
    exchange_name: str = "Wallex"
    data_provider: str = "wallex"                     # "wallex" (default) or "binance" (public data only)
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframe: str = "1h"
    binance_urls: list[str] = field(default_factory=lambda: [
        "https://api.binance.com", "https://data-api.binance.vision"])
    # --- mode ------------------------------------------------------------------------
    # suggest : only analyses and shows advice (no orders, no API key needed for prices)
    # dry_run : full automatic logic with SIMULATED orders on real Wallex prices (no money)
    # live    : REAL orders on Wallex with your API key (requires typed confirmation in GUI)
    trading_mode: str = "suggest"
    # --- your risk settings (in percent, as you type them) ----------------------------
    max_loss_pct: float = 5.0                         # stop-loss: sell if price falls this much below entry
    min_profit_pct: float = 5.0                       # take-profit: sell when price rises this much above entry
    max_hold_hours: float = 72.0                      # close a position after this many hours
    # --- money -----------------------------------------------------------------------
    capital: float = 1000.0                           # USDT the bot may use
    max_total_exposure: float = 0.60                  # at most 60% of capital invested at once
    min_order_quote: float = 10.0
    max_position_frac: float = 0.25                   # at most 25% of capital in one coin
    max_daily_loss_frac: float = 0.03                 # halt new buys after -3% realised loss in a day
    max_drawdown_kill: float = 0.15                   # halt all trading after -15% total realised loss
    # --- costs -----------------------------------------------------------------------
    fee_rate: float = 0.002                           # Wallex spot fee (0.2% per side in its public docs)
    slippage: float = 0.0005
    kelly_fraction: float = 0.25
    entry_buffer: float = 0.001                       # buy limit placed 0.1% above ask, to fill fast
    exit_buffer: float = 0.003                        # sell limit placed 0.3% below bid, to fill fast
    order_timeout_s: float = 45.0
    # --- forecasting -----------------------------------------------------------------
    horizon_bars: int = 4
    train_bars: int = 1000
    test_bars: int = 168
    seed: int = 42
    # --- entry rules (overridden by output/advisor_params.json after a backtest) -----
    p_enter: float = 0.60
    p_exit: float = 0.50
    max_hold_bars: int = 24
    # --- timing ----------------------------------------------------------------------
    cycle_minutes: int = 15                           # full analysis every 15 minutes
    tick_seconds: int = 10                            # real-time price check / stop-loss monitor
    # --- alerts ----------------------------------------------------------------------
    alert_min_prob: float = 0.70
    alert_min_net_edge: float = 0.03
    alert_repeat_hours: float = 4.0
    volume_spike_z: float = 2.5
    sharp_move_pct: float = 0.02
    beep: bool = True
    # --- backtest --------------------------------------------------------------------
    calibration_periods: int = 4
    target_weekly_returns: list[float] = field(default_factory=lambda: [0.10, 0.15])
    # --- output ----------------------------------------------------------------------
    output_dir: str = "output"

    @property
    def out(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else PROJECT_DIR / p

    @property
    def stop_frac(self) -> float:
        return self.max_loss_pct / 100.0

    @property
    def take_frac(self) -> float:
        return self.min_profit_pct / 100.0

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        p = Path(path) if path else PROJECT_DIR / "config.json"
        if not p.exists():
            return cfg
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}   # "_help" style comments
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown keys in config.json: {sorted(unknown)}")
        for k, v in raw.items():
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else PROJECT_DIR / "config.json"
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def validate(self) -> None:
        if self.trading_mode not in TRADING_MODES:
            raise ValueError(f"trading_mode must be one of {TRADING_MODES}")
        if not 0 < self.max_loss_pct <= MAX_LOSS_CAP_PCT:
            raise ValueError(f"max_loss_pct must be > 0 and <= {MAX_LOSS_CAP_PCT} (hard limit)")
        round_trip_pct = 2 * (self.fee_rate + self.slippage) * 100
        if self.min_profit_pct <= round_trip_pct:
            raise ValueError(f"min_profit_pct must be above the round-trip cost ({round_trip_pct:.2f}%)")
        if not 0 < self.max_total_exposure <= 1:
            raise ValueError("max_total_exposure must be in (0, 1]")
        if not 0 < self.max_position_frac <= 1:
            raise ValueError("max_position_frac must be in (0, 1]")
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if self.cycle_minutes < 1 or self.tick_seconds < 2:
            raise ValueError("cycle_minutes >= 1 and tick_seconds >= 2")
        if self.data_provider not in ("wallex", "binance"):
            raise ValueError("data_provider must be 'wallex' or 'binance'")

    def to_dict(self) -> dict:
        return asdict(self)
