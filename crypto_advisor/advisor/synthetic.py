"""Random-walk candles for offline testing only. They have no real predictive structure."""
import numpy as np
import pandas as pd


def synthetic_ohlcv(n_bars: int = 1000, seed: int = 0, start_price: float = 100.0,
                    vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.where(np.sin(np.arange(n_bars) / 400.0) > 0, 1.5, 0.7)
    close = start_price * np.exp(np.cumsum(rng.normal(0, vol * regime)))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, vol * regime * 0.5, n_bars))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(5.0, 0.5, n_bars) * regime
    taker = volume * rng.uniform(0.4, 0.6, n_bars)
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume, "taker_buy": taker}, index=idx)
