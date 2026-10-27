"""Market data: CSV loader, optional ccxt exchange fetcher, and a synthetic generator for tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV candles from CSV with a 'timestamp' column (ms epoch or ISO string)."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column")
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        idx = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)
    df = df.set_index(idx).drop(columns=["timestamp"])
    missing = set(OHLCV_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df = df[OHLCV_COLS].astype(float).sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_ccxt(exchange: str, symbol: str, timeframe: str = "1h", bars: int = 1500) -> pd.DataFrame:
    """Fetch the most recent `bars` public OHLCV candles through ccxt (no API key needed).

    Pages backwards-to-forwards because exchanges cap each request (often at 1000 candles).
    The last returned candle may still be forming; callers should drop it.
    """
    import ccxt  # imported lazily so tests do not need network libraries

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - bars * tf_ms
    rows: list = []
    while len(rows) < bars:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLS]).drop_duplicates("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop(columns=["timestamp"])[OHLCV_COLS].astype(float).sort_index().tail(bars)


def synthetic_ohlcv(n_bars: int = 24 * 365, seed: int = 0, start: str = "2024-01-01",
                    start_price: float = 100.0, drift: float = 0.0, vol: float = 0.01) -> pd.DataFrame:
    """Random-walk candles with volatility regimes. For testing ONLY: it has no real predictive structure."""
    rng = np.random.default_rng(seed)
    regime = np.where(np.sin(np.arange(n_bars) / 400.0) > 0, 1.5, 0.7)  # calm / volatile regimes
    rets = rng.normal(drift, vol * regime)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, vol * regime * 0.5, n_bars))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=5.0, sigma=0.5, size=n_bars) * regime
    idx = pd.date_range(start, periods=n_bars, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume}, index=idx)


class CcxtFeed:
    """Live candle source used by the 24/7 loop."""

    def __init__(self, exchange: str):
        self.exchange = exchange

    def fetch(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        return fetch_ccxt(self.exchange, symbol, timeframe, bars)
