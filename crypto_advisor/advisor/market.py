"""Historical candles and prices. Default source: Wallex. Optional: Binance public data."""
from __future__ import annotations

import time

import pandas as pd
import requests

from .wallex import WallexClient

COLS = ["open", "high", "low", "close", "volume", "taker_buy"]
RESOLUTION = {"15m": 15, "1h": 60, "4h": 240}          # minutes per candle
CHUNK = 500                                            # candles per Wallex history request


class MarketDataError(RuntimeError):
    pass


def _seconds(timeframe: str) -> int:
    if timeframe not in RESOLUTION:
        raise MarketDataError(f"unsupported timeframe {timeframe}; use one of {sorted(RESOLUTION)}")
    return RESOLUTION[timeframe] * 60


def wallex_candles(client: WallexClient, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """Latest `bars` CLOSED candles from Wallex, fetched in chunks going back in time."""
    step = _seconds(timeframe)
    res = str(RESOLUTION[timeframe])
    now = int(time.time())
    end = now
    rows: dict[int, tuple] = {}
    for _ in range(max(1, bars // CHUNK + 2)):
        frm = end - CHUNK * step
        data = client.history(symbol, res, frm, end)
        if data.get("s") == "ok":
            for t, o, h, l, c, v in zip(data["t"], data["o"], data["h"], data["l"], data["c"], data["v"]):
                rows[int(t)] = (float(o), float(h), float(l), float(c), float(v))
        end = frm
        if len(rows) >= bars:
            break
    if not rows:
        raise MarketDataError(f"Wallex returned no candles for {symbol}")
    ts = sorted(rows)
    df = pd.DataFrame([rows[t] for t in ts], columns=["open", "high", "low", "close", "volume"],
                      index=pd.to_datetime(ts, unit="s", utc=True))
    df = df[df.index + pd.Timedelta(seconds=step) <= pd.Timestamp.now(tz="UTC")]   # drop forming candle
    return df.tail(bars)


def _get(urls: list[str], path: str, params: dict):
    last_err = None
    for base in urls:
        try:
            r = requests.get(base + path, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
    raise MarketDataError(f"all Binance hosts failed: {last_err}")


def binance_candles(symbol: str, timeframe: str, bars: int, urls: list[str]) -> pd.DataFrame:
    """Public Binance klines (used only when data_provider='binance')."""
    interval = {"15m": "15m", "1h": "1h", "4h": "4h"}[timeframe]
    rows, end_time = [], None
    while len(rows) < bars:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _get(urls, "/api/v3/klines", params)
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1
        if len(batch) < 1000:
            break
    if not rows:
        raise MarketDataError(f"no candles for {symbol}")
    df = pd.DataFrame({
        "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows], "taker_buy": [float(r[9]) for r in rows],
    }, index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
    now_ms = int(time.time() * 1000)
    closed = [r[6] < now_ms for r in rows]                 # close_time already passed = closed candle
    df = df[closed]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.tail(bars)


def fetch_candles(cfg, client: WallexClient | None, symbol: str, bars: int) -> pd.DataFrame:
    if cfg.data_provider == "binance":
        return binance_candles(symbol, cfg.timeframe, bars, cfg.binance_urls)
    if client is None:
        client = WallexClient()
    return wallex_candles(client, symbol, cfg.timeframe, bars)


def demo_frame(symbol: str, bars: int, seed: int) -> pd.DataFrame:
    """Offline synthetic candles for testing without internet. NOT real prices."""
    from .synthetic import synthetic_ohlcv
    return synthetic_ohlcv(n_bars=bars, seed=seed, start_price=100.0 + seed * 10)
