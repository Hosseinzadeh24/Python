"""Public Binance market data (no API key). Falls back to a second Binance host if one is blocked."""
from __future__ import annotations

import time

import pandas as pd
import requests

COLS = ["open", "high", "low", "close", "volume", "taker_buy"]
PAGE = 1000


class MarketDataError(RuntimeError):
    pass


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


def klines_to_frame(rows: list) -> pd.DataFrame:
    """Binance kline row: [openTime, open, high, low, close, volume, closeTime, quoteVol, trades,
    takerBuyBaseVol, takerBuyQuoteVol, ignore]."""
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "open": df[1].astype(float), "high": df[2].astype(float), "low": df[3].astype(float),
        "close": df[4].astype(float), "volume": df[5].astype(float), "taker_buy": df[9].astype(float),
        "close_time": df[6].astype("int64"),
    })
    out.index = pd.to_datetime(df[0].astype("int64"), unit="ms", utc=True)
    return out


def fetch_candles(symbol: str, interval: str, bars: int, urls: list[str]) -> pd.DataFrame:
    """Latest `bars` candles, paginated backwards. Only CLOSED candles are returned."""
    rows: list = []
    end_time = None
    while len(rows) < bars:
        params = {"symbol": symbol, "interval": interval, "limit": PAGE}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _get(urls, "/api/v3/klines", params)
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1
        if len(batch) < PAGE:
            break
    if not rows:
        raise MarketDataError(f"no candles for {symbol}")
    df = klines_to_frame(rows)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]          # drop the candle that is still forming
    return df.drop(columns=["close_time"]).tail(bars)


def fetch_price(symbol: str, urls: list[str]) -> float:
    data = _get(urls, "/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


def demo_frame(symbol: str, bars: int, seed: int) -> pd.DataFrame:
    """Offline synthetic candles for testing the program without internet. NOT real prices."""
    from .synthetic import synthetic_ohlcv
    return synthetic_ohlcv(n_bars=bars, seed=seed, start_price=100.0 + seed * 10)
