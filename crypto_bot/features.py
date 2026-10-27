"""Technical / statistical features. Every feature at time t uses only data up to and including t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum, trend, mean-reversion, volatility, volume and calendar features."""
    c, v = df["close"], df["volume"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    # momentum / returns
    for k in (1, 3, 12, 24, 72):
        f[f"ret_{k}"] = np.log(c / c.shift(k))

    # volatility and regime
    f["vol_24"] = lr.rolling(24).std()
    f["vol_72"] = lr.rolling(72).std()
    f["vol_ratio"] = f["vol_24"] / f["vol_72"]
    f["atr_pct"] = atr(df, 14) / c
    f["hl_range"] = (df["high"] - df["low"]) / c

    # moving-average trend (moving averages)
    for n in (10, 30, 90):
        f[f"close_sma{n}"] = c / c.rolling(n).mean() - 1
    f["sma10_sma30"] = c.rolling(10).mean() / c.rolling(30).mean() - 1

    # MACD
    macd = ema(c, 12) - ema(c, 26)
    f["macd"] = macd / c
    f["macd_signal"] = ema(macd, 9) / c
    f["macd_hist"] = f["macd"] - f["macd_signal"]

    # oscillators / mean reversion
    f["rsi_14"] = rsi(c, 14) / 100
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_z"] = (c - m20) / (2 * s20)
    f["bb_width"] = 4 * s20 / m20
    f["zscore_48"] = (c - c.rolling(48).mean()) / c.rolling(48).std()

    # volume / order-flow proxies (volume-based indicators)
    lv = np.log1p(v)
    f["vol_z"] = (lv - lv.rolling(48).mean()) / lv.rolling(48).std()
    f["vol_chg_3"] = lv.diff(3)
    signed = np.sign(c.diff()).fillna(0) * v
    f["signed_vol_ratio_12"] = signed.rolling(12).sum() / v.rolling(12).sum()

    # calendar (crypto trades 24/7; hour-of-day effects)
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return f.replace([np.inf, -np.inf], np.nan)


def forward_labels(df: pd.DataFrame, horizon: int, cost: float) -> tuple[pd.Series, pd.Series]:
    """Future log return over `horizon` bars and label = 1 if it beats round-trip costs.

    NOTE: these use future prices, so they are only ever used as training targets.
    """
    fwd = np.log(df["close"].shift(-horizon) / df["close"])
    label = (fwd > cost).astype(float).where(fwd.notna())
    return fwd, label
