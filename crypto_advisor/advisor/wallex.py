"""Wallex exchange REST client (official API: https://api-docs.wallex.ir).

Public endpoints need no key. Account endpoints need your API key in the X-API-Key header.
The client never calls any withdrawal endpoint.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal

import requests

BASE_URL = "https://api.wallex.ir"
FILLED_STATES = {"FILLED", "EXECUTED", "DONE"}
TERMINAL_STATES = FILLED_STATES | {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}


class WallexError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers
def to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def round_down(x: float, dp: int) -> float:
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_DOWN))


def round_up(x: float, dp: int) -> float:
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_UP))


def fmt_fixed(x: float, dp: int) -> str:
    """Exact string with dp decimals (the API takes price/quantity as strings)."""
    return f"{Decimal(str(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_DOWN):f}"


def _dp(v, default: int) -> int:
    """stepSize/tickSize are given as decimal places (e.g. 6 or 2). Accept 0.01-style values too."""
    if v is None:
        return default
    f = to_float(v, -1)
    if f < 0:
        return default
    if f == int(f):
        return int(f)
    return max(0, round(-math.log10(f)))


@dataclass
class MarketMeta:
    symbol: str
    base: str
    quote: str
    price_dp: int
    qty_dp: int
    min_qty: float
    min_notional: float


@dataclass
class OrderInfo:
    client_id: str
    symbol: str
    side: str
    price: float
    qty: float
    executed_qty: float
    avg_price: float
    status: str
    active: bool

    @property
    def filled(self) -> bool:
        if self.status.upper() in FILLED_STATES:
            return True
        return self.qty > 0 and self.executed_qty >= self.qty * 0.9999

    @property
    def terminal(self) -> bool:
        return self.filled or self.status.upper() in TERMINAL_STATES or not self.active


def parse_order(d: dict, client_id: str = "") -> OrderInfo:
    return OrderInfo(
        client_id=str(d.get("clientOrderId") or client_id),
        symbol=str(d.get("symbol", "")),
        side=str(d.get("side", "")).upper(),
        price=to_float(d.get("price")),
        qty=to_float(d.get("origQty")),
        executed_qty=to_float(d.get("executedQty")),
        avg_price=to_float(d.get("executedPrice")),
        status=str(d.get("status", "")),
        active=bool(d.get("active", False)),
    )


def parse_markets(symbols: dict) -> dict[str, MarketMeta]:
    out = {}
    for sym, m in symbols.items():
        out[sym] = MarketMeta(
            symbol=sym, base=str(m.get("baseAsset", "")), quote=str(m.get("quoteAsset", "")),
            price_dp=_dp(m.get("tickSize"), 2), qty_dp=_dp(m.get("stepSize"), 6),
            min_qty=to_float(m.get("minQty")), min_notional=to_float(m.get("minNotional")),
        )
    return out


def parse_snapshot(symbols: dict) -> dict[str, dict]:
    """Realtime-style snapshot of every market from /v1/markets (prices and 24h stats)."""
    out = {}
    for sym, m in symbols.items():
        st = m.get("stats") or {}
        last = to_float(st.get("lastPrice"))
        bid = to_float(st.get("bidPrice")) or last
        ask = to_float(st.get("askPrice")) or last
        out[sym] = {
            "last": last, "bid": bid, "ask": ask,
            "change_24h": to_float(st.get("24h_ch")),
            "volume_quote_24h": to_float(st.get("24h_quoteVolume")),
            "high_24h": to_float(st.get("24h_highPrice")),
            "low_24h": to_float(st.get("24h_lowPrice")),
        }
    return out


# ---------------------------------------------------------------- client
class WallexClient:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL, timeout: float = 15.0,
                 min_interval: float = 0.25, session: requests.Session | None = None):
        self.api_key = api_key or None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self._last = 0.0

    def _request(self, method: str, path: str, params: dict | None = None, body: dict | None = None,
                 auth: bool = False):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if auth:
            if not self.api_key:
                raise WallexError("No API key set. Enter your Wallex API key first.")
            headers["X-API-Key"] = self.api_key
        last_exc = None
        for attempt in range(4):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.request(method, self.base_url + path, params=params, json=body,
                                         headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429,) or r.status_code >= 500:
                last_exc = WallexError(f"HTTP {r.status_code}")
                time.sleep(2 ** attempt)
                continue
            try:
                data = r.json()
            except ValueError:
                raise WallexError(f"HTTP {r.status_code}: non-JSON answer")
            if r.status_code >= 400 or (isinstance(data, dict) and data.get("success") is False):
                msg = data.get("message") if isinstance(data, dict) else None
                raise WallexError(f"HTTP {r.status_code}: {msg or 'request rejected'}")
            return data
        raise WallexError(f"Wallex did not answer after retries: {last_exc}")

    # ---- public market data
    def markets(self) -> dict:
        """Full symbol dictionary: precision, minimums and live stats."""
        return self._request("GET", "/v1/markets")["result"]["symbols"]

    def history(self, symbol: str, resolution: str, frm: int, to: int) -> dict:
        """Candles (UDF format): dict with lists t, o, h, l, c, v; s == 'ok' or 'no_data'."""
        return self._request("GET", "/v1/udf/history",
                             params={"symbol": symbol, "resolution": resolution, "from": frm, "to": to})

    # ---- account (needs API key; read-only calls + order placement/cancel)
    def balances(self) -> dict:
        return self._request("GET", "/v1/account/balances", auth=True)["result"]["balances"]

    def fee_table(self) -> dict:
        return self._request("GET", "/v1/account/fee", auth=True)["result"]

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else None
        res = self._request("GET", "/v1/account/openOrders", params=params, auth=True)["result"]
        return res.get("orders", []) if isinstance(res, dict) else res

    def place_limit(self, symbol: str, side: str, price: str, quantity: str, client_id: str) -> dict:
        body = {"symbol": symbol, "type": "LIMIT", "side": side.upper(),
                "price": price, "quantity": quantity, "client_id": client_id}
        return self._request("POST", "/v1/account/orders", body=body, auth=True)["result"]

    def order(self, client_id: str) -> dict:
        return self._request("GET", f"/v1/account/orders/{client_id}", auth=True)["result"]

    def cancel(self, client_id: str) -> dict:
        return self._request("DELETE", "/v1/account/orders", params={"clientOrderId": client_id}, auth=True)
