"""Runs the real HTTP client against a local mock server that follows the documented Wallex shapes.
This checks headers, JSON envelopes, error handling and order payloads. It does NOT prove the live API
behaves the same way; that needs a real-key test (see README)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.wallex import WallexClient, WallexError  # noqa: E402

SEEN = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        SEEN.append(("GET", self.path, self.headers.get("X-API-Key")))
        if self.path.startswith("/v1/markets"):
            return self._send(200, {"success": True, "message": "", "result": {"symbols": {
                "BTCUSDT": {"baseAsset": "BTC", "quoteAsset": "USDT", "tickSize": 2, "stepSize": 6,
                            "minQty": 1e-6, "minNotional": 10,
                            "stats": {"bidPrice": "99", "askPrice": "100", "lastPrice": "99.5", "24h_ch": -0.4}}}}})
        if self.path.startswith("/v1/udf/history"):
            return self._send(200, {"s": "ok", "t": [1, 2], "o": ["1", "1"], "h": ["1", "1"],
                                    "l": ["1", "1"], "c": ["1", "1"], "v": ["1", "1"]})
        if self.path.startswith("/v1/account/balances"):
            if self.headers.get("X-API-Key") != "GOODKEY":
                return self._send(403, {"success": False, "message": "forbidden", "result": None})
            return self._send(200, {"success": True, "message": "", "result": {"balances": {
                "USDT": {"asset": "USDT", "value": "300", "locked": "50"}}}})
        return self._send(404, {"success": False, "message": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        SEEN.append(("POST", self.path, body))
        if self.path == "/v1/account/orders":
            return self._send(201, {"success": True, "message": "", "result": {
                "clientOrderId": body.get("client_id"), "symbol": body.get("symbol"),
                "side": body.get("side"), "price": body.get("price"), "origQty": body.get("quantity"),
                "executedQty": "0", "status": "NEW", "active": True}})
        return self._send(404, {"success": False, "message": "not found"})


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_public_markets_need_no_key(server):
    c = WallexClient(None, base_url=server, min_interval=0)
    m = c.markets()
    assert m["BTCUSDT"]["baseAsset"] == "BTC"
    assert SEEN[-1][2] is None


def test_history_returns_udf_arrays(server):
    c = WallexClient(None, base_url=server, min_interval=0)
    d = c.history("BTCUSDT", "60", 0, 10)
    assert d["s"] == "ok" and len(d["c"]) == 2


def test_account_call_sends_api_key_header(server):
    c = WallexClient("GOODKEY", base_url=server, min_interval=0)
    bal = c.balances()
    assert bal["USDT"]["value"] == "300"
    assert SEEN[-1][2] == "GOODKEY"


def test_account_call_without_key_is_refused_locally(server):
    c = WallexClient(None, base_url=server, min_interval=0)
    with pytest.raises(WallexError, match="No API key"):
        c.balances()


def test_rejected_key_raises_clear_error_without_retry(server):
    c = WallexClient("BADKEY", base_url=server, min_interval=0)
    with pytest.raises(WallexError, match="403"):
        c.balances()


def test_limit_order_payload_matches_docs(server):
    c = WallexClient("GOODKEY", base_url=server, min_interval=0)
    res = c.place_limit("BTCUSDT", "buy", "100.10", "1.998002", "ADVBabc123")
    assert res["status"] == "NEW"
    method, path, body = SEEN[-1]
    assert (method, path) == ("POST", "/v1/account/orders")
    assert body == {"symbol": "BTCUSDT", "type": "LIMIT", "side": "BUY", "price": "100.10",
                    "quantity": "1.998002", "client_id": "ADVBabc123"}
