"""Where advice goes: console, output files, Windows beep, optional Telegram. De-duplicates alerts."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

from .config import Config
from .strategy import Advice
from .wallex import base_quote

KIND_TEXT = {
    "BUY_SIGNAL": "BUY SUGGESTION",
    "SELL_SIGNAL": "SELL SUGGESTION",
    "OPPORTUNITY": "INVESTMENT OPPORTUNITY",
    "UNUSUAL_VOLUME": "UNUSUAL VOLUME",
    "SHARP_MOVE": "SHARP PRICE MOVE",
}


def format_alert(adv: Advice, kind: str) -> str:
    head = f"*** {KIND_TEXT.get(kind, kind)}: {adv.symbol} @ {adv.price:.6g} ***"
    lines = [head, f"   P(up) {adv.prob_up:.0%} | expected net gain {adv.net_edge:+.2%} | "
                   f"1h move {adv.ret_1h:+.2%} | volume z {adv.vol_z:+.1f}"]
    base, quote = base_quote(adv.symbol)
    if kind in ("BUY_SIGNAL", "OPPORTUNITY") and adv.action == "BUY":
        lines.append(f"   Suggested amount: {adv.amount_quote:,.2f} {quote} (~{adv.qty:.6g} {base})"
                     f" | stop {adv.stop_price:.6g} | target {adv.take_price:.6g}")
    if kind == "SELL_SIGNAL":
        lines.append(f"   Suggest selling {adv.qty:.6g} {base} (~{adv.amount_quote:,.2f} {quote}). {adv.reason}")
    lines.append(f"   {adv.reason}")
    return "\n".join(lines)


def beep() -> None:
    try:
        import winsound  # Windows only
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except Exception:
        pass


class Notifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.out
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {"alerts": {}}
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    def should_alert(self, symbol: str, kind: str, now: datetime) -> bool:
        last = self.state["alerts"].get(f"{symbol}:{kind}")
        if last is None:
            return True
        return now - datetime.fromisoformat(last) >= timedelta(hours=self.cfg.alert_repeat_hours)

    def mark(self, symbol: str, kind: str, now: datetime) -> None:
        self.state["alerts"][f"{symbol}:{kind}"] = now.isoformat(timespec="seconds")

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def alert(self, text: str, record: dict, important: bool) -> None:
        print(text, flush=True)
        with (self.dir / "alerts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.token and self.chat_id:
            try:
                requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              json={"chat_id": self.chat_id, "text": text}, timeout=10)
            except requests.RequestException as exc:
                print(f"  (telegram not sent: {exc})", flush=True)
        if important and self.cfg.beep:
            beep()

    def write_history(self, advices: list[Advice], now: datetime) -> None:
        path = self.dir / "suggestions_history.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time_utc", "symbol", "action", "price", "prob_up", "net_edge",
                            "amount_quote", "qty", "stop", "take", "reason"])
            for a in advices:
                w.writerow([now.isoformat(timespec="seconds"), a.symbol, a.action, f"{a.price:.8g}",
                            f"{a.prob_up:.4f}", f"{a.net_edge:.5f}", f"{a.amount_quote:.2f}",
                            f"{a.qty:.8g}", a.stop_price or "", a.take_price or "", a.reason])

    def write_latest(self, text: str) -> None:
        (self.dir / "latest.txt").write_text(text, encoding="utf-8")
