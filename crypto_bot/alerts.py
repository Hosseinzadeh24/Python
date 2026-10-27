"""High-opportunity alerts: console, log file, and optional Telegram message."""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


@dataclass
class Opportunity:
    symbol: str
    time: str
    price: float
    prob_up: float
    net_edge: float
    take_price: float
    stop_price: float

    def message(self) -> str:
        return (f"[ALERT] {self.symbol} @ {self.price:.6g}  P(up)={self.prob_up:.0%}  "
                f"expected net gain={self.net_edge:+.2%}  target={self.take_price:.6g}  "
                f"stop={self.stop_price:.6g}  time={self.time}")


class Notifier:
    def __init__(self, data_dir: str | Path):
        self.log_path = Path(data_dir) / "alerts.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.recent: deque[dict] = deque(maxlen=20)   # shown in status.json for the phone app

    def send(self, opp: Opportunity) -> None:
        msg = opp.message()
        print(msg, flush=True)
        self.recent.append(asdict(opp))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(opp), ensure_ascii=False) + "\n")
        if self.token and self.chat_id:
            try:
                requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              json={"chat_id": self.chat_id, "text": msg}, timeout=10)
            except requests.RequestException as exc:  # never let a notification crash trading
                print(f"telegram failed: {exc}", flush=True)
