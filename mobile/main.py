"""Android monitor (Kivy). Shows the 24/7 bot's status fetched from its status server.

The phone does NOT run the trading engine: Android suspends background apps, so the
bot runs on a server (see deploy/) and this app only displays its status and alerts.
"""
import json
import threading
import urllib.request

from kivy.app import App
from kivy.clock import Clock
from kivy.storage.jsonstore import JsonStore
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput


class MonitorRoot(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", padding=10, spacing=6, **kw)
        self.store = JsonStore("settings.json")
        saved = self.store.get("conn") if self.store.exists("conn") else {"url": "", "token": ""}
        self.url = TextInput(text=saved["url"], hint_text="http://SERVER:8080/status", multiline=False,
                             size_hint_y=None, height=44)
        self.token = TextInput(text=saved["token"], hint_text="status token", password=True,
                               multiline=False, size_hint_y=None, height=44)
        self.add_widget(self.url)
        self.add_widget(self.token)
        btn = Button(text="Refresh", size_hint_y=None, height=50)
        btn.bind(on_release=lambda *_: self.refresh())
        self.add_widget(btn)
        self.info = Label(text="Not connected yet", halign="left", valign="top", font_size="15sp")
        self.info.bind(size=lambda w, s: setattr(w, "text_size", (s[0], None)))
        scroll = ScrollView()
        scroll.add_widget(self.info)
        self.add_widget(scroll)
        self._seen_alerts = set()
        Clock.schedule_interval(lambda *_: self.refresh(), 60)

    def refresh(self):
        self.store.put("conn", url=self.url.text.strip(), token=self.token.text.strip())
        threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self):
        try:
            req = urllib.request.Request(self.url.text.strip(),
                                         headers={"X-Token": self.token.text.strip()})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = self._format(data)
        except Exception as exc:  # show the error instead of crashing the app
            text = f"Connection error:\n{exc}"
        Clock.schedule_once(lambda *_: setattr(self.info, "text", text))

    def _format(self, d):
        if not d.get("updated_utc"):
            return "No status yet. Is the bot running?"
        lines = [
            f"Mode: {d.get('mode')}   Halted: {d.get('halted')}",
            f"Updated: {d.get('updated_utc')}",
            f"Equity: {d.get('equity'):.2f}   Capital: {d.get('capital'):.2f}",
            f"Total return: {d.get('total_return', 0):+.2%}",
            f"Daily P/L: {d.get('daily_pnl', 0):+.2f}   Cash: {d.get('cash', 0):.2f}",
            "",
            "Open positions:",
        ]
        for sym, p in (d.get("positions") or {}).items():
            lines.append(f"  {sym}: entry {p['entry']:.6g}  stop {p['stop']:.6g}  take {p['take']:.6g}")
        if not d.get("positions"):
            lines.append("  none")
        lines += ["", "P(up) by coin:"]
        for sym, v in (d.get("prob_up") or {}).items():
            lines.append(f"  {sym}: {v:.0%}")
        lines += ["", "Recent alerts:"]
        for a in (d.get("recent_alerts") or [])[-5:]:
            lines.append(f"  {a['time']} {a['symbol']} P={a['prob_up']:.0%} net={a['net_edge']:+.2%}")
        if not d.get("recent_alerts"):
            lines.append("  none")
        return "\n".join(lines)


class CryptoMonitorApp(App):
    title = "Crypto Monitor"

    def build(self):
        return MonitorRoot()


if __name__ == "__main__":
    CryptoMonitorApp().run()
