"""Window (Tkinter, comes with Windows Python). All trading logic is in bot.py / trader.py.

The bot runs in a background thread; the window only reads messages from a queue, so it never freezes.
"""
from __future__ import annotations

import queue
import threading

import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk

from .bot import Bot
from .runner import Runner
from .config import Config
from .form import LIVE_PHRASE, live_confirmed, parse_form
from .secrets_store import delete_key, load_key, mask, save_key, source


class App:
    def __init__(self, root: tk.Tk, cfg: Config):
        self.root = root
        self.cfg = cfg
        self.q: queue.Queue = queue.Queue()
        self.runner: Runner | None = None
        self.api_key: str | None = load_key()
        root.title("Crypto advisor - Wallex")
        root.geometry("1080x760")
        self._build()
        self._fill_form()
        self._refresh_key_label()
        self.root.after(250, self._pump)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        r = self.root
        top = ttk.LabelFrame(r, text="Wallex API key  (کلید API والکس)")
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="API key:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(top, textvariable=self.key_var, show="*", width=60)
        self.key_entry.grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Save key", command=self.on_save_key).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Test key", command=self.on_test_key).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="Delete key", command=self.on_delete_key).grid(row=0, column=4, padx=4)
        self.key_label = ttk.Label(top, text="")
        self.key_label.grid(row=1, column=0, columnspan=5, sticky="w", padx=4, pady=(0, 4))
        top.columnconfigure(1, weight=1)

        risk = ttk.LabelFrame(r, text="Settings  (تنظیمات)")
        risk.pack(fill="x", padx=8, pady=4)
        self.vars = {k: tk.StringVar() for k in ("max_loss", "min_profit", "capital", "exposure", "symbols")}
        fields = [
            ("Max loss % (1 to 15)", "max_loss"),
            ("Min profit % (take-profit)", "min_profit"),
            ("Capital (USDT)", "capital"),
            ("Max invested %", "exposure"),
            ("Markets (comma separated)", "symbols"),
        ]
        for i, (label, key) in enumerate(fields):
            ttk.Label(risk, text=label).grid(row=i // 3, column=(i % 3) * 2, sticky="e", padx=(8, 2), pady=3)
            ttk.Entry(risk, textvariable=self.vars[key], width=18 if key != "symbols" else 30).grid(
                row=i // 3, column=(i % 3) * 2 + 1, sticky="w", padx=(0, 10), pady=3)
        self.mode_var = tk.StringVar(value="suggest")
        mode_box = ttk.Frame(risk)
        mode_box.grid(row=2, column=0, columnspan=6, sticky="w", padx=8, pady=3)
        ttk.Label(mode_box, text="Mode:").pack(side="left")
        for val, txt in (("suggest", "Suggest only (no orders)"), ("dry_run", "Dry run (simulated, no money)"),
                         ("live", "LIVE (real orders)")):
            ttk.Radiobutton(mode_box, text=txt, value=val, variable=self.mode_var).pack(side="left", padx=8)
        self.msg = ttk.Label(risk, text="", foreground="#b00020", wraplength=1000, justify="left")
        self.msg.grid(row=3, column=0, columnspan=6, sticky="w", padx=8, pady=2)

        btns = ttk.Frame(r)
        btns.pack(fill="x", padx=8, pady=4)
        ttk.Button(btns, text="Save settings", command=self.on_save_settings).pack(side="left", padx=4)
        self.start_btn = ttk.Button(btns, text="Start", command=self.on_start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.sell_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btns, text="also sell all positions", variable=self.sell_all_var).pack(side="left", padx=12)
        tk.Button(btns, text="KILL SWITCH (cancel + stop)", bg="#b00020", fg="white",
                  command=self.on_kill).pack(side="left", padx=6)
        ttk.Button(btns, text="Resume trading", command=self.on_resume).pack(side="left", padx=4)

        self.status = ttk.Label(r, text="Not running.", font=("Segoe UI", 10, "bold"))
        self.status.pack(fill="x", padx=10, pady=2)

        cols = ("symbol", "bid", "ask", "last", "24h %", "P(up)", "advice", "position")
        self.tree = ttk.Treeview(r, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.pack(fill="x", padx=8, pady=4)

        self.log = scrolledtext.ScrolledText(r, height=18, font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, padx=8, pady=(4, 8))

    def _fill_form(self) -> None:
        c = self.cfg
        self.vars["max_loss"].set(f"{c.max_loss_pct:g}")
        self.vars["min_profit"].set(f"{c.min_profit_pct:g}")
        self.vars["capital"].set(f"{c.capital:g}")
        self.vars["exposure"].set(f"{c.max_total_exposure * 100:g}")
        self.vars["symbols"].set(", ".join(c.symbols))
        self.mode_var.set(c.trading_mode)

    def _refresh_key_label(self) -> None:
        self.key_label.config(text=f"Saved key: {mask(self.api_key)}   source: {source()}")

    def _values(self) -> dict:
        return {**{k: v.get() for k, v in self.vars.items()}, "mode": self.mode_var.get()}

    def _log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _show_errors(self, errs: list[str]) -> None:
        self.msg.config(text="\n".join(errs))

    # ------------------------------------------------------------------ actions
    def on_save_key(self) -> None:
        key = self.key_var.get().strip()
        if not key:
            self._show_errors(["Paste the key in the box first."])
            return
        try:
            save_key(key)
        except Exception as exc:
            self._show_errors([f"Could not save the key: {exc}"])
            return
        self.api_key = key
        self.key_var.set("")
        self._show_errors([])
        self._refresh_key_label()
        self._log("API key saved to Windows Credential Manager (it is not written to any file).")

    def on_delete_key(self) -> None:
        if messagebox.askyesno("Delete key", "Delete the saved Wallex API key from this computer?"):
            delete_key()
            self.api_key = None
            self._refresh_key_label()

    def _current_key(self) -> str | None:
        return self.key_var.get().strip() or self.api_key

    def on_test_key(self) -> None:
        key = self._current_key()
        if not key:
            self._show_errors(["No key: paste it and press Save key first."])
            return
        self._log("Testing key (read-only request) ...")
        self._in_thread(lambda: self._test_key_work(key))

    def _test_key_work(self, key: str) -> None:
        from .wallex import WallexClient
        try:
            bal = WallexClient(key).balances()
            self.q.put(("log", f"Key works. Assets on the account: {len(bal)}. No orders were placed."))
        except Exception as exc:
            self.q.put(("log", f"Key test failed: {exc}"))

    def on_save_settings(self) -> None:
        cfg, errs = parse_form(self.cfg, self._values(), has_key=bool(self._current_key()))
        if errs:
            self._show_errors(errs)
            return
        self._show_errors([])
        cfg.save()
        self.cfg = cfg
        self._log(f"Settings saved to config.json (max loss {cfg.max_loss_pct:g}%, "
                  f"min profit {cfg.min_profit_pct:g}%, mode {cfg.trading_mode}).")

    def on_start(self) -> None:
        if self.runner is not None:
            return
        cfg, errs = parse_form(self.cfg, self._values(), has_key=bool(self._current_key()))
        if errs:
            self._show_errors(errs)
            return
        if cfg.trading_mode == "live":
            text = simpledialog.askstring(
                "Live trading", "LIVE mode places REAL orders with REAL money on Wallex.\n"
                "You can lose money, including the whole capital you allow.\n"
                "Stop-loss only works while this window is running.\n\n"
                f"Type exactly:  {LIVE_PHRASE}", parent=self.root)
            if not live_confirmed(text or ""):
                self._show_errors(["Live trading was not started (confirmation text did not match)."])
                return
        if cfg.trading_mode == "suggest":
            key = None
        else:
            key = self._current_key()
        self._show_errors([])
        try:
            cfg.save()
            self.cfg = cfg
            self.runner = Runner(cfg, key, self.q)
        except Exception as exc:
            self._show_errors([f"Could not start: {exc}"])
            self.runner = None
            return
        self.runner.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text=f"Running: {cfg.trading_mode}")

    def on_stop(self) -> None:
        if self.runner:
            self.runner.stop()
            self.stop_btn.config(state="disabled")

    def on_kill(self) -> None:
        sell_all = self.sell_all_var.get()
        what = "cancel all orders placed by this app AND sell all its positions" if sell_all else \
            "cancel all orders placed by this app and stop new buys (positions are kept and protected)"
        if not messagebox.askyesno("Kill switch", f"This will {what}.\n\nContinue?"):
            return
        if self.runner is None:
            key = self._current_key() if self.cfg.trading_mode != "suggest" else None
            self._in_thread(lambda: self._kill_work(key, sell_all, None))
            return
        self._in_thread(lambda: self._kill_work(None, sell_all, self.runner))

    def _kill_work(self, key, sell_all, runner) -> None:
        try:
            if runner is not None:
                with runner.lock:
                    events = runner.bot.kill_switch(sell_all)
            else:
                bot = Bot(self.cfg, api_key=key, log=lambda m: self.q.put(("log", m)))
                bot.refresh_markets()
                events = bot.kill_switch(sell_all)
            for e in events:
                self.q.put(("log", f"KILL: {e}"))
        except Exception as exc:
            self.q.put(("log", f"Kill switch error: {exc}. Check your orders on wallex.ir."))

    def on_resume(self) -> None:
        if self.runner is None:
            self._show_errors(["Start the bot first, then resume."])
            return
        if messagebox.askyesno("Resume", "Resume trading (new buys allowed again)?"):
            with self.runner.lock:
                self.runner.bot.resume()

    def _in_thread(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    # ------------------------------------------------------------------ message pump
    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind in ("log", "log_detail"):
                    self._log(str(payload))
                elif kind == "snapshot":
                    self._show_snapshot(payload)
                elif kind == "stopped":
                    self.runner = None
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status.config(text="Not running.")
        except queue.Empty:
            pass
        self.root.after(250, self._pump)

    def _show_snapshot(self, data: dict) -> None:
        self.tree.delete(*self.tree.get_children())
        for row in data["rows"]:
            sym, bid, ask, last, ch, pup, act, pos = row
            self.tree.insert("", "end", values=(sym, f"{bid:.8g}", f"{ask:.8g}", f"{last:.8g}",
                                                f"{ch:+.2f}", pup, act, pos))
        s = data["summary"]
        halt = f"HALTED: {s['halt_reason']}" if s["halted"] else "trading allowed"
        self.status.config(text=f"{self.cfg.trading_mode}  |  {halt}  |  realised today {s['realised_today']:+.2f}"
                                f"  total {s['realised_total']:+.2f}  |  open positions {s['open_positions']}"
                                f"  |  exposure {s['exposure']:.2f} USDT")


def run_gui(cfg: Config) -> None:
    root = tk.Tk()
    App(root, cfg)
    root.mainloop()
