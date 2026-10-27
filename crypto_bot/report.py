"""Matplotlib charts for calibration results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _pct(x: float) -> str:
    return f"{x:+.2%}"


def plot_period(period: dict, eq_df, path: str | Path, targets: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    base = eq_df.iloc[0]
    for col, style in (("bot", "-"), ("sma_baseline", "--"), ("buy_hold", ":")):
        ax.plot(eq_df.index, eq_df[col] / eq_df[col].iloc[0] * 100 - 100, style, label=col)
    ax.axhline(0, color="grey", lw=0.8)
    for t in targets:
        ax.axhline(t * 100, color="green", lw=0.6, ls="--", alpha=0.6)
    ax.set_title(f"Period {period['period']}  |  bot {_pct(period['bot_return'])}  "
                 f"baseline {_pct(period['baseline_return'])}  buy&hold {_pct(period['buy_hold_return'])}  "
                 f"| trades {period['bot_trades']}  maxDD {period['bot_max_drawdown']:.2%}")
    ax.set_ylabel("return since period start (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(periods: list[dict], path: str | Path, targets: list[float]) -> None:
    import numpy as np

    x = np.arange(len(periods))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, [p["bot_return"] * 100 for p in periods], w, label="bot (out-of-sample)")
    ax.bar(x, [p["baseline_return"] * 100 for p in periods], w, label="SMA baseline")
    ax.bar(x + w, [p["buy_hold_return"] * 100 for p in periods], w, label="buy & hold")
    for t in targets:
        ax.axhline(t * 100, color="green", ls="--", lw=0.8, label=f"target {int(t*100)}%")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"week {p['period']}" for p in periods])
    ax.set_ylabel("weekly return (%)")
    ax.set_title("Calibration: weekly results per period")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(history: list[dict], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    runs = [h["run"] for h in history]
    ax.plot(runs, [h["mean_period_return"] * 100 for h in history], "o-", label="mean weekly return (bot)")
    ax.plot(runs, [h["min_period_return"] * 100 for h in history], "s--", label="worst week (bot)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("%")
    ax.set_title("Calibration runs: comparison with previous runs")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
