"""Walk-forward calibration.

For each evaluation period (default 7 days):
  1. Train the ensemble on a rolling window of past data.
  2. Choose trading parameters (entry/exit thresholds, stop/take sizes, time stop)
     by testing a small grid on the LAST part of the training window only (validation).
  3. Retrain on the whole training window and trade the NEXT 7 days (out-of-sample).

The test week is never used to pick parameters. Reported returns are therefore
out-of-sample estimates. They are NOT a promise of future profit.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import SimResult, TradeParams, metrics, simulate
from .config import Config
from .features import build_features, forward_labels
from .models import Ensemble, sma_crossover_prob
from .risk import RiskLimits

PARAM_GRID = {
    "p_enter": [0.55, 0.60, 0.65],
    "p_exit": [0.45, 0.50],
    "stop_mult": [1.5, 2.5],
    "take_mult": [2.0, 3.0],
    "max_hold_bars": [12, 24],
}
WARMUP = 72  # bars of history needed by the indicators before a window starts
VALIDATION_SHARE = 0.25


def param_grid(grid: dict = PARAM_GRID) -> list[TradeParams]:
    keys = list(grid)
    return [TradeParams(**dict(zip(keys, vals))) for vals in itertools.product(*(grid[k] for k in keys))]


def _score(res: SimResult, start_equity: float) -> float:
    """Validation objective: return penalised by drawdown. Chosen on training data only."""
    m = metrics(res, start_equity)
    return m["total_return"] - 0.5 * abs(m["max_drawdown"])


def _window(df: pd.DataFrame, prob: pd.Series, lo: int, hi: int) -> tuple[pd.DataFrame, pd.Series]:
    """Return rows [lo-WARMUP, hi) with probabilities masked during warm-up (no trades there)."""
    start = max(0, lo - WARMUP)
    sub = df.iloc[start:hi]
    p = prob.iloc[start:hi].copy()
    p.iloc[: lo - start] = np.nan
    return sub, p


def _run_window(df, prob, lo, hi, params, limits, start_equity, min_edge=0.0) -> SimResult:
    sub, p = _window(df, prob, lo, hi)
    res = simulate(sub, p, params, limits, start_equity=start_equity, min_net_edge=min_edge)
    k = lo - max(0, lo - WARMUP)
    eq = res.equity.iloc[k:]
    trades = res.trades[res.trades["entry_time"] >= df.index[lo]] if len(res.trades) else res.trades
    return SimResult(equity=eq, trades=trades.reset_index(drop=True),
                     in_position=res.in_position.iloc[k:], halted=res.halted)


def walk_forward(frames: dict[str, pd.DataFrame], cfg: Config, periods: int | None = None,
                 grid: list[TradeParams] | None = None,
                 log: Callable[[str], None] = print) -> dict:
    periods = periods or cfg.calibration_periods
    grid = grid or param_grid()
    symbols = list(frames)
    common = frames[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(frames[s].index)
    frames = {s: frames[s].loc[common] for s in symbols}
    n = len(common)
    H, train, test = cfg.horizon_bars, cfg.train_bars, cfg.test_bars
    need = train + periods * test
    if n < need:
        raise ValueError(f"Need at least {need} aligned bars for {periods} periods, got {n}")

    limits = RiskLimits(cfg.fee_rate, cfg.slippage, cfg.kelly_fraction,
                        cfg.max_position_frac, cfg.max_daily_loss_frac, cfg.max_drawdown_kill)
    share = cfg.capital / len(symbols)
    feats = {s: build_features(frames[s]) for s in symbols}
    labels = {}
    for s in symbols:
        _, lab = forward_labels(frames[s], H, limits.round_trip_cost)
        labels[s] = lab

    out_periods, eq_frames = [], []
    t_start = time.time()
    for k in range(periods):
        t0 = train + k * test          # first bar of the test week
        t1 = t0 + test                 # end (exclusive)
        tr_lo = t0 - train
        val_cut = tr_lo + int(train * (1 - VALIDATION_SHARE))
        p_bot, p_base = {}, {}
        chosen, sym_metrics, sym_bh, sym_base = {}, {}, {}, {}
        sym_eqs, sym_eq_base = [], []
        for s in symbols:
            X, y = feats[s], labels[s]
            ok = X.notna().all(axis=1) & y.notna()

            def rows(lo, hi_excl_label):
                # training rows whose label horizon ends before hi_excl_label (no leakage)
                idx = np.arange(lo, hi_excl_label - H)
                sel = X.index[idx][ok.iloc[idx].to_numpy()]
                return X.loc[sel], y.loc[sel]

            # (a) inner validation: fit on the early part, score grid on the late part
            Xa, ya = rows(tr_lo, val_cut)
            model_a = Ensemble(cfg.seed).fit(Xa, ya)
            prob_val = model_a.predict_proba(X.iloc[val_cut:t0].dropna()).reindex(X.index[val_cut:t0])
            prob_val_full = pd.Series(np.nan, index=frames[s].index)
            prob_val_full.loc[prob_val.index] = prob_val.values
            best, best_score = None, -np.inf
            for prm in grid:
                r = _run_window(frames[s], prob_val_full, val_cut, t0, prm, limits, share)
                sc = _score(r, share)
                if sc > best_score:
                    best, best_score = prm, sc
            chosen[s] = asdict(best)

            # (b) final model on the whole training window -> trade the test week
            Xf, yf = rows(tr_lo, t0)
            model_f = Ensemble(cfg.seed).fit(Xf, yf)
            prob_t = model_f.predict_proba(X.iloc[t0 - WARMUP:t1].dropna()).reindex(frames[s].index[t0 - WARMUP:t1])
            prob_full = pd.Series(np.nan, index=frames[s].index)
            prob_full.loc[prob_t.index] = prob_t.values

            res = _run_window(frames[s], prob_full, t0, t1, best, limits, share)
            sym_metrics[s] = metrics(res, share)
            sym_eqs.append(res.equity.rename(s))

            # benchmarks on the same week
            c = frames[s]["close"]
            sym_bh[s] = float(c.iloc[t1 - 1] / c.iloc[t0] - 1 - 2 * cfg.fee_rate)
            base_prob = pd.Series(np.nan, index=frames[s].index)
            base_prob.iloc[t0 - WARMUP:t1] = sma_crossover_prob(X).iloc[t0 - WARMUP:t1].values
            rb = _run_window(frames[s], base_prob, t0, t1, TradeParams(), limits, share)
            sym_base[s] = metrics(rb, share)
            sym_eq_base.append(rb.equity.rename(s))

        bot_eq = sum(sym_eqs).rename("bot")
        base_eq = sum(sym_eq_base).rename("sma_baseline")
        bh_series = sum(share * frames[s]["close"].iloc[t0:t1] / frames[s]["close"].iloc[t0] for s in symbols)
        bh_eq = (bh_series * (1 - 2 * cfg.fee_rate)).rename("buy_hold")
        eq_df = pd.concat([bot_eq, base_eq, bh_eq], axis=1)
        eq_frames.append(eq_df)

        bot_ret = float(bot_eq.iloc[-1] / cfg.capital - 1)
        base_ret = float(base_eq.iloc[-1] / cfg.capital - 1)
        bh_ret = float(np.mean(list(sym_bh.values())))
        trades = int(sum(m["trades"] for m in sym_metrics.values()))
        period = {
            "period": k + 1,
            "start": str(common[t0]),
            "end": str(common[t1 - 1]),
            "bot_return": bot_ret,
            "bot_max_drawdown": float(min(m["max_drawdown"] for m in sym_metrics.values())),
            "bot_trades": trades,
            "bot_win_rate": float(np.average([m["win_rate"] for m in sym_metrics.values()],
                                             weights=[max(m["trades"], 1) for m in sym_metrics.values()])),
            "baseline_return": base_ret,
            "buy_hold_return": bh_ret,
            "beats_baseline": bot_ret > base_ret,
            "per_symbol_return": {s: sym_metrics[s]["total_return"] for s in symbols},
            "per_symbol_buy_hold": sym_bh,
            "chosen_params": chosen,
            "target_hits": {f"{int(t*100)}%": bot_ret >= t for t in cfg.target_weekly_returns},
            "end_equity": float(bot_eq.iloc[-1]),
        }
        out_periods.append(period)
        log(f"period {k+1}/{periods}  {period['start'][:10]}..{period['end'][:10]}  "
            f"bot {bot_ret:+.2%}  baseline {base_ret:+.2%}  buy&hold {bh_ret:+.2%}  "
            f"trades {trades}  ({time.time()-t_start:.0f}s)")

    rets = np.array([p["bot_return"] for p in out_periods])
    summary = {
        "periods": periods,
        "mean_period_return": float(rets.mean()),
        "std_period_return": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "min_period_return": float(rets.min()),
        "max_period_return": float(rets.max()),
        "periods_beating_baseline": int(sum(p["beats_baseline"] for p in out_periods)),
        "target_hit_counts": {f"{int(t*100)}%": int(sum(p["target_hits"][f"{int(t*100)}%"] for p in out_periods))
                              for t in cfg.target_weekly_returns},
        "note": "Out-of-sample estimates on the data provided. Not a guarantee of future profit.",
    }
    return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "periods": out_periods, "summary": summary, "equity": eq_frames,
            "config": cfg.to_dict()}


def save_results(result: dict, out_dir: str | Path) -> Path:
    from .report import plot_history, plot_period, plot_summary

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "equity"}
    (out / "results.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    for i, (p, eq) in enumerate(zip(result["periods"], result["equity"]), start=1):
        eq.to_csv(out / f"period_{i}_equity.csv")
        plot_period(p, eq, charts / f"period_{i}.png", result["config"]["target_weekly_returns"])
    plot_summary(result["periods"], charts / "summary.png", result["config"]["target_weekly_returns"])
    hist_path = Path(out_dir).parent / "history.json"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    hist.append({"run": out.name, "created_utc": result["created_utc"], **result["summary"]})
    hist_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    plot_history(hist, charts / "history.png")
    return out
