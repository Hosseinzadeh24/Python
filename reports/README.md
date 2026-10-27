# Calibration reports

- `demo_synthetic/` — **داده مصنوعی (synthetic) — فقط برای نشان دادن خروجی**.
  این نتایج از بازار واقعی گرفته نشده و هیچ ارزشی برای قضاوت درباره سودآوری ندارند.
  The files here were generated from random-walk data to demonstrate the report format.
  They say nothing about real-market profitability.

خروجی واقعی را با دستور زیر بسازید (داده عمومی صرافی):

```bash
python -m crypto_bot calibrate --source ccxt --periods 4
```

Each run creates `run_<time>/` with `results.json`, per-period equity CSVs and charts
(`charts/period_N.png`, `charts/summary.png`). `history.json` compares runs with each other.
