
# AI Trading Bot — MEXC Tailored (Python + CCXT + scikit-learn)

> **Default = Paper Trading.** Live trading is **OFF**. Turn on only if you understand the risks and have tested thoroughly.

### What’s different vs the generic version?
- `.env` defaults to **MEXC** (`EXCHANGE_ID=mexc`).
- CCXT client sets `options.defaultType='spot'` for MEXC.
- **Bugfix**: `trade.py` now records `entry` price correctly on position entry.

---

## Quick Start on Windows (PowerShell)

```powershell
# 1) Create & activate a venv
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2) Install deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) Configure
Copy-Item .env.example .env
# Edit .env as needed (SYMBOL, TIMEFRAME, risk, etc.)

# 4) Train model on historical data
python model.py

# 5) Backtest rules
python backtest.py

# 6) Paper trade (simulated)
python trade.py
```

### Live trading (optional; at your own risk)
- Set `LIVE_MODE=1` in `.env`.
- Add your **MEXC** API key/secret (spot permissions).
- Consider IP whitelisting. Keep keys safe.

---

## Files

```
ai-trading-bot-mexc/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ config.py
├─ data.py
├─ features.py
├─ model.py
├─ backtest.py
├─ trade.py         # bugfix: sets entry price on entry
├─ exchange.py      # CCXT client for MEXC with defaultType=spot
├─ utils.py
├─ models/
└─ logs/
```
