# risk_engine.py
from __future__ import annotations
import os, math
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# ---- env ----
ATR_MULT = float(os.getenv("RISK_ATR_MULT", "2.5"))
RV_MULT  = float(os.getenv("RISK_RV_MULT",  "2.0"))
MIN_STOP_FRAC = float(os.getenv("RISK_MIN_STOP_FRAC", "0.002"))

SPREAD_PAUSE_BPS = float(os.getenv("SPREAD_PAUSE_BPS", "15"))
SPREAD_WIDEN_MULT = float(os.getenv("SPREAD_WIDEN_MULT", "2.5"))
SPREAD_LOOKBACK = int(os.getenv("SPREAD_LOOKBACK", "20"))
SLIPPAGE_MAX_BPS = float(os.getenv("SLIPPAGE_MAX_BPS", "40"))

DAILY_DD_MAX = float(os.getenv("DAILY_DD_MAX", "0.05"))
DATA_GAP_MAX_MS = int(os.getenv("DATA_GAP_MAX_MS", "180000"))
TIME_STOP_BARS = int(os.getenv("TIME_STOP_BARS", os.getenv("MAX_HOLD_BARS", "48")))

MAX_PORTFOLIO_EXPOSURE_USDT = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_USDT", "0"))
PER_SYMBOL_NOTIONAL_USDT    = float(os.getenv("PER_SYMBOL_NOTIONAL_USDT", "0"))

def stop_distance_abs(price: float, atr: float, rv: float) -> float:
    """Absolute stop size in price units using ATR and RV."""
    atr_part = ATR_MULT * max(atr, 0.0)
    rv_part  = RV_MULT  * max(rv, 0.0) * max(price, 1e-12)
    min_part = MIN_STOP_FRAC * max(price, 1e-12)
    return max(atr_part, rv_part, min_part)

def size_from_risk(equity: float, risk_per_trade: float, price: float, stop_abs: float) -> float:
    """
    Position base qty sized so that 'risk_per_trade * equity' is lost if stop is hit.
    qty = (equity * R) / stop_abs  (1x, no leverage considered here)
    """
    if price <= 0 or stop_abs <= 0: return 0.0
    return (equity * risk_per_trade) / stop_abs

def cap_notional(price: float, qty: float, leverage: int, wallet_usdt: Optional[float]) -> float:
    """
    Reuse live caps: MAX_NOTIONAL_USDT, MAX_MARGIN_FRACTION. Plus new caps:
    - PER_SYMBOL_NOTIONAL_USDT
    - MAX_PORTFOLIO_EXPOSURE_USDT (handled outside per symbol)
    """
    from math import inf
    q = max(qty, 0.0)
    if price <= 0: return 0.0
    max_not_global = float(os.getenv("MAX_NOTIONAL_USDT", "0"))
    if max_not_global > 0:
        q = min(q, max_not_global / price)
    if PER_SYMBOL_NOTIONAL_USDT > 0:
        q = min(q, PER_SYMBOL_NOTIONAL_USDT / price)
    max_margin_frac = float(os.getenv("MAX_MARGIN_FRACTION", "0.0"))
    if wallet_usdt is not None and max_margin_frac > 0 and leverage > 0:
        max_margin   = wallet_usdt * max_margin_frac
        max_notional = max_margin * leverage
        q = min(q, max_notional / price)
    return max(q, 0.0)

def should_pause_spread(cur_spread_bps: float, roll_median_bps: Optional[float]) -> bool:
    """Pause entries if spread too wide or widened abnormally."""
    if not np.isfinite(cur_spread_bps): return False
    if cur_spread_bps > SPREAD_PAUSE_BPS: return True
    if roll_median_bps and np.isfinite(roll_median_bps):
        if cur_spread_bps > SPREAD_WIDEN_MULT * roll_median_bps:
            return True
    return False

def has_data_gap(df: pd.DataFrame, timeframe_ms: int = 60000) -> bool:
    """True if the last gap exceeds DATA_GAP_MAX_MS."""
    if df is None or df.empty or "ts" not in df.columns: return True
    ts = df["ts"].dropna().astype("int64")
    if ts.size < 3: return False
    gap = int(ts.iloc[-1]) - int(ts.iloc[-2])
    return gap > max(DATA_GAP_MAX_MS, timeframe_ms * 3)

def portfolio_exposure_ok(open_positions: Dict[str, Dict[str, Any]], price_map: Dict[str, float]) -> bool:
    """Check against MAX_PORTFOLIO_EXPOSURE_USDT."""
    if MAX_PORTFOLIO_EXPOSURE_USDT <= 0: return True
    total = 0.0
    for sym, st in (open_positions or {}).items():
        if not st.get("in_pos"): continue
        px = price_map.get(sym)
        if px and st.get("qty"):
            total += float(px) * float(st["qty"])
    return total <= MAX_PORTFOLIO_EXPOSURE_USDT
