from __future__ import annotations
import os, asyncio, math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import ccxt.async_support as ccxt_async  # asyncio version of ccxt

# Env controls (with safe defaults)
ANALYZE_EXCHANGES = [
    e.strip() for e in os.getenv("ANALYZE_EXCHANGES", "binance,bybit,mexc,bitget").split(",") if e.strip()
]
ROUTER_DEFAULT_TAKER_BPS = float(os.getenv("ROUTER_DEFAULT_TAKER_BPS", "10"))
ROUTER_ENABLED = bool(int(os.getenv("ENABLE_ROUTER", "1")))

@dataclass
class Quote:
    venue: str
    vwap: float           # book VWAP for the requested amount
    taker_bps: float      # taker fee in bps (not "edge" vs. mid)
    depth_ok: bool
    mtype: str            # 'spot' or 'swap'
    reason: str = ""

def _guess_type_order(symbol: str) -> List[str]:
    """
    If the symbol looks like a perpetual (contains ':USDT'), try swap first.
    Otherwise try spot first, then swap.
    """
    return ["swap", "spot"] if (":USDT" in symbol or ":" in symbol) else ["spot", "swap"]

def _build_opts(ex_id: str, mtype: str) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"enableRateLimit": True, "timeout": 20000, "options": {}}
    # sensible defaults per exchange
    if ex_id == "mexc":
        # we only need spot for MEXC quotes
        opts["options"]["defaultType"] = "spot"
    elif ex_id == "bitget":
        if mtype == "swap":
            opts["options"].update({"defaultType": "swap", "defaultSubType": "linear", "defaultSettle": "USDT"})
        else:
            opts["options"]["defaultType"] = "spot"
    elif ex_id == "bybit":
        if mtype == "swap":
            opts["options"].update({"defaultType": "swap", "defaultSubType": "linear"})
        else:
            opts["options"]["defaultType"] = "spot"
    elif ex_id == "binance":
        if mtype == "swap":
            # USDT-M futures
            opts["options"]["defaultType"] = "future"
        else:
            opts["options"]["defaultType"] = "spot"
    return opts

async def _open_client_for_symbol(ex_id: str, symbol: str) -> Tuple[ccxt_async.Exchange, str]:
    """
    Try to open a client with the right market type so that `symbol` exists.
    Returns (client, mtype).
    """
    last_err: Optional[Exception] = None
    for mtype in _guess_type_order(symbol):
        ex_cls = getattr(ccxt_async, ex_id)
        ex = ex_cls(_build_opts(ex_id, mtype))
        try:
            await ex.load_markets()
            if symbol in getattr(ex, "symbols", []):
                return ex, mtype
            # symbol not found with this type
            await ex.close()
        except Exception as e:
            last_err = e
            try:
                await ex.close()
            except Exception:
                pass
    raise last_err or Exception(f"{ex_id}: symbol {symbol!r} not available in spot/swap")

def _vwap_from_orderbook(ob: Dict[str, Any], side: str, amount: float) -> Tuple[Optional[float], bool]:
    levels = ob.get("asks" if side == "buy" else "bids", [])
    if not levels or amount <= 0:
        return None, False
    left = float(amount)
    cost = 0.0
    for px, sz in levels:
        take = min(left, float(sz))
        cost += take * float(px)
        left -= take
        if left <= 1e-12:
            break
    if left > 1e-12:
        return None, False
    return cost / float(amount), True

async def _quote_one(ex_id: str, symbol: str, side: str, amount: float, default_bps: float) -> Optional[Quote]:
    try:
        ex, mtype = await _open_client_for_symbol(ex_id, symbol)
    except Exception:
        return None
    try:
        ob = await ex.fetch_order_book(symbol, limit=50)
        vwap, depth_ok = _vwap_from_orderbook(ob, side, amount)
        if vwap is None:
            return Quote(ex_id, math.inf, float("inf"), False, mtype, reason="no_depth")

        fee_bps = default_bps
        try:
            mk = ex.markets.get(symbol, {})
            if isinstance(mk.get("taker"), (int, float,)):
                fee_bps = float(mk["taker"]) * 10000.0  # fraction → bps
        except Exception:
            pass

        return Quote(ex_id, float(vwap), float(fee_bps), True, mtype)
    except Exception:
        return None
    finally:
        try:
            await ex.close()
        except Exception:
            pass

async def best_cex_quote_async(
    exchanges: List[str],
    symbol: str,
    side: str,                # 'buy' or 'sell'
    amount: float,            # base amount (spot) or contracts (swap) matching the venue
    default_bps: float
) -> Optional[Dict[str, Any]]:
    tasks = [
        asyncio.create_task(_quote_one(e, symbol, side, amount, default_bps))
        for e in exchanges if e
    ]
    if not tasks:
        return None

    results = await asyncio.gather(*tasks, return_exceptions=False)
    best: Optional[Tuple[Quote, float]] = None

    for q in results:
        if q is None or not q.depth_ok or not math.isfinite(q.vwap):
            continue

        # pick the best *effective* price including taker fee
        if side == "buy":
            eff_px = q.vwap * (1.0 + q.taker_bps / 10000.0)
            better = lambda a, b: a < b
        else:
            eff_px = q.vwap * (1.0 - q.taker_bps / 10000.0)
            better = lambda a, b: a > b

        if best is None or better(eff_px, best[1]):
            best = (q, eff_px)

    if best is None:
        return None

    q = best[0]
    # Return a dict to stay compatible with existing trade.py usage
    return {
        "venue": q.venue,
        "mtype": q.mtype,
        "vwap": q.vwap,
        "eff_bps": q.taker_bps,   # for backward-compatibility: this is taker fee bps
        "depth_ok": q.depth_ok,
        "reason": q.reason,
    }

# Synchronous facade so existing code can just call best_cex_quote(...)
def best_cex_quote(symbol: str, side: str, amount: float) -> Optional[Dict[str, Any]]:
    try:
        return asyncio.run(
            best_cex_quote_async(
                ANALYZE_EXCHANGES, symbol, side, amount, ROUTER_DEFAULT_TAKER_BPS
            )
        )
    except RuntimeError:
        # If an event loop is already running (rare in your CLI app), fall back to a new loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                best_cex_quote_async(
                    ANALYZE_EXCHANGES, symbol, side, amount, ROUTER_DEFAULT_TAKER_BPS
                )
            )
        finally:
            loop.close()
