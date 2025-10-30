"""
Async, multi-symbol scanner using ccxt.async_support + CPU process pool
- Bitget USDT-M universe (top-N by quote volume)
- Pre-sanitize symbols
- Async OHLCV fetch for all symbols
- Feature build + scoring in ProcessPoolExecutor (configurable cores)
- Optional cross-venue last prices (binance/bybit/bitget/mexc)
"""

from __future__ import annotations
import os, math, json, asyncio, itertools
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import ccxt.async_support as ccxt_async
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local project imports
import joblib
from features import build_features

# ===================== ENV / knobs =====================
TIMEFRAME          = os.getenv("TIMEFRAME", "5m")
LOOKBACK           = int(os.getenv("LOOKBACK_CANDLES", "2000"))
SCAN_TOPN          = int(os.getenv("SCAN_TOPN", "25"))
SCAN_MIN_NOTIONAL  = float(os.getenv("SCAN_MIN_NOTIONAL", "100000"))
SYMBOL_WHITELIST   = [s.strip() for s in os.getenv("SYMBOL_WHITELIST","").split(",") if s.strip()]
BITGET_SANDBOX     = bool(int(os.getenv("BITGET_SANDBOX","0")))
MODEL_PATH         = os.getenv("MODEL_PATH", "models/model.pkl")

# Async concurrency
HTTP_WORKERS       = int(os.getenv("HTTP_WORKERS", "16"))
SANITIZE_WORKERS   = int(os.getenv("SANITIZE_WORKERS", "16"))
CROSS_VENUES       = [e.strip() for e in os.getenv("ANALYZE_EXCHANGES", "binance,bybit,bitget,mexc").split(",") if e.strip()]
FETCH_CROSS        = bool(int(os.getenv("FETCH_CROSS_TICKERS","0")))

# CPU workers (env default; can be overridden via run_scan(cpu_workers=...))
CPU_WORKERS_ENV    = int(os.getenv("CPU_WORKERS", str(os.cpu_count() or 2)))

# Output
RUNTIME_LOG        = os.getenv("RUNTIME_LOG", "logs/runtime.log")
SCAN_OUT           = os.getenv("SCAN_OUT", "logs/scan.jsonl")
QUIET_MODE         = bool(int(os.getenv("QUIET_MODE", "1")))
SPINNER            = bool(int(os.getenv("SPINNER", "1")))
COLOR              = bool(int(os.getenv("COLOR", "1")))
RUNTIME_THROTTLE_MS= int(os.getenv("RUNTIME_THROTTLE_MS","800"))

# ===================== Utilities =====================

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

_last_write_ms = 0.0
def write_runtime_progress(done: int, total: int, last_symbol: str, phase: str = "scan"):
    global _last_write_ms
    import time
    now = time.time() * 1000.0
    if now - _last_write_ms < RUNTIME_THROTTLE_MS:
        return
    _last_write_ms = now
    _ensure_dir(RUNTIME_LOG)
    with open(RUNTIME_LOG, "a", encoding="utf-8") as f:
        f.write(f"{_ts_utc()} | {phase} {done}/{total} | {last_symbol}\n")
    if not QUIET_MODE:
        print(f"{_ts_utc()} | {phase} {done}/{total} | {last_symbol}")

def _c(txt: str, code: str) -> str:
    if not COLOR:
        return txt
    return f"\033[{code}m{txt}\033[0m"

_SPIN = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
def spinner_batch(done: int, total: int, last: str, phase: str = "scan"):
    if not SPINNER:
        return
    spin = next(_SPIN)
    line = f"\r\033[2K{_c(spin,'90')} {_c(phase,'36')} {done}/{total} {_c(last,'33')}"
    print(line, end="", flush=True)

# ===================== Async Bitget helpers =====================

async def _bitget_client() -> ccxt_async.Exchange:
    ex = ccxt_async.bitget({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType":"swap","defaultSubType":"linear","defaultSettle":"USDT"},
    })
    try:
        ex.set_sandbox_mode(BITGET_SANDBOX)
    except Exception:
        pass
    try:
        await ex.load_markets()
    except Exception:
        await ex.close()
        raise
    return ex

async def build_universe_bitget(topn: int, min_qvol: float) -> List[str]:
    ex = await _bitget_client()
    try:
        tickers = await ex.fetch_tickers()
        cands: List[Tuple[str, float]] = []
        for sym, m in ex.markets.items():
            if not m.get("contract", False):           continue
            if m.get("settle","").upper() != "USDT":   continue
            if m.get("linear") is False:               continue
            t = tickers.get(sym, {})
            last = t.get("last")
            qvol = t.get("quoteVolume")
            if qvol is None:
                base_vol = t.get("baseVolume")
                if base_vol is not None and last:
                    qvol = base_vol * last
                else:
                    info = t.get("info", {}) or {}
                    qvol = float(info.get("usdtVol") or 0.0)
            if last and qvol and float(qvol) >= min_qvol:
                cands.append((sym, float(qvol)))
        cands.sort(key=lambda x: x[1], reverse=True)
        out = [s for s,_ in cands[:topn]]
        return out
    finally:
        try: await ex.close()
        except Exception: pass

async def fetch_ohlcv_async(ex: ccxt_async.Exchange, symbol: str, timeframe: str, want: int) -> List[List[float]]:
    per = 1000
    ms_per_bar = int(ex.parse_timeframe(timeframe) * 1000)
    end_ms   = ex.milliseconds()
    start_ms = end_ms - (want + 200) * ms_per_bar
    since    = start_ms

    rows: List[List[float]] = []
    last_seen_ts = None
    pages = 0
    max_pages = 2000
    while len(rows) < want and pages < max_pages:
        part = await ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=per)
        if not part:
            break
        if last_seen_ts is not None and part[-1][0] <= last_seen_ts:
            break
        rows += part
        last_seen_ts = part[-1][0]
        since = last_seen_ts + ms_per_bar
        pages += 1
        if since >= end_ms - ms_per_bar:
            break
        await asyncio.sleep((ex.rateLimit or 250)/1000)
    if not rows:
        rows = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(1000, want))
    return rows

async def _probe_one(ex: ccxt_async.Exchange, symbol: str, timeframe: str, sem: asyncio.Semaphore) -> Optional[str]:
    async with sem:
        try:
            await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            return symbol
        except Exception:
            return None

async def sanitize_symbols(symbols: List[str], timeframe: str, workers: int) -> List[str]:
    if not symbols:
        return []
    ex = await _bitget_client()
    sem = asyncio.Semaphore(max(1, workers))
    try:
        tasks = [_probe_one(ex, s, timeframe, sem) for s in symbols]
        keep = await asyncio.gather(*tasks)
        kept = set([k for k in keep if k])
        return [s for s in symbols if s in kept]
    finally:
        try: await ex.close()
        except Exception: pass

# ===================== Cross-venue tickers (optional) =====================

async def _with_ex(ex_id: str) -> Optional[ccxt_async.Exchange]:
    try:
        cls = getattr(ccxt_async, ex_id)
    except AttributeError:
        return None
    opts = {"enableRateLimit": True, "timeout": 15000}
    if ex_id in ("bybit","bitget","mexc"):
        opts["options"] = {"defaultType": "swap"}
    ex = cls(opts)
    try:
        await ex.load_markets()
        return ex
    except Exception:
        try: await ex.close()
        except Exception: pass
        return None

async def cross_venue_last(base_symbol: str, venues: List[str], workers: int = 8) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}
    async def one(ex_id: str):
        ex = await _with_ex(ex_id)
        if not ex:
            results[ex_id] = None
            return
        try:
            sym = base_symbol
            if sym not in ex.symbols and base_symbol.endswith(":USDT"):
                sym = base_symbol.split(":")[0]
            if sym not in ex.symbols:
                results[ex_id] = None
            else:
                t = await ex.fetch_ticker(sym)
                results[ex_id] = float(t["last"]) if t and t.get("last") else None
        except Exception:
            results[ex_id] = None
        finally:
            try: await ex.close()
            except Exception: pass
    await asyncio.gather(*[one(v) for v in venues])
    return results

# ===================== CPU worker =====================

_MODEL = None
_FCOLS = None

def _cpu_features_and_score(symbol: str, rows: List[List[float]], model_path: str) -> Optional[Dict[str, Any]]:
    global _MODEL, _FCOLS
    if not rows:
        return None
    try:
        if _MODEL is None:
            bundle = joblib.load(model_path)
            _MODEL = bundle["model"]
            _FCOLS = bundle["features"]
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        for c in ("open","high","low","close","volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        feats = build_features(df)
        need = [c for c in ("ema_20","ema_50","atr_14","close") if c not in feats.columns]
        if need:
            return None
        if "trend_ok" not in feats.columns:
            feats["trend_ok"] = (feats["ema_20"] > feats["ema_50"]).astype(int)
        if "momentum_ok" not in feats.columns:
            feats["momentum_ok"] = ((feats["close"] > feats["ema_20"]) & (feats["ema_20"] > feats["ema_50"])).astype(int)
        if "vol_ok" not in feats.columns:
            feats["vol_ok"] = (feats["atr_14"] / feats["close"] > 0.002).astype(int)
        last = feats.iloc[-1]
        X = feats[_FCOLS].iloc[[-1]]
        p_raw = float(_MODEL.predict_proba(X)[:,1][0])
        price = float(last["close"])
        atr   = float(last["atr_14"])
        mom_ok   = bool(int(last["momentum_ok"]) == 1)
        trend_ok = bool(int(last["trend_ok"]) == 1)
        vol_ok   = bool(int(last["vol_ok"]) == 1)
        rule_prob = 0.8 if mom_ok else 0.2
        p_ens = 0.7*p_raw + 0.3*rule_prob
        return {
            "symbol": symbol,
            "price": price,
            "atr": atr,
            "p_raw": p_raw,
            "p_ens": p_ens,
            "trend_ok": trend_ok,
            "vol_ok": vol_ok,
            "mom_ok": mom_ok,
            "n_bars": int(len(feats)),
            "ts_last": feats.index[-1].isoformat(),
        }
    except Exception:
        return None

# ===================== Main async pipeline =====================

def _effective_cpu_workers(requested: Optional[int]) -> int:
    if requested is None:
        requested = CPU_WORKERS_ENV
    if requested <= 0:
        requested = os.cpu_count() or 2
    return max(1, int(requested))

async def run_scan(cpu_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Returns list of per-symbol dicts with scores and gates. Also writes JSONL to SCAN_OUT.
    cpu_workers: override number of processes (<=0 → use all cores)
    """
    # 1) Universe
    universe = SYMBOL_WHITELIST[:] if SYMBOL_WHITELIST else await build_universe_bitget(SCAN_TOPN, SCAN_MIN_NOTIONAL)
    total_n = len(universe)

    # 2) Pre-sanitize
    if SANITIZE_WORKERS > 0:
        universe = await sanitize_symbols(universe, TIMEFRAME, SANITIZE_WORKERS)
        total_n = len(universe)

    # 3) Async fetch OHLCV
    bitget = await _bitget_client()
    sem = asyncio.Semaphore(max(1, HTTP_WORKERS))

    async def fetch_one(sym: str) -> Tuple[str, Optional[List[List[float]]]]:
        async with sem:
            try:
                rows = await fetch_ohlcv_async(bitget, sym, TIMEFRAME, LOOKBACK)
                return sym, rows
            except Exception:
                return sym, None

    try:
        tasks = [fetch_one(s) for s in universe]
        done, last = 0, ""
        all_rows: Dict[str, List[List[float]]] = {}
        for fut in asyncio.as_completed(tasks):
            sym, rows = await fut
            last = sym
            done += 1
            spinner_batch(done, total_n, last, phase="fetch")
            write_runtime_progress(done, total_n, last, phase="fetch")
            if rows:
                all_rows[sym] = rows
    finally:
        try: await bitget.close()
        except Exception: pass

    # 4) CPU features + scoring (configurable cores)
    eff_workers = _effective_cpu_workers(cpu_workers)
    _ensure_dir(SCAN_OUT)
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=eff_workers) as pool:
        futs = { pool.submit(_cpu_features_and_score, s, all_rows[s], MODEL_PATH): s for s in all_rows }
        done, last = 0, ""
        for future in as_completed(futs):
            sym = futs[future]
            try:
                res = future.result()
            except Exception:
                res = None
            done += 1
            last = sym
            spinner_batch(done, total_n, last, phase="score")
            write_runtime_progress(done, total_n, last, phase="score")
            if res:
                results.append(res)
                with open(SCAN_OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")

    # 5) Optional cross-venue
    if FETCH_CROSS and results:
        async def fill_cross(item: Dict[str, Any]):
            base_sym = item["symbol"].split(":")[0]
            item["cross"] = await cross_venue_last(base_sym, CROSS_VENUES)
        await asyncio.gather(*[fill_cross(r) for r in results])

    results.sort(key=lambda d: d.get("p_ens", 0.0), reverse=True)
    return results
