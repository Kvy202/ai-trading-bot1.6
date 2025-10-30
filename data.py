# data.py
import os, time
from typing import Optional, List, Tuple
import ccxt
import pandas as pd
import numpy as np

from config import EXCHANGE_ID, TIMEFRAME, LOOKBACK_CANDLES

DERIVS = bool(int(os.getenv("DERIVS", "1")))  # default futures ON

def _quiet_data() -> bool:
    return os.getenv("QUIET_DATA", "0") == "1"

# ---------- exchange helpers ----------
def get_exchange(exchange_id: Optional[str] = None, kind: Optional[str] = None):
    """
    Create a ccxt exchange. kind: 'swap' or 'spot'. If None, infer from DERIVS.
    Enforces USDT-M linear for swaps on Bitget/Bybit/MEXC.
    """
    ex_id = (exchange_id or EXCHANGE_ID).lower()
    cls = getattr(ccxt, ex_id)
    use_kind = (kind or ("swap" if DERIVS else "spot"))

    options = {"defaultType": use_kind}
    if ex_id in ("bitget", "bybit", "mexc") and use_kind == "swap":
        options.update({"defaultSubType": "linear", "defaultSettle": "USDT"})

    ex = cls({
        "enableRateLimit": True,
        "timeout": int(os.getenv("CCXT_TIMEOUT_MS", "20000")),
        "options": options,
    })
    ex.verbose = bool(int(os.getenv("CCXT_DEBUG", "0")))
    return ex

def _ensure_markets(ex: ccxt.Exchange, kind: str):
    """Load markets with correct params for the venue/kind."""
    try:
        params = {}
        if ex.id == "bitget":
            params = {"productType": "USDT-FUTURES"} if kind == "swap" else {"type": "spot"}
        elif ex.id == "mexc" and kind == "spot":
            params = {"type": "spot"}
        elif ex.id == "bybit":
            # bybit is okay without special params for v5 OHLCV
            pass
        ex.load_markets(False, params)
    except Exception:
        ex.load_markets()

# ---------- OHLCV ----------
def fetch_ohlcv(symbol: str,
                timeframe: Optional[str] = None,
                limit: Optional[int] = None) -> pd.DataFrame:
    """
    Paginated OHLCV. Uses SWAP for perps (USDT-M).
    Falls back to a single limit-only fetch if pagination returns nothing.
    """
    tf = (timeframe or TIMEFRAME).lower()
    want = int(limit or LOOKBACK_CANDLES or 1000)

    looks_contract = (":USDT" in symbol) or ("-SWAP" in symbol)
    kind = "swap" if (DERIVS or looks_contract) else "spot"

    ex = get_exchange(kind=kind)
    _ensure_markets(ex, kind)

    per = 1000 if not (ex.id == "mexc" and kind == "spot") else 500

    ms_per_bar = int(ex.parse_timeframe(tf) * 1000)
    end_ms = ex.milliseconds()
    start_ms = end_ms - (want + 200) * ms_per_bar
    since = start_ms

    params = {}
    if ex.id == "bitget":
        params = {"productType": "USDT-FUTURES"} if kind == "swap" else {"type": "spot"}
    elif ex.id == "mexc" and kind == "spot":
        params = {"type": "spot"}

    rows, retries, pages = [], 0, 0
    max_pages = 2000
    last_seen_ts = None

    while len(rows) < want and pages < max_pages:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=per, params=params)
        except Exception as e:
            retries += 1
            if retries > 3:
                raise
            time.sleep(min(2 ** retries, 8))
            continue

        if not ohlcv:
            break
        if last_seen_ts is not None and ohlcv[-1][0] <= last_seen_ts:
            break

        rows += ohlcv
        pages += 1
        last_seen_ts = ohlcv[-1][0]
        since = last_seen_ts + ms_per_bar
        if since >= end_ms - ms_per_bar:
            break

        time.sleep((ex.rateLimit or 250) / 1000)

    # Fallback: latest N
    if not rows:
        try:
            fallback = ex.fetch_ohlcv(symbol, timeframe=tf, limit=min(1000, want), params=params)
            if fallback:
                rows = fallback
        except Exception:
            pass

    if not rows:
        raise RuntimeError(f"No OHLCV returned for {symbol} {tf} on {ex.id} ({kind})")

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df.tail(want)

# ---------- DL helper: features + prices ----------
def _build_one_symbol(sym: str, tf: str, lb: int,
                      feature_cols: Optional[List[str]]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    from features import make_dataset
    try:
        bars = fetch_ohlcv(sym, tf, lb)
    except Exception as e:
        if not _quiet_data():
            print(f"[load_prices_and_features] skipping {sym}: {e}")
        return None

    X_df, _y_unused, _feats = make_dataset(bars)
    if X_df is None or len(X_df) == 0:
        if not _quiet_data():
            print(f"[load_prices_and_features] {sym}: empty feature set")
        return None

    # optionally align to a requested feature order
    if feature_cols:
        missing = [c for c in feature_cols if c not in X_df.columns]
        if missing:
            if not _quiet_data():
                print(f"[load_prices_and_features] {sym}: missing features {missing[:6]}{'...' if len(missing)>6 else ''}")
            keep = [c for c in feature_cols if c in X_df.columns]
            if not keep:
                return None
            X_df = X_df[keep]
        else:
            X_df = X_df[feature_cols]

    prices = bars.loc[X_df.index, "close"].astype("float32")
    return X_df, prices

def load_prices_and_features(symbols: Optional[List[str]] = None,
                             timeframe: Optional[str] = None,
                             lookback: Optional[int] = None,
                             feature_cols: Optional[List[str]] = None,
                             add_symbol_id: bool = False,
                             return_dfs: bool = False):
    """
    Build a training matrix across one or many symbols.
    """
    tf = (timeframe or TIMEFRAME)
    lb = int(lookback or LOOKBACK_CANDLES or 1000)

    if not symbols:
        wl = os.getenv("SYMBOL_WHITELIST", "").strip()
        if wl:
            symbols = [s.strip() for s in wl.split(",") if s.strip()]
        else:
            sym_env = os.getenv("SYMBOL", "BTC/USDT:USDT")
            symbols = [sym_env]

    built: List[Tuple[pd.DataFrame, pd.Series, str]] = []
    for si, sym in enumerate(symbols):
        one = _build_one_symbol(sym, tf, lb, feature_cols)
        if not one:
            continue
        X_df, prices = one
        if add_symbol_id:
            X_df = X_df.copy()
            X_df["symbol_id"] = float(si)
        built.append((X_df, prices, sym))

    if not built:
        raise RuntimeError("No features could be built for any symbol.")

    # align columns across symbols
    cols = set(built[0][0].columns)
    for X_df, _, _ in built[1:]:
        cols &= set(X_df.columns)
    cols = list(sorted(cols))
    if not cols:
        raise RuntimeError("No common feature columns across symbols.")

    X_all, p_all = [], []
    for X_df, prices, sym in built:
        Xc = X_df[cols].astype("float32")
        pc = prices.astype("float32")
        idx = Xc.index.intersection(pc.index)
        Xc = Xc.loc[idx]
        pc = pc.loc[idx]
        X_all.append(Xc)
        p_all.append(pc)

    X_cat = pd.concat(X_all, axis=0).sort_index()
    p_cat = pd.concat(p_all, axis=0).loc[X_cat.index]

    if return_dfs:
        return X_cat, p_cat

    X_np = X_cat.values.astype(np.float32, copy=False)
    p_np = p_cat.values.astype(np.float32, copy=False)
    return X_np, p_np
