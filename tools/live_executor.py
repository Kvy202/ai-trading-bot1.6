"""
Refined live signal executor for the AI trading bot.

This script acts as a paper‐trading execution engine that listens for live
signals produced by the writer process.  It applies configurable gate
logic (fixed or adaptive thresholds, risk limits, side restrictions) to
decide when to enter, scale in, flip or exit positions.  When enabled,
take‑profit (TP) and stop‑loss (SL) guards automatically close positions
when price moves outside the configured bounds.  All operations are
recorded to a daily paper trades CSV and an optional closed trades CSV.

Key improvements over the previous version include:

* Better parsing of the signals file: extra columns like ``kinds_used``
  and ``side_hint`` are ignored and the core fields are extracted
  robustly.  The executor no longer assumes a fixed number of columns.
* Honour the ``allow`` field from the writer – if the writer marks a
  signal as disallowed (``allow`` = 0) the executor will now skip that
  symbol entirely instead of treating it as advisory.
* Cross‑platform single instance lock: the executor no longer relies on
  Windows‐specific ``tasklist``.  Instead, it writes its PID and
  timestamp to a lock file and removes stale locks based on age.
* Clearer log messages during idle periods to reassure the user that
  the executor is still running when no new signals arrive.

This file is meant to reside in the ``tools`` directory of the trading
project.  It uses only the Python standard library so it can run on
Windows, Linux and macOS without modification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parents[1]
LOGS_DIR: Path = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OUT_LOG: Path = LOGS_DIR / "live_executor.out"
ERR_LOG: Path = LOGS_DIR / "live_executor.err"
LOCK_PATH: Path = LOGS_DIR / "live_executor.lock"
STATE_JSON: Path = LOGS_DIR / "executor_state.json"

# Closed trades logs (master and per‑day) reside under LOGS_DIR
CLOSED_MASTER_CSV: Path = LOGS_DIR / "trades_closed.csv"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return the current UTC timestamp in ISO format with timezone."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def _write(fp: Path, line: str) -> None:
    """Append a line to a file, ensuring its directory exists."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(line)


def log(msg: str) -> None:
    _write(OUT_LOG, f"[{_ts()}] {msg}\n")


def log_err(msg: str) -> None:
    _write(ERR_LOG, f"[{_ts()}] {msg}\n")


# ---------------------------------------------------------------------------
# File helpers for paper trades and closed trades
# ---------------------------------------------------------------------------

def paper_path_for_day(d: date) -> Path:
    return LOGS_DIR / f"trades_paper_{d:%Y%m%d}.csv"


def closed_path_for_day(d: date) -> Path:
    return LOGS_DIR / f"trades_closed_{d:%Y%m%d}.csv"


def ensure_header(path: Path, header: Iterable[str]) -> None:
    """Ensure a CSV file exists with the given header if missing."""
    if not path.exists():
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(list(header))


def record_paper_trade(path: Path, row: List[Any]) -> None:
    """Append a row to the daily paper trades CSV."""
    ensure_header(path, ["ts", "symbol", "side", "price", "qty", "reason"])
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def record_closed_trade(ts: str, symbol: str, closed_side: str, qty: float,
                        entry_avg: float, exit_price: float, realized_pnl: float,
                        reason: str) -> None:
    """Append a row to both the master closed trades CSV and today's closed CSV."""
    master_cols = ["ts", "symbol", "closed_side", "qty", "entry_avg",
                   "exit_price", "realized_pnl", "reason"]
    ensure_header(CLOSED_MASTER_CSV, master_cols)
    with open(CLOSED_MASTER_CSV, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([ts, symbol, closed_side, qty, entry_avg,
                                exit_price, realized_pnl, reason])
    today_path = closed_path_for_day(datetime.now(timezone.utc).date())
    ensure_header(today_path, master_cols)
    with open(today_path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([ts, symbol, closed_side, qty, entry_avg,
                                exit_price, realized_pnl, reason])


# ---------------------------------------------------------------------------
# Locking to enforce a single executor instance
# ---------------------------------------------------------------------------

def single_instance_lock(stale_sec: int = 600) -> None:
    """Acquire a simple PID lock; exit if another alive executor holds the lock.

    The lock file stores ``pid,timestamp``.  If the process is still
    running and the timestamp is fresh, this instance will exit.  If the
    lock appears stale (no process exists or age > stale_sec) it will be
    replaced.
    """
    if LOCK_PATH.exists():
        try:
            pid_str, when_str = LOCK_PATH.read_text(encoding="utf-8").strip().split(",", 1)
            pid = int(pid_str)
            started = float(when_str)
            # If this PID is us, we simply keep running
            if pid == os.getpid():
                log("lock: self PID detected; continuing")
            else:
                # On Unix we can check /proc, on Windows we simply try to
                # send signal 0.  Failure indicates a dead process.
                alive = True
                try:
                    if hasattr(os, "kill"):
                        # On Unix: kill(pid,0) raises OSError if dead
                        os.kill(pid, 0)
                    else:
                        # On Windows: os.kill still works for signal 0 in Python 3.9+
                        os.kill(pid, 0)
                except OSError:
                    alive = False
                fresh = (time.time() - started) < stale_sec
                if alive and fresh:
                    log_err(f"lock: another live_executor running (PID={pid}); exiting.")
                    sys.exit(0)
                else:
                    log("lock: stale or dead lock found; replacing.")
        except Exception:
            log("lock: unparsable lock; replacing.")
    # Write our PID and timestamp
    try:
        LOCK_PATH.write_text(f"{os.getpid()},{time.time()}", encoding="utf-8")
    except Exception as e:
        log_err(f"cannot create lock file: {e}")


def unlock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Signal parsing helpers
# ---------------------------------------------------------------------------

def parse_signal_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a CSV line from the live signals file into a dictionary.

    The writer emits rows with at least the following columns:
    ``ts, symbol, px, p_meta, rv_mean, allow, thr, mode`` and may append
    extra fields such as ``kinds_used`` or ``side_hint``.  Only the
    first eight columns are required; extra columns are ignored.  If the
    row is malformed or cannot be parsed, ``None`` is returned.
    """
    try:
        parts = next(csv.reader([line]))
    except Exception:
        return None
    if len(parts) < 8:
        return None
    try:
        ts, symbol, px, p_meta, rv_mean, allow, thr, mode = parts[:8]
        return {
            "ts": ts,
            "symbol": symbol,
            "price": float(px) if px else 0.0,
            "p_meta": float(p_meta),
            "rv_mean": float(rv_mean),
            # Allow is an int (0 or 1); treat any non‑1 as 0
            "allow": 1 if str(allow).strip() in ("1", "True", "true", "YES", "yes") else 0,
            "thr": float(thr),
            "mode": (mode or "abs").lower(),
        }
    except Exception:
        return None


def mode_value(p_meta: float, mode: str) -> float:
    return abs(p_meta) if (mode or "abs") == "abs" else p_meta


def threshold_pass(p_meta: float, thr_sig: float, mode: str,
                   allow: int, thr_exec: float, respect_writer_thr: bool) -> Tuple[bool, str]:
    """Determine whether a signal passes the combined threshold gate.

    Args:
        p_meta: The model's p_long or p_short meta value for this symbol.
        thr_sig: The writer's threshold.
        mode: "abs" or "raw" to select absolute or signed p_meta.
        allow: 1 if the writer allows trading, 0 if advisory or disallowed.
        thr_exec: The executor's threshold (possibly adaptive).
        respect_writer_thr: If true, use max(thr_sig, thr_exec) as the
            effective threshold; otherwise use thr_exec alone.

    Returns:
        A tuple (ok, reason).  ok is True if the signal passes the gate.
        reason contains a human‑readable reason for skipping when ok is False.
    """
    val = mode_value(p_meta, mode)
    thr_eff = max(thr_sig, thr_exec) if respect_writer_thr else thr_exec
    if allow != 1:
        return False, "allow=0 (writer gate)"
    if val < thr_eff:
        return False, f"below_thr({val:.4f}<{thr_eff:.4f}) writer_thr={thr_sig:.4f} exec_thr={thr_exec:.4f}"
    return True, ""


def qty_for(price: float, risk_usd: float, min_notional: float, min_qty: float) -> float:
    """Compute quantity from risk budget.

    Ensures the resulting notional (price * qty) meets ``min_notional`` and
    quantity is at least ``min_qty``.  Returns 0.0 if quantity is too
    small or price is invalid.
    """
    if price <= 0:
        return 0.0
    qty = risk_usd / price
    if qty < min_qty:
        return 0.0
    if price * qty < min_notional:
        return 0.0
    return round(qty, 6)


def side_allowed_by_cfg(p_meta: float, sides: str) -> bool:
    sides = (sides or "both").lower()
    if sides == "both":
        return True
    if sides == "long_only":
        return p_meta >= 0
    if sides == "short_only":
        return p_meta < 0
    return True


def symbol_allowed(symbol: str, whitelist: Optional[List[str]]) -> bool:
    return (not whitelist) or (symbol in whitelist)


def place_order(symbol: str, side: str, price: float, qty: float) -> bool:
    """Stub for broker integration.  Always returns True for paper trades."""
    return True


def read_recent_signals(path: Path, last_ts_map: Dict[str, str], window: int = 50) -> List[Dict[str, Any]]:
    """Return a list of the most recent unseen signal per symbol.

    ``last_ts_map`` stores the timestamp of the last processed signal for
    each symbol.  Only signals with a newer ``ts`` are returned.  Up to
    ``window`` rows from the tail of the file are considered.  The
    resulting list is sorted chronologically.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return []
    if not lines:
        return []
    newest: Dict[str, Dict[str, Any]] = {}
    for raw in reversed(lines[-window:]):  # newest to older
        sig = parse_signal_line(raw)
        if not sig:
            continue
        sym = sig["symbol"]
        ts = sig["ts"]
        # Skip if we've already processed this ts for the symbol
        if last_ts_map.get(sym) == ts:
            continue
        # Keep the most recent unseen signal per symbol
        if sym not in newest:
            newest[sym] = sig
    out = list(newest.values())
    out.sort(key=lambda s: s["ts"])  # chronological
    return out


def _tail_lines(path: Path, n: int) -> List[str]:
    """Return the last n lines from a file (at most n, ignoring blank lines)."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        size = 8192
        data = b""
        while end > 0 and data.count(b"\n") <= n + 1:
            take = min(size, end)
            end -= take
            f.seek(end)
            data = f.read(take) + data
        lines = [ln.decode("utf-8", errors="ignore").strip() for ln in data.splitlines() if ln.strip()]
    return lines[-n:]


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round(q * (len(vals) - 1)))
    return vals[max(0, min(idx, len(vals) - 1))]


def compute_adaptive_thr(signals_path: Path, window: int, pmode: str,
                         target_pass: float, thr_min: float, thr_max: float,
                         ema_prev: Optional[float], alpha: float) -> Optional[float]:
    """Compute a new adaptive threshold based on recent p_meta values.

    Uses the (1 - target_pass) percentile of recent |p_meta| or raw p_meta
    values.  An exponential moving average (EMA) with parameter ``alpha``
    smooths the threshold.  The result is clamped between ``thr_min`` and
    ``thr_max``.  Returns ``None`` if fewer than 12 samples are available.
    """
    lines = _tail_lines(signals_path, window)
    if not lines:
        return None
    vals: List[float] = []
    pmode = (pmode or "abs").lower()
    for ln in lines:
        sig = parse_signal_line(ln)
        if not sig:
            continue
        v = abs(sig["p_meta"]) if pmode == "abs" else sig["p_meta"]
        vals.append(v)
    if len(vals) < 12:
        return None
    q = max(0.0, min(1.0, 1.0 - float(target_pass)))
    thr = _percentile(vals, q)
    if ema_prev is not None:
        thr = float(alpha) * thr + (1.0 - float(alpha)) * float(ema_prev)
    thr = max(float(thr_min), min(float(thr), float(thr_max)))
    return thr


# ---------------------------------------------------------------------------
# Position bookkeeping (paper trades)
# ---------------------------------------------------------------------------

class Pos:
    __slots__ = ("side", "qty", "avg")

    def __init__(self, side: str, qty: float, avg: float) -> None:
        self.side = side  # "long" or "short"
        self.qty = qty
        self.avg = avg


def pnl_on_close(pos: Pos, price: float) -> float:
    """Compute realized PnL when closing a position at a given price."""
    if pos.side == "long":
        return (price - pos.avg) * pos.qty
    else:
        return (pos.avg - price) * pos.qty


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Live signal executor (paper with flip/TP/SL + closed CSV)"
    )
    parser.add_argument("--signals", default=str(LOGS_DIR / "live_signals.csv"),
                        help="Path to live_signals.csv")
    parser.add_argument("--plong", type=float, default=float(os.getenv("DL_P_LONG", "0.45")),
                        help="Fixed threshold if not adaptive")
    parser.add_argument("--pmode", type=str, default=(os.getenv("DL_P_LONG_MODE", "abs") or "abs"),
                        choices=["abs", "raw"], help="p_meta mode")
    parser.add_argument("--rv-max", type=float, default=float(os.getenv("EXEC_RV_MAX", "60")))
    parser.add_argument("--poll", type=float, default=float(os.getenv("EXEC_POLL_SEC", "3")))
    parser.add_argument("--cooldown", type=float, default=float(os.getenv("EXEC_COOLDOWN_SEC", "30")))
    parser.add_argument("--one-position", action="store_true",
                        default=os.getenv("EXEC_ONE_POSITION", "0") == "1",
                        help="Allow at most one open symbol at a time")
    parser.add_argument("--sides", default=(os.getenv("EXEC_SIDES", "both") or "both"),
                        choices=["both", "long_only", "short_only"])
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("RISK_MAX_SYMBOLS", "5")),
                        help="Maximum number of concurrent active symbols")
    parser.add_argument("--risk-usd", type=float, default=float(os.getenv("RISK_MAX_POS_USD", "100")))
    parser.add_argument("--tail", type=int, default=50, help="Signals tail window for new signals")
    parser.add_argument("--tp-pct", type=float, default=float(os.getenv("EXEC_TP_PCT", "0.01")),
                        help="Take‑profit fraction (0.01 = 1%)")
    parser.add_argument("--sl-pct", type=float, default=float(os.getenv("EXEC_SL_PCT", "0.02")),
                        help="Stop‑loss fraction (0.02 = 2%)")
    parser.add_argument("--flip-open", action="store_true", default=os.getenv("EXEC_FLIP_OPEN", "1") == "1",
                        help="Open immediately after flip close")
    parser.add_argument("--scale-in", action="store_true", default=os.getenv("EXEC_SCALE_IN", "0") == "1",
                        help="Allow adding to same‑side position")
    parser.add_argument("--respect-writer-thr", action="store_true",
                        help="Use max(writer_thr, exec_thr) for entry")
    # Adaptive gate options
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive exec threshold")
    parser.add_argument("--target-pass", type=float, default=float(os.getenv("EXEC_TARGET_PASS", "0.20")))
    parser.add_argument("--window-signals", type=int, default=int(os.getenv("EXEC_WINDOW_SIGNALS", "180")))
    parser.add_argument("--thr-min", type=float, default=float(os.getenv("EXEC_THR_MIN", "0.40")))
    parser.add_argument("--thr-max", type=float, default=float(os.getenv("EXEC_THR_MAX", "0.60")))
    parser.add_argument("--thr-alpha", type=float, default=float(os.getenv("EXEC_THR_EMA_ALPHA", "0.2")))

    args = parser.parse_args(argv)

    # Resolve signals path relative to project root
    signals_path = Path(args.signals)
    if not signals_path.is_absolute():
        signals_path = (BASE_DIR / signals_path).resolve()

    single_instance_lock()

    try:
        current_day = datetime.now(timezone.utc).date()
        paper_path = paper_path_for_day(current_day)
        ensure_header(paper_path, ["ts", "symbol", "side", "price", "qty", "reason"])
        ensure_header(CLOSED_MASTER_CSV,
                      ["ts", "symbol", "closed_side", "qty", "entry_avg",
                       "exit_price", "realized_pnl", "reason"])
        ensure_header(closed_path_for_day(current_day),
                      ["ts", "symbol", "closed_side", "qty", "entry_avg",
                       "exit_price", "realized_pnl", "reason"])

        # Initial threshold and state
        exec_thr_cur: float = args.plong
        state = {
            "ts": _ts(),
            "exec_thr": exec_thr_cur,
            "exec_mode": args.pmode,
            "adaptive": bool(args.adaptive),
            "target_pass": args.target_pass,
            "window": args.window_signals,
        }
        try:
            STATE_JSON.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        log(f"GATE exec_thr={exec_thr_cur:.4f} mode={args.pmode} adaptive={args.adaptive} respect_writer_thr={args.respect_writer_thr}")

        last_signal_ts: Dict[str, str] = {}
        last_fill_time: Dict[str, float] = {}
        last_fill_price: Dict[str, float] = {}
        active_symbols: set[str] = set()
        positions: Dict[str, Pos] = {}

        idle_ticks_no_file = 0
        idle_ticks_no_new = 0

        whitelist = [s.strip() for s in os.getenv("EXEC_SYMBOL_WHITELIST", "").split(",") if s.strip()]

        while True:
            # Daily rollover for paper and closed logs
            now_utc = datetime.now(timezone.utc)
            if now_utc.date() != current_day:
                current_day = now_utc.date()
                paper_path = paper_path_for_day(current_day)
                ensure_header(paper_path, ["ts", "symbol", "side", "price", "qty", "reason"])
                ensure_header(closed_path_for_day(current_day),
                              ["ts", "symbol", "closed_side", "qty", "entry_avg",
                               "exit_price", "realized_pnl", "reason"])

            # Adaptive threshold update
            if args.adaptive and signals_path.exists():
                thr_new = compute_adaptive_thr(
                    signals_path, args.window_signals, args.pmode,
                    args.target_pass, args.thr_min, args.thr_max,
                    ema_prev=exec_thr_cur, alpha=args.thr_alpha
                )
                if thr_new is not None and abs(thr_new - exec_thr_cur) > 1e-6:
                    exec_thr_cur = float(thr_new)
                    state.update({"ts": _ts(), "exec_thr": exec_thr_cur})
                    try:
                        STATE_JSON.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass
                    log(f"ADAPT_THR updated exec_thr={exec_thr_cur:.6f} mode={args.pmode}")

            # Wait for signals file to appear
            if not signals_path.exists():
                idle_ticks_no_file += 1
                mod1 = max(1, int(round(10.0 / max(1.0, args.poll))))
                if idle_ticks_no_file % mod1 == 0:
                    log("waiting: signals.csv not found yet…")
                time.sleep(args.poll)
                continue

            # Read new signals (most recent unseen per symbol)
            new_sigs = read_recent_signals(signals_path, last_signal_ts, window=args.tail)
            if not new_sigs:
                idle_ticks_no_new += 1
                mod2 = max(1, int(round(20.0 / max(1.0, args.poll))))
                if idle_ticks_no_new % mod2 == 0:
                    log("idle: no new signals")
                time.sleep(args.poll)
                continue

            # Reset idlers when work arrives
            idle_ticks_no_file = 0
            idle_ticks_no_new = 0

            for sig in new_sigs:
                symbol = sig["symbol"]
                ts_sig = sig["ts"]
                p_meta = sig["p_meta"]
                rv_mean = sig["rv_mean"]
                allow = sig["allow"]
                mode = (sig.get("mode") or args.pmode).lower()
                px_sig = sig["price"]
                thr_sig = sig["thr"]

                last_signal_ts[symbol] = ts_sig

                # Whitelist filter
                if not symbol_allowed(symbol, whitelist):
                    log(f"SKIP {symbol} reason=not_whitelisted")
                    continue

                # Derive price: use px_sig if positive, else rv_mean
                price = px_sig if px_sig > 0 else rv_mean
                # Validate position record
                pos = positions.get(symbol)

                # 1) Check TP/SL on existing position
                if pos and price > 0:
                    hit_tp = hit_sl = False
                    if pos.side == "long":
                        if args.tp_pct > 0 and price >= pos.avg * (1 + args.tp_pct):
                            hit_tp = True
                        if args.sl_pct > 0 and price <= pos.avg * (1 - args.sl_pct):
                            hit_sl = True
                    else:
                        if args.tp_pct > 0 and price <= pos.avg * (1 - args.tp_pct):
                            hit_tp = True
                        if args.sl_pct > 0 and price >= pos.avg * (1 + args.sl_pct):
                            hit_sl = True
                    if hit_tp or hit_sl:
                        side_close = "SELL" if pos.side == "long" else "BUY_TO_COVER"
                        realized = pnl_on_close(pos, price)
                        reason = f"EXIT_{'TP' if hit_tp else 'SL'} p={p_meta:.4f} rv={rv_mean:.2f} pnl={realized:.6f}"
                        record_paper_trade(paper_path, [ts_sig, symbol, side_close, price, pos.qty, reason])
                        record_closed_trade(ts_sig, symbol, side_close, pos.qty, pos.avg, price, realized, reason)
                        log(f"TRADE {side_close} {symbol} {pos.qty}@{price:.8f} ({reason})")
                        last_fill_time[symbol] = time.time()
                        last_fill_price[symbol] = price
                        positions.pop(symbol, None)
                        active_symbols.discard(symbol)
                        continue

                # 2) Gate entry / flip logic
                # Side restriction (long_only/short_only)
                if not side_allowed_by_cfg(p_meta, args.sides):
                    log(f"SKIP {symbol} reason=side_blocked p={p_meta:.4f} cfg={args.sides}")
                    continue
                # Determine effective threshold and test
                exec_thr_use = exec_thr_cur if args.adaptive else args.plong
                ok_thr, why_thr = threshold_pass(p_meta, thr_sig, mode, allow,
                                                 exec_thr_use, args.respect_writer_thr)
                if not ok_thr:
                    log(f"SKIP {symbol} reason={why_thr} p={p_meta:.4f}")
                    continue
                # Check volatility guard
                if rv_mean > args.rv_max:
                    log(f"SKIP {symbol} reason=rv>{args.rv_max:.2f} rv={rv_mean:.2f}")
                    continue

                want = "long" if p_meta >= 0 else "short"
                now_s = time.time()
                last_t = last_fill_time.get(symbol, 0.0)
                cooldown_ok = (now_s - last_t) >= args.cooldown

                # Global one‑position guard
                if args.one_position and symbol not in active_symbols and len(active_symbols) >= 1:
                    log(f"SKIP {symbol} reason=one_position_active({list(active_symbols)})")
                    continue
                # Max symbols guard: prevent opening new symbol if limit reached
                if len(active_symbols) >= args.max_symbols and symbol not in active_symbols:
                    log(f"SKIP {symbol} reason=max_symbols({args.max_symbols})")
                    continue

                # Refresh pos pointer after any modifications
                pos = positions.get(symbol)

                # Same side: optional scale‑in
                if pos and pos.side == want:
                    if args.scale_in and cooldown_ok:
                        qty = qty_for(price, args.risk_usd, min_notional=float(os.getenv("EXEC_MIN_NOTIONAL", "5")),
                                       min_qty=float(os.getenv("EXEC_MIN_QTY", "0")))
                        if qty > 0 and place_order(symbol, "BUY" if want == "long" else "SELL_SHORT", price, qty):
                            new_qty = pos.qty + qty
                            pos.avg = (pos.avg * pos.qty + price * qty) / new_qty
                            pos.qty = new_qty
                            side_txt = "BUY" if want == "long" else "SELL_SHORT"
                            reason = f"SCALE_IN p={p_meta:.4f} rv={rv_mean:.2f}"
                            record_paper_trade(paper_path, [ts_sig, symbol, side_txt, price, qty, reason])
                            log(f"TRADE {side_txt} {symbol} {qty}@{price:.8f} ({reason})")
                            last_fill_time[symbol] = now_s
                            last_fill_price[symbol] = price
                        else:
                            log(f"SKIP {symbol} reason=no_qty_scale_in")
                    else:
                        log(f"SKIP {symbol} reason=already_{pos.side}")
                    continue

                # Opposite side: flip close (and optionally open new)
                if pos and pos.side != want:
                    side_close = "SELL" if pos.side == "long" else "BUY_TO_COVER"
                    realized = pnl_on_close(pos, price)
                    reason_close = f"FLIP_CLOSE p={p_meta:.4f} rv={rv_mean:.2f} pnl={realized:.6f}"
                    record_paper_trade(paper_path, [ts_sig, symbol, side_close, price, pos.qty, reason_close])
                    record_closed_trade(ts_sig, symbol, side_close, pos.qty, pos.avg, price, realized, reason_close)
                    log(f"TRADE {side_close} {symbol} {pos.qty}@{price:.8f} ({reason_close})")
                    last_fill_time[symbol] = now_s
                    last_fill_price[symbol] = price
                    positions.pop(symbol, None)
                    active_symbols.discard(symbol)
                    if not args.flip_open:
                        continue  # wait until next signal to open
                    # fall through to open new position without cooldown

                # Fresh entry (or flip‑open continuation)
                if symbol not in positions and not cooldown_ok:
                    log(f"SKIP {symbol} reason=cooldown({args.cooldown:.0f}s)")
                    continue
                side_entry = "BUY" if want == "long" else "SELL_SHORT"
                qty = qty_for(price, args.risk_usd, min_notional=float(os.getenv("EXEC_MIN_NOTIONAL", "5")),
                               min_qty=float(os.getenv("EXEC_MIN_QTY", "0")))
                if qty <= 0:
                    log(f"SKIP {symbol} reason=no_qty price={price:.8f}")
                    continue
                if last_fill_price.get(symbol) == price and (now_s - last_t) < (args.cooldown * 2):
                    log(f"SKIP {symbol} reason=dup_price price={price:.8f}")
                    continue
                if not place_order(symbol, side_entry, price, qty):
                    log_err(f"ERROR order_rejected {symbol} {side_entry} {qty}@{price:.8f}")
                    continue
                reason = f"ENTRY p={p_meta:.4f} rv={rv_mean:.2f} eff_thr={(exec_thr_cur if args.adaptive else args.plong):.4f}"
                record_paper_trade(paper_path, [ts_sig, symbol, side_entry, price, qty, reason])
                log(f"TRADE {side_entry} {symbol} {qty}@{price:.8f} ({reason})")
                positions[symbol] = Pos("long" if side_entry == "BUY" else "short", qty, price)
                last_fill_time[symbol] = now_s
                last_fill_price[symbol] = price
                active_symbols.add(symbol)

            time.sleep(args.poll)

    except KeyboardInterrupt:
        log("executor stopped (ctrl-c)")
    except Exception as e:
        log_err(f"FATAL: {e}")
    finally:
        unlock()
        log("executor stopped")


if __name__ == "__main__":
    main()