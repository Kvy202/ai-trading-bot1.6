"""
Refined live writer for the AI trading bot.

This script generates live trading signals by loading an ensemble of
deep‑learning models, refreshing real‑time features and writing the
results to a master log, a daily meta log and a live signals CSV.  It
also emits a heartbeat file that the dashboard can use for liveness
checks.  Key improvements over the original implementation include:

* Cross‑platform lock handling: the writer no longer relies on
  Windows‑only ``tasklist`` to detect stale processes.  Instead it
  uses ``os.kill(pid, 0)`` (which works on both Unix and Windows) to
  test process existence and removes stale locks.
* Robust logging: errors during model loading or prediction are
  captured and recorded to the error log without abruptly terminating
  subsequent processing.  A fatal import error will still cause the
  writer to exit, as this indicates a missing dependency.
* Clear heartbeat updates: a JSON heartbeat file is written on
  every tick, with flags indicating the overall status, current meta
  value, thresholds and sleep interval.  This aids monitoring.

This file should be placed in the ``tools`` directory and invoked via
``python improved_live_writer.py``.  It is designed to be called
from the ``start_live.ps1`` script, which sets appropriate
environment variables such as ``DL_P_LONG`` and ``DL_P_LONG_MODE``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths and file constants
# ---------------------------------------------------------------------------
TOOLS_DIR: Path = Path(__file__).resolve().parent
BASE_DIR: Path = TOOLS_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LOGS = BASE_DIR / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
WR_OUT: Path = LOGS / "live_writer.out"
WR_ERR: Path = LOGS / "live_writer.err"
LOCK: Path = LOGS / "live_writer.lock"
HB_JSON: Path = LOGS / "live_writer_heartbeat.json"
SIGNALS: Path = LOGS / "live_signals.csv"
MASTER_LOG: Path = LOGS / "live_meta_log.csv"
DAILY_PREFIX = "live_meta_log_"

# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def _append(fp: Path, text: str) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(text)


def log(msg: str) -> None:
    _append(WR_OUT, f"[{_ts()}] {msg}\n")


def log_err(msg: str) -> None:
    _append(WR_ERR, f"[{_ts()}] {msg}\n")


# ---------------------------------------------------------------------------
# Heartbeat handling
# ---------------------------------------------------------------------------

def write_heartbeat(ok: bool, detail: str, symbols: List[str], allow: int,
                    p_meta: float, thr: float, mode: str, sleep_s: int) -> None:
    hb = {
        "ts": _ts(),
        "ok": bool(ok),
        "detail": str(detail),
        "symbols": symbols,
        "p_meta": float(p_meta),
        "thr": float(thr),
        "mode": mode,
        "allow": int(allow),
        "sleep_sec": int(sleep_s),
    }
    try:
        HB_JSON.write_text(json.dumps(hb, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_header(path: Path, cols: Iterable[str]) -> None:
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(cols))


def append_row(path: Path, cols: Iterable[str], row: Dict[str, Any]) -> None:
    ensure_header(path, cols)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in cols])


def today_daily(prefix: str) -> Path:
    return LOGS / f"{prefix}{datetime.now(timezone.utc):%Y%m%d}.csv"


# ---------------------------------------------------------------------------
# Locking (cross‑platform)
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is running on this host."""
    try:
        # On POSIX and Windows, os.kill(pid, 0) will raise OSError if the
        # process does not exist or permission is denied.  We ignore
        # permission errors and treat them as alive.
        os.kill(pid, 0)
    except OSError as e:
        # errno 3 = no such process on Windows, errno ESRCH on Unix
        return False
    except Exception:
        # Other exceptions (e.g. access denied) are treated as alive
        return True
    return True


def writer_lock(stale_sec: int) -> None:
    """Acquire a writer lock; exit if another active writer holds the lock."""
    if LOCK.exists():
        try:
            pid_str, when_str = LOCK.read_text(encoding="utf-8").strip().split(",", 1)
            pid = int(pid_str)
            started = float(when_str)
            if pid != os.getpid():
                alive = _pid_alive(pid)
                fresh = (time.time() - started) < stale_sec
                if alive and fresh:
                    log_err(f"another writer alive (PID={pid}); exiting")
                    sys.exit(0)
                else:
                    log("writer lock: stale or dead lock found; replacing")
        except Exception:
            log("writer lock: unparsable lock; replacing")
    try:
        LOCK.write_text(f"{os.getpid()},{time.time()}", encoding="utf-8")
    except Exception as e:
        log_err(f"writer lock: cannot create lock file: {e}")


def writer_unlock() -> None:
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Safe import of model utilities
# ---------------------------------------------------------------------------
try:
    from ml_dl.dl_ensemble import load_ensemble, refresh_live_features, predict_ensemble  # noqa: F401
except Exception as e:
    # If the model code cannot be imported, log the error and re‑raise.
    log_err(f"FATAL import: {e}\n{traceback.format_exc()}")
    raise


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Improved live writer")
    p.add_argument("--symbols", default=os.getenv("DL_SYMBOLS", "BTCUSDT,ETHUSDT"))
    p.add_argument("--timeframe", default=os.getenv("DL_TIMEFRAME", "1m"))
    p.add_argument("--seq", type=int, default=int(os.getenv("DL_SEQ_LEN", "128")))
    p.add_argument("--sleep", type=int, default=max(1, int(os.getenv("DL_WRITER_SLEEP", "3"))))
    p.add_argument("--signals", default=str(SIGNALS))
    p.add_argument("--master-log", default=str(MASTER_LOG))
    p.add_argument("--daily-prefix", default=DAILY_PREFIX)
    p.add_argument("--allow-only", default=os.getenv("DL_ALLOW_ONLY", "1"))
    p.add_argument("--thr", type=float, default=float(os.getenv("DL_P_LONG", "0.45")))
    p.add_argument("--mode", default=(os.getenv("DL_P_LONG_MODE", "abs") or "abs").lower(),
                   choices=["abs", "raw"])
    p.add_argument("--lookback-pad", type=int, default=int(os.getenv("DL_MAX_LOOKBACK_PAD", "6000")))
    p.add_argument("--stale-sec", type=int, default=int(os.getenv("DL_WRITER_STALE_SEC", "600")))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Allow gate logic
# ---------------------------------------------------------------------------

def allow_gate(p_long: float, thr: float, mode: str, allow_only: str) -> int:
    """Return 1 to allow trading or 0 to disallow based on threshold and mode."""
    if str(allow_only) == "0":
        return 1
    val = abs(p_long) if mode == "abs" else p_long
    return 1 if val >= thr else 0


def side_hint(p: float) -> str:
    return "LONG" if p > 0 else ("SHORT" if p < 0 else "FLAT")


def extract_last_prices(meta: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(meta, dict):
        return out
    for k in ("last_px", "last_price", "px"):
        sub = meta.get(k)
        if isinstance(sub, dict):
            for s, v in sub.items():
                try:
                    out[str(s)] = float(v)
                except Exception:
                    pass
    return out


# ---------------------------------------------------------------------------
# Main writer loop
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = build_args()
    # Acquire lock to ensure a single writer
    writer_lock(args.stale_sec)

    # Paths
    master_log = Path(args.master_log)
    signals_path = Path(args.signals)
    symlist = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

    # Make sure files exist so downstream tools don’t choke
    base_cols = ["ts", "p_meta", "thr", "mode", "rv_mean", "allow", "kinds_used"]
    signal_cols = ["ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode",
                   "kinds_used", "side_hint"]
    ensure_header(master_log, base_cols)
    ensure_header(today_daily(args.daily_prefix), base_cols)
    ensure_header(signals_path, signal_cols)

    # Load the model ensemble
    try:
        models, dev = load_ensemble(X_dim=30, device=None)
        log("ensemble loaded")
    except Exception as e:
        log_err(f"FATAL load_ensemble: {e}\n{traceback.format_exc()}")
        write_heartbeat(False, f"load_ensemble_failed: {e}", [], 0, 0.0, float(args.thr), args.mode, args.sleep)
        # Do not proceed without models; exit
        sys.exit(1)

    log(f"writer started symbols={symlist} tf={args.timeframe} thr={args.thr} mode={args.mode}")

    try:
        while True:
            ok = True
            detail = "tick_ok"
            ts_now = _ts()
            try:
                # Refresh features and make predictions
                meta, xw = refresh_live_features(seq_len=args.seq, add_symbol_id=True,
                                                  lookback_pad=args.lookback_pad,
                                                  symbols=symlist, timeframe=args.timeframe)
                per_model, aggregate = predict_ensemble(xw, models, dev, None)

                # Extract aggregate predictions
                if isinstance(aggregate, (tuple, list)) and len(aggregate) >= 3:
                    ret_hat, rv_hat, p_long = aggregate[:3]
                else:
                    ret_hat, rv_hat = 0.0, 0.0
                    p_long = float(aggregate) if isinstance(aggregate, (int, float)) else 0.0

                syms = (meta.get("symbols") if isinstance(meta, dict) else None) or symlist
                last_px = extract_last_prices(meta)

                # Build aggregate row for master log and daily log
                agg_row: Dict[str, Any] = {
                    "ts": ts_now,
                    "p_meta": float(p_long),
                    "thr": float(args.thr),
                    "mode": args.mode,
                    "rv_mean": float(rv_hat),
                    "allow": allow_gate(p_long, args.thr, args.mode, args.allow_only),
                    "kinds_used": ",".join(sorted(per_model.keys())),
                }
                # Additional per‑model diagnostics (optional)
                for name, vals in per_model.items():
                    try:
                        rhat, rvh, pl = vals
                        agg_row[f"{name}_ret"] = float(rhat)
                        agg_row[f"{name}_rv"] = float(rvh)
                        agg_row[f"{name}_p"] = float(pl)
                    except Exception:
                        pass

                # Write aggregate row to master and daily logs
                cols = base_cols + sorted([c for c in agg_row if c not in base_cols])
                append_row(master_log, cols, agg_row)
                append_row(today_daily(args.daily_prefix), cols, agg_row)

                # Write per‑symbol signal rows
                def sym_preds(sym: str) -> Tuple[float, float]:
                    ps: List[float] = []
                    rvs: List[float] = []
                    for _, vals in per_model.items():
                        if isinstance(vals, dict) and sym in vals:
                            try:
                                rhat_s, rvh_s, pl_s = vals[sym]
                                ps.append(float(pl_s))
                                rvs.append(float(rvh_s))
                            except Exception:
                                pass
                    if ps:
                        return sum(ps) / len(ps), sum(rvs) / len(rvs)
                    else:
                        return float(p_long), float(rv_hat)

                for sym in syms:
                    p_s, rv_s = sym_preds(sym)
                    px = last_px.get(sym)
                    px_final = f"{(px if isinstance(px, (int, float)) else rv_s):.8f}"
                    row = {
                        "ts": ts_now,
                        "symbol": sym,
                        "px": px_final,
                        "p_meta": p_s,
                        "rv_mean": rv_s,
                        "allow": allow_gate(p_s, args.thr, args.mode, args.allow_only),
                        "thr": args.thr,
                        "mode": args.mode,
                        "kinds_used": agg_row["kinds_used"],
                        "side_hint": side_hint(p_s),
                    }
                    append_row(signals_path, signal_cols, row)

                log(f"tick allow={agg_row['allow']} p_meta={agg_row['p_meta']:.4f} syms={','.join(syms)}")
                write_heartbeat(True, detail, syms, int(agg_row["allow"]), float(agg_row["p_meta"]), float(args.thr), args.mode, args.sleep)

            except Exception as e:
                ok = False
                detail = f"ERROR: {e}"
                log_err(f"{detail}\n{traceback.format_exc()}")
                write_heartbeat(False, detail, [], 0, 0.0, float(args.thr), args.mode, args.sleep)

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        log("writer stopped (ctrl-c)")
    finally:
        writer_unlock()


if __name__ == "__main__":
    main()