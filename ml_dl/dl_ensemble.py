# ml_dl/dl_ensemble.py
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch

from .dl_infer import load_model, predict_next
from .dl_dataset import load_prices_and_features  # real loader

# ------------------------
# Device selection helper
# ------------------------
def _pick_device() -> str:
    pref = os.getenv("DL_DEVICE", "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("cuda", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

# ------------------------
# Helpers for file discovery
# ------------------------
def _default_model_dir() -> str:
    return os.getenv("DL_MODEL_DIR", "model_artifacts")

def _fallback_scaler_path() -> str:
    return os.path.join(_default_model_dir(), "scaler_latest.joblib")

def _fallback_model_paths(kind: str) -> List[str]:
    d = _default_model_dir()
    return [
        os.path.join(d, f"dl_{kind}_latest.pt"),
        os.path.join(d, f"dl_{kind}.pt"),
    ]

def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

# ------------------------
# Load all base models
# ------------------------
def load_ensemble(X_dim: int, device: Optional[str] = None) -> Tuple[Dict[str, dict], str]:
    dev = device or _pick_device()

    def maybe_load(kind: str, scaler_env: str, model_env: str):
        scaler_path_env = os.getenv(scaler_env, "").strip()
        model_path_env  = os.getenv(model_env, "").strip()

        scaler_path = _first_existing([scaler_path_env, _fallback_scaler_path()])
        model_path  = _first_existing([model_path_env] + _fallback_model_paths(kind))

        if not scaler_path or not model_path:
            print(f"[dl_ensemble] skip {kind}: missing files scaler={scaler_path!r} model={model_path!r}")
            return None

        try:
            scaler, model, _dev = load_model(kind, X_dim, scaler_path, model_path, device=dev)
            return {"scaler": scaler, "model": model}
        except Exception as e:
            print(f"[dl_ensemble] WARNING: failed to load {kind}: {e}")
            return None

    models: Dict[str, dict] = {}
    models["tcn"]  = maybe_load("tcn",  "DL_TCN_SCALER_PATH",  "DL_TCN_MODEL_PATH")
    models["lstm"] = maybe_load("lstm", "DL_LSTM_SCALER_PATH", "DL_LSTM_MODEL_PATH")
    models["tx"]   = maybe_load("tx",   "DL_TX_SCALER_PATH",   "DL_TX_MODEL_PATH")

    models = {k: v for k, v in models.items() if v is not None}
    if not models:
        raise RuntimeError(
            "No ensemble members loaded. Provide *_SCALER_PATH and *_MODEL_PATH envs, "
            "or ensure defaults exist: model_artifacts/scaler_latest.joblib and "
            "model_artifacts/dl_{tcn,lstm,tx}_latest.pt"
        )

    return models, dev

# ------------------------
# Feature alignment
# ------------------------
def _align_feat_dim(x_window: np.ndarray, target_dim: int) -> np.ndarray:
    x = np.asarray(x_window, dtype=np.float32)
    T, F = x.shape
    if F == target_dim:
        return x
    if F > target_dim:
        return x[:, :target_dim]
    pad = np.zeros((T, target_dim - F), dtype=np.float32)
    return np.concatenate([x, pad], axis=1)

# ------------------------
# Per-model + blended preds
# ------------------------
def predict_ensemble(
    x_window: np.ndarray,
    models: Dict[str, dict],
    device: str,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, tuple], tuple]:
    kinds: List[str] = list(models.keys())
    if not kinds:
        raise ValueError("predict_ensemble: empty models dict")

    if weights is None:
        weights_env = os.getenv("DL_MODEL_WEIGHTS", "").strip()
        parsed: Dict[str, float] = {}
        if weights_env:
            for part in weights_env.split(','):
                if ':' not in part:
                    continue
                k, v = part.split(':', 1)
                try:
                    parsed[k.strip()] = float(v)
                except Exception:
                    continue
        if parsed:
            s = sum(max(0.0, parsed.get(k, 0.0)) for k in kinds) or 1.0
            weights = {k: max(0.0, parsed.get(k, 0.0)) / s for k in kinds}
        else:
            weights = {k: 1.0 / len(kinds) for k in kinds}
    else:
        s = sum(max(0.0, weights.get(k, 0.0)) for k in kinds) or 1.0
        weights = {k: max(0.0, weights.get(k, 0.0)) / s for k in kinds}

    per_model: Dict[str, tuple] = {}
    for k, pack in models.items():
        scaler = pack["scaler"]
        target_dim = int(getattr(scaler, "n_features_in_", x_window.shape[1]))
        xw_aligned = _align_feat_dim(x_window, target_dim)
        ret_hat, rv_hat, p_long = predict_next(xw_aligned, scaler, pack["model"], device)
        per_model[k] = (float(ret_hat), float(rv_hat), float(p_long))

    rets  = np.array([per_model[k][0] for k in kinds], dtype=np.float32)
    rvs   = np.array([per_model[k][1] for k in kinds], dtype=np.float32)
    plons = np.array([per_model[k][2] for k in kinds], dtype=np.float32)
    ws    = np.array([weights[k] for k in kinds], dtype=np.float32)

    blend = (float(np.dot(rets, ws)), float(np.dot(rvs, ws)), float(np.dot(plons, ws)))
    return per_model, blend

# ------------------------
# Robust live feature refresh
# ------------------------
def refresh_live_features(
    seq_len: int,
    add_symbol_id: bool,
    lookback_pad: int = 200,
    symbols: Optional[list] = None,
    timeframe: Optional[str] = None,
):
    """
    Pull a recent feature window and return (X_live, x_window),
    auto-increasing lookback_pad until we have >= seq_len rows,
    up to a safe cap.
    """
    max_pad = int(os.getenv("DL_MAX_LOOKBACK_PAD", "5000"))
    pad = max(lookback_pad, 64)

    last_err = None
    while pad <= max_pad:
        try:
            X_live, _ = load_prices_and_features(
                symbols=symbols,
                timeframe=timeframe,
                lookback=seq_len + pad,
                add_symbol_id=add_symbol_id,
                return_dfs=False,
            )
            if X_live.shape[0] >= seq_len:
                xw = X_live[-seq_len:, :]
                return X_live, xw
            last_err = f"got {X_live.shape[0]} rows with pad={pad}"
        except Exception as e:
            last_err = str(e)
        pad = int(pad * 2)  # exponential backoff

    raise RuntimeError(
        f"refresh_live_features: insufficient rows for seq_len={seq_len} "
        f"even after pad up to {max_pad} (last error: {last_err})"
    )
