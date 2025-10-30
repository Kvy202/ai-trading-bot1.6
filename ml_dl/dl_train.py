# ml_dl/dl_train.py
import os as os_mod
import math
import argparse
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from .dl_models import TemporalConvNet, TinyTransformer, TinyLSTM
# Pull in advanced architectures if available.  These imports are
# optional; training will skip them unless the corresponding ``kind``
# argument is supplied.  See ``ml_dl/dl_models_adv.py`` for details.
try:
    from .dl_models_adv import AdvancedTransformer
except Exception:
    AdvancedTransformer = None  # type: ignore
from .dl_metrics import auc, mse_mae, information_coefficient, calibration_ece
from .dl_labels import next_k_logret, next_k_rv, binarize_return
from .dl_dataset import SeqDataset, load_prices_and_features  # <-- fixed
from .dl_walkforward import rolling_windows


# ---------------- model factory ----------------
def make_model(kind: str, in_dim: int):
    if kind == "tcn":
        return TemporalConvNet(in_dim)
    if kind == "tx":
        return TinyTransformer(in_dim)
    if kind == "lstm":
        return TinyLSTM(in_dim)
    # Advanced Transformer – more expressive than TinyTransformer
    if kind in ("adv", "tft", "transformer"):
        if AdvancedTransformer is None:
            raise ValueError(
                "Advanced model requested but not available. Ensure ml_dl/dl_models_adv.py exists."
            )
        return AdvancedTransformer(in_dim)
    raise ValueError(
        "kind must be one of 'tcn', 'tx', 'lstm' or an advanced kind like 'adv'/'tft'"
    )


def _slice_len(s, T=None):
    if isinstance(s, slice):
        return (s.stop - s.start)
    return len(s)


def _align_global_to_X(T: int, prices: np.ndarray, r: np.ndarray, rv: np.ndarray, y_cls: np.ndarray):
    """Keep most-recent T rows across all label arrays so they match X."""
    if not (len(r) == len(rv) == len(y_cls)):
        raise RuntimeError(f"Label arrays disagree: r={len(r)} rv={len(rv)} y={len(y_cls)}")
    if len(r) != T:
        r = r[-T:]
        rv = rv[-T:]
        y_cls = y_cls[-T:]
        if isinstance(prices, np.ndarray) and len(prices) >= T:
            prices = prices[-T:]
    return prices, r, rv, y_cls


def _align_fold_arrays(Xp: np.ndarray, r: np.ndarray, y: np.ndarray, rv: np.ndarray):
    """
    Ensure equal lengths per fold (small off-by-ones happen with rolling windows / horizon).
    Returns sliced views with n = min lens.
    """
    n = min(len(Xp), len(r), len(y), len(rv))
    if not (len(Xp) == len(r) == len(y) == len(rv)):
        print(f"[WARN] aligning fold lengths: X={len(Xp)} r={len(r)} y={len(y)} rv={len(rv)} -> {n}")
    return Xp[:n], r[:n], y[:n], rv[:n]


# ---------------- one epoch loop ----------------
def train_once(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: str,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
):
    crit_reg = nn.MSELoss()
    crit_cls = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)
    best_loss, best_state, bad = math.inf, None, 0

    # default if no val data
    A = MSE = MAE = IC = ECE = float("nan")

    for _ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for batch in loaders["train"]:

            x = batch["x"].to(device, non_blocking=True)
            y_rr = batch["y_ret_reg"].to(device, non_blocking=True)
            y_rc = batch["y_ret_cls"].to(device, non_blocking=True)
            y_rv = batch["y_rv_reg"].to(device, non_blocking=True)

            out = model(x)
            loss = (
                weights[0] * crit_reg(out["ret_reg"], y_rr)
                + weights[1] * crit_reg(out["rv_reg"], y_rv)
                + weights[2] * crit_cls(out["ret_cls_logits"], y_rc)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())

        # ---- val ----
        model.eval()
        with torch.no_grad():
            vloss, probs, ycls, rhat, rtrue = 0.0, [], [], [], []
            for batch in loaders["val"]:
                x = batch["x"].to(device, non_blocking=True)
                y_rr = batch["y_ret_reg"].to(device, non_blocking=True)
                y_rc = batch["y_ret_cls"].to(device, non_blocking=True)
                y_rv = batch["y_rv_reg"].to(device, non_blocking=True)

                out = model(x)
                loss = (
                    weights[0] * crit_reg(out["ret_reg"], y_rr)
                    + weights[1] * crit_reg(out["rv_reg"], y_rv)
                    + weights[2] * crit_cls(out["ret_cls_logits"], y_rc)
                )
                vloss += float(loss.item())

                p = out["ret_cls_logits"].softmax(-1)[:, 1]
                probs.append(p.cpu())
                ycls.append(y_rc.cpu())
                rhat.append(out["ret_reg"].cpu())
                rtrue.append(y_rr.cpu())

            if probs:
                probs = torch.cat(probs)
                ycls = torch.cat(ycls)
                rhat = torch.cat(rhat)
                rtrue = torch.cat(rtrue)
                A = auc(ycls, probs)
                MSE, MAE = mse_mae(rtrue, rhat)
                IC = information_coefficient(rtrue, rhat)
                ECE = calibration_ece(ycls, probs)
            else:
                vloss = float("inf")

        # ---- early stop ----
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "val_loss": float(best_loss),
        "val_auc": float(A),
        "val_mse": float(MSE),
        "val_mae": float(MAE),
        "val_ic": float(IC),
        "val_ece": float(ECE),
    }


# ---------------- helpers ----------------
def _parse_symbols(s: str) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def _derive_windows(
    T: int,
    seq_len: int,
    horizon: int,
    train_len: Optional[int],
    val_len: Optional[int],
    step: Optional[int],
) -> Tuple[int, int, int]:
    # If not provided, scale with data length (with minimums)
    tr = train_len or max(int(0.7 * T), 5000)
    va = val_len or max(int(0.1 * T), 1000)
    st = step or max(int(0.1 * T), 1000)

    # Ensure they fit with some cushion for seq/horizon
    cushion = seq_len + horizon + 32
    need = tr + va + cushion
    if T < need:
        raise RuntimeError(
            f"Not enough rows (T={T}) for windows: train={tr}, val={va}, "
            f"seq={seq_len}, horizon={horizon}. "
            f"Increase --lookback or reduce --train-len/--val-len."
        )
    return tr, va, st


# ---------------- script entry ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["tcn", "tx", "lstm"], default=os_mod.getenv("DL_MODEL_KIND", "tcn"))
    parser.add_argument("--seq-len", type=int, default=int(os_mod.getenv("DL_SEQ_LEN", "64")))
    parser.add_argument("--horizon", type=int, default=int(os_mod.getenv("DL_HORIZON_K", "12")))
    parser.add_argument("--batch", type=int, default=int(os_mod.getenv("DL_BATCH", "256")))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default=os_mod.getenv("DL_SAVE_DIR", "model_artifacts"))
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--device", type=str, default="auto")  # auto|cpu|cuda
    parser.add_argument("--symbols", type=str, default=os_mod.getenv("SYMBOL_WHITELIST", ""))  # CSV
    parser.add_argument("--timeframe", type=str, default=os_mod.getenv("TIMEFRAME", "5m"))
    parser.add_argument("--lookback", type=int, default=int(os_mod.getenv("LOOKBACK_CANDLES", "8000")))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-ret", type=float, default=1.0)  # regression weight
    parser.add_argument("--w-rv", type=float, default=1.0)   # rv regression weight
    parser.add_argument("--w-cls", type=float, default=1.0)  # classification weight
    parser.add_argument("--seed", type=int, default=42)
    # configurable walk-forward windows
    parser.add_argument("--train-len", type=int, default=None)
    parser.add_argument("--val-len", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()

    # device
    device = "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # optional feature order bundle
    feature_cols = None
    try:
        bundle = joblib.load(os_mod.getenv("MODEL_PATH", "models/model.pkl"))
        feature_cols = list(bundle.get("features", [])) or None
    except Exception:
        pass

    # symbols
    symbols = _parse_symbols(args.symbols)

    # load features + prices
    X, prices = load_prices_and_features(
        symbols=symbols,
        timeframe=args.timeframe,
        lookback=args.lookback,
        feature_cols=feature_cols,
        add_symbol_id=True,
        return_dfs=False,
    )
    T, F = X.shape
    print(f"[data] X shape={X.shape}, prices={prices.shape}, symbols={symbols or 'default SYMBOL'}")

    # labels (computed from prices), then ALIGN to X length
    r  = next_k_logret(prices, args.horizon)
    rv = next_k_rv(np.log(prices), args.horizon)
    y_cls = binarize_return(r, tau=0.0005)

    prices, r, rv, y_cls = _align_global_to_X(T, prices, r, rv, y_cls)

    # quick label balance peek (after alignment)
    if os_mod.getenv("QUIET_LABELS", "0") != "1":
        pos = int((y_cls == 1).sum())
        n = int(y_cls.shape[0])
        print(f"[Label balance] n={n}  positives={pos}  frac={pos/max(n,1)}")

    os_mod.makedirs(args.save_dir, exist_ok=True)

    # derive windows
    train_len, val_len, step = _derive_windows(
        T=T,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_len=args.train_len,
        val_len=args.val_len,
        step=args.step,
    )
    print(f"[windows] train_len={train_len}  val_len={val_len}  step={step}")

    # walk-forward training
    folds = 0
    for tr, va in rolling_windows(T, train_len=train_len, val_len=val_len, step=step):
        folds += 1

        # standardize by train window only
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32, copy=False)
        Xva = scaler.transform(X[va]).astype(np.float32, copy=False)

        # slice labels to the fold
        r_tr, y_tr, rv_tr = r[tr], y_cls[tr], rv[tr]
        r_va, y_va, rv_va = r[va], y_cls[va], rv[va]

        # --- per-fold alignment (prevents assertion) ---
        Xtr, r_tr, y_tr, rv_tr = _align_fold_arrays(Xtr, r_tr, y_tr, rv_tr)
        Xva, r_va, y_va, rv_va = _align_fold_arrays(Xva, r_va, y_va, rv_va)

        # datasets
        ds_tr = SeqDataset(Xtr, r_tr, y_tr, rv_tr, args.seq_len)
        ds_va = SeqDataset(Xva, r_va, y_va, rv_va, args.seq_len)

        dl_tr = DataLoader(
            ds_tr,
            batch_size=args.batch,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )

        print("fold lens:", _slice_len(tr), _slice_len(va))
        print("Xtr/Xva:", Xtr.shape, Xva.shape)
        print("r_tr/y_tr/rv_tr:", r_tr.shape, y_tr.shape, rv_tr.shape)
        print("r_va/y_va/rv_va:", r_va.shape, y_va.shape, rv_va.shape)
        print("ds_tr size:", len(ds_tr), " ds_va size:", len(ds_va))

        # model
        model = make_model(args.kind, in_dim=F).to(device)

        # train
        res = train_once(
            model,
            {"train": dl_tr, "val": dl_va},
            device,
            weights=(args.w_ret, args.w_rv, args.w_cls),
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
        )
        print(f"[VAL fold {folds}] {res}")

        # ---- save OOF for this fold (canonical keys) ----
        if os_mod.getenv("DL_SAVE_OOF", "0") == "1":
            import numpy as _np
            oof_dir = os_mod.getenv("DL_OOF_DIR", "oof")
            os_mod.makedirs(oof_dir, exist_ok=True)

            model.eval()
            with torch.no_grad():
                probs, ret_hat, rv_hat, y_cls_list = [], [], [], []
                for batch in dl_va:
                    x   = batch["x"].to(device, non_blocking=True)
                    yrc = batch["y_ret_cls"].to(device, non_blocking=True)
                    out = model(x)
                    p   = out["ret_cls_logits"].softmax(-1)[:, 1]
                    probs.append(p.cpu().numpy())
                    ret_hat.append(out["ret_reg"].cpu().numpy())
                    rv_hat.append(out["rv_reg"].cpu().numpy())
                    y_cls_list.append(yrc.cpu().numpy())

            p_long = _np.concatenate(probs).astype("float32")
            ret    = _np.concatenate(ret_hat).reshape(-1).astype("float32")
            rv     = _np.concatenate(rv_hat).reshape(-1).astype("float32")
            y_out  = _np.concatenate(y_cls_list).reshape(-1).astype("int8")

            # keep equal length in case last batch was pruned in SeqDataset
            L = min(len(p_long), len(ret), len(rv), len(y_out))
            if L < len(p_long) or L < len(ret) or L < len(rv) or L < len(y_out):
                print(f"[WARN] aligning OOF arrays -> {L}")
            p_long, ret, rv, y_out = p_long[:L], ret[:L], rv[:L], y_out[:L]

            tag  = args.tag or "latest"
            kind = args.kind
            out_path = os_mod.path.join(oof_dir, f"oof_{kind}_{tag}_{va.start}_{va.stop}.npz")
            _np.savez_compressed(
                out_path,
                p_long=p_long,
                y=y_out,
                rv=rv,
                ret=ret,
                meta=dict(
                    kind=kind, tag=tag, start=int(va.start), stop=int(va.stop),
                    seq_len=int(args.seq_len), horizon=int(args.horizon),
                    n_features=int(F),
                ),
            )
            print(f"[OOF] wrote {out_path}  | len={len(y_out)}  pos={int(y_out.sum())}")

        # ---- save artifacts for this fold ----
        sfx = f"{args.tag}_{va.start}_{va.stop}"
        joblib.dump(scaler, os_mod.path.join(args.save_dir, f"scaler_{sfx}.joblib"))
        torch.save(model.state_dict(), os_mod.path.join(args.save_dir, f"dl_{args.kind}_{sfx}.pt"))

        # refresh "latest"
        joblib.dump(scaler, os_mod.path.join(args.save_dir, "scaler_latest.joblib"))
        torch.save(model.state_dict(), os_mod.path.join(args.save_dir, f"dl_{args.kind}_latest.pt"))

    if folds == 0:
        raise RuntimeError(
            "No folds produced — not enough data? "
            "Increase --lookback or reduce --train-len/--val-len."
        )


if __name__ == "__main__":
    main()
