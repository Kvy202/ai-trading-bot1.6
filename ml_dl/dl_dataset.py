import numpy as np
import torch
from torch.utils.data import Dataset

# --- Re-export loader so `from ml_dl.dl_dataset import load_prices_and_features` works ---
try:
    from data import load_prices_and_features as _lpaf
except Exception as e:
    _lpaf = None
    _lpaf_err = e

def load_prices_and_features(
    symbols=None,
    timeframe: str = "1m",
    lookback: int = 8000,
    feature_cols=None,
    add_symbol_id: bool = True,
    return_dfs: bool = False,
    **kwargs,
):
    """
    Thin wrapper that forwards to data.load_prices_and_features with the same signature
    your data.py supports. We intentionally DO NOT pass unknown kwargs like start/end.
    """
    if _lpaf is None:
        raise ImportError(
            "data.load_prices_and_features is not available. "
            "Ensure data.py is present and importable."
        ) from _lpaf_err

    # Only forward the parameters that data.py actually supports.
    return _lpaf(
        symbols=symbols,
        timeframe=timeframe,
        lookback=lookback,
        feature_cols=feature_cols,
        add_symbol_id=add_symbol_id,
        return_dfs=return_dfs,
    )

class SeqDataset(Dataset):
    """
    Builds fixed-length windows of size L from a *single contiguous slice* of X and aligned targets.
    Drops first L-1 rows, any window with NaN/inf, and any target with NaN/inf.
    Inputs:
      X   : np.ndarray [T, F]
      r   : np.ndarray [T]   (regression target)
      y   : np.ndarray [T]   (classification target, int64 {0,1})
      rv  : np.ndarray [T]   (realized vol target)
      L   : int              (sequence length)
    """
    def __init__(self, X: np.ndarray, r: np.ndarray, y: np.ndarray, rv: np.ndarray, L: int):
        assert len(X) == len(r) == len(y) == len(rv), "X and targets must have same T"
        self.X = np.asarray(X, dtype=np.float32)
        self.r = np.asarray(r, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.rv = np.asarray(rv, dtype=np.float32)
        self.L = int(L)

        T = len(self.X)
        valid = []
        targ_ok = np.isfinite(self.r) & np.isfinite(self.rv)

        for i in range(self.L - 1, T):
            if not targ_ok[i]:
                continue
            s = i - self.L + 1
            win = self.X[s : i + 1]
            if win.shape[0] != self.L:
                continue
            if not np.isfinite(win).all():
                continue
            valid.append(i)

        self.idx = np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        s = i - self.L + 1
        x = self.X[s : i + 1]           # [L, F]
        rr = self.r[i]                  # scalar
        rc = self.y[i]                  # class id
        rv = self.rv[i]                 # scalar

        return {
            "x": torch.from_numpy(x),               # [L, F], float32
            "y_ret_reg": torch.tensor(rr, dtype=torch.float32),
            "y_ret_cls": torch.tensor(rc, dtype=torch.long),
            "y_rv_reg": torch.tensor(rv, dtype=torch.float32),
        }
