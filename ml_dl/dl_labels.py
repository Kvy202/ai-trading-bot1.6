import numpy as np

def next_k_logret(prices: np.ndarray, k: int) -> np.ndarray:
    # prices shape (T,)
    r = np.full_like(prices, fill_value=np.nan, dtype=float)
    r[:-k] = np.log(prices[k:]) - np.log(prices[:-k])
    return r

def next_k_rv(log_prices: np.ndarray, k: int) -> np.ndarray:
    # log_prices shape (T,)
    rv = np.full_like(log_prices, np.nan, dtype=float)
    diffs = np.diff(log_prices)
    # rolling sum of squared diffs
    sq = diffs**2
    # cumulative sum trick
    csum = np.cumsum(sq)
    rv[:-(k)] = np.sqrt(csum[k-1:] - np.concatenate(([0.0], csum[:-k])))
    return rv

def binarize_return(r: np.ndarray, tau: float = 0.0) -> np.ndarray:
    y = np.where(r >= tau, 1, 0)
    return y
