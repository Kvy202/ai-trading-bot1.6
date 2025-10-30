import os
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
from labels import triple_barrier_labels

# 0.15% ATR gate (tweak via VOL_RATIO_THRESH if needed)
VOL_RATIO_THRESH = float(os.getenv("VOL_RATIO_THRESH", "0.0015"))

def make_label(df: pd.DataFrame) -> pd.Series:
    """Simple next-candle label (unused when using triple_barrier_labels)."""
    fut_ret = df['close'].pct_change().shift(-1)
    return (fut_ret > 0).astype(int)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # basic returns/vol
    out['ret'] = out['close'].pct_change()
    out['vol'] = out['ret'].rolling(30).std()

    # momentum
    rsi = RSIIndicator(close=out['close'], window=14)
    out['rsi_14'] = rsi.rsi()

    stoch = StochasticOscillator(
        high=out['high'], low=out['low'], close=out['close'],
        window=14, smooth_window=3
    )
    out['stoch_k'] = stoch.stoch()
    out['stoch_d'] = stoch.stoch_signal()

    # trend
    macd = MACD(close=out['close'], window_slow=26, window_fast=12, window_sign=9)
    out['macd'] = macd.macd()
    out['macd_signal'] = macd.macd_signal()
    out['macd_hist'] = macd.macd_diff()

    out['ema_12'] = EMAIndicator(close=out['close'], window=12).ema_indicator()
    out['ema_26'] = EMAIndicator(close=out['close'], window=26).ema_indicator()
    out['ema_20'] = EMAIndicator(close=out['close'], window=20).ema_indicator()
    out['ema_50'] = EMAIndicator(close=out['close'], window=50).ema_indicator()
    out['sma_50'] = SMAIndicator(close=out['close'], window=50).sma_indicator()
    out['sma_200'] = SMAIndicator(close=out['close'], window=200).sma_indicator()

    # ATR
    atr = AverageTrueRange(high=out['high'], low=out['low'], close=out['close'], window=14)
    out['atr_14'] = atr.average_true_range()

    # lags & rolling stats
    for k in [1, 2, 3, 4, 5]:
        out[f'ret_lag{k}'] = out['ret'].shift(k)
    out['roll_std_10']  = out['close'].rolling(10).std()
    out['roll_mean_10'] = out['close'].rolling(10).mean() / out['close'] - 1
    out['roll_mean_50'] = out['close'].rolling(50).mean() / out['close'] - 1
    out['roll_std_50']  = out['close'].rolling(50).std()

    # multi-timeframe proxies
    out['ema_20_4h_proxy'] = out['close'].ewm(span=20*4, adjust=False).mean()
    out['ema_50_4h_proxy'] = out['close'].ewm(span=50*4, adjust=False).mean()

    # gates (single definitions)
    out['trend_ok']     = (out['ema_12'] > out['ema_26']).astype(int)
    out['momentum_ok']  = ((out['close'] > out['ema_20']) & (out['ema_20'] > out['ema_50'])).astype(int)
    out['vol_ok']       = (out['atr_14'] / out['close'] > VOL_RATIO_THRESH).astype(int)

    out = out.dropna().copy()
    return out

def make_dataset(df: pd.DataFrame):
    feats = build_features(df)

    tf = os.getenv("TIMEFRAME", "1h").lower()
    if tf in ("1m","3m","5m"):
        pt, sl, max_h = 0.003, 0.003, 36   # ~0.3%
    elif tf in ("15m","30m"):
        pt, sl, max_h = 0.006, 0.006, 48   # ~0.6%
    else:
        pt, sl, max_h = 0.01, 0.01, 20     # ~1%

    y = triple_barrier_labels(feats, pt=pt, sl=sl, max_h=max_h).dropna()
    feats = feats.loc[y.index]

    feature_cols = [
        'ret','rsi_14','macd','macd_signal','macd_hist','stoch_k','stoch_d',
        'ema_12','ema_26','ema_20','ema_50','sma_50','sma_200','atr_14','vol',
        'ret_lag1','ret_lag2','ret_lag3','ret_lag4','ret_lag5',
        'roll_mean_10','roll_std_10','roll_mean_50','roll_std_50',
        'ema_20_4h_proxy','ema_50_4h_proxy','trend_ok','momentum_ok','vol_ok'
    ]
    X = feats[feature_cols].apply(pd.to_numeric, errors='coerce').dropna()
    y = y.loc[X.index]

    if os.getenv("QUIET_LABELS","0") != "1":
        print("[Label balance] n=", len(y), "  positives=", int((y==1).sum()), "  frac=", float((y==1).mean()))

    return X, y, feats.loc[X.index]
