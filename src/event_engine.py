"""Causal support/resistance levels + early event labels (zero lookahead).

Features are computed from OHLCV at time <= t only (trailing windows, never a
centered pivot).  The event label for bar t is resolved over a FUTURE horizon
[t+1 .. t+H] in the last pass: that future is used ONLY to create the training
label, never as a feature.

Event classes:
    0 NO_EVENT
    1 SUPPORT_BREAKDOWN     (break low + confirm -X% before reclaim)
    2 RESISTANCE_BREAKOUT   (break high + confirm +X% before reclaim)
    3 FALSE_BREAK           (cross level then reclaim within N bars)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_CLASSES = 4
CLASS_NAMES = {0: "NO_EVENT", 1: "SUPPORT_BREAKDOWN", 2: "RESISTANCE_BREAKOUT", 3: "FALSE_BREAK"}


@dataclass
class EventConfig:
    swing_window: int = 7            # trailing bars for a causal swing pivot
    level_lookback: int = 500        # trailing bars used to form each level
    touch_tol: float = 0.0005        # +/-0.05% = a "touch" of a level
    break_threshold: float = 0.0005  # 0.05% beyond the level = a break
    confirmation_target: float = 0.003  # 0.30% move = confirmation
    horizon: int = 30                # future bars used to resolve the label
    false_reclaim_bars: int = 5      # reclaim within N bars => false break
    stage_near_pct: float = 0.002    # STAGE_1: within 0.20% of a level


def _shift(a, k: int) -> np.ndarray:
    """Return a copy of a float array shifted by k steps; fill with NaN."""
    a = np.asarray(a, dtype=np.float64)
    if k > 0:
        return np.concatenate([a[k:], np.full(k, np.nan)])
    return np.concatenate([np.full(-k, np.nan), a[:k]])


def swing_flags(high, low, window):
    """Causal swing pivots: a bar is a swing high/low if it is the trailing
    max/min over `window` bars AND strictly above/below its left neighbour."""
    hi = pd.Series(high)
    lo = pd.Series(low)
    is_high = (hi == hi.rolling(window, min_periods=window).max()) & (hi > hi.shift(1))
    is_low = (lo == lo.rolling(window, min_periods=window).min()) & (lo < lo.shift(1))
    return is_high.to_numpy(), is_low.to_numpy()


def build_levels(wide, cfg, c, h, l) -> pd.DataFrame:
    """Dynamic causal support/resistance from historical swing pivots."""
    close = wide[c].to_numpy(np.float64)
    high = wide[h].to_numpy(np.float64)
    low = wide[l].to_numpy(np.float64)

    is_high, is_low = swing_flags(high, low, cfg.swing_window)
    ph = pd.Series(np.where(is_high, high, np.nan), index=wide.index)
    pl = pd.Series(np.where(is_low, low, np.nan), index=wide.index)

    # nearest resistance = trailing min of swing highs; support = trailing max
    # of swing lows.  Trailing windows only => no future data.
    resistance = ph.ffill().rolling(cfg.level_lookback, min_periods=1).min()
    support = pl.ffill().rolling(cfg.level_lookback, min_periods=1).max()

    touch = cfg.touch_tol
    low_s = pd.Series(low, index=wide.index)
    high_s = pd.Series(high, index=wide.index)
    close_s = pd.Series(close, index=wide.index)
    sup_touch = (low_s <= support * (1 + touch)) & (high_s >= support * (1 - touch))
    res_touch = (low_s <= resistance * (1 + touch)) & (high_s >= resistance * (1 - touch))
    sup_touches = sup_touch.astype(int).rolling(cfg.level_lookback, min_periods=1).sum().shift(1)
    res_touches = res_touch.astype(int).rolling(cfg.level_lookback, min_periods=1).sum().shift(1)

    dS = (close_s / support - 1.0).clip(-0.03, 0.03)
    dR = (close_s / resistance - 1.0).clip(-0.03, 0.03)
    rng = ((close_s - support) / (resistance - support + 1e-12)).clip(0.0, 1.0)

    return pd.DataFrame({
        "support": support.to_numpy(),
        "resistance": resistance.to_numpy(),
        "dist_support": dS.to_numpy(),
        "dist_resistance": dR.to_numpy(),
        "support_touches": sup_touches.to_numpy(),
        "resistance_touches": res_touches.to_numpy(),
        "support_strength": (sup_touches / cfg.level_lookback).to_numpy(),
        "resistance_strength": (res_touches / cfg.level_lookback).to_numpy(),
        "range_norm": rng.to_numpy(),
    })


def build_event_labels(wide, levels, cfg, c, h, l) -> pd.DataFrame:
    close = wide[c].to_numpy(np.float64)
    high = wide[h].to_numpy(np.float64)
    low = wide[l].to_numpy(np.float64)
    support = levels["support"].to_numpy(np.float64)
    resistance = levels["resistance"].to_numpy(np.float64)
    n = len(close)
    H = cfg.horizon
    sup_brk = support * (1.0 - cfg.break_threshold)
    sup_conf = support * (1.0 - cfg.confirmation_target)
    res_brk = resistance * (1.0 + cfg.break_threshold)
    res_conf = resistance * (1.0 + cfg.confirmation_target)
    Bb = np.zeros((n, H), dtype=bool)
    Cb = np.zeros((n, H), dtype=bool)
    Rb = np.zeros((n, H), dtype=bool)
    Bu = np.zeros((n, H), dtype=bool)
    Cu = np.zeros((n, H), dtype=bool)
    Ru = np.zeros((n, H), dtype=bool)
    fwd_min = np.full(n, np.inf)
    fwd_max = np.full(n, -np.inf)
    for k in range(H):
        cl = _shift(close, k + 1)
        Bb[:, k] = cl < sup_brk
        Cb[:, k] = _shift(low, k + 1) <= sup_conf
        Rb[:, k] = cl > support
        Bu[:, k] = cl > res_brk
        Cu[:, k] = _shift(high, k + 1) >= res_conf
        Ru[:, k] = cl < resistance
        r = np.where(np.isnan(cl), np.nan, cl / close - 1.0)
        fwd_min = np.fmin(fwd_min, r)
        fwd_max = np.fmax(fwd_max, r)
    def argfirst(mat):
        has = mat.any(axis=1)
        out = np.full(n, H, dtype=np.int64)
        out[has] = np.argmax(mat[has], axis=1)
        return out, has
    def first_after(mat, p):
        allowed = np.arange(H)[None, :] > p[:, None]
        masked = np.where(allowed, mat, False)
        has = masked.any(axis=1)
        out = np.full(n, H, dtype=np.int64)
        out[has] = np.argmax(masked[has], axis=1)
        return out, has
    b_dn, has_dn_break = argfirst(Bb)
    b_up, has_up_break = argfirst(Bu)
    c_dn, has_c_dn = first_after(Cb, b_dn)
    r_dn, has_r_dn = first_after(Rb, b_dn)
    c_up, has_c_up = first_after(Cu, b_up)
    r_up, has_r_up = first_after(Ru, b_up)
    near_up = close <= resistance * (1.0 + cfg.break_threshold)
    near_dn = close >= support * (1.0 - cfg.break_threshold)
    conf_dn = near_dn & has_dn_break & has_c_dn & (c_dn < r_dn)
    conf_up = near_up & has_up_break & has_c_up & (c_up < r_up)
    false_dn = near_dn & has_dn_break & has_r_dn & (r_dn <= cfg.false_reclaim_bars) & ~conf_dn
    false_up = near_up & has_up_break & has_r_up & (r_up <= cfg.false_reclaim_bars) & ~conf_up
    label = np.full(n, 0, dtype=np.int8)
    label[conf_dn] = 1
    label[conf_up] = 2
    label[(false_dn | false_up) & (label == 0)] = 3
    tb = np.full(n, np.nan)
    tb[has_dn_break] = b_dn[has_dn_break]
    tb[has_up_break & ~has_dn_break] = b_up[has_up_break & ~has_dn_break]
    return pd.DataFrame({
        "event": label,
        "fwd_min_ret": np.where(np.isinf(fwd_min), np.nan, fwd_min),
        "fwd_max_ret": np.where(np.isinf(fwd_max), np.nan, fwd_max),
        "time_to_break": tb,
    })
def build_stage(levels, cfg) -> np.ndarray:
    dS = levels["dist_support"].to_numpy()
    dR = levels["dist_resistance"].to_numpy()
    n = len(dS)
    near = cfg.stage_near_pct
    touch = cfg.break_threshold
    conf = cfg.confirmation_target
    near_any = np.logical_or((dS >= 0) & (dS <= near), (dR <= 0) & (-dR <= near))
    touch_any = np.logical_or((dS >= 0) & (dS <= touch), (dR <= 0) & (-dR <= touch))
    broke = np.logical_or(dS <= -touch, dR >= touch)
    confirmed = np.logical_or(dS <= -conf, dR >= conf)
    stage = np.zeros(n, dtype=np.int8)
    stage[near_any] = 1
    stage[touch_any] = 2
    stage[broke] = 3
    stage[confirmed] = 4
    return stage

def add_wick_features(wide, cfg, c, o, h, l) -> pd.DataFrame:
    close = wide[c].to_numpy(np.float64)
    open_ = wide[o].to_numpy(np.float64)
    high = wide[h].to_numpy(np.float64)
    low = wide[l].to_numpy(np.float64)
    rng = (high - low) + 1e-12
    close_s = pd.Series(close, index=wide.index)
    hh = close_s.rolling(20, min_periods=20).max()
    ll = close_s.rolling(20, min_periods=20).min()
    hh_prev = hh.shift(1)
    ll_prev = ll.shift(1)
    return pd.DataFrame({
        "upper_wick": (high - np.maximum(close, open_)) / rng,
        "lower_wick": (np.minimum(close, open_) - low) / rng,
        "body_abs": np.abs(close - open_) / rng,
        "body_dir": (close - open_) / rng,
        "higher_high": (hh > hh_prev).astype(int).to_numpy(),
        "higher_low": (ll > ll_prev).astype(int).to_numpy(),
        "lower_high": (hh < hh_prev).astype(int).to_numpy(),
        "lower_low": (ll < ll_prev).astype(int).to_numpy(),
    })
