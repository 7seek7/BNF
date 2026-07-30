# -*- coding: utf-8 -*-
"""
V8.4 修复版策略引擎
证据系统后置为评分器 + 震荡优先分类 + 真实P1波谷 + BaseScore×证据加权
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple

EPS = 1e-12


class MarketTrendType(Enum):
    UPTREND = 'UPTREND'
    DOWNTREND = 'DOWNTREND'
    RANGEBOUND = 'RANGEBOUND'
    HIGH_VOLATILITY_TREND = 'HIGH_VOLATILITY_TREND'
    HALT = 'HALT'
    TRIANGLE = 'TRIANGLE'
    UNKNOWN = 'UNKNOWN'


class V83SignalType(Enum):
    BO_LONG = 'BO_LONG'
    BO_SHORT = 'BO_SHORT'
    RB_LONG = 'RB_LONG'
    RB_SHORT = 'RB_SHORT'
    TR_LONG = 'TR_LONG'
    TR_SHORT = 'TR_SHORT'
    TRI_LONG = 'TRI_LONG'
    TRI_SHORT = 'TRI_SHORT'


@dataclass
class V83Signal:
    signal_type: V83SignalType
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_t1: float
    take_profit_t2: float
    take_profit_t3: float
    score: float
    leverage: int
    position_ratio: float
    bar_idx: int
    evidence_score: int
    reason: str = ''


# ======================================================================
# 趋势分类 — 震荡优先识别
# ======================================================================

def _find_swing_points(typical, left=2, right=2):
    n = len(typical)
    highs = []
    lows = []
    for i in range(left, n - right):
        if typical[i] == max(typical[i - left:i + right + 1]):
            highs.append((i, typical[i]))
        if typical[i] == min(typical[i - left:i + right + 1]):
            lows.append((i, typical[i]))
    return highs, lows


def _verify_swing_high(highs, low_arr, atr7, idx, price):
    for j, (si, _) in enumerate(highs):
        if si == idx:
            if j + 1 < len(highs):
                next_idx = highs[j + 1][0]
            else:
                next_idx = min(idx + 5, len(low_arr) - 1)
            sub_low = np.min(low_arr[idx:next_idx + 1])
            drop_pct = (price - sub_low) / max(price, EPS)
            return drop_pct >= 0.4 * atr7[idx] / max(price, EPS) if atr7[idx] > 0 else False
    return False


def _verify_swing_low(lows, high_arr, atr7, idx, price):
    for j, (si, _) in enumerate(lows):
        if si == idx:
            if j + 1 < len(lows):
                next_idx = lows[j + 1][0]
            else:
                next_idx = min(idx + 5, len(high_arr) - 1)
            sub_high = np.max(high_arr[idx:next_idx + 1])
            bounce_pct = (sub_high - price) / max(price, EPS)
            return bounce_pct >= 0.5 * atr7[idx] / max(price, EPS) if atr7[idx] > 0 else False
    return False


def classify_trend_v83(high, low, close, atr14, atr7, idx, adx_arr=None, lookback=120):
    """
    震荡优先趋势分类（修复版）
    第一层：硬性震荡判定（振幅 < 3×ATR → 强制震荡）
    第二层：ADX辅助（ADX<20 且 振幅<5×ATR → 震荡）
    第三层：纯结构判定（仅当振幅≥3×ATR且有HH/HL时才单边）
    返回 (trend, has_hhhl, has_lhll, swing_count, r2_score, slope_pct, lock_bar)
    """
    if idx < 40:
        return MarketTrendType.UNKNOWN, False, False, 0, 0.0, 0.0, 0

    cur_atr = atr14[idx] if idx < len(atr14) else 1.0
    cur_close = close[idx]
    cur_adx = adx_arr[idx] if adx_arr is not None and idx < len(adx_arr) else 0

    # 最近20根K线的价格振幅（绝对值）
    start20 = max(0, idx - 19)
    high20 = np.max(high[start20:idx + 1])
    low20 = np.min(low[start20:idx + 1])
    amp_20 = (high20 - low20) / max(cur_close, EPS)

    # ========== 第一层：硬性震荡判定 ==========
    if amp_20 < 3.0 * (cur_atr / max(cur_close, EPS)):
        return MarketTrendType.RANGEBOUND, False, False, 0, 0.0, 0.0, 0

    # ========== 第二层：ADX辅助 ==========
    if cur_adx < 20 and amp_20 < 5.0 * (cur_atr / max(cur_close, EPS)):
        return MarketTrendType.RANGEBOUND, False, False, 0, 0.0, 0.0, 0

    # ========== 第三层：纯结构判定 ==========
    start = max(0, idx - lookback + 1)
    typical = (high[start:idx + 1] + low[start:idx + 1] + close[start:idx + 1]) / 3

    raw_highs, raw_lows = _find_swing_points(typical, left=2, right=2)

    valid_highs = [(i, typical[i]) for i, v in raw_highs
                   if _verify_swing_high(raw_highs, low[start:idx + 1],
                                          atr7[start:idx + 1], i, typical[i])]
    valid_lows = [(i, typical[i]) for i, v in raw_lows
                  if _verify_swing_low(raw_lows, high[start:idx + 1],
                                       atr7[start:idx + 1], i, typical[i])]

    recent_highs = valid_highs[-6:] if len(valid_highs) > 6 else valid_highs
    recent_lows = valid_lows[-6:] if len(valid_lows) > 6 else valid_lows

    h_prices = [p for _, p in recent_highs]
    l_prices = [p for _, p in recent_lows]

    hh = sum(1 for j in range(1, len(h_prices)) if h_prices[j] > h_prices[j - 1])
    hl = sum(1 for j in range(1, len(l_prices)) if l_prices[j] > l_prices[j - 1])
    lh = sum(1 for j in range(1, len(h_prices)) if h_prices[j] < h_prices[j - 1])
    ll = sum(1 for j in range(1, len(l_prices)) if l_prices[j] < l_prices[j - 1])

    has_hhhl = hl >= 2 and hh >= 2
    has_lhll = lh >= 2 and ll >= 2
    swing_count = max(hh, hl, lh, ll)

    cur_atr_ratio = cur_atr / max(cur_close, EPS) * 100

    # 高波动分轨
    if cur_atr_ratio >= 2.5:
        if has_hhhl or has_lhll:
            return MarketTrendType.HIGH_VOLATILITY_TREND, has_hhhl, has_lhll, swing_count, 0.0, 0.0, 0
        return MarketTrendType.HALT, has_hhhl, has_lhll, swing_count, 0.0, 0.0, 0

    if has_hhhl and swing_count >= 3:
        return MarketTrendType.UPTREND, has_hhhl, has_lhll, swing_count, 0.0, 0.0, 3
    if has_lhll and swing_count >= 3:
        return MarketTrendType.DOWNTREND, has_hhhl, has_lhll, swing_count, 0.0, 0.0, 3

    return MarketTrendType.RANGEBOUND, has_hhhl, has_lhll, swing_count, 0.0, 0.0, 0


# ======================================================================
# 零延迟证据系统 — 仅作评分，不再拦截
# ======================================================================

def evaluate_evidence(close, high, low, volume, atr14, vsr, idx):
    if idx < 14:
        return 0, 'NEUTRAL', {}, {}

    c = close[idx]; o = close[idx - 1]
    h = high[idx]; l_ = low[idx]
    body = abs(c - o)
    m = {}

    m['structure_score'] = 0
    lookback = min(15, idx)
    prev_high = np.max(close[idx - lookback:idx]) if lookback > 0 else c
    prev_low = np.min(close[idx - lookback:idx]) if lookback > 0 else c
    if c > prev_high:
        m['structure_score'] = 3
    elif c < prev_low:
        m['structure_score'] = 3

    m['body_ratio'] = body / max(atr14[idx], EPS) if atr14[idx] > 0 else 0
    m['strong_body'] = m['body_ratio'] >= 0.50  # 15m单K线body/ATR很难>0.8，放宽到0.5

    if idx >= 4:
        cur_slope = c - close[idx - 1]
        prev_slopes = [close[i] - close[i - 1] for i in range(idx - 2, idx)]
        avg_prev = np.mean(prev_slopes) if prev_slopes else 0
        m['momentum_accel'] = abs(cur_slope) > abs(avg_prev) * 1.5 if abs(avg_prev) > EPS else False
        m['momentum_dir'] = 1 if cur_slope > 0 else (-1 if cur_slope < 0 else 0)
    else:
        m['momentum_accel'] = False
        m['momentum_dir'] = 0

    m['vsr'] = vsr[idx] if idx < len(vsr) else 1.0

    if idx >= 10:
        amp_hist = [(high[i] - low[i]) / max(close[i], EPS) for i in range(idx - 10, idx)]
        avg_amp = np.mean(amp_hist) if amp_hist else 0
        cur_amp = (h - l_) / max(c, EPS)
        m['vol_expand'] = cur_amp > avg_amp * 1.5 if avg_amp > EPS else False
    else:
        m['vol_expand'] = False

    total = m['structure_score']
    bull_count = bear_count = 0
    if m['structure_score'] >= 3:
        if c > prev_high: bull_count += 1
        else: bear_count += 1
    if m['strong_body'] and c > o and c > close[idx - 1]:
        total += 2; bull_count += 1
    elif m['strong_body'] and c < o and c < close[idx - 1]:
        total += 2; bear_count += 1
    if m['momentum_accel'] and m['momentum_dir'] > 0:
        total += 2; bull_count += 1
    elif m['momentum_accel'] and m['momentum_dir'] < 0:
        total += 2; bear_count += 1
    if m['vsr'] >= 1.3 and c > o:
        total += 2; bull_count += 1
    elif m['vsr'] >= 1.3 and c < o:
        total += 2; bear_count += 1
    if m['vol_expand'] and c > o:
        total += 1; bull_count += 1
    elif m['vol_expand'] and c < o:
        total += 1; bear_count += 1

    direction = 'BULL' if bull_count >= bear_count and bull_count > 0 else ('BEAR' if bear_count > 0 else 'NEUTRAL')
    return total, direction, m, {}


# ======================================================================
# 激进通道
# ======================================================================

def check_aggressive_channel(close, high, low, open_, atr14, vsr, idx):
    if idx < 5:
        return False, None
    cur_vsr = vsr[idx] if idx < len(vsr) else 1.0
    body = abs(close[idx] - open_[idx])
    # 15m回测：比规范收紧，补偿无法模拟盘中入场
    is_massive = body >= 1.5 * atr14[idx] if atr14[idx] > 0 else False
    is_vsr_surge = cur_vsr >= 2.5

    if not (is_massive and is_vsr_surge):
        return False, None

    # 突破前10根极值
    prev_highs = high[max(0, idx - 10):idx]
    prev_lows = low[max(0, idx - 10):idx]
    if len(prev_highs) > 0 and close[idx] > np.max(prev_highs):
        return True, 'LONG'
    if len(prev_lows) > 0 and close[idx] < np.min(prev_lows):
        return True, 'SHORT'
    return False, None


# ======================================================================
# P1查找 — 真实回调波谷
# ======================================================================

def find_p1_support(close, high, low, volume, atr14, atr7, idx, lookback=60):
    """
    寻找真实回调低点（修复版）
    条件：1) 有明确上行：前20根涨>1.5%
          2) 回调深度 > 1.2%（原0.5%→提高2.4倍）
          3) 反弹验证：0.8×ATR7（原0.5×ATR7）
          4) 缩量验证：回调区成交量 < 均值60%
    返回 P1价格，或None
    """
    # 趋势背景检查：前20根涨超1.5%
    prior_idx = max(0, idx - 20)
    prior_rise = (close[idx] - close[prior_idx]) / max(close[prior_idx], EPS) * 100
    if prior_rise < 1.5:
        return None

    start = max(0, idx - lookback)

    for search_idx in range(idx - 2, start, -1):
        if search_idx < 2:
            continue
        # 局部极值低点
        w_low = np.min(low[max(0, search_idx - 2):search_idx + 3])
        if low[search_idx] != w_low:
            continue

        p1 = low[search_idx]
        high_after = np.max(high[search_idx + 1:idx + 1]) if search_idx + 1 <= idx else high[search_idx]

        # 回调深度 > 1.2%
        retrace_pct = (high_after - p1) / max(p1, EPS) * 100
        if retrace_pct < 1.2:
            continue

        # 反弹验证：0.8×ATR7
        if search_idx + 3 < idx:
            bounce = np.max(close[search_idx + 1:min(search_idx + 4, idx + 1)]) - p1
            if atr7[search_idx] > 0 and bounce < 0.8 * atr7[search_idx]:
                continue
        else:
            continue

        # 缩量验证：回调区成交量 < 均值60%
        vol_start = max(start, search_idx - 10)
        avg_vol_before = np.mean(volume[vol_start:search_idx + 1]) if search_idx > vol_start else 0
        if avg_vol_before > 0:
            vol_after = np.mean(volume[search_idx + 1:idx + 1]) if search_idx + 1 <= idx else volume[search_idx]
            if vol_after > avg_vol_before * 0.6:
                continue  # 未缩量，不是有效回调

        return float(p1)

    return None


def find_p1_resistance(close, high, low, volume, atr14, atr7, idx, lookback=60):
    """
    寻找真实反弹高点（空头P1，修复版）
    """
    prior_idx = max(0, idx - 20)
    prior_drop = (close[prior_idx] - close[idx]) / max(close[prior_idx], EPS) * 100
    if prior_drop < 1.5:
        return None

    start = max(0, idx - lookback)
    for search_idx in range(idx - 2, start, -1):
        if search_idx < 2:
            continue
        w_high = np.max(high[max(0, search_idx - 2):search_idx + 3])
        if high[search_idx] != w_high:
            continue

        p1 = high[search_idx]
        low_after = np.min(low[search_idx + 1:idx + 1]) if search_idx + 1 <= idx else low[search_idx]
        retrace_pct = (p1 - low_after) / max(p1, EPS) * 100
        if retrace_pct < 1.2:
            continue

        if search_idx + 3 < idx:
            bounce = p1 - np.min(close[search_idx + 1:min(search_idx + 4, idx + 1)])
            if atr7[search_idx] > 0 and bounce < 0.8 * atr7[search_idx]:
                continue
        else:
            continue

        vol_start = max(start, search_idx - 10)
        avg_vol_before = np.mean(volume[vol_start:search_idx + 1]) if search_idx > vol_start else 0
        if avg_vol_before > 0:
            vol_after = np.mean(volume[search_idx + 1:idx + 1]) if search_idx + 1 <= idx else volume[search_idx]
            if vol_after > avg_vol_before * 0.6:
                continue

        return float(p1)

    return None


# ======================================================================
# MTF检查
# ======================================================================

def check_mtf(sig_type: str, dir1h: int, dir4h: int) -> Tuple[bool, float]:
    if sig_type in ('TR_LONG',):
        return (dir1h != 2 and dir4h == 1), 0.0
    if sig_type in ('TR_SHORT',):
        return (dir1h != 1 and dir4h == 2), 0.0
    if sig_type in ('BO_LONG',):
        is_bull = dir1h != 2 and dir4h == 1
        return True, 1.0 if is_bull else -1.0
    if sig_type in ('BO_SHORT',):
        is_bear = dir1h != 1 and dir4h == 2
        return True, 1.0 if is_bear else -1.0
    is_bull = dir1h != 2 and dir4h == 1
    is_bear = dir1h != 1 and dir4h == 2
    bonus = 0.5 if (is_bull or is_bear) else -0.5
    return True, bonus


# ======================================================================
# 新评分系统 — BaseScore × 证据加权
# ======================================================================

def get_base_score(sig_type: str, trend: MarketTrendType) -> float:
    is_trend = trend in (MarketTrendType.UPTREND, MarketTrendType.DOWNTREND)
    is_range = trend == MarketTrendType.RANGEBOUND
    if sig_type.startswith('BO'):
        return 6.0 if is_range else 5.0
    if sig_type.startswith('TR'):
        return 6.5 if is_trend else 5.0
    if sig_type.startswith('RB'):
        return 6.0 if is_range else 4.0
    return 5.0


def evidence_multipliers(evidence_metrics: dict, direction: str) -> Tuple[float, float, float, float, float]:
    """
    5项证据加权乘数
    返回 (m1_structure, m2_morphology, m3_momentum, m4_volume, m5_volatility)
    """
    m1 = 1.2 if evidence_metrics.get('structure_score', 0) >= 3 else 1.0
    m2 = 1.1 if evidence_metrics.get('strong_body', False) else 1.0
    m3 = 1.1 if evidence_metrics.get('momentum_accel', False) else 1.0
    m4 = 1.15 if evidence_metrics.get('vsr', 1.0) >= 1.3 else 1.0
    m5 = 1.05 if evidence_metrics.get('vol_expand', False) else 1.0
    return m1, m2, m3, m4, m5


def calculate_score_v83(
    sig_type: str, trend: MarketTrendType, evidence_metrics: dict,
    vsr: float, rsi: float, adx: float, r2: float,
    mtf_bonus: float, bar_idx: int, atr_pct: float, evidence_dir: str
) -> Tuple[float, int, float]:
    if bar_idx < 14:
        return 0.0, 0, 0.0

    warmup = bar_idx < 20
    base_score = get_base_score(sig_type, trend)
    m1, m2, m3, m4, m5 = evidence_multipliers(evidence_metrics, evidence_dir)

    score = base_score * m1 * m2 * m3 * m4 * m5 + mtf_bonus

    # 条件拮抗
    if vsr >= 2.0 and adx < 20:
        score *= 0.75

    score = max(1.0, min(10.0, score))

    base_lev = 15
    dynamic_max_lev = min(30, max(10, 30 / max(0.5, atr_pct * 100)))
    lev = int(max(1, min(dynamic_max_lev, base_lev * np.sqrt(score / 10))))
    position_ratio = (score / 10) * 0.02

    if warmup:
        lev = int(lev * 0.5)
        position_ratio *= 0.5

    return round(score, 2), lev, position_ratio


# ======================================================================
# 信号冲突仲裁
# ======================================================================

def arbitrate_signals(signals: List[V83Signal]) -> Optional[V83Signal]:
    if not signals:
        return None
    directions = set(s.direction for s in signals)
    if len(directions) > 1:
        return None

    tri = [s for s in signals if s.signal_type in (V83SignalType.TRI_LONG, V83SignalType.TRI_SHORT)]
    if tri:
        return tri[0]

    # 优先级: RB > BO > TR
    for st in (V83SignalType.RB_LONG, V83SignalType.RB_SHORT,
               V83SignalType.BO_LONG, V83SignalType.BO_SHORT):
        match = [s for s in signals if s.signal_type == st]
        if match:
            return match[0]

    return signals[0]
