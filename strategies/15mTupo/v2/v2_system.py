"""
v2 系统核心：完全按照新开发文档实现
- 数据清洗：影线比值>=3丢弃尖端
- S/R计分系统：影线+1/实体+2，24h滑动窗口，Weighted_Score>=3激活
- 防抖机制：<0.2%沿用旧位
- 形态分类：120根K线分形→线性回归→斜率分类（RECTANGLE/三角形/趋势）
- 策略7条件决策树
- 硬编码禁止：ATR>2.5x熔断、三角形末端1/3禁区、有效位<2个
"""
import numpy as np
from enum import Enum
from collections import OrderedDict
import time as _time

class TrendType(Enum):
    UPTREND = 'UPTREND'
    DOWNTREND = 'DOWNTREND'
    RECTANGLE = 'RECTANGLE'
    SYM_TRIANGLE = 'SYM_TRIANGLE'
    ASC_TRIANGLE = 'ASC_TRIANGLE'
    DESC_TRIANGLE = 'DESC_TRIANGLE'
    UNKNOWN = 'UNKNOWN'

class SignalType(Enum):
    REBOUND_LONG = 'REBOUND_LONG'
    REBOUND_SHORT = 'REBOUND_SHORT'
    BREAKOUT_LONG = 'BREAKOUT_LONG'
    BREAKOUT_SHORT = 'BREAKOUT_SHORT'
    PULLBACK_LONG = 'PULLBACK_LONG'
    PULLBACK_SHORT = 'PULLBACK_SHORT'
    TRI_BREAK = 'TRI_BREAK'
    WAIT = 'WAIT'

EPS = 1e-12

# ======================================================================
# 第一章：数据清洗 — 影线过滤
# ======================================================================
def clean_wicks(o, h, l, c):
    """毛刺过滤：影线/实体 >= 3 时丢弃尖端"""
    body = np.abs(c - o)
    body = np.where(body < EPS, EPS, body)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    h_clean = h.copy()
    l_clean = l.copy()
    mask_up = upper_wick / body >= 3
    mask_dn = lower_wick / body >= 3
    h_clean[mask_up] = np.maximum(o[mask_up], c[mask_up])
    l_clean[mask_dn] = np.minimum(o[mask_dn], c[mask_dn])
    return h_clean, l_clean

# ======================================================================
# 第二~四章：S/R计分系统
# ======================================================================
class SRLevel:
    __slots__ = ('price', 'wick_hits', 'body_hits', 'weighted_score',
                 'first_hit_time', 'last_hit_time')
    def __init__(self, price, t):
        self.price = price
        self.wick_hits = 0
        self.body_hits = 0
        self.weighted_score = 0.0
        self.first_hit_time = t
        self.last_hit_time = t

class SRScoringEngine:
    """分形极值点驱动的S/R系统
    候选池 = 分形极值点（由形态分类提供），不再记录任意价位
    t参数为15m bar index，window_bars=96即24h"""
    def __init__(self, window_bars=96, hit_tolerance_pct=0.001,
                 min_interval_bars=4, activate_threshold=3.0,
                 debounce_pct=0.002):
        self.window_bars = window_bars
        self.hit_tol_pct = hit_tolerance_pct
        self.min_interval = min_interval_bars
        self.activate_thr = activate_threshold
        self.debounce_pct = debounce_pct
        # key=price, val=[wick_hits, body_hits, score, last_bar, first_bar]
        self.records = {}
        self.prev_support = None
        self.prev_resistance = None

    def set_fractal_levels(self, fractal_highs, fractal_lows, current_bar):
        """从分形极值点设置S/R候选池
        fractal_highs: [(bar_idx, price), ...]
        fractal_lows: [(bar_idx, price), ...]
        只添加新的极值点，已有的保留分数"""
        for bar_idx, price in fractal_highs:
            pk = self._round_key(price)
            if pk not in self.records:
                self.records[pk] = [0, 0, 0.0, float(current_bar), float(bar_idx)]
        for bar_idx, price in fractal_lows:
            pk = self._round_key(price)
            if pk not in self.records:
                self.records[pk] = [0, 0, 0.0, float(current_bar), float(bar_idx)]

    def _round_key(self, price):
        if price > 100:
            return round(price, 1)
        elif price > 1:
            return round(price, 2)
        else:
            return round(price, 4)

    def on_bar(self, t, high, low, close, open_):
        """只检查当前bar的H/L/C是否命中已有候选池中的极值点"""
        for price, is_body in ((high, False), (low, False), (close, True)):
            pk = self._find_nearby(price)
            if pk is not None and pk in self.records:
                rec = self.records[pk]
                if t - rec[3] < self.min_interval:
                    rec[3] = t
                    continue
                if is_body:
                    rec[1] += 1
                    rec[2] += 2
                else:
                    rec[0] += 1
                    rec[2] += 1
                rec[3] = t
        # 清理过期记录
        cutoff = t - self.window_bars
        to_del = [k for k, v in self.records.items() if v[4] < cutoff]
        for k in to_del:
            del self.records[k]

    def _find_nearby(self, price):
        """在±0.1%容差内查找最近的候选极值点，找不到返回None"""
        tol = price * self.hit_tol_pct
        best_key = None
        best_diff = tol
        for pk in self.records:
            diff = abs(pk - price)
            if diff < best_diff:
                best_diff = diff
                best_key = pk
        return best_key

    def get_levels(self, current_price):
        if not self.records:
            return None, None, 0
        sup = None
        res = None
        cnt = 0
        for pk, rec in self.records.items():
            if rec[2] >= self.activate_thr:
                cnt += 1
                if pk < current_price:
                    if sup is None or pk > sup:
                        sup = pk
                elif pk > current_price:
                    if res is None or pk < res:
                        res = pk
        if sup is not None and self.prev_support is not None:
            if abs(sup - self.prev_support) / max(self.prev_support, EPS) < self.debounce_pct:
                sup = self.prev_support
        if res is not None and self.prev_resistance is not None:
            if abs(res - self.prev_resistance) / max(self.prev_resistance, EPS) < self.debounce_pct:
                res = self.prev_resistance
        self.prev_support = sup
        self.prev_resistance = res
        return sup, res, cnt

# ======================================================================
# 第五章：市场背景形态分类
# ======================================================================
def _linear_regression(x, y):
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0
    sx = np.sum(x); sy = np.sum(y)
    sxx = np.sum(x * x); sxy = np.sum(x * y)
    denom = n * sxx - sx * sx
    if abs(denom) < EPS:
        return 0.0, sy / max(n, 1), 0.0
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    y_var = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / max(y_var, EPS) if y_var > EPS else 0.0
    return m, b, max(r2, 0.0)

def _frac_extrema(high, low, left=2, right=2, start=0, end=None):
    """分形极值点提取：左2右2，支持start/end范围"""
    n = end if end is not None else len(high)
    highs = []
    lows = []
    for i in range(max(left, start), min(n - right, len(high))):
        is_high = True
        is_low = True
        for j in range(1, left + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_high = False
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_low = False
        if is_high:
            highs.append((i, high[i]))
        if is_low:
            lows.append((i, low[i]))
    return highs, lows

def classify_morphology(high_arr, low_arr, ci, lookback=120):
    """第五章：形态分类（每1小时全量运算一次）"""
    start = max(0, ci - lookback + 1)
    h_seg = high_arr[start:ci + 1]
    l_seg = low_arr[start:ci + 1]
    if len(h_seg) < 20:
        return TrendType.UNKNOWN, 0.0, 0.0
    highs, lows = _frac_extrema(h_seg, l_seg, left=2, right=2, start=0, end=len(h_seg))
    if len(highs) < 2 or len(lows) < 2:
        return TrendType.UNKNOWN, 0.0, 0.0
    h_idx = np.array([x[0] for x in highs], dtype=float)
    h_val = np.array([x[1] for x in highs])
    l_idx = np.array([x[0] for x in lows], dtype=float)
    l_val = np.array([x[1] for x in lows])
    slope_h, _, r2_h = _linear_regression(h_idx, h_val)
    slope_l, _, r2_l = _linear_regression(l_idx, l_val)
    norm = np.mean(h_val) if np.mean(h_val) > EPS else 1.0
    slope_h_pct = slope_h / norm
    slope_l_pct = slope_l / norm
    abs_sh = abs(slope_h_pct)
    abs_sl = abs(slope_l_pct)
    if abs_sh < 0.3 and abs_sl < 0.3:
        return TrendType.RECTANGLE, slope_h_pct, slope_l_pct
    if slope_h_pct < -0.5 and slope_l_pct > 0.5:
        return TrendType.SYM_TRIANGLE, slope_h_pct, slope_l_pct
    if abs_sh < 0.3 and slope_l_pct > 0.5:
        return TrendType.ASC_TRIANGLE, slope_h_pct, slope_l_pct
    if slope_h_pct < -0.5 and abs_sl < 0.3:
        return TrendType.DESC_TRIANGLE, slope_h_pct, slope_l_pct
    if slope_h_pct > 0.5 and slope_l_pct > 0.5:
        return TrendType.UPTREND, slope_h_pct, slope_l_pct
    if slope_h_pct < -0.5 and slope_l_pct < -0.5:
        return TrendType.DOWNTREND, slope_h_pct, slope_l_pct
    return TrendType.UNKNOWN, slope_h_pct, slope_l_pct

# ======================================================================
# 第六章 + 第七章：策略决策树 + 硬编码禁止
# ======================================================================
class V2Signal:
    __slots__ = ('signal_type', 'side', 'sl', 'tp', 'entry_price', 'support',
                 'resistance', 'morphology', 'sr_count', 'reason')
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

def _calc_sl_long(entry, support, atr, morph):
    if morph == TrendType.RECTANGLE and support is not None:
        return support - 0.5 * atr
    return entry - 0.5 * atr

def _calc_sl_short(entry, resistance, atr, morph):
    if morph == TrendType.RECTANGLE and resistance is not None:
        return resistance + 0.5 * atr
    return entry + 0.5 * atr

def generate_signal(ci15, c15, h15, l15, o15, v15, atr, ema20,
                     support, resistance, sr_count, morph, slope_h, slope_l,
                     atr_100_mean, triangle_span=None, triangle_pos_ratio=None):
    """
    第六章决策树 + 第七章硬编码禁止
    返回 V2Signal 或 SignalType.WAIT
    """
    price = c15[ci15]
    cur_atr = atr[ci15] if ci15 < len(atr) else atr[-1]
    if cur_atr < EPS:
        return SignalType.WAIT

    # 第七章·熔断1：ATR > 2.5x
    if atr_100_mean > EPS and cur_atr > atr_100_mean * 2.5:
        return SignalType.WAIT

    # 第七章·数据稀疏
    if sr_count < 2:
        return SignalType.WAIT

    # 三角形末尾禁区（仅前2/3有效）
    if triangle_pos_ratio is not None and triangle_pos_ratio > 2.0 / 3.0:
        if morph in (TrendType.SYM_TRIANGLE, TrendType.ASC_TRIANGLE, TrendType.DESC_TRIANGLE):
            return SignalType.WAIT

    cur_ema = ema20[ci15] if ci15 < len(ema20) else price

    # ---- RECTANGLE 策略 ----
    if morph == TrendType.RECTANGLE:
        if support is not None and resistance is not None:
            zone_height = resistance - support
            if zone_height < EPS:
                return SignalType.WAIT
            # 区间太窄不反弹，只允许突破
            zone_too_narrow = zone_height < 1.5 * cur_atr
            if not zone_too_narrow:
                # 条件1：REBOUND_LONG
                if price <= support + 0.3 * cur_atr and price >= support:
                    sl = support - 0.5 * cur_atr
                    return V2Signal(signal_type=SignalType.REBOUND_LONG, side='LONG',
                                    sl=sl, tp=resistance, entry_price=price,
                                    support=support, resistance=resistance,
                                    morphology=morph, sr_count=sr_count, reason='矩形反弹多')
                # 条件2：REBOUND_SHORT
                if price >= resistance - 0.3 * cur_atr and price <= resistance:
                    sl = resistance + 0.5 * cur_atr
                    return V2Signal(signal_type=SignalType.REBOUND_SHORT, side='SHORT',
                                    sl=sl, tp=support, entry_price=price,
                                    support=support, resistance=resistance,
                                    morphology=morph, sr_count=sr_count, reason='矩形反弹空')
            # 条件3：BREAKOUT_LONG
            if price > resistance + 0.3 * cur_atr:
                vol_arr = v15[max(0, ci15 - 19):ci15 + 1]
                vol_ma = np.mean(vol_arr) if len(vol_arr) > 0 else 0
                vol_std = np.std(vol_arr) if len(vol_arr) > 1 else 0
                if v15[ci15] > vol_ma + 2 * vol_std:
                    sl = resistance - 0.3 * cur_atr
                    return V2Signal(signal_type=SignalType.BREAKOUT_LONG, side='LONG',
                                    sl=sl, tp=price + zone_height, entry_price=price,
                                    support=support, resistance=resistance,
                                    morphology=morph, sr_count=sr_count, reason='矩形突破多')
            # 条件4：BREAKOUT_SHORT
            if price < support - 0.3 * cur_atr:
                vol_arr = v15[max(0, ci15 - 19):ci15 + 1]
                vol_ma = np.mean(vol_arr) if len(vol_arr) > 0 else 0
                vol_std = np.std(vol_arr) if len(vol_arr) > 1 else 0
                if v15[ci15] > vol_ma + 2 * vol_std:
                    sl = support + 0.3 * cur_atr
                    return V2Signal(signal_type=SignalType.BREAKOUT_SHORT, side='SHORT',
                                    sl=sl, tp=price - zone_height, entry_price=price,
                                    support=support, resistance=resistance,
                                    morphology=morph, sr_count=sr_count, reason='矩形突破空')

    # ---- UPTREND 策略 ----
    if morph == TrendType.UPTREND:
        # 条件5：PULLBACK_LONG
        if support is not None:
            dist_to_support = abs(price - support)
            if dist_to_support < 0.3 * cur_atr and price >= support:
                sl = support - 0.5 * cur_atr
                return V2Signal(signal_type=SignalType.PULLBACK_LONG, side='LONG',
                                sl=sl, tp=cur_ema, entry_price=price,
                                support=support, resistance=resistance,
                                morphology=morph, sr_count=sr_count, reason='趋势回调多')
        if abs(price - cur_ema) < 0.3 * cur_atr:
            recent_low = np.min(l15[max(0, ci15 - 5):ci15 + 1])
            sl = recent_low - 0.5 * cur_atr
            return V2Signal(signal_type=SignalType.PULLBACK_LONG, side='LONG',
                            sl=sl, tp=cur_ema, entry_price=price,
                            support=support, resistance=resistance,
                            morphology=morph, sr_count=sr_count, reason='趋势EMA回调多')

    # ---- DOWNTREND 策略 ----
    if morph == TrendType.DOWNTREND:
        if resistance is not None:
            dist_to_res = abs(price - resistance)
            if dist_to_res < 0.3 * cur_atr and price <= resistance:
                sl = resistance + 0.5 * cur_atr
                return V2Signal(signal_type=SignalType.PULLBACK_SHORT, side='SHORT',
                                sl=sl, tp=cur_ema, entry_price=price,
                                support=support, resistance=resistance,
                                morphology=morph, sr_count=sr_count, reason='趋势回调空')
        if abs(price - cur_ema) < 0.3 * cur_atr:
            recent_high = np.max(h15[max(0, ci15 - 5):ci15 + 1])
            sl = recent_high + 0.5 * cur_atr
            return V2Signal(signal_type=SignalType.PULLBACK_SHORT, side='SHORT',
                            sl=sl, tp=cur_ema, entry_price=price,
                            support=support, resistance=resistance,
                            morphology=morph, sr_count=sr_count, reason='趋势EMA回调空')

    # ---- TRIANGLE 策略 ----
    if morph in (TrendType.SYM_TRIANGLE, TrendType.ASC_TRIANGLE, TrendType.DESC_TRIANGLE):
        if triangle_span is not None and len(triangle_span) == 2:
            tri_start, tri_end = triangle_span
            tri_len = tri_end - tri_start
            if tri_len > 0:
                pos_ratio = (ci15 - tri_start) / tri_len
                if pos_ratio > 2.0 / 3.0:
                    return SignalType.WAIT
        if resistance is not None and price > resistance:
            sl = resistance - 0.3 * cur_atr
            tp = price + (price - (support if support else price - cur_atr))
            return V2Signal(signal_type=SignalType.TRI_BREAK, side='LONG',
                            sl=sl, tp=tp, entry_price=price,
                            support=support, resistance=resistance,
                            morphology=morph, sr_count=sr_count, reason='三角形突破多')
        if support is not None and price < support:
            sl = support + 0.3 * cur_atr
            tp = price - ((resistance if resistance else price + cur_atr) - price)
            return V2Signal(signal_type=SignalType.TRI_BREAK, side='SHORT',
                            sl=sl, tp=tp, entry_price=price,
                            support=support, resistance=resistance,
                            morphology=morph, sr_count=sr_count, reason='三角形突破空')

    return SignalType.WAIT
