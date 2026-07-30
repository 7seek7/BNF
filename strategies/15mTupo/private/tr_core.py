# -*- coding: utf-8 -*-
"""
TR信号核心逻辑 - 基于TR建议文档实现
包含：基石线、操作线、触碰次数、缩量验证、止跌形态
"""
import numpy as np
from typing import Optional, Tuple, Dict, List


class TrendLineState:
    """趋势线状态管理"""
    
    def __init__(self):
        self.foundation_line = None  # 基石线：永久固定
        self.operating_line = None   # 操作线：动态调整
        self.touch_count = 0         # 操作线触碰次数
        self.last_adjust_idx = -100  # 最后调整操作线的idx
        self.last_high = 0.0         # 用于检测新高
        
    def to_dict(self) -> dict:
        return {
            'foundation': self.foundation_line,
            'operating': self.operating_line,
            'touch_count': self.touch_count,
            'last_adjust_idx': self.last_adjust_idx,
        }


def create_trend_line(idx: int, price: float, slope: float = None) -> dict:
    """创建趋势线"""
    return {'idx': idx, 'price': price, 'slope': slope}


def update_operating_line(
    state: TrendLineState,
    new_idx: int,
    new_price: float,
    current_high: float,
    min_gap: int = 5
) -> bool:
    """
    更新操作线
    约束条件：
    1. 每5根K线内最多调整1次
    2. 新线斜率必须比旧线更陡峭
    3. 价格创新高时才更新
    """
    # 检查是否创新高
    if current_high <= state.last_high:
        return False
    
    state.last_high = current_high
    
    # 检查调整频率
    if new_idx - state.last_adjust_idx < min_gap:
        return False
    
    # 计算新斜率
    if state.foundation_line:
        foundation = state.foundation_line
        new_slope = (new_price - foundation['price']) / (new_idx - foundation['idx'])
        
        # 检查斜率是否更陡峭
        if state.operating_line and state.operating_line.get('slope') is not None:
            old_slope = state.operating_line['slope']
            if new_slope <= old_slope:
                return False
    
    # 更新操作线
    state.operating_line = create_trend_line(new_idx, new_price, new_slope if state.foundation_line else None)
    state.last_adjust_idx = new_idx
    state.touch_count = 0  # 重置触碰次数
    return True


def count_touches(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    trend_line: dict,
    current_idx: int,
    lookback: int = 20,
    tolerance_pct: float = 0.015
) -> int:
    """
    统计趋势线触碰次数
    触碰条件：价格接近趋势线（距离≤tolerance_pct）
    """
    if not trend_line:
        return 0
    
    idx = trend_line['idx']
    price = trend_line['price']
    slope = trend_line.get('slope', 0)
    
    touches = 0
    start = max(0, current_idx - lookback)
    
    for i in range(start, current_idx):
        trend_price = price + slope * (i - idx) if slope else price
        dist = abs(low[i] - trend_price) / trend_price if trend_price > 0 else 999
        if dist <= tolerance_pct:
            touches += 1
    
    return touches


def check_volume_shrink(
    volume: np.ndarray,
    avg_volume: np.ndarray,
    current_idx: int,
    lookback: int = 10,
    shrink_ratio: float = 0.6
) -> bool:
    """
    缩量验证：回调时成交量 < 均量×shrink_ratio
    """
    if current_idx < lookback or current_idx >= len(volume):
        return False
    
    # 计算回调期间的平均成交量
    callback_vol = np.mean(volume[current_idx-lookback:current_idx])
    
    # 获取均量
    if current_idx < len(avg_volume):
        avg_vol = avg_volume[current_idx]
    else:
        avg_vol = np.mean(volume[max(0, current_idx-20):current_idx])
    
    if avg_vol <= 0:
        return False
    
    return callback_vol < avg_vol * shrink_ratio


def check_hammer_pattern(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    current_idx: int
) -> bool:
    """
    止跌形态：锤子线
    条件：阳线 + 下影线 > 实体×2
    """
    if current_idx >= len(close):
        return False
    
    o = open_[current_idx]
    h = high[current_idx]
    l = low[current_idx]
    c = close[current_idx]
    
    # 阳线
    if c <= o:
        return False
    
    body = c - o
    lower_shadow = o - l
    
    if body <= 0:
        return False
    
    return lower_shadow > body * 2


def check_doji_pattern(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    avg_volume: np.ndarray,
    current_idx: int,
    body_ratio: float = 0.1,
    vol_ratio: float = 0.6
) -> bool:
    """
    缩量十字星
    条件：实体很小 + 成交量缩小
    """
    if current_idx >= len(close):
        return False
    
    o = open_[current_idx]
    h = high[current_idx]
    l = low[current_idx]
    c = close[current_idx]
    v = volume[current_idx]
    
    rng = h - l
    if rng <= 0:
        return False
    
    body = abs(c - o)
    body_pct = body / rng
    
    # 成交量缩小
    if current_idx < len(avg_volume) and avg_volume[current_idx] > 0:
        vol_shrink = v < avg_volume[current_idx] * vol_ratio
    else:
        vol_shrink = True
    
    return body_pct < body_ratio and vol_shrink


def check_tr_entry_conditions(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    volume: np.ndarray,
    avg_volume: np.ndarray,
    atr: np.ndarray,
    current_idx: int,
    trend_line: dict,
    side: str = 'LONG'
) -> Tuple[bool, str]:
    """
    TR入场四重共振检查
    返回：(是否满足, 原因)
    """
    if not trend_line:
        return False, "无趋势线"
    
    # 1. 空间到位：收盘价距趋势线 ≤ ATR×0.15
    idx = trend_line['idx']
    price = trend_line['price']
    slope = trend_line.get('slope', 0)
    trend_price = price + slope * (current_idx - idx) if slope else price
    
    if trend_price <= 0:
        return False, "趋势线价格无效"
    
    dist = (close[current_idx] - trend_price) / trend_price
    atrv = atr[current_idx] if current_idx < len(atr) else 0
    atr_thresh = (atrv / close[current_idx] * 0.15) if close[current_idx] > 0 and atrv > 0 else 0.015
    
    if side == 'LONG':
        if dist > atr_thresh and dist > 0.015:
            return False, f"距离过远: {dist*100:.2f}% > {atr_thresh*100:.2f}%"
    else:  # SHORT
        if dist < -atr_thresh and dist < -0.015:
            return False, f"距离过远: {dist*100:.2f}% < {-atr_thresh*100:.2f}%"
    
    # 2. 缩量验证
    if not check_volume_shrink(volume, avg_volume, current_idx):
        return False, "未缩量"
    
    # 3. 止跌形态
    if side == 'LONG':
        if not (check_hammer_pattern(open_, high, low, close, current_idx) or
                check_doji_pattern(open_, high, low, close, volume, avg_volume, current_idx)):
            return False, "无止跌形态"
    else:  # SHORT
        # 空头的止跌形态：阴线 + 上影线 > 实体×2
        o, h, l, c = open_[current_idx], high[current_idx], low[current_idx], close[current_idx]
        if c >= o:  # 需要阴线
            return False, "非阴线"
        body = o - c
        upper_shadow = h - o
        if body <= 0 or upper_shadow <= body * 2:
            return False, "无止跌形态"
    
    # 4. 触碰次数过滤（在外部处理）
    
    return True, "满足"


def check_foundation_break(
    close: np.ndarray,
    current_idx: int,
    foundation_line: dict
) -> bool:
    """
    基石线跌破检查
    价格实体（收盘价）跌破基石线，立即触发清仓
    """
    if not foundation_line:
        return False
    
    idx = foundation_line['idx']
    price = foundation_line['price']
    
    # 基石线是水平线（价格固定）
    return close[current_idx] < price
