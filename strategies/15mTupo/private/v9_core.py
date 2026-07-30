# -*- coding: utf-8 -*-
"""
v9_core.py — 统一结构驱动引擎（状态机版）
SwingFSM: FORMING→CANDIDATE→CONFIRMED→ARCHIVED
RegimeFSM: TREND_UP/DOWN ↔ TRANSITION ↔ RANGE/TRIANGLE
"""
import numpy as np
from enum import Enum


class SwingState(Enum):
    FORMING = "FORMING"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    INVALIDATED = "INVALIDATED"
    ARCHIVED = "ARCHIVED"


class MarketTrendType(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    CONSOLIDATION = "CONSOLIDATION"
    TRIANGLE = "TRIANGLE"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class ValidPivot:
    __slots__ = ('type', 'price', 'idx', 'strength', 'confirmed_time')
    def __init__(self, type, price, idx, strength, confirmed_time=-1):
        self.type = type
        self.price = price
        self.idx = idx
        self.strength = strength
        self.confirmed_time = confirmed_time


class SwingPoint:
    """单个摆动点的状态机 — FORMING→CANDIDATE→CONFIRMED/INVALIDATED→ARCHIVED"""
    __slots__ = ('type', 'state', 'extreme_price', 'extreme_idx',
                 'candidate_time', 'confirmed_time', 'invalidated_reason',
                 'atr_val', 'strength')
    def __init__(self, point_type, extreme_price, extreme_idx, atr_val):
        self.type = point_type
        self.state = SwingState.FORMING
        self.extreme_price = extreme_price
        self.extreme_idx = extreme_idx
        self.candidate_time = -1
        self.confirmed_time = -1
        self.invalidated_reason = None
        self.atr_val = atr_val
        self.strength = 0.0

    def update(self, high, low, idx):
        """PRICE_UPDATE事件 + 自动检测阈值/覆盖"""
        if self.state not in (SwingState.FORMING, SwingState.CANDIDATE):
            return
        threshold = 0.5 * self.atr_val
        if self.type == 'HIGH':
            retrace = self.extreme_price - low
            if high > self.extreme_price and idx - self.extreme_idx > 3:
                self.state = SwingState.INVALIDATED
                self.invalidated_reason = 'EXTREME_OVERRIDDEN'
                return
        else:
            retrace = high - self.extreme_price
            if low < self.extreme_price and idx - self.extreme_idx > 3:
                self.state = SwingState.INVALIDATED
                self.invalidated_reason = 'EXTREME_OVERRIDDEN'
                return
        if self.state == SwingState.FORMING and retrace >= threshold:
            self.state = SwingState.CANDIDATE
            self.candidate_time = idx


class StructureSnapshot:
    """规格书3.3: 不可变结构快照 — 每根15m K线闭合后冻结"""
    __slots__ = ('bar_idx', 'trend', 'triangle', 'pivots',
                 'p1', 'p1_idx', 'trend_start_price', 'trend_start_idx',
                 'forming_count')
    def __init__(self, bar_idx, trend, triangle, pivots,
                 p1, p1_idx, trend_start_price, trend_start_idx,
                 forming_count):
        self.bar_idx = bar_idx
        self.trend = trend
        self.triangle = triangle
        self.pivots = tuple(pivots)
        self.p1 = p1
        self.p1_idx = p1_idx
        self.trend_start_price = trend_start_price
        self.trend_start_idx = trend_start_idx
        self.forming_count = forming_count


class StructureEngine:
    def __init__(self, atr7=None, atr14=None, max_pivots=10, vol_arr=None):
        self.atr7 = atr7
        self.atr14 = atr14
        self._max_pivots = max_pivots
        self.vol_arr = vol_arr
        # ConfirmedSwingDeque：只包含CONFIRMED点，最多10个
        self.pivots = []
        # FORMING/CANDIDATE追踪
        self._forming = []
        # 当前市场状态
        self.current_trend = MarketTrendType.UNKNOWN
        self._prev_trend = MarketTrendType.UNKNOWN
        self._transition_bar = -1  # 状态切换时的bar index
        # 三角形
        self.triangle = None
        self._triangle_frozen = False
        # prev_close追踪（三角形解冻用）
        self._prev_close = None
        # TREND_UP专用：P1追踪
        self.p1 = None
        self.p1_idx = None
        self.trend_start_idx = None
        self.trend_start_price = None
        # 快照管理
        self.snapshots = []  # 保留最近10个快照
        self._last_snapshot_bar = -1

    # ────────────── 快照接口 ──────────────

    def get_snapshot(self):
        """获取最新的结构快照（规格书3.3）"""
        return self.snapshots[-1] if self.snapshots else None

    def get_snapshots(self, n=5):
        """获取最近n个快照"""
        return list(self.snapshots[-n:]) if self.snapshots else []

    def _freeze_snapshot(self, bar_idx):
        """创建当前状态的不可变快照"""
        snap = StructureSnapshot(
            bar_idx=bar_idx,
            trend=self.current_trend,
            triangle=self.triangle,
            pivots=list(self.pivots),
            p1=self.p1, p1_idx=self.p1_idx,
            trend_start_price=self.trend_start_price,
            trend_start_idx=self.trend_start_idx,
            forming_count=len(self._forming),
        )
        self.snapshots.append(snap)
        if len(self.snapshots) > 10:
            self.snapshots.pop(0)

    # ────────────── 公开接口（与旧版兼容）──────────────

    def get_trend(self):
        return self.current_trend

    def get_triangle(self):
        return self.triangle

    def get_pivots(self):
        return self.pivots

    def get_p1(self, direction=None):
        if direction == 'LONG' and self.current_trend == MarketTrendType.UPTREND:
            return self.p1
        if direction == 'SHORT' and self.current_trend == MarketTrendType.DOWNTREND:
            return self.p1
        return None

    def get_trend_start(self):
        return self.trend_start_price, self.trend_start_idx

    def get_box(self):
        if self.current_trend not in (MarketTrendType.CONSOLIDATION, MarketTrendType.TRANSITION):
            return None
        if len(self.pivots) < 4:
            return None
        highs = [p.price for p in self.pivots if p.type == 'HIGH']
        lows = [p.price for p in self.pivots if p.type == 'LOW']
        if not highs or not lows:
            return None
        return {
            'top_upper': max(highs),
            'top_lower': min(highs),
            'bottom_upper': max(lows),
            'bottom_lower': min(lows),
            'range_high': (max(highs) + min(highs)) / 2,
            'range_low': (max(lows) + min(lows)) / 2,
            'width': max(highs) - min(lows),
        }

    # ────────────── 核心入口 ──────────────

    def add_bar(self, idx, high, low, close, h_arr, l_arr):
        if idx < 2:
            return
        # 1. 更新所有FORMING/CANDIDATE点
        for sp in self._forming[:]:
            sp.update(high, low, idx)
            if sp.state == SwingState.INVALIDATED:
                self._forming.remove(sp)
            elif sp.state == SwingState.CANDIDATE:
                self._try_confirm(sp, idx, high, low)

        # 2. 检测新极值点
        is_high = (high > h_arr[idx-1] and high > h_arr[idx-2] and
                   high > h_arr[idx+1] and high > h_arr[idx+2]) if idx < len(h_arr)-2 else False
        is_low = (low < l_arr[idx-1] and low < l_arr[idx-2] and
                  low < l_arr[idx+1] and low < l_arr[idx+2]) if idx < len(l_arr)-2 else False
        if not is_high and not is_low:
            self._update_trend(idx, close, high)
            self._freeze_snapshot(idx)
            return

        atr7_val = self.atr7[idx] if self.atr7 is not None and idx < len(self.atr7) else 0.01

        if is_low:
            sp = SwingPoint('LOW', low, idx, atr7_val)
            self._forming.append(sp)
            self._confirm_opposite('HIGH', idx)
        else:
            sp = SwingPoint('HIGH', high, idx, atr7_val)
            self._forming.append(sp)
            self._confirm_opposite('LOW', idx)

        if len(self._forming) > self._max_pivots * 2:
            self._forming = self._forming[-self._max_pivots * 2:]

        self._update_trend(idx, close, high)
        self._freeze_snapshot(idx)

    # ────────────── SwingFSM 内部 ──────────────

    def _confirm_opposite(self, opposite_type, idx):
        """新极值点确认所有同类型CANDIDATE（新LOW确认CANDIDATE HIGH，反之亦然）"""
        for sp in self._forming[:]:
            if sp.type == opposite_type and sp.state == SwingState.CANDIDATE:
                sp.state = SwingState.CONFIRMED
                sp.confirmed_time = idx
                vp = ValidPivot(sp.type, sp.extreme_price, sp.extreme_idx,
                                sp.strength, confirmed_time=idx)
                self.pivots.append(vp)
                if len(self.pivots) > self._max_pivots:
                    old = self.pivots.pop(0)
                self._forming.remove(sp)

    def _try_confirm(self, sp, idx, high, low):
        """检查CANDIDATE是否可被已存在的相反CONFIRMED点确认"""
        if sp.type == 'HIGH':
            for cp in self.pivots:
                if cp.type == 'LOW' and cp.idx > sp.extreme_idx and cp.idx <= idx:
                    sp.state = SwingState.CONFIRMED
                    sp.confirmed_time = idx
                    vp = ValidPivot(sp.type, sp.extreme_price, sp.extreme_idx,
                                    sp.strength, confirmed_time=idx)
                    self.pivots.append(vp)
                    if len(self.pivots) > self._max_pivots:
                        self.pivots.pop(0)
                    self._forming.remove(sp)
                    return
        else:
            for cp in self.pivots:
                if cp.type == 'HIGH' and cp.idx > sp.extreme_idx and cp.idx <= idx:
                    sp.state = SwingState.CONFIRMED
                    sp.confirmed_time = idx
                    vp = ValidPivot(sp.type, sp.extreme_price, sp.extreme_idx,
                                    sp.strength, confirmed_time=idx)
                    self.pivots.append(vp)
                    if len(self.pivots) > self._max_pivots:
                        self.pivots.pop(0)
                    self._forming.remove(sp)
                    return

    # ────────────── RegimeFSM ──────────────

    def _update_trend(self, idx, close, high):
        """市场状态机：三角形→趋势→震荡/过渡"""
        self._prev_trend = self.current_trend

        # 三角形冻结检查
        if self._triangle_frozen and self.triangle is not None:
            self.current_trend = MarketTrendType.TRIANGLE
            return

        lows = [p for p in self.pivots if p.type == 'LOW']
        highs = [p for p in self.pivots if p.type == 'HIGH']

        if len(self.pivots) < 2:
            self.current_trend = MarketTrendType.UNKNOWN
            return

        # 三角形检测
        if len(highs) >= 3 and len(lows) >= 3:
            rh = highs[-3:]; rl = lows[-3:]
            if all(rh[i].price > rh[i+1].price for i in range(2)) and \
               all(rl[i].price < rl[i+1].price for i in range(2)):
                old_w = rh[0].price - rl[0].price
                new_w = rh[-1].price - rl[-1].price
                if old_w > 0 and new_w / old_w < 0.6:
                    self.current_trend = MarketTrendType.TRIANGLE
                    self.triangle = {
                        'top': rh[-1].price, 'top_idx': rh[-1].idx,
                        'bottom': rl[-1].price, 'bottom_idx': rl[-1].idx,
                        'top_slope': (rh[-1].price - rh[-2].price) / max(rh[-1].idx - rh[-2].idx, 1),
                        'bottom_slope': (rl[-1].price - rl[-2].price) / max(rl[-1].idx - rl[-2].idx, 1),
                    }
                    self._triangle_frozen = True
                    return

        # 趋势破坏检查（从趋势状态进入）
        if self._prev_trend in (MarketTrendType.UPTREND, MarketTrendType.DOWNTREND):
            trend_broken = self._check_trend_break(idx, close, high)
            if trend_broken:
                self.current_trend = MarketTrendType.TRANSITION
                self._transition_bar = idx
                self.p1 = None; self.p1_idx = None
                return

        # 冷却期：切换后3根K线内不切换（除非趋势破坏）
        if self.current_trend != self._prev_trend and self.current_trend != MarketTrendType.TRANSITION:
            if idx - self._transition_bar <= 3:
                self.current_trend = self._prev_trend
                return

        # 趋势判定
        # UPTREND: 至少2个依次抬高的CONFIRMED LOW
        if len(lows) >= 2:
            if lows[-1].price > lows[-2].price and lows[-1].idx - lows[-2].idx >= 4:
                self.current_trend = MarketTrendType.UPTREND
                self._transition_bar = idx
                if self.trend_start_idx is None:
                    self.trend_start_idx = lows[0].idx
                    self.trend_start_price = lows[0].price
                if len(lows) >= 2:
                    self.p1 = lows[-2].price
                    self.p1_idx = lows[-2].idx
                return

        if len(highs) >= 2:
            if highs[-1].price < highs[-2].price and highs[-1].idx - highs[-2].idx >= 4:
                self.current_trend = MarketTrendType.DOWNTREND
                self._transition_bar = idx
                if len(highs) >= 2:
                    self.p1 = highs[-2].price
                    self.p1_idx = highs[-2].idx
                return

        # CONSOLIDATION 或 TRANSITION
        if len(self.pivots) >= 4:
            self.current_trend = MarketTrendType.CONSOLIDATION
        else:
            self.current_trend = MarketTrendType.TRANSITION
        self.p1 = None; self.p1_idx = None

    def _check_trend_break(self, idx, close, high):
        """趋势破坏3条件（规格书4.5节）
        条件1：收盘价 < 最近HL（UPTREND）或 > 最近LH（DOWNTREND）
        条件2：跌破幅度 > 0.5×ATR14
        条件3：下一根K线收盘价未收回
        """
        if idx < 3 or self.atr14 is None or idx >= len(self.atr14):
            return False
        if self._prev_trend == MarketTrendType.UPTREND:
            lows = [p for p in self.pivots if p.type == 'LOW']
            if len(lows) < 2:
                return False
            recent_hl = lows[-2].price  # 倒数第二个低点=最近HL
            # 条件1
            if close[idx] < recent_hl:
                # 条件2
                breach = (recent_hl - close[idx]) / max(close[idx], 0.001) * 100
                if breach > 0.5 * (self.atr14[idx] / max(close[idx], 0.001) * 100):
                    # 条件3：上一根K线已跌破，本根未收回
                    if idx >= 1 and close[idx-1] < recent_hl and close[idx] < recent_hl:
                        return True
        elif self._prev_trend == MarketTrendType.DOWNTREND:
            highs = [p for p in self.pivots if p.type == 'HIGH']
            if len(highs) < 2:
                return False
            recent_lh = highs[-2].price
            if close[idx] > recent_lh:
                breach = (close[idx] - recent_lh) / max(close[idx], 0.001) * 100
                if breach > 0.5 * (self.atr14[idx] / max(close[idx], 0.001) * 100):
                    if idx >= 1 and close[idx-1] > recent_lh and close[idx] > recent_lh:
                        return True
        return False
