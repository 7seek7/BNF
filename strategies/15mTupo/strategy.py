# -*- coding: utf-8 -*-
"""
15mTupo 实盘策略 - 基于 run_final.py (权威版)
与回测共享核心位于 strategies/15mTupo/private/tupo_core.py
"""

import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from framework.strategy.base import StrategyBase, Signal, SignalType
from framework.business.kline_manager import KlineBar
from framework.core.config import StrategyConfig
from framework.core.logger import get_logger
import importlib
_tupo_core = importlib.import_module('strategies.15mTupo.private.tupo_core')
TrendType = _tupo_core.TrendType
PositionSide = _tupo_core.PositionSide
ExitState = _tupo_core.ExitState
_calc_score = _tupo_core._calc_score
calc_sl_lev = _tupo_core.calc_sl_lev
entry_ok = _tupo_core.entry_ok
quality_ok = _tupo_core.quality_ok
check_exit = _tupo_core.check_exit
calculate_adx_fast = _tupo_core.calculate_adx_fast
calculate_rsi = _tupo_core.calculate_rsi
calculate_avg_volume_fast = _tupo_core.calculate_avg_volume_fast
analyze_trend_fast = _tupo_core.analyze_trend_fast
env_float = _tupo_core.env_float
env_int = _tupo_core.env_int
set_atr_ma40_buffer = _tupo_core.set_atr_ma40_buffer

# ── 命名常量 ──────────────────────────────────────────────
ADX_THRESHOLD = 20              # ADX 趋势强度阈值
LOOKBACK_BARS = 20              # 高低点回看周期
MIN_BARS_BEFORE_ADX = 20        # ADX 可用最小 K 线数
RETEST_TIMEOUT_BARS = 40        # 回测超时判断周期
SMA5_LOOKBACK = 5               # 5 分钟 MA5 计算窗口
MIN_15M_BARS = 100              # analyze() 最小 15m K 线数
EPSILON = 1e-12                 # 除零保护 / 极小值
SMALL_NUM = 0.001               # 除法分母安全下限
ATR_DEFAULT_RATIO = 0.02        # ATR/价格 默认值
TRENDLINE_TOLERANCE_PCT = 0.3   # 趋势线触碰容差百分比
# MTF 方向判断阈值
MTF_UP_THRESHOLD = 1.005        # 收盘 > MA * 此值 = up
MTF_DOWN_THRESHOLD = 0.995      # 收盘 < MA * 此值 = down
# ──────────────────────────────────────────────────────────


@dataclass
class SignalCtx:
    """analyze() 信号检测上下文，减少参数传递"""
    symbol: str
    ti: TrendType
    info: dict
    ci: int
    cp: float
    vv: float
    c15s: float
    o15s: float
    h15s: float
    l15s: float
    v15s: float
    res: Optional[float]
    sup: Optional[float]
    cons_trend: str
    mtf_dir: str
    tf4d: int
    n5: int
    c5_above_ma5: bool
    c5_below_ma5: bool
    adx: np.ndarray
    rsi_vals: np.ndarray
    atr: np.ndarray
    avgv: np.ndarray
    ma20v: np.ndarray
    body15_arr: np.ndarray
    pos20_arr: np.ndarray
    rng15pct_arr: np.ndarray
    v5_arr: np.ndarray
    dt_short_touches: int
    cur_retest: Optional[dict]
    high_arr: np.ndarray
    low_arr: np.ndarray


class TupoStrategy(StrategyBase):
    def __init__(self):
        super().__init__()
        self._name = '15mTupo'
        self._lock = threading.RLock()
        self._retest_state = {}
        self._last_5m_high = 0.0
        self._last_5m_low = 0.0
        self._last_5m_close = 0.0
        self._cached_body15_arr = np.array([])
        self._cached_pos20_arr = np.array([])
        self._cached_rng15pct_arr = np.array([])
        self._cached_v5_arr = np.array([])
        self._last_signal_key = {}  # symbol → signal key，用于去重Telegram通知
        self._auto_trade = False  # 自働交易开关（默认关闭，Telegram消息带交易按钮）

    def get_intervals(self) -> List[str]:
        """返回策略所需K线周期"""
        return ['5m', '15m', '4h']

    def set_auto_trade(self, enabled: bool) -> bool:
        """设置自働交易开关"""
        self._auto_trade = enabled
        self.logger.info(f"自働交易已{'开启' if enabled else '关闭'}")
        return self._auto_trade

    def initialize(self, config: StrategyConfig) -> bool:
        """初始化策略配置和日志"""
        if not super().initialize(config):
            return False
        self._min_sr_room = env_float('MIN_SR_ROOM', 0.0)
        self.logger.info(f"15mTupo策略初始化完成 (BASE_LEV={os.environ.get('BASE_LEV','15')}, "
                         f"MAX_LOSS={os.environ.get('MAX_LOSS','30')}, "
                         f"MIN_SR_ROOM={self._min_sr_room})")
        return True

    def _get_mtf_dir(self, n15, ci15, close, high, low, open_, bars4h=None):
        """Get 1h and 4h direction from 15m bars + actual 4h data"""
        hi = ci15 // 4
        mtf = 'neutral'
        tf4 = 0
        if hi >= 14:
            n1h = n15 // 4
            o1h = np.array([open_[i * 4] for i in range(n1h)])
            c1h = np.array([close[min((i + 1) * 4 - 1, n15 - 1)] for i in range(n1h)])
            if hi < len(c1h):
                ma20_1h = pd.Series(c1h[:hi + 1]).rolling(20, min_periods=20).mean().values
                if not np.isnan(ma20_1h[-1]):
                    if c1h[hi] > ma20_1h[-1] * MTF_UP_THRESHOLD: mtf = 'up'
                    elif c1h[hi] < ma20_1h[-1] * MTF_DOWN_THRESHOLD: mtf = 'down'
                    else: mtf = 'neutral'

                    # 优先使用实际4h K线数据
                    if bars4h and len(bars4h) >= 21:
                        c4h = np.array([b.close for b in bars4h], dtype=float)
                        ma20_4h = pd.Series(c4h).rolling(20, min_periods=20).mean().values
                        if not np.isnan(ma20_4h[-1]):
                            if c4h[-1] > ma20_4h[-1] * MTF_UP_THRESHOLD: tf4 = 1
                            elif c4h[-1] < ma20_4h[-1] * MTF_DOWN_THRESHOLD: tf4 = 2
                    else:
                        # 回退: 从15m聚合4h
                        hi4 = ci15 // 16
                        n4h = n15 // 16
                        if hi4 >= 20:
                            c4h = np.array([close[min((i + 1) * 16 - 1, n15 - 1)] for i in range(n4h)])
                            if hi4 < len(c4h):
                                ma20_4h = pd.Series(c4h[:hi4 + 1]).rolling(20, min_periods=20).mean().values
                                if not np.isnan(ma20_4h[-1]):
                                    if c4h[hi4] > ma20_4h[-1] * MTF_UP_THRESHOLD: tf4 = 1
                                    elif c4h[hi4] < ma20_4h[-1] * MTF_DOWN_THRESHOLD: tf4 = 2
                    return mtf, tf4
        return mtf, tf4

    # ── Retest 状态机 ─────────────────────────────────────────
    def _update_retest_state(self, symbol: str, ti: TrendType, res, sup,
                             c15s: float, cons_trend: str, mtf_dir: str,
                             tf4d: int, ci: int, retest: Optional[dict],
                             _breakout_threshold: float) -> Optional[dict]:
        cur_retest = None
        if ti == TrendType.CONSOLIDATION and res and sup:
            if c15s > res and (c15s - res) / res * 100 > _breakout_threshold and \
               cons_trend != 'down' and mtf_dir in ('neutral', 'up') and tf4d == 1:
                if retest is None or retest.get('level') != res or retest.get('dir') != 'long':
                    cur_retest = {'dir': 'long', 'level': res, 'ci': ci,
                                  'max_dist': (c15s - res) / res * 100, 'state': 'breakout'}
                else:
                    retest['max_dist'] = max(retest['max_dist'], (c15s - res) / res * 100)
                    cur_retest = retest
            elif c15s < sup and (sup - c15s) / sup * 100 > _breakout_threshold and \
                 cons_trend != 'up' and mtf_dir in ('neutral', 'down') and tf4d == 2:
                if retest is None or retest.get('level') != sup or retest.get('dir') != 'short':
                    cur_retest = {'dir': 'short', 'level': sup, 'ci': ci,
                                  'max_dist': (sup - c15s) / sup * 100, 'state': 'breakout'}
                else:
                    retest['max_dist'] = max(retest['max_dist'], (sup - c15s) / sup * 100)
                    cur_retest = retest

        if cur_retest is None:
            if retest and ((retest.get('dir') == 'long' and retest.get('level') == res) or
                           (retest.get('dir') == 'short' and retest.get('level') == sup)):
                cur_retest = retest
        if cur_retest and cur_retest.get('state') == 'breakout' and ci > cur_retest.get('ci', 0):
            rp = cur_retest
            if rp['dir'] == 'long':
                d = (c15s - rp['level']) / rp['level'] * 100
                rp['max_dist'] = max(rp['max_dist'], d)
                if rp['max_dist'] >= 0.3 and d <= 1.0:
                    rp['state'] = 'retest'
            else:
                d = (rp['level'] - c15s) / rp['level'] * 100
                rp['max_dist'] = max(rp['max_dist'], d)
                if rp['max_dist'] >= 0.3 and d <= 1.0:
                    rp['state'] = 'retest'
        if cur_retest and ci - cur_retest.get('ci', 0) > 40:
            cur_retest = None
        self._retest_state[symbol] = cur_retest
        return cur_retest

    # ── 趋势线触碰计数 ──────────────────────────────────────
    def _count_dt_short_touches(self, ti: TrendType, ci: int,
                                info: dict, close: np.ndarray) -> int:
        tl_slp = info.get('trend_line_slope', 0.0) or 0.0
        tl_bv = info.get('trend_line_base_val', 0.0) or 0.0
        tl_bi = info.get('trend_line_base_idx', 0) or 0
        tl_valid = tl_slp != 0 and tl_bv != 0 and tl_bi >= 0
        if not (tl_valid and ci >= 20 and ti == TrendType.DOWNTREND):
            return 0
        touches = 0
        lb = LOOKBACK_BARS
        for j in range(max(0, ci - lb), ci):
            tv = tl_bv + tl_slp * (j - tl_bi)
            if tv <= 0: continue
            if close[j] >= tv and (close[j] - tv) / tv * 100 < TRENDLINE_TOLERANCE_PCT:
                touches += 1
        return touches

    # ── 信号检测: BO_LONG ────────────────────────────────────
    def _check_bo_long(self, ctx: SignalCtx,
                       _breakout_threshold: float, _bo_retest: bool,
                       _bo_strong_breakout: float) -> Optional[Tuple]:
        if not (ctx.ti == TrendType.CONSOLIDATION and ctx.res and ctx.sup and
                ctx.c15s > ctx.res and (ctx.c15s - ctx.res) / ctx.res * 100 > _breakout_threshold and
                ctx.cons_trend != 'down' and ctx.mtf_dir in ('neutral', 'up') and ctx.tf4d == 1):
            return None
        can_enter = False
        if not _bo_retest:
            can_enter = True
        elif ctx.cur_retest and ctx.cur_retest.get('dir') == 'long' and \
             ctx.cur_retest.get('level') == ctx.res and ctx.cur_retest.get('state') == 'retest':
            can_enter = True
        elif _bo_strong_breakout > 0 and ctx.cur_retest and \
             ctx.cur_retest.get('dir') == 'long' and ctx.cur_retest.get('level') == ctx.res and \
             ctx.cur_retest.get('max_dist', 0) >= _bo_strong_breakout:
            can_enter = True
        if not can_enter:
            return None
        ci = ctx.ci
        adv_v = ctx.adx[ci] if ci < len(ctx.adx) else ADX_THRESHOLD
        volr_v = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
        sc = _calc_score('BO_LONG', adv_v, ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50,
                         ctx.atr[ci] if ci < len(ctx.atr) else 0,
                         (ctx.c15s - ctx.res) / ctx.res * 100, volr_v, 2)
        sn = f'BO_LONG_S{sc:.0f}'
        atr_pct = (ctx.atr[ci] / ctx.cp * 100) if ci < len(ctx.atr) and ctx.cp > 0 else 0
        if not (entry_ok(sc, adv_v, volr_v, sn, atr_pct,
                         ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50, 'LONG', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'LONG', ctx.v5_arr, ctx.rng15pct_arr)):
            return None
        sl, lev = calc_sl_lev(PositionSide.LONG, ctx.sup, ctx.cp, sc, 'BO_LONG', ctx.vv,
                              ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
        if sl is None:
            return None
        self._retest_state[ctx.symbol] = None
        return (PositionSide.LONG, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))

    # ── 信号检测: BO_SHORT ───────────────────────────────────
    def _check_bo_short(self, ctx: SignalCtx,
                        _breakout_threshold: float, _bo_retest: bool,
                        _bo_strong_breakout: float) -> Optional[Tuple]:
        if not (ctx.ti == TrendType.CONSOLIDATION and ctx.res and ctx.sup and
                ctx.c15s < ctx.sup and (ctx.sup - ctx.c15s) / ctx.sup * 100 > _breakout_threshold and
                ctx.cons_trend != 'up' and ctx.mtf_dir in ('neutral', 'down') and ctx.tf4d == 2):
            return None
        can_enter = False
        if not _bo_retest:
            can_enter = True
        elif ctx.cur_retest and ctx.cur_retest.get('dir') == 'short' and \
             ctx.cur_retest.get('level') == ctx.sup and ctx.cur_retest.get('state') == 'retest':
            can_enter = True
        elif _bo_strong_breakout > 0 and ctx.cur_retest and \
             ctx.cur_retest.get('dir') == 'short' and ctx.cur_retest.get('level') == ctx.sup and \
             ctx.cur_retest.get('max_dist', 0) >= _bo_strong_breakout:
            can_enter = True
        if not can_enter:
            return None
        ci = ctx.ci
        adv_v = ctx.adx[ci] if ci < len(ctx.adx) else ADX_THRESHOLD
        volr_v = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
        sc = _calc_score('BO_SHORT', adv_v, ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50,
                         ctx.atr[ci] if ci < len(ctx.atr) else 0,
                         (ctx.sup - ctx.c15s) / ctx.sup * 100, volr_v, 2)
        sn = f'BO_SHORT_S{sc:.0f}'
        atr_pct = (ctx.atr[ci] / ctx.cp * 100) if ctx.cp > 0 else 0
        if not (entry_ok(sc, adv_v, volr_v, sn, atr_pct,
                         ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50, 'SHORT', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'SHORT', ctx.v5_arr, ctx.rng15pct_arr)):
            return None
        sl, lev = calc_sl_lev(PositionSide.SHORT, ctx.res, ctx.cp, sc, 'BO_SHORT', ctx.vv,
                              ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
        if sl is None:
            return None
        self._retest_state[ctx.symbol] = None
        return (PositionSide.SHORT, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))

    # ── 信号检测: RB_LONG ────────────────────────────────────
    def _check_rb_long(self, ctx: SignalCtx,
                       _body_ratio: float, _close_position: float) -> Optional[Tuple]:
        if not (ctx.ti == TrendType.CONSOLIDATION and ctx.res and ctx.sup and
                ctx.c15s > ctx.sup * 0.995 and ctx.c15s > ctx.o15s and
                ctx.c15s > ctx.ma20v[ctx.ci] * 0.998 and
                ctx.mtf_dir in ('neutral', 'up') and ctx.tf4d == 1 and
                ctx.l15s <= ctx.sup * 1.003 and ctx.c15s < ctx.res * 0.998):
            return None
        br = (ctx.c15s - ctx.o15s) / max(ctx.h15s - ctx.l15s, 0.001)
        cp2 = (ctx.c15s - ctx.l15s) / max(ctx.h15s - ctx.l15s, 0.001)
        if not (ctx.n5 >= 5 and ctx.c5_above_ma5 and br > _body_ratio and cp2 > _close_position):
            return None
        # MIN_SR_ROOM过滤：确保距阻力位有足够空间
        msr = self._min_sr_room
        if msr > 0 and ctx.res > 0:
            sr_room = (ctx.res - ctx.c15s) / ctx.c15s * 100
            if sr_room < msr:
                return None
        ci = ctx.ci
        adv_v = ctx.adx[ci] if ci < len(ctx.adx) else ADX_THRESHOLD
        volr_v = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
        sc = _calc_score('RB_LONG', adv_v, ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50,
                         ctx.atr[ci] if ci < len(ctx.atr) else 0,
                         (ctx.c15s - ctx.sup) / ctx.sup * 100, volr_v, 2)
        sn = f'RB_LONG_S{sc:.0f}'
        atr_pct = (ctx.atr[ci] / ctx.cp * 100) if ctx.cp > 0 else 0
        if not (entry_ok(sc, adv_v, volr_v, sn, atr_pct,
                         ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50, 'LONG', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'LONG', ctx.v5_arr, ctx.rng15pct_arr)):
            return None
        sl, lev = calc_sl_lev(PositionSide.LONG, ctx.sup, ctx.cp, sc, 'RB_LONG', ctx.vv,
                              ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
        if sl is None:
            return None
        return (PositionSide.LONG, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))

    # ── 信号检测: RB_SHORT ───────────────────────────────────
    def _check_rb_short(self, ctx: SignalCtx,
                        _body_ratio: float, _close_position: float) -> Optional[Tuple]:
        if not (ctx.ti == TrendType.CONSOLIDATION and ctx.res and ctx.sup and
                ctx.h15s >= ctx.res * 0.997 and ctx.h15s < ctx.res * 1.005 and ctx.c15s < ctx.o15s and
                ctx.mtf_dir in ('neutral', 'down') and ctx.tf4d == 2):
            return None
        br2 = (ctx.o15s - ctx.c15s) / max(ctx.h15s - ctx.l15s, 0.001)
        cp22 = (ctx.h15s - ctx.c15s) / max(ctx.h15s - ctx.l15s, 0.001)
        if not (ctx.n5 >= 5 and ctx.c5_below_ma5 and br2 > _body_ratio and cp22 > _close_position):
            return None
        # MIN_SR_ROOM过滤：确保距支撑位有足够空间
        msr = self._min_sr_room
        if msr > 0 and ctx.sup > 0:
            sr_room = (ctx.c15s - ctx.sup) / ctx.c15s * 100
            if sr_room < msr:
                return None
        ci = ctx.ci
        adv_v = ctx.adx[ci] if ci < len(ctx.adx) else ADX_THRESHOLD
        volr_v = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
        sc = _calc_score('RB_SHORT', adv_v, ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50,
                         ctx.atr[ci] if ci < len(ctx.atr) else 0,
                         (ctx.res - ctx.c15s) / ctx.res * 100, volr_v, 2)
        sn = f'RB_SHORT_S{sc:.0f}'
        atr_pct = (ctx.atr[ci] / ctx.cp * 100) if ctx.cp > 0 else 0
        if not (entry_ok(sc, adv_v, volr_v, sn, atr_pct,
                         ctx.rsi_vals[ci] if ci < len(ctx.rsi_vals) else 50, 'SHORT', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'SHORT', ctx.v5_arr, ctx.rng15pct_arr)):
            return None
        sl, lev = calc_sl_lev(PositionSide.SHORT, ctx.res, ctx.cp, sc, 'RB_SHORT', ctx.vv,
                              ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
        if sl is None:
            return None
        return (PositionSide.SHORT, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))

    # ── 信号检测: TR_LONG ────────────────────────────────────
    def _check_tr_long(self, ctx: SignalCtx,
                       _body_ratio: float, _close_position: float) -> Optional[Tuple]:
        """TR_LONG信号检测 - 基于基石线体系（TR策略规范v6.0）
        模式A：P1附近3根K线内早鸟试仓
        模式B：突破H1放量确认主仓
        """
        if ctx.ti != TrendType.UPTREND:
            return None
        foundation = ctx.info.get('foundation_line')
        if foundation is None:
            return None
        ci = ctx.ci
        # 触碰次数（>=2放弃）
        touch_count = ctx.info.get('touch_count', 0)
        if touch_count >= 2:
            return None
        # MIN_SR_ROOM过滤：确保距阻力位有足够上升空间（TR_LONG用resistance）
        msr = self._min_sr_room
        if msr > 0:
            _res = ctx.info.get('resistance', 0)
            if _res > 0:
                sr_room = (_res - ctx.cp) / ctx.cp * 100
                if sr_room < msr:
                    return None

        entry_ci = ci + 1
        p1_idx = ctx.info.get('p1_idx', foundation.get('idx', 0))
        p1_price = ctx.info.get('p1_price', foundation.get('price', 0))
        h1_price = ctx.info.get('h1_price', 0)
        atrv = ctx.atr[ci]

        mode_a = False
        mode_b = False

        # 模式A：P1附近3根K线内（EARLY_WINDOW_BARS=3）
        bars_since_p1 = entry_ci - p1_idx
        if bars_since_p1 >= 0 and bars_since_p1 <= 3:
            pullback_dist = abs(ctx.c15s - p1_price) / max(p1_price, 0.001)
            a_threshold = (atrv / max(ctx.c15s, 0.001)) * 0.5
            if pullback_dist <= a_threshold:
                mode_a = True

        # 模式B：突破H1（放量确认）
        if not mode_a and h1_price > 0:
            if ctx.h15s > h1_price:
                breakout_amp = (ctx.h15s - h1_price) / max(h1_price, 0.001)
                if breakout_amp <= atrv / max(ctx.c15s, 0.001) * 0.3:
                    volr = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
                    if volr > 1.5:
                        mode_b = True

        if not mode_a and not mode_b:
            return None

        # 反转K线（锤子线或十字星）
        body = abs(ctx.c15s - ctx.o15s)
        lower_shadow = min(ctx.o15s, ctx.c15s) - ctx.l15s
        is_hammer = ctx.c15s > ctx.o15s and lower_shadow > body * 2
        is_doji = body < (ctx.h15s - ctx.l15s) * 0.1
        if not (is_hammer or is_doji):
            return None

        adv = ctx.adx[ci]; rsiv = ctx.rsi_vals[ci]
        volr = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
        sc = _calc_score('TR_LONG', adv, rsiv, atrv,
                         (ctx.c15s - ctx.ma20v[ci]) / ctx.ma20v[ci] * 100, volr, 2)
        sn = f'TR_LONG_S{sc:.0f}'
        atr_pct = (atrv / ctx.cp * 100) if ctx.cp > 0 else 0
        if not (entry_ok(sc, adv, volr, sn, atr_pct, rsiv, 'LONG', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'LONG', ctx.v5_arr, ctx.rng15pct_arr)):
            return None
        sl, lev = calc_sl_lev(PositionSide.LONG, None, ctx.cp, sc, 'TR_LONG', ctx.vv,
                              ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
        if sl is None:
            return None
        return (PositionSide.LONG, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))

    # ── 信号检测: TR_SHORT ───────────────────────────────────
    def _check_tr_short(self, ctx: SignalCtx,
                        _body_ratio: float, _close_position: float) -> Optional[Tuple]:
        """TR_SHORT信号检测 - 基于基石线体系（TR策略规范v6.0）
        模式A：P1附近3根K线内早鸟试仓
        模式B：跌破L1放量确认主仓
        """
        if ctx.ti != TrendType.DOWNTREND:
            return None
        foundation = ctx.info.get('foundation_line')
        if foundation is None:
            return None
        ci = ctx.ci
        # 触碰次数（>=2放弃）
        touch_count = ctx.info.get('touch_count', 0)
        if touch_count >= 2:
            return None
        # MIN_SR_ROOM过滤：确保距支撑位有足够下跌空间（TR_SHORT用support）
        msr = self._min_sr_room
        if msr > 0:
            _sup = ctx.info.get('support', 0)
            if _sup > 0:
                sr_room = (ctx.cp - _sup) / ctx.cp * 100
                if sr_room < msr:
                    return None
        # 多时间框架
        if ctx.mtf_dir not in ('neutral', 'down') or ctx.tf4d != 2:
            return None

        entry_ci = ci + 1
        p1_idx = ctx.info.get('p1_idx', foundation.get('idx', 0))
        p1_price = ctx.info.get('p1_price', foundation.get('price', 0))
        h1_price = ctx.info.get('h1_price', 0)
        atrv = ctx.atr[ci]

        mode_a = False
        mode_b = False

        # 模式A：P1附近3根K线内（EARLY_WINDOW_BARS=3）
        bars_since_p1 = entry_ci - p1_idx
        if bars_since_p1 >= 0 and bars_since_p1 <= 3:
            pullback_dist = abs(ctx.c15s - p1_price) / max(p1_price, 0.001)
            a_threshold = (atrv / max(ctx.c15s, 0.001)) * 0.6
            if pullback_dist <= a_threshold:
                mode_a = True

        # 模式B：跌破L1（放量确认）
        if not mode_a and h1_price > 0:
            if ctx.l15s < h1_price:
                breakout_amp = (h1_price - ctx.l15s) / max(h1_price, 0.001)
                if breakout_amp <= atrv / max(ctx.c15s, 0.001) * 0.3:
                    volr = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
                    if volr > 1.5:
                        mode_b = True

        if not mode_a and not mode_b:
            return None

        # 反转K线（射击之星或十字星）
        body = abs(ctx.c15s - ctx.o15s)
        upper_shadow = ctx.h15s - max(ctx.o15s, ctx.c15s)
        is_shooting_star = ctx.c15s < ctx.o15s and upper_shadow > body * 2
        is_doji = body < (ctx.h15s - ctx.l15s) * 0.1
        if not (is_shooting_star or is_doji):
            return None

        adv = ctx.adx[ci]; rsiv = ctx.rsi_vals[ci]
        volr = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
        sc = _calc_score('TR_SHORT', adv, rsiv, atrv,
                         (ctx.c15s - ctx.ma20v[ci]) / ctx.ma20v[ci] * 100, volr, 2)
        sn = f'TR_SHORT_S{sc:.0f}'
        atr_pct = (atrv / ctx.cp * 100) if ctx.cp > 0 else 0
        if not (entry_ok(sc, adv, volr, sn, atr_pct, rsiv, 'SHORT', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'SHORT', ctx.v5_arr, ctx.rng15pct_arr)):
            return None
        sl, lev = calc_sl_lev(PositionSide.SHORT, None, ctx.cp, sc, 'TR_SHORT', ctx.vv,
                              ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
        if sl is None:
            return None
        return (PositionSide.SHORT, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))

    # ── 信号检测: TRIANGLE_LONG / TRIANGLE_SHORT ─────────────
    def _check_triangle(self, ctx: SignalCtx,
                        _triangle_breakout: float, _body_pct: float) -> Optional[Tuple]:
        if ctx.ti != TrendType.TRIANGLE:
            return None
        tdet = ctx.info.get('triangle')
        if not tdet:
            return None
        ci = ctx.ci
        adv_v = ctx.adx[ci]
        volr_v = ctx.v15s / max(ctx.avgv[ci], SMALL_NUM) if 0 <= ci < len(ctx.avgv) else 1.0
        # TRIANGLE_LONG
        if (ctx.c15s > tdet['up_at_ci'] and
            ctx.c15s > ctx.o15s and (ctx.c15s - ctx.o15s) / ctx.o15s * 100 > _body_pct and ctx.tf4d == 1):
            sc = _calc_score('TRIANGLE_LONG', adv_v, ctx.rsi_vals[ci], ctx.atr[ci],
                             (ctx.c15s - tdet['up_at_ci']) / tdet['up_at_ci'] * 100, volr_v, 4)
            sn = f'TRIANGLE_LONG_S{sc:.0f}'
            atr_pct = (ctx.atr[ci] / ctx.cp * 100) if ctx.cp > 0 else 0
            if (entry_ok(sc, adv_v, volr_v, sn, atr_pct, ctx.rsi_vals[ci], 'LONG', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'LONG', ctx.v5_arr, ctx.rng15pct_arr)):
                sl, lev = calc_sl_lev(PositionSide.LONG, tdet['lo_at_ci'], ctx.cp, sc,
                                      'TRIANGLE_LONG', ctx.vv,
                                      ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
                if sl is not None:
                    return (PositionSide.LONG, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))
        # TRIANGLE_SHORT
        if (ctx.c15s < tdet['lo_at_ci'] and
            ctx.c15s < ctx.o15s and (ctx.o15s - ctx.c15s) / ctx.o15s * 100 > _body_pct and ctx.tf4d == 2):
            sc = _calc_score('TRIANGLE_SHORT', adv_v, ctx.rsi_vals[ci], ctx.atr[ci],
                             (tdet['lo_at_ci'] - ctx.c15s) / tdet['lo_at_ci'] * 100, volr_v, 4)
            sn = f'TRIANGLE_SHORT_S{sc:.0f}'
            atr_pct = (ctx.atr[ci] / ctx.cp * 100) if ctx.cp > 0 else 0
            if (entry_ok(sc, adv_v, volr_v, sn, atr_pct, ctx.rsi_vals[ci], 'SHORT', ci) and
                quality_ok(ci, ctx.body15_arr, ctx.pos20_arr, 'SHORT', ctx.v5_arr, ctx.rng15pct_arr)):
                sl, lev = calc_sl_lev(PositionSide.SHORT, tdet['up_at_ci'], ctx.cp, sc,
                                      'TRIANGLE_SHORT', ctx.vv,
                                      ctx.info.get('sup_is_cluster', False) or ctx.info.get('res_is_cluster', False))
                if sl is not None:
                    return (PositionSide.SHORT, sl, sn, lev,
                ctx.info.get('sup_is_cluster', False),
                ctx.info.get('res_is_cluster', False))
        return None

    def analyze(self, symbol: str, klines: Dict[str, List[KlineBar]]) -> Optional[Signal]:
        """分析K线数据生成交易信号"""
        bars15 = klines.get('15m', [])
        bars5 = klines.get('5m', [])
        bars4h = klines.get('4h', [])
        if len(bars15) < MIN_15M_BARS:
            return None

        n15 = len(bars15)
        close = np.array([b.close for b in bars15], dtype=float)
        high = np.array([b.high for b in bars15], dtype=float)
        low = np.array([b.low for b in bars15], dtype=float)
        open_ = np.array([b.open for b in bars15], dtype=float)
        volume = np.array([b.volume for b in bars15], dtype=float)

        if np.any(np.isnan(close)) or np.any(np.isnan(high)) or \
           np.any(np.isnan(low)) or np.any(np.isnan(open_)) or \
           np.any(close <= 0) or np.any(high <= 0) or \
           np.any(high < low):
            self.logger.warning(f"{symbol} K线数据异常（包含NaN/零值/高价<低价），跳过")
            return None

        adx, pdm, mdm, atr = calculate_adx_fast(high, low, close)
        rsi_vals = calculate_rsi(close)
        avgv = calculate_avg_volume_fast(volume)
        ma20v = pd.Series(close).rolling(20).mean().values
        atr_pct_full = np.where(close > 0, atr / close * 100, 0)
        set_atr_ma40_buffer(atr_pct_full)
        rng15 = np.where(high - low < EPSILON, EPSILON, high - low)
        body15_arr = np.abs(close - open_) / rng15
        rng15pct_arr = (high - low) / np.where(close > 0, close, 1)
        h20 = pd.Series(high).rolling(LOOKBACK_BARS).max().values
        l20 = pd.Series(low).rolling(LOOKBACK_BARS).min().values
        pos20_arr = (close - l20) / np.where(h20 - l20 < EPSILON, EPSILON, h20 - l20)
        with self._lock:
            self._cached_body15_arr = body15_arr
            self._cached_pos20_arr = pos20_arr
            self._cached_rng15pct_arr = rng15pct_arr

            n5 = len(bars5)
            c5_above_ma5 = False
            c5_below_ma5 = False
            if n5 >= 1:
                self._last_5m_high = bars5[-1].high
                self._last_5m_low = bars5[-1].low
                self._last_5m_close = bars5[-1].close
            self._cached_v5_arr = np.array([b.volume for b in bars5], dtype=float) if n5 > 0 else np.array([])
        if n5 >= 6:
            c5_closes = np.array([b.close for b in bars5], dtype=float)
            c5_avg_prev5 = float(np.mean(c5_closes[-6:-1]))
            c5_above_ma5 = c5_closes[-1] > c5_avg_prev5
            c5_below_ma5 = c5_closes[-1] < c5_avg_prev5

        ci = n15 - 2
        if ci < 20 or np.isnan(adx[ci]):
            return None

        ti, info = analyze_trend_fast(
            close, high, low, ci, adx, pdm, mdm,
            adx_threshold=20, atr=atr, return_sr_extra=True)
        if ti == TrendType.UNKNOWN:
            return None

        mtf_dir, tf4d = self._get_mtf_dir(n15, ci, close, high, low, open_, bars4h)
        res = info.get('resistance')
        sup = info.get('support')
        cons_trend = info.get('cons_trend_dir', 'neutral')

        c15s = bars15[-1].close; o15s = bars15[-1].open
        h15s = bars15[-1].high; l15s = bars15[-1].low; v15s = bars15[-1].volume
        cp = c15s;         vv = atr[ci] / close[ci] if atr[ci] > 0 else ATR_DEFAULT_RATIO

        _breakout_threshold = env_float('BREAKOUT_THRESHOLD', 0.3)
        _body_ratio = env_float('BODY_RATIO', 0.5)
        _close_position = env_float('CLOSE_POSITION', 0.6)
        _body_pct = env_float('BODY_PCT', 0.3)
        _triangle_breakout = env_float('TRIANGLE_BREAKOUT', 0.5)
        _bo_retest = bool(int(os.environ.get('BO_RETEST', '1')))
        _bo_strong_breakout = env_float('BO_STRONG_BREAKOUT', 0)

        with self._lock:
            retest = self._retest_state.get(symbol)
        cur_retest = self._update_retest_state(
            symbol, ti, res, sup, c15s, cons_trend, mtf_dir, tf4d, ci,
            retest, _breakout_threshold)

        dt_short_touches = self._count_dt_short_touches(ti, ci, info, close)

        ctx = SignalCtx(
            symbol=symbol, ti=ti, info=info, ci=ci, cp=cp, vv=vv,
            c15s=c15s, o15s=o15s, h15s=h15s, l15s=l15s, v15s=v15s,
            res=res, sup=sup, cons_trend=cons_trend, mtf_dir=mtf_dir, tf4d=tf4d,
            n5=n5, c5_above_ma5=c5_above_ma5, c5_below_ma5=c5_below_ma5,
            adx=adx, rsi_vals=rsi_vals, atr=atr, avgv=avgv, ma20v=ma20v,
            body15_arr=body15_arr, pos20_arr=pos20_arr, rng15pct_arr=rng15pct_arr,
            v5_arr=self._cached_v5_arr,
            dt_short_touches=dt_short_touches, cur_retest=cur_retest,
            high_arr=high, low_arr=low,
        )

        sig = None
        for checker in (
            lambda: self._check_bo_long(ctx, _breakout_threshold, _bo_retest, _bo_strong_breakout),
            lambda: self._check_bo_short(ctx, _breakout_threshold, _bo_retest, _bo_strong_breakout),
            lambda: self._check_rb_long(ctx, _body_ratio, _close_position),
            lambda: self._check_rb_short(ctx, _body_ratio, _close_position),
            lambda: self._check_tr_long(ctx, _body_ratio, _close_position),
            lambda: self._check_tr_short(ctx, _body_ratio, _close_position),
            lambda: self._check_triangle(ctx, _triangle_breakout, _body_pct),
        ):
            sig = checker()
            if sig is not None:
                break

        if sig is None:
            adv_v = float(adx[ci]) if ci < len(adx) else 0
            volr_v = float(v15s / max(avgv[ci], 1e-9)) if 0 <= ci < len(avgv) else 0
            retest_info = cur_retest.get('state') if cur_retest else 'None'
            res_s = f'{res:.2f}' if res is not None else 'None'
            sup_s = f'{sup:.2f}' if sup is not None else 'None'
            self.logger.info(
                f"{symbol} 无信号 ti={ti.value} res={res_s} sup={sup_s} "
                f"cons_trend={cons_trend} mtf={mtf_dir} tf4d={tf4d} "
                f"adv={adv_v:.1f} volr={volr_v:.2f} retest={retest_info}")
            return None

        side, sl, sn, lev, sig_isc_sup, sig_isc_res = sig
        _is_rb = sn.startswith('RB_') or 'REBOUND' in sn
        _EXCLUDE_FALLBACK = bool(int(os.environ.get('EXCLUDE_FALLBACK', '0')))
        _RB_EXCLUDE_FALLBACK = bool(int(os.environ.get('RB_EXCLUDE_FALLBACK', '1')))
        if _EXCLUDE_FALLBACK and not (sig_isc_sup or sig_isc_res):
            return None
        if _RB_EXCLUDE_FALLBACK and _is_rb and not (sig_isc_sup or sig_isc_res):
            return None
        framework_side = SignalType.LONG if side == PositionSide.LONG else SignalType.SHORT
        entry_price = cp
        base_lev = env_int('BASE_LEV', 15)
        lev = max(1, min(int(base_lev * 1.5), lev))

        sig_out = Signal(
            signal_type=framework_side, symbol=symbol, strategy='15mTupo',
            price=entry_price, stop_loss=sl, leverage=lev, reason=sn,
        )
        sig_out.metadata = {
            'signal_type_str': sn, 'trend_type': ti.value,
            'entry_res': sup, 'entry_sup': res,
            'foundation_line': info.get('foundation_line'),
            'operating_line': info.get('operating_line'),
        }
        self.logger.info(f"{symbol} 信号: {sn} dir={framework_side.value} lev={lev}")
        _auto_trade = self._auto_trade

        # 去重：相同symbol+side+signal_type不重复发通知
        _side_str = 'LONG' if side == PositionSide.LONG else 'SHORT'
        _sig_key = f'{symbol}_{_side_str}_{sn}'
        if _sig_key != self._last_signal_key.get(symbol):
            self._last_signal_key[symbol] = _sig_key
            try:
                from alert.telegram_bot import TelegramBot as _TB
                from framework.shared.signal_store import store_signal, make_signal_id
                _n = min(80, len(bars15))
                _ts = [getattr(b, 'timestamp', getattr(b, 'open_time', b.time if hasattr(b, 'time') else i*300000))
                       for i, b in enumerate(bars15[-_n:])]
                _df = pd.DataFrame({
                    'datetime': pd.to_datetime(_ts, unit='ms'),
                    'open': open_[-_n:], 'high': high[-_n:],
                    'low': low[-_n:], 'close': close[-_n:],
                    'volume': volume[-_n:],
                })
                _signal_id = make_signal_id(symbol)
                store_signal(_signal_id, {
                    'symbol': symbol, 'side': _side_str,
                    'entry_price': entry_price, 'stop_loss': sl,
                    'leverage': lev, 'signal_type': sn,
                    'trend_type': ti.value, 'entry_sup': sup,
                    'entry_res': res, 'strategy': self._name,
                })
                _bot = _TB(bot_type='alert')
                _side_label = '做多' if side == PositionSide.LONG else '做空'
                _caption_extra = f"[{self._name}] {symbol} {_side_label}\n" \
                                 f"入场: {entry_price:.6f}  止损: {sl:.6f}\n" \
                                 f"杠杆: {lev}x  趋势: {ti.value}\n"
                _tl_bi_raw = info.get('trend_line_base_idx')
                _tl_bi_adj = (_tl_bi_raw - (n15 - _n)) if _tl_bi_raw is not None else None
                _bot.send_signal_chart(
                    _df, symbol, entry_price, side=_side_str,
                    signal_type=sn, trend_type=ti.value,
                    stop_loss=sl, leverage=lev,
                    entry_sup=sup, entry_res=res,
                    tl_slope=info.get('trend_line_slope'),
                    tl_base_idx=_tl_bi_adj,
                    tl_base_val=info.get('trend_line_base_val'),
                    triangle_up=info.get('triangle_up_at_ci'),
                    triangle_lo=info.get('triangle_lo_at_ci'),
                    signal_id=_signal_id,
                    with_trade_button=not _auto_trade,
                    caption_extra=_caption_extra,
                    strategy_name=self._name,
                )
                _bot.close()
            except Exception as e:
                self.logger.debug(f"Telegram通知跳过 (非生产环境): {e}")

        if not _auto_trade:
            return None

        return sig_out

    def check_stop_loss_take_profit(self, position, current_price: float) -> Optional[str]:
        """实盘出口检查：委托 check_exit() 执行退出逻辑"""
        ep = position.entry_price
        lev = position.leverage
        if ep is None or lev is None or ep <= 0 or lev <= 0:
            return None

        side_str = getattr(position.side, 'value', str(position.side))
        side = PositionSide.LONG if 'LONG' in side_str.upper() else PositionSide.SHORT
        st = getattr(position, 'signal_type', '') or getattr(position, 'metadata', {}).get('signal_type_str', '')
        if not hasattr(position, '_tupo_exit_state'):
            position._tupo_exit_state = ExitState()
        es = position._tupo_exit_state
        elapsed_5m = int((time.time() - position.opened_at) / 300) if hasattr(position, 'opened_at') else 0
        hb = elapsed_5m
        mx = getattr(position, 'max_profit_pct', 0.0)
        mn = getattr(position, 'mae_pct', -100.0)
        _er = getattr(position, 'entry_res', 0)
        eres = float(_er) if _er is not None and _er != '' else 0.0
        _es = getattr(position, 'entry_sup', 0)
        esup = float(_es) if _es is not None and _es != '' else 0.0
        tt_str = getattr(position, 'trend_type', '')
        _VALID_TRENDS = ('UPTREND', 'DOWNTREND', 'CONSOLIDATION', 'TRIANGLE')
        tt = TrendType(tt_str) if tt_str in _VALID_TRENDS else TrendType.UNKNOWN

        # ── 半仓保本 ─────────────────────────────────────────
        _half_exit_enabled = bool(int(os.environ.get('HALF_EXIT_ENABLED', '1')))
        _half_exit_trigger = float(os.environ.get('HALF_EXIT_TRIGGER_PCT', '5.0'))
        _half_exit_reentry = float(os.environ.get('HALF_EXIT_REENTRY_PCT', '1.0'))
        pr = getattr(position, '_remaining_ratio', 1.0)
        half_exited = getattr(position, '_half_exited', False)
        half_reentered = getattr(position, '_half_reentered', False)
        half_exit_price = getattr(position, '_half_exit_price', 0.0)

        if _half_exit_enabled and not half_exited and not half_reentered and mx >= _half_exit_trigger:
            # 半仓保本：卖50%，PnL归零
            half_exited = True
            position._half_exited = True
            position._half_exit_price = current_price
            position._remaining_ratio = max(0.0, pr - 0.5)
            position._partial_close_ratio = 0.5
            self.logger.info(f"{position.symbol} 半仓保本 @ {current_price:.4f}")
            return '半仓保本'

        # 半仓再入场
        if _half_exit_enabled and half_exited and not half_reentered and half_exit_price > 0:
            _reentry_px = half_exit_price * (1 + _half_exit_reentry / 100) if side == PositionSide.LONG \
                else half_exit_price * (1 - _half_exit_reentry / 100)
            if (side == PositionSide.LONG and current_price >= _reentry_px) or \
               (side == PositionSide.SHORT and current_price <= _reentry_px):
                half_reentered = True
                position._half_reentered = True
                position._remaining_ratio = 1.0  # 恢复满仓
                self.logger.info(f"{position.symbol} 半仓再入场触发 @ {current_price:.4f}（需调度器加仓）")
                return None

        # P0-1: 从kline_manager获取真实数据，替代硬编码0值
        bars5 = self._kline_manager.get_bars(position.symbol, '5m', 60) if self._kline_manager else []
        bars15 = self._kline_manager.get_bars(position.symbol, '15m', 40) if self._kline_manager else []
        n5 = len(bars5); n15 = len(bars15)

        avgv_ci = 0.0; v15s = 0.0; v5_i = 0.0
        c5_i = current_price; c5_prev = current_price
        with self._lock:
            l5_i = self._last_5m_low if self._last_5m_low > 0 else current_price * 0.99
            h5_i = self._last_5m_high if self._last_5m_high > 0 else current_price * 1.01
            c15s = current_price
            body15_arr = self._cached_body15_arr
            pos20_arr = self._cached_pos20_arr
            rng15pct_arr = self._cached_rng15pct_arr
            v5_arr = self._cached_v5_arr

        if n5 >= 2:
            v5_arr = np.array([b.volume for b in bars5], dtype=float)
            v5_i = bars5[-1].volume
            c5_i = bars5[-1].close; c5_prev = bars5[-2].close
            l5_i = bars5[-1].low; h5_i = bars5[-1].high

        if n15 >= 1:
            c15s = bars15[-1].close

        # avgv_ci: 15m均量(20根SMA), 与回测run_final.py的calculate_avg_volume_fast(df15['volume'])一致
        # 注意: 不能用5m均量, 否则cur_vr=v15s/avgv_ci会偏3倍
        # 修复: n15<20时使用可用数据计算, 避免avgv_ci=0导致volume ratio失真
        if n15 >= 1:
            v15s = bars15[-1].volume
            _v15_arr = np.array([b.volume for b in bars15], dtype=float)
            avgv_ci = float(np.mean(_v15_arr)) if len(_v15_arr) > 0 else v15s
            # 15m均量为0时(启动初期蜡烛未形成), 用5m volume估算
            if avgv_ci <= 0 and v5_i > 0:
                avgv_ci = v5_i * 3
        elif v5_i > 0:
            # 15m数据不可用时，用5m volume的3倍作为近似（15m≈3根5m）
            avgv_ci = v5_i * 3
            if n15 >= 20:
                c15_a = np.array([b.close for b in bars15], dtype=float)
                o15_a = np.array([b.open for b in bars15], dtype=float)
                h15_a = np.array([b.high for b in bars15], dtype=float)
                l15_a = np.array([b.low for b in bars15], dtype=float)
                r15 = np.where(h15_a - l15_a < 1e-12, 1e-12, h15_a - l15_a)
                body15_arr = np.abs(c15_a - o15_a) / r15
                h20 = pd.Series(h15_a).rolling(20).max().values
                l20 = pd.Series(l15_a).rolling(20).min().values
                pos20_arr = (c15_a - l20) / np.where(h20 - l20 < 1e-12, 1e-12, h20 - l20)
                rng15pct_arr = (h15_a - l15_a) / np.where(c15_a > 0, c15_a, 1)

        pr = getattr(position, '_remaining_ratio', 1.0)
        _pos_md = getattr(position, 'metadata', {})
        fl_info = _pos_md.get('foundation_line', {})
        op_info = _pos_md.get('operating_line', {})
        should_close, er, cr, new_sl, new_mx, new_mn, pnl, exit_p = check_exit(
            pos=side, ep=ep, sl=getattr(position, 'stop_loss_price', ep * 0.98),
            lev=lev, cp=current_price,             hb=hb, hb15=hb // 3,
            mx=mx, mn=mn, st=st, tt=tt,
            eidx=0, eidx15=0,
            eres=eres, esup=esup, es=es,
            ci=0, i=-1, avgv_ci=avgv_ci, v15s=v15s, v5_i=v5_i,
            c5_i=c5_i, c5_prev=c5_prev,
            l5_i=l5_i, h5_i=h5_i,
            c15s=c15s, pr=pr,
            sr_room=((eres - ep) / ep * 100) if side == PositionSide.LONG and eres > 0
            else ((ep - esup) / ep * 100) if side == PositionSide.SHORT and esup > 0 else 0.0,
            fl_info=fl_info, op_info=op_info,
        )
        if should_close:
            if 0 < cr < 1.0:
                position._remaining_ratio = max(0.0, pr * (1.0 - cr))
            position._partial_close_ratio = cr
            if exit_p > 0:
                position._exit_price = exit_p
            return er
        # 持久化 mx/mn/sl，确保下次调用时max_profit、MAE和SL不重置
        position.max_profit_pct = new_mx
        position.mae_pct = new_mn
        if new_sl is not None and new_sl > 0:
            position.stop_loss_price = new_sl
        return None

    def on_position_opened(self, position):
        position._tupo_exit_state = ExitState()
        position._remaining_ratio = 1.0
        md = getattr(position, 'metadata', {}) or {}
        st = md.get('signal_type_str', '')
        position.signal_type = st
        _ev = md.get('entry_res', 0)
        position.entry_res = float(_ev) if _ev is not None and _ev != '' else 0.0
        _sv = md.get('entry_sup', 0)
        position.entry_sup = float(_sv) if _sv is not None and _sv != '' else 0.0
        position.trend_type = md.get('trend_type', '')
        # P1-1: 确保止损价已设置（框架已设stop_loss_price，此处明确校验）
        if getattr(position, 'stop_loss_price', None) is None or position.stop_loss_price <= 0:
            _sl = md.get('stop_loss', 0) or md.get('sl', 0)
            if _sl > 0:
                position.stop_loss_price = float(_sl)
            else:
                position.stop_loss_price = position.entry_price * 0.98
                self.logger.warning(f"{position.symbol} 止损未设置，使用默认2%: {position.stop_loss_price:.6f}")
        self.logger.info(f"持仓已开: {position.symbol} {st} SL={position.stop_loss_price:.6f}")

    def on_position_closed(self, position):
        self.logger.info(f"持仓已平: {position.symbol} "
                         f"原因={getattr(position,'close_reason','')} "
                         f"盈亏={getattr(position,'pnl_pct',0):.2f}%")

    def get_status(self) -> Dict[str, Any]:
        return {'strategy': '15mTupo', 'enabled': True, 'auto_trade': self._auto_trade}
