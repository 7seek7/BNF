# -*- coding: utf-8 -*-
"""
15mTupo 共享核心 - 基于 run_final.py (权威版)
指标、趋势分析、信号评分、杠杆计算、退出逻辑
"""

import numpy as np
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import os
import pandas as pd

from framework.shared.enums import PositionSide, SignalType


@dataclass
class TradeResult:
    symbol: str; entry_idx_5m: int; entry_price: float
    exit_idx_5m: int; exit_price: float; side: str
    leverage: float; pnl_pct: float; hold_bars_5m: int
    exit_reason: str; trend_type: str; signal_type: str
    max_pnl_pct: float; min_pnl_pct: float
    continuation_after_exit: float
    entry_adx: float; entry_atr: float; entry_rsi: float
    exit_volume: float; exit_body_ratio: float; exit_rsi: float
    is_cluster_sup: bool = False
    is_cluster_res: bool = False
    sr_room_pct: float = 0.0
    entry_support: float = 0.0
    entry_resistance: float = 0.0
    stop_loss: float = 0.0
    trend_line_slope: float = 0.0
    trend_line_base_idx: int = 0
    trend_line_base_val: float = 0.0
    search_start: int = 0
    search_end: int = 0

@dataclass
class WatchlistEntry:
    """盈利出场后监控条目，用于再入场"""
    symbol: str
    exit_idx_5m: int
    exit_idx_15m: int
    exit_price: float
    exit_reason: str
    original_side: str
    original_signal: str
    original_trend: str
    peak_profit: float
    entry_price: float
    leverage: int
    reentry_count: int = 0
    cooldown_expiry_5m: int = 0
    last_reentry_idx_5m: int = -1


class TrendType(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    CONSOLIDATION = "CONSOLIDATION"
    TRIANGLE = "TRIANGLE"
    SYM_TRIANGLE = "SYM_TRIANGLE"
    ASC_TRIANGLE = "ASC_TRIANGLE"
    DESC_TRIANGLE = "DESC_TRIANGLE"
    RECTANGLE = "RECTANGLE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExitState:
    trail_high: float = 0.0
    trail_active: bool = False
    tsp: float = 0.0
    tc: int = 0
    hrt: bool = False
    prev_pnl: float = 0.0
    ph: List[float] = field(default_factory=list)
    peak_vol_ratio: float = 0.0
    fl_breach_count: int = 0
    op_breach_count: int = 0


def _int_env(key, default):
    """Safe env int: handles float and non-numeric strings"""
    v = os.environ.get(key)
    if v is None or v == '':
        return int(default) if isinstance(default, (int, float, str)) else default
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))
        except ValueError:
            return int(default) if isinstance(default, (int, float, str)) else default


def _resolve_env_key(key):
    """自动尝试15MTUPO_前缀，找不到则用裸键"""
    prefixed = f'15MTUPO_{key}'
    if prefixed in os.environ:
        return prefixed
    return key


def env_float(key, default):
    try:
        resolved = _resolve_env_key(key)
        val = os.environ.get(resolved)
        return float(val) if val is not None else (float(default) if default is not None else default)
    except (ValueError, TypeError):
        return float(default) if default is not None else default


def env_int(key, default):
    try:
        resolved = _resolve_env_key(key)
        val = os.environ.get(resolved)
        return int(float(val)) if val is not None else (int(default) if default is not None else default)
    except (ValueError, TypeError):
        return int(default) if default is not None else default


def env_bool(key, default):
    return os.environ.get(key, str(default)).lower() in ('true', '1', 'yes')


def adjust_leverage_fast(base_leverage, volatility, entry_price, stop_loss,
                         max_loss_pct=100, vol_high=0.05, vol_high_factor=0.5,
                         vol_med=0.03, vol_med_factor=0.75,
                         vol_low=0.02, vol_low_factor=1.2,
                         vol_very_low=0.01, vol_very_low_factor=1.5,
                         leverage_min=5, stop_loss_min_dist=0.5):
    leverage = float(base_leverage)
    if volatility > vol_high:
        leverage = base_leverage * vol_high_factor
    elif volatility > vol_med:
        leverage = base_leverage * vol_med_factor
    elif volatility < vol_very_low:
        leverage = base_leverage * vol_very_low_factor
    elif volatility < vol_low:
        leverage = base_leverage * vol_low_factor
    loss_at_stop = abs(entry_price - stop_loss) / entry_price * 100
    if loss_at_stop < stop_loss_min_dist:
        return 0
    max_loss = loss_at_stop * leverage
    if max_loss > max_loss_pct:
        leverage = max(leverage_min, int(max_loss_pct / loss_at_stop))
    else:
        leverage = max(leverage_min, int(leverage))
    return int(leverage)


class BacktestSettings:
    """兼容层：旧代码 BacktestSettings → 实际读 env var"""
    def __init__(self):
        self.sl_uptrend = env_int('SL_UPTREND', 15)
        self.sl_rebound = env_int('SL_REBOUND', 30)
        self.sl_triangle = env_int('SL_TRIANGLE', 40)
        self.leverage = env_int('BASE_LEV', 15)
        self.adx_threshold = env_int('ADX_THRESHOLD', 20)
        self.trend_price_change = env_float('TREND_PRICE_CHANGE', 2.0)
        self.volume_threshold = env_float('VOLUME_THRESHOLD', 1.3)
        self.body_ratio = env_float('BODY_RATIO', 0.5)
        self.close_position = env_float('CLOSE_POSITION', 0.6)
        self.breakout_threshold = env_float('BREAKOUT_THRESHOLD', 0.3)
        self.range_pct = env_float('RANGE_PCT', 1.0)
        self.body_pct = env_float('BODY_PCT', 0.3)
        self.triangle_breakout = env_float('TRIANGLE_BREAKOUT', 0.5)
        self.leverage_base = env_int('BASE_LEV', 15)
        self.leverage_max = env_int('LEVERAGE_MAX', 30)
        self.leverage_min = env_int('LEVERAGE_MIN', 1)
        self.leverage_max_loss = env_int('MAX_LOSS', 100)
        self.stop_loss_min_dist = env_float('STOP_LOSS_MIN_DIST', 0.5)
        self.ladder_peak_uptrend = env_int('LADDER_PEAK_UPTREND', 25)
        self.ladder_peak_rebound = env_int('LADDER_PEAK_REBOUND', 10)
        self.ladder_peak_tr = env_int('LADDER_PEAK_TR', 0)
        self.ladder_dd_t1_uptrend = env_int('LADDER_DD_T1_UPTREND', 20)
        self.ladder_dd_t2_uptrend = env_int('LADDER_DD_T2_UPTREND', 30)
        self.ladder_dd_t3_uptrend = env_int('LADDER_DD_T3_UPTREND', 40)
        self.ladder_dd_t4_uptrend = env_int('LADDER_DD_T4_UPTREND', 0)
        self.ladder_dd_t5_uptrend = env_int('LADDER_DD_T5_UPTREND', 0)
        self.ladder_dd_t1_rebound = env_int('LADDER_DD_T1_REBOUND', 8)
        self.ladder_dd_t2_rebound = env_int('LADDER_DD_T2_REBOUND', 12)
        self.ladder_dd_t3_rebound = env_int('LADDER_DD_T3_REBOUND', 15)
        self.ladder_dd_t4_rebound = env_int('LADDER_DD_T4_REBOUND', 0)
        self.ladder_dd_t5_rebound = env_int('LADDER_DD_T5_REBOUND', 0)
        self.ladder_dd_t1_tr = env_int('LADDER_DD_T1_TR', 0)
        self.ladder_dd_t2_tr = env_int('LADDER_DD_T2_TR', 0)
        self.ladder_dd_t3_tr = env_int('LADDER_DD_T3_TR', 0)
        self.ladder_dd_t4_tr = env_int('LADDER_DD_T4_TR', 0)
        self.ladder_dd_t5_tr = env_int('LADDER_DD_T5_TR', 0)
        self.ladder_dynamic_scale = True
        self.ladder_scale_midpoint = 50
        self.ladder_close_t1 = env_int('LADDER_CLOSE_T1', 30)
        self.ladder_close_t2 = env_int('LADDER_CLOSE_T2', 40)
        self.ladder_close_t3 = env_int('LADDER_CLOSE_T3', 30)
        self.ladder_close_t4 = env_int('LADDER_CLOSE_T4', 0)
        self.ladder_close_t5 = env_int('LADDER_CLOSE_T5', 0)
        self.ladder_speed_factor = env_int('LADDER_SPEED_FACTOR', 10)
        self.ladder_vol_shrink = env_float('LADDER_VOL_SHRINK', 0.7)
        self.breakeven_peak = env_int('BREAKEVEN_PEAK', 10)
        self.breakeven_return = env_int('BREAKEVEN_RETURN', 3)
        self.rebound_opposite_enabled = env_bool('REBOUND_OPPOSITE_ENABLED', True)
        self.rebound_opposite_distance = env_float('REBOUND_OPPOSITE_DISTANCE', 0.005)
        self.rebound_opposite_close_ratio = env_float('REBOUND_OPPOSITE_CLOSE_RATIO', 0.50)
        self.entry_stop_bars = env_int('ENTRY_STOP_BARS', 30)
        self.stuck_bars_min = env_int('STUCK_BARS_MIN', 60)
        self.stuck_pnl_range = env_float('STUCK_PNL_RANGE', 5)
        self.stuck_history_bars = env_int('STUCK_HISTORY_BARS', 20)
        self.stuck_max_peak = env_int('STUCK_MAX_PEAK', 10)
        self.time_exit_bars_min = env_int('TIME_EXIT_BARS_MIN', 200)
        self.time_exit_history = env_int('TIME_EXIT_HISTORY', 30)
        self.time_exit_range = env_float('TIME_EXIT_RANGE', 5)
        self.capital_ratio = env_float('CAPITAL_RATIO', 70)
        self.max_positions = env_int('MAX_POSITIONS', 1)
        self.single_symbol_max = env_float('SINGLE_SYMBOL_MAX', 5000)
        self.monitor_symbols = env_int('MONITOR_SYMBOLS', 50)
        self.trading_fee_rate = env_float('TRADING_FEE_RATE', 0.0005)
        self.taker_fee = env_float('TAKER_FEE', 0.0005)
        self.vol_high = env_float('VOL_HIGH', 0.05)
        self.vol_high_factor = env_float('VOL_HIGH_FACTOR', 0.5)
        self.vol_med = env_float('VOL_MED', 0.03)
        self.vol_med_factor = env_float('VOL_MED_FACTOR', 0.75)
        self.vol_low = env_float('VOL_LOW', 0.02)
        self.vol_low_factor = env_float('VOL_LOW_FACTOR', 1.2)
        self.vol_very_low = env_float('VOL_VERY_LOW', 0.01)
        self.vol_very_low_factor = env_float('VOL_VERY_LOW_FACTOR', 1.5)
    @property
    def ENABLED(self): return self.capital_ratio > 0
    @property
    def MAX_POSITIONS(self): return self.max_positions
    @property
    def MONITOR_SYMBOLS(self): return self.monitor_symbols
    @property
    def SINGLE_SYMBOL_MAX(self): return self.single_symbol_max
    @property
    def TRADING_FEE_RATE(self): return self.trading_fee_rate
    @property
    def LEVERAGE(self): return self.leverage
    @property
    def ADX_THRESHOLD(self): return self.adx_threshold
    @property
    def VOLUME_THRESHOLD(self): return self.volume_threshold
    @property
    def SIGNAL_BODY_RATIO(self): return self.body_ratio
    @property
    def SIGNAL_BREAKOUT_THRESHOLD(self): return self.breakout_threshold
    @property
    def SIGNAL_RANGE_PCT(self): return self.range_pct
    @property
    def SIGNAL_VOLUME_THRESHOLD(self): return self.volume_threshold


SIG_QUAL = {
    # 扩展SIG_QUAL覆盖ADX/RSI调整后的分数变体
    # 原则: 同信号类型内, 分越高→sig_qual越高
    # RB_SHORT
    "RB_SHORT_S10": 1.7, "RB_SHORT_S9": 1.5, "RB_SHORT_S8": 1.2, "RB_SHORT_S7": 0.9,
    # BO_SHORT
    "BO_SHORT_S10": 1.5, "BO_SHORT_S9": 1.4, "BO_SHORT_S8": 1.3, "BO_SHORT_S7": 1.1, "BO_SHORT_S6": 0.9,
    # TRIANGLE_SHORT
    "TRIANGLE_SHORT_S10": 1.4, "TRIANGLE_SHORT_S9": 1.3, "TRIANGLE_SHORT_S8": 1.2,
    # RB_LONG
    "RB_LONG_S9": 1.3, "RB_LONG_S8": 1.2, "RB_LONG_S7": 1.1, "RB_LONG_S6": 0.9, "RB_LONG_S5": 0.7,
    # BO_LONG
    "BO_LONG_S8": 1.1, "BO_LONG_S7": 1.0, "BO_LONG_S6": 0.9, "BO_LONG_S5": 0.8, "BO_LONG_S4": 0.7,
    # TR_SHORT
    "TR_SHORT_S9": 1.3, "TR_SHORT_S8": 1.2, "TR_SHORT_S7": 1.0, "TR_SHORT_S6": 0.9, "TR_SHORT_S5": 0.7,
    # TRIANGLE_LONG
    "TRIANGLE_LONG_S8": 1.1, "TRIANGLE_LONG_S7": 1.0, "TRIANGLE_LONG_S6": 0.9,
    # TR_LONG
    "TR_LONG_S7": 1.0, "TR_LONG_S6": 0.9, "TR_LONG_S5": 0.7, "TR_LONG_S4": 0.6, "TR_LONG_S3": 0.5,
}


def _calc_score(sig_type, adv=None, rsiv=None, atrv=None, srd=None, volr=None, swc=None,
                adx_mode=None, srd_mode=None):
    # 基础分：基于信号类型的历史EV排名
    # SHORT > LONG, RB_SHORT > BO_SHORT > RB_LONG > BO_LONG > TR
    score_map = {
        'BO_LONG': 6, 'BO_SHORT': 8,
        'RB_LONG': 7, 'RB_SHORT': 9,
        'TR_LONG': 5, 'TR_SHORT': 7,
        'TRIANGLE_LONG': 6, 'TRIANGLE_SHORT': 8,
    }
    sc = score_map.get(sig_type, 5)

    # ADX调整：低ADX(盘整后突破)普遍比高ADX(趋势中追)好
    if adv is not None:
        if adv < 17:
            sc += 1
        elif adv > 30:
            sc -= 1

    # RSI调整：方向感知
    if rsiv is not None:
        is_long = sig_type.endswith('LONG')
        if is_long:
            if rsiv < 40:
                sc += 1
            elif rsiv > 70:
                sc -= 1
        else:
            if rsiv > 55:
                sc += 1
            elif rsiv < 35:
                sc -= 1

    return max(1, min(10, int(sc)))


def entry_ok(sc, adv, volr, sig_name="", atr_pct=0, rsi=50, side="", ci=-1):
    # 分数现在是基于信号类型的固定值（真实EV排名），不再用通用阈值过滤
    # ENTRY_SC_MIN 保留但实效化, 由 MARKET_BIAS 做方向过滤
    market_bias = os.environ.get('MARKET_BIAS', 'NEUTRAL').upper()
    if market_bias == 'SHORT' and side == 'LONG':
        if sc < env_float('MARKET_BIAS_LONG_MIN_SC', 7): return False
    elif market_bias == 'LONG' and side == 'SHORT':
        if sc < env_float('MARKET_BIAS_SHORT_MIN_SC', 7): return False
    min_adx = env_float('MIN_ADX', 0)
    if adv < min_adx: return False
    max_adx = env_float('MAX_ADX', 0)
    if max_adx > 0 and adv > max_adx: return False
    if volr < env_float('MIN_VOLR', 0): return False
    max_atr = env_float('MAX_ATR', 0)
    if max_atr > 0 and atr_pct > max_atr: return False
    max_atr_long = env_float('MAX_ATR_LONG', 0)
    if max_atr_long > 0 and side == "LONG" and atr_pct > max_atr_long: return False
    if side == "LONG" and rsi > env_float('RSI_LONG_MAX', 100): return False
    if side == "SHORT" and rsi < env_float('RSI_SHORT_MIN', 0): return False
    max_atr_ratio = env_float('MAX_ATR_RATIO', 0)
    if max_atr_ratio > 0:
        buf = _atr_ma40_buffer
        if len(buf) > 0 and ci >= 0 and ci < len(buf) and buf[ci] > 0:
            if atr_pct > buf[ci] * max_atr_ratio:
                return False
    blacklist = set(os.environ.get('SIGNAL_BLACKLIST', '').split(',')) - {''}
    for blk in blacklist:
        if sig_name.startswith(blk):
            return False
    return True


def check_double_pullback(high_arr, low_arr, ci, level, side, lookback=15, min_bounce_pct=0.3):
    """二次回踩检测：价格曾触碰level→反弹→再次触碰"""
    if level is None or ci < lookback + 2:
        return False
    touch_dist = level * 0.005  # 0.5%容差
    first_touch_idx = -1
    for i in range(max(0, ci - lookback), ci):
        if side == 'LONG':
            if abs(low_arr[i] - level) <= touch_dist or low_arr[i] < level:
                first_touch_idx = i
                break
        else:
            if abs(high_arr[i] - level) <= touch_dist or high_arr[i] > level:
                first_touch_idx = i
                break
    if first_touch_idx < 0:
        return False
    if side == 'LONG':
        max_after = max(high_arr[first_touch_idx+1:ci+1]) if first_touch_idx + 1 <= ci else 0
        bounce = (max_after - level) / level * 100
    else:
        min_after = min(low_arr[first_touch_idx+1:ci+1]) if first_touch_idx + 1 <= ci else float('inf')
        bounce = (level - min_after) / level * 100
    return bounce >= min_bounce_pct


_atr_ma40_buffer = np.array([])
_atr_ma40_lock = threading.Lock()

# Score config cache — avoids os.environ.get() per _calc_score() call
_ADX_MODE = None
_SRD_MODE = None
_score_cfg_lock = threading.Lock()

def _get_score_cfg():
    global _ADX_MODE, _SRD_MODE
    if _ADX_MODE is None or _SRD_MODE is None:
        with _score_cfg_lock:
            if _ADX_MODE is None:
                _ADX_MODE = _int_env('ADX_MODE', '0')
            if _SRD_MODE is None:
                _SRD_MODE = _int_env('SRD_MODE', '0')
    return _ADX_MODE, _SRD_MODE

# Exit config cache — populated once per process, avoids ~40 os.environ.get() per check_exit() call
_EXIT_CFG = None
_exit_cfg_lock = threading.Lock()

def _get_exit_cfg():
    global _EXIT_CFG
    if _EXIT_CFG is None:
        with _exit_cfg_lock:
            if _EXIT_CFG is None:
                _EXIT_CFG = {
                    'vel_floor': float(os.environ.get('VEL_FLOOR', '11')),
                    'vel_min_mx': float(os.environ.get('VEL_MIN_MX', '8')),
                    'vel_floor_tr': float(os.environ.get('VEL_FLOOR_TR', '0')),
                    'vel_tr_mx_cap': float(os.environ.get('VEL_TR_MX_CAP', '20')),
                    'vel_window': int(os.environ.get('VEL_WINDOW', '3')),
                    'vel_floor_multi': float(os.environ.get('VEL_FLOOR_MULTI', '15')),
                    'trail_act': float(os.environ.get('TRAIL_ACT', '5.0')),
                    'trail_dist': float(os.environ.get('TRAIL_DIST', '1.5')),
                    'trail_tight_act': float(os.environ.get('TRAIL_TIGHT_ACT', '0')),
                    'trail_tight_dist': float(os.environ.get('TRAIL_TIGHT_DIST', '0')),
                    'trail_act_tr': float(os.environ.get('TRAIL_ACT_TR', '0')),
                    'trail_dist_tr': float(os.environ.get('TRAIL_DIST_TR', '0')),
                    'slippage': float(os.environ.get('SLIPPAGE', '0.03')),
                    'funding': float(os.environ.get('FUNDING', '0.0001')),
                    'rebound_opp': os.environ.get('REBOUND_OPPOSITE_ENABLED', os.environ.get('REBOUND_OPP', '0')).lower() in ('1', 'true', 'yes'),
                    'rebound_be': bool(int(os.environ.get('REBOUND_BE', '0'))),
                    'ladder_vs': bool(int(os.environ.get('LADDER_VS', '0'))),
                    'ladder_peak_uptrend': int(os.environ.get('LADDER_PEAK_UPTREND', '25')),
                    'ladder_peak_rebound': int(os.environ.get('LADDER_PEAK_REBOUND', '10')),
                    'ladder_peak_tr': int(os.environ.get('LADDER_PEAK_TR', '0')),
                    'ladder_dd_t1_uptrend': int(os.environ.get('LADDER_DD_T1_UPTREND', '20')),
                    'ladder_dd_t2_uptrend': int(os.environ.get('LADDER_DD_T2_UPTREND', '30')),
                    'ladder_dd_t3_uptrend': int(os.environ.get('LADDER_DD_T3_UPTREND', '40')),
                    'ladder_dd_t4_uptrend': int(os.environ.get('LADDER_DD_T4_UPTREND', '0')),
                    'ladder_dd_t5_uptrend': int(os.environ.get('LADDER_DD_T5_UPTREND', '0')),
                    'ladder_dd_t1_rebound': int(os.environ.get('LADDER_DD_T1_REBOUND', '8')),
                    'ladder_dd_t2_rebound': int(os.environ.get('LADDER_DD_T2_REBOUND', '12')),
                    'ladder_dd_t3_rebound': int(os.environ.get('LADDER_DD_T3_REBOUND', '15')),
                    'ladder_dd_t4_rebound': int(os.environ.get('LADDER_DD_T4_REBOUND', '0')),
                    'ladder_dd_t5_rebound': int(os.environ.get('LADDER_DD_T5_REBOUND', '0')),
                    'ladder_dd_t1_tr': int(os.environ.get('LADDER_DD_T1_TR', '0')),
                    'ladder_dd_t2_tr': int(os.environ.get('LADDER_DD_T2_TR', '0')),
                    'ladder_dd_t3_tr': int(os.environ.get('LADDER_DD_T3_TR', '0')),
                    'ladder_dd_t4_tr': int(os.environ.get('LADDER_DD_T4_TR', '0')),
                    'ladder_dd_t5_tr': int(os.environ.get('LADDER_DD_T5_TR', '0')),
                    'ladder_close_t1': int(os.environ.get('LADDER_CLOSE_T1', '30')),
                    'ladder_close_t2': int(os.environ.get('LADDER_CLOSE_T2', '40')),
                    'ladder_close_t3': int(os.environ.get('LADDER_CLOSE_T3', '30')),
                    'ladder_close_t4': int(os.environ.get('LADDER_CLOSE_T4', '0')),
                    'ladder_close_t5': int(os.environ.get('LADDER_CLOSE_T5', '0')),
                    'entry_stop_bars': int(os.environ.get('ENTRY_STOP_BARS', '30')),
                    'be_mx_pct': float(os.environ.get('BE_MX_PCT', '5.0')),
                    'be_sl_buffer_pct': float(os.environ.get('BE_SL_BUFFER_PCT', '0')),
                    'be_hard_mx': float(os.environ.get('BE_HARD_MX', '0')),
                    'be_weak_mx': float(os.environ.get('BE_WEAK_MX', '0')),
                    'be_weak_vol': float(os.environ.get('BE_WEAK_VOL', '5.0')),
                    'be_unconditional_mx': float(os.environ.get('BE_UNCONDITIONAL_MX', '0')),
                    'trend_be_mx': float(os.environ.get('TREND_BE_MX', '0')),
                    'vol_confirm_thresh': float(os.environ.get('VOL_CONFIRM_THRESH', '5.0')),
                    'liq_mm_rate': float(os.environ.get('LIQ_MM_RATE', '0.5')),
                    'liq_threshold': float(os.environ.get('LIQ_THRESHOLD', '-90')),
                    'profit_protect_pct': float(os.environ.get('PROFIT_PROTECT_PCT', '0')),
                    'reentry_enabled': bool(int(os.environ.get('REENTRY_ENABLED', '0'))),
                    'reentry_cooldown': int(os.environ.get('REENTRY_COOLDOWN', '8')),
                    'reentry_pullback_pct': float(os.environ.get('REENTRY_PULLBACK_PCT', '3.0')),
                    'reentry_min_peak': float(os.environ.get('REENTRY_MIN_PEAK', '5.0')),
                    'reentry_max_times': int(os.environ.get('REENTRY_MAX_TIMES', '2')),
                    'reentry_max_wait': int(os.environ.get('REENTRY_MAX_WAIT', '40')),
                }
    return _EXIT_CFG

def reload_exit_cfg():
    """Force reload exit config from env vars (call after env changes)"""
    global _EXIT_CFG
    with _exit_cfg_lock:
        _EXIT_CFG = None


def set_atr_ma40_buffer(atr_pct_full: np.ndarray) -> None:
    """计算并设置 _atr_ma40_buffer（40周期ATR%滚动均值），供回测和实盘共享"""
    global _atr_ma40_buffer
    with _atr_ma40_lock:
        _atr_ma40_buffer = pd.Series(atr_pct_full).shift(1).rolling(40, min_periods=10).mean().values


def quality_ok(ci, body15_arr, pos20_arr, side, v5_arr, rng15pct_arr):
    min_body15 = env_float('MIN_BODY15', 0)
    min_pos20 = env_float('MIN_POS20', 0)
    min_vol5 = env_float('MIN_VOL5', 0)
    min_rng15 = env_float('MIN_RNG15', 0)
    if min_body15 <= 0 and min_pos20 <= 0 and min_vol5 <= 0 and min_rng15 <= 0:
        return True
    if ci < 20: return True
    if min_body15 > 0:
        b = body15_arr[ci] if ci < len(body15_arr) else 0
        if b < min_body15: return False
    if min_pos20 > 0:
        p = pos20_arr[ci] if ci < len(pos20_arr) else 0.5
        if p < min_pos20: return False
    if min_vol5 > 0:
        i5 = ci * 3
        if i5 < 6: return True
        if i5 >= len(v5_arr): return True
        prev = v5_arr[i5-6:i5]
        if len(prev) == 0: return True
        avg5 = float(prev.mean())
        if avg5 <= 0: return True
        v5r = float(v5_arr[i5]) / avg5
        if v5r < min_vol5: return False
    if min_rng15 > 0:
        r = rng15pct_arr[ci] if ci < len(rng15pct_arr) else 0
        if r < min_rng15: return False
    return True


def calc_sl_lev(side, ref, cp, sc, sig_type, vol, is_cluster_sig=True):
    side_filter = os.environ.get('SIDE_FILTER', '')
    if side_filter == "SHORT" and side == PositionSide.LONG: return None, None
    if side_filter == "LONG" and side == PositionSide.SHORT: return None, None
    boost_sigs = set(os.environ.get('BOOST_SIGS', '').split(',')) - {''}
    boost_qual = float(os.environ.get('BOOST_QUAL', '2.0'))
    sq_key = sig_type + "_S" + str(int(sc))
    sig_qual = SIG_QUAL.get(sq_key, 1.0)
    if boost_sigs and sq_key in boost_sigs:
        sig_qual *= boost_qual
    cluster_lev_mult = float(os.environ.get('CLUSTER_LEV_MULT', '1.0'))
    fallback_lev_mult = float(os.environ.get('FALLBACK_LEV_MULT', '1.0'))
    if is_cluster_sig and cluster_lev_mult != 1.0:
        sig_qual *= cluster_lev_mult
    elif not is_cluster_sig and fallback_lev_mult != 1.0:
        sig_qual *= fallback_lev_mult

    base_lev = _int_env('BASE_LEV', '15')
    max_loss = _int_env('MAX_LOSS', '30')
    buf = 0.005
    # TR专用初始止损加宽：SL_TR_ATR_MULT 作为默认2%止损的乘数
    if side == PositionSide.LONG:
        if ref is not None:
            raw = ref * (1 - buf)
        elif sig_type.startswith('TR_'):
            _sl_mult = float(os.environ.get('SL_TR_ATR_MULT', '2.0'))
            raw = cp * (1 - 0.02 * _sl_mult)
        else:
            raw = cp * 0.98
        sp = (cp - raw) / cp * 100
    else:
        if ref is not None:
            raw = ref * (1 + buf)
        elif sig_type.startswith('TR_'):
            _sl_mult = float(os.environ.get('SL_TR_ATR_MULT', '2.0'))
            raw = cp * (1 + 0.02 * _sl_mult)
        else:
            raw = cp * 1.02
        sp = (raw - cp) / cp * 100
    sp = max(sp, 0.5)
    max_sp = float(os.environ.get('MAX_SP', '5.0'))
    if sp > max_sp: sp = max_sp
    if side == PositionSide.LONG: sl = cp * (1 - sp / 100)
    else: sl = cp * (1 + sp / 100)

    base_lev_calc = max(1, int(base_lev * sc / 10))
    if vol > 0.05: lev_factor = 0.5
    elif vol > 0.03: lev_factor = 0.75
    elif vol <= 0.01: lev_factor = 1.5
    elif vol < 0.02: lev_factor = 1.2
    else: lev_factor = 1.0
    lev = max(1, int(base_lev_calc * lev_factor * sig_qual))

    lev_mult = 1.0
    if sig_type.startswith('TR_'):
        lev_mult = float(os.environ.get('LEV_MULT_TR', '1.0'))
    elif sig_type.startswith('RB_'):
        lev_mult = float(os.environ.get('LEV_MULT_RB', '1.0'))
    elif sig_type.startswith('BO_'):
        lev_mult = float(os.environ.get('LEV_MULT_BO', '1.0'))
    lev = max(1, int(lev * lev_mult))

    rb_mx = _int_env('RB_MAX_LEV', '15')
    if rb_mx > 0 and sig_type.startswith('RB_') and lev > rb_mx:
        lev = rb_mx
    tr_mx = _int_env('TR_MAX_LEV', '0')
    if tr_mx > 0 and sig_type.startswith('TR_') and lev > tr_mx:
        lev = tr_mx
    bo_mx = _int_env('BO_MAX_LEV', '0')
    if bo_mx > 0 and sig_type.startswith('BO_') and lev > bo_mx:
        lev = bo_mx
    max_loss_calc = sp * lev
    if max_loss_calc > max_loss:
        lev = max(1, int(max_loss / sp))
    return sl, lev


def aggregate_5m_to_15m_fast(df_5m):
    df = df_5m.reset_index(drop=True).copy()
    df['5m_idx'] = df.index // 3
    return df.groupby('5m_idx').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()


def calculate_adx_fast(high, low, close, period=14):
    n = len(close)
    tr = np.maximum(high[1:]-low[1:], np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
    up_move = high[1:]-high[:-1]; down_move = low[:-1]-low[1:]
    plus_dm = np.where((up_move>down_move)&(up_move>0), up_move, 0)
    minus_dm = np.where((down_move>up_move)&(down_move>0), down_move, 0)
    tr_avg = np.full(n, np.nan); plus_di = np.full(n, np.nan); minus_di = np.full(n, np.nan); adx = np.full(n, np.nan)
    tr_avg[period] = np.mean(tr[:period])
    if tr_avg[period] > 0:
        plus_di[period] = 100*np.mean(plus_dm[:period])/tr_avg[period]
        minus_di[period] = 100*np.mean(minus_dm[:period])/tr_avg[period]
    for i in range(period+1, n):
        tr_avg[i] = (tr_avg[i-1]*(period-1)+tr[i-1])/period
        if tr_avg[i] > 0:
            plus_di[i] = (plus_di[i-1]*(period-1)+100*plus_dm[i-1]/tr_avg[i])/period
            minus_di[i] = (minus_di[i-1]*(period-1)+100*minus_dm[i-1]/tr_avg[i])/period
        else:
            plus_di[i] = plus_di[i-1]
            minus_di[i] = minus_di[i-1]
    di_sum = np.where(np.isnan(plus_di)|np.isnan(minus_di), 0, plus_di+minus_di)
    with np.errstate(invalid='ignore', divide='ignore'):
        dx = np.where(di_sum > 0, np.abs(plus_di-minus_di)/di_sum*100, 0)
    adx[2*period] = np.mean(dx[period:2*period])
    for i in range(2*period+1, n):
        adx[i] = (adx[i-1]*(period-1)+dx[i])/period
    return adx, plus_di, minus_di, tr_avg


def calculate_rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta>0, delta, 0); loss = np.where(delta<0, -delta, 0)
    avg_g = np.full(len(close), np.nan); avg_l = np.full(len(close), np.nan)
    avg_g[period] = np.mean(gain[:period]); avg_l[period] = np.mean(loss[:period])
    for i in range(period+1, len(close)):
        avg_g[i] = (avg_g[i-1]*13+gain[i-1])/14; avg_l[i] = (avg_l[i-1]*13+loss[i-1])/14
    rs = avg_g/np.maximum(avg_l, 1e-10)
    return 100-100/(1+rs)


def calculate_avg_volume_fast(volume, period=20):
    r = np.full(len(volume), np.nan)
    for i in range(period, len(volume)):
        r[i] = np.mean(volume[i-period:i])
    return r


def _reg_slope_pct(arr, start, end):
    if end - start < 10:
        return 0, 0, 0
    x = np.arange(start, end+1, dtype=float)
    y = arr[start:end+1]
    n = float(len(x))
    sx = float(np.sum(x)); sy = float(np.sum(y))
    sxx = float(np.sum(x*x)); sxy = float(np.sum(x*y))
    denom = n*sxx - sx*sx
    if abs(denom) < 1e-15:
        return 0, 0, 0
    m = (n*sxy - sx*sy) / denom
    b = (sy - m*sx) / n
    y_pred = m*x + b
    ss_res = float(np.sum((y - y_pred)**2))
    ss_tot = float(np.sum((y - float(np.mean(y)))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    total_change = m * (end - start)
    avg_price = float(np.mean(y))
    return m / (abs(b) + 0.01) * 100, float(m), float(r2)


def find_trend_line_fast(high, low, idx, lookback, is_up, atr=None):
    if idx < 3:
        return None, None, None, None
    arr = low if is_up else high
    opp = high if is_up else low
    if atr is not None:
        a = atr[max(0, idx-lookback):idx+1]
        tol = max((a.sum() / a.size) * 0.4, 0.01)
    else:
        tol = 0.02
    start = max(1, idx - lookback)
    sw_pts = []
    for i in range(start, idx - 1):
        ai = arr[i]
        if is_up:
            if ai < arr[i-1] and ai < arr[i+1]:
                sw_pts.append((float(ai), int(i)))
        else:
            if ai > arr[i-1] and ai > arr[i+1]:
                sw_pts.append((float(ai), int(i)))
    if len(sw_pts) < 2:
        return None, None, None, None
    xv = np.array([p[1] for p in sw_pts], dtype=float)
    yv = np.array([p[0] for p in sw_pts], dtype=float)
    x_rel = xv - start
    nf = float(len(x_rel))
    sx = float(x_rel.sum()); sy = float(yv.sum())
    sxx = float((x_rel*x_rel).sum()); sxy = float((x_rel*yv).sum())
    d = nf*sxx - sx*sx
    if abs(d) < 1e-15:
        return None, None, None, None
    slope = (nf*sxy - sx*sy) / d
    intercept = (sy - slope*sx) / nf
    if is_up and slope <= 0:
        return None, None, None, None
    if not is_up and slope >= 0:
        return None, None, None, None
    touches = 0
    crosses = 0
    n_bars = idx - start + 1
    for k in range(start, idx + 1):
        lv = slope * (k - start) + intercept
        if is_up:
            if lv > opp[k]:
                crosses += 1
            if abs(lv - low[k]) <= tol:
                touches += 1
        else:
            if lv < opp[k]:
                crosses += 1
            if abs(lv - high[k]) <= tol:
                touches += 1
    if touches < 2:
        return None, None, None, None, None
    if crosses > n_bars * 0.1:
        return None, None, None, None, None
    # 计算R²（规格书3.4.1 LineScore）
    y_pred = slope * x_rel + intercept
    ss_res = float(((yv - y_pred)**2).sum())
    ss_tot = float(((yv - sy / nf)**2).sum())
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
    return slope, intercept, start, idx, r_squared


def _detect_market_segments(adx, idx, adx_threshold=20, max_lookback=120):
    start = max(0, idx - max_lookback)
    segments = []
    seg_type = None
    seg_end = idx
    for i in range(idx, start - 1, -1):
        is_trend = not np.isnan(adx[i]) and adx[i] >= adx_threshold
        cur = 'trend' if is_trend else 'consol'
        if seg_type is None:
            seg_type = cur
        elif cur != seg_type:
            segments.append({'type': seg_type, 'start': i + 1, 'end': seg_end})
            seg_type = cur
            seg_end = i
    if seg_type is not None:
        segments.append({'type': seg_type, 'start': start, 'end': seg_end})
    merged = []
    for seg in segments:
        if merged and merged[-1]['type'] == seg['type']:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(dict(seg))
    return merged


def find_horizontal_SR(high, low, close, idx, atr,
                       lookback=60, cluster_atr_mult=0.15,
                       penetration_bars=40, penetration_threshold=0.05,
                        penetration_price=0.002, min_cluster=4,
                        eff_high=None, eff_low=None, spike_flags=None,
                        bbw_percentile=None):
    """
    规格书5.1节: 水平S/R提取（仅用于CONSOLIDATION）
    3.2节: DBSCAN聚类，eps=0.15×ATR×(1+BBW分位数)，minPts=4
    使用Typical Price + 左3右3极值 + DBSCAN聚类 + 穿透率验证
    4.1节: 使用effective价格进行支撑阻力识别
    附录A Q8: 插针过滤穿透标记 — 连续3次过滤穿透视为有效穿透
    """
    if idx < 20 or atr is None or idx >= len(high):
        return None, None, {}
    atr_val = float(atr[idx])
    if atr_val <= 0:
        return None, None, {}
    # 规格书3.2: eps=0.15×ATR×(1+BBW分位数)
    bbw_factor = (1 + bbw_percentile) if bbw_percentile is not None else 1.0
    eps = cluster_atr_mult * atr_val * bbw_factor
    # 使用有效价格提取极值（如果提供）
    sr_high = eff_high if eff_high is not None else high
    sr_low = eff_low if eff_low is not None else low
    start = max(0, idx - lookback)
    end = min(idx - 4, len(high) - 4)
    # 提取极值点（使用有效价格）
    sh = []  # 峰顶
    sl = []  # 谷底
    for i in range(start + 3, end + 1):
        tp = (float(sr_high[i]) + float(sr_low[i]) + float(close[i])) / 3.0
        left = [(float(sr_high[j])+float(sr_low[j])+float(close[j]))/3.0 for j in range(i-3, i)]
        right = [(float(sr_high[j])+float(sr_low[j])+float(close[j]))/3.0 for j in range(i+1, i+4)]
        if len(left) < 3 or len(right) < 3:
            continue
        if tp > max(left) and tp > max(right):
            sh.append((float(sr_high[i]), i))
        if tp < min(left) and tp < min(right):
            sl.append((float(sr_low[i]), i))
    # 规格书3.2: DBSCAN聚类（eps=0.15×ATR×(1+BBW), minPts=4）
    def _dbscan_cluster(vals):
        if len(vals) < min_cluster:
            return None, 0, []
        arr = sorted(vals, key=lambda x: x[0])
        n = len(arr)
        # 邻接矩阵：每个点找到eps内的邻居
        neighbors = []
        for i in range(n):
            nbrs = []
            for j in range(n):
                if i != j and abs(arr[i][0] - arr[j][0]) <= eps:
                    nbrs.append(j)
            neighbors.append(nbrs)
        # 标记核心点（minPts个邻居 = min_cluster个点含自身）
        core = [False] * n
        for i in range(n):
            if len(neighbors[i]) >= min_cluster - 1:
                core[i] = True
        # 通过核心点连通性形成簇（包含边界点：非核心邻居也加入簇）
        visited = [False] * n
        best_cluster = None
        best_count = 0
        for i in range(n):
            if visited[i]:
                continue
            if not core[i]:
                visited[i] = True
                continue
            # BFS扩展簇（核心点+边界点）
            cluster = [i]
            visited[i] = True
            q = [i]
            while q:
                cur = q.pop(0)
                for nb in neighbors[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        cluster.append(nb)
                        if core[nb]:
                            q.append(nb)
            if len(cluster) >= min_cluster and len(cluster) > best_count:
                prices = [arr[m][0] for m in cluster]
                best_cluster = float(np.median(prices))
                best_count = len(cluster)
                _best_members = prices
        # 返回簇中位数、簇大小、完整簇成员列表（供后续验证使用）
        return best_cluster, best_count, _best_members if best_cluster is not None else []
    res_level, res_cnt, _ = _dbscan_cluster(sh)
    sup_level, sup_cnt, _ = _dbscan_cluster(sl)
    # 穿透率验证（使用有效close + spike_flags过滤穿透标记）
    p_start = max(0, idx - penetration_bars)
    p_end = idx
    def _penetration_ok(level, is_resistance):
        if level is None:
            return False, 0, 0
        cross = 0
        filtered_cross = 0
        total = max(1, p_end - p_start)
        for i in range(p_start, p_end):
            penetrated = False
            if is_resistance:
                # 规格书3.2.3: 有效穿透判定 — close突破 或 eff_high突破边界
                if float(close[i]) > level * (1 + penetration_price):
                    penetrated = True
                elif eff_high is not None and i < len(eff_high) and eff_high[i] > level * (1 + penetration_price):
                    penetrated = True
            else:
                if float(close[i]) < level * (1 - penetration_price):
                    penetrated = True
                elif eff_low is not None and i < len(eff_low) and eff_low[i] < level * (1 - penetration_price):
                    penetrated = True
            if penetrated:
                cross += 1
                # 附录A Q8: 穿透来自过滤后K线则计数
                if spike_flags is not None and i < len(spike_flags) and spike_flags[i]:
                    filtered_cross += 1
        rate = cross / total
        return rate <= penetration_threshold, rate, filtered_cross
    res_ok, res_rate, res_filtered = _penetration_ok(res_level, True)
    sup_ok, sup_rate, sup_filtered = _penetration_ok(sup_level, False)
    # 附录A Q8: 过滤穿透计数记录（供调用方做风险调整）
    if not res_ok:
        res_level = None
        res_cnt = 0
    if not sup_ok:
        sup_level = None
        sup_cnt = 0
    extra = {
        'res_is_cluster': res_cnt >= 3,
        'sup_is_cluster': sup_cnt >= 3,
        'res_cluster_count': res_cnt,
        'sup_cluster_count': sup_cnt,
        'res_filtered_cross': res_filtered,
        'sup_filtered_cross': sup_filtered,
        'search_start': start,
        'search_end': idx,
        'res_penetration_rate': res_rate,
        'sup_penetration_rate': sup_rate,
        'res_filtered_penetrations': res_filtered,
        'sup_filtered_penetrations': sup_filtered,
    }
    return res_level, sup_level, extra


def detect_triangle_proper(high_arr, low_arr, atr, idx, lookback=45):
    start = max(0, idx - lookback)
    if idx - start < 20:
        return None
    atr_seg = atr[start:idx+1]
    min_amp = (atr_seg.sum() / len(atr_seg)) * 0.1
    left = right = 3
    end = min(idx - right, len(high_arr) - right - 1)
    if end <= start + left:
        return None
    rng = np.arange(start + left, end)
    from numpy.lib.stride_tricks import sliding_window_view
    h7 = sliding_window_view(high_arr, left + right + 1)
    l7 = sliding_window_view(low_arr, left + right + 1)
    win_idx = rng - left
    win_max_h = np.max(h7[win_idx], axis=1)
    win_min_l = np.min(l7[win_idx], axis=1)
    is_high = (high_arr[rng] == win_max_h) & (high_arr[rng] - np.maximum(high_arr[rng-left], high_arr[rng+right]) >= min_amp)
    is_low = (low_arr[rng] == win_min_l) & (np.minimum(low_arr[rng-left], low_arr[rng+right]) - low_arr[rng] >= min_amp)
    sw_highs = [(float(high_arr[i]), int(i)) for i in rng[is_high]]
    sw_lows = [(float(low_arr[i]), int(i)) for i in rng[is_low]]
    if len(sw_highs) < 3 or len(sw_lows) < 3:
        return None

    def _fast_ls(xv, yv):
        nf = float(len(xv)); sx = float(xv.sum()); sy = float(yv.sum())
        sxx = float((xv*xv).sum()); sxy = float((xv*yv).sum())
        d = nf*sxx - sx*sx
        if abs(d) < 1e-15:
            return 0, 0, 0
        m = (nf*sxy - sx*sy) / d
        b = (sy - m*sx) / nf
        yp = m*xv + b
        ssr = float(((yv-yp)**2).sum())
        sst = float(((yv-float(yv.mean()))**2).sum())
        r2 = 1 - ssr/sst if sst > 0 else 0
        return m, b, r2

    best = None
    for nh in range(3, min(4, len(sw_highs)+1)):
        highs = sw_highs[-nh:]
        if not all(highs[i][0] > highs[i+1][0] for i in range(nh-1)):
            continue
        xh = np.array([p[1] for p in highs], dtype=float)
        yh = np.array([p[0] for p in highs], dtype=float)
        slope_h, intercept_h, r2_h = _fast_ls(xh, yh)
        if slope_h >= -0.001 or r2_h < 0.5:
            continue
        for nl in range(3, min(4, len(sw_lows)+1)):
            lows = sw_lows[-nl:]
            if not all(lows[i][0] < lows[i+1][0] for i in range(nl-1)):
                continue
            xl = np.array([p[1] for p in lows], dtype=float)
            yl = np.array([p[0] for p in lows], dtype=float)
            slope_l, intercept_l, r2_l = _fast_ls(xl, yl)
            if slope_l <= 0.001 or r2_l < 0.5:
                continue
            up_s = slope_h * start + intercept_h
            lo_s = slope_l * start + intercept_l
            up_e = slope_h * idx + intercept_h
            lo_e = slope_l * idx + intercept_l
            gap_s = up_s - lo_s
            gap_e = up_e - lo_e
            min_gap = np.mean(atr[start:idx+1]) * 0.3
            # 末端不交易：收窄到10%以下或gap太小（<0.5*ATR）
            if gap_s <= 0 or gap_e <= 0 or gap_e >= gap_s:
                continue
            gap_ratio = gap_e / gap_s
            if gap_ratio < 0.1 or gap_e < np.mean(atr[start:idx+1]) * 0.5:
                continue
            if gap_ratio > 0.2 or gap_e < min_gap:
                continue
            score = (nh+nl)*10 + (r2_h+r2_l)*5
            if best is None or score > best['score']:
                best = {
                    'slope_h': slope_h, 'inter_h': intercept_h,
                    'slope_l': slope_l, 'inter_l': intercept_l,
                    'nh': nh, 'nl': nl, 'r2_h': r2_h, 'r2_l': r2_l,
                    'up_at_ci': up_e, 'lo_at_ci': lo_e,
                    'up_at_start': up_s, 'lo_at_start': lo_s,
                    'gap_start': gap_s, 'gap_end': gap_e, 'score': score
                }
    return best


def classify_morphology_doc(high, low, idx, lookback=120):
    """文档第五章：分形极值+线性回归形态分类
    返回 (TrendType, slope_h_pct, slope_l_pct)"""
    start = max(0, idx - lookback + 1)
    h_seg = high[start:idx + 1]
    l_seg = low[start:idx + 1]
    if len(h_seg) < 20:
        return TrendType.UNKNOWN, 0.0, 0.0
    # 分形极值提取（左2右2）
    n = len(h_seg)
    highs = []
    lows = []
    for i in range(2, n - 2):
        is_high = True
        is_low = True
        for j in range(1, 3):
            if h_seg[i] <= h_seg[i - j] or h_seg[i] <= h_seg[i + j]:
                is_high = False
            if l_seg[i] >= l_seg[i - j] or l_seg[i] >= l_seg[i + j]:
                is_low = False
        if is_high:
            highs.append((i, h_seg[i]))
        if is_low:
            lows.append((i, l_seg[i]))
    if len(highs) < 2 or len(lows) < 2:
        return TrendType.UNKNOWN, 0.0, 0.0
    h_idx = np.array([x[0] for x in highs], dtype=float)
    h_val = np.array([x[1] for x in highs])
    l_idx = np.array([x[0] for x in lows], dtype=float)
    l_val = np.array([x[1] for x in lows])
    # 线性回归
    def _lr(x, y):
        nn = len(x)
        if nn < 3:
            return 0.0, 0.0, 0.0
        sx, sy = x.sum(), y.sum()
        sxx, sxy = (x * x).sum(), (x * y).sum()
        denom = nn * sxx - sx * sx
        if abs(denom) < 1e-12:
            return 0.0, sy / max(nn, 1), 0.0
        m = (nn * sxy - sx * sy) / denom
        b = (sy - m * sx) / nn
        y_hat = m * x + b
        ss_res = ((y - y_hat) ** 2).sum()
        y_var = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(y_var, 1e-12) if y_var > 1e-12 else 0.0
        return m, b, max(r2, 0.0)
    slope_h, _, _ = _lr(h_idx, h_val)
    slope_l, _, _ = _lr(l_idx, l_val)
    norm = h_val.mean() if h_val.mean() > 1e-12 else 1.0
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


# ── 趋势状态锁定（防横跳，规格书4.5节） ──
# 全局单例锁（回测逐symbol执行，不存在并发冲突）
_trend_lock = {
    'trend': None,          # 锁定的趋势类型（TrendType枚举）
    'expiry_idx': 0,        # 锁到期bar索引
    'pivot_price': 0.0,     # 锁定时的关键枢轴价格（最近谷底/峰顶）
    'pivot_idx': 0,         # 枢轴bar索引
}
TREND_LOCK_BARS = 3

def _is_lock_active(idx):
    """检查锁定状态。锁有效期间返回锁定的趋势，越过强制解锁条件则解锁"""
    global _trend_lock
    lock = _trend_lock
    if lock['trend'] is None:
        return None
    if idx > lock['expiry_idx'] + 5:  # 过期后宽限5根再强制清除
        lock['trend'] = None
        return None
    return lock['trend']

def _set_trend_lock(trend, idx, pivot_price, pivot_idx):
    """设置趋势锁定"""
    global _trend_lock
    _trend_lock['trend'] = trend
    _trend_lock['expiry_idx'] = idx + TREND_LOCK_BARS
    _trend_lock['pivot_price'] = pivot_price
    _trend_lock['pivot_idx'] = pivot_idx

LOCK_BREAK_TOLERANCE = 0.015  # 强制解锁容差1.5%

def _check_forced_unlock(idx, low, high):
    """检查是否触发强制解锁（价格击穿枢轴）"""
    global _trend_lock
    lock = _trend_lock
    if lock['trend'] is None or lock['pivot_idx'] <= 0:
        return False
    if lock['trend'] == TrendType.UPTREND:
        # 价格跌破锁定时的最近谷底（含1.5%容差）→ 强制解锁
        if low[idx] < lock['pivot_price'] * (1 - LOCK_BREAK_TOLERANCE):
            lock['trend'] = None
            return True
    elif lock['trend'] == TrendType.DOWNTREND:
        # 价格升破锁定时的最近峰顶（含1.5%容差）→ 强制解锁
        if high[idx] > lock['pivot_price'] * (1 + LOCK_BREAK_TOLERANCE):
            lock['trend'] = None
            return True
    return False


def analyze_trend_fast(close, high, low, idx, adx, plus_di, minus_di, lookback=20,
                       consolidation_period=80, adx_threshold=20, price_change_threshold=2.0,
                       atr=None, return_sr_extra=False, ema20=None, ema50=None,
                       eff_high=None, eff_low=None, spike_flags=None,
                       atr_struct=None, atr_struct_ma100=None):
    """
    宏观趋势识别引擎 V7.0（基于15mTupokaifa.md规格书）
    仲裁优先级：TRIANGLE > UPTREND > DOWNTREND > CONSOLIDATION > UNKNOWN
    核心原则：结构定义趋势（极值点排列），指标不定义趋势（ADX仅辅助过滤）
    """
    info = {}

    # ── 全局常量（对应.env规格，硬编码确保一致性） ──
    SWING_LOOKBACK = 3
    SWING_REBOUND_MULT = 0.5
    SHORT_REBOUND_MULT = 0.4
    UNKNOWN_VOL_ATR = 0.40
    TRIANGLE_R2 = 0.60
    TRIANGLE_LOOKBACK = 24

    # ── 4.1: 数据有效性过滤 ──
    if idx < 30 or np.isnan(adx[idx]) or idx >= len(close) - 3:
        info['filter'] = 'insufficient_data'
        return TrendType.UNKNOWN, info

    # ── 4.1b: HIGH_VOLATILITY 检测（规格书 P1 优先级）──
    # ATR14 / Close ≥ max(2.5%, 0.3×ATR_struct_ma100/Close) 时强制停开仓
    atr_ratio = (atr[idx] / close[idx] * 100) if atr is not None and idx < len(atr) and close[idx] > 0 else 0
    vol_threshold = 2.5
    if atr_struct_ma100 is not None and idx < len(atr_struct_ma100) and close[idx] > 0:
        vol_threshold = max(2.5, 0.3 * atr_struct_ma100[idx] / close[idx] * 100)
    if atr_ratio >= vol_threshold:
        info['filter'] = 'high_volatility'
        info['atr_ratio'] = atr_ratio
        info['vol_threshold'] = vol_threshold
        return TrendType.HIGH_VOLATILITY, info

    # ── 4.2: 波动率休眠检测（规格书 Step 1）──
    body_short = np.mean(high[idx - 9:idx + 1] - low[idx - 9:idx + 1])
    body_long = np.mean(high[max(0, idx - 49):idx + 1] - low[max(0, idx - 49):idx + 1])
    if body_long > 0 and body_short < body_long * UNKNOWN_VOL_ATR:
        info['filter'] = 'low_volatility'
        return TrendType.UNKNOWN, info

    # ── 4.3: 有效极值点提取（核心滤波） ──
    sw_highs = []
    sw_lows = []
    swing_start = max(3, idx - 60)
    swing_end = min(idx - SWING_LOOKBACK - 1, len(high) - 4)

    for i in range(swing_start + SWING_LOOKBACK, swing_end + 1):
        if i < 0 or i >= len(high) - 1:
            continue
        tp = (float(high[i]) + float(low[i]) + float(close[i])) / 3.0

        left_start = max(0, i - SWING_LOOKBACK)
        left_end = i
        right_start = i + 1
        right_end = min(len(high), i + SWING_LOOKBACK + 1)

        tp_left = [(float(high[j]) + float(low[j]) + float(close[j])) / 3.0 for j in range(left_start, left_end)]
        tp_right = [(float(high[j]) + float(low[j]) + float(close[j])) / 3.0 for j in range(right_start, right_end)]

        if len(tp_left) < SWING_LOOKBACK or len(tp_right) < SWING_LOOKBACK:
            continue

        min_left = min(tp_left)
        max_left = max(tp_left)
        min_right = min(tp_right)
        max_right = max(tp_right)

        # 谷底验证（注意：用idx限制未来数据，实盘中未来K线未知）
        if tp < min_left and tp < min_right:
            future_end = min(idx, i + 4)  # 锁住实盘不可见的未来K线
            rebound = float(np.max(high[i + 1:future_end + 1])) - tp
            atr_val = float(atr[i]) if atr is not None and i < len(atr) else 0
            if rebound > SWING_REBOUND_MULT * atr_val:
                sw_lows.append((float(tp), i))

        # 峰顶验证
        if tp > max_left and tp > max_right:
            future_end = min(idx, i + 4)
            drop = tp - float(np.min(low[i + 1:future_end + 1]))
            atr_val = float(atr[i]) if atr is not None and i < len(atr) else 0
            if drop > SHORT_REBOUND_MULT * atr_val:
                sw_highs.append((float(tp), i))

    # 基础信息（供所有分支共用）
    info['search_start'] = swing_start
    info['search_end'] = idx
    info['swing_highs'] = [(p, j) for (p, j) in sw_highs]
    info['swing_lows'] = [(p, j) for (p, j) in sw_lows]

    # ── 4.4: 仲裁优先级 ──

    # ========== 优先级1: TRIANGLE ==========
    if len(sw_highs) >= 3 and len(sw_lows) >= 3:
        h_prices_arr = [p for p, _ in sw_highs[-3:]]
        h_idxs_arr = [j for _, j in sw_highs[-3:]]
        l_prices_arr = [p for p, _ in sw_lows[-3:]]
        l_idxs_arr = [j for _, j in sw_lows[-3:]]

        # 使用线性代数直接对3个点做回归（_reg_slope_pct要求连续数组）
        def _simple_reg(ys, xs):
            n = len(xs)
            sx = float(np.sum(xs)); sy = float(np.sum(ys))
            sxx = float(np.sum(xs * xs)); sxy = float(np.sum(xs * ys))
            denom = n * sxx - sx * sx
            if abs(denom) < 1e-15:
                return 0, 0
            m = (n * sxy - sx * sy) / denom
            y_pred = m * xs + (sy - m * sx) / n
            ss_res = float(np.sum((ys - y_pred) ** 2))
            ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            return float(m), r2

        h_slope, h_r2 = _simple_reg(np.array(h_prices_arr), np.array(h_idxs_arr, dtype=float))
        l_slope, l_r2 = _simple_reg(np.array(l_prices_arr), np.array(l_idxs_arr, dtype=float))

        if h_slope < -0.001 and h_r2 > TRIANGLE_R2 and l_slope > 0.001 and l_r2 > TRIANGLE_R2:
            up_at_ci = (l_slope * (idx - l_idxs_arr[0]) + l_prices_arr[0])
            lo_at_ci = (h_slope * (idx - h_idxs_arr[0]) + h_prices_arr[0])
            if up_at_ci < lo_at_ci:  # 上沿低于下沿（三角形未突破）
                info['triangle'] = {
                    'slope_up': l_slope,
                    'slope_lo': h_slope,
                    'up_at_ci': float(up_at_ci),
                    'lo_at_ci': float(lo_at_ci),
                    'up_price': l_prices_arr[-1],
                    'lo_price': h_prices_arr[-1],
                }
                return TrendType.TRIANGLE, info

    # ========== 优先级2: UPTREND ==========
    if len(sw_lows) >= 3:
        lows_sorted = sorted(sw_lows, key=lambda x: x[1])
        three_lows = lows_sorted[-3:]
        p0, j0 = three_lows[0]
        p1, j1 = three_lows[1]
        p2, j2 = three_lows[2]

        # 条件：谷底依次抬高 + 收盘在P2上方 + EMA20>EMA50（规格书要求）
        ema_ok = True
        if ema20 is not None and ema50 is not None and idx < len(ema20) and idx < len(ema50):
            ema_ok = ema20[idx] > ema50[idx]
        if p0 < p1 < p2 and close[idx] > p2 and ema_ok:
            # 状态锁定检查在此不做（外部调用者负责）
            # 构建基石线和操作线
            vreversal_start = max(0, idx - 60)
            vreversal_seg = low[vreversal_start:idx + 1]
            if len(vreversal_seg) > 0:
                tl_anchor_val = float(vreversal_seg.min())
                tl_anchor_idx = vreversal_start + int(np.argmin(vreversal_seg))
            else:
                tl_anchor_val = float(low[idx])
                tl_anchor_idx = idx

            # 基石线: V反极值点 → 最近谷底(sw_low[-1])
            fl_start_price = tl_anchor_val
            fl_start_idx = tl_anchor_idx
            fl_end_price = p2
            fl_end_idx = j2
            if fl_end_idx > fl_start_idx:
                fl_slope = (fl_end_price - fl_start_price) / (fl_end_idx - fl_start_idx)
            else:
                fl_slope = 0.001

            info['trend_line_base_idx'] = fl_start_idx
            info['trend_line_base_val'] = fl_start_price
            info['trend_line_slope'] = fl_slope
            info['trend_line_second_idx'] = fl_end_idx
            info['trend_line_second_val'] = fl_end_price

            info['foundation_line'] = {
                'idx': fl_start_idx,
                'price': fl_start_price,
                'slope': fl_slope,
            }

            # 操作线: 连接最近两个谷底
            if len(lows_sorted) >= 2:
                op_p0, op_j0 = lows_sorted[-2]
                op_p1, op_j1 = lows_sorted[-1]
                if op_j1 > op_j0:
                    op_slope = (op_p1 - op_p0) / (op_j1 - op_j0)
                else:
                    op_slope = fl_slope

                if abs(op_slope) > abs(fl_slope):
                    info['operating_line'] = {
                        'idx': op_j0,
                        'price': op_p0,
                        'slope': op_slope,
                        'second_idx': op_j1,
                        'second_val': op_p1,
                    }
                else:
                    info['operating_line'] = {
                        'idx': fl_start_idx,
                        'price': fl_start_price,
                        'slope': fl_slope,
                        'second_idx': fl_end_idx,
                        'second_val': fl_end_price,
                    }
            else:
                info['operating_line'] = {
                    'idx': fl_start_idx,
                    'price': fl_start_price,
                    'slope': fl_slope,
                    'second_idx': fl_end_idx,
                    'second_val': fl_end_price,
                }

            # 操作线摩擦失效检测（规格书 §4）：连续8Bar距操作线<0.1×ATR14
            friction_decayed = False
            op_line = info.get('operating_line')
            if op_line and atr is not None:
                op_idx = op_line.get('idx', 0)
                op_price = op_line.get('price', 0)
                op_slope = op_line.get('slope', 0)
                if op_idx > 0 and op_price > 0:
                    close_count = 0
                    for check_i in range(max(op_idx, idx - 15), idx + 1):
                        if check_i < 0 or check_i >= len(close) or check_i >= len(atr):
                            continue
                        op_val = op_price + op_slope * (check_i - op_idx)
                        if op_val > 0 and atr[check_i] > 0:
                            dist_pct = abs(close[check_i] - op_val) / max(op_val, 0.001) * 100
                            threshold = 0.1 * atr[check_i] / max(op_val, 0.001) * 100
                            if dist_pct < threshold:
                                close_count += 1
                            else:
                                close_count = 0
                        if close_count >= 8:
                            friction_decayed = True
                            break
            info['operating_line_friction_decayed'] = friction_decayed

            # TR入场附加数据
            info['p1_price'] = p2
            info['p1_idx'] = j2
            # H1: 规格书要求 — P1后扫描连续3Bar振幅<0.3%的平台
            h1_price = 0.0
            h1_idx = 0
            scan_start = j2 + 1
            scan_end = min(idx + 1, len(high))
            platform_found = False
            for scan_i in range(scan_start, scan_end - 2):
                if scan_i < 0 or scan_i >= len(high):
                    continue
                amp1 = (high[scan_i] - low[scan_i]) / max(close[scan_i], 0.001) * 100
                amp2 = (high[scan_i+1] - low[scan_i+1]) / max(close[scan_i+1], 0.001) * 100
                amp3 = (high[scan_i+2] - low[scan_i+2]) / max(close[scan_i+2], 0.001) * 100
                if max(amp1, amp2, amp3) < 0.3:
                    # 取3根bar的最高点
                    plat_high = max(high[scan_i:scan_i+3])
                    if plat_high > h1_price:
                        h1_price = float(plat_high)
                        h1_idx = scan_i + int(np.argmax(high[scan_i:scan_i+3]))
                        platform_found = True
            if not platform_found:
                h1_seg = high[j2:scan_end]
                if len(h1_seg) > 0:
                    h1_price = float(h1_seg.max())
                    h1_idx = j2 + int(np.argmax(h1_seg))
            info['h1_price'] = h1_price
            info['h1_idx'] = h1_idx
            info['h1_platform_found'] = platform_found
            info['touch_count'] = 0
            # 触碰次数统计：只统计最近30根K线，避免从趋势线创建开始累计导致天量触踫
            LOOKBACK_TOUCH = 30
            if atr is not None:
                tc_count = 0
                last_tc_bar = -100
                fl = info.get('foundation_line')
                if fl:
                    fl_idx = fl.get('idx', 0)
                    fl_price = fl.get('price', 0)
                    fl_slope = fl.get('slope', 0)
                    scan_start = max(idx - LOOKBACK_TOUCH, fl_idx + 5, j2)
                    for j in range(scan_start, idx):
                        if j < len(atr) and atr[j] > 0:
                            trend_p = fl_price + fl_slope * (j - fl_idx) if fl_slope else fl_price
                            if trend_p > 0:
                                dist_line = abs(float(close[j]) - trend_p)
                                is_touch = dist_line <= float(atr[j]) * 0.15
                                band_low = trend_p - float(atr[j]) * 0.10
                                band_high = trend_p + float(atr[j]) * 0.10
                                is_touch_band = band_low <= float(close[j]) <= band_high
                                if (is_touch or is_touch_band) and (j - last_tc_bar) >= 3:
                                    tc_count += 1
                                    last_tc_bar = j
                    info['touch_count'] = tc_count

            # S/R（UPTREND下同方向S/R）
            if len(sw_highs) >= 2:
                res = float(max(sw_highs[-1][0], sw_highs[-2][0]))
                info['resistance'] = res
            else:
                info['resistance'] = None
            if len(sw_lows) >= 2:
                sup = float(min(sw_lows[-1][0], sw_lows[-2][0]))
                info['support'] = sup
            else:
                info['support'] = None

            info['zone_high'] = float(high[max(0, idx - 60):idx + 1].max())
            info['zone_low'] = float(low[max(0, idx - 60):idx + 1].min())
            _set_trend_lock(TrendType.UPTREND, idx, p2, j2)
            return TrendType.UPTREND, info

    # ========== 优先级3: DOWNTREND ==========
    if len(sw_highs) >= 3:
        highs_sorted = sorted(sw_highs, key=lambda x: x[1])
        three_highs = highs_sorted[-3:]
        p0, j0 = three_highs[0]
        p1, j1 = three_highs[1]
        p2, j2 = three_highs[2]

        # 条件：峰顶依次降低 + 收盘在P2下方 + EMA20<EMA50（规格书要求）
        ema_ok = True
        if ema20 is not None and ema50 is not None and idx < len(ema20) and idx < len(ema50):
            ema_ok = ema20[idx] < ema50[idx]
        if p0 > p1 > p2 and close[idx] < p2 and ema_ok:
            vreversal_start = max(0, idx - 60)
            vreversal_seg = high[vreversal_start:idx + 1]
            if len(vreversal_seg) > 0:
                tl_anchor_val = float(vreversal_seg.max())
                tl_anchor_idx = vreversal_start + int(np.argmax(vreversal_seg))
            else:
                tl_anchor_val = float(high[idx])
                tl_anchor_idx = idx

            # 基石线: V反极值点 → 最近峰顶
            fl_start_price = tl_anchor_val
            fl_start_idx = tl_anchor_idx
            fl_end_price = p2
            fl_end_idx = j2
            if fl_end_idx > fl_start_idx:
                fl_slope = (fl_end_price - fl_start_price) / (fl_end_idx - fl_start_idx)
            else:
                fl_slope = -0.001

            info['trend_line_base_idx'] = fl_start_idx
            info['trend_line_base_val'] = fl_start_price
            info['trend_line_slope'] = fl_slope
            info['trend_line_second_idx'] = fl_end_idx
            info['trend_line_second_val'] = fl_end_price

            info['foundation_line'] = {
                'idx': fl_start_idx,
                'price': fl_start_price,
                'slope': fl_slope,
            }

            # 操作线: 连接最近两个峰顶
            if len(highs_sorted) >= 2:
                op_p0, op_j0 = highs_sorted[-2]
                op_p1, op_j1 = highs_sorted[-1]
                if op_j1 > op_j0:
                    op_slope = (op_p1 - op_p0) / (op_j1 - op_j0)
                else:
                    op_slope = fl_slope

                if abs(op_slope) > abs(fl_slope):
                    info['operating_line'] = {
                        'idx': op_j0,
                        'price': op_p0,
                        'slope': op_slope,
                        'second_idx': op_j1,
                        'second_val': op_p1,
                    }
                else:
                    info['operating_line'] = {
                        'idx': fl_start_idx,
                        'price': fl_start_price,
                        'slope': fl_slope,
                        'second_idx': fl_end_idx,
                        'second_val': fl_end_price,
                    }
            else:
                info['operating_line'] = {
                    'idx': fl_start_idx,
                    'price': fl_start_price,
                    'slope': fl_slope,
                    'second_idx': fl_end_idx,
                    'second_val': fl_end_price,
                }

            # TR入场附加数据
            info['p1_price'] = p2
            info['p1_idx'] = j2
            # L1: P1到当前之间的最低点（DOWNTREND）
            l1_seg = low[j2:min(idx + 1, len(low))]
            if len(l1_seg) > 0:
                info['h1_price'] = float(l1_seg.min())
                info['h1_idx'] = j2 + int(np.argmin(l1_seg))
            else:
                info['h1_price'] = 0
                info['h1_idx'] = 0
            info['touch_count'] = 0
            # DOWNTREND触碰次数统计：只统计最近30根K线
            LOOKBACK_TOUCH = 30
            if atr is not None:
                tc_count = 0
                last_tc_bar = -100
                fl = info.get('foundation_line')
                if fl:
                    fl_idx = fl.get('idx', 0)
                    fl_price = fl.get('price', 0)
                    fl_slope = fl.get('slope', 0)
                    scan_start = max(idx - LOOKBACK_TOUCH, fl_idx + 5, j2)
                    for j in range(scan_start, idx):
                        if j < len(atr) and atr[j] > 0:
                            trend_p = fl_price + fl_slope * (j - fl_idx) if fl_slope else fl_price
                            if trend_p > 0:
                                dist_line = abs(float(close[j]) - trend_p)
                                is_touch = dist_line <= float(atr[j]) * 0.15
                                band_low = trend_p - float(atr[j]) * 0.10
                                band_high = trend_p + float(atr[j]) * 0.10
                                is_touch_band = band_low <= float(close[j]) <= band_high
                                if (is_touch or is_touch_band) and (j - last_tc_bar) >= 3:
                                    tc_count += 1
                                    last_tc_bar = j
                    info['touch_count'] = tc_count

            # S/R
            if len(sw_lows) >= 2:
                sup = float(min(sw_lows[-1][0], sw_lows[-2][0]))
                info['support'] = sup
            else:
                info['support'] = None
            if len(sw_highs) >= 2:
                res = float(max(sw_highs[-1][0], sw_highs[-2][0]))
                info['resistance'] = res
            else:
                info['resistance'] = None

            info['zone_high'] = float(high[max(0, idx - 60):idx + 1].max())
            info['zone_low'] = float(low[max(0, idx - 60):idx + 1].min())
            _set_trend_lock(TrendType.DOWNTREND, idx, p2, j2)
            return TrendType.DOWNTREND, info

    # ========== 优先级4: CONSOLIDATION（有S/R） ==========
    sr_res, sr_sup, sr_extra = find_horizontal_SR(high, low, close, idx, atr,
                                                   eff_high=eff_high, eff_low=eff_low,
                                                   spike_flags=spike_flags)
    if sr_sup is not None and sr_res is not None and \
       abs(sr_res - sr_sup) > (float(atr[idx]) * 2 if atr is not None else 0):
        info['resistance'] = sr_res
        info['support'] = sr_sup
        if return_sr_extra and sr_extra:
            info['res_is_cluster'] = sr_extra.get('res_is_cluster', False)
            info['sup_is_cluster'] = sr_extra.get('sup_is_cluster', False)
            info['res_cluster_count'] = sr_extra.get('res_cluster_count', 0)
            info['sup_cluster_count'] = sr_extra.get('sup_cluster_count', 0)
            info['swing_highs'] = sr_extra.get('swing_highs', [])
            info['swing_lows'] = sr_extra.get('swing_lows', [])
            info['res_members'] = sr_extra.get('res_members', [])
            info['sup_members'] = sr_extra.get('sup_members', [])
            info['search_start'] = sr_extra.get('search_start')
            info['search_end'] = sr_extra.get('search_end')
            info['segments'] = sr_extra.get('segments', [])
        cons_start = max(0, idx - consolidation_period)
        info['zone_high'] = float(high[cons_start:idx + 1].max())
        info['zone_low'] = float(low[cons_start:idx + 1].min())
        info['cons_trend_dir'] = 'neutral'
        # 状态锁定：锁有效时覆盖CONSOLIDATION
        locked = _is_lock_active(idx)
        if locked is not None and not _check_forced_unlock(idx, low, high):
            return locked, info
        return TrendType.CONSOLIDATION, info

    # ========== 优先级5: UNKNOWN ==========
    # 再次尝试TRIANGLE（无S/R时的fallback）
    tri = detect_triangle_proper(high, low, atr, idx, lookback=45)
    if tri is not None:
        locked = _is_lock_active(idx)
        if locked is not None and not _check_forced_unlock(idx, low, high):
            return locked, info
        info['triangle'] = tri
        info['search_start'] = max(0, idx - 45)
        info['search_end'] = idx
        return TrendType.TRIANGLE, info

    locked = _is_lock_active(idx)
    if locked is not None and not _check_forced_unlock(idx, low, high):
        return locked, info

    info['zone_high'] = float(high[max(0, idx - 20):idx + 1].max())
    info['zone_low'] = float(low[max(0, idx - 20):idx + 1].min())
    return TrendType.UNKNOWN, info


# =====================================================================
# ⚠️  双副本架构说明 (DUAL-COPY ARCHITECTURE)
# 本函数(check_exit)与 run_final.py inline退出逻辑是双副本!
# 两份代码逻辑完全一致，通过同步测试保护:
#   python framework/backtest/test_exit_sync.py
#
# 为什么不能合并?
#   回测: 每根5m bar调一次, 39币×360k根=1400万次调用
#   Python函数调用开销(~200ns)×1400万=3s额外开销, 加上dict lookup
#   实际慢10x(350s→3500s+), 不可接受
#   实盘: 每5分钟最多触发一次, 函数调用开销可忽略
#
# 所以: 回测用run_final.py inline(快), 实盘用check_exit()(干净)
# 修改此处必须同步修改 framework/backtest/run_final.py (lines ~202-320)
# =====================================================================
def check_exit(
    pos: PositionSide,
    ep: float,
    sl: float,
    lev: int,
    cp: float,
    hb: int,
    hb15: int,
    mx: float,
    mn: float,
    st: str,
    tt: TrendType,
    eidx: int,
    eidx15: int,
    eres: float,
    esup: float,
    es: ExitState,
    ci: int,
    i: int,
    avgv_ci: float,
    v15s: float,
    v5_i: float,
    c5_i: float,
    c5_prev: float,
    l5_i: float,
    h5_i: float,
    c15s: float,
    pr: float = 1.0,
    sr_room: float = 99.0,
    fl_info: dict = None,
    op_info: dict = None,
) -> Tuple[bool, str, float, float, float, float, float, float]:
    """
    run_final.py 退出检查逻辑（9级）— 实盘唯一退出路径
    支持 i=-1 (live mode): always run PnL update and all exit checks
    支持 i>=0 (backtest): 5m-level on every i, 15m-level at i%%3==2

    ⚠️  双副本: 此函数与 run_final.py inline退出逻辑完全一致
    回测不用此函数(10x慢), 仅实盘(strategy.py)调用
    修改逻辑必须同步两处, 用 test_exit_sync.py 验证

    Returns: (should_close, exit_reason, close_ratio, new_sl, new_mx, new_mn, pnl, exit_p)
    - should_close: 是否平仓
    - exit_reason: 退出原因字符串
    - close_ratio: 平仓比例 (0.0~1.0)
    - new_sl, new_mx, new_mn: 更新后的止损/最大盈利/最小盈利
    - pnl: 本bar盈亏% (用于账户更新)
    - exit_p: 触发价格 (用于TradeResult记录)

    设计决策说明:
    1. 液化检查用elif链: 当TREND_BE匹配时跳过, 因为TREND_BE已收紧SL, 5m区间触到爆仓价时SL更近会先触发
    2. 阶梯close_ratio除以pr: pr<1时(已部分平仓), 需放大cr以平剩余仓位中的目标比例
    3. sr_room: 入场时S/R距离, 用于ladder CONSOLIDATION分支判断
    4. hrt覆盖: 阶梯触发后, 若后续velocity/trailing/entry_stop平仓, er改为'阶梯止盈'
    """
    if ep <= 0 or lev <= 0:
        return False, "", 1.0, sl, mx, mn, 0.0, cp

    live = i < 0
    c = _get_exit_cfg()
    vel_floor = c['vel_floor']
    vel_min_mx = c['vel_min_mx']
    vel_floor_tr = c['vel_floor_tr']
    vel_tr_mx_cap = c['vel_tr_mx_cap']
    vel_window = c['vel_window']
    vel_floor_multi = c['vel_floor_multi']
    trail_on = True
    trail_act = c['trail_act']
    trail_dist = c['trail_dist']
    trail_tight_act = c['trail_tight_act']
    trail_tight_dist = c['trail_tight_dist']
    trail_act_tr = c['trail_act_tr']
    trail_dist_tr = c['trail_dist_tr']
    slippage = c['slippage']
    funding = c['funding']
    rebound_opp = c['rebound_opp']
    rebound_be = c['rebound_be']
    ladder_vs = c['ladder_vs']
    ladder_peak_uptrend = c['ladder_peak_uptrend']
    ladder_peak_rebound = c['ladder_peak_rebound']
    ladder_peak_tr = c['ladder_peak_tr']
    ladder_dd_t1_uptrend = c['ladder_dd_t1_uptrend']
    ladder_dd_t2_uptrend = c['ladder_dd_t2_uptrend']
    ladder_dd_t3_uptrend = c['ladder_dd_t3_uptrend']
    ladder_dd_t4_uptrend = c['ladder_dd_t4_uptrend']
    ladder_dd_t5_uptrend = c['ladder_dd_t5_uptrend']
    ladder_dd_t1_rebound = c['ladder_dd_t1_rebound']
    ladder_dd_t2_rebound = c['ladder_dd_t2_rebound']
    ladder_dd_t3_rebound = c['ladder_dd_t3_rebound']
    ladder_dd_t4_rebound = c['ladder_dd_t4_rebound']
    ladder_dd_t5_rebound = c['ladder_dd_t5_rebound']
    ladder_dd_t1_tr = c['ladder_dd_t1_tr']
    ladder_dd_t2_tr = c['ladder_dd_t2_tr']
    ladder_dd_t3_tr = c['ladder_dd_t3_tr']
    ladder_dd_t4_tr = c['ladder_dd_t4_tr']
    ladder_dd_t5_tr = c['ladder_dd_t5_tr']
    ladder_close_t1 = c['ladder_close_t1']
    ladder_close_t2 = c['ladder_close_t2']
    ladder_close_t3 = c['ladder_close_t3']
    ladder_close_t4 = c['ladder_close_t4']
    ladder_close_t5 = c['ladder_close_t5']
    entry_stop_bars = c['entry_stop_bars']
    be_mx_pct = c['be_mx_pct']
    be_sl_buffer_pct = c['be_sl_buffer_pct']
    be_hard_mx = c['be_hard_mx']
    be_weak_mx = c['be_weak_mx']
    be_weak_vol = c['be_weak_vol']
    trend_be_mx = c['trend_be_mx']
    vol_confirm_thresh = c['vol_confirm_thresh']
    liq_mm_rate = c['liq_mm_rate']
    liq_threshold = c['liq_threshold']

    sc2 = False
    er = ""
    cr = 1.0
    exit_p = cp
    new_sl = sl
    new_mx = mx
    new_mn = mn

    pnl = 0.0
    sl_hit = False

    # 5m SL check — 返回8元组 (should_close, er, cr, new_sl, new_mx, new_mn, pnl, exit_p)
    if (pos == PositionSide.LONG and l5_i <= sl) or (pos == PositionSide.SHORT and h5_i >= sl):
        sl_hit = True
        if pos == PositionSide.LONG:
            pnl = abs(ep - sl) / ep * 100 * lev
        else:
            pnl = abs(sl - ep) / ep * 100 * lev
        pnl = -pnl
        pnl -= funding * lev
        pnl = max(pnl, -100.0)
        pnl -= slippage * 2 * lev
        sc2, er = True, "trigger_stop"
        cr = 1.0
        exit_p = sl
        return sl_hit, er, cr, new_sl, new_mx, new_mn, pnl, exit_p

    # Opposite S/R (REBOUND) — live mode skips hold_bars gate
    if not sl_hit and rebound_opp and (live or i >= 2) and sl != ep:
        if pos == PositionSide.LONG and eres > 0 and h5_i >= eres * (1 - 0.005) and c5_i < c5_prev:
            cr = min(0.5 / pr, 1.0) if pr > 0 else 0.5
            new_sl = ep
            sc2, er = True, "res_stop"
        elif pos == PositionSide.SHORT and esup > 0 and l5_i <= esup * (1 + 0.005) and c5_i > c5_prev:
            cr = min(0.5 / pr, 1.0) if pr > 0 else 0.5
            new_sl = ep
            sc2, er = True, "sup_stop"

    # BE_MX: volume confirm
    if not sl_hit and be_mx_pct > 0:
        if pos == PositionSide.LONG:
            if mx >= be_mx_pct and new_sl < ep and es.peak_vol_ratio >= vol_confirm_thresh:
                new_sl = ep * (1 - be_sl_buffer_pct / 100)
            elif hb >= 6 and max(mx, (h5_i - ep) / ep * 100 * lev) >= be_mx_pct and new_sl < ep:
                _5m_vr = v5_i / max(avgv_ci / 3, 0.001) if avgv_ci > 0 else 0
                if _5m_vr >= vol_confirm_thresh:
                    new_sl = ep * (1 - be_sl_buffer_pct / 100)
        elif pos == PositionSide.SHORT:
            if mx >= be_mx_pct and new_sl > ep and es.peak_vol_ratio >= vol_confirm_thresh:
                new_sl = ep * (1 + be_sl_buffer_pct / 100)
            elif hb >= 6 and max(mx, (ep - l5_i) / ep * 100 * lev) >= be_mx_pct and new_sl > ep:
                _5m_vr = v5_i / max(avgv_ci / 3, 0.001) if avgv_ci > 0 else 0
                if _5m_vr >= vol_confirm_thresh:
                    new_sl = ep * (1 + be_sl_buffer_pct / 100)

    # BE_HARD: unconditional
    if not sl_hit and be_hard_mx > 0 and mx >= be_hard_mx:
        if pos == PositionSide.LONG and new_sl < ep:
            new_sl = ep * (1 - be_sl_buffer_pct / 100)
        elif pos == PositionSide.SHORT and new_sl > ep:
            new_sl = ep * (1 + be_sl_buffer_pct / 100)

    # BE_WEAK: low volume confirm
    if not sl_hit and be_weak_mx > 0 and mx >= be_weak_mx:
        _cur_vr = v5_i / max(avgv_ci / 3, 0.001) if avgv_ci > 0 else 0
        if _cur_vr < be_weak_vol:
            if pos == PositionSide.LONG and new_sl < ep:
                new_sl = ep * (1 - be_sl_buffer_pct / 100)
            elif pos == PositionSide.SHORT and new_sl > ep:
                new_sl = ep * (1 + be_sl_buffer_pct / 100)

    # 无条件保本：mx达到BE_UNCONDITIONAL_MX时无条件缩SL（不依赖量比）
    # 针对3-8%盈利但因量比不足未触发BE的交易
    if not sl_hit and c['be_unconditional_mx'] > 0 and mx >= c['be_unconditional_mx']:
        if pos == PositionSide.LONG and new_sl < ep:
            new_sl = ep * (1 - be_sl_buffer_pct / 100)
        elif pos == PositionSide.SHORT and new_sl > ep:
            new_sl = ep * (1 + be_sl_buffer_pct / 100)

    # TREND_BE: trend context — 当mx>=阈值且趋势非盘整时，将SL移至入场价附近
    if not sl_hit and trend_be_mx > 0 and mx >= trend_be_mx:
        if tt not in (TrendType.CONSOLIDATION, TrendType.UNKNOWN):
            if pos == PositionSide.LONG and new_sl < ep:
                new_sl = ep * (1 - be_sl_buffer_pct / 100)
            elif pos == PositionSide.SHORT and new_sl > ep:
                new_sl = ep * (1 + be_sl_buffer_pct / 100)
    # 爆仓检查 — 独立if: 液化检查不受TREND_BE影响
    if not sl_hit and ((pos == PositionSide.LONG and l5_i <= ep * (1 - 1.0 / lev + liq_mm_rate / 100)) or \
         (pos == PositionSide.SHORT and h5_i >= ep * (1 + 1.0 / lev - liq_mm_rate / 100))):
        sl_hit = True
        liq_loss_pct = (1.0 / lev - liq_mm_rate / 100) * 100 * lev
        pnl = -liq_loss_pct
        pnl -= funding * lev
        pnl = max(pnl, -100.0)
        pnl -= slippage * 2 * lev
        if pos == PositionSide.LONG:
            exit_p = ep * (1 - 1.0 / lev + liq_mm_rate / 100)
        else:
            exit_p = ep * (1 + 1.0 / lev - liq_mm_rate / 100)
        pnl = max(pnl, liq_threshold)
        sc2, er = True, "liq"
        cr = 1.0

    # PnL update + exit checks (15m close in backtest, every tick in live)
    is_15m_close = (i >= 0 and i % 3 == 2) or live
    # 15m收盘PnL更新 — 用15m收盘价计算浮盈, 5m期间只触发SL/爆仓, 不更新PnL
    # i>=0 回测: 3根5m bar扣3倍资金费率; i=-1 实盘: 扣1倍(已实时累积)
    if not sl_hit and is_15m_close:
        if pos == PositionSide.LONG:
            pnl_c = (c15s - ep) / ep * 100 * lev
        else:
            pnl_c = (ep - c15s) / ep * 100 * lev
        pnl = pnl_c
        pnl -= funding * lev * (3 if i >= 0 else 1)
        pnl = max(pnl, -100.0)
        cur_vr = v15s / max(avgv_ci, 0.001) if avgv_ci > 0 else 0
        if pnl_c > mx:
            es.peak_vol_ratio = max(es.peak_vol_ratio, cur_vr)
        mx = max(mx, pnl_c)
        mn = min(mn, pnl_c)
        es.ph.append(pnl_c)
        if len(es.ph) > 10:
            es.ph.pop(0)
        er = ""
        cr = 1.0
        exit_p = c15s

        # 操作线跌破（优先级2）：连续2根收盘跌破操作线→平70%，止损移至入场价
        if not sc2 and op_info:
            op_idx = op_info.get('idx', 0)
            op_price = op_info.get('price', 0)
            op_slope = op_info.get('slope', 0)
            if op_idx > 0 and op_price > 0:
                op_val = op_price + op_slope * (ci - op_idx)
                if op_val > 0:
                    if (pos == PositionSide.LONG and c15s < op_val) or \
                       (pos == PositionSide.SHORT and c15s > op_val):
                        es.op_breach_count += 1
                        if es.op_breach_count >= 2:
                            cr = min(0.7 / pr, 1.0) if pr > 0 else 0.7
                            new_sl = ep
                            sc2, er = True, "op_line_breach"
                    else:
                        if es.op_breach_count > 0:
                            es.op_breach_count = 0
        # 基石线跌破（优先级3-4）：连续2根收盘跌破基石线，按成交量区分
        if not sc2 and fl_info:
            fl_idx = fl_info.get('idx', 0)
            fl_price = fl_info.get('price', 0)
            fl_slope = fl_info.get('slope', 0)
            if fl_idx > 0 and fl_price > 0:
                fl_val = fl_price + fl_slope * (ci - fl_idx)
                if fl_val > 0:
                    is_breach = (pos == PositionSide.LONG and c15s < fl_val) or \
                                (pos == PositionSide.SHORT and c15s > fl_val)
                    if is_breach:
                        es.fl_breach_count += 1
                        if es.fl_breach_count >= 2:
                            if cur_vr >= 0.8:
                                cr = 1.0
                                sc2, er = True, "fl_breach_high_vol"
                            elif cur_vr <= 0.2:
                                cr = min(0.5 / pr, 1.0) if pr > 0 else 0.5
                                sc2, er = True, "fl_breach_low_vol"
                            else:
                                cr = 1.0
                                sc2, er = True, "fl_breach"
                    else:
                        if es.fl_breach_count > 0:
                            es.fl_breach_count = 0

        # 速度退出: 盈利快速回落时平仓
        # 单根: prev_pnl - cur_pnl >= vel_floor (单bar亏损速度)
        _vf = vel_floor
        if vel_floor_tr > 0 and st.startswith('TR_') and mx < vel_tr_mx_cap:
            _vf = vel_floor_tr
        if _vf > 0 and mx >= vel_min_mx and es.prev_pnl != 0 and es.prev_pnl - pnl_c >= _vf:
            sc2, er = True, "velocity"
        # 多根: 窗口内峰值 - 当前 >= vel_floor_multi (连续回落)
        if not sc2 and vel_window > 0 and vel_floor_multi > 0 and mx >= vel_min_mx and len(es.ph) >= vel_window:
            if max(es.ph[-vel_window:]) - es.ph[-1] >= vel_floor_multi:
                sc2, er = True, "velocity"
        es.prev_pnl = pnl_c

        # 移动止盈: 盈利达到trail_act后激活, 从最高点回落trail_dist%平仓
        if trail_on and es.trail_active:
            _td = trail_tight_dist if (trail_tight_dist > 0 and mx >= trail_tight_act) else trail_dist
            if st.startswith('TR_') and trail_dist_tr > 0:
                _td = trail_dist_tr
            if pos == PositionSide.LONG:
                es.trail_high = max(es.trail_high, c15s)
                if c15s <= es.trail_high * (1 - _td / 100):
                    sc2, er = True, "trailing"
            else:
                es.trail_high = min(es.trail_high, c15s)
                if c15s >= es.trail_high * (1 + _td / 100):
                    sc2, er = True, "trailing"
        if trail_on and not es.trail_active:
            _ta = trail_act
            if st.startswith('TR_') and trail_act_tr > 0:
                _ta = trail_act_tr
            if pos == PositionSide.LONG and pnl_c >= _ta:
                es.trail_high = c15s
                es.trail_active = True
            elif pos == PositionSide.SHORT and pnl_c >= _ta:
                es.trail_high = c15s
                es.trail_active = True

        # 盈利回撤保护：当盈利从最高点回撤PROFIT_PROTECT_PCT%时退出
        # 例如：PROFIT_PROTECT_PCT=50表示当盈利从最高点回撤50%时退出
        # 如果最大盈利是5%，当盈利回撤到2.5%时退出
        if not sc2 and c['profit_protect_pct'] > 0 and mx > 0 and pnl_c > 0:
            # 计算从最高点回撤的比例
            drawback_pct = (mx - pnl_c) / mx * 100
            if drawback_pct >= c['profit_protect_pct']:
                sc2, er = True, "profit_protect"

        # 阶梯止盈: 最多5次部分平仓, 每次在回撤触发时平仓比例=cr
        if not sc2 and es.tc < 5:
            if pnl_c > es.tsp:
                es.tsp = pnl_c
            ir = st.startswith('RB_') or 'REBOUND' in st
            is_tr = (st == 'TR' or st.startswith('TR_'))
            _is_consolidation_sr = (not ir and tt == TrendType.CONSOLIDATION and sr_room < 2.0)
            if ir or _is_consolidation_sr:
                lp = ladder_peak_rebound
                ddt = [ladder_dd_t1_rebound, ladder_dd_t2_rebound, ladder_dd_t3_rebound, ladder_dd_t4_rebound, ladder_dd_t5_rebound]
            elif is_tr and ladder_peak_tr > 0:
                lp = ladder_peak_tr
                ddt = [ladder_dd_t1_tr, ladder_dd_t2_tr, ladder_dd_t3_tr, ladder_dd_t4_tr, ladder_dd_t5_tr]
            else:
                lp = ladder_peak_uptrend
                ddt = [ladder_dd_t1_uptrend, ladder_dd_t2_uptrend, ladder_dd_t3_uptrend, ladder_dd_t4_uptrend, ladder_dd_t5_uptrend]
            if es.tsp >= lp:
                dd_trigger = es.tsp - pnl_c >= ddt[es.tc]
                if ladder_vs and dd_trigger and ci >= 0 and avgv_ci > 0:
                    if v15s >= avgv_ci * 0.7:
                        dd_trigger = False
                if dd_trigger:
                    ratios = [ladder_close_t1/100, ladder_close_t2/100, ladder_close_t3/100, ladder_close_t4/100, ladder_close_t5/100]
                    cr = max(0, min(ratios[es.tc] / pr if pr > 0 else ratios[es.tc], 1.0))
                    sc2, er = True, "ladder"
                    es.tc += 1
                    es.hrt = True

        # 反弹保本: RB/REBOUND信号盈利10%后回落至3%以下 → 平仓保本
        if not sc2 and rebound_be and (st.startswith('RB_') or 'REBOUND' in st) and mx >= 10.0 and pnl_c <= 3.0:
            sc2, er = True, "rebound_be"

        # 入场止损: 持仓超过entry_stop_bars/3根15m bar且从未盈利 → 平仓
        if not sc2 and hb15 >= entry_stop_bars // 3 and mx <= 0:
            sc2, er = True, "entry_stop"

        # 时间衰减（优先级9）：持仓16根15m且最大浮盈<3%→平50%
        if not sc2 and hb15 >= 16 and mx < 3.0:
            cr = min(0.5 / pr, 1.0) if pr > 0 else 0.5
            sc2, er = True, "time_decay"

        if sc2:
            pnl -= slippage * 2 * lev
            # Partial close: reset tsp to funded pnl (after slippage)
            # Design: run_final.py sets tsp=pnl after partial close so next dd_trigger uses current pnl as baseline
            if cr < 1.0:
                es.tsp = pnl

    new_mx = mx
    new_mn = mn
    should_close = sl_hit or (is_15m_close and sc2)
    # hrt覆盖: 阶梯触发后, 若后续velocity/trailing/entry_stop平仓, er改为'阶梯止盈'
    if es.hrt and should_close and er != "ladder":
        er = "ladder"
    return should_close, er, cr, new_sl, new_mx, new_mn, pnl, exit_p


class V8ExitState:
    """
    V8.2规格书9章: 9级出场管线状态
    """
    def __init__(self):
        self.trail_high: float = 0.0
        self.trail_active: bool = False
        self.tsp: float = 0.0
        self.tc: int = 0
        self.ph: List[float] = field(default_factory=list)
        self.fl_breach_count: int = 0
        self.op_breach_count: int = 0


def check_exit_v8(
    pos: PositionSide,
    ep: float,
    sl: float,
    lev: int,
    cp: float,
    hb15: int,
    mx: float,
    mn: float,
    st: str,
    tt: TrendType,
    es: V8ExitState,
    c15s: float,
    avgv_ci: float = 0.0,
    v15s: float = 0.0,
    pr: float = 1.0,
    fl_info: dict = None,
    op_info: dict = None,
    atr_ci: float = 0.0,
    bar_idx: int = 0,
) -> Tuple[bool, str, float, float, float, float, float, float]:
    """
    V8.2规格书9章: 9级出场管线（严格按优先级顺序评估）
      Level 1: QuickStop (hold>=1, 杠杆PnL<-1.5%, 平100%)
      Level 2: MAE Early Exit (hold<=8, MAE<-15%, 平100%)
      Level 3A: Hard Stop SL2 (pnl<=-30%, 平100%)
      Level 3B: Hard Stop SL1 (pnl<=-25%, 平50%)
      Level 4: 操作线跌破 (连续2根收盘跌破, 平70%, 止损移至入场价)
      Level 5A: 放量基石线跌破 (连续2根+VSR>=0.8, 平100%)
      Level 5B: 缩量基石线跌破 (连续2根+VSR<0.2, 平50%, 止损下移0.5ATR)
      Level 6: Ratchet Trailing (最高浮盈>=+0.2%激活, 回撤0.3%平仓)
      Level 7: Tiered TP (T1:+12%平30% T2:+15%平40% T3:+20%平30%)
      Level 8A: 僵尸持仓超时 (30根15m无盈利, 平100%)
      Level 8B: 动量衰减超时 (16根15m浮盈<3%, 平50%)
      Level 9: Liquidation Guard (pnl<=-90%, 平100%)
    Returns: (should_close, exit_reason, close_ratio, new_sl, new_mx, new_mn, pnl, exit_p)
    """
    if ep <= 0 or lev <= 0:
        return False, "", 1.0, sl, mx, mn, 0.0, cp

    cr = 1.0
    exit_p = cp
    new_sl = sl

    # 计算当前杠杆盈亏
    if pos == PositionSide.LONG:
        pnl_raw = (c15s - ep) / ep * 100 * lev
    else:
        pnl_raw = (ep - c15s) / ep * 100 * lev

    new_mx = max(mx, pnl_raw)
    new_mn = min(mn, pnl_raw)

    # ---- Level 1: QuickStop (hold>=1, pnl<-1.5%) ----
    if hb15 >= 1 and pnl_raw < -1.5:
        return True, 'quick_stop', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 2: MAE Early Exit (hold<=8, MAE<-15%) ----
    if hb15 <= 8 and new_mn < -15.0:
        return True, 'mae_early_exit', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 3A: Hard Stop SL2 (pnl<=-30%, 平100%) ----
    if pnl_raw <= -30.0:
        return True, 'hard_stop_sl2', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 3B: Hard Stop SL1 (pnl<=-25%, 平50%) ----
    if pnl_raw <= -25.0:
        close_ratio = min(0.5 / pr, 1.0) if pr > 0 else 0.5
        return True, 'hard_stop_sl1', close_ratio, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 4: 操作线跌破 (连续2根收盘跌破→平70%, 止损移至入场价) ----
    if op_info:
        op_idx = op_info.get('idx', 0)
        op_price = op_info.get('price', 0)
        op_slope = op_info.get('slope', 0)
        if op_idx > 0 and op_price > 0:
            op_val = op_price + op_slope * (bar_idx - op_idx)
            if op_val > 0:
                is_breach = (pos == PositionSide.LONG and c15s < op_val) or \
                            (pos == PositionSide.SHORT and c15s > op_val)
                if is_breach:
                    es.op_breach_count += 1
                    if es.op_breach_count >= 2:
                        cr = min(0.7 / pr, 1.0) if pr > 0 else 0.7
                        new_sl = ep
                        return True, 'operational_line_breach', cr, new_sl, new_mx, new_mn, pnl_raw, c15s
                else:
                    if es.op_breach_count > 0:
                        es.op_breach_count = 0

    # ---- Level 5A/B: 基石线跌破 ----
    if fl_info:
        fl_idx = fl_info.get('idx', 0)
        fl_price = fl_info.get('price', 0)
        fl_slope = fl_info.get('slope', 0)
        if fl_idx > 0 and fl_price > 0:
            fl_val = fl_price + fl_slope * (bar_idx - fl_idx)
            if fl_val > 0:
                is_breach = (pos == PositionSide.LONG and c15s < fl_val) or \
                            (pos == PositionSide.SHORT and c15s > fl_val)
                if is_breach:
                    es.fl_breach_count += 1
                    if es.fl_breach_count >= 2:
                        cur_vr = v15s / max(avgv_ci, 0.001) if avgv_ci > 0 else 1.0
                        if cur_vr >= 0.8:
                            return True, 'foundation_breach_high_vol', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s
                        elif cur_vr <= 0.2:
                            cr = min(0.5 / pr, 1.0) if pr > 0 else 0.5
                            if atr_ci > 0:
                                new_sl = fl_val - atr_ci * 0.5 if pos == PositionSide.LONG else fl_val + atr_ci * 0.5
                            return True, 'foundation_breach_low_vol', cr, new_sl, new_mx, new_mn, pnl_raw, c15s
                        else:
                            cr = min(0.5 / pr, 1.0) if pr > 0 else 0.5
                            return True, 'foundation_breach', cr, new_sl, new_mx, new_mn, pnl_raw, c15s
                else:
                    if es.fl_breach_count > 0:
                        es.fl_breach_count = 0

    # ---- Level 6: Ratchet Trailing Stop ----
    if pos == PositionSide.LONG:
        es.trail_high = max(es.trail_high, c15s) if es.trail_high > 0 else c15s
        if not es.trail_active and pnl_raw >= 0.2:
            es.trail_active = True
        if es.trail_active and c15s <= es.trail_high * 0.997:
            return True, 'trailing_stop', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s
    else:
        es.trail_high = min(es.trail_high, c15s) if es.trail_high > 0 else c15s
        if not es.trail_active and pnl_raw >= 0.2:
            es.trail_active = True
        if es.trail_active and c15s >= es.trail_high * 1.003:
            return True, 'trailing_stop', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 7: Tiered TP ----
    if es.tc < 3:
        if pnl_raw > es.tsp:
            es.tsp = pnl_raw
        tiers = [(12, 3), (15, 4), (20, 5)]
        if es.tsp >= tiers[es.tc][0]:
            dd = es.tsp - pnl_raw
            if dd >= tiers[es.tc][1]:
                tier_close = [0.3, 0.4, 0.3][es.tc]
                cr = min(tier_close / pr, 1.0) if pr > 0 else tier_close
                es.tc += 1
                return True, f'tiered_tp_t{es.tc}', cr, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 8A: 僵尸持仓超时 (30根15m无盈利) ----
    if hb15 >= 30 and pnl_raw <= 0:
        return True, 'zombie_timeout', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 8B: 动量衰减超时 (16根15m浮盈<3%) ----
    if hb15 >= 16 and new_mx < 3.0:
        cr = min(0.5 / pr, 1.0) if pr > 0 else 0.5
        return True, 'momentum_decay', cr, new_sl, new_mx, new_mn, pnl_raw, c15s

    # ---- Level 9: Liquidation Guard (最低优先级, 最后检查, 防穿仓) ----
    if pnl_raw <= -90.0:
        return True, 'liquidation', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s

    return False, '', 1.0, new_sl, new_mx, new_mn, pnl_raw, c15s


class TRState(Enum):
    """TR策略状态机（规格书6.1节）"""
    IDLE = 0
    SEARCHING = 1
    FROZEN = 2
    CONFIRMING = 3
    ENTRY_WAIT = 4
    HOLDING = 5
    COOLDOWN = 6


def detect_p1_candidate(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    breakout_idx: int,
    breakout_price: float,
    current_idx: int,
    atr7_current: float,
) -> Optional[Dict]:
    """
    V8.2规格书8.1节: detectP1Candidate
    确认突破后的P1支撑点 (8根Bar检索窗口, 最小回踩0.5%, 反弹验证>=0.5*ATR7)
    返回 {'p1Price': float, 'p1Idx': int, 'frozen': bool} | None
    """
    if current_idx - breakout_idx > 8:
        return None

    min_low = float(low[breakout_idx + 1])
    min_idx = breakout_idx + 1
    for i in range(breakout_idx + 1, current_idx + 1):
        if i >= len(low):
            break
        if float(low[i]) < min_low:
            min_low = float(low[i])
            min_idx = i

    pullback_depth = (breakout_price - min_low) / breakout_price if breakout_price > 0 else 0
    if pullback_depth < 0.005:
        return None

    max_post_high = min_low
    for k in range(min_idx + 1, current_idx + 1):
        if k >= len(high):
            break
        if float(high[k]) > max_post_high:
            max_post_high = float(high[k])

    if max_post_high - min_low < 0.5 * atr7_current:
        return None

    frozen = False
    if current_idx - min_idx >= 3:
        max_c = float(close[current_idx])
        min_c = float(close[current_idx])
        for k in range(current_idx - 2, current_idx + 1):
            if k >= len(close):
                break
            if float(close[k]) > max_c:
                max_c = float(close[k])
            if float(close[k]) < min_c:
                min_c = float(close[k])
        if min_c > 0 and (max_c - min_c) / min_c <= 0.002:
            frozen = True

    return {'p1Price': min_low, 'p1Idx': min_idx, 'frozen': frozen}


class TupoEngine:
    """
    V8.2规格书5章: 5状态微观状态机
    状态: SEARCHING → FOUNDATION_SET → TRIGGER_ARMED → SIGNAL_FIRED / VACUUM_CANCELLED
    支持 LONG/SHORT 方向
    """
    def __init__(self, symbol: str = '', direction: str = 'LONG'):
        self.symbol = symbol
        self.direction = direction.upper()
        self.state = 'SEARCHING'
        self.anchor_p1: Optional[Dict] = None  # 对于LONG是支撑低点, 对于SHORT是阻力高点
        self.anchor_h1: Optional[Dict] = None  # 对于LONG是阻力高点, 对于SHORT是支撑低点

    def reset(self):
        self.state = 'SEARCHING'
        self.anchor_p1 = None
        self.anchor_h1 = None

    def evaluate_bar(self, i: int, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     atr14: np.ndarray, vsr: np.ndarray, market_trend: str) -> Optional[Dict]:
        """
        V8.2规格书5.2节: evaluateBar
        每根bar评估一次, 返回信号字典或None
        LONG方向: P1(swing low) → H1(swing high) → 突破H1向上
        SHORT方向: P1(swing high) → H1(swing low) → 跌破H1向下
        """
        if i < 4:
            return None

        current_close = float(close[i])
        current_open = float(close[i])
        current_atr = float(atr14[i]) if i < len(atr14) else 0
        current_vsr = float(vsr[i]) if i < len(vsr) else 0
        is_long = self.direction == 'LONG'

        # ---- 状态1: SEARCHING ----
        if self.state == 'SEARCHING':
            k = i - 2
            if k - 2 >= 0 and k + 2 < len(low):
                if is_long:
                    is_swing = (
                        float(low[k]) <= float(low[k - 2]) and
                        float(low[k]) <= float(low[k - 1]) and
                        float(low[k]) <= float(low[k + 1]) and
                        float(low[k]) <= float(low[k + 2])
                    )
                    if is_swing and market_trend in ('UPTREND', 'RANGEBOUND'):
                        self.anchor_p1 = {'price': float(low[k]), 'index': k}
                        self.state = 'FOUNDATION_SET'
                else:
                    is_swing = (
                        float(high[k]) >= float(high[k - 2]) and
                        float(high[k]) >= float(high[k - 1]) and
                        float(high[k]) >= float(high[k + 1]) and
                        float(high[k]) >= float(high[k + 2])
                    )
                    if is_swing and market_trend in ('DOWNTREND', 'RANGEBOUND'):
                        self.anchor_p1 = {'price': float(high[k]), 'index': k}
                        self.state = 'FOUNDATION_SET'

        # ---- 状态2: FOUNDATION_SET ----
        elif self.state == 'FOUNDATION_SET':
            k = i - 2
            if k - 2 >= 0 and k + 2 < len(high):
                if is_long:
                    is_swing = (
                        float(high[k]) >= float(high[k - 2]) and
                        float(high[k]) >= float(high[k - 1]) and
                        float(high[k]) >= float(high[k + 1]) and
                        float(high[k]) >= float(high[k + 2])
                    )
                    if is_swing and self.anchor_p1 and k > self.anchor_p1['index']:
                        height = float(high[k]) - self.anchor_p1['price']
                        if height >= 1.0 * current_atr:
                            self.anchor_h1 = {'price': float(high[k]), 'index': k}
                            self.state = 'TRIGGER_ARMED'
                else:
                    is_swing = (
                        float(low[k]) <= float(low[k - 2]) and
                        float(low[k]) <= float(low[k - 1]) and
                        float(low[k]) <= float(low[k + 1]) and
                        float(low[k]) <= float(low[k + 2])
                    )
                    if is_swing and self.anchor_p1 and k > self.anchor_p1['index']:
                        height = self.anchor_p1['price'] - float(low[k])
                        if height >= 1.0 * current_atr:
                            self.anchor_h1 = {'price': float(low[k]), 'index': k}
                            self.state = 'TRIGGER_ARMED'

        # ---- 状态3: TRIGGER_ARMED ----
        elif self.state == 'TRIGGER_ARMED' and self.anchor_h1 and self.anchor_p1:
            if is_long:
                if current_close < self.anchor_p1['price']:
                    self.state = 'SEARCHING'
                    return None
                is_breakout = current_close > self.anchor_h1['price']
            else:
                if current_close > self.anchor_p1['price']:
                    self.state = 'SEARCHING'
                    return None
                is_breakout = current_close < self.anchor_h1['price']

            is_volume_surge = current_vsr >= 1.5
            body_size = abs(current_close - current_open)
            is_solid_body = body_size >= 0.8 * current_atr

            if is_breakout and is_volume_surge and is_solid_body:
                self.state = 'SIGNAL_FIRED'
                if is_long:
                    stop_loss = min(self.anchor_p1['price'], current_close - 1.5 * current_atr)
                else:
                    stop_loss = max(self.anchor_p1['price'], current_close + 1.5 * current_atr)
                return {
                    'direction': self.direction,
                    'entry_price': current_close,
                    'stop_loss': stop_loss,
                    'anchor_p1': self.anchor_p1,
                    'anchor_h1': self.anchor_h1,
                    'vsr': current_vsr,
                    'confidence': 0.95 if current_vsr >= 3.0 else 0.85,
                }

        return None


def detect_p1(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_7: np.ndarray,
    breakout_price: float,
    breakout_idx: int,
    current_idx: int,
    side: str = 'LONG',
    p1_min_pullback: float = 0.005,
    p1_rebound_mult: float = 0.5,
    p1_freeze_bars: int = 3,
    p1_freeze_threshold: float = 0.002,
    p1_lookback: int = 8,
) -> Optional[Dict]:
    """
    规格书6.2节：P1确认算法

    从 breakout_idx+1 开始追踪候选低点（LONG）或高点（SHORT）。
    返回 P1 字典或 None:
        {'price': float, 'idx': int, 'rsi': float}
    """
    scan_end = min(current_idx, breakout_idx + p1_lookback)
    if scan_end <= breakout_idx + 1:
        return None

    if side == 'LONG':
        best_idx = None
        best_price = None
        for j in range(breakout_idx + 1, scan_end):
            if j >= len(low):
                break
            lv = low[j]
            depth = (breakout_price - lv) / breakout_price if breakout_price > 0 else 0
            if depth >= p1_min_pullback:
                if best_idx is None or lv < best_price:
                    best_idx = j
                    best_price = lv
        if best_idx is None:
            return None
        # 反弹验证
        rebound_high = float(np.max(high[best_idx + 1:min(scan_end + 1, len(high))]))
        rebound = (rebound_high - best_price) / best_price if best_price > 0 else 0
        atrv = atr_7[best_idx] if best_idx < len(atr_7) else 0
        rebound_ok = rebound > p1_rebound_mult * atrv / max(best_price, 0.001) if atrv > 0 else False
        if not rebound_ok:
            return None
        # 窄幅冻结检查
        freeze_ok = True
        if best_idx > 0:
            freeze_window = min(p1_freeze_bars, scan_end - best_idx)
            if freeze_window >= 2:
                seg = high[best_idx:best_idx + freeze_window] - low[best_idx:best_idx + freeze_window]
                amp_max = np.max(seg) / best_price if best_price > 0 else 999
                if amp_max < p1_freeze_threshold:
                    freeze_ok = True
        return {'price': best_price, 'idx': best_idx, 'confirmed': freeze_ok}
    else:
        best_idx = None
        best_price = None
        for j in range(breakout_idx + 1, scan_end):
            if j >= len(high):
                break
            hv = high[j]
            depth = (hv - breakout_price) / breakout_price if breakout_price > 0 else 0
            if depth >= p1_min_pullback:
                if best_idx is None or hv > best_price:
                    best_idx = j
                    best_price = hv
        if best_idx is None:
            return None
        rebound_low = float(np.min(low[best_idx + 1:min(scan_end + 1, len(low))]))
        rebound = (best_price - rebound_low) / best_price if best_price > 0 else 0
        atrv = atr_7[best_idx] if best_idx < len(atr_7) else 0
        rebound_ok = rebound > p1_rebound_mult * atrv / max(best_price, 0.001) if atrv > 0 else False
        if not rebound_ok:
            return None
        freeze_ok = True
        if best_idx > 0:
            freeze_window = min(p1_freeze_bars, scan_end - best_idx)
            if freeze_window >= 2:
                seg = high[best_idx:best_idx + freeze_window] - low[best_idx:best_idx + freeze_window]
                amp_max = np.max(seg) / best_price if best_price > 0 else 999
                if amp_max < p1_freeze_threshold:
                    freeze_ok = True
        return {'price': best_price, 'idx': best_idx, 'confirmed': freeze_ok}


def find_dynamic_trendline(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    foundation_line: dict,
    current_idx: int,
    side: str = 'LONG',
    op_min_bars: int = 5,
    op_expire_bars: int = 8,
    slope_check: bool = True,
    op_breach_close: float = 0.70,
) -> Optional[Dict]:
    """
    规格书5.2节+6.5节：生成/更新操作线

    从 foundation.end_idx 至 current_idx 寻找有效回调点。
    返回操作线字典或 None:
        {'idx': int, 'price': float, 'slope': float, 'active': bool, 'update_idx': int}
    """
    if not foundation_line:
        return None
    fl_idx = foundation_line.get('idx', 0)
    fl_price = foundation_line.get('price', 0)
    fl_slope = foundation_line.get('slope', 0)
    if fl_idx >= current_idx:
        return None

    scan_start = max(fl_idx + op_min_bars, foundation_line.get('end_idx', fl_idx))
    if scan_start >= current_idx:
        return None

    if side == 'LONG':
        best_idx = None
        best_price = None
        for j in range(scan_start, current_idx):
            if j < len(low):
                trend_p = fl_price + fl_slope * (j - fl_idx) if fl_slope else fl_price
                if low[j] > trend_p:
                    if best_idx is None or low[j] < best_price:
                        best_idx = j
                        best_price = low[j]
        if best_idx is None or best_price is None:
            return None
    else:
        best_idx = None
        best_price = None
        for j in range(scan_start, current_idx):
            if j < len(high):
                trend_p = fl_price + fl_slope * (j - fl_idx) if fl_slope else fl_price
                if high[j] < trend_p:
                    if best_idx is None or high[j] > best_price:
                        best_idx = j
                        best_price = high[j]
        if best_idx is None or best_price is None:
            return None

    new_slope = (best_price - fl_price) / (best_idx - fl_idx) if best_idx != fl_idx else fl_slope

    ol = {
        'idx': fl_idx,
        'price': fl_price,
        'slope': new_slope,
        'second_idx': best_idx,
        'second_val': best_price,
        'update_idx': best_idx,
        'active': True,
        'friction_count': 0,
    }

    # 摩擦检查：价格贴线OP_EXPIRE_BARS根→失效
    friction_count = 0
    for j in range(max(fl_idx + 1, best_idx), current_idx):
        if j >= len(close):
            break
        ol_val = fl_price + new_slope * (j - fl_idx)
        if ol_val > 0 and atr is not None and j < len(atr):
            dist = abs(close[j] - ol_val) / ol_val
            if dist < 0.1 * atr[j] / max(close[j], 0.001):
                friction_count += 1
                if friction_count >= op_expire_bars:
                    ol['active'] = False
                    break
            else:
                friction_count = 0
    ol['friction_count'] = friction_count
    return ol


def update_operational(
    op_line: dict,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    current_idx: int,
    side: str = 'LONG',
    op_min_bars: int = 5,
    op_expire_bars: int = 8,
    slope_check: bool = True,
) -> Optional[Dict]:
    """
    规格书6.5节：动态更新操作线

    扫描新出现的回调点，更新操作线。
    约束：min_gap=5，斜率加速检查，过期检查。
    """
    if not op_line:
        return None
    fl_idx = op_line.get('idx', 0)
    fl_price = op_line.get('price', 0)
    last_update = op_line.get('update_idx', fl_idx)
    if current_idx - last_update < op_min_bars:
        return op_line

    new_idx = None
    new_price = None
    if side == 'LONG':
        for j in range(last_update + 1, current_idx):
            if j < len(low):
                if new_idx is None or low[j] < new_price:
                    new_idx = j
                    new_price = low[j]
    else:
        for j in range(last_update + 1, current_idx):
            if j < len(high):
                if new_idx is None or high[j] > new_price:
                    new_idx = j
                    new_price = high[j]

    if new_idx is None or new_price is None:
        return op_line

    new_slope = (new_price - fl_price) / (new_idx - fl_idx) if new_idx != fl_idx else op_line.get('slope', 0)

    if slope_check:
        old_slope = op_line.get('slope', 0)
        if side == 'LONG' and new_slope <= old_slope:
            return op_line
        if side == 'SHORT' and new_slope >= old_slope:
            return op_line

    op_line['slope'] = new_slope
    op_line['second_idx'] = new_idx
    op_line['second_val'] = new_price
    op_line['update_idx'] = new_idx

    # 重置摩擦计数后重新检查
    friction_count = 0
    for j in range(max(fl_idx + 1, new_idx), current_idx):
        if j >= len(close):
            break
        ol_val = fl_price + new_slope * (j - fl_idx)
        if ol_val > 0 and atr is not None and j < len(atr):
            dist = abs(close[j] - ol_val) / ol_val
            if dist < 0.1 * atr[j] / max(close[j], 0.001):
                friction_count += 1
                if friction_count >= op_expire_bars:
                    op_line['active'] = False
                    break
            else:
                friction_count = 0
    op_line['friction_count'] = friction_count
    return op_line


# ============================================================
# 以下为 15mTupo_量化交易系统_完整开发文档 新增功能
# ============================================================

# ---------- 2.2 插针过滤 + 有效价格 ----------
def detect_spike(high, low, close_, open_, volume, atr_prev, avg_vol_prev10, k_spike=3.0, vol_ratio=0.3):
    """检测K线是否为插针（规格书2.2节）
    返回 (is_spike, eff_high, eff_low)
    """
    upper_wick = high - max(open_, close_)
    lower_wick = min(open_, close_) - low
    if atr_prev <= 0 or avg_vol_prev10 <= 0:
        return False, high, low
    if (upper_wick > k_spike * atr_prev or lower_wick > k_spike * atr_prev) and volume < avg_vol_prev10 * vol_ratio:
        return True, max(open_, close_), min(open_, close_)
    return False, high, low


def generate_effective_prices(high, low, close, open_, volume, atr_smooth, k_spike=3.0, vol_ratio=0.3):
    """批量生成有效价格数组（规格书2.2节）
    返回 (eff_high, eff_low, spike_flags)
    """
    n = len(high)
    eff_high = high.copy()
    eff_low = low.copy()
    spike_flags = np.zeros(n, dtype=bool)
    for i in range(1, n):
        avg_vol = np.mean(volume[max(0, i-10):i]) if i >= 10 else np.mean(volume[:i])
        spike, eh, el = detect_spike(high[i], low[i], close[i], open_[i], volume[i], atr_smooth[i-1], avg_vol, k_spike, vol_ratio)
        if spike:
            eff_high[i] = eh
            eff_low[i] = el
            spike_flags[i] = True
    return eff_high, eff_low, spike_flags


# ---------- 2.3 三套ATR系统 ----------
def calculate_atr_system(high, low, close, period=14, eff_high=None, eff_low=None):
    """计算三套ATR（规格书2.3节）
    返回 (atr_trade_raw, atr_trade_risk, atr_struct, atr_struct_ma100)
    ATR_struct 使用 eff 价格计算（规格书2.3.3）
    """
    n = len(close)
    tr_raw = np.full(n, np.nan)
    tr_struct = np.full(n, np.nan)
    _h = eff_high if eff_high is not None else high
    _l = eff_low if eff_low is not None else low
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr_raw[i] = max(hl, hc, lc)
        # ATR_struct 使用有效价格（规格书2.3.3）
        hl_eff = _h[i] - _l[i]
        hc_eff = abs(_h[i] - close[i-1])
        lc_eff = abs(_l[i] - close[i-1])
        tr_struct[i] = max(hl_eff, hc_eff, lc_eff)

    # ATR_trade_raw
    atr_raw = np.full(n, np.nan)
    if n > period:
        atr_raw[period] = np.mean(tr_raw[1:period+1])
        for i in range(period+1, n):
            atr_raw[i] = (atr_raw[i-1] * (period-1) + tr_raw[i]) / period

    # ATR_trade_risk（带异常保护）
    atr_risk = np.full(n, np.nan)
    if n > period:
        atr_risk[period] = atr_raw[period]
        for i in range(period+1, n):
            tr_risk = tr_raw[i]
            if atr_raw[i] > 0:
                pct = atr_raw[i] / atr_raw[i-1] if atr_raw[i-1] > 0 else 1.0
                if pct > 1.5:  # ATR跳增>50%
                    tr_risk = min(tr_raw[i], 3 * atr_risk[i-1])
            atr_risk[i] = (atr_risk[i-1] * (period-1) + tr_risk) / period

    # ATR_struct
    atr_struct = np.full(n, np.nan)
    if n > period:
        atr_struct[period] = np.mean(tr_struct[1:period+1])
        for i in range(period+1, n):
            atr_struct[i] = (atr_struct[i-1] * (period-1) + tr_struct[i]) / period

    # ATR_struct_ma100
    ma100 = np.full(n, np.nan)
    for i in range(min(100, n)):
        if i < n:
            slice_ = atr_struct[max(0, i-99):i+1]
            if np.any(~np.isnan(slice_)):
                ma100[i] = np.nanmean(slice_)
    return atr_raw, atr_risk, atr_struct, ma100


# ---------- 2.4 BBW + 分位数 ----------
def calculate_bbw(close, period=20, std_mult=2.0):
    """计算布林带宽度（规格书2.4节）
    返回 (bbw, bbw_percentile_queue)
    队列维护最近1000个BBW值
    """
    n = len(close)
    bbw = np.full(n, np.nan)
    for i in range(period-1, n):
        ma = np.mean(close[i-period+1:i+1])
        std = np.std(close[i-period+1:i+1])
        if ma > 0:
            ub = ma + std_mult * std
            lb = ma - std_mult * std
            bbw[i] = (ub - lb) / ma
        else:
            bbw[i] = 0
    return bbw


def get_bbw_percentile(bbw_queue, current_bbw):
    """计算当前BBW在历史队列中的分位数（规格书3.1.1节）
    分位数 = 当前BBW在排序队列中的索引 / (队列长度 - 1)
    """
    if len(bbw_queue) < 20:
        return 0.5  # 冷启动
    import bisect
    sorted_q = sorted(bbw_queue)
    idx = bisect.bisect_left(sorted_q, current_bbw)
    pct = idx / (len(sorted_q) - 1) if len(sorted_q) > 1 else 0.5
    return max(0.0, min(1.0, pct))


# ---------- 3.1.1 K_dynamic ----------
def calculate_k_dynamic(k_base, alpha, bbw_percentile):
    """计算动态反转阈值系数（规格书3.1.1节）
    返回 K_dynamic
    """
    kd = k_base * (1 + alpha * (bbw_percentile - 0.5))
    return max(0.25, min(1.20, kd))


# ---------- 4.2 TrendScore ----------
def calc_trend_score(lows, highs, adx_val, line_score=50):
    """趋势强度评分（规格书4.2节）
    lows/highs: 最近的摆动低点/高点列表 (每个元素有.price)
    """
    if len(lows) < 2 or len(highs) < 2:
        return 0
    # HH/HL一致性评分 (40%)
    n_hl = len(lows)
    hl_up = sum(1 for i in range(1, n_hl) if lows[i].price > lows[i-1].price)
    hh_up = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i-1].price)
    consistency = min(hl_up, hh_up) / max(n_hl - 1, 1) * 100

    # ADX评分 (10%)
    adx_score = max(0, min(100, (adx_val - 15) / 30 * 100))

    # 趋势线质量 (20%)
    tl_score = min(100, line_score)

    # 推动浪比例评分 (30%)
    if len(lows) >= 2 and len(highs) >= 2:
        impulse = highs[-1].price - lows[-1].price if highs[-1].price > lows[-1].price else lows[-1].price - highs[-1].price
        prev_impulse = highs[-2].price - lows[-2].price if highs[-2].price > lows[-2].price else lows[-2].price - highs[-2].price
        if prev_impulse > 0:
            ratio = min(impulse / prev_impulse, 2.0) / 2.0 * 100
        else:
            ratio = 50
    else:
        ratio = 50

    score = 0.40 * consistency + 0.30 * ratio + 0.10 * adx_score + 0.20 * tl_score
    return score


# ---------- 5.6 SignalScore ----------
def calc_signal_score(volume_pct, breakout_atr_pct, bbw_percentile, trend_consistency=0, liquidity=50):
    """入场质量评分（规格书5.6节）
    返回 (score, should_trade)
    """
    score = (0.25 * min(volume_pct, 100)
           + 0.25 * min(breakout_atr_pct * 100, 100)
           + 0.15 * ((1 - bbw_percentile) * 100)
           + 0.20 * trend_consistency
           + 0.15 * liquidity)
    should_trade = score >= 50
    return score, should_trade


# ---------- 5.5 信号冷却机制 ----------
class SignalCooldown:
    """信号冷却管理器（规格书5.5节）"""

    def __init__(self):
        self._state = {}  # symbol+direction -> {'losses':0, 'cool_until':0}

    def _key(self, symbol, direction):
        return f"{symbol}_{direction}"

    def check(self, symbol, direction, current_bar):
        """检查是否在冷却期"""
        key = self._key(symbol, direction)
        s = self._state.get(key)
        if s is None:
            return True
        return current_bar >= s.get('cool_until', 0)

    def record_loss(self, symbol, direction, current_bar, cool_bars=(10, 20)):
        """记录一次亏损，设置冷却（规格书5.5节）
        cool_bars: (首次冷却bars, 二次冷却bars)，三次以上当日禁止
        第一次失败：冷却10根15m K线（2.5小时）
        第二次连续失败：冷却20根15m K线（5小时）
        第三次连续失败：当日禁止（≈40根≈10小时）
        """
        key = self._key(symbol, direction)
        s = self._state.setdefault(key, {'losses': 0, 'cool_until': 0})
        s['losses'] += 1
        if s['losses'] == 1:
            s['cool_until'] = current_bar + cool_bars[0]
        elif s['losses'] == 2:
            s['cool_until'] = current_bar + cool_bars[1]
        else:
            s['cool_until'] = current_bar + 40  # 当日禁止（≈10小时）
        return s['cool_until']

    def record_win(self, symbol, direction):
        """记录一笔盈利，重置计数器"""
        key = self._key(symbol, direction)
        s = self._state.get(key)
        if s:
            s['losses'] = 0
            s['cool_until'] = 0

    def reset(self):
        self._state.clear()


__all__ = [
    'TradeResult', 'TrendType', 'PositionSide', 'ExitState', 'TRState', 'V8ExitState',
    'SIG_QUAL', '_calc_score', 'entry_ok', 'quality_ok', 'calc_sl_lev',
    'aggregate_5m_to_15m_fast',
    'calculate_adx_fast', 'calculate_rsi', 'calculate_avg_volume_fast',
    '_reg_slope_pct', 'find_trend_line_fast', 'find_horizontal_SR',
    'detect_triangle_proper', 'analyze_trend_fast',
    'check_exit', 'check_exit_v8', 'reload_exit_cfg', '_atr_ma40_buffer',
    'detect_p1', 'detect_p1_candidate', 'TupoEngine',
    'find_dynamic_trendline', 'update_operational',
    # 新增文档功能
    'detect_spike', 'generate_effective_prices',
    'calculate_atr_system',
    'calculate_bbw', 'get_bbw_percentile',
    'calculate_k_dynamic',
    'calc_trend_score', 'calc_signal_score',
    'SignalCooldown',
]
