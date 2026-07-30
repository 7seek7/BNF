# -*- coding: utf-8 -*-
"""
V8.3 九级离场管线 (Section 10)
L1速度急撤 → L2 MAE → L3 ATR自适应 → L4阶梯止盈 → L5 Trailing → L6保本
→ L7 REBOUND对侧缩损 → L8时间衰减 → L9爆仓防护
"""
import numpy as np
from typing import Tuple, Optional

EPS = 1e-12

# 默认参数 (可被覆盖)
EXIT_CFG = {
    'L1_VELOCITY_THRESH': 0.08,           # 8% 单Bar暴跌
    'L1_VELOCITY_ATR_MULT': 2.5,           # 2.5×ATR%
    'L2_MAE_BO': -8.0,                     # BO MAE阈值(%)
    'L2_MAE_TR': -12.0,                    # TR MAE阈值(%)
    'L2_MAE_RB': -15.0,                    # RB MAE阈值(%)
    'L2_MAE_BARS': 10,                     # MAE检查Bar数
    'L3_SL_K_BO': 2.0,                     # ATR SL乘数(BO)
    'L3_SL_K_TR': 3.0,                     # ATR SL乘数(TR)
    'L3_SL_K_RB': 3.0,                     # ATR SL乘数(RB)
    'L3_SL_MIN_PCT_BO': 2.0,               # BO最小SL%
    'L3_SL_MIN_PCT_TR': 2.5,               # TR最小SL%
    'L3_SL_MIN_PCT_RB': 1.5,               # RB最小SL%
    'L4_PEAK_TR': 12.0,                    # L4激活峰值TR(%)
    'L4_PEAK_BO': 15.0,                    # L4激活峰值BO(%)
    'L4_PEAK_RB': 8.0,                     # L4激活峰值RB(%)
    'L4_T1_DD': 10.0,                      # T1回撤%
    'L4_T2_DD': 15.0,                      # T2回撤%
    'L4_T3_DD': 20.0,                      # T3回撤%
    'L4_T1_CLOSE': 0.30,                   # T1平仓比例
    'L4_T2_CLOSE': 0.40,                   # T2平仓比例
    'L4_T3_CLOSE': 0.30,                   # T3平仓比例
    'L5_ACTIVATE_PCT': 5.0,                # Trailing激活峰值(%)
    'L5_TRAIL_DIST': 1.5,                  # Trailing回撤距离(%)
    'L6_BE_MX': 5.0,                       # BE激活峰值(%)
    'L6_BE_VSR': 1.5,                      # BE VSR门槛
    'L6_BE_RETRACE': 0.3,                  # BE回落至+0.3×ATR
    'L6_BE_CLOSE_RATIO': 0.30,             # BE平仓比例
    'L8_TIME_BO': 12,                      # BO时间衰减Bar数
    'L8_TIME_TR': 24,                      # TR时间衰减Bar数
    'L8_TIME_RB': 8,                       # RB时间衰减Bar数
    'L8_TIME_EXIT_RATIO': 0.50,            # 时间衰减平仓比例
    'L8_TIME_MAX_PNL': 3.0,                # 时间衰减最大浮盈门槛(%)
    'L9_LIQ_THRESHOLD': -90.0,             # 爆仓阈值(%)
    'L9_LIQ_PENALTY': 0.5,                 # 爆仓罚金(%)
    'L9_LIQ_MM_RATE': 0.5,                 # 维持保证金率(%)
}


def _get_signal_family(sig_type: str) -> str:
    """提取信号类型家族"""
    if sig_type.startswith('BO'): return 'BO'
    if sig_type.startswith('TR'): return 'TR'
    if sig_type.startswith('RB'): return 'RB'
    if sig_type.startswith('TRI'): return 'TRI'
    return 'OTHER'


def _get_trail_distance(bbw_percentile: float) -> float:
    """L5跟踪距离：根据BBW分位数动态调整（规格书6.1.3）"""
    if bbw_percentile < 0.3:
        return 2.5
    elif bbw_percentile > 0.7:
        return 8.0
    return 5.0


def check_exit_v83(
    direction: str,
    entry_price: float,
    entry_bar: int,
    current_bar: int,
    current_price: float,
    high: float,
    low: float,
    atr: float,
    atr_pct: float,
    stop_loss: float,
    signal_type: str,
    signal_score: float,
    max_pnl: float,
    min_pnl: float,
    hold_bars: int,
    trail_active: bool,
    trail_high: float,
    ladder_tier: int,
    ladder_peak: float,
    leverage: int,
    bbw_percentile: float = 0.5,
    be_sl_adjusted: bool = False
) -> Tuple[bool, float, str, float]:
    """
    九级离场管线检查
    返回 (should_exit, exit_price, exit_reason, close_ratio)
    close_ratio: 平仓比例 (0~1.0)
    """
    pnl_pct = ((current_price - entry_price) / entry_price * 100) if direction == 'LONG' else \
              ((entry_price - current_price) / entry_price * 100)

    # 盘中极端价格
    if direction == 'LONG':
        intra_high_pnl = ((high - entry_price) / entry_price * 100)
        intra_low_pnl = ((low - entry_price) / entry_price * 100)
    else:
        intra_high_pnl = ((entry_price - low) / entry_price * 100)
        intra_low_pnl = ((entry_price - high) / entry_price * 100)

    cur_max_pnl = max(max_pnl, intra_high_pnl)
    cur_min_pnl = min(min_pnl, intra_low_pnl)

    fam = _get_signal_family(signal_type)

    # ========== L1: 速度急撤 ==========
    if cur_max_pnl >= 8 and pnl_pct < -max(8, 2.5 * atr_pct) and hold_bars <= 5:
        return True, current_price, 'L1_velocity', 1.0

    # ========== L2: MAE早期风控（统一保证金%：阈值和3×ATR都转为杠杆后比较）==========
    if hold_bars <= EXIT_CFG['L2_MAE_BARS']:
        mae_thresh = max(EXIT_CFG.get(f'L2_MAE_{fam}', -8.0), -3 * atr_pct * leverage)
        pnl_lev = pnl_pct * leverage
        if pnl_lev < mae_thresh:
            return True, current_price, 'L2_mae', 1.0

    # ========== L3: ATR自适应止损 ==========
    k = EXIT_CFG.get('L3_SL_K_BO', 2.0) if fam == 'BO' else \
        EXIT_CFG.get('L3_SL_K_TR', 3.0) if fam in ('TR', 'TRI') else \
        EXIT_CFG.get('L3_SL_K_RB', 3.0) if fam == 'RB' else 2.0
    min_sl_map = {'BO': EXIT_CFG['L3_SL_MIN_PCT_BO'],
                  'TR': EXIT_CFG['L3_SL_MIN_PCT_TR'],
                  'RB': EXIT_CFG['L3_SL_MIN_PCT_RB']}
    sl_pct = max(min_sl_map.get(fam, EXIT_CFG['L3_SL_MIN_PCT_BO']), k * atr_pct)
    if direction == 'LONG' and current_price < entry_price * (1 - sl_pct / 100):
        return True, current_price, 'L3_atr_stop', 1.0
    if direction == 'SHORT' and current_price > entry_price * (1 + sl_pct / 100):
        return True, current_price, 'L3_atr_stop', 1.0
    if direction == 'LONG' and stop_loss > 0 and current_price <= stop_loss:
        return True, current_price, 'L3_hard_stop', 1.0
    if direction == 'SHORT' and stop_loss > 0 and current_price >= stop_loss:
        return True, current_price, 'L3_hard_stop', 1.0

    # ========== L4: 阶梯止盈（规格书：仅RB；浮盈回撤10%/15%/20%平30%/40%/30%）==========
    if fam == 'RB' and ladder_tier < 3:
        cur_max_lev = max(max_pnl, intra_high_pnl * leverage)  # 统一为PnL%
        pnl_lev = pnl_pct * leverage
        if cur_max_lev >= EXIT_CFG.get(f'L4_PEAK_{fam}', 8.0):
            dd_pcts = [EXIT_CFG['L4_T1_DD'], EXIT_CFG['L4_T2_DD'], EXIT_CFG['L4_T3_DD']]
            close_ratios = [EXIT_CFG['L4_T1_CLOSE'], EXIT_CFG['L4_T2_CLOSE'], EXIT_CFG['L4_T3_CLOSE']]
            if cur_max_lev > 0:
                dd = (cur_max_lev - pnl_lev) / cur_max_lev * 100
                if dd >= dd_pcts[ladder_tier]:
                    return True, current_price, f'L4_ladder_t{ladder_tier + 1}', close_ratios[ladder_tier]

    # ========== L5: Trailing Stop（规格书6.1.3：仅BO/TR，BBW分位数动态距离）==========
    if fam in ('BO', 'TR', 'TRI') and cur_max_pnl >= EXIT_CFG['L5_ACTIVATE_PCT']:
        trail_dist = _get_trail_distance(bbw_percentile)
        if trail_active:
            if direction == 'LONG':
                new_trail = max(trail_high, current_price)
                if current_price <= new_trail * (1 - trail_dist / 100):
                    return True, current_price, 'L5_trailing', 1.0
            else:
                new_trail = min(trail_high, current_price)
                if current_price >= new_trail * (1 + trail_dist / 100):
                    return True, current_price, 'L5_trailing', 1.0

    # ========== L6: 保本模式（规格书：浮盈≥5%后止损上移至成本）==========
    if not be_sl_adjusted and cur_max_pnl >= EXIT_CFG['L6_BE_MX']:
        if direction == 'LONG' and stop_loss < entry_price:
            return False, current_price, 'L6_breakeven', 0.0
        if direction == 'SHORT' and stop_loss > entry_price:
            return False, current_price, 'L6_breakeven', 0.0

    # ========== L7: REBOUND对侧缩损 ==========
    if fam == 'RB' and cur_max_pnl > 0:
        if direction == 'LONG' and current_price > entry_price:
            return False, current_price, 'L7_rebound', 0.0  # 仅缩止损不退出
        if direction == 'SHORT' and current_price < entry_price:
            return False, current_price, 'L7_rebound', 0.0

    # ========== L8: 时间衰减（规格书：超时未盈利=当前浮盈）==========
    time_limit = EXIT_CFG.get(f'L8_TIME_{fam}', 12)
    if hold_bars >= time_limit and pnl_pct < EXIT_CFG['L8_TIME_MAX_PNL']:
        return True, current_price, 'L8_time_decay', EXIT_CFG['L8_TIME_EXIT_RATIO']

    # ========== L9: 爆仓防护 ==========
    liq_loss = 100.0 / leverage - EXIT_CFG['L9_LIQ_MM_RATE']
    if pnl_pct <= -liq_loss:
        return True, current_price, 'L9_liquidation', 1.0

    return False, current_price, '', 0.0


def update_trail(direction: str, current_price: float, trail_high: float, trail_active: bool,
                 max_pnl: float, activate_pct: float = 5.0) -> Tuple[float, bool]:
    """更新Trailing Stop状态"""
    if trail_active:
        if direction == 'LONG':
            return max(trail_high, current_price), True
        else:
            return min(trail_high, current_price), True
    else:
        if max_pnl >= activate_pct:
            return current_price, True
    return trail_high, trail_active
