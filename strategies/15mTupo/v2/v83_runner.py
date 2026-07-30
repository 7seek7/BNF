# -*- coding: utf-8 -*-
"""
V9 统一结构引擎回测运行器
拐点序列趋势判定 + 三角识别 + 8信号矩阵 + 简化评分 + 9级退出
"""
import sys, os, gc, pandas as pd, numpy as np, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from dotenv import load_dotenv
_main_env = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
_strat_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(_main_env, override=False)
load_dotenv(_strat_env, override=False)

import importlib
_tupo_core = importlib.import_module('strategies.15mTupo.private.tupo_core')
TradeResult = _tupo_core.TradeResult
PositionSide = _tupo_core.PositionSide
aggregate_5m_to_15m_fast = _tupo_core.aggregate_5m_to_15m_fast
calculate_adx_fast = _tupo_core.calculate_adx_fast
calculate_rsi = _tupo_core.calculate_rsi
calculate_avg_volume_fast = _tupo_core.calculate_avg_volume_fast
calculate_bbw = _tupo_core.calculate_bbw
generate_effective_prices = _tupo_core.generate_effective_prices
calculate_atr_system = _tupo_core.calculate_atr_system
SignalCooldown = _tupo_core.SignalCooldown
calc_signal_score = _tupo_core.calc_signal_score
calc_trend_score = _tupo_core.calc_trend_score
get_bbw_percentile = _tupo_core.get_bbw_percentile
find_horizontal_SR = _tupo_core.find_horizontal_SR

_v9 = importlib.import_module('strategies.15mTupo.private.v9_core')
MarketTrendType = _v9.MarketTrendType
StructureEngine = _v9.StructureEngine
EPS = 1e-8

_v83 = importlib.import_module('strategies.15mTupo.private.v83_core')
V83SignalType = _v83.V83SignalType
find_p1_support = _v83.find_p1_support
find_p1_resistance = _v83.find_p1_resistance
check_mtf = _v83.check_mtf
arbitrate_signals = _v83.arbitrate_signals
V83Signal = _v83.V83Signal

_v83_exit = importlib.import_module('strategies.15mTupo.private.v83_exit')
check_exit_v83 = _v83_exit.check_exit_v83
update_trail = _v83_exit.update_trail
EXIT_CFG = _v83_exit.EXIT_CFG

DR = r"E:\BNFF\BNFRich\data\historical"
if os.environ.get('YZDIR'):
    DR = r"E:\BNFF\BNFRich\data\historical\yanzheng"

BASE_LEV_CONFIG = int(os.environ.get("BASE_LEV", "15"))
MAX_LOSS_CONFIG = int(os.environ.get("MAX_LOSS", "30"))
SLIPPAGE_PCT = float(os.environ.get("SLIPPAGE", "0.03"))
FUNDING_PER_BAR = float(os.environ.get("FUNDING", "0.0001"))
BE_HARD_MX = float(os.environ.get("BE_HARD_MX", "0"))
BE_WEAK_MX = float(os.environ.get("BE_WEAK_MX", "0"))
BE_WEAK_VOL = float(os.environ.get("BE_WEAK_VOL", "5.0"))
TREND_BE_MX = float(os.environ.get("TREND_BE_MX", "0"))
BE_SL_BUFFER_PCT = float(os.environ.get("BE_SL_BUFFER_PCT", "0"))

def _calc_atr_pct(atr_arr, close_arr, ci):
    if ci < len(atr_arr) and ci < len(close_arr) and close_arr[ci] > 0:
        return atr_arr[ci] / close_arr[ci] * 100
    return 1.0


def _vsr_calc(v15, ci, avgv):
    if ci >= 0 and ci < len(avgv) and avgv[ci] > 0 and ci < len(v15):
        return v15[ci] / avgv[ci]
    return 1.0


def _calc_leverage(atr_pct):
    base_lev = BASE_LEV_CONFIG
    if atr_pct > 0:
        lev_calc = MAX_LOSS_CONFIG / max(atr_pct * 0.5, 0.1)
        lev = max(5, min(int(lev_calc), base_lev))
    else:
        lev = base_lev
    return min(lev, 30)


def run_v83(df_5m, sym, s_obj):
    c5 = df_5m['close'].values.astype(float)
    h5 = df_5m['high'].values.astype(float)
    l5 = df_5m['low'].values.astype(float)
    o5 = df_5m['open'].values.astype(float)
    v5 = df_5m['volume'].values.astype(float)

    df15 = aggregate_5m_to_15m_fast(df_5m)
    if len(df15) < 100:
        return []

    c15 = df15['close'].values
    h15 = df15['high'].values
    l15 = df15['low'].values
    o15 = df15['open'].values
    v15 = df15['volume'].values.astype(float)

    adx, pdm, mdm, atr14 = calculate_adx_fast(h15, l15, c15, 14)
    _, _, _, atr7_arr = calculate_adx_fast(h15, l15, c15, 7)
    rsi = calculate_rsi(c15, 14)
    avgv = calculate_avg_volume_fast(v15, 20)

    # 文档2.4: BBW + 队列
    bbw_arr = calculate_bbw(c15)
    bbw_queue = []
    # 文档2.2: 有效价格
    eff_high, eff_low, spike_flags = generate_effective_prices(h15, l15, c15, o15, v15, atr14)
    # 文档2.3: 三套ATR（ATR_struct使用有效价格）
    atr_raw, atr_risk, atr_struct, atr_ma100 = calculate_atr_system(h15, l15, c15, eff_high=eff_high, eff_low=eff_low)

    # 替代atr14校验
    c15_safe = np.where(c15 > 0, c15, 1.0)

    n_5 = len(c5)
    n_15 = len(c15)

    # VSR
    vsr = np.full(n_15, 1.0, dtype=float)
    ewma_v = np.full(n_15, 0.0, dtype=float)
    for i in range(1, n_15):
        if i == 1:
            ewma_v[i] = v15[i]
        else:
            ewma_v[i] = 0.10 * v15[i] + 0.90 * ewma_v[i - 1]
        vsr[i] = v15[i] / max(ewma_v[i], EPS) if ewma_v[i] > 0 else 1.0

    # 统一结构引擎
    engine = StructureEngine(atr7=atr7_arr, atr14=atr14)

    # EMA20
    ema20 = pd.Series(c15).ewm(span=20).mean().values
    # EMA50 (用于4h方向近似)
    ema50 = pd.Series(c15).ewm(span=50).mean().values

    rlist = []
    cooldown_mgr = SignalCooldown()
    pos = None
    eidx = eidx15 = 0
    last_engine_bar = -1
    epr = sl = 0.0
    lev = 1
    sig_type_str = ''
    eadx = 20
    eatr = 0.0
    ersi = 50.0
    mx = mn = 0.0
    hold_bars = 0
    pr = 1.0  # partial close ratio
    ap = 0.0  # accumulated pnl
    trail_active = False
    trail_high = 0.0
    ladder_tier = 0
    ladder_peak = 0.0
    prev_pnl = 0.0
    be_sl_adjusted = False
    entry_trend_str = 'UNKNOWN'
    entry_score = 5.0
    entry_sup = 0.0
    entry_res = 0.0
    # SignalScore缓存（ci变化时更新）
    _last_sig_ci = -1
    _bbw_percentile_val = 0.5
    _cur_vsr_pct = 50.0
    _trend_consistency = 50

    def _check_friction_decayed(c_arr, atr_arr, p1_price, ci_idx, lookback=5):
        """摩擦衰减检测：检查最近lookback根K线价格是否持续靠近P1（规格书附录A Q5）
        若≥60%的K线收盘价在P1的0.1×ATR范围内，返回True
        """
        if p1_price is None or ci_idx <= lookback:
            return False
        count = 0
        total = 0
        for j in range(max(0, ci_idx - lookback), ci_idx + 1):
            if j >= len(c_arr):
                continue
            total += 1
            dist_pct = abs(c_arr[j] - p1_price) / max(p1_price, 0.001) * 100
            thr = 0.1 * atr_arr[j] / max(p1_price, 0.001) * 100 if atr_arr[j] > 0 else 0.01
            if dist_pct < thr:
                count += 1
        return count >= total * 0.6 if total > 0 else False

    # 7.3 弱突破追踪：记录弱突破状态，4h内同方向可再入场
    _weak_breakouts = {}  # (symbol, direction) -> {'ci': ..., 'entry_price': ..., 'atr_trade': ...}

    # 9.1 历史确认函数：RB信号需要历史反弹确认
    def _check_rb_history(close, low, high, atr, ci, box_bottom, box_top, lookback=100, direction='LONG'):
        """搜索最近lookback根K线，确认历史支撑/阻力行为（规格书9.1节）
        LONG: 价格到达支撑区→20根内确认摆动低点→反弹≥0.5×ATR
        SHORT: 价格到达阻力区→20根内确认摆动高点→回落≥0.5×ATR
        """
        if ci <= lookback + 20:
            return False
        for j in range(ci - lookback, ci - 20):
            if j < 0 or j >= len(close):
                continue
            if direction == 'SHORT':
                # 阻力侧：高点触及或突破阻力区间
                if high[j] >= box_top:
                    confirm_idx = -1
                    for k in range(j + 1, min(j + 20, ci)):
                        if high[k] <= high[j] and k < len(close) and close[k] < high[j]:
                            confirm_idx = k
                            break
                    if confirm_idx > 0:
                        fallback = high[j] - close[confirm_idx]
                        if fallback >= 0.5 * atr[j]:
                            return True
            else:
                # 支撑侧：低点触及或跌入支撑区间
                if low[j] <= box_bottom:
                    confirm_idx = -1
                    for k in range(j + 1, min(j + 20, ci)):
                        if low[k] >= low[j] and k < len(close) and close[k] > low[j]:
                            confirm_idx = k
                            break
                    if confirm_idx > 0:
                        rebound = close[confirm_idx] - low[j]
                        if rebound >= 0.5 * atr[j]:
                            return True
        return False

    _rb_pullback_counter = 0  # 8.2 TR回调计数器
    _tracked_trend_start = -1  # 记录已追踪的趋势起点

    for i in range(60, n_5):
        bi = i % 3
        r15 = i // 3
        if r15 >= n_15:
            continue
        ci = max(0, r15 - 1)

        # 当前15m K线实时聚合
        if bi == 0:
            o15s = o5[i]
            h15s = h5[i]
            l15s = l5[i]
            c15s = c5[i]
            v15s = v5[i]
        else:
            h15s = max(h15s, h5[i])
            l15s = min(l15s, l5[i])
            c15s = c5[i]
            v15s += v5[i]

        c15v = c15[ci] if ci < n_15 else c15[-1]
        atr_pct = _calc_atr_pct(atr_risk, c15_safe, ci)
        cur_vsr = _vsr_calc(v15, ci, avgv)
        cur_ema20 = ema20[ci] if ci < len(ema20) else c15v

        # ========== 退出逻辑 ==========
        if pos is not None:
            hold_bars = i - eidx
            sl_hit = False
            exit_triggered = False
            exit_p = epr
            pnl = 0.0
            cr = 1.0
            er = ''

            # L9跳空保护：5m开盘价穿越强平价格+缓冲时立即以开盘价全平
            if pos is not None:
                mm_rate = 0.5  # 维持保证金率(%)
                buffer = 0.5  # 缓冲距离(%)
                if pos == PositionSide.LONG:
                    liq_price = epr * (1 - (1 - mm_rate / 100) / lev)
                    gap_hit = o5[i] <= liq_price * (1 + buffer / 100)
                else:
                    liq_price = epr * (1 + (1 - mm_rate / 100) / lev)
                    gap_hit = o5[i] >= liq_price * (1 - buffer / 100)
                if gap_hit:
                    pnl = abs(epr - o5[i]) / epr * 100 * lev
                    pnl = -pnl - FUNDING_PER_BAR * lev
                    pnl = max(pnl, -MAX_LOSS_CONFIG)
                    exit_p = o5[i]
                    sl_hit = True
                    er = '跳空保护'
                    cr = 1.0

            # 止损检查：规格书8.2要求用raw_high/raw_low，插针也真实触发止损
            if (pos == PositionSide.LONG and l5[i] <= sl) or \
               (pos == PositionSide.SHORT and h5[i] >= sl):
                sl_hit = True
                if sl_hit:
                    pnl = abs(epr - sl) / epr * 100 * lev if pos == PositionSide.LONG else \
                          abs(sl - epr) / epr * 100 * lev
                    pnl = -pnl - FUNDING_PER_BAR * lev
                    pnl = max(pnl, -MAX_LOSS_CONFIG)
                    exit_p = sl
                    er = '止损'
                    cr = 1.0
                    if pos == PositionSide.LONG:
                        exit_p = min(sl, l5[i])
                    else:
                        exit_p = max(sl, h5[i])

            # 15m收盘管理
            if not sl_hit and bi == 2:
                pnl_c = ((c15s - epr) / epr * 100 * lev) if pos == PositionSide.LONG else \
                        ((epr - c15s) / epr * 100 * lev)
                pnl_c -= FUNDING_PER_BAR * lev * 3
                pnl_c = max(pnl_c, -MAX_LOSS_CONFIG)
                pnl = pnl_c
                mx = max(mx, pnl_c)
                mn = min(mn, pnl_c)

                # V8.3 退出管线
                should_exit, ex_price, ex_reason, close_ratio = check_exit_v83(
                    direction='LONG' if pos == PositionSide.LONG else 'SHORT',
                    entry_price=epr,
                    entry_bar=eidx15,
                    current_bar=ci,
                    current_price=c15s,
                    high=h15s,
                    low=l15s,
                    atr=atr_risk[ci] if ci < n_15 else atr_risk[-1],
                    atr_pct=atr_pct,
                    stop_loss=sl,
                    signal_type=sig_type_str,
                    signal_score=entry_score,
                    max_pnl=mx,
                    min_pnl=mn,
                    hold_bars=hold_bars,
                    trail_active=trail_active,
                    trail_high=trail_high,
                    ladder_tier=ladder_tier,
                    ladder_peak=ladder_peak,
                    leverage=lev,
                    bbw_percentile=_bbw_percentile_val,
                    be_sl_adjusted=be_sl_adjusted
                )

                if should_exit:
                    if ex_reason == 'L7_rebound':
                        # L7: 不退出，缩紧止损至 entry + 浮盈×0.3
                        if pos == PositionSide.LONG and c15s > epr:
                            sl = max(sl, epr + (c15s - epr) * 0.3)
                        elif pos == PositionSide.SHORT and c15s < epr:
                            sl = min(sl, epr - (epr - c15s) * 0.3)
                    elif ex_reason == 'L6_breakeven':
                        # L6: 不退出，止损上移至成本+缓冲
                        if pos == PositionSide.LONG and sl < epr:
                            sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                            be_sl_adjusted = True
                        elif pos == PositionSide.SHORT and sl > epr:
                            sl = epr * (1 + BE_SL_BUFFER_PCT / 100)
                            be_sl_adjusted = True
                    else:
                        cr = close_ratio
                        er = ex_reason
                        exit_p = ex_price
                        exit_triggered = True

                # Trailing更新
                if not exit_triggered:
                    trail_high, trail_active = update_trail(
                        'LONG' if pos == PositionSide.LONG else 'SHORT',
                        c15s, trail_high, trail_active, mx
                    )

                # 保本模式 (L6简化-在主循环中处理)
                if not exit_triggered and not be_sl_adjusted:
                    if mx >= BE_HARD_MX and BE_HARD_MX > 0:
                        if pos == PositionSide.LONG and sl < epr:
                            sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                            be_sl_adjusted = True
                        elif pos == PositionSide.SHORT and sl > epr:
                            sl = epr * (1 + BE_SL_BUFFER_PCT / 100)
                            be_sl_adjusted = True
                    if mx >= TREND_BE_MX and TREND_BE_MX > 0:
                        if pos == PositionSide.LONG and sl < epr:
                            sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                            be_sl_adjusted = True
                        elif pos == PositionSide.SHORT and sl > epr:
                            sl = epr * (1 + BE_SL_BUFFER_PCT / 100)
                            be_sl_adjusted = True

                prev_pnl = pnl_c

            # 止损 + 退出执行
            if sl_hit or (bi == 2 and exit_triggered):
                pnl -= SLIPPAGE_PCT * 2 * lev
                pnl = max(pnl, -MAX_LOSS_CONFIG)
                ap += pnl * cr * pr
                pr *= (1 - cr)
                if pr < 0.01 or cr >= 1.0:
                    # 7.3 弱突破检测：止损发生在3根15m K线内(hold_bars<=9)，且最大浮盈<0.3×ATR
                    dir_str = 'LONG' if pos == PositionSide.LONG else 'SHORT'
                    if sl_hit and hold_bars <= 9 and mx < 0.3 * eatr:
                        _weak_breakouts[(sym, dir_str)] = {'ci': ci, 'entry_price': epr, 'atr_trade': eatr}
                    # 冷却记录
                    if ap > 0:
                        cooldown_mgr.record_win(sym, dir_str)
                    else:
                        cooldown_mgr.record_loss(sym, dir_str, ci)
                    rlist.append(TradeResult(
                        sym, eidx, epr, i, exit_p,
                        pos.value, lev,
                        ap, hold_bars, er,
                        entry_trend_str,
                        sig_type_str, mx, mn, 0.0,
                        eadx, eatr, ersi,
                        v5[i] if i < len(v5) else 0, 0,
                        rsi[ci] if ci < len(rsi) else 50,
                        False, False, 0.0, entry_sup, entry_res, sl,
                        foundation_line_slope, foundation_line_idx, foundation_line_val
                    ))
                    pos = None
                    pr = 1.0
                    ap = 0.0
                    mx = mn = 0.0
                    trail_active = False
                    trail_high = 0.0
                    ladder_tier = 0
                    ladder_peak = 0.0
                    be_sl_adjusted = False
                else:
                    # 部分平仓后继续
                    pass
                continue
            continue

        if pos is not None:
            continue

        # ========== 入场逻辑 ==========
        if ci < 14:
            continue

        # 使用实时聚合的15m数据更新引擎
        if bi == 2 and ci > 2 and ci < n_15 - 2:
            # add_bar(idx, high, low, close, h_arr, l_arr)
            #   high/low = eff_high[r15]/eff_low[r15] — 该15m bar的有效价格（用于极值比较）
            #   close = c15 (raw close数组, 用于趋势更新)
            #   h_arr/l_arr = eff_high/eff_low (完整有效价格数组, 用于极值检测左右比较)
            engine.add_bar(r15, eff_high[r15], eff_low[r15], c15, eff_high, eff_low)
            last_engine_bar = r15
            # 文档2.4: 维护BBW队列
            if not np.isnan(bbw_arr[ci]):
                bbw_queue.append(bbw_arr[ci])
                if len(bbw_queue) > 1000:
                    bbw_queue.pop(0)
        trend = engine.get_trend()

        # 激进通道（暂时关闭，待其他修复稳定后再启用）
        # agg_triggered, aggressive_dir = check_aggressive_channel(
        #     c15, h15, l15, o15, atr14, vsr, ci)
        # if agg_triggered and aggressive_dir:
        #     sl_agg = c15[ci] - 0.5 * atr14[ci] if aggressive_dir == 'LONG' else c15[ci] + 0.5 * atr14[ci]
        #     if sl_agg <= 0: continue
        #     pos = PositionSide.LONG if aggressive_dir == 'LONG' else PositionSide.SHORT
        #     eidx = i; eidx15 = r15; epr = c15s; sl = sl_agg
        #     sig_type_str = 'AGGRESSIVE_' + aggressive_dir
        #     lev = _calc_leverage(atr_pct)
        #     eadx = adx[ci] if ci < len(adx) else 20
        #     eatr = atr14[ci] if ci < n_15 else 0
        #     ersi = rsi[ci] if ci < len(rsi) else 50
        #     continue

        # 6.10 时效性：最新摆动点>4小时→无结构（停止信号生成）
        all_pivots = engine.get_pivots()
        if all_pivots:
            latest_pivot_ci = max(p.idx for p in all_pivots)
            if ci - latest_pivot_ci > 16:  # 4小时=16根15m K线
                trend = MarketTrendType.UNKNOWN

        # MTF方向
        dir1h = 1 if cur_ema20 > 0 and c15v > cur_ema20 else (2 if c15v < cur_ema20 * 0.998 else 0)
        dir4h = dir1h

        # ========== 信号检测前参数准备 ==========
        cur_adx_val = adx[ci] if ci < len(adx) else 0
        cur_rsi = rsi[ci] if ci < len(rsi) else 50

        # ========== SignalScore 共用参数（ci变化时才重新计算） ==========
        if ci != _last_sig_ci:
            _last_sig_ci = ci
            if ci < len(bbw_arr) and not np.isnan(bbw_arr[ci]) and len(bbw_queue) > 0:
                _bbw_percentile_val = get_bbw_percentile(bbw_queue, bbw_arr[ci])
            else:
                _bbw_percentile_val = 0.5
            _cur_vsr_pct = min(cur_vsr * 100, 100.0)
            _all_pivots = engine.get_pivots()
            _pivot_lows = [p for p in _all_pivots if p.type == 'LOW']
            _pivot_highs = [p for p in _all_pivots if p.type == 'HIGH']
            _trend_consistency = calc_trend_score(_pivot_lows, _pivot_highs, cur_adx_val)
            # 8.2 回调计数器：统计趋势开始后的回调次数
            trend_start_price, trend_start_idx = engine.get_trend_start() or (None, None)
            if trend_start_idx is not None and trend == MarketTrendType.UPTREND:
                # 1.4 仅统计价格>趋势起点的回调低点
                _pullback_count = sum(1 for p in _pivot_lows if p.idx > trend_start_idx and p.price > trend_start_price)
            elif trend_start_idx is not None and trend == MarketTrendType.DOWNTREND:
                # 1.4 仅统计价格<趋势起点的回调高点
                _pullback_count = sum(1 for p in _pivot_highs if p.idx > trend_start_idx and p.price < trend_start_price)
            else:
                _pullback_count = 99  # 非趋势状态，TR信号不会触发
            if _trend_consistency <= 0:
                if trend == MarketTrendType.UPTREND or trend == MarketTrendType.DOWNTREND:
                    _trend_consistency = 70
                elif trend == MarketTrendType.TRIANGLE:
                    _trend_consistency = 50
                elif trend == MarketTrendType.CONSOLIDATION:
                    _trend_consistency = 30
                elif trend == MarketTrendType.TRANSITION:
                    _trend_consistency = 15
                else:
                    _trend_consistency = 50
        bbw_percentile_val = _bbw_percentile_val
        cur_vsr_pct = _cur_vsr_pct
        trend_consistency = _trend_consistency
        pullback_count = _pullback_count if '_pullback_count' in dir() else 99

        # 只使用闭合15m K线生成信号（bi==2）
        if bi != 2:
            continue

        # ========== 8信号检测 ==========
        signals = []

        # 使用已闭合K线数据（c15[ci] = 上一根完成的15m收盘价）
        _close = c15[ci]
        _open_ = o15[ci]
        body_ratio = abs(_close - _open_) / max(atr14[ci], EPS) if atr14[ci] > 0 else 0
        solid_body = body_ratio >= 0.80
        is_bull = _close > _open_
        is_bear = _close < _open_

        recent_high = np.max(h15[max(0, ci-40):ci+1])
        recent_low = np.min(l5[max(0, ci-40):ci+1])

        # 规格书3.2: 使用DBSCAN聚类+穿透测试提取水平S/R，替代简单get_box()
        box = None
        _res_level, _sup_level, _sr_extra = find_horizontal_SR(
            h15, l15, c15, ci, atr_struct,
            eff_high=eff_high, eff_low=eff_low,
            spike_flags=spike_flags,
            bbw_percentile=bbw_percentile_val
        )
        if _res_level is not None and _sup_level is not None:
            _width = _res_level - _sup_level
            if 0.5 * atr14[ci] <= _width <= 3.0 * atr14[ci]:
                box = {
                    'top_upper': _res_level,
                    'top_lower': _res_level - 0.2 * atr_struct[ci],
                    'bottom_upper': _sup_level + 0.2 * atr_struct[ci],
                    'bottom_lower': _sup_level,
                    'range_high': (_res_level + _sup_level) / 2,
                    'range_low': (_res_level + _sup_level) / 2,
                    'width': _width,
                    'is_weak': (_sr_extra.get('res_filtered_penetrations', 0) >= 5
                                or _sr_extra.get('sup_filtered_penetrations', 0) >= 5),
                    'sr_source': 'find_horizontal_SR'
                }
        # fallback: 引擎get_box（当DBSCAN未找到有效S/R时）
        if box is None:
            _box_old = engine.get_box()
            if _box_old is not None:
                _w = _box_old['width'] / max(atr14[ci], EPS) if atr14[ci] > 0 else 0
                if 0.5 <= _w <= 3.0:
                    _pts = engine.get_pivots()
                    _nt = sum(1 for p in _pts if p.type == 'HIGH' and p.price >= _box_old['top_lower'] and p.price <= _box_old['top_upper'])
                    _nb = sum(1 for p in _pts if p.type == 'LOW' and p.price >= _box_old['bottom_lower'] and p.price <= _box_old['bottom_upper'])
                    if _nt >= 2 and _nb >= 2:
                        # 对get_box结果补充穿透验证
                        if ci >= 30:
                            _s = ci - 30
                            _abs = 0.5 * atr_struct[ci]
                            _pt = sum(1 for j in range(_s, ci) if c15[j] > _box_old['top_upper'] or eff_high[j] > _box_old['top_upper'] + _abs)
                            _pb = sum(1 for j in range(_s, ci) if c15[j] < _box_old['bottom_lower'] or eff_low[j] < _box_old['bottom_lower'] - _abs)
                            _ct = any(c15[j] > _box_old['top_upper'] and c15[j-1] > _box_old['top_upper'] for j in range(_s + 1, ci))
                            _cb = any(c15[j] < _box_old['bottom_lower'] and c15[j-1] < _box_old['bottom_lower'] for j in range(_s + 1, ci))
                            if _pt <= 3 and _pb <= 3 and not _ct and not _cb:
                                _rpt = sum(1 for j in range(_s, ci) if h15[j] > _box_old['top_upper'])
                                _rpb = sum(1 for j in range(_s, ci) if l15[j] < _box_old['bottom_lower'])
                                _box_old['is_weak'] = (_rpt >= 5 or _rpb >= 5)
                                _box_old['sr_source'] = 'get_box'
                                box = _box_old

        # 杠杆计算：根据止损距离反算，确保风险固定
        def calc_leverage_from_sl(entry, sl_price, direction='LONG'):
            """根据止损位反算杠杆，确保每笔最大亏损=MAX_LOSS"""
            if direction == 'LONG':
                sl_dist_pct = (entry - sl_price) / entry * 100
            else:
                sl_dist_pct = (sl_price - entry) / entry * 100
            sl_dist_pct = max(sl_dist_pct, 0.5)  # 至少0.5%
            lev = MAX_LOSS_CONFIG / sl_dist_pct  # 杠杆 = 最大亏损 / 止损距离
            return max(5, min(int(lev), BASE_LEV_CONFIG))

        # TR额外过滤：前一根需是回调K线（或小实体/反向）
        def _is_pullback_bar(ci):
            if ci < 1: return True
            prev_close = c15[ci - 1]
            prev_open = o15[ci - 1]
            prev_dir = 1 if prev_close > prev_open else -1
            cur_dir = 1 if is_bull else -1
            # 反向=回调，或小实体横盘
            return cur_dir != prev_dir or abs(_close - _open_) < 0.3 * max(atr14[ci], EPS)

        # 下一根已闭合K线的开盘价（作为入场价格，避免使用未闭合数据）
        _next_open = o15[min(ci + 1, n_15 - 1)]

        # ========== BO/RB信号（CONSOLIDATION） ==========
        if trend == MarketTrendType.CONSOLIDATION and box:
            box_top_upper = box['top_upper']
            box_top_lower = box['top_lower']
            box_bottom_upper = box['bottom_upper']
            box_bottom_lower = box['bottom_lower']
            box_range_high = box['range_high']
            box_range_low = box['range_low']
            box_width = box['width']

            ### BO_LONG：收盘价完全脱离箱体上沿区间 ###
            # 5.4.2 A级突破条件（全部满足可跳过双K线确认）
            is_agrade_long = is_bull and _close > box_top_upper * 1.01 and cur_vsr >= 1.5 and solid_body and \
                             bbw_percentile_val > 0.8 and cur_adx_val > 30 and \
                             (_close - box_top_upper) < atr14[ci]
            # 标准BO条件（需双K线确认）
            prev_closed_above = c15[ci - 1] > box_top_upper * 1.005 if ci >= 1 else False
            cur_closed_above = is_bull and _close > box_top_upper * 1.005 and cur_vsr >= 1.3
            bo_long_allowed = cur_closed_above and (is_agrade_long or prev_closed_above)
            # 7.3 弱突破重入检查：4h内弱突破后同方向可再入场（仓位减半）
            wb = _weak_breakouts.get((sym, 'LONG'))
            _pos_ratio_bo = 1.0
            if not bo_long_allowed and wb and ci - wb['ci'] <= 16:
                bo_long_allowed = cur_closed_above or (is_bull and _close > box_top_upper * 1.002 and cur_vsr >= 1.0)
                _pos_ratio_bo = 0.5
            if bo_long_allowed:
                mtf_pass, mtf_bonus = check_mtf('BO_LONG', dir1h, dir4h)
                if mtf_pass:
                    breakout_atr_pct = (_close - box_top_upper) / max(atr14[ci], EPS)
                    sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                    if should_trade:
                        # 5.4.4 止损：max(R - 0.2×ATR_trade_risk, 最后一个CONFIRMED低点)
                        _all_pts = engine.get_pivots()
                        _last_low = None
                        for _p in reversed(_all_pts):
                            if _p.type == 'LOW' and getattr(_p, 'confirmed_time', -1) >= 0 and _p.idx < ci:
                                _last_low = _p.price
                                break
                        _sl_r = box_top_upper - 0.2 * atr_risk[ci]
                        sl_bo = max(_sl_r, _last_low) if _last_low is not None else _sl_r
                        # 箱体弱边界：先锋仓位减半
                        if box.get('is_weak'):
                            _pos_ratio_bo *= 0.5
                        lev_sc = calc_leverage_from_sl(_next_open, sl_bo, 'LONG')
                        tp1 = _next_open + box_width * 0.5  # 目标：半个箱体宽度
                        signals.append(V83Signal(V83SignalType.BO_LONG, 'LONG', _next_open, sl_bo,
                                                 tp1, tp1 * 1.05, tp1 * 1.10,
                                                 sig_score, lev_sc, _pos_ratio_bo, ci, 5))

            ### BO_SHORT：收盘价完全脱离箱体下沿区间 ###
            # 5.4.2 A级突破条件
            is_agrade_short = is_bear and _close < box_bottom_lower * 0.99 and cur_vsr >= 1.5 and solid_body and \
                              bbw_percentile_val > 0.8 and cur_adx_val > 30 and \
                              (box_bottom_lower - _close) < atr14[ci]
            prev_closed_below = c15[ci - 1] < box_bottom_lower * 0.995 if ci >= 1 else False
            cur_closed_below = is_bear and _close < box_bottom_lower * 0.995 and cur_vsr >= 1.3
            bo_short_allowed = cur_closed_below and (is_agrade_short or prev_closed_below)
            # 7.3 弱突破重入检查（仓位减半）
            wb = _weak_breakouts.get((sym, 'SHORT'))
            _pos_ratio_bo = 1.0
            if not bo_short_allowed and wb and ci - wb['ci'] <= 16:
                bo_short_allowed = cur_closed_below or (is_bear and _close < box_bottom_lower * 0.997 and cur_vsr >= 1.0)
                _pos_ratio_bo = 0.5
            if bo_short_allowed:
                mtf_pass, mtf_bonus = check_mtf('BO_SHORT', dir1h, dir4h)
                if mtf_pass:
                    breakout_atr_pct = (box_bottom_lower - _close) / max(atr14[ci], EPS)
                    sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                    if should_trade:
                        # 5.4.4 止损：min(S + 0.2×ATR_trade_risk, 最后一个CONFIRMED高点)
                        _all_pts = engine.get_pivots()
                        _last_high = None
                        for _p in reversed(_all_pts):
                            if _p.type == 'HIGH' and getattr(_p, 'confirmed_time', -1) >= 0 and _p.idx < ci:
                                _last_high = _p.price
                                break
                        _sl_s = box_bottom_lower + 0.2 * atr_risk[ci]
                        sl_bo = min(_sl_s, _last_high) if _last_high is not None else _sl_s
                        # 箱体弱边界：先锋仓位减半
                        if box.get('is_weak'):
                            _pos_ratio_bo *= 0.5
                        lev_sc = calc_leverage_from_sl(_next_open, sl_bo, 'SHORT')
                        tp1 = _next_open - box_width * 0.5  # 目标：半个箱体宽度
                        signals.append(V83Signal(V83SignalType.BO_SHORT, 'SHORT', _next_open, sl_bo,
                                                 tp1, tp1 * 0.95, tp1 * 0.90,
                                                 sig_score, lev_sc, _pos_ratio_bo, ci, 5))

            ### RB_LONG：价格进入下沿区间 + 反弹阳线 ###
            elif l15[ci] <= box_bottom_upper and is_bull and cur_adx_val < 25:
                # 9.1 历史确认：过去100根K线至少有一次支撑反弹确认
                if not _check_rb_history(c15, l15, h15, atr14, ci, box_bottom_upper, box_top_upper):
                    pass  # 无历史确认，放弃RB_LONG
                else:
                    mtf_pass, mtf_bonus = check_mtf('RB_LONG', dir1h, dir4h)
                    if mtf_pass:
                        breakout_atr_pct = (box_bottom_upper - l15[ci]) / max(atr14[ci], EPS)
                        sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                        if should_trade:
                            # 5.3.2 止损：min(S - 0.3×ATR, 摆动低点 - 0.2×ATR)
                            _sl_s_minus = box_bottom_lower - 0.3 * atr_risk[ci]
                            _sl_low_minus = l15[ci] - 0.2 * atr_risk[ci]
                            sl_rb = min(_sl_s_minus, _sl_low_minus)
                            lev_sc = calc_leverage_from_sl(_next_open, sl_rb, 'LONG')
                            tp1 = box_range_high  # 目标：箱体中段
                            signals.append(V83Signal(V83SignalType.RB_LONG, 'LONG', _next_open, sl_rb,
                                                     tp1, tp1 * 1.05, tp1 * 1.10,
                                                     sig_score, lev_sc, 1.0, ci, 5))

            ### RB_SHORT：价格进入上沿区间 + 回落阴线 ###
            elif h15[ci] >= box_top_lower and is_bear and cur_adx_val < 25:
                # 9.1 历史确认：过去100根K线至少有一次阻力回落实证
                if not _check_rb_history(c15, l15, h15, atr14, ci, box_bottom_lower, box_top_lower, direction='SHORT'):
                    pass  # 无历史确认，放弃RB_SHORT
                else:
                    mtf_pass, mtf_bonus = check_mtf('RB_SHORT', dir1h, dir4h)
                    if mtf_pass:
                        breakout_atr_pct = (h15[ci] - box_top_lower) / max(atr14[ci], EPS)
                        sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                        if should_trade:
                            # 止损位：箱体上沿最高点上方
                            # 5.3.4 止损：max(R + 0.3×ATR, 摆动高点 + 0.2×ATR)
                            _sl_s_plus = box_top_upper + 0.3 * atr_risk[ci]
                            _sl_high_plus = h15[ci] + 0.2 * atr_risk[ci]
                            sl_rb = max(_sl_s_plus, _sl_high_plus)
                            lev_sc = calc_leverage_from_sl(_next_open, sl_rb, 'SHORT')
                            tp1 = box_range_low  # 目标：箱体中段
                            signals.append(V83Signal(V83SignalType.RB_SHORT, 'SHORT', _next_open, sl_rb,
                                                     tp1, tp1 * 0.95, tp1 * 0.90,
                                                     sig_score, lev_sc, 1.0, ci, 5))

        # ========== 10章 三角形信号（TRIANGLE） ==========
        if trend == MarketTrendType.TRIANGLE:
            tri = engine.get_triangle()
            if tri is not None:
                # 投影斜线到当前K线位置
                top_at_ci = tri['top'] + tri['top_slope'] * (ci - tri['top_idx'])
                bottom_at_ci = tri['bottom'] + tri['bottom_slope'] * (ci - tri['bottom_idx'])
                tri_width = top_at_ci - bottom_at_ci

                ### TRIANGLE_LONG：价格向上突破下降上轨 ###
                if is_bull and _close > top_at_ci * 1.005 and solid_body:
                    mtf_pass, mtf_bonus = check_mtf('BO_LONG', dir1h, dir4h)
                    if mtf_pass:
                        breakout_atr_pct = (_close - top_at_ci) / max(atr14[ci], EPS)
                        sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                        if should_trade:
                            # 5.4.4 止损：三角形下轨 - 0.2×ATR(结构与ATR类比)
                            sl_tri = bottom_at_ci - 0.2 * atr_risk[ci]
                            min_sl_tri = _next_open * (1 - 0.005)  # 至少0.5%距离
                            if sl_tri > min_sl_tri:
                                sl_tri = min_sl_tri
                            lev_sc = calc_leverage_from_sl(_next_open, sl_tri, 'LONG')
                            tp1 = _next_open + tri_width * 0.5
                            signals.append(V83Signal(V83SignalType.BO_LONG, 'LONG', _next_open, sl_tri,
                                                     tp1, tp1 * 1.05, tp1 * 1.10,
                                                     sig_score, lev_sc, 1.0, ci, 5))

                ### TRIANGLE_SHORT：价格向下突破上升下轨 ###
                elif is_bear and _close < bottom_at_ci * 0.995 and solid_body:
                    mtf_pass, mtf_bonus = check_mtf('BO_SHORT', dir1h, dir4h)
                    if mtf_pass:
                        breakout_atr_pct = (bottom_at_ci - _close) / max(atr14[ci], EPS)
                        sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                        if should_trade:
                            # 5.4.4 止损：三角形上轨 + 0.2×ATR
                            sl_tri = top_at_ci + 0.2 * atr_risk[ci]
                            min_sl_tri = _next_open * (1 + 0.005)  # 至少0.5%距离
                            if sl_tri < min_sl_tri:
                                sl_tri = min_sl_tri
                            lev_sc = calc_leverage_from_sl(_next_open, sl_tri, 'SHORT')
                            tp1 = _next_open - tri_width * 0.5
                            signals.append(V83Signal(V83SignalType.BO_SHORT, 'SHORT', _next_open, sl_tri,
                                                     tp1, tp1 * 0.95, tp1 * 0.90,
                                                     sig_score, lev_sc, 1.0, ci, 5))

        ### TR_LONG ###
        if trend == MarketTrendType.UPTREND and is_bull:
            # 8.2 回调计数器：仅第1/2次回调可入场
            # 1.4 价格过滤：仅计数价格>趋势起点的回调
            if pullback_count > 2:
                pass
            elif cur_rsi < 75 and _is_pullback_bar(ci):
                mtf_pass, mtf_bonus = check_mtf('TR_LONG', dir1h, dir4h)
                if mtf_pass:
                    p1 = engine.get_p1('LONG')   # 从引擎获取P1
                    if p1 is not None:
                        # 附录A Q5: 摩擦衰减检查
                        if _check_friction_decayed(c15, atr14, p1, ci):
                            friction_decayed = True
                        else:
                            friction_decayed = False
                        if friction_decayed:
                            pass  # TR入场前摩擦衰减，放弃信号
                        else:
                            # 1.5 趋势强度过滤：最近推动浪/前次推动浪 >= 0.5
                            _allow_tr = True
                            if len(_pivot_lows) >= 3 and len(_pivot_highs) >= 3:
                                recent_impulse = _pivot_highs[-1].price - _pivot_lows[-1].price
                                prev_impulse = _pivot_highs[-2].price - _pivot_lows[-2].price
                                if prev_impulse > 0 and recent_impulse / prev_impulse < 0.5:
                                    _allow_tr = False  # 趋势衰竭，跳过
                            if _allow_tr:
                                dist_to_p1 = abs(_close - p1) / max(_close, EPS) * 100
                                if dist_to_p1 <= 1.5 * atr_pct:
                                    breakout_atr_pct = abs(_close - p1) / max(atr14[ci], EPS)
                                    sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                                    if should_trade:
                                        # 5.2.5 止损：回调摆动低点 - 0.2×ATR_trade_risk
                                        sl_tr = p1 - 0.2 * atr_risk[ci]
                                        lev_sc = calc_leverage_from_sl(_next_open, sl_tr, 'LONG')
                                        tp1 = _next_open + 1.5 * atr14[ci]  # 目标：1.5倍ATR
                                        signals.append(V83Signal(V83SignalType.TR_LONG, 'LONG', _next_open, sl_tr,
                                                                 tp1, tp1 * 1.05, tp1 * 1.10,
                                                                 sig_score, lev_sc, 1.0, ci, 5))

        ### TR_SHORT ###
        if trend == MarketTrendType.DOWNTREND and is_bear:
            # 8.2 回调计数器：仅第1/2次回调可入场
            if pullback_count > 2:
                pass
            elif cur_rsi > 25 and _is_pullback_bar(ci):
                mtf_pass, mtf_bonus = check_mtf('TR_SHORT', dir1h, dir4h)
                if mtf_pass:
                    p1 = engine.get_p1('SHORT')   # 从引擎获取P1
                    if p1 is not None:
                        # 附录A Q5: 摩擦衰减检查
                        if _check_friction_decayed(c15, atr14, p1, ci):
                            friction_decayed = True
                        else:
                            friction_decayed = False
                        if friction_decayed:
                            pass  # TR入场前摩擦衰减，放弃信号
                        else:
                            # 1.5 趋势强度过滤：最近推动浪/前次推动浪 >= 0.5
                            _allow_tr = True
                            if len(_pivot_lows) >= 3 and len(_pivot_highs) >= 3:
                                recent_impulse = _pivot_highs[-1].price - _pivot_lows[-1].price
                                prev_impulse = _pivot_highs[-2].price - _pivot_lows[-2].price
                                if prev_impulse > 0 and recent_impulse / prev_impulse < 0.5:
                                    _allow_tr = False  # 趋势衰竭，跳过
                            if _allow_tr:
                                dist_to_p1 = abs(_close - p1) / max(_close, EPS) * 100
                                if dist_to_p1 <= 1.5 * atr_pct:
                                    breakout_atr_pct = abs(_close - p1) / max(atr14[ci], EPS)
                                    sig_score, should_trade = calc_signal_score(cur_vsr_pct, breakout_atr_pct, bbw_percentile_val, trend_consistency)
                                    if should_trade:
                                        # 5.2.5 止损：反弹摆动高点 + 0.2×ATR_trade_risk
                                        sl_tr = p1 + 0.2 * atr_risk[ci]
                                        lev_sc = calc_leverage_from_sl(_next_open, sl_tr, 'SHORT')
                                        tp1 = _next_open - 1.5 * atr14[ci]  # 目标：1.5倍ATR
                                        signals.append(V83Signal(V83SignalType.TR_SHORT, 'SHORT', _next_open, sl_tr,
                                                                 tp1, tp1 * 0.95, tp1 * 0.90,
                                                                 sig_score, lev_sc, 1.0, ci, 5))

        # 仲裁
        final_sig = arbitrate_signals(signals)
        if final_sig is None:
            continue

        sl_price = final_sig.stop_loss
        if sl_price is None or sl_price <= 0:
            continue

        # 冷却检查
        if not cooldown_mgr.check(sym, final_sig.direction, ci):
            continue

        pos = PositionSide.LONG if final_sig.direction == 'LONG' else PositionSide.SHORT
        eidx = i
        eidx15 = r15
        epr = final_sig.entry_price
        sl = sl_price
        sig_type_str = final_sig.signal_type.value
        lev = final_sig.leverage
        eadx = adx[ci] if ci < len(adx) else 20
        eatr = atr14[ci] if ci < n_15 else 0
        ersi = rsi[ci] if ci < len(rsi) else 50
        entry_trend_str = trend.value if trend else 'UNKNOWN'
        entry_score = final_sig.score
        # S/R：根据信号类型设置
        entry_sup = 0.0
        entry_res = 0.0
        # 基石线（foundation_line）：趋势的起点，作为支撑/阻力
        foundation_line_idx = 0
        foundation_line_val = 0.0
        foundation_line_slope = 0.0

        if box:
            if 'BO_LONG' in sig_type_str or 'RB_LONG' in sig_type_str:
                entry_sup = box['bottom_lower']
                entry_res = box['top_upper']
            elif 'BO_SHORT' in sig_type_str or 'RB_SHORT' in sig_type_str:
                entry_sup = box['bottom_lower']
                entry_res = box['top_upper']
        elif 'TR_LONG' in sig_type_str:
            p1_val = engine.get_p1('LONG')
            if p1_val:
                entry_sup = p1_val
                # 基石线 = P1，从P1到入场点的趋势线
                foundation_line_idx = engine.p1_idx if engine.p1_idx else ci
                foundation_line_val = p1_val
                foundation_line_slope = (_close - p1_val) / max(ci - foundation_line_idx, 1)
        elif 'TR_SHORT' in sig_type_str:
            p1_val = engine.get_p1('SHORT')
            if p1_val:
                entry_res = p1_val
                # 基石线 = P1，从P1到入场点的趋势线
                foundation_line_idx = engine.p1_idx if engine.p1_idx else ci
                foundation_line_val = p1_val
                foundation_line_slope = (_close - p1_val) / max(ci - foundation_line_idx, 1)

    return rlist


if __name__ == '__main__':
    if os.environ.get('YZDIR'):
        fs = [f for f in os.listdir(DR) if '_5m_2025-04-01_2026-05-31.csv' in f]
    else:
        fs = [f for f in os.listdir(DR) if f.endswith('_5m_2025-03-01_2026-05-01.csv')]
    _save_sample = os.environ.get('SAVE_SAMPLE', '')
    if _save_sample:
        _wanted = set(_save_sample.split(','))
        fs = [f for f in fs if any(w in f for w in _wanted)]
    fs_all = sorted(fs)
    print('Found %d 5m files total' % len(fs_all), flush=True)
    t_start = time.time()
    all_trades = []
    results = []
    for f in fs_all:
        sym = f.split('_5m')[0]
        print('Loading %s...' % sym, end=' ', flush=True)
        t0 = time.time()
        df = pd.read_csv(DR + '/' + f).dropna(subset=['close']).reset_index(drop=True)
        print('running...', end=' ', flush=True)
        trades = run_v83(df, sym, None)
        print('done %d trades (%.1fs)' % (len(trades), time.time() - t0), flush=True)
        total_pnl = sum(t.pnl_pct for t in trades)
        wins = [t for t in trades if t.pnl_pct > 0]
        results.append((sym, len(trades), total_pnl, 100 * len(wins) / len(trades) if trades else 0))
        all_trades.extend(trades)
        gc.collect()
    print('Total %.1fs for %d coins' % (time.time() - t_start, len(results)), flush=True)

    import csv as _csv
    out_path = r'E:\BNFF\BNFRich\logs\all_trades_v83.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as _f:
        _w = _csv.writer(_f)
        _w.writerow(['sym','entry_idx','exit_idx','side','entry_price','sl','leverage','pnl_pct',
                     'hold_bars_5m','exit_reason','signal_type','trend_type','max_pnl_pct','min_pnl_pct',
                     'entry_adx','entry_atr','entry_rsi'])
        for _t in all_trades:
            _w.writerow([_t.symbol, _t.entry_idx_5m, _t.exit_idx_5m, _t.side,
                         _t.entry_price, _t.stop_loss, _t.leverage, _t.pnl_pct,
                         _t.hold_bars_5m, _t.exit_reason, _t.signal_type, _t.trend_type,
                         _t.max_pnl_pct, _t.min_pnl_pct, _t.entry_adx, _t.entry_atr, _t.entry_rsi])
    print('Saved %d trades to %s' % (len(all_trades), out_path), flush=True)

    results.sort(key=lambda x: -x[2])
    print('\n========== 全币结果（按PnL降序）==========')
    for sym, n, pnl, wr in results:
        print('%s: %d笔 PnL=%+.0f%% 胜率=%.0f%%' % (sym, n, pnl, wr))

    total_pnl = sum(t.pnl_pct for t in all_trades)
    wins = [t for t in all_trades if t.pnl_pct > 0]
    print('\n========== 合计 ==========')
    print('Total: %d trades, PnL=%.0f%%, Win=%d/%d=%.0f%%' % (
        len(all_trades), total_pnl, len(wins), len(all_trades),
        100 * len(wins) / len(all_trades) if all_trades else 0))

    sigs = {}
    for t in all_trades:
        sigs.setdefault(t.signal_type[:20], []).append(t)
    print('\n========== 按信号类型 ==========')
    for sn in sorted(sigs):
        tl = sigs[sn]
        w = [t for t in tl if t.pnl_pct > 0]
        if len(tl) >= 1:
            print('%s: %d笔 PnL=%.0f%% 胜=%d/%d=%.0f%%' % (
                sn, len(tl), sum(t.pnl_pct for t in tl), len(w), len(tl),
                100 * len(w) / len(tl)))

    print('\n========== 期望值(EV)分析 ==========')
    for sn in sorted(sigs):
        tl = sigs[sn]
        if len(tl) < 3: continue
        w = [t for t in tl if t.pnl_pct > 0]
        l_ = [t for t in tl if t.pnl_pct <= 0]
        wr = len(w) / len(tl)
        avg_win = sum(t.pnl_pct for t in w) / len(w) if w else 0
        avg_loss = sum(t.pnl_pct for t in l_) / len(l_) if l_ else 0
        ev = wr * avg_win - (1 - wr) * abs(avg_loss)
        print('%s: %d笔 胜率=%.0f%% avg_win=+%.1f%% avg_loss=%.1f%% EV=%.2f%%' % (
            sn, len(tl), wr * 100, avg_win, avg_loss, ev))
