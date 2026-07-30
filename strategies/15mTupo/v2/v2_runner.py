# -*- coding: utf-8 -*-
"""
v2 回测运行器 — 完全按照新开发文档实现
复用 run_final.py 的退出逻辑，替换入场逻辑
运行方式: python strategies/15mTupo/v2/v2_runner.py
"""
import sys, os, gc, pandas as pd, numpy as np, time, datetime
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
BacktestSettings = _tupo_core.BacktestSettings
aggregate_5m_to_15m_fast = _tupo_core.aggregate_5m_to_15m_fast
calculate_adx_fast = _tupo_core.calculate_adx_fast
calculate_rsi = _tupo_core.calculate_rsi
calculate_avg_volume_fast = _tupo_core.calculate_avg_volume_fast
set_atr_ma40_buffer = _tupo_core.set_atr_ma40_buffer

_v2_system = importlib.import_module('strategies.15mTupo.v2.v2_system')
clean_wicks = _v2_system.clean_wicks
SRScoringEngine = _v2_system.SRScoringEngine
classify_morphology = _v2_system.classify_morphology
_frac_extrema = _v2_system._frac_extrema
generate_signal = _v2_system.generate_signal
V2TrendType = _v2_system.TrendType
SignalType = _v2_system.SignalType
V2Signal = _v2_system.V2Signal

DR = r"E:\BNFF\BNFRich\data\historical"
if os.environ.get('YZDIR'):
    DR = r"E:\BNFF\BNFRich\data\historical\yanzheng"

s = BacktestSettings()
BASE_LEV_CONFIG = int(os.environ.get("BASE_LEV", "15"))
MAX_LOSS_CONFIG = int(os.environ.get("MAX_LOSS", "30"))
SLIPPAGE_PCT = float(os.environ.get("SLIPPAGE", "0.03"))
FUNDING_PER_BAR = float(os.environ.get("FUNDING", "0.0001"))
BE_MX_PCT = float(os.environ.get("BE_MX_PCT", "5.0"))
BE_SL_BUFFER_PCT = float(os.environ.get("BE_SL_BUFFER_PCT", "0"))
BE_HARD_MX = float(os.environ.get("BE_HARD_MX", "0"))
BE_WEAK_MX = float(os.environ.get("BE_WEAK_MX", "0"))
BE_WEAK_VOL = float(os.environ.get("BE_WEAK_VOL", "5.0"))
TREND_BE_MX = float(os.environ.get("TREND_BE_MX", "0"))
VOL_CONFIRM_THRESH = float(os.environ.get("VOL_CONFIRM_THRESH", "2.0"))
LIQ_THRESHOLD = float(os.environ.get("LIQ_THRESHOLD", "-90"))
LIQ_PENALTY = float(os.environ.get("LIQ_PENALTY", "0.5"))
LIQ_MM_RATE = float(os.environ.get("LIQ_MM_RATE", "0.5"))
trail_on = True
trail_act = float(os.environ.get('TRAIL_ACT', '8.0'))
trail_dist = float(os.environ.get('TRAIL_DIST', '2.5'))
vel_floor = float(os.environ.get('VEL_FLOOR', '12'))
vel_min_mx = float(os.environ.get('VEL_MIN_MX', '8'))
vel_window = int(os.environ.get('VEL_WINDOW', '3'))
vel_floor_multi = float(os.environ.get('VEL_FLOOR_MULTI', '15'))


def run_v2(df_5m, sym, s_obj):
    c5 = df_5m['close'].values.astype(float)
    h5 = df_5m['high'].values.astype(float)
    l5 = df_5m['low'].values.astype(float)
    v5 = df_5m['volume'].values.astype(float)
    o5 = df_5m['open'].values.astype(float)

    # 毛刺清洗
    h5c, l5c = clean_wicks(o5, h5, l5, c5)

    df15 = aggregate_5m_to_15m_fast(df_5m)
    if len(df15) < 100:
        return []

    c15 = df15['close'].values
    h15 = df15['high'].values
    l15 = df15['low'].values
    o15 = df15['open'].values
    v15 = df15['volume'].values.astype(float)

    # 毛刺清洗15m
    h15c, l15c = clean_wicks(o15, h15, l15, c15)

    adx, pdm, mdm, atr = calculate_adx_fast(h15c, l15c, c15, 14)
    rsi = calculate_rsi(c15, 14)
    ema20 = pd.Series(c15).ewm(span=20).mean().values
    avgv = calculate_avg_volume_fast(v15, 20)
    set_atr_ma40_buffer(np.where(c15 > 0, atr / c15 * 100, 0))

    # ATR 100根均值（用于熔断）
    atr_100_mean = 0.0
    if len(atr) > 100:
        atr_100_mean = np.mean(atr[-100:])

    n_15 = len(c15)

    # S/R计分引擎（15m bar index，96根=24h，文档原文参数）
    sr_engine = SRScoringEngine(window_bars=96, hit_tolerance_pct=0.001,
                                min_interval_bars=4, activate_threshold=3.0,
                                debounce_pct=0.002)

    # 形态分类缓存（每1小时/4根15m更新一次）
    morph = V2TrendType.UNKNOWN
    slope_h = 0.0
    slope_l = 0.0
    last_morph_ci = -999

    rlist = []
    pos = None
    eidx = 0
    eidx15 = 0
    epr = 0.0
    sl = 0.0
    lev = 1
    st = ''
    eadx = 20
    eatr = 0.0
    ersi = 50.0
    tsp = 0.0
    tc = 0
    pr = 1.0
    ap = 0.0
    mx = 0.0
    mn = 0.0
    hrt = False
    post_ladder_peak = 0.0
    ladder_exits = []
    _after_ladder = False
    ph = []
    eres = 0.0
    esup = 0.0
    sr_room = 0.0
    o15s = 0.0
    h15s = 0.0
    l15s = 0.0
    c15s = 0.0
    v15s = 0.0
    trail_high = 0.0
    trail_active = False
    prev_pnl = 0
    peak_vol_ratio = 0.0

    for i in range(60, len(c5)):
        cp = c5[i]
        bi = i % 3
        r15 = i // 3
        if r15 >= len(c15):
            continue
        ci = max(0, r15 - 1)
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
        c15v = c15[ci] if ci < len(c15) else c15[-1]
        vv = atr[ci] / c15v if atr[ci] > 0 else 0.02

        # ========== 退出逻辑（从 run_final.py 复制） ==========
        if pos is not None:
            pnl = 0.0
            sl_hit = False
            hb = i - eidx
            sc2 = False
            exit_p = epr

            # 5m SL 检查
            if (pos == PositionSide.LONG and l5[i] <= sl) or (pos == PositionSide.SHORT and h5[i] >= sl):
                sl_hit = True
                pnl = abs(epr - sl) / epr * 100 * lev if pos == PositionSide.LONG else abs(sl - epr) / epr * 100 * lev
                pnl = -pnl
                pnl -= FUNDING_PER_BAR * lev
                pnl = max(pnl, -100.0)
                pnl -= SLIPPAGE_PCT * 2 * lev
                sc2, er = True, '触发止损'
                cr = 1.0
                exit_p = sl

            # BE逻辑
            if not sl_hit:
                if pos == PositionSide.LONG:
                    if mx >= BE_MX_PCT and sl < epr and peak_vol_ratio >= VOL_CONFIRM_THRESH:
                        sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                    elif hb >= 6 and max(mx, (h5[i] - epr) / epr * 100 * lev) >= BE_MX_PCT and sl < epr:
                        _5m_vr = v5[i] / max(avgv[ci] / 3, 0.001) if ci >= 0 and ci < len(avgv) and avgv[ci] > 0 else 0
                        if _5m_vr >= VOL_CONFIRM_THRESH:
                            sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                elif pos == PositionSide.SHORT:
                    if mx >= BE_MX_PCT and sl > epr and peak_vol_ratio >= VOL_CONFIRM_THRESH:
                        sl = epr * (1 + BE_SL_BUFFER_PCT / 100)
                    elif hb >= 6 and max(mx, (epr - l5[i]) / epr * 100 * lev) >= BE_MX_PCT and sl > epr:
                        _5m_vr = v5[i] / max(avgv[ci] / 3, 0.001) if ci >= 0 and ci < len(avgv) and avgv[ci] > 0 else 0
                        if _5m_vr >= VOL_CONFIRM_THRESH:
                            sl = epr * (1 + BE_SL_BUFFER_PCT / 100)

            if not sl_hit and BE_HARD_MX > 0 and mx >= BE_HARD_MX:
                if pos == PositionSide.LONG and sl < epr:
                    sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                elif pos == PositionSide.SHORT and sl > epr:
                    sl = epr * (1 + BE_SL_BUFFER_PCT / 100)

            if not sl_hit and BE_WEAK_MX > 0 and mx >= BE_WEAK_MX:
                _cur_vr = v5[i] / max(avgv[ci] / 3, 0.001) if ci >= 0 and ci < len(avgv) and avgv[ci] > 0 else 0
                if _cur_vr < BE_WEAK_VOL:
                    if pos == PositionSide.LONG and sl < epr:
                        sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                    elif pos == PositionSide.SHORT and sl > epr:
                        sl = epr * (1 + BE_SL_BUFFER_PCT / 100)

            if not sl_hit and TREND_BE_MX > 0 and mx >= TREND_BE_MX:
                _morph_tt = 'CONSOLIDATION' if morph in (V2TrendType.RECTANGLE,) else morph.value
                if _morph_tt not in ('CONSOLIDATION', 'UNKNOWN'):
                    if pos == PositionSide.LONG and sl < epr:
                        sl = epr * (1 - BE_SL_BUFFER_PCT / 100)
                    elif pos == PositionSide.SHORT and sl > epr:
                        sl = epr * (1 + BE_SL_BUFFER_PCT / 100)

            # 爆仓检查
            if not sl_hit and ((pos == PositionSide.LONG and l5[i] <= epr * (1 - 1.0 / lev + LIQ_MM_RATE / 100)) or \
                 (pos == PositionSide.SHORT and h5[i] >= epr * (1 + 1.0 / lev - LIQ_MM_RATE / 100))):
                sl_hit = True
                liq_loss_pct = (1.0 / lev - LIQ_MM_RATE / 100) * 100 * lev
                pnl = -liq_loss_pct
                pnl -= FUNDING_PER_BAR * lev
                pnl = max(pnl, -100.0)
                pnl -= SLIPPAGE_PCT * 2 * lev
                if pos == PositionSide.LONG:
                    exit_p = epr * (1 - 1.0 / lev + LIQ_MM_RATE / 100)
                else:
                    exit_p = epr * (1 + 1.0 / lev - LIQ_MM_RATE / 100)
                pnl = max(pnl, LIQ_THRESHOLD)
                sc2, er = True, '爆仓'
                cr = 1.0

            # 15m收盘管理
            if not sl_hit and bi == 2:
                pnl_c = (c15s - epr) / epr * 100 * lev if pos == PositionSide.LONG else (epr - c15s) / epr * 100 * lev
                pnl = pnl_c
                pnl -= FUNDING_PER_BAR * lev * 3
                pnl = max(pnl, -100.0)
                cur_vr = v15s / max(avgv[ci], 0.001) if ci >= 0 and ci < len(avgv) and avgv[ci] > 0 else 0
                if pnl_c > mx:
                    peak_vol_ratio = max(peak_vol_ratio, cur_vr)
                mx = max(mx, pnl_c)
                mn = min(mn, pnl_c)
                ph.append(pnl_c)
                if len(ph) > 10:
                    ph.pop(0)
                if _after_ladder and pnl_c > post_ladder_peak:
                    post_ladder_peak = pnl_c
                hb15 = r15 - eidx15
                sc2 = False
                er = ''
                cr = 1.0
                exit_p = c15s

                # 速度退出
                if vel_floor > 0 and mx >= vel_min_mx and prev_pnl != 0 and prev_pnl - pnl_c >= vel_floor:
                    sc2, er = True, '速度退出'
                if not sc2 and vel_window > 0 and vel_floor_multi > 0 and mx >= vel_min_mx and len(ph) >= vel_window:
                    if max(ph[-vel_window:]) - ph[-1] >= vel_floor_multi:
                        sc2, er = True, '速度退出'
                if not sc2 and vel_floor > 0 and mx >= vel_min_mx and epr > 0:
                    pnl_h5 = (h5[i] - epr) / epr * 100 * lev if pos == PositionSide.LONG else (epr - l5[i]) / epr * 100 * lev
                    if pnl_h5 > mx:
                        pnl_h5 = mx
                    if pnl_h5 - pnl_c >= vel_floor:
                        sc2, er = True, '速度退出'

                prev_pnl = pnl_c

                # 移动止盈
                if trail_on and trail_active:
                    if pos == PositionSide.LONG:
                        trail_high = max(trail_high, c15s)
                        if c15s <= trail_high * (1 - trail_dist / 100):
                            sc2, er = True, '移动止盈'
                    else:
                        trail_high = min(trail_high, c15s)
                        if c15s >= trail_high * (1 + trail_dist / 100):
                            sc2, er = True, '移动止盈'
                if trail_on and not trail_active:
                    if pos == PositionSide.LONG and pnl_c >= trail_act:
                        trail_high = c15s
                        trail_active = True
                    elif pos == PositionSide.SHORT and pnl_c >= trail_act:
                        trail_high = c15s
                        trail_active = True

                # 阶梯止盈
                if not sc2 and tc < 5:
                    if pnl_c > tsp:
                        tsp = pnl_c
                    _lp = s.ladder_peak_uptrend
                    if morph in (V2TrendType.RECTANGLE,):
                        _lp = s.ladder_peak_rebound
                    elif morph in (V2TrendType.SYM_TRIANGLE, V2TrendType.ASC_TRIANGLE, V2TrendType.DESC_TRIANGLE):
                        if s.ladder_peak_tr > 0:
                            _lp = s.ladder_peak_tr
                    if tsp >= _lp:
                        ddt = [s.ladder_dd_t1_uptrend, s.ladder_dd_t2_uptrend, s.ladder_dd_t3_uptrend,
                               s.ladder_dd_t4_uptrend, s.ladder_dd_t5_uptrend]
                        if morph in (V2TrendType.RECTANGLE,):
                            ddt = [s.ladder_dd_t1_rebound, s.ladder_dd_t2_rebound, s.ladder_dd_t3_rebound,
                                   s.ladder_dd_t4_rebound, s.ladder_dd_t5_rebound]
                        elif morph in (V2TrendType.SYM_TRIANGLE, V2TrendType.ASC_TRIANGLE, V2TrendType.DESC_TRIANGLE):
                            if s.ladder_peak_tr > 0:
                                ddt = [s.ladder_dd_t1_tr, s.ladder_dd_t2_tr, s.ladder_dd_t3_tr,
                                       s.ladder_dd_t4_tr, s.ladder_dd_t5_tr]
                        dd_trigger = tsp - pnl_c >= ddt[tc]
                        if dd_trigger:
                            cr = max(0, min([s.ladder_close_t1 / 100, s.ladder_close_t2 / 100,
                                             s.ladder_close_t3 / 100, s.ladder_close_t4 / 100,
                                             s.ladder_close_t5 / 100][tc] / pr, 1.0))
                            ladder_exits.append({'tc': tc, 'pnl': pnl_c, 'mx_so_far': mx, 'cr': cr,
                                                 'price': c15s, 'step': tc + 1})
                            _after_ladder = True
                            post_ladder_peak = pnl_c
                            sc2, er = True, '阶梯止盈'
                            tc += 1
                            hrt = True

                if not sc2 and hb15 >= s.entry_stop_bars // 3 and mx <= 0:
                    sc2, er = True, '入场止损'

                if sc2:
                    pnl -= SLIPPAGE_PCT * 2 * lev

            if sl_hit or sc2:
                ap += pnl * cr * pr
                pr *= (1 - cr)
                if pr < 0.01 or cr >= 1.0:
                    if hrt:
                        er = '阶梯止盈'
                    if ladder_exits:
                        final_pnl = pnl_c if not sl_hit else -abs(epr - sl) / epr * 100 * lev
                        for le in ladder_exits:
                            missed = post_ladder_peak - le['pnl']
                    rlist.append(TradeResult(sym, eidx, epr, i, exit_p, pos.value, lev,
                                             ap - 2 * lev * s_obj.trading_fee_rate * 100, hb, er,
                                             morph.value, st, mx, mn, 0, eadx, eatr, ersi,
                                             v5[i] if i < len(v5) else 0, 0,
                                             rsi[ci] if ci < len(rsi) else 50,
                                             False, False, sr_room, sl))
                    pos = None
                    pr = 1.0
                    ap = 0.0
                    mx = mn = 0.0
                    hrt = False
                    tsp = 0.0
                    tc = 0
                    ph.clear()
                    trail_active = False
                    trail_high = 0.0
                    prev_pnl = 0
                    peak_vol_ratio = 0.0
                    ladder_exits = []
                    _after_ladder = False
                    post_ladder_peak = 0.0
                else:
                    tsp = pnl
                continue
            continue

        if pos is not None:
            continue

        # ========== v2 入场逻辑 ==========
        # S/R更新
        sr_engine.on_bar(ci, h15c[ci], l15c[ci], c15[ci], o15[ci])
        support, resistance, sr_count = sr_engine.get_levels(c15s)

        # 形态分类 + 分形极值提取（每4根15m更新一次）
        if ci - last_morph_ci >= 4 or last_morph_ci < 0:
            morph, slope_h, slope_l = classify_morphology(h15c, l15c, ci, lookback=120)
            last_morph_ci = ci
            # 提取分形极值点，喂给SR引擎作为候选池
            start = max(0, ci - 119)
            seg_h = h15c[start:ci + 1]
            seg_l = l15c[start:ci + 1]
            fh, fl = _frac_extrema(seg_h, seg_l, left=2, right=2)
            fractal_highs = [(start + idx, val) for idx, val in fh]
            fractal_lows = [(start + idx, val) for idx, val in fl]
            sr_engine.set_fractal_levels(fractal_highs, fractal_lows, ci)

        if morph == V2TrendType.UNKNOWN:
            continue

        # 生成信号
        sig = generate_signal(
            ci15=ci, c15=c15, h15=h15c, l15=l15c, o15=o15, v15=v15,
            atr=atr, ema20=ema20, support=support, resistance=resistance,
            sr_count=sr_count, morph=morph, slope_h=slope_h, slope_l=slope_l,
            atr_100_mean=atr_100_mean
        )

        if isinstance(sig, SignalType):
            continue

        if isinstance(sig, V2Signal):
            side = PositionSide.LONG if sig.side == 'LONG' else PositionSide.SHORT
            entry_price = c15s

            # 杠杆计算
            atr_pct = atr[ci] / c15v * 100 if c15v > 0 else 1.0
            base_lev = BASE_LEV_CONFIG
            if atr_pct > 0:
                lev_calc = MAX_LOSS_CONFIG / max(atr_pct * 0.5, 0.1)
                lev = max(5, min(int(lev_calc), base_lev))
            else:
                lev = base_lev
            lev = min(lev, 30)

            sl_price = sig.sl
            if sl_price is None or sl_price <= 0:
                continue

            pos = side
            eidx = i
            eidx15 = r15
            epr = entry_price
            sl = sl_price
            st = sig.signal_type.value
            eadx = adx[ci] if ci < len(adx) else 20
            eatr = atr[ci] if ci < len(atr) else 0
            ersi = rsi[ci] if ci < len(rsi) else 50
            eres = resistance or 0.0
            esup = support or 0.0
            sr_room = 0.0
            if side == PositionSide.LONG and eres > 0:
                sr_room = (eres - epr) / epr * 100
            elif side == PositionSide.SHORT and esup > 0:
                sr_room = (epr - esup) / epr * 100

    return rlist


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
    trades = run_v2(df, sym, s)
    print('done %d trades (%.1fs)' % (len(trades), time.time() - t0), flush=True)
    total_pnl = sum(t.pnl_pct for t in trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    results.append((sym, len(trades), total_pnl, 100 * len(wins) / len(trades) if trades else 0))
    all_trades.extend(trades)
    gc.collect()
print('Total %.1fs for %d coins' % (time.time() - t_start, len(results)), flush=True)

# 保存CSV
import csv as _csv
with open(r'E:\BNFF\BNFRich\logs\all_trades_v2.csv', 'w', newline='', encoding='utf-8') as _f:
    _w = _csv.writer(_f)
    _w.writerow(['sym', 'entry_idx', 'exit_idx', 'side', 'entry_price', 'sl', 'leverage', 'pnl_pct',
                 'hold_bars_5m', 'exit_reason', 'signal_type', 'trend_type', 'max_pnl_pct', 'min_pnl_pct',
                 'entry_adx', 'entry_atr', 'entry_rsi', 'is_cluster_sup', 'is_cluster_res', 'sr_room_pct'])
    for _t in all_trades:
        _w.writerow([_t.symbol, _t.entry_idx_5m, _t.exit_idx_5m, _t.side,
                     _t.entry_price, _t.sl, _t.leverage, _t.pnl_pct,
                     _t.hold_bars_5m, _t.exit_reason, _t.signal_type, _t.trend_type,
                     _t.max_pnl_pct, _t.min_pnl_pct, _t.entry_adx, _t.entry_atr, _t.entry_rsi,
                     False, False, getattr(_t, 'sr_room_pct', 0.0)])
print('Saved %d trades to logs/all_trades_v2.csv' % len(all_trades), flush=True)

results.sort(key=lambda x: -x[2])
print('\n========== 全39币结果（按PnL降序）==========')
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

# EV分析
print('\n========== 期望值(EV)分析 ==========')
for sn in sorted(sigs):
    tl = sigs[sn]
    if len(tl) < 3:
        continue
    wins = [t for t in tl if t.pnl_pct > 0]
    losses = [t for t in tl if t.pnl_pct <= 0]
    wr = len(wins) / len(tl)
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
    ev = wr * avg_win - (1 - wr) * abs(avg_loss)
    print('%s: %d笔 胜率=%.0f%% avg_win=+%.1f%% avg_loss=%.1f%% EV=%.2f%%' % (
        sn, len(tl), wr * 100, avg_win, avg_loss, ev))
