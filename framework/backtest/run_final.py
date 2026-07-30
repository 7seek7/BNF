# -*- coding: utf-8 -*-
"""
15mTupo 全币种回测 + 复利模拟

运行方式:
  powershell> python framework/backtest/run_final.py

环境变量（全部可选，有默认值）:
  运行模式:
  MODE            = backtest(默认) | live      回测/实盘模式
  YZDIR           = 空(默认)      | 1          验证集(21币)
  SAVE_SAMPLE     = 空(默认)      | "BTC,ETH"  仅跑指定币种

  资金管理(推荐配置见docs):
  BASE_MARGIN_PCT = 0.20  每笔资金%(推荐0.20, 默认0.10)
  DD_T1           = 0.35  回撤减仓阈值(推荐0.35, 0=禁用)
  DD_T2           = 0.55  回撤停开阈值(推荐0.55, 0=禁用)
  MAX_CONCURRENT  = 10    最大并发持仓

  风控:
  BASE_LEV        = 15    基础杠杆
  MAX_LOSS        = 30    单笔最大亏损%
  SLIPPAGE        = 0.03  滑点%(双边)
  FUNDING         = 0.0001 资金费率(每5m)

  入口过滤(默认宽松):
  ENTRY_SC_MIN=4  MIN_ADX=0  MAX_ADX=0  MIN_VOLR=0
  MAX_ATR=0  MAX_ATR_LONG=0  RSI_LONG_MAX=100  RSI_SHORT_MIN=0
  MAX_ATR_RATIO=0  SIGNAL_BLACKLIST=""

  质量过滤(默认全关):
  MIN_BODY15=0  MIN_POS20=0  MIN_VOL5=0  MIN_RNG15=0

  仓位管控(默认全关):
  REBOUND_HALF=0  REBOUND_OPP=0  REBOUND_BE=0  LADDER_VS=0
  EXCLUDE_FALLBACK=0  BO_RETEST=1  BO_STRONG_BREAKOUT=0
  VOL_CONFIRM_THRESH=5.0  BE_MX_PCT=5.0  BE_SL_BUFFER_PCT=0
  BE_HARD_MX=0  BE_WEAK_MX=0  BE_WEAK_VOL=5.0  TREND_BE_MX=0

  速度退出:
  VEL_FLOOR=11  VEL_FLOOR_TR=7  VEL_TR_MX_CAP=20
  VEL_MIN_MX=8  VEL_WINDOW=3  VEL_FLOOR_MULTI=15

  信号杠杆:
  RB_MAX_LEV=15  TR_MAX_LEV=0  BO_MAX_LEV=0
  LEV_MULT_RB=1.0  LEV_MULT_TR=1.0  LEV_MULT_BO=1.0
  CLUSTER_LEV_MULT=1.0  FALLBACK_LEV_MULT=1.0

  追踪止盈:
  TRAIL_ACT=5.0  TRAIL_DIST=1.5
  TRAIL_TIGHT_ACT=0  TRAIL_TIGHT_DIST=0
  PER_SYM_TRAIL=""  SIDE_FILTER=""  MAX_SP=5.0

  评分模式:
  ADX_MODE=0  SRD_MODE=0
"""

import sys, os, gc, pandas as pd, numpy as np, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
_main_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
_strat_env = os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "15mTupo", ".env")
load_dotenv(_main_env, override=False)
load_dotenv(_strat_env, override=False)

import importlib
_tupo_core = importlib.import_module('strategies.15mTupo.private.tupo_core')
print(f'[VERIFY] _trend_lock: {hasattr(_tupo_core, "_trend_lock")}, LOCK_BREAK_TOLERANCE: {getattr(_tupo_core, "LOCK_BREAK_TOLERANCE", "MISSING")}, TREND_LOCK_BARS: {getattr(_tupo_core, "TREND_LOCK_BARS", "MISSING")}', flush=True)

# 计数器
_trend_overrides = 0
TradeResult = _tupo_core.TradeResult
WatchlistEntry = _tupo_core.WatchlistEntry
PositionSide = _tupo_core.PositionSide
TrendType = _tupo_core.TrendType
BacktestSettings = _tupo_core.BacktestSettings
aggregate_5m_to_15m_fast = _tupo_core.aggregate_5m_to_15m_fast
calculate_adx_fast = _tupo_core.calculate_adx_fast
calculate_rsi = _tupo_core.calculate_rsi
calculate_avg_volume_fast = _tupo_core.calculate_avg_volume_fast
analyze_trend_fast = _tupo_core.analyze_trend_fast
set_atr_ma40_buffer = _tupo_core.set_atr_ma40_buffer
entry_ok = _tupo_core.entry_ok
quality_ok = _tupo_core.quality_ok
_calc_score = _tupo_core._calc_score
calc_sl_lev = _tupo_core.calc_sl_lev
TupoEngine = _tupo_core.TupoEngine
V8ExitState = _tupo_core.V8ExitState
check_exit_v8 = _tupo_core.check_exit_v8

MODE = os.environ.get("MODE", "backtest").lower()
DR = r"E:\BNFF\BNFRich\data\historical"
if os.environ.get("YZDIR"):
    DR = r"E:\BNFF\BNFRich\data\historical\yanzheng"
s = BacktestSettings()

BASE_LEV_CONFIG = int(os.environ.get("BASE_LEV", "15"))
MAX_LOSS_CONFIG = int(os.environ.get("MAX_LOSS", "30"))
SLIPPAGE_PCT = float(os.environ.get("SLIPPAGE", "0.03"))
FUNDING_PER_BAR = float(os.environ.get("FUNDING", "0.0001"))

_REBOUND_HALF = bool(int(os.environ.get("REBOUND_HALF", "0")))
_REBOUND_OPP = os.environ.get("REBOUND_OPPOSITE_ENABLED", os.environ.get("REBOUND_OPP", "0")).lower() in ('1', 'true', 'yes')
_REBOUND_BE = bool(int(os.environ.get("REBOUND_BE", "0")))
_LADDER_VS = bool(int(os.environ.get("LADDER_VS", "0")))
_EXCLUDE_FALLBACK = bool(int(os.environ.get("EXCLUDE_FALLBACK", "0")))
_RB_EXCLUDE_FALLBACK = bool(int(os.environ.get("RB_EXCLUDE_FALLBACK", "0")))
_BO_RETEST = bool(int(os.environ.get("BO_RETEST", "1")))
_BO_STRONG_BREAKOUT = float(os.environ.get("BO_STRONG_BREAKOUT", "0"))
VOL_CONFIRM_THRESH = float(os.environ.get("VOL_CONFIRM_THRESH", "5.0"))
BE_MX_PCT = float(os.environ.get("BE_MX_PCT", "5.0"))
BE_SL_BUFFER_PCT = float(os.environ.get("BE_SL_BUFFER_PCT", "0"))
BE_HARD_MX = float(os.environ.get("BE_HARD_MX", "0"))
BE_WEAK_MX = float(os.environ.get("BE_WEAK_MX", "0"))
BE_WEAK_VOL = float(os.environ.get("BE_WEAK_VOL", "5.0"))
TREND_BE_MX = float(os.environ.get("TREND_BE_MX", "0"))
BE_UNCONDITIONAL_MX = float(os.environ.get("BE_UNCONDITIONAL_MX", "0"))

V8_EXIT = bool(int(os.environ.get("V8_EXIT", "0")))

HALF_EXIT_ENABLED = bool(int(os.environ.get("HALF_EXIT_ENABLED", "0")))
HALF_EXIT_TRIGGER_PCT = float(os.environ.get("HALF_EXIT_TRIGGER_PCT", "5.0"))
HALF_EXIT_REENTRY_PCT = float(os.environ.get("HALF_EXIT_REENTRY_PCT", "1.0"))

LIQ_THRESHOLD = float(os.environ.get("LIQ_THRESHOLD", "-90"))
LIQ_PENALTY = float(os.environ.get("LIQ_PENALTY", "0.5"))
LIQ_MM_RATE = float(os.environ.get("LIQ_MM_RATE", "0.5"))

# 半仓保本参数
HALF_EXIT_ENABLED = bool(int(os.environ.get("HALF_EXIT_ENABLED", "0")))
HALF_EXIT_TRIGGER_PCT = float(os.environ.get("HALF_EXIT_TRIGGER_PCT", "5.0"))
HALF_EXIT_REENTRY_PCT = float(os.environ.get("HALF_EXIT_REENTRY_PCT", "1.0"))


def run_final(df_5m, sym, s):
    # base_lev: 基础杠杆（10/15/20/30），由信号分数缩放
    # max_loss_pct: 单笔最大亏损占本金比例（避免极端情况）
    c5=df_5m['close'].values.astype(float);h5=df_5m['high'].values.astype(float)
    l5=df_5m['low'].values.astype(float);v5=df_5m['volume'].values.astype(float)
    o5=df_5m['open'].values.astype(float)
    df15=aggregate_5m_to_15m_fast(df_5m)
    if len(df15)<100: return []
    c15=df15['close'].values;h15=df15['high'].values;l15=df15['low'].values;o15=df15['open'].values
    adx,pdm,mdm,atr=calculate_adx_fast(h15,l15,c15,14)
    rsi=calculate_rsi(c15,14);ma20v=pd.Series(c15).rolling(20).mean().values
    # V8.2: EMA20/EMA50（用于趋势分类验证）
    ema20 = pd.Series(c15).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(c15).ewm(span=50, adjust=False).mean().values
    avgv=calculate_avg_volume_fast(df15['volume'].values,20)
    # V8.2 VSR计算: EWMA(volume, alpha=0.1)
    v15_arr = df15['volume'].values.astype(float)
    vsr_arr = np.ones_like(v15_arr)
    ewma_vol = np.copy(v15_arr)
    for vi in range(1, len(v15_arr)):
        ewma_vol[vi] = 0.1 * v15_arr[vi] + 0.9 * ewma_vol[vi - 1]
        if ewma_vol[vi] > 0:
            vsr_arr[vi] = v15_arr[vi] / ewma_vol[vi]
    atr_pct_full = np.where(c15 > 0, atr / c15 * 100, 0)
    set_atr_ma40_buffer(atr_pct_full)
    # body15[ci] = abs(c15[ci]-o15[ci])/(h15[ci]-l15[ci]+eps); pos20[ci] = (c15[ci]-minLow20)/(maxHigh20-minLow20+eps)
    rng15=h15-l15; rng15=np.where(rng15<1e-12,1e-12,rng15)
    body15_arr=np.abs(c15-o15)/rng15
    rng15pct_arr=rng15/np.where(c15>0,c15,1)
    h20=pd.Series(h15).rolling(20).max().values
    l20=pd.Series(l15).rolling(20).min().values
    pos20_arr=(c15-l20)/np.where(h20-l20<1e-12,1e-12,h20-l20)
    
    n_15=len(c15);n_1h=n_15//4
    o1h=np.zeros(n_1h);h1h=np.zeros(n_1h);l1h=np.zeros(n_1h);c1h=np.zeros(n_1h)
    for i in range(n_1h):
        s15=i*4;e15=min(n_15,(i+1)*4)
        if s15>=e15:break
        o1h[i]=o15[s15];h1h[i]=np.max(h15[s15:e15]);l1h[i]=np.min(l15[s15:e15]);c1h[i]=c15[e15-1]
    ma20_1h=pd.Series(c1h).rolling(20).mean().values
    def get_1h(ci15):
        hi=ci15//4
        if hi<14 or hi>=len(c1h): return 'neutral'
        if c1h[hi]>ma20_1h[hi]*1.005: return 'up'
        elif c1h[hi]<ma20_1h[hi]*0.995: return 'down'
        return 'neutral'

    trail_on=True;trail_act=float(os.environ.get('TRAIL_ACT', '5.0'));trail_dist=float(os.environ.get('TRAIL_DIST', '1.5'))  # TRAIL_ACT 激活阈值, TRAIL_DIST 回撤触发
    # 按币种覆写trail_dist（PER_SYM_TRAIL=SYM:VAL,SYM2:VAL2）
    for _pse in os.environ.get('PER_SYM_TRAIL', '').split(','):
        _pse = _pse.strip()
        if _pse and ':' in _pse:
            _ps, _pv = _pse.split(':', 1)
            if _ps in sym: trail_dist = float(_pv)
    trail_tight_act=float(os.environ.get('TRAIL_TIGHT_ACT', '0'))  # mx>=此值时收紧trail（0=不收紧）
    trail_tight_dist=float(os.environ.get('TRAIL_TIGHT_DIST', '0'))  # 收紧后的trail距离（0=不收紧）
    trail_act_tr=float(os.environ.get('TRAIL_ACT_TR', '0'))  # TR专用trail激活阈值（0=使用全局）
    trail_dist_tr=float(os.environ.get('TRAIL_DIST_TR', '0'))  # TR专用trail回撤触发（0=使用全局）
    # VELOCITY: 单根15m内PnL暴跌超过阈值即退出（捕获急剧反转，不影响缓跌）
    vel_floor=float(os.environ.get('VEL_FLOOR', '11'))  # 单根暴跌阈值（0=禁用）
    vel_min_mx=float(os.environ.get('VEL_MIN_MX', '8'))  # mx>=此值才检查
    vel_floor_tr=float(os.environ.get('VEL_FLOOR_TR', '0'))  # TR信号专用阈值（0=使用全局）
    vel_tr_mx_cap=float(os.environ.get('VEL_TR_MX_CAP', '20'))  # TR信号仅mx<此值才用VEL_FLOOR_TR
    # 多K线速度退出：检查窗口内累计PnL回撤（捕获慢速反转）
    vel_window=int(os.environ.get('VEL_WINDOW', '3'))  # 回看窗口（0=禁用）
    vel_floor_multi=float(os.environ.get('VEL_FLOOR_MULTI', '15'))  # 窗口内累计回撤阈值
    # 盈利回撤保护：当盈利从最高点回撤PROFIT_PROTECT_PCT%时退出（0=禁用）
    # 例如：PROFIT_PROTECT_PCT=50表示当盈利从最高点回撤50%时退出
    # 如果最大盈利是5%，当盈利回撤到2.5%时退出
    profit_protect_pct=float(os.environ.get('PROFIT_PROTECT_PCT', '0'))
    only_rebound=False;only_long=False

    n_4h=n_15//16
    o4h=np.zeros(n_4h);h4h=np.zeros(n_4h);l4h=np.zeros(n_4h);c4h=np.zeros(n_4h)
    for i in range(n_4h):
        s15=i*16;e15=min(n_15,(i+1)*16)
        if s15>=e15:break
        o4h[i]=o15[s15];h4h[i]=np.max(h15[s15:e15]);l4h[i]=np.min(l15[s15:e15]);c4h[i]=c15[e15-1]
    ma20_4h=pd.Series(c4h).rolling(20).mean().values
    tf4_arr=np.zeros(n_15,dtype=np.int8)
    for ci15 in range(n_15):
        hi=ci15//16
        if hi>=20 and hi<len(c4h):
            if c4h[hi]>ma20_4h[hi]*1.005: tf4_arr[ci15]=1
            elif c4h[hi]<ma20_4h[hi]*0.995: tf4_arr[ci15]=2
    
    rlist=[];pos=None;eidx=0;eidx15=0;epr=0.0;sl=0.0;lev=1
    tt=TrendType.UNKNOWN;st='';eadx=20;eatr=0.0;ersi=50.0
    tsp=0.0;tc=0;pr=1.0;ap=0.0;mx=0.0;mn=0.0;hrt=False
    ph=[];eres=0.0;esup=0.0;sr_room=0.0;o15s=0.0;h15s=0.0;l15s=0.0;c15s=0.0;v15s=0.0
    trail_high=0.0;trail_active=False;prev_pnl=0;cache_ci=-1;cache_ti=None;cache_info={}
    fl_info={};op_info={};fl_breach_count=0;op_breach_count=0
    cache_tl_slp=0.0;cache_tl_bv=0.0;cache_tl_bi=0;cache_tl_valid=False
    cache_search_start=0;cache_search_end=0
    dt_short_touches=0
    retest=None
    _debug_t={k:0.0 for k in ['prep','loop','manage','sig','trend']}
    t_data=0.0; n_loop=0
    _rb_half_debug=[]
    _debug_touch_dist = {}
    watchlist = {}  # dict[str, WatchlistEntry] 币种→监控条目
    # TR状态机（6.1节）
    tr_state = _tupo_core.TRState.IDLE
    tr_breakout_price = 0.0
    tr_breakout_idx = -100
    tr_p1 = None
    tr_foundation = None
    tr_operational = None
    tr_cooldown_end = -100
    tr_p1_frozen_bars = 0
    # V8.2状态机
    tupo_engine_long = _tupo_core.TupoEngine(sym + '_LONG', 'LONG')
    tupo_engine_short = _tupo_core.TupoEngine(sym + '_SHORT', 'SHORT')
    v8_exit_state = _tupo_core.V8ExitState()
    
    for i in range(60,len(c5)):
        t0=time.time()
        cp=c5[i];bi=i%3;r15=i//3
        if r15>=len(c15):continue
        ci=max(0,r15-1)
        if bi==0:
            o15s=o5[i];h15s=h5[i];l15s=l5[i];c15s=c5[i];v15s=v5[i]
        else:
            h15s=max(h15s,h5[i]);l15s=min(l15s,l5[i]);c15s=c5[i];v15s+=v5[i]
        c15v=c15[ci] if ci<len(c15) else c15[-1];vv=atr[ci]/c15v if atr[ci]>0 else 0.02
        t_data+=time.time()-t0;n_loop+=1
        
        if pos is not None:
            t0=time.time()
            ...
            pnl=0.0;sl_hit=False;hb=i-eidx;sc2=False;exit_p=epr
            # =====================================================================
            # ⚠️  双副本架构说明 (DUAL-COPY ARCHITECTURE)
            #
            # 本文件(run_final.py)的退出逻辑与 tupo_core.py check_exit() 是双副本!
            # 两份代码逻辑完全一致，通过同步测试保护:
            #   python framework/backtest/test_exit_sync.py
            #
            # 为什么不能合并?
            #   回测: 每根5m bar调一次, 39币×360k根=1400万次调用
            #   Python函数调用开销(~200ns)×1400万=3s额外开销, 加上dict lookup
            #   实际慢10x(350s→3500s+), 不可接受
            #   实盘: 每5分钟最多触发一次, 函数调用开销可忽略
            #
            # 所以: 回测用inline(快), 实盘用check_exit()(干净)
            #
            # 修改此处必须同步修改 strategies/15mTupo/private/tupo_core.py check_exit()
            # 必须同步: env vars (BE_MX_PCT, BE_SL_BUFFER_PCT, BE_HARD_MX,
            #   BE_WEAK_MX, BE_WEAK_VOL, TREND_BE_MX, VOL_CONFIRM_THRESH,
            #   LIQ_MM_RATE, LIQ_THRESHOLD, TRAIL_*, VEL_*, LADDER_*, REBOUND_*),
            #   退出级别顺序, 液化elif链结构, ladder volume scaling, tsp partial close
            # =====================================================================
            # 5m SL 检查（每根5m都用l5/h5检测，SL价成交）
            if (pos==PositionSide.LONG and l5[i]<=sl) or (pos==PositionSide.SHORT and h5[i]>=sl):
                sl_hit=True
                pnl=abs(epr-sl)/epr*100*lev if pos==PositionSide.LONG else abs(sl-epr)/epr*100*lev
                pnl=-pnl;pnl-=FUNDING_PER_BAR*lev;pnl=max(pnl,-100.0)
                pnl-=SLIPPAGE_PCT*2*lev;sc2,er=True,'触发止损';cr=1.0;exit_p=sl
            # 量比确认后才缩SL：爆量(>=3x)是真突破→缩SL保本；无量(<3x)是震荡反弹→保持宽SL继续跑
            if not sl_hit:
                if pos==PositionSide.LONG:
                    if mx>=BE_MX_PCT and sl<epr and peak_vol_ratio>=VOL_CONFIRM_THRESH: sl=epr*(1-BE_SL_BUFFER_PCT/100)
                    elif hb>=6 and max(mx, (h5[i]-epr)/epr*100*lev)>=BE_MX_PCT and sl<epr:
                        _5m_vr = v5[i]/max(avgv[ci]/3,0.001) if ci>=0 and ci<len(avgv) and avgv[ci]>0 else 0
                        if _5m_vr >= VOL_CONFIRM_THRESH: sl=epr*(1-BE_SL_BUFFER_PCT/100)
                elif pos==PositionSide.SHORT:
                    if mx>=BE_MX_PCT and sl>epr and peak_vol_ratio>=VOL_CONFIRM_THRESH: sl=epr*(1+BE_SL_BUFFER_PCT/100)
                    elif hb>=6 and max(mx, (epr-l5[i])/epr*100*lev)>=BE_MX_PCT and sl>epr:
                        _5m_vr = v5[i]/max(avgv[ci]/3,0.001) if ci>=0 and ci<len(avgv) and avgv[ci]>0 else 0
                        if _5m_vr >= VOL_CONFIRM_THRESH: sl=epr*(1+BE_SL_BUFFER_PCT/100)
            # 硬BE：mx达到BE_HARD_MX时无条件缩SL（不依赖量比），针对9-10%关键价位反转
            if not sl_hit and BE_HARD_MX>0 and mx>=BE_HARD_MX:
                if pos==PositionSide.LONG and sl<epr: sl=epr*(1-BE_SL_BUFFER_PCT/100)
                elif pos==PositionSide.SHORT and sl>epr: sl=epr*(1+BE_SL_BUFFER_PCT/100)
            # 弱量BE：mx达到BE_WEAK_MX且当前量比<BE_WEAK_VOL时缩SL（关键价位缩量=反转信号）
            if not sl_hit and BE_WEAK_MX>0 and mx>=BE_WEAK_MX:
                _cur_vr = v5[i]/max(avgv[ci]/3,0.001) if ci>=0 and ci<len(avgv) and avgv[ci]>0 else 0
                if _cur_vr < BE_WEAK_VOL:
                    if pos==PositionSide.LONG and sl<epr: sl=epr*(1-BE_SL_BUFFER_PCT/100)
                    elif pos==PositionSide.SHORT and sl>epr: sl=epr*(1+BE_SL_BUFFER_PCT/100)
            # 无条件保本：mx达到BE_UNCONDITIONAL_MX时无条件缩SL（不依赖量比），针对3-8%盈利但因量比不足未触发BE的交易
            if not sl_hit and BE_UNCONDITIONAL_MX>0 and mx>=BE_UNCONDITIONAL_MX:
                if pos==PositionSide.LONG and sl<epr: sl=epr*(1-BE_SL_BUFFER_PCT/100)
                elif pos==PositionSide.SHORT and sl>epr: sl=epr*(1+BE_SL_BUFFER_PCT/100)
            # 趋势BE：mx达到TREND_BE_MX且入场趋势为非CONSOLIDATION→缩SL（弱势趋势的反弹更易反转）
            if not sl_hit and TREND_BE_MX>0 and mx>=TREND_BE_MX:
                if tt not in (TrendType.CONSOLIDATION, TrendType.UNKNOWN):
                    if pos==PositionSide.LONG and sl<epr: sl=epr*(1-BE_SL_BUFFER_PCT/100)
                    elif pos==PositionSide.SHORT and sl>epr: sl=epr*(1+BE_SL_BUFFER_PCT/100)
            # 5m 爆仓检查 (币安真实公式): 5m bar 区间 [L5, H5] 触到 liq_price 即爆仓
            # liq_price = epr * (1 ± (1/lev - mm_rate/100))
            # SHORT: 涨到 liq_price 爆仓; LONG: 跌到 liq_price 爆仓
            if not sl_hit and ((pos==PositionSide.LONG and l5[i]<=epr*(1-1.0/lev+LIQ_MM_RATE/100)) or \
                 (pos==PositionSide.SHORT and h5[i]>=epr*(1+1.0/lev-LIQ_MM_RATE/100))):
                sl_hit=True
                # 5m bar 区间触到 liq 价 → 爆仓 (亏损 = 1 - mm_rate ≈ 99.5%)
                liq_loss_pct = (1.0/lev - LIQ_MM_RATE/100) * 100 * lev  # margin % 损失
                pnl=-liq_loss_pct;pnl-=FUNDING_PER_BAR*lev;pnl=max(pnl,-100.0)
                pnl-=SLIPPAGE_PCT*2*lev
                if pos==PositionSide.LONG:
                    exit_p=epr*(1-1.0/lev+LIQ_MM_RATE/100)
                else:
                    exit_p=epr*(1+1.0/lev-LIQ_MM_RATE/100)
                # 维持 爆仓 标识和罚金逻辑兼容
                pnl=max(pnl,LIQ_THRESHOLD);sc2,er=True,'爆仓';cr=1.0
            # 15m收盘管理（用c15成交+触发，H/L仅用于阶梯/移动峰值跟踪）
            if not sl_hit and bi==2:
                pnl_c=(c15s-epr)/epr*100*lev if pos==PositionSide.LONG else (epr-c15s)/epr*100*lev
                pnl=pnl_c;pnl-=FUNDING_PER_BAR*lev*3;pnl=max(pnl,-100.0)
                cur_vr = v15s / max(avgv[ci], 0.001) if ci>=0 and ci<len(avgv) and avgv[ci]>0 else 0
                if pnl_c > mx:
                    peak_vol_ratio = max(peak_vol_ratio, cur_vr)
                mx=max(mx,pnl_c);mn=min(mn,pnl_c);ph.append(pnl_c)
                if len(ph)>10:ph.pop(0)
                hb15=r15-eidx15;sc2=False;er='';cr=1.0;exit_p=c15s
                # ─── V8.2 出场管线（仅用于 BO_V8 信号） ───
                _use_v8_exit = V8_EXIT and st.startswith('BO_V8')
                if _use_v8_exit:
                    _should_close, _er, _cr, _new_sl, _new_mx, _new_mn, _pnl, _exit_p = check_exit_v8(
                        pos, epr, sl, lev, c15s, hb15, mx, mn, st, tt,
                        v8_exit_state, c15s,
                        avgv[ci] if ci>=0 and ci<len(avgv) else 0,
                        v15s, pr, fl_info, op_info,
                        atr[ci] if ci<len(atr) else 0,
                        bar_idx=ci,
                    )
                    if _should_close:
                        sc2 = True; er = _er; cr = _cr; pnl = _pnl; exit_p = _exit_p
                        mx = _new_mx; mn = _new_mn
                # ─── 原退出逻辑（非 BO_V8 信号使用） ───
                if not _use_v8_exit:
                    if not sc2 and op_info:
                        op_idx = op_info.get('idx', 0)
                        op_price = op_info.get('price', 0)
                        op_slope = op_info.get('slope', 0)
                        if op_idx > 0 and op_price > 0:
                            op_val = op_price + op_slope * (ci - op_idx)
                            if op_val > 0:
                                if (pos == PositionSide.LONG and c15s < op_val) or \
                                   (pos == PositionSide.SHORT and c15s > op_val):
                                    op_breach_count += 1
                                    if op_breach_count >= 2:
                                        cr = min(0.7/pr, 1.0) if pr > 0 else 0.7
                                        sl = epr
                                        sc2,er=True,'操作线跌破'
                                else:
                                    if op_breach_count > 0: op_breach_count = 0
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
                                    fl_breach_count += 1
                                    if fl_breach_count >= 2:
                                        # 放量>80%：100%平仓
                                        if cur_vr >= 0.8:
                                            cr = 1.0
                                            sc2,er=True,'基石线跌破(放量)'
                                        # 缩量<20%：平50%，止损下移0.5ATR
                                        elif cur_vr <= 0.2:
                                            cr = min(0.5/pr, 1.0) if pr > 0 else 0.5
                                            if pos == PositionSide.LONG and ci < len(atr):
                                                sl = fl_val - atr[ci] * 0.5
                                            elif pos == PositionSide.SHORT and ci < len(atr):
                                                sl = fl_val + atr[ci] * 0.5
                                            sc2,er=True,'基石线跌破(缩量)'
                                        else:
                                            cr = min(0.5/pr, 1.0) if pr > 0 else 0.5
                                            sc2,er=True,'基石线跌破'
                                else:
                                    if fl_breach_count > 0: fl_breach_count = 0
                    # 半仓保本：盈利达到触发阈值时，卖出50%在入场价（保本），SL不变
                    if HALF_EXIT_ENABLED and not half_exited and mx>=HALF_EXIT_TRIGGER_PCT:
                        half_exited=True;cr=0.5;sc2=True;er='半仓保本';pnl_c=0;pnl=0
                        half_exit_price=c15s
                    # 半仓再入场：价格涨回卖出价+阈值时，买回50%
                    if HALF_EXIT_ENABLED and half_exited and not half_reentered and half_exit_price>0:
                        _reentry_price = half_exit_price * (1 + HALF_EXIT_REENTRY_PCT/100)
                        if (pos==PositionSide.LONG and c15s >= _reentry_price) or \
                           (pos==PositionSide.SHORT and c15s <= _reentry_price):
                            half_reentered=True;cr=-1.0;sc2=True;er='半仓再入场';pnl_c=0;pnl=0
                    # 速度检测：单根15m PnL 急剧反转退出
                    _vf = vel_floor
                    if vel_floor_tr>0 and st.startswith('TR_') and mx<vel_tr_mx_cap:
                        _vf = vel_floor_tr
                    if _vf>0 and mx>=vel_min_mx and prev_pnl!=0 and prev_pnl-pnl_c>=_vf:
                        sc2,er=True,'速度退出'
                    # 多K线速度检测：窗口内累计PnL回撤（捕获慢速反转）
                    if not sc2 and vel_window>0 and vel_floor_multi>0 and mx>=vel_min_mx and len(ph)>=vel_window:
                        if max(ph[-vel_window:]) - ph[-1] >= vel_floor_multi:
                            sc2,er=True,'速度退出'
                    prev_pnl=pnl_c
                    if trail_on and trail_active:
                        _td = trail_tight_dist if (trail_tight_dist>0 and mx>=trail_tight_act) else trail_dist
                        if st.startswith('TR_') and trail_dist_tr>0: _td = trail_dist_tr
                        if pos==PositionSide.LONG:
                            trail_high=max(trail_high,c15s)
                            if c15s<=trail_high*(1-_td/100): sc2,er=True,'移动止盈'
                        else:
                            trail_high=min(trail_high,c15s)
                            if c15s>=trail_high*(1+_td/100): sc2,er=True,'移动止盈'
                    if trail_on and not trail_active:
                        _ta = trail_act
                        if st.startswith('TR_') and trail_act_tr>0: _ta = trail_act_tr
                        if pos==PositionSide.LONG and pnl_c>=_ta: trail_high=c15s;trail_active=True
                        elif pos==PositionSide.SHORT and pnl_c>=_ta: trail_high=c15s;trail_active=True
                    # 盈利回撤保护：当盈利从最高点回撤PROFIT_PROTECT_PCT%时退出
                    if not sc2 and profit_protect_pct>0 and mx>0 and pnl_c>0:
                        drawback_pct = (mx - pnl_c) / mx * 100
                        if drawback_pct >= profit_protect_pct:
                            sc2,er=True,'盈利回撤保护'
                    if not sc2 and tc<5:
                        if pnl_c>tsp: tsp=pnl_c
                        ir=st.startswith('RB_') or 'REBOUND' in st
                        is_tr=(st=='TR' or st.startswith('TR_'))
                        if ir or (tt==TrendType.CONSOLIDATION and sr_room<2.0):
                            _lp=s.ladder_peak_rebound
                        elif is_tr and s.ladder_peak_tr>0:
                            _lp=s.ladder_peak_tr
                        else:
                            _lp=s.ladder_peak_uptrend
                        if tsp >= _lp:
                            if ir or (tt==TrendType.CONSOLIDATION and sr_room<2.0):
                                ddt=[s.ladder_dd_t1_rebound,s.ladder_dd_t2_rebound,s.ladder_dd_t3_rebound,s.ladder_dd_t4_rebound,s.ladder_dd_t5_rebound]
                            elif is_tr and s.ladder_peak_tr>0:
                                ddt=[s.ladder_dd_t1_tr,s.ladder_dd_t2_tr,s.ladder_dd_t3_tr,s.ladder_dd_t4_tr,s.ladder_dd_t5_tr]
                            else:
                                ddt=[s.ladder_dd_t1_uptrend,s.ladder_dd_t2_uptrend,s.ladder_dd_t3_uptrend,s.ladder_dd_t4_uptrend,s.ladder_dd_t5_uptrend]
                            dd_trigger = tsp-pnl_c >= ddt[tc]
                            if _LADDER_VS and dd_trigger and ci>=0 and avgv[ci]>0:
                                cur_v = v15s*3.0/(bi+1)
                                if cur_v >= avgv[ci] * 0.7: dd_trigger = False
                            if dd_trigger:
                                cr=max(0,min([s.ladder_close_t1/100,s.ladder_close_t2/100,s.ladder_close_t3/100,s.ladder_close_t4/100,s.ladder_close_t5/100][tc]/pr,1.0))
                                sc2,er=True,'阶梯止盈';tc+=1;hrt=True
                    if not sc2 and _REBOUND_BE and (st.startswith('RB_') or 'REBOUND' in st) and mx>=10.0 and pnl_c<=3.0:
                        sc2,er=True,'反弹保本'
                    if not sl_hit and _REBOUND_OPP and sl!=epr:
                        if pos==PositionSide.LONG and eres>0 and h5[i] >= eres*(1-0.005) and c5[i] < c5[i-1]:
                            cr = min(0.5/pr, 1.0) if pr>0 else 0.5
                            sl = epr; sc2,er=True,'阻力止盈'
                        elif pos==PositionSide.SHORT and esup>0 and l5[i] <= esup*(1+0.005) and c5[i] > c5[i-1]:
                            cr = min(0.5/pr, 1.0) if pr>0 else 0.5
                            sl = epr; sc2,er=True,'支撑止盈'
                    if not sc2 and hb15>=s.entry_stop_bars//3 and mx<=0: sc2,er=True,'入场止损'
                    if not sc2 and hb15>=16 and mx<3.0:
                        cr = min(0.5/pr, 1.0) if pr>0 else 0.5
                        sc2,er=True,'时间衰减退出'
                if sc2: pnl-=SLIPPAGE_PCT*2*lev
            if sl_hit or sc2:
                ap+=pnl*cr*pr;pr*=(1-cr)
                if pr<0.01 or cr>=1.0:
                    if hrt: er='阶梯止盈'
                    rlist.append(TradeResult(sym,eidx,epr,i,exit_p,pos.value,lev,ap-2*lev*s.trading_fee_rate*100,hb,er,tt.value,st,mx,mn,0,eadx,eatr,ersi,v5[i] if i<len(v5) else 0,0,rsi[ci] if ci<len(rsi) else 50,bool(sig_isc_sup),bool(sig_isc_res),sr_room,esup,eres,cache_tl_slp,cache_tl_bi,cache_tl_bv,cache_search_start,cache_search_end))
                    # 再入场：盈利出场时加入观察列表（已废弃，改用半仓保本）
                    pass
                    pos=None;pr=1.0;ap=0.0;mx=mn=0.0;hrt=False;tsp=0.0;tc=0;ph.clear();trail_active=False;trail_high=0.0;prev_pnl=0;peak_vol_ratio=0.0;fl_breach_count=0;op_breach_count=0
                    if tr_state == _tupo_core.TRState.HOLDING:
                        tr_state = _tupo_core.TRState.COOLDOWN; tr_cooldown_end = ci + 8
                else:
                    sc2=False;cr=1.0  # 部分平仓后复位sc2/cr，防止泄漏到后续K线重复执行
                    tsp=pnl
                continue
            continue
        if pos is not None: continue
        
        # 信号初始化（再入场机制已废弃，改用半仓保本）
        sig = None
        
        if ci!=cache_ci:
            cache_ti,cache_info=analyze_trend_fast(c15,h15,l15,ci,adx,pdm,mdm,adx_threshold=s.adx_threshold,price_change_threshold=s.trend_price_change,atr=atr,return_sr_extra=True,ema20=ema20,ema50=ema50)
            cache_ci=ci
            cache_tl_slp=cache_info.get('trend_line_slope',0.0) or 0.0
            cache_tl_bv=cache_info.get('trend_line_base_val',0.0) or 0.0
            cache_tl_bi=cache_info.get('trend_line_base_idx',0) or 0
            cache_tl_valid=cache_tl_slp!=0 and cache_tl_bv!=0 and cache_tl_bi>=0
            cache_search_start=cache_info.get('search_start',0) or 0
            cache_search_end=cache_info.get('search_end',0) or 0
            dt_short_touches=0
            if cache_tl_valid and ci>=20 and cache_ti==TrendType.DOWNTREND:
                lb=20
                for j in range(max(0,ci-lb), ci):
                    tv=cache_tl_bv+cache_tl_slp*(j-cache_tl_bi)
                    if tv<=0: continue
                    if c15[j]>=tv and (c15[j]-tv)/tv*100<0.3: dt_short_touches+=1
        ti,info=cache_ti,cache_info
        if ti in (TrendType.UNKNOWN, TrendType.HIGH_VOLATILITY): continue
        mtf_dir=get_1h(ci)
        tf4d=tf4_arr[ci]
        res,sup=info.get('resistance'),info.get('support')
        cons_trend=info.get('cons_trend_dir','neutral')

        # ─── V8.2 状态机评估（规格书5章） ───
        trend_v8 = ti.value  # 'UPTREND'|'DOWNTREND'|'CONSOLIDATION'|'TRIANGLE'
        if trend_v8 == 'CONSOLIDATION':
            trend_v8 = 'RANGEBOUND'
        v8_sig_long = tupo_engine_long.evaluate_bar(ci, h15, l15, c15, atr, vsr_arr, trend_v8)
        v8_sig_short = tupo_engine_short.evaluate_bar(ci, h15, l15, c15, atr, vsr_arr, trend_v8)
        if v8_sig_long is not None and sig is None:
            adv_v = adx[ci] if ci < len(adx) else 20
            volr_v = v15s / max(avgv[ci], 0.001) if 0 <= ci < len(avgv) else 1.0
            sc = _calc_score('BO_LONG', adv_v, rsi[ci] if ci < len(rsi) else 50, atr[ci] if ci < len(atr) else 0, 0.5, volr_v, 4)
            sn = f'BO_V8_S{sc:.0f}'
            atr_pct = (atr[ci] / c15[ci] * 100) if ci < len(atr) and ci < len(c15) and c15[ci] > 0 else 0
            if entry_ok(sc, adv_v, volr_v, sn, atr_pct, rsi[ci] if ci < len(rsi) else 50, 'LONG', ci) and quality_ok(ci, body15_arr, pos20_arr, 'LONG', v5, rng15pct_arr):
                sl_v8, lev_v8 = calc_sl_lev(PositionSide.LONG, v8_sig_long['stop_loss'], cp, sc, 'BO_V8', vv, info.get('sup_is_cluster', False) or info.get('res_is_cluster', False))
                if sl_v8 is not None:
                    sig = (PositionSide.LONG, sl_v8, sn, lev_v8, info.get('sup_is_cluster', False), info.get('res_is_cluster', False))
        if v8_sig_short is not None and sig is None:
            adv_v = adx[ci] if ci < len(adx) else 20
            volr_v = v15s / max(avgv[ci], 0.001) if 0 <= ci < len(avgv) else 1.0
            sc = _calc_score('BO_SHORT', adv_v, rsi[ci] if ci < len(rsi) else 50, atr[ci] if ci < len(atr) else 0, 0.5, volr_v, 4)
            sn = f'BO_V8_S{sc:.0f}'
            atr_pct = (atr[ci] / c15[ci] * 100) if ci < len(atr) and ci < len(c15) and c15[ci] > 0 else 0
            if entry_ok(sc, adv_v, volr_v, sn, atr_pct, rsi[ci] if ci < len(rsi) else 50, 'SHORT', ci) and quality_ok(ci, body15_arr, pos20_arr, 'SHORT', v5, rng15pct_arr):
                sl_v8, lev_v8 = calc_sl_lev(PositionSide.SHORT, v8_sig_short['stop_loss'], cp, sc, 'BO_V8', vv, info.get('sup_is_cluster', False) or info.get('res_is_cluster', False))
                if sl_v8 is not None:
                    sig = (PositionSide.SHORT, sl_v8, sn, lev_v8, info.get('sup_is_cluster', False), info.get('res_is_cluster', False))

        # ─── TR状态机更新（规格书6.1节） ───
        is_tr_trend = ti in (TrendType.UPTREND, TrendType.DOWNTREND)
        tr_side = 'LONG' if ti == TrendType.UPTREND else 'SHORT' if ti == TrendType.DOWNTREND else None
        if tr_state == _tupo_core.TRState.IDLE:
            if is_tr_trend and tr_side:
                tr_state = _tupo_core.TRState.SEARCHING
                tr_breakout_price = c15s
                tr_breakout_idx = ci
                tr_p1 = None; tr_foundation = None; tr_operational = None; tr_p1_frozen_bars = 0
        if tr_state == _tupo_core.TRState.SEARCHING and is_tr_trend and tr_side:
            # Section 10: 如果P1_LOOKBACK根bar内未找到P1，创建虚拟基石线
            if ci - tr_breakout_idx >= 8 and tr_p1 is None:
                atrv = atr[ci] if ci < len(atr) else 0
                if tr_side == 'LONG':
                    virt_price = tr_breakout_price - 0.2 * atrv * 5 / max(tr_breakout_price, 0.001) * tr_breakout_price
                else:
                    virt_price = tr_breakout_price + 0.2 * atrv * 5 / max(tr_breakout_price, 0.001) * tr_breakout_price
                virt_price = max(virt_price, 0.001)
                tr_p1 = {'price': virt_price, 'idx': tr_breakout_idx, 'virtual': True, 'confirmed': True}
                tr_foundation = {'idx': tr_breakout_idx, 'price': virt_price, 'slope': 0.0, 'end_idx': tr_breakout_idx}
                ol = _tupo_core.find_dynamic_trendline(c15, h15, l15, atr, tr_foundation, ci, tr_side)
                if ol is not None: tr_operational = ol
                tr_state = _tupo_core.TRState.ENTRY_WAIT
            else:
                p1_res = _tupo_core.detect_p1(c15, h15, l15, atr, tr_breakout_price, tr_breakout_idx, ci, tr_side)
                if p1_res is not None:
                    tr_p1 = p1_res
                    tr_state = _tupo_core.TRState.FROZEN
                    tr_p1_frozen_bars = 0
        if tr_state == _tupo_core.TRState.FROZEN:
            tr_p1_frozen_bars += 1
            if tr_p1_frozen_bars >= 3 and tr_p1 and tr_side:
                fl_price = tr_p1['price']; fl_idx = tr_p1['idx']
                tr_foundation = {'idx': fl_idx, 'price': fl_price, 'slope': 0.0, 'end_idx': fl_idx}
                ol = _tupo_core.find_dynamic_trendline(c15, h15, l15, atr, tr_foundation, ci, tr_side)
                if ol is not None:
                    tr_operational = ol
                tr_state = _tupo_core.TRState.CONFIRMING
        if tr_state == _tupo_core.TRState.CONFIRMING:
            tr_state = _tupo_core.TRState.ENTRY_WAIT
        if not is_tr_trend and tr_state not in (_tupo_core.TRState.IDLE, _tupo_core.TRState.COOLDOWN):
            if tr_state != _tupo_core.TRState.HOLDING:
                tr_state = _tupo_core.TRState.IDLE; tr_p1 = None; tr_foundation = None; tr_operational = None
        if tr_state == _tupo_core.TRState.COOLDOWN and ci >= tr_cooldown_end:
            tr_state = _tupo_core.TRState.IDLE; tr_p1 = None; tr_foundation = None; tr_operational = None
        # 操作线更新（规格书6.5节）：每8根bar检查一次
        if tr_operational and tr_side and ci % 8 == 0:
            tr_operational = _tupo_core.update_operational(tr_operational, c15, h15, l15, atr, ci, tr_side)

        # 突破跟踪
        if ti==TrendType.CONSOLIDATION and res and sup:
            if c15s>res and (c15s-res)/res*100>s.breakout_threshold and cons_trend!='down' and mtf_dir in ('neutral','up') and tf4d==1:
                if retest is None or retest['level']!=res or retest['dir']!='long':
                    retest={'dir':'long','level':res,'ci':ci,'i':i,'max_dist':(c15s-res)/res*100,'state':'breakout'}
                else: retest['max_dist']=max(retest['max_dist'],(c15s-res)/res*100)
            elif c15s<sup and (sup-c15s)/sup*100>s.breakout_threshold and cons_trend!='up' and mtf_dir in ('neutral','down') and tf4d==2:
                if retest is None or retest['level']!=sup or retest['dir']!='short':
                    retest={'dir':'short','level':sup,'ci':ci,'i':i,'max_dist':(sup-c15s)/sup*100,'state':'breakout'}
                else: retest['max_dist']=max(retest['max_dist'],(sup-c15s)/sup*100)
        
        if retest and retest['state']=='breakout' and ci>retest['ci']:
            rp=retest
            if rp['dir']=='long':
                d=(c15s-rp['level'])/rp['level']*100;rp['max_dist']=max(rp['max_dist'],d)
                if rp['max_dist']>=0.3 and d<=1.0: rp['state']='retest'
            else:
                d=(rp['level']-c15s)/rp['level']*100;rp['max_dist']=max(rp['max_dist'],d)
                if rp['max_dist']>=0.3 and d<=1.0: rp['state']='retest'
        if retest and ci-retest['ci']>40: retest=None
        
        if ti==TrendType.CONSOLIDATION and res and sup and not only_rebound:
            if c15s>res and (c15s-res)/res*100>s.breakout_threshold and cons_trend!='down' and mtf_dir in ('neutral','up') and tf4d==1:
                can_enter = False
                if _BO_RETEST == 0:
                    can_enter = True  # 突破即入, 不需回踩
                elif retest and retest['dir']=='long' and retest['level']==res and retest['state']=='retest':
                    can_enter = True
                elif _BO_STRONG_BREAKOUT > 0 and retest and retest['dir']=='long' and retest['level']==res and retest['max_dist'] >= _BO_STRONG_BREAKOUT:
                    can_enter = True  # 强势突破立即入
                if can_enter:
                    adv_v=adx[ci] if ci<len(adx) else 20
                    volr_v=v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 1.0
                    sc=_calc_score('BO_LONG',adv_v,rsi[ci] if ci<len(rsi) else 50,atr[ci] if ci<len(atr) else 0,(c15s-res)/res*100,volr_v,len(info.get('swing_highs',[])) if info.get('swing_highs') else 2)
                    sn=f'BO_LONG_S{sc:.0f}'
                    atr_pct = (atr[ci]/c15[ci]*100) if ci<len(atr) and ci<len(c15) and c15[ci]>0 else 0
                    if entry_ok(sc, adv_v, volr_v, sn, atr_pct, rsi[ci] if ci<len(rsi) else 50, 'LONG', ci) and quality_ok(ci, body15_arr, pos20_arr, 'LONG', v5, rng15pct_arr):
                        sl,lev=calc_sl_lev(PositionSide.LONG,sup,cp,sc,'BO_LONG',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                        if sl is None: continue
                        sig=(PositionSide.LONG,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False));retest=None
        if sig is None and ti==TrendType.CONSOLIDATION and res and sup and not only_rebound:
            if c15s<sup and (sup-c15s)/sup*100>s.breakout_threshold and cons_trend!='up' and mtf_dir in ('neutral','down') and tf4d==2:
                can_enter = False
                if _BO_RETEST == 0:
                    can_enter = True
                elif retest and retest['dir']=='short' and retest['level']==sup and retest['state']=='retest':
                    can_enter = True
                elif _BO_STRONG_BREAKOUT > 0 and retest and retest['dir']=='short' and retest['level']==sup and retest['max_dist'] >= _BO_STRONG_BREAKOUT:
                    can_enter = True
                if can_enter:
                    adv_v=adx[ci] if ci<len(adx) else 20
                    volr_v=v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 1.0
                    sc=_calc_score('BO_SHORT',adv_v,rsi[ci] if ci<len(rsi) else 50,atr[ci] if ci<len(atr) else 0,(sup-c15s)/sup*100,volr_v,len(info.get('swing_lows',[])) if info.get('swing_lows') else 2)
                    sn=f'BO_SHORT_S{sc:.0f}'
                    atr_pct = (atr[ci]/c15[ci]*100) if ci<len(atr) and ci<len(c15) and c15[ci]>0 else 0
                    if entry_ok(sc, adv_v, volr_v, sn, atr_pct, rsi[ci] if ci<len(rsi) else 50, 'SHORT', ci) and quality_ok(ci, body15_arr, pos20_arr, 'SHORT', v5, rng15pct_arr):
                        sl,lev=calc_sl_lev(PositionSide.SHORT,res,cp,sc,'BO_SHORT',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                        if sl is None: continue
                        sig=(PositionSide.SHORT,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False));retest=None
        if sig is None and ti==TrendType.CONSOLIDATION and res and sup:
            if c15s>sup*0.995 and c15s>o15s and c15s>ma20v[ci]*0.998 and mtf_dir in ('neutral','up') and tf4d==1 and l15s<=sup*1.003 and c15s<res*0.998:
                br=(c15s-o15s)/max(h15s-l15s,0.001);cp2=(c15s-l15s)/max(h15s-l15s,0.001)
                if i>=5 and c15s>sum(c5[i-5:i])/5 and br>s.body_ratio and cp2>s.close_position:
                    adv_v=adx[ci] if ci<len(adx) else 20
                    volr_v=v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 1.0
                    sc=_calc_score('RB_LONG',adv_v,rsi[ci] if ci<len(rsi) else 50,atr[ci] if ci<len(atr) else 0,(c15s-sup)/sup*100,volr_v,2)
                    sn=f'RB_LONG_S{sc:.0f}'
                    atr_pct = (atr[ci]/c15[ci]*100) if ci<len(atr) and ci<len(c15) and c15[ci]>0 else 0
                    if entry_ok(sc, adv_v, volr_v, sn, atr_pct, rsi[ci] if ci<len(rsi) else 50, 'LONG', ci) and quality_ok(ci, body15_arr, pos20_arr, 'LONG', v5, rng15pct_arr):
                        sl,lev=calc_sl_lev(PositionSide.LONG,sup,cp,sc,'RB_LONG',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                        if sl is None: continue
                        sig=(PositionSide.LONG,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False))
            if h15s>=res*0.997 and h15s<res*1.005 and c15s<o15s and mtf_dir in ('neutral','down') and tf4d==2:
                br2=(o15s-c15s)/max(h15s-l15s,0.001);cp22=(h15s-c15s)/max(h15s-l15s,0.001)
                if i>=5 and c15s<sum(c5[i-5:i])/5 and br2>s.body_ratio and cp22>s.close_position:
                    adv_v=adx[ci] if ci<len(adx) else 20
                    volr_v=v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 1.0
                    sc=_calc_score('RB_SHORT',adv_v,rsi[ci] if ci<len(rsi) else 50,atr[ci] if ci<len(atr) else 0,(res-c15s)/res*100,volr_v,2)
                    sn=f'RB_SHORT_S{sc:.0f}'
                    atr_pct = (atr[ci]/c15[ci]*100) if ci<len(atr) and ci<len(c15) and c15[ci]>0 else 0
                    if entry_ok(sc, adv_v, volr_v, sn, atr_pct, rsi[ci] if ci<len(rsi) else 50, 'SHORT', ci) and quality_ok(ci, body15_arr, pos20_arr, 'SHORT', v5, rng15pct_arr):
                        sl,lev=calc_sl_lev(PositionSide.SHORT,res,cp,sc,'RB_SHORT',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                        if sl is None: continue
                        sig=(PositionSide.SHORT,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False))
        if sig is None and ti==TrendType.UPTREND and not only_rebound:
            foundation = info.get('foundation_line')
            if foundation is None: pass
            elif mtf_dir not in ('neutral','up') or tf4d!=1: pass
            elif i<20: pass
            elif tr_state == _tupo_core.TRState.COOLDOWN: pass  # 冷却期禁止入场（6.1节）
            else:
                _tc = info.get('touch_count', 0)
                _debug_touch_dist[_tc] = _debug_touch_dist.get(_tc, 0) + 1
                if _tc < 8:  # 触碰<8次（原始阈值，经验证有效）
                    p1_idx = info.get('p1_idx', foundation.get('idx', 0))
                    p1_price = info.get('p1_price', foundation.get('price', 0))
                    h1_price = info.get('h1_price', 0)
                    atrv = atr[ci] if ci<len(atr) else 0
                    ent_ci = ci + 1
                    mode_a = False; mode_b = False
                    # SIG_MODE_A（规格书5章）：P1附近回踩 + 缩量 + 止跌形态
                    bars_since_p1 = ent_ci - p1_idx
                    if bars_since_p1 >= 0 and bars_since_p1 <= 10:
                        zone_low = p1_price + 0.10 * atrv  # Low ≤ P1 + 0.10×ATR
                        zone_high = p1_price - 0.15 * atrv  # Close ≥ P1 - 0.15×ATR
                        if l15s <= zone_low and c15s >= zone_high:
                            vol_ratio = v15s / max(avgv[ci], 0.001) if 0<=ci<len(avgv) else 99
                            if vol_ratio < 0.60:  # 缩量至60%以下
                                body = abs(c15s - o15s)
                                lower_shadow = min(o15s, c15s) - l15s
                                total_range = h15s - l15s
                                is_hammer = c15s > o15s and lower_shadow > body * 2
                                is_long_lower_shadow = total_range > 0 and lower_shadow >= 0.5 * total_range
                                if is_hammer or is_long_lower_shadow:
                                    mode_a = True
                    # SIG_MODE_B（规格书5章）：突破H1放量确认
                    if not mode_a and h1_price > 0 and c15s > h1_price:
                        close_above_h1 = c15s > h1_price + 0.10 * atrv
                        vol_ratio_b = v15s / max(avgv[ci], 0.001) if 0<=ci<len(avgv) else 1.0
                        vsr_ok = vol_ratio_b >= 1.5
                        body_ok = abs(c15s - o15s) >= 0.80 * atrv
                        if close_above_h1 and vsr_ok and body_ok:
                            mode_b = True
                    tr_qualified = mode_a or mode_b
                    if not tr_qualified and _tc < 8 and c15s > o15s and c15s < ma20v[ci] * 1.003:
                        # 回退：原始body/candle条件
                        br15 = (c15s-o15s)/max(h15s-l15s,0.001)
                        cp15 = (c15s-l15s)/max(h15s-l15s,0.001)
                        tr_qualified = br15 > s.body_ratio and cp15 > s.close_position
                    if tr_qualified:
                        adv=adx[ci] if ci<len(adx) else 20; rsiv=rsi[ci] if ci<len(rsi) else 50
                        volr=v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 1.0
                        srd=(c15s-ma20v[ci])/ma20v[ci]*100
                        sc=_calc_score('TR_LONG',adv,rsiv,atrv,srd,volr,2)
                        sn=f'TR_LONG_S{sc:.0f}'
                        atr_pct = (atrv/c15[ci]*100) if c15[ci]>0 else 0
                        if entry_ok(sc, adv, volr, sn, atr_pct, rsiv, 'LONG', ci) and quality_ok(ci, body15_arr, pos20_arr, 'LONG', v5, rng15pct_arr):
                            sl,lev=calc_sl_lev(PositionSide.LONG,None,cp,sc,'TR_LONG',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                            if sl is not None:
                                sig=(PositionSide.LONG,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False))
        if sig is None and ti==TrendType.DOWNTREND and not only_rebound:
            foundation = info.get('foundation_line')
            if foundation is None: pass
            elif mtf_dir not in ('neutral','down') or tf4d!=2: pass
            elif i<20: pass
            elif tr_state == _tupo_core.TRState.COOLDOWN: pass  # 冷却期禁止入场（6.1节）
            else:
                _tc = info.get('touch_count', 0)
                if _tc < 8:
                    p1_idx = info.get('p1_idx', foundation.get('idx', 0))
                    p1_price = info.get('p1_price', foundation.get('price', 0))
                    l1_price = info.get('h1_price', 0)
                    atrv = atr[ci] if ci<len(atr) else 0
                    ent_ci = ci + 1
                    mode_a = False; mode_b = False
                    bars_since_p1 = ent_ci - p1_idx
                    if bars_since_p1 >= 0 and bars_since_p1 <= 10:
                        bounce_dist = abs(c15s - p1_price) / max(p1_price, 0.001)
                        a_threshold = (atrv / max(c15s, 0.001)) * 0.5
                        if bounce_dist <= a_threshold:
                            mode_a = True
                    if not mode_a and l1_price > 0 and l15s < l1_price:
                        breakdown_amp = (l1_price - l15s) / max(l1_price, 0.001)
                        if breakdown_amp <= atrv / max(c15s, 0.001) * 0.3:
                            volr_b = v15s / max(avgv[ci], 0.001) if 0<=ci<len(avgv) else 1.0
                            if volr_b > 1.5:
                                if c15s < l1_price - 0.1 * (atr[ci] if ci<len(atr) else 0):
                                    mode_b = True
                    tr_qualified = mode_a or mode_b
                    if tr_qualified:
                        body = abs(c15s - o15s)
                        upper_shadow = h15s - max(o15s, c15s)
                        is_shooting = c15s < o15s and upper_shadow > body * 2
                        is_doji = body < (h15s - l15s) * 0.1
                        tr_qualified = is_shooting or is_doji
                    if not tr_qualified and _tc < 8 and c15s < o15s and c15s > ma20v[ci] * 0.997:
                        br15 = (o15s-c15s)/max(h15s-l15s,0.001)
                        cp15 = (h15s-c15s)/max(h15s-l15s,0.001)
                        tr_qualified = br15 > s.body_ratio and cp15 > s.close_position
                    if tr_qualified:
                        adv=adx[ci] if ci<len(adx) else 20; rsiv=rsi[ci] if ci<len(rsi) else 50
                        volr=v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 1.0
                        srd=(ma20v[ci]-c15s)/ma20v[ci]*100
                        sc=_calc_score('TR_SHORT',adv,rsiv,atrv,srd,volr,2)
                        sn=f'TR_SHORT_S{sc:.0f}'
                        atr_pct = (atrv/c15[ci]*100) if c15[ci]>0 else 0
                        if entry_ok(sc, adv, volr, sn, atr_pct, rsiv, 'SHORT', ci) and quality_ok(ci, body15_arr, pos20_arr, 'SHORT', v5, rng15pct_arr):
                            sl,lev=calc_sl_lev(PositionSide.SHORT,None,cp,sc,'TR_SHORT',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                            if sl is not None:
                                sig=(PositionSide.SHORT,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False))
        if sig is None and ti==TrendType.TRIANGLE:
            tdet=info.get('triangle')
            volr_tri = v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 0
            if tdet and c15s>tdet['up_at_ci'] and (c15s-tdet['up_at_ci'])/tdet['up_at_ci']*100>s.triangle_breakout and c15s>o15s and (c15s-o15s)/o15s*100>s.body_pct and volr_tri >= 1.5 and tf4d==1:
                adv_v=adx[ci] if ci<len(adx) else 20
                sc=_calc_score('TRIANGLE_LONG',adv_v,rsi[ci] if ci<len(rsi) else 50,atr[ci] if ci<len(atr) else 0,(c15s-tdet['up_at_ci'])/tdet['up_at_ci']*100,volr_tri,4)
                sn=f'TRIANGLE_LONG_S{sc:.0f}'
                atr_pct = (atr[ci]/c15[ci]*100) if ci<len(atr) and ci<len(c15) and c15[ci]>0 else 0
                if entry_ok(sc, adv_v, volr_tri, sn, atr_pct, rsi[ci] if ci<len(rsi) else 50, 'LONG', ci) and quality_ok(ci, body15_arr, pos20_arr, 'LONG', v5, rng15pct_arr):
                    sl,lev=calc_sl_lev(PositionSide.LONG,tdet['lo_at_ci'],cp,sc,'TRIANGLE_LONG',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                    if sl is None: continue
                    sig=(PositionSide.LONG,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False))
            elif tdet and c15s<tdet['lo_at_ci'] and (tdet['lo_at_ci']-c15s)/tdet['lo_at_ci']*100>s.triangle_breakout and c15s<o15s and (o15s-c15s)/o15s*100>s.body_pct and volr_tri >= 1.5 and tf4d==2:
                adv_v=adx[ci] if ci<len(adx) else 20
                volr_v=v15s/max(avgv[ci],0.001) if ci>=0 and ci<len(avgv) else 1.0
                sc=_calc_score('TRIANGLE_SHORT',adv_v,rsi[ci] if ci<len(rsi) else 50,atr[ci] if ci<len(atr) else 0,(tdet['lo_at_ci']-c15s)/tdet['lo_at_ci']*100,volr_v,4)
                sn=f'TRIANGLE_SHORT_S{sc:.0f}'
                atr_pct = (atr[ci]/c15[ci]*100) if ci<len(atr) and ci<len(c15) and c15[ci]>0 else 0
                if entry_ok(sc, adv_v, volr_v, sn, atr_pct, rsi[ci] if ci<len(rsi) else 50, 'SHORT', ci) and quality_ok(ci, body15_arr, pos20_arr, 'SHORT', v5, rng15pct_arr):
                    sl,lev=calc_sl_lev(PositionSide.SHORT,tdet['up_at_ci'],cp,sc,'TRIANGLE_SHORT',vv,info.get('sup_is_cluster',False) or info.get('res_is_cluster',False))
                    if sl is None: continue
                    sig=(PositionSide.SHORT,sl,sn,lev,info.get('sup_is_cluster',False),info.get('res_is_cluster',False))
        if sig is not None:
            # 检查是否需要聚类确认
            _is_rb = sig[2].startswith('RB_') or 'REBOUND' in sig[2]
            if _EXCLUDE_FALLBACK and not (sig[4] or sig[5]):
                continue  # 全局EXCLUDE_FALLBACK：非聚类信号跳过
            if _RB_EXCLUDE_FALLBACK and _is_rb and not (sig[4] or sig[5]):
                continue  # RB专用EXCLUDE_FALLBACK：非聚类RB信号跳过
            pos,sl,st,lev,sig_isc_sup,sig_isc_res=sig;eidx=i;eidx15=r15;epr=cp;mx=0.0;mn=0.0;peak_vol_ratio=0.0
            v8_exit_state = _tupo_core.V8ExitState()
            if st.startswith('TR_') and tr_state == _tupo_core.TRState.ENTRY_WAIT:
                tr_state = _tupo_core.TRState.HOLDING
            eadx=adx[ci] if ci<len(adx) else 0;eatr=atr[ci] if ci<len(atr) else 0;ersi=rsi[ci] if ci<len(rsi) else 50
            eres=res or 0;esup=sup or 0;tt=ti
            sr_room = ((eres-epr)/epr*100) if pos==PositionSide.LONG and eres>0 else ((epr-esup)/epr*100) if pos==PositionSide.SHORT and esup>0 else 0
            fl_info = info.get('foundation_line') or {}
            op_info = info.get('operating_line') or {}
            fl_breach_count = 0
            op_breach_count = 0
            trail_active=False;trail_high=0;tsp=0;tc=0;ph.clear();hrt=False;pr=1.0;ap=0.0;prev_pnl=0
            half_exited=False;half_reentered=False;half_exit_price=0.0
            # REBOUND 半仓开仓（精细仓位管控1）— v6 干净版信号名是 RB_*
            if _REBOUND_HALF and (st.startswith('RB_') or 'REBOUND' in st) and lev >= 6:
                _orig_lev = lev
                lev = max(5, int(lev/2))
                _rb_half_debug.append((sym, st, _orig_lev, lev))
    # ─── 调试：触碰次数分布 ───
    if _debug_touch_dist:
        print('\n========== TR触碰次数分布（调试） ==========')
        for tc in sorted(_debug_touch_dist.keys()):
            print(f'  touch_count={tc}: {_debug_touch_dist[tc]}次')
    
    return rlist


if __name__ == '__main__':
    if os.environ.get('YZDIR'):
        fs=[f for f in os.listdir(DR) if '_5m_2025-04-01_2026-05-31.csv' in f]
    else:
        fs=[f for f in os.listdir(DR) if f.endswith('_5m_2025-03-01_2026-05-01.csv')]
    # Filter to specific coins if SAVE_SAMPLE env var is set
    _save_sample = os.environ.get('SAVE_SAMPLE', '')
    if _save_sample:
        _wanted = set(_save_sample.split(','))
        fs = [f for f in fs if any(w in f for w in _wanted)]
        print(f'SAMPLE MODE: {len(fs)} files match {_wanted}', flush=True)
    fs_all=sorted(fs)
    print('Found %d 5m files total' % len(fs_all), flush=True)
    import time
    t_start=time.time()
    all_trades=[];results=[]
    for f in fs_all:
        _tupo_core._trend_lock = {'trend': None, 'expiry_idx': 0, 'pivot_price': 0.0, 'pivot_idx': 0}
        sym=f.split('_5m')[0]
        print('Loading %s...' % sym, end=' ', flush=True);t0=time.time()
        df=pd.read_csv(DR+'/'+f).dropna(subset=['close']).reset_index(drop=True)
        print('running...', end=' ', flush=True)
        trades=run_final(df, sym, s)
        print('done %d trades (%.1fs)' % (len(trades), time.time()-t0), flush=True)
        total_pnl=sum(t.pnl_pct for t in trades)
        wins=[t for t in trades if t.pnl_pct>0]
        results.append((sym,len(trades),total_pnl,100*len(wins)/len(trades) if trades else 0))
        all_trades.extend(trades)
        gc.collect()
    print('Total %.1fs for %d coins' % (time.time()-t_start, len(results)), flush=True)
    
    # Save all trades to CSV for analysis
    import csv as _csv
    _csv_tag = f'_vol{VOL_CONFIRM_THRESH:.0f}_mx{BE_MX_PCT:.0f}_buf{BE_SL_BUFFER_PCT:.0f}'
    with open(fr'E:\BNFF\BNFRich\logs\all_trades{_csv_tag}.csv', 'w', newline='', encoding='utf-8') as _f:
        _w = _csv.writer(_f)
        _w.writerow(['sym','entry_idx','exit_idx','side','entry_price','sl','leverage','pnl_pct',
                     'hold_bars_5m','exit_reason','signal_type','trend_type','max_pnl_pct','min_pnl_pct',
                     'entry_adx','entry_atr','entry_rsi','is_cluster_sup','is_cluster_res','sr_room_pct',
                     'entry_support','entry_resistance','trend_line_slope','trend_line_base_idx','trend_line_base_val',
                     'search_start','search_end'])
        for _t in all_trades:
            _w.writerow([_t.symbol,
                         _t.entry_idx_5m, _t.exit_idx_5m, _t.side,
                         _t.entry_price, 0.0, _t.leverage, _t.pnl_pct,
                         _t.hold_bars_5m, _t.exit_reason, _t.signal_type, _t.trend_type,
                         _t.max_pnl_pct, _t.min_pnl_pct, _t.entry_adx, _t.entry_atr, _t.entry_rsi,
                         getattr(_t, 'is_cluster_sup', False), getattr(_t, 'is_cluster_res', False),
                         getattr(_t, 'sr_room_pct', 0.0),
                         getattr(_t, 'entry_support', 0.0), getattr(_t, 'entry_resistance', 0.0),
                         getattr(_t, 'trend_line_slope', 0.0), getattr(_t, 'trend_line_base_idx', 0),
                         getattr(_t, 'trend_line_base_val', 0.0),
                         getattr(_t, 'search_start', 0), getattr(_t, 'search_end', 0)])
    print('Saved %d trades to logs/all_trades.csv' % len(all_trades), flush=True)
    
    if MODE != 'live':
        results.sort(key=lambda x: -x[2])
        print('\n========== 全39币结果（按PnL降序）==========')
        for sym,n,pnl,wr in results:
            print('%s: %d笔 PnL=%+.0f%% 胜率=%.0f%%' % (sym,n,pnl,wr))
    
        total_pnl=sum(t.pnl_pct for t in all_trades);wins=[t for t in all_trades if t.pnl_pct>0]
        pos_coins=sum(1 for _,_,p,_ in results if p>0)
        neg_coins=sum(1 for _,_,p,_ in results if p<0)
        print('\n========== 合计 ==========')
        print('Total: %d trades, PnL=%.0f%%, Win=%d/%d=%.0f%%' % (len(all_trades), total_pnl, len(wins), len(all_trades), 100*len(wins)/len(all_trades) if all_trades else 0))
        print('正收益币种: %d/%d (%.0f%%)' % (pos_coins, pos_coins+neg_coins, 100*pos_coins/(pos_coins+neg_coins)))
    
        sigs={}
        for t in all_trades:
            sigs.setdefault(t.signal_type[:15],[]).append(t)
        print('\n========== 按信号类型 ==========')
        for sn in sorted(sigs):
            tl=sigs[sn];w=[t for t in tl if t.pnl_pct>0]
            if len(tl)>=3:
                print('%s: %d笔 PnL=%.0f%% 胜=%d/%d=%.0f%%' % (sn, len(tl), sum(t.pnl_pct for t in tl), len(w), len(tl), 100*len(w)/len(tl)))
    
        print('\n========== 盈亏特征分析 ==========')
        for sig_name in sorted(sigs):
            tl=sigs[sig_name]
            wins=[t for t in tl if t.pnl_pct>0]
            los=[t for t in tl if t.pnl_pct<=0]
            if len(tl)<5: continue
            print('%s (%d笔): 胜=%d(%.0f%%) 亏=%d' % (sig_name, len(tl), len(wins), 100*len(wins)/len(tl), len(los)))
            if wins:
                w_lev=sum(t.leverage for t in wins)/len(wins)
                w_adx=sum(t.entry_adx for t in wins)/len(wins)
                w_rsi=sum(t.entry_rsi for t in wins)/len(wins)
                w_atr=sum(t.entry_atr for t in wins)/len(wins)
                w_hb=sum(t.hold_bars_5m for t in wins)/len(wins)
                print('  胜: 杠杆%.1fx ADX%.0f RSI%.0f ATR%.3f 持仓%.0f根5m 均收益%.1f%%' % (w_lev,w_adx,w_rsi,w_atr,w_hb,sum(t.pnl_pct for t in wins)/len(wins)))
            if los:
                l_lev=sum(t.leverage for t in los)/len(los)
                l_adx=sum(t.entry_adx for t in los)/len(los)
                l_rsi=sum(t.entry_rsi for t in los)/len(los)
                l_atr=sum(t.entry_atr for t in los)/len(los)
                l_hb=sum(t.hold_bars_5m for t in los)/len(los)
                print('  亏: 杠杆%.1fx ADX%.0f RSI%.0f ATR%.3f 持仓%.0f根5m 均收益%.1f%%' % (l_lev,l_adx,l_rsi,l_atr,l_hb,sum(t.pnl_pct for t in los)/len(los)))
            diffs=[]
            if wins and los:
                if abs(w_adx-l_adx)>1: diffs.append('ADX差%.0f' % (w_adx-l_adx))
                if abs(w_rsi-l_rsi)>2: diffs.append('RSI差%.0f' % (w_rsi-l_rsi))
                if abs(w_atr-l_atr)/max(l_atr,0.001)>0.1: diffs.append('ATR差%.1f%%' % ((w_atr-l_atr)/max(l_atr,0.001)*100))
                if abs(w_hb-l_hb)>3: diffs.append('持仓差%.0f' % (w_hb-l_hb))
            if diffs: print('  差异: %s' % ', '.join(diffs))
    
        # 复利模拟（回撤统计）
        print('\n========== 复利模拟（最多10个币种同时持仓）==========')
        all_sorted=sorted(all_trades, key=lambda r: r.entry_idx_5m)
        n_coins = len(set(t.symbol for t in all_trades)) if all_trades else len(results)
        total_capital=n_coins*1000.0; global_capital=total_capital
        open_positions=[]
        n_liquidations=0
        liq_penalty_cost=0.0
        max_capital=total_capital
        min_capital=total_capital
        running_peak=total_capital
        max_drawdown=0.0
        DD_T1 = float(os.environ.get('DD_T1', '0.20'))
        DD_T2 = float(os.environ.get('DD_T2', '0.30'))
        _dd_active=False; _dd_total_bars=0; _dd_periods=[]; _step=0
        _daily_pnl = {}  # Section 9: 日亏损限制
        _daily_loss_limit = float(os.environ.get('DAILY_LOSS_PCT', '5.0'))
        _tr_positions = []  # Section 9: TR持仓计数
        _max_tr_concurrent = int(os.environ.get('MAX_TR_CONCURRENT', '5'))
        for r in all_sorted:
            still_open=[]
            for pos in open_positions:
                if r.entry_idx_5m<pos['exit_time']:
                    still_open.append(pos)
                else:
                    pnl_capped = max(pos['pnl_pct'], -100.0)
                    was_liquidated = pos.get('exit_reason') == '爆仓'
                    if was_liquidated:
                        n_liquidations += 1
                        liq_penalty_cost += pos['margin'] * LIQ_PENALTY / 100
                    global_capital += pos['margin'] * (1 + pnl_capped / 100)
            open_positions=still_open
            max_capital=max(max_capital, global_capital)
            min_capital=min(min_capital, global_capital)
            running_peak=max(running_peak, global_capital)
            current_dd=(running_peak-global_capital)/running_peak if running_peak>0 else 0
            max_drawdown=max(max_drawdown, current_dd)
            _step+=1
            if current_dd>1e-10:
                _dd_total_bars+=1
                if not _dd_active:
                    _dd_active=True; _dd_trough=global_capital; _dd_start_step=_step
                else:
                    _dd_trough=min(_dd_trough,global_capital)
            elif _dd_active:
                _depth=(running_peak-_dd_trough)/running_peak
                _bars=_step-_dd_start_step
                _dd_periods.append({'depth':_depth,'bars':_bars})
                _dd_active=False
            _dd_factor = 1.0
            if DD_T2 > 0 and current_dd >= DD_T2: continue
            if DD_T1 > 0 and DD_T2 > DD_T1 and current_dd > DD_T1:
                _ratio = (current_dd - DD_T1) / (DD_T2 - DD_T1)
                _dd_factor = 1.0 - _ratio * 0.75
                _dd_factor = max(_dd_factor, 0.25)
            _boost_sigs = set(os.environ.get('BOOST_SIGS', '').split(',')) - {''}
            _boost_margin_mult = float(os.environ.get('BOOST_MARGIN_MULT', '2.0'))
            _base_margin_pct = float(os.environ.get('BASE_MARGIN_PCT', '0.10'))
            _max_concurrent = int(os.environ.get('MAX_CONCURRENT', '10'))
            is_boosted = any(r.signal_type == b for b in _boost_sigs) and _boost_margin_mult > 1
            cur_slots = len(open_positions)
            new_slots = _boost_margin_mult if is_boosted else 1.0
            if (cur_slots + new_slots) <= _max_concurrent and global_capital>=1000.0:
                margin = global_capital * _base_margin_pct * (_boost_margin_mult if is_boosted else 1.0) * _dd_factor
                global_capital -= margin
                open_positions.append({'entry_time':r.entry_idx_5m,'exit_time':r.exit_idx_5m,'pnl_pct':r.pnl_pct,'exit_reason':r.exit_reason,'margin':margin,'is_boosted':is_boosted,'signal':r.signal_type,'day':r.entry_idx_5m//(24*60)})
                # Section 9: 日亏损限制
                _day_key = r.entry_idx_5m//(24*60)
                if _day_key not in _daily_pnl: _daily_pnl[_day_key] = 0.0
                if r.pnl_pct < 0:
                    _daily_pnl[_day_key] += abs(r.pnl_pct) * margin / 100
                if _daily_loss_limit > 0 and _day_key in _daily_pnl and _daily_pnl[_day_key] > total_capital * _daily_loss_limit / 100:
                    continue  # 超过日亏损上限，跳过本日后续信号
                # Section 9: TR并发限制
                if r.signal_type and r.signal_type.startswith('TR_'):
                    _tr_positions.append(r)
                _active_tr = [p for p in _tr_positions if p.exit_idx_5m > r.entry_idx_5m]
                if len(_active_tr) > _max_tr_concurrent:
                    continue  # TR持仓超过并发限制
        for pos in open_positions:
            pnl_capped = max(pos['pnl_pct'], -100.0)
            was_liquidated = pos.get('exit_reason') == '爆仓'
            if was_liquidated:
                n_liquidations += 1
                liq_penalty_cost += pos['margin'] * LIQ_PENALTY / 100
            global_capital += pos['margin'] * (1 + pnl_capped / 100)
        global_capital -= liq_penalty_cost
        total_ret=(global_capital-total_capital)/total_capital*100
        if _dd_active:
            _depth=(running_peak-_dd_trough)/running_peak
            _bars=_step-_dd_start_step
            _dd_periods.append({'depth':_depth,'bars':_bars})
        print('初始资金: %.0f USDT (39币x1000U)' % total_capital)
        print('最终资金: %.2f USDT' % global_capital)
        print('爆仓笔数: %d (罚金: %.0f USDT)' % (n_liquidations, liq_penalty_cost))
        print('最低资金: %.0f USDT' % min_capital)
        print('最大回撤: %.1f%%' % (max_drawdown*100))
        print('回撤次数: %d' % len(_dd_periods))
        if _dd_periods:
            _depths=[p['depth'] for p in _dd_periods]
            print('平均回撤: %.1f%%' % (sum(_depths)/len(_depths)*100))
            print('中位回撤: %.1f%%' % (sorted(_depths)[len(_depths)//2]*100))
            _bars_list=[p['bars'] for p in _dd_periods]
            print('平均回撤时长: %.0f根5m | 最长: %.0f根5m' % (sum(_bars_list)/len(_bars_list), max(_bars_list)))
            print('回撤时间占比: %.1f%%' % (_dd_total_bars/max(_step,1)*100))
            _buckets=[(0.05,0.10,'5-10'),(0.10,0.20,'10-20'),(0.20,0.30,'20-30'),(0.30,0.40,'30-40'),(0.40,0.50,'40-50'),(0.50,1.0,'50%+')]
            _bktxt='  '.join('%s%%:%d' % (lbl,sum(1 for d in _depths if lo<=d<hi)) for lo,hi,lbl in _buckets)
            print('回撤分布: %s' % _bktxt)
        print('总收益率: %.2f%%' % total_ret)
        avg_lev = sum(t.leverage for t in all_trades) / len(all_trades) if all_trades else 0
        print('平均杠杆: %.1fx' % avg_lev)
        print('平均每笔收益: %.1f%%' % (sum(t.pnl_pct for t in all_trades)/len(all_trades) if all_trades else 0))
        reasons = {}
        for t in all_trades:
            reasons.setdefault(t.exit_reason, []).append(t)
        if reasons:
            print('按退出原因统计:')
            for r, ts in sorted(reasons.items(), key=lambda x: -len(x[1])):
                avg_pnl=sum(t.pnl_pct for t in ts)/len(ts)
                avg_l=sum(t.leverage for t in ts)/len(ts)
                w=sum(1 for t in ts if t.pnl_pct>0)
                print('  %s: %.1f%%/笔, 杠杆%.1fx, 胜率%d/%d=%d%%, %d笔' % (r, avg_pnl, avg_l, w, len(ts), 100*w//len(ts), len(ts)))
    
        # ─── 生成权益曲线并发送Telegram ───
        if all_trades and os.environ.get('SEND_TELEGRAM', ''):
            try:
                from framework.shared.chart_utils import plot_equity_with_trades
                # 等权复利
                cum = 0.0
                _trades_for_chart = []
                for t in all_sorted:
                    cum += t.pnl_pct
                    _trades_for_chart.append({'pnl_pct': cum})
                title = f'15mTupo {len(all_trades)}笔 | PnL={total_pnl:.0f}% | Win={100*len(wins)//len(all_trades)}% | DD={max_drawdown*100:.1f}%'
                chart_path = plot_equity_with_trades(_trades_for_chart, title=title, out_name=f'run_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                if chart_path:
                    from alert.telegram_bot import TelegramBot
                    bot = TelegramBot(bot_type='alert')
                    caption = f'<b>15mTupo 回测完成</b>\n{len(all_trades)}笔 | PnL={total_pnl:+.0f}% | 胜率={100*len(wins)//len(all_trades)}% | 最大回撤={max_drawdown*100:.1f}% | 结束资金={global_capital:.0f}U'
                    bot.send_photo(chart_path, caption=caption)
                    bot.close()
                    print(f'  权益曲线已发送至Telegram: {chart_path}')
            except Exception as e:
                print(f'  Telegram图表发送失败: {e}')
