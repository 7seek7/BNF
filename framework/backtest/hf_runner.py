"""
HF Strategy Official Backtest (1m-based)
=========================================
设计原则：回测是独立实现的完整脚本，不依赖 strategies/hf/strategy.py。
- 入场信号：回测与实盘逻辑等价但代码独立（各自实现 3m 价格变化+成交量检测），
  避免回测与实盘耦合导致的维护问题。
- 出场逻辑：通过 framework/shared/exit_logic.py 共享，回测和实盘完全一致。
- 资金模型：回测用 calc_pool_model() 模拟资金分配，实盘用真实仓位管理。
- 参数来源：与实盘同源 strategies/hf/.env，通过 Config 类读取。

Usage:
  python -m framework.backtest.hf_runner
  python -m framework.backtest.hf_runner --direction BOTH --days 500 --pullback 2
"""

import glob, os, sys
import random
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))

import logging
# 日志级别由运行环境控制，不在模块级禁用


# ═══════════════════════════════════════════════════════════════════════
# 加载环境变量（公共/私有分离，与hf_backtest.py相同）
# ═══════════════════════════════════════════════════════════════════════

def _load_env_file(env_path, override=False):
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if override or k not in os.environ:
                        os.environ[k] = v

root_dir = Path(__file__).resolve().parent.parent.parent
_load_env_file(root_dir / '.env')
_load_env_file(root_dir / 'strategies' / 'hf' / '.env')

# 在 env 加载后 import（避免 strategies/hf/__init__ 提前触发 settings.py 覆盖 env 变量）
from strategies.hf.exit_logic import should_exit as shared_should_exit
from strategies.hf._params import load_shared_params


# ═══════════════════════════════════════════════════════════════════════
# 配置（从 .env 读取，共享参数与 HFSettings 共用 _params.py 定义）
# ═══════════════════════════════════════════════════════════════════════

class Config:
    """回测配置，从 .env 读取。—— 共享参数通过 _params.py 与 HFSettings 统一。
    设计意图：回测是独立脚本，不依赖 strategy.py；两套类通过 _params.py 确保默认值一致。"""

    # 回测独有参数（不需要与 HFSettings 共享）
    _data_dirs_env = os.getenv('HF_DATA_DIRS', '')
    if _data_dirs_env:
        DATA_DIRS = [d.strip() for d in _data_dirs_env.split(',') if d.strip()]
    else:
        DATA_DIRS = [
            BASE / 'data' / 'historical' / 'yanzheng',
            BASE / 'data' / 'historical'
        ]
    _exclude_env = os.getenv('HF_EXCLUDE', '')
    if _exclude_env:
        EXCLUDE = {s.strip() for s in _exclude_env.split(',') if s.strip()}
    else:
        EXCLUDE = {'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
                   'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT'}
    VOL_AVG_BARS = 10
    POSITION_PCT = float(os.getenv('HF_POSITION_PCT', '4'))
    DIRECTION = os.getenv('HF_DIRECTION', 'LONG')
    COMPOUND = os.getenv('HF_COMPOUND', 'true').lower() == 'true'

    # 共享参数（与 HFSettings 共用 _params.py 定义，在类体外赋值）

# 注入共享参数到 Config
_shared = load_shared_params()
for _k, _v in _shared.items():
    setattr(Config, _k, _v)
Config.TREND_FILTER = Config.TREND_FILTER_ENABLED
Config.PCT_THRESHOLD = Config.PRICE_CHANGE_THRESHOLD
Config.VOL_SURGE = Config.VOLUME_SURGE_THRESHOLD
del _shared, _k, _v


# ── Helpers ─────────────────────────────────────────────────────────
def get_symbols(data_dirs=None, exclude=None):
    if data_dirs is None:
        data_dirs = Config.DATA_DIRS
    if exclude is None:
        exclude = Config.EXCLUDE
    symbols = set()
    for data_dir in data_dirs:
        files = glob.glob(f'{data_dir}/*_1m_*.csv')
        for f in files:
            s = os.path.basename(f).split('_1m_')[0]
            if s not in exclude:
                symbols.add(s)
    return sorted(symbols)


from collections import OrderedDict
_1m_cache = OrderedDict()
_CACHE_MAX = 50

def load_1m(symbol, data_dirs=None, days=0):
    if data_dirs is None:
        data_dirs = Config.DATA_DIRS
    cache_key = (symbol, days)
    if cache_key in _1m_cache:
        cached = _1m_cache[cache_key]
        return cached.tail(days * 1440).copy() if days > 0 else cached.copy()
    files = []
    for data_dir in data_dirs:
        files.extend(sorted(glob.glob(f'{data_dir}/{symbol}_1m_*')))
    if not files:
        return None
    need_rows = days * 1440 if days > 0 else 0
    usecols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    dfs = []
    loaded = 0
    for f in sorted(files, key=lambda x: os.path.basename(x)):
        n = _count_lines(f) - 1
        if need_rows > 0 and loaded + n < need_rows:
            dfs.append(pd.read_csv(f, usecols=usecols))
            loaded += n
        elif need_rows > 0:
            remaining = need_rows - loaded
            if remaining > 0:
                dfs.append(pd.read_csv(f, usecols=usecols, nrows=remaining))
            break
        else:
            dfs.append(pd.read_csv(f, usecols=usecols))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    df = df.dropna(subset=['close']).reset_index(drop=True)
    if len(_1m_cache) >= _CACHE_MAX:
        _1m_cache.popitem(last=False)  # FIFO淘汰最旧条目
    _1m_cache[cache_key] = df
    return df.tail(days * 1440).copy() if days > 0 else df.copy()


def _count_lines(path):
    with open(path, 'rb') as f:
        return f.read().count(b'\n')


def build_3m_from_1m(df1m):
    """Manually aggregate every 3 consecutive 1m bars into one 3m bar.
    
    This ensures 3m close == last 1m close in the group (no future function).
    Timestamp = last 1m bar's timestamp in each group.
    """
    n = len(df1m) // 3 * 3
    if n == 0:
        return pd.DataFrame()
    df = df1m.iloc[:n]
    closes = df['close'].values.reshape(-1, 3)
    highs = df['high'].values.reshape(-1, 3)
    lows = df['low'].values.reshape(-1, 3)
    opens = df['open'].values.reshape(-1, 3)
    volumes = df['volume'].values.reshape(-1, 3)
    time_col = pd.to_datetime(df['open_time'].values).values.reshape(-1, 3)

    result = pd.DataFrame({
        'open': opens[:, 0],
        'high': highs.max(axis=1),
        'low': lows.min(axis=1),
        'close': closes[:, -1],
        'volume': volumes.sum(axis=1),
    }, index=pd.DatetimeIndex(time_col[:, -1], name='datetime'))
    return result


def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return 0.0
    trs = []
    for j in range(-period, 0):
        h, l, cp = highs[j], lows[j], closes[j - 1]
        trs.append(max(h - l, abs(h - cp), abs(l - cp)))
    return sum(trs) / len(trs) if trs else 0.0


def _simulate_reversion_exit(entry_idx, entry_price, closes, highs, lows,
                             side, leverage, taker_fee, slippage,
                             ref_mean, max_hold, rev_stop_pct=2.0):
    """均值回归出场：价格回到 ref_mean(基准均值)即止盈；若继续沿逆势拉伸超过
    rev_stop_pct% 则止损。返回 (net_pnl, gross_pnl, hold_bars, reason)"""
    is_long = (side == 'LONG')
    n = len(closes)
    end = min(entry_idx + max_hold, n - 1)
    stop_mul = (1.0 - rev_stop_pct / 100.0) if is_long else (1.0 + rev_stop_pct / 100.0)
    reason = 'Timeout'
    exit_idx = end
    for j in range(entry_idx + 1, end + 1):
        if is_long:
            if highs[j] >= ref_mean:                 # 价格回升至均值 -> 回归止盈
                exit_idx, reason = j, 'RevTP'
                break
            if lows[j] <= entry_price * stop_mul:    # 继续下挫 -> 止损
                exit_idx, reason = j, 'RevSL'
                break
        else:
            if lows[j] <= ref_mean:                  # 价格回落至均值 -> 回归止盈
                exit_idx, reason = j, 'RevTP'
                break
            if highs[j] >= entry_price * stop_mul:   # 继续上冲 -> 止损
                exit_idx, reason = j, 'RevSL'
                break
    exit_price = closes[exit_idx]
    eff_entry = entry_price * (1 + slippage) if is_long else entry_price * (1 - slippage)
    eff_exit = exit_price * (1 - slippage) if is_long else exit_price * (1 + slippage)
    pnl_mult = leverage * 100
    inv_ep = 1.0 / eff_entry
    gross_pnl = (eff_exit - eff_entry) * inv_ep * pnl_mult if is_long else (eff_entry - eff_exit) * inv_ep * pnl_mult
    net_pnl = gross_pnl - 2 * taker_fee * leverage * 100
    hold_bars = exit_idx - entry_idx
    return net_pnl, gross_pnl, hold_bars, reason


def _simulate_trend_exit(entry_idx, entry_price, closes, highs, lows,
                         side, leverage, taker_fee, slippage,
                         ref_mean, max_hold, rev_stop_pct=2.0, rev_tp_pct=2.0):
    """趋势跟随出场：顺偏离方向进场后，价格继续跑 rev_tp_pct% 止盈；
    若回到基准均值(ref_mean,趋势失败)或逆势 rev_stop_pct% 则止损。"""
    is_long = (side == 'LONG')
    n = len(closes)
    end = min(entry_idx + max_hold, n - 1)
    tp_mul = (1.0 + rev_tp_pct / 100.0) if is_long else (1.0 - rev_tp_pct / 100.0)
    stop_mul = (1.0 - rev_stop_pct / 100.0) if is_long else (1.0 + rev_stop_pct / 100.0)
    reason = 'Timeout'
    exit_idx = end
    for j in range(entry_idx + 1, end + 1):
        if is_long:
            if highs[j] >= entry_price * tp_mul:                 # 顺势继续 -> 止盈
                exit_idx, reason = j, 'TrendTP'
                break
            if lows[j] <= ref_mean or lows[j] <= entry_price * stop_mul:   # 回均值/逆势 -> 止损
                exit_idx, reason = j, 'TrendSL'
                break
        else:
            if lows[j] <= entry_price * tp_mul:                  # 顺势继续 -> 止盈
                exit_idx, reason = j, 'TrendTP'
                break
            if highs[j] >= ref_mean or highs[j] >= entry_price * stop_mul:  # 回均值/逆势 -> 止损
                exit_idx, reason = j, 'TrendSL'
                break
    exit_price = closes[exit_idx]
    eff_entry = entry_price * (1 + slippage) if is_long else entry_price * (1 - slippage)
    eff_exit = exit_price * (1 - slippage) if is_long else exit_price * (1 + slippage)
    pnl_mult = leverage * 100
    inv_ep = 1.0 / eff_entry
    gross_pnl = (eff_exit - eff_entry) * inv_ep * pnl_mult if is_long else (eff_entry - eff_exit) * inv_ep * pnl_mult
    net_pnl = gross_pnl - 2 * taker_fee * leverage * 100
    hold_bars = exit_idx - entry_idx
    return net_pnl, gross_pnl, hold_bars, reason


# ── Exit simulation (pure 1m, with liquidation + costs) ──────────────
def simulate_exit(entry_idx, entry_price,
                  closes, highs, lows,
                  side='LONG', leverage=15,
                  taker_fee=0.0005, slippage=0.0002,
                  quick_stop_bars=0, quick_stop_pnl=-1.5,
                  qs_tier1_bars=0, qs_tier1_pnl=0.0,
                  mae_exit_bars=0, mae_exit_threshold=0.0,
                  liq_pct=-100.0, max_hold=360,
                  sl_trigger1=-25.0, sl_trigger2=-30.0,
                  sl_close1=50, sl_close2=100,
                  sl_activate_bars=0,
                  trail_stop=False, trail_activate=0.2, trail_step=0.05,
                  high_profit_threshold=50.0,
                  high_profit_drawback1=15, high_profit_close1=50,
                  high_profit_drawback2=20, high_profit_close2=50,
                  low_profit_threshold=20.0,
                  low_profit_drawback1=10, low_profit_close1=60,
                     micro_stop_bars=0, micro_stop_pnl=0.0,
                     early_filter=False,
                     liq_mm_rate=Config.LIQ_MM_RATE,
                     tp_tiers=None,
                     tp_tiers_low=None,
                     tp_tiers_high=None,
                     tp_profit_threshold=15.0,
                     reversion_exit=False, ref_mean=None, rev_stop_pct=2.0,
                     trend_exit=False, rev_tp_pct=2.0):
    """模拟1m粒度出场，内部调用共用退出逻辑 strategies.hf.exit_logic.should_exit()
    返回：(net_pnl, gross_pnl, hold_bars, reason)"""
    if reversion_exit and ref_mean is not None:
        return _simulate_reversion_exit(
            entry_idx, entry_price, closes, highs, lows,
            side, leverage, taker_fee, slippage, ref_mean, max_hold, rev_stop_pct)
    if trend_exit and ref_mean is not None:
        return _simulate_trend_exit(
            entry_idx, entry_price, closes, highs, lows,
            side, leverage, taker_fee, slippage, ref_mean, max_hold, rev_stop_pct, rev_tp_pct)
    should_exit_flag, reason, details = shared_should_exit(
        entry_price=entry_price,
        entry_idx=entry_idx,
        closes=closes, highs=highs, lows=lows,
        side=side, leverage=leverage,
        taker_fee=taker_fee, slippage=slippage,
        quick_stop_bars=quick_stop_bars, quick_stop_pnl=quick_stop_pnl,
        qs_tier1_bars=qs_tier1_bars, qs_tier1_pnl=qs_tier1_pnl,
        mae_exit_bars=mae_exit_bars, mae_exit_threshold=mae_exit_threshold,
        liq_pct=liq_pct, max_hold=max_hold,
        sl_trigger1=sl_trigger1, sl_trigger2=sl_trigger2,
        sl_close1=sl_close1, sl_close2=sl_close2,
        sl_activate_bars=sl_activate_bars,
        trail_stop=trail_stop, trail_activate=trail_activate, trail_step=trail_step,
        high_profit_threshold=high_profit_threshold,
        high_profit_drawback1=high_profit_drawback1, high_profit_close1=high_profit_close1,
        high_profit_drawback2=high_profit_drawback2, high_profit_close2=high_profit_close2,
        low_profit_threshold=low_profit_threshold,
        low_profit_drawback1=low_profit_drawback1, low_profit_close1=low_profit_close1,
        micro_stop_bars=micro_stop_bars, micro_stop_pnl=micro_stop_pnl,
        early_filter=early_filter,
        liq_mm_rate=liq_mm_rate,
        tp_tiers=tp_tiers,
        tp_tiers_low=tp_tiers_low,
        tp_tiers_high=tp_tiers_high,
        tp_profit_threshold=tp_profit_threshold,
    )
    net_pnl = details.get('net_pnl', 0)
    hold_bars = details.get('hold_bars', max_hold)
    exit_price = details.get('exit_price', entry_price)
    is_long = (side == 'LONG')
    # gross_pnl：不含费用，只含滑点（清算价不加滑点）
    exit_slip = slippage if reason != 'Liquidation' else 0
    eff_entry = entry_price * (1 + slippage) if is_long else entry_price * (1 - slippage)
    eff_exit = exit_price * (1 - exit_slip) if is_long else exit_price * (1 + exit_slip)
    pnl_mult = leverage * 100
    inv_ep = 1.0 / eff_entry
    if is_long:
        gross_pnl = (eff_exit - eff_entry) * inv_ep * pnl_mult
    else:
        gross_pnl = (eff_entry - eff_exit) * inv_ep * pnl_mult
    return net_pnl, gross_pnl, hold_bars, reason

# ── Core backtest ───────────────────────────────────────────────────
def run_backtest(direction=Config.DIRECTION, compound=Config.COMPOUND,
    taker_fee=Config.TAKER_FEE, slippage=Config.SLIPPAGE,
    days=0, trend_filter=Config.TREND_FILTER,
    confirm_bars=1, atr_sl_mult=0.0,
    trail_stop=Config.TRAIL_STOP, trail_activate=Config.TRAIL_ACTIVATE, trail_step=Config.TRAIL_STEP,
    sl1=Config.STOPLOSS_TRIGGER1, sl2=Config.STOPLOSS_TRIGGER2,
    sl_close1=Config.SL_CLOSE1_PCT, sl_close2=Config.SL_CLOSE2_PCT,
    sl_activate_bars=Config.SL_ACTIVATE_BARS, max_hold=Config.MAX_HOLD_1M,
    leverage=Config.LEVERAGE, dynamic_leverage=False,
    liq_pct=Config.LIQUIDATION_PCT, pct_threshold=Config.PCT_THRESHOLD, vol_surge=Config.VOL_SURGE,
    quick_stop_bars=Config.QUICK_STOP_BARS, quick_stop_pnl=Config.QUICK_STOP_PNL,
    qs_tier1_bars=Config.QS_TIER1_BARS, qs_tier1_pnl=Config.QS_TIER1_PNL,
    vol_persist=False,
    mae_exit_bars=Config.MAE_EXIT_BARS, mae_exit_threshold=Config.MAE_EXIT_THRESHOLD,
    micro_stop_bars=0, micro_stop_pnl=0.0,
    early_filter=False,
    pullback_bars=0,
    adaptive_vol=False,
    reversal_entry=False, reversal_pct=0.3,
     random_entry=False, flip_entry=False,
     reversion_signal=False, reversion_exit=False,
     rev_window=60, rev_entry_pct=2.0, rev_stop_pct=2.0,
     rev_direction='against', rev_tp_pct=2.0,
     diagnose=False,
     equal_alloc=True, equity_cap_multiple=100.0,
     position_pct=Config.POSITION_PCT):
    """..."""
    symbols = get_symbols()
    print(f"Symbols: {len(symbols)} | Direction: {direction} | Compound: {compound}")
    print(f"Costs: fee={taker_fee*100:.3f}% slip={slippage*100:.3f}% per side | Leverage: {leverage}x")
    print(f"Filters: trend={trend_filter}")
    if dynamic_leverage:
        print(f"Dynamic leverage: ON (ATR-based, base={leverage}x)")
    if trail_stop:
        print(f"Trailing stop: ON activate={trail_activate}% step={trail_step}%")
    print(f"SL: L1={sl1}%({sl_close1}%) L2={sl2}%({sl_close2}%) act={sl_activate_bars}b | MaxHold: {max_hold}m | Liquidation: {liq_pct}%")
    if mae_exit_bars > 0:
        print(f"MAE_Exit: {mae_exit_bars} bars, threshold={mae_exit_threshold}% (MAE<threshold -> exit)")
    if random_entry:
        print(f"[DIAG] RANDOM ENTRY: direction randomized at signal bars (signal timing preserved)")
    if flip_entry:
        print(f"[DIAG] FLIP ENTRY: direction reversed at signal bars (signal timing preserved)")
    if reversion_signal or reversion_exit:
        print(f"[REV] direction={rev_direction} window={rev_window} entry_pct={rev_entry_pct}% stop_pct={rev_stop_pct}% tp_pct={rev_tp_pct}%")

    print(f"Pool: {Config.MAX_POSITIONS} pos x 100% equal (max 100% utilized)\n")

    if random_entry:
        random.seed(42)

    all_trades = []

    for si, sym in enumerate(symbols):
        print(f"[{si+1}/{len(symbols)}] {sym}...", end=' ', flush=True)
        df1m = load_1m(sym, days=days)
        if df1m is None or len(df1m) < 600:
            print("skip (data)")
            continue

        df3m = build_3m_from_1m(df1m)
        if len(df3m) < 200:
            print("skip (3m data)")
            continue

        closes1m = df1m['close'].values.astype(np.float64)
        highs1m = df1m['high'].values.astype(np.float64)
        lows1m = df1m['low'].values.astype(np.float64)
        timestamps1m = pd.to_datetime(df1m['open_time']).astype(np.int64).values // 1_000  # 微秒→毫秒 (datetime64[us])

        closes3m = df3m['close'].values.astype(np.float64)
        highs3m = df3m['high'].values.astype(np.float64)
        lows3m = df3m['low'].values.astype(np.float64)
        volumes3m = df3m['volume'].values.astype(np.float64)
        timestamps3m = np.array([int(t.timestamp() * 1000) for t in df3m.index])

        # Map: 3m bar timestamp -> last 1m bar index in that 3m window
        ts1m_to_idx = {int(ts): idx for idx, ts in enumerate(timestamps1m)}
        ts3m_to_1m_idx = {}
        for i3, ts3 in enumerate(timestamps3m):
            if ts3 in ts1m_to_idx:
                ts3m_to_1m_idx[i3] = ts1m_to_idx[ts3]

        n3 = len(df3m)
        if reversion_signal:
            mean3m_arr = df3m['close'].rolling(rev_window).mean().shift(1).values.astype(np.float64)
        else:
            mean3m_arr = None
        active_until_3m = 0
        trades_this = []

        # 入场信号检测 —— 与 strategies/hf/strategy.py:analyze() 逻辑等价但独立实现
        # 两者都检测 3m 价格变化 >= PCT_THRESHOLD + 成交量 >= VOL_SURGE * avg_vol
        for i in range(60, n3):
                if active_until_3m > i:
                    continue

                if i not in ts3m_to_1m_idx:
                    continue
                entry_1m_idx = ts3m_to_1m_idx[i]
                ref_mean = None

                # 跳过信号位置太靠后、没有足够数据模拟出场的情况
                if entry_1m_idx > len(closes1m) - max_hold - 1:
                    continue

                # ── 波动率自适应阈值：高波提高门槛，低波降低门槛 ──
                _use_pct = pct_threshold
                if adaptive_vol and i >= 30:
                    _atr20 = [calc_atr(highs3m[j-14:j], lows3m[j-14:j], closes3m[j-14:j], 14) for j in range(i-20, i) if j >= 14]
                    if len(_atr20) >= 10:
                        _recent = np.mean(_atr20[-5:])
                        _hist = np.mean(_atr20)
                        if _hist > 0:
                            _vr = _recent / _hist
                            if _vr > 1.3:
                                _use_pct = max(_use_pct, pct_threshold * 1.5)
                            elif _vr < 0.7:
                                _use_pct = min(_use_pct, pct_threshold * 0.7)

# ── Step 1: 初始信号 — 3m 价格变化 + 成交量爆发/反转 ──
                price = closes3m[i]
                prev_price = closes3m[i - 1]
                price_change = (price - prev_price) / prev_price * 100

                vol_start = max(0, i - Config.VOL_AVG_BARS)
                avg_vol = volumes3m[vol_start:i].mean()
                if avg_vol == 0:
                    continue

                side = None
                if reversal_entry and not reversion_signal:
                    if i >= 2:
                        _prev = closes3m[i-1]
                        _pp = closes3m[i-2]
                        _prev_chg = (_prev - _pp) / _pp * 100
                        _prev_avg_vol = volumes3m[max(0, i-1-Config.VOL_AVG_BARS):i-1].mean()
                        _is_surge = _prev_avg_vol > 0 and volumes3m[i-1] >= _prev_avg_vol * vol_surge and abs(_prev_chg) >= pct_threshold
                        if _is_surge:
                            _rev_chg = (price - _prev) / _prev * 100
                            if _prev > _pp and _rev_chg <= -reversal_pct and direction in ('SHORT', 'BOTH'):
                                side = 'SHORT'  # 大阳→SHORT反转
                            elif _prev < _pp and _rev_chg >= reversal_pct and direction in ('LONG', 'BOTH'):
                                side = 'LONG'   # 大阴→LONG反转
                if not reversal_entry and not reversion_signal:
                    long_ok = (direction in ('LONG', 'BOTH')
                        and price > prev_price
                        and abs(price_change) >= _use_pct
                        and volumes3m[i] >= avg_vol * vol_surge)
                    if long_ok and confirm_bars > 1:
                        for cb in range(1, confirm_bars):
                            if i - cb < 1:
                                long_ok = False
                                break
                            prev_p = closes3m[i - cb]
                            pp = closes3m[i - cb - 1]
                            if prev_p <= pp:
                                long_ok = False
                                break
                    if long_ok and trend_filter and i >= 50:
                        ma20 = closes3m[i-20:i].mean()
                        ma50 = closes3m[i-50:i].mean()
                        if ma20 <= ma50:
                            long_ok = False

                    short_ok = (direction in ('SHORT', 'BOTH')
                            and price < prev_price
                            and abs(price_change) >= _use_pct
                            and volumes3m[i] >= avg_vol * vol_surge)
                    if short_ok and confirm_bars > 1:
                        for cb in range(1, confirm_bars):
                            if i - cb < 1:
                                short_ok = False
                                break
                            prev_p = closes3m[i - cb]
                            pp = closes3m[i - cb - 1]
                            if prev_p >= pp:
                                short_ok = False
                                break
                    if short_ok and trend_filter and i >= 50:
                        ma20 = closes3m[i-20:i].mean()
                        ma50 = closes3m[i-50:i].mean()
                        if ma20 >= ma50:
                            short_ok = False

                    if flip_entry:
                        # 同入场时刻/频率，方向取反：测试信号方向预测是否有贡献
                        if long_ok:
                            side = 'SHORT'
                        elif short_ok:
                            side = 'LONG'
                    elif random_entry:
                        # 在信号本应触发的相同时刻，随机选方向（验证信号方向预测是否有贡献）
                        _choices = []
                        if direction in ('LONG', 'BOTH'):
                            _choices.append('LONG')
                        if direction in ('SHORT', 'BOTH'):
                            _choices.append('SHORT')
                        side = random.choice(_choices) if _choices else None
                    else:
                        if long_ok:
                            side = 'LONG'
                        elif short_ok:
                            side = 'SHORT'

                if reversion_signal and side is None:
                    _mean = mean3m_arr[i]
                    if _mean > 0:
                        _dev = (closes3m[i] - _mean) / _mean * 100
                        if rev_direction == 'with':
                            # 趋势跟随：顺偏离方向进场，赌继续跑
                            if _dev >= rev_entry_pct and direction in ('LONG', 'BOTH'):
                                side = 'LONG'
                            elif _dev <= -rev_entry_pct and direction in ('SHORT', 'BOTH'):
                                side = 'SHORT'
                        else:
                            # 均值回归：逆偏离方向进场，赌回归
                            if _dev <= -rev_entry_pct and direction in ('LONG', 'BOTH'):
                                side = 'LONG'
                            elif _dev >= rev_entry_pct and direction in ('SHORT', 'BOTH'):
                                side = 'SHORT'
                    if side is not None:
                        ref_mean = _mean

                if side is None:
                    continue

                # ── Step 3: 可选 — 前一根3m bar也高于均量 ──
                if vol_persist and i >= 2:
                    prev_avg_vol = volumes3m[max(0, i-1-Config.VOL_AVG_BARS):i-1].mean()
                    if prev_avg_vol == 0 or volumes3m[i-1] < prev_avg_vol:
                        continue

                # ── Step 3b: 延迟入口 — 等待1m方向确认再入场
                if pullback_bars > 0:
                    _orig_entry = entry_1m_idx
                    _signal_close = closes1m[entry_1m_idx]
                    _confirmed = False
                    for _pb in range(1, min(pullback_bars + 1, len(closes1m) - entry_1m_idx)):
                        _ci = entry_1m_idx + _pb
                        if (side == 'LONG' and closes1m[_ci] > _signal_close) or \
                           (side == 'SHORT' and closes1m[_ci] < _signal_close):
                            entry_1m_idx = _ci
                            _confirmed = True
                            break
                    if not _confirmed:
                        continue

                entry = closes1m[entry_1m_idx]
                if ref_mean is None:
                    ref_mean = entry
                cur_sl1 = sl1
                cur_sl2 = sl2
                cur_lev = leverage

                if dynamic_leverage and i >= 15:
                    atr = calc_atr(highs3m[i-15:i+1], lows3m[i-15:i+1], closes3m[i-15:i+1], 14)
                    if atr > 0:
                        atr_pct = atr / entry * 100
                        target_risk = 5.0
                        atr_sl_pct = atr_pct * 2.0
                        if atr_sl_pct > 0:
                            cur_lev = min(leverage, max(5, int(target_risk / atr_sl_pct)))
                            cur_sl1 = -atr_sl_pct * cur_lev * 0.8
                            cur_sl2 = -atr_sl_pct * cur_lev

                pnl, gross_pnl, hold_1m, reason = simulate_exit(
                entry_1m_idx, entry,
                closes1m, highs1m, lows1m,
                side=side, leverage=cur_lev,
                taker_fee=taker_fee, slippage=slippage,
                quick_stop_bars=quick_stop_bars, quick_stop_pnl=quick_stop_pnl,
                qs_tier1_bars=qs_tier1_bars, qs_tier1_pnl=qs_tier1_pnl,
                mae_exit_bars=mae_exit_bars, mae_exit_threshold=mae_exit_threshold,
                liq_pct=liq_pct, max_hold=max_hold,
                sl_trigger1=cur_sl1, sl_trigger2=cur_sl2,
                sl_close1=sl_close1, sl_close2=sl_close2,
                sl_activate_bars=sl_activate_bars,
                trail_stop=trail_stop, trail_activate=trail_activate, trail_step=trail_step,
                high_profit_threshold=Config.HIGH_PROFIT_THRESHOLD,
                high_profit_drawback1=Config.HIGH_PROFIT_DRAWBACK1, high_profit_close1=Config.HIGH_PROFIT_CLOSE1,
                high_profit_drawback2=Config.HIGH_PROFIT_DRAWBACK2, high_profit_close2=Config.HIGH_PROFIT_CLOSE2,
                low_profit_threshold=Config.LOW_PROFIT_THRESHOLD,
                low_profit_drawback1=Config.LOW_PROFIT_DRAWBACK1, low_profit_close1=Config.LOW_PROFIT_CLOSE1,
                micro_stop_bars=micro_stop_bars, micro_stop_pnl=micro_stop_pnl,
                early_filter=early_filter,
                liq_mm_rate=Config.LIQ_MM_RATE,
                tp_tiers=getattr(Config, 'TP_TIERS', None),
                tp_tiers_low=getattr(Config, 'TP_TIERS_LOW', None),
                tp_tiers_high=getattr(Config, 'TP_TIERS_HIGH', None),
                tp_profit_threshold=getattr(Config, 'TP_PROFIT_THRESHOLD', 15.0),
                reversion_exit=(reversion_exit and rev_direction == 'against'),
                trend_exit=(reversion_exit and rev_direction == 'with'),
                ref_mean=ref_mean,
                rev_stop_pct=rev_stop_pct, rev_tp_pct=rev_tp_pct,
                )

                entry_ts = int(timestamps1m[entry_1m_idx])
                exit_1m_idx = min(entry_1m_idx + hold_1m, len(timestamps1m) - 1)
                exit_ts = int(timestamps1m[exit_1m_idx])
                
                # 出场后价格分析（后悔值）
                _post_exit = {}
                _post_horizons = [5, 10, 30, 60]  # 分钟
                for _h in _post_horizons:
                    _target_idx = min(exit_1m_idx + _h, len(closes1m) - 1)
                    if _target_idx > exit_1m_idx:
                        # 出场后h分钟的最高价
                        _max_high = max(highs1m[exit_1m_idx+1:_target_idx+1]) if exit_1m_idx+1 <= _target_idx else closes1m[exit_1m_idx]
                        # 计算后悔值（相对于出场价）
                        if side == 'LONG':
                            _regret = (_max_high - closes1m[exit_1m_idx]) / closes1m[exit_1m_idx] * 100
                        else:
                            _regret = (closes1m[exit_1m_idx] - _max_high) / closes1m[exit_1m_idx] * 100
                        _post_exit[f'max_after_{_h}m'] = _max_high
                        _post_exit[f'regret_{_h}m'] = _regret
                    else:
                        _post_exit[f'max_after_{_h}m'] = closes1m[exit_1m_idx]
                        _post_exit[f'regret_{_h}m'] = 0
                
                # 入场方向一致性检测
                _horizons = [1, 3, 5, 10, 15, 30, 60, 120, 360]
                _eq_ret = {}
                for _n in _horizons:
                    _hi = min(entry_1m_idx + _n, len(closes1m) - 1)
                    _px = closes1m[_hi]
                    # 价格收益率(%)：LONG为正=继续涨，SHORT对称为正=继续跌
                    _ret = (_px - entry) / entry * 100 if side == 'LONG' else (entry - _px) / entry * 100
                    _eq_ret[_n] = _ret
                trades_this.append((sym, entry_ts, pnl, exit_ts, gross_pnl, side, reason, _eq_ret, _post_exit))

                delay_1m = entry_1m_idx - ts3m_to_1m_idx[i]
                hold_3m = (hold_1m + delay_1m + 2) // 3  # 向上取整
                active_until_3m = i + hold_3m

                if diagnose and len(trades_this) < 100:
                    _path_pnls = []
                    for _pt in range(hold_1m + 1):
                        _pi = entry_1m_idx + _pt
                        if _pi >= len(closes1m):
                            break
                        _pc = closes1m[_pi]
                        _raw = (_pc - entry) / entry * cur_lev * 100
                        _path_pnls.append(_raw)
                    trades_this[-1] = trades_this[-1] + (entry_1m_idx, hold_1m, _path_pnls)

        if trades_this:
            print(f"{len(trades_this)} trades")
            all_trades.extend(trades_this)
        else:
            print(f"no signal")

        # ── Results ──
    if not all_trades:
        print("\nNo trades at all!")
        return [], 0, 0, []

    all_pnls = np.array([t[2] for t in all_trades])
    wins = all_pnls[all_pnls > 0]
    losses = all_pnls[all_pnls <= 0]
    n_trades = len(all_pnls)
    win_rate = len(wins) / n_trades * 100
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    print(f"\n{'='*70}")
    print(f"HF Backtest (1m) - {direction} | {'Compound' if compound else 'Single'}")
    print(f"Costs (embedded): fee={taker_fee*100:.3f}% slip={slippage*100:.3f}% per side | Lev: {leverage}x")
    print(f"{'='*70}")

    ppnl, pdd, curve = calc_pool_model(
        all_trades, compound=compound,
        max_positions=Config.MAX_POSITIONS, position_pct=position_pct,
        equal_alloc=equal_alloc, equity_cap_multiple=equity_cap_multiple
    )
    alloc_mode = "equal" if equal_alloc else f"{position_pct}%"
    cap_note = f"+compound(cap{equity_cap_multiple:.0f}x)" if compound else ""
    print(f"\nPool ({Config.MAX_POSITIONS}x{alloc_mode}{cap_note}): {ppnl:+.2f}% MaxDD: {pdd:.2f}%")
    print(f"Ratio: {ppnl/pdd:.2f}" if pdd > 0 else "Ratio: N/A")

    print(f"\nTrades: {n_trades}")
    print(f"Win Rate: {win_rate:.1f}% ({len(wins)}W/{len(losses)}L)")
    print(f"Avg Win: {avg_win:+.2f}% Avg Loss: {avg_loss:+.2f}%")
    print(f"Profit Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Profit Ratio: N/A")

    sym_stats = {}
    for t in all_trades:
        sym = t[0]
        if sym not in sym_stats:
            sym_stats[sym] = {'pnls': [], 'wins': 0, 'total': 0}
        sym_stats[sym]['pnls'].append(t[2])
        sym_stats[sym]['total'] += 1
        if t[2] > 0:
            sym_stats[sym]['wins'] += 1

    print(f"\n{'Symbol':<14} {'Trades':>6} {'Win%':>6} {'AvgPnL':>8} {'NetPnL':>8}")
    print('-' * 46)
    for sym, s in sorted(sym_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True):
        wr = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0
        ap = np.mean(s['pnls']) if s['pnls'] else 0
        net = sum(s['pnls'])
        print(f"{sym:<14} {s['total']:>6d} {wr:>5.1f}% {ap:>+7.2f}% {net:>+7.2f}%")

    dir_stats = {}
    for t in all_trades:
        d = t[5]
        if d not in dir_stats:
            dir_stats[d] = {'pnls': [], 'wins': 0, 'total': 0}
        dir_stats[d]['pnls'].append(t[2])
        dir_stats[d]['total'] += 1
        if t[2] > 0:
            dir_stats[d]['wins'] += 1
    print(f"\n--- By Direction ---")
    for d, s in dir_stats.items():
        wr = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0
        ap = np.mean(s['pnls']) if s['pnls'] else 0
        net = sum(s['pnls'])
        print(f"  {d:<6}: {s['total']:>4d} trades, WR {wr:.1f}%, Avg {ap:+.2f}%, Net {net:+.2f}%")

    reason_stats = {}
    for t in all_trades:
        r = t[6]
        if r not in reason_stats:
            reason_stats[r] = {'pnls': [], 'total': 0}
        reason_stats[r]['pnls'].append(t[2])
        reason_stats[r]['total'] += 1
    print(f"\n--- Exit Reasons ---")
    for r, s in sorted(reason_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True):
        ap = np.mean(s['pnls']) if s['pnls'] else 0
        net = sum(s['pnls'])
        print(f"  {r:<14}: {s['total']:>4d} trades, Avg {ap:+.2f}%, Net {net:+.2f}%")

    # 入场方向预测力诊断（按多/空分开，含平均N分钟收益率）
    if len(all_trades) > 0 and isinstance(all_trades[0][7], dict):
        _hors = [1, 3, 5, 10, 15, 30, 60, 120, 360]
        for _sd in ['LONG', 'SHORT']:
            _sub = [t for t in all_trades if t[5] == _sd]
            if not _sub:
                continue
            print(f"\n--- Entry Predictive Power: {_sd} (n={len(_sub)}) ---")
            print(f"  {'H(min)':>8} {'%favorable':>12} {'avgRet%':>10}")
            for _n in _hors:
                _rets = [t[7].get(_n, 0.0) for t in _sub]
                _fav = sum(1 for r in _rets if r > 0) / len(_rets) * 100
                _avg = sum(_rets) / len(_rets)
                print(f"  {_n:>8} {_fav:>11.1f}% {_avg:>+9.2f}%")

    if diagnose:
        _paths = [t for t in all_trades if len(t) > 10]
        if _paths:
            _bp, _pk, _nr, _rev = [], [], [], []
            for _t in _paths:
                _path = _t[10]
                _pos = next((i for i, v in enumerate(_path) if v > 0), _t[9])
                _bp.append(_pos)
                _pk.append(max(_path))
                _neg_rate = sum(1 for v in _path if v < 0) / max(len(_path), 1)
                _nr.append(_neg_rate)
                _rev.append(1 if any(v < 0 for v in _path[1:]) and any(v > 0 for v in _path[1:]) else 0)
            print(f"\n--- Holding Process (n={len(_paths)}) ---")
            print(f"  Avg bars to first +PnL: {np.mean(_bp):.1f}")
            print(f"  Avg peak PnL: {np.mean(_pk):+.2f}%")
            print(f"  Reversal rate (neg->pos): {sum(_rev)/len(_rev)*100:.1f}%")
            print(f"  Time in red: {np.mean(_nr)*100:.1f}%")
            print(f"  3 best final PnL: {', '.join(f'{_t[2]:+.2f}%' for _t in sorted(_paths, key=lambda x: x[2], reverse=True)[:3])}")
            print(f"  3 worst final PnL: {', '.join(f'{_t[2]:+.2f}%' for _t in sorted(_paths, key=lambda x: x[2])[:3])}")
            # 前几根bar的PnL分布
            _early = {}
            for _i in [1, 2, 3, 5]:
                _vals = [t[10][_i] if len(t[10]) > _i else 0 for t in _paths]
                _pos = sum(1 for v in _vals if v > 0)
                _early[_i] = (_pos / len(_vals) * 100, np.mean(_vals))
            print(f"  Bar1: {_early[1][0]:.0f}% positive, avg {_early[1][1]:+.2f}%")
            print(f"  Bar2: {_early[2][0]:.0f}% positive, avg {_early[2][1]:+.2f}%")
            print(f"  Bar3: {_early[3][0]:.0f}% positive, avg {_early[3][1]:+.2f}%")

    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"{direction}_{'c' if compound else 's'}_fee{str(taker_fee).replace('.','')}"
    out_csv = BASE / f"hf_backtest_{tag}_{ts_str}.csv"
    
    # 构建输出数据，包含出场后价格分析
    rows = []
    for t in all_trades:
        row = {
            'symbol': t[0], 'entry_ts': t[1], 'pnl_net': t[2],
            'exit_ts': t[3], 'pnl_gross': t[4], 'direction': t[5],
            'exit_reason': t[6]
        }
        # 添加出场后价格数据（如果有）
        if len(t) > 8 and isinstance(t[8], dict):
            post_exit = t[8]
            for key, val in post_exit.items():
                row[key] = val
        rows.append(row)
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {out_csv}")

    if curve:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            xs, ys = zip(*curve)
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(xs, ys, color='royalblue', linewidth=0.8)
            ax.set_title(f"HF Equity (1m) - {direction} {'compound' if compound else 'single'}")
            ax.set_ylabel("Equity")
            ax.grid(alpha=0.3)
            out_png = BASE / f"hf_equity_{tag}_{ts_str}.png"
            fig.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Chart: {out_png}")
        except Exception:
            pass

    return all_trades, ppnl, pdd, curve


def calc_pool_model(trades_all, compound=True, max_positions=5, position_pct=4, equal_alloc=False, equity_cap_multiple=100.0):
    if not trades_all:
        return None, None, None

    pool = float(max_positions * 100.0)
    initial_pool = float(max_positions * 100.0)

    entries_by_bar = {}
    for idx, t in enumerate(trades_all):
        sym, entry_ts, pnl_pct, exit_ts = t[0], t[1], t[2], t[3]
        trade_key = (sym, entry_ts, idx)
        entries_by_bar.setdefault(entry_ts, []).append((trade_key, pnl_pct, exit_ts))

    all_bars = sorted(set(
        [t[1] for t in trades_all] + [t[3] for t in trades_all]
    ))

    active = {}
    peak = pool
    max_dd = 0.0
    curve = []
    total = pool
    first_ts = all_bars[0] if all_bars else 0

    for bar in all_bars:
        hours = (bar - first_ts) / 3600000.0
        for key in list(active.keys()):
            pos = active[key]
            if bar >= pos['exit_bar']:
                clamped_pnl = max(pos['pnl_pct'], -100.0)
                pool += pos['allocated'] * (1 + clamped_pnl / 100)
                if equity_cap_multiple and pool > initial_pool * equity_cap_multiple:
                    pool = initial_pool * equity_cap_multiple
                del active[key]

        if bar in entries_by_bar:
            if equal_alloc:
                base = pool if compound else initial_pool
                unit_size = base / max_positions
            else:
                if compound:
                    unit_size = pool * position_pct / 100.0
                else:
                    unit_size = initial_pool * position_pct / 100.0
            for trade_key, pnl_pct, exit_bar in entries_by_bar[bar]:
                if len(active) >= max_positions:
                    break
                if equal_alloc:
                    if pool >= unit_size:
                        allocated = unit_size
                        pool -= allocated
                        active[trade_key] = {
                            'entry_bar': bar,
                            'allocated': allocated,
                            'pnl_pct': pnl_pct,
                            'exit_bar': exit_bar,
                        }
                else:
                    if pool >= unit_size:
                        allocated = pool * position_pct / 100.0 if compound else initial_pool * position_pct / 100.0
                        pool -= allocated
                        active[trade_key] = {
                            'entry_bar': bar,
                            'allocated': allocated,
                            'pnl_pct': pnl_pct,
                            'exit_bar': exit_bar,
                        }

        margin_used = sum(p['allocated'] for p in active.values())
        total = pool + margin_used
        curve.append((hours, total))
        if total > peak:
            peak = total
        dd_pct = (peak - total) / peak * 100 if peak > 0 else 0.0
        if dd_pct > max_dd:
            max_dd = dd_pct

    total_pnl = (total / initial_pool - 1) * 100
    return total_pnl, max_dd, curve


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HF Official Backtest (1m)')
    parser.add_argument('--direction', default=Config.DIRECTION, choices=['LONG','SHORT','BOTH'])
    parser.add_argument('--compound', action='store_true', default=Config.COMPOUND)
    parser.add_argument('--no-compound', dest='compound', action='store_false')
    parser.add_argument('--cap', type=float, default=100.0,
                        help='Compound equity cap as multiple of initial capital (default 100x). Profits above cap are withdrawn to keep compounding realistic.')
    parser.add_argument('--fee', type=float, default=Config.TAKER_FEE)
    parser.add_argument('--slippage', type=float, default=Config.SLIPPAGE)
    parser.add_argument('--days', type=int, default=0, help='Max backtest days (0=all available)')
    parser.add_argument('--trend', dest='trend_filter', action='store_true', default=Config.TREND_FILTER)
    parser.add_argument('--no-trend', dest='trend_filter', action='store_false')
    parser.add_argument('--trail', dest='trail_stop', action='store_true', default=Config.TRAIL_STOP)
    parser.add_argument('--no-trail', dest='trail_stop', action='store_false')
    parser.add_argument('--trail-activate', type=float, default=Config.TRAIL_ACTIVATE)
    parser.add_argument('--trail-step', type=float, default=Config.TRAIL_STEP)
    parser.add_argument('--leverage', type=int, default=Config.LEVERAGE)
    parser.add_argument('--dynamic-lev', action='store_true', help='ATR-based dynamic leverage')
    parser.add_argument('--liq', type=float, default=Config.LIQUIDATION_PCT, help='Liquidation pct (default -100)')
    parser.add_argument('--sl1', type=float, default=Config.STOPLOSS_TRIGGER1)
    parser.add_argument('--sl2', type=float, default=Config.STOPLOSS_TRIGGER2)
    parser.add_argument('--mae-bars', dest='mae_exit_bars', type=int, default=Config.MAE_EXIT_BARS,
                        help='MAE exit: exit if MAE < threshold after N bars (0=disable)')
    parser.add_argument('--mae-threshold', dest='mae_exit_threshold', type=float, default=Config.MAE_EXIT_THRESHOLD,
                        help='MAE exit threshold in %% (e.g., -10 means MAE < -10%% triggers exit)')
    parser.add_argument('--qs-t1-bars', dest='qs_tier1_bars', type=int, default=Config.QS_TIER1_BARS,
                        help='QuickStop Tier1: activate after N bars (0=disable)')
    parser.add_argument('--qs-t1-pnl', dest='qs_tier1_pnl', type=float, default=Config.QS_TIER1_PNL,
                        help='QuickStop Tier1: trigger when PnL below this %%')
    parser.add_argument('--micro-stop-bars', type=int, default=0,
                        help='Micro stop: exit within N bars if PnL below threshold')
    parser.add_argument('--micro-stop-pnl', type=float, default=0.0,
                        help='Micro stop: PnL threshold %% (e.g., -0.5)')
    parser.add_argument('--early-filter', action='store_true',
                        help='Close 50%% if bar1 PnL negative')
    parser.add_argument('--confirm-bars', type=int, default=Config.CONFIRM_BARS,
                        help='confirm bars (default Config)')
    parser.add_argument('--vol-surge', type=float, default=Config.VOL_SURGE,
                        help='volume surge threshold (default Config)')
    parser.add_argument('--vol-persist', action='store_true', default=Config.VOL_PERSIST)
    parser.add_argument('--no-vol-persist', dest='vol_persist', action='store_false')
    parser.add_argument('--pct', type=float, default=Config.PCT_THRESHOLD,
                        help='price change threshold %% (default Config)')
    parser.add_argument('--pullback-bars', type=int, default=0,
                        help='Wait N 1m bars for direction confirmation before entry (0=off)')
    parser.add_argument('--adaptive-vol', action='store_true',
                        help='Adjust pct_threshold based on ATR volatility regime')
    parser.add_argument('--reversal', action='store_true',
                        help='Reversal entry: surge bar + opposite follow-through')
    parser.add_argument('--reversal-pct', type=float, default=0.3,
                        help='Reversal follow-through threshold %% (default 0.3)')
    parser.add_argument('--random-entry', action='store_true',
                        help='[DIAG] Randomize entry direction at signal bars (test signal value)')
    parser.add_argument('--flip-entry', action='store_true',
                        help='[DIAG] Reverse entry direction at signal bars (test signal value)')
    parser.add_argument('--reversion', action='store_true',
                        help='Mean-reversion mode: deviation-from-mean entry + revert-to-mean exit')
    parser.add_argument('--rev-window', type=int, default=60,
                        help='Reversion mean window in 3m bars (default 60 = 3h)')
    parser.add_argument('--rev-entry-pct', type=float, default=2.0,
                        help='Reversion entry deviation %% from mean (default 2.0)')
    parser.add_argument('--rev-stop-pct', type=float, default=2.0,
                        help='Reversion stop %% beyond entry if stretch continues (default 2.0)')
    parser.add_argument('--rev-direction', choices=['against', 'with'], default='against',
                        help='Reversion signal direction: against=mean-reversion, with=trend-following the deviation')
    parser.add_argument('--rev-tp-pct', type=float, default=2.0,
                        help='Trend-follow TP %% continuation beyond entry (default 2.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for --random-entry')
    parser.add_argument('--diagnose', action='store_true',
                        help='Detailed holding process diagnosis')
    parser.add_argument('--position-pct', type=float, default=Config.POSITION_PCT,
                        help='Position size as %% of pool (default 4)')
    parser.add_argument('--no-equal-alloc', dest='equal_alloc', action='store_false',
                        help='Use POSITION_PCT instead of equal allocation')
    args = parser.parse_args()
    run_backtest(
        direction=args.direction,
        compound=args.compound,
        equity_cap_multiple=args.cap,
        taker_fee=args.fee,
        slippage=args.slippage,
        days=args.days,
        trend_filter=args.trend_filter,
        trail_stop=args.trail_stop,
        trail_activate=args.trail_activate,
        trail_step=args.trail_step,
        leverage=args.leverage,
        dynamic_leverage=args.dynamic_lev,
        liq_pct=args.liq,
        sl1=args.sl1,
        sl2=args.sl2,
        confirm_bars=args.confirm_bars,
        vol_surge=args.vol_surge,
        vol_persist=args.vol_persist,
        pct_threshold=args.pct,
        mae_exit_bars=args.mae_exit_bars,
        mae_exit_threshold=args.mae_exit_threshold,
        qs_tier1_bars=args.qs_tier1_bars,
        qs_tier1_pnl=args.qs_tier1_pnl,
        pullback_bars=args.pullback_bars,
        adaptive_vol=args.adaptive_vol,
        micro_stop_bars=args.micro_stop_bars,
        micro_stop_pnl=args.micro_stop_pnl,
        early_filter=args.early_filter,
        reversal_entry=args.reversal,
        reversal_pct=args.reversal_pct,
        random_entry=args.random_entry,
        flip_entry=args.flip_entry,
        reversion_signal=args.reversion,
        reversion_exit=args.reversion,
        rev_window=args.rev_window,
        rev_entry_pct=args.rev_entry_pct,
        rev_stop_pct=args.rev_stop_pct,
        rev_direction=args.rev_direction,
        rev_tp_pct=args.rev_tp_pct,
        diagnose=args.diagnose,
        equal_alloc=args.equal_alloc,
        position_pct=args.position_pct,
    )
