# pyright: reportRedeclaration=false
# 说明: 此文件使用热加载设计，类属性在_reload_env_vars方法中重新赋值是预期行为
# 禁用LSP的重定义警告以避免误报

import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import time
import threading

logger = logging.getLogger(__name__)

# ========== 类型转换安全辅助函数（防御性编程）==========
def safe_int(value: Any, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """安全地将值转换为整数，带防御性检查"""
    try:
        result = int(float(str(value)))
        if min_val is not None and result < min_val:
            logger.warning(f"值 {value} 低于最小值 {min_val}，使用最小值")
            return min_val
        if max_val is not None and result > max_val:
            logger.warning(f"值 {value} 超过最大值 {max_val}，使用最大值")
            return max_val
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"整数转换失败: {value} ({e})，使用默认值 {default}")
        return default

def safe_float(value: Any, default: float = 0.0, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """安全地将值转换为浮点数，带防御性检查"""
    try:
        result = float(str(value))
        if min_val is not None and result < min_val:
            logger.warning(f"值 {value} 低于最小值 {min_val}，使用最小值")
            return min_val
        if max_val is not None and result > max_val:
            logger.warning(f"值 {value} 超过最大值 {max_val}，使用最大值")
            return max_val
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"浮点数转换失败: {value} ({e})，使用默认值 {default}")
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """安全地将值转换为布尔值"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    try:
        return bool(value)
    except (ValueError, TypeError):
        logger.warning(f"布尔值转换失败: {value}，使用默认值 {default}")
        return default

# 环境变量路径（延迟加载，避免模块导入时污染环境变量）
env_path = Path(__file__).parent.parent / '.env'
strategies_dir = Path(__file__).parent.parent / 'strategies'
hf_env = strategies_dir / 'hf' / '.env'
tupo_env = strategies_dir / '15mTupo' / '.env'

# 延迟加载环境变量
_env_loaded = False
_env_lock = threading.Lock()

def _load_env_vars():
    """加载环境变量（延迟加载，避免模块导入时污染）"""
    global _env_loaded
    if _env_loaded:
        return
    
    with _env_lock:
        if _env_loaded:
            return
        
        # 加载主配置（不覆盖已有环境变量）
        if env_path.exists():
            load_dotenv(env_path, override=False)
        
        # 加载策略私有配置（不覆盖已有环境变量）
        if hf_env.exists():
            load_dotenv(hf_env, override=False)
            print(f"[配置] HF私有配置已加载: {hf_env}")
        
        if tupo_env.exists():
            load_dotenv(tupo_env, override=False)
            print(f"[配置] 15mTupo私有配置已加载: {tupo_env}")
        
        _env_loaded = True

# 加载环境变量（在类定义之前）
_load_env_vars()

class Settings:
    """系统配置类 - 所有可调参数均从环境变量读取，带中文注释，便于维护与分块实现

    支持参数热读取：修改 .env 文件后无需重启，下次读取时自动更新
    """
    # 最后修改时间（用于检测文件变化）
    _last_modified = 0
    _reload_interval = 60  # 热重载间隔（秒），默认60秒
    _reload_thread = None
    _stop_reload = threading.Event()
    _reload_lock = threading.RLock()  # 热重载锁，防止竞态条件
    # API 配置
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')  # 实盘：币安API Key
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')  # 实盘：币安API Secret
    TESTNET_API_KEY = os.getenv('TESTNET_API_KEY', '')  # 测试网：API Key
    TESTNET_API_SECRET = os.getenv('TESTNET_API_SECRET', '')  # 测试网：API Secret

    # Telegram 配置
    TELEGRAM_BOT_TOKEN_ALERT = os.getenv('TELEGRAM_BOT_TOKEN_ALERT', '')  # 警报 Telegram 机器人 Token
    TELEGRAM_BOT_TOKEN_TRADE = os.getenv('TELEGRAM_BOT_TOKEN_TRADE', '')  # 交易 Telegram 机器人 Token
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')  # Telegram 聊天 ID

    # 警报参数 - 监控阈值与频率
    MONITOR_INTERVAL = int(float(os.getenv('MONITOR_INTERVAL', '120')))  # 监控周期（分钟），默认2小时
    MONITOR_SYMBOLS_COUNT = int(float(os.getenv('MONITOR_SYMBOLS_COUNT', '100')))  # 按成交量排序前 N 名币种进行监控
    SYMBOLS_UPDATE_INTERVAL = int(float(os.getenv('SYMBOLS_UPDATE_INTERVAL', '60')))  # 监控币种排序更新间隔（秒）
    PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '3.0'))  # 价格变化阈值（%）
    VOLUME_THRESHOLD = float(os.getenv('VOLUME_THRESHOLD', '1.5'))  # 成交量放大阈值（倍数）
    
    # ==================== BNFF 策略参数 (基于回测结果) ====================
    # 做空: 5连跌 > 3% (期望 +11.13%, 胜率 69.8%)
    SHORT_DROP_COUNT = int(os.getenv('SHORT_DROP_COUNT', '5'))
    SHORT_DROP_THRESHOLD = float(os.getenv('SHORT_DROP_THRESHOLD', '3.0'))
    SHORT_DROP_VOLUME_REQUIRED = os.getenv('SHORT_DROP_VOLUME_REQUIRED', 'false').lower() == 'true'  # 做空是否需要成交量放大
    SHORT_DROP_VOLUME_THRESHOLD = float(os.getenv('SHORT_DROP_VOLUME_THRESHOLD', '1.5'))  # 做空成交量放大阈值
    
    # 做多: 3连跌后反弹 > 0.2% (期望 +6.62%, 胜率 83.3%)
    LONG_BOUNCE_COUNT = int(os.getenv('LONG_BOUNCE_COUNT', '3'))
    LONG_BOUNCE_THRESHOLD = float(os.getenv('LONG_BOUNCE_THRESHOLD', '0.2'))
    LONG_BOUNCE_VOLUME_THRESHOLD = float(os.getenv('LONG_BOUNCE_VOLUME_THRESHOLD', '1.5'))  # 成交量倍数阈值（做多反弹需成交量放大）
    LONG_BOUNCE_VOLUME_REQUIRED = os.getenv('LONG_BOUNCE_VOLUME_REQUIRED', 'true').lower() == 'true'  # 做多反弹是否需要成交量放大
    LONG_BOUNCE_OI_REQUIRED = os.getenv('LONG_BOUNCE_OI_REQUIRED', 'true').lower() == 'true'  # 做多是否需要持仓量增加
    
    # 做多: 单根涨 > 5% (提高阈值减少反转假信号，原2%胜率仅38.8%)
    LONG_SURGE_THRESHOLD = float(os.getenv('LONG_SURGE_THRESHOLD', '5.0'))
    LONG_SURGE_VOLUME_REQUIRED = os.getenv('LONG_SURGE_VOLUME_REQUIRED', 'true').lower() == 'true'  # 做多暴涨是否需要成交量放大
    LONG_SURGE_VOLUME_THRESHOLD = float(os.getenv('LONG_SURGE_VOLUME_THRESHOLD', '2.0'))  # 做多暴涨成交量放大阈值
    # ==================== BNFF 策略参数结束 ====================
    
    # 持仓量监控参数
    OPEN_INTEREST_MONITOR_ENABLED = os.getenv('OPEN_INTEREST_MONITOR_ENABLED', 'false').lower() == 'true'  # 持仓增量监控开关（默认关闭）
    OPEN_INTEREST_INCREASE_THRESHOLD = float(os.getenv('OPEN_INTEREST_INCREASE_THRESHOLD', '5.0'))  # 持仓增量阈值（%）
    
    # 退出时平仓
    EXIT_CLOSE_ALL_POSITIONS = os.getenv('EXIT_CLOSE_ALL_POSITIONS', 'true').lower() == 'true'  # 退出时是否平仓
    
    # 警报冷却时间
    ALERT_COOLDOWN = int(os.getenv('ALERT_COOLDOWN', '60'))  # 警报冷却时间（分钟）
    
    # 做空策略持仓量过滤
    SHORT_DROP_OI_REQUIRED = os.getenv('SHORT_DROP_OI_REQUIRED', 'false').lower() == 'true'  # 做空是否需要持仓量增加
    TREND_INTERVAL = os.getenv('TREND_INTERVAL', '2h')  # 趋势判断K线周期（如：1h, 2h, 4h）

    # ==================== 策略开关 ====================
    # 15mTupo策略开关（原V23策略）- 兼容旧配置V23_ENABLED
    _tmp_str_enabled = os.getenv('15MTUPO_ENABLED') or os.getenv('V23_ENABLED') or 'true'
    STRAT15_ENABLED = _tmp_str_enabled.lower() == 'true'
    # 高频超短线策略开关
    HF_ENABLED = os.getenv('HF_ENABLED', 'false').lower() == 'true'
    # 策略资金分配模式：SEPARATE=分开, COMBINED=组合
    STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'SEPARATE')
    
    # V23策略参数
    
    # V23策略参数
    V23_MAX_POSITIONS = int(float(os.getenv('V23_MAX_POSITIONS', '5')))  # V23最大持仓数
    V23_SINGLE_SYMBOL_MAX = float(os.getenv('V23_SINGLE_SYMBOL_MAX', '5000'))  # V23单币种最大投资
    V23_MONITOR_SYMBOLS = int(float(os.getenv('V23_MONITOR_SYMBOLS', '50')))  # V23监控币种数量（按成交量排序取前N）
    V23_LOAD_KLINES_COUNT = int(float(os.getenv('V23_LOAD_KLINES_COUNT', '0')))  # 启动时预加载K线的币种数量（0=不预加载，使用实时数据）
    
    # V23实盘监控参数
    V23_SIGNAL_COOLDOWN = int(os.getenv('V23_SIGNAL_COOLDOWN', '300'))  # 信号冷却时间（秒）
    V23_SIGNAL_CHECK_INTERVAL = int(os.getenv('V23_SIGNAL_CHECK_INTERVAL', '2'))  # 信号检查间隔（秒）
    V23_KLINE_HISTORY_COUNT = int(os.getenv('V23_KLINE_HISTORY_COUNT', '100'))  # K线历史数量
    V23_FALLBACK_SYMBOLS = os.getenv('V23_FALLBACK_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT')  # 备用币种列表

    # 15mTupo策略参数
    TREND_PERIOD = int(os.getenv('15MTUPO_TREND_PERIOD') or os.getenv('TREND_PERIOD', '16'))  # 趋势判断K线数
    CONSOLIDATION_PERIOD = int(os.getenv('15MTUPO_CONSOLIDATION_PERIOD') or os.getenv('CONSOLIDATION_PERIOD', '40'))  # 震荡判断周期
    SIGNAL_VOLUME_THRESHOLD = float(os.getenv('15MTUPO_SIGNAL_VOLUME_THRESHOLD') or os.getenv('SIGNAL_VOLUME_THRESHOLD', '1.5'))  # 信号成交量倍数
    VOLUME_COMPARE_PERIODS = int(os.getenv('VOLUME_COMPARE_PERIODS', '10'))  # 成交量对比周期数
    SIGNAL_BODY_RATIO = float(os.getenv('15MTUPO_SIGNAL_BODY_RATIO') or os.getenv('SIGNAL_BODY_RATIO', '0.5'))  # K线实体比阈值
    SIGNAL_CLOSE_POSITION = float(os.getenv('15MTUPO_SIGNAL_CLOSE_POSITION') or os.getenv('SIGNAL_CLOSE_POSITION', '0.6'))  # 收盘位置阈值
    SIGNAL_BREAKOUT_THRESHOLD = float(os.getenv('15MTUPO_SIGNAL_BREAKOUT_THRESHOLD') or os.getenv('SIGNAL_BREAKOUT_THRESHOLD', '0.3'))  # 突破幅度阈值
    TREND_PRICE_CHANGE = float(os.getenv('15MTUPO_TREND_PRICE_CHANGE') or os.getenv('TREND_PRICE_CHANGE', '2.0'))  # 趋势涨幅/跌幅阈值(%)
    EXIT_DRAWDOWN_30 = float(os.getenv('15MTUPO_EXIT_DRAWDOWN_30') or os.getenv('EXIT_DRAWDOWN_30', '30'))  # 回撤30%平仓25%
    EXIT_DRAWDOWN_50 = float(os.getenv('15MTUPO_EXIT_DRAWDOWN_50') or os.getenv('EXIT_DRAWDOWN_50', '50'))  # 回撤50%平仓100%
    POSITION_COMPLETE_STEP_LOSS = float(os.getenv('POSITION_COMPLETE_STEP_LOSS', '-10.0'))  # 分批建仓完成后的止损阈值
    EXIT_HOLD_BARS_STOP = int(os.getenv('15MTUPO_EXIT_HOLD_BARS_STOP') or os.getenv('EXIT_HOLD_BARS_STOP', '30'))  # 入场止损K线数
    EXIT_BREAKEVEN_PNL = float(os.getenv('15MTUPO_EXIT_BREAKEVEN_PNL') or os.getenv('EXIT_BREAKEVEN_PNL', '10'))  # 保本触发阈值
    REBUILD_FIRST_RATIO = float(os.getenv('15MTUPO_REBUILD_FIRST_RATIO') or os.getenv('REBUILD_FIRST_RATIO', '50'))  # 分批建仓首笔比例
    REBUILD_SECOND_RATIO = float(os.getenv('15MTUPO_REBUILD_SECOND_RATIO') or os.getenv('REBUILD_SECOND_RATIO', '50'))  # 分批建仓第二笔比例
    
    # 组合模式资金分配
    V23_CAPITAL_RATIO = float(os.getenv('V23_CAPITAL_RATIO', '70'))  # V23占用资金比例%
    HF_CAPITAL_RATIO = float(os.getenv('HF_CAPITAL_RATIO', '30'))  # 高频占用资金比例%
    
    # 数据下载参数
    MIN_HISTORY_BARS = int(float(os.getenv('MIN_HISTORY_BARS', '1500')))  # 最小历史K线数量
    DOWNLOAD_DAYS = int(float(os.getenv('DOWNLOAD_DAYS', '90')))  # 下载历史数据天数
    AUTO_DOWNLOAD_ON_START = os.getenv('AUTO_DOWNLOAD_ON_START', 'true').lower() == 'true'  # 启动时自动下载缺失数据

    # 持仓监控参数（根据杠杆调整刷新频率）
    POSITION_MONITOR_SLEEP_TIME = int(float(os.getenv('POSITION_MONITOR_SLEEP_TIME', '15')))  # 持仓监控刷新间隔（秒）
    POSITION_SYNC_INTERVAL = int(os.getenv('POSITION_SYNC_INTERVAL', '10'))  # 持仓同步间隔（秒）
    # 推荐值：
    # - 5x杠杆: 30秒（价格波动影响小）
    # - 10x杠杆: 20秒（中等风险）
    # - 15-20x杠杆: 15秒（推荐，当前设置）
    # - 20-30x杠杆: 10秒（高风险）
    # - >30x杠杆: 5秒（极度危险，但API限流风险高）
    
    # 交易参数
    MARGIN_MODE = os.getenv('MARGIN_MODE', 'ISOLATED')  # 保证金模式（ISOLATED / CROSSED）
    LEVERAGE = int(float(os.getenv('LEVERAGE', '20')))  # 最大杠杆（在币种允许范围内取小值）
    MAX_POSITIONS = int(float(os.getenv('MAX_POSITIONS', '5')))  # 最大同时持仓数量
    MAX_POSITIONS_PER_SYMBOL = int(float(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3')))  # 单币种最大持仓数
    SINGLE_SYMBOL_MAX_INVESTMENT = float(os.getenv('SINGLE_SYMBOL_MAX_INVESTMENT', '5000'))  # 单币种最大投资金额（USDT）
    POSITION_ALLOCATION_MODE = os.getenv('POSITION_ALLOCATION_MODE', 'EQUAL')  # 资金分配模式（EQUAL / DYNAMIC）
    ORDER_MODE = os.getenv('ORDER_MODE', 'LIMIT_TO_MARKET')  # 订单执行模式（LIMIT_ONLY / MARKET_ONLY / LIMIT_TO_MARKET）
    ORDER_TYPE = os.getenv('ORDER_TYPE', 'MARKET')  # 下单类型（MARKET / LIMIT），已废弃，使用 ORDER_MODE 代替
    INITIAL_POSITION = float(os.getenv('INITIAL_POSITION', '15'))  # 初始下单占总资金比例（%），降低初始开仓比例
    DELAY_RATIO = float(os.getenv('DELAY_RATIO', '0.003'))  # 下单延迟系数（0.3%以确保止盈限价单能成交）
    MAX_MARGIN_PER_SYMBOL = float(os.getenv('MAX_MARGIN_PER_SYMBOL', '25000'))  # 单币种最大保证金（USDT）
    ORDER_LIMIT_TIMEOUT = int(os.getenv('ORDER_LIMIT_TIMEOUT', '5'))  # 限价单转市价超时时间（秒）

    # 订单类型判断参数
    ORDER_TYPE_MARKET_THRESHOLD = float(os.getenv('ORDER_TYPE_MARKET_THRESHOLD', '1000'))  # 小额订单用市价单的阈值(USDT)
    ORDER_TYPE_LOW_VOLUME_THRESHOLD = float(os.getenv('ORDER_TYPE_LOW_VOLUME_THRESHOLD', '100000'))  # 低流动性用限价单的阈值(日交易量USDT)
    ORDER_TYPE_LIMIT_PRICE_BUY_BIAS = float(os.getenv('ORDER_TYPE_LIMIT_PRICE_BUY_BIAS', '1.001'))  # 买入限价单价格偏移(1.001=+0.1%)
    ORDER_TYPE_LIMIT_PRICE_SELL_BIAS = float(os.getenv('ORDER_TYPE_LIMIT_PRICE_SELL_BIAS', '0.999'))  # 卖出限价单价格偏移(0.999=-0.1%)

    # 限价单转市价阈值（用于未成交时价格偏离或亏损扩大时自动转市价）
    LIMIT_TO_MARKET_PRICE_DEVIATION_THRESHOLD = float(os.getenv('LIMIT_TO_MARKET_PRICE_DEVIATION_THRESHOLD', '0.5'))  # 价格偏离阈值（%），超过此值转市价
    LIMIT_TO_MARKET_TIME_THRESHOLD = float(os.getenv('LIMIT_TO_MARKET_TIME_THRESHOLD', '8'))  # 限价单等待时间（秒），超过此值转市价
    LIMIT_TO_MARKET_PNL_DECLINE_THRESHOLD = float(os.getenv('LIMIT_TO_MARKET_PNL_DECLINE_THRESHOLD', '2.0'))  # 盈利率下降阈值（%），超过此值转市价

    # 回测与策略参数
    # 回测基础参数
    BACKTEST_INITIAL_CAPITAL = float(os.getenv('BACKTEST_INITIAL_CAPITAL', '10000')) # 回测初始资金（USDT）
    BACKTEST_RESULT_DIR = os.getenv('BACKTEST_RESULT_DIR', 'results') # 回测结果目录
    BACKTEST_DATA_DIR = os.getenv('BACKTEST_DATA_DIR', 'data') # 回测数据目录

    # 入场止损参数
    ENTRY_STOP_LOSS_BARS = int(float(os.getenv('ENTRY_STOP_LOSS_BARS', '30'))) # 入场止损K线数（持仓超过此数无盈利则平仓）
    ENTRY_STOP_LOSS_PCT = float(os.getenv('ENTRY_STOP_LOSS_PCT', '0')) # 入场止损盈亏率（<=此值触发入场止损）

    # 保本出场参数
    PROFIT_PROTECT_THRESHOLD = float(os.getenv('PROFIT_PROTECT_THRESHOLD', '10')) # 保本出场阈值（最大盈利超过此值且回到±1%内平仓）

    # 回撤止盈参数
    DRAWDOWN_30_THRESHOLD = float(os.getenv('DRAWDOWN_30_THRESHOLD', '30')) # 回撤30%止盈阈值
    DRAWDOWN_50_THRESHOLD = float(os.getenv('DRAWDOWN_50_THRESHOLD', '50')) # 回撤50%止盈阈值

    # 逆势出场参数
    COUNTER_LOSS_10 = float(os.getenv('COUNTER_LOSS_10', '10')) # 逆势亏损10%减半
    COUNTER_LOSS_20 = float(os.getenv('COUNTER_LOSS_20', '20')) # 逆势亏损20%离场

    # 趋势反转参数
    TREND_REVERSAL_BARS = int(float(os.getenv('TREND_REVERSAL_BARS', '60'))) # 趋势反转检测K线数

    LOSS_STEP1 = float(os.getenv('LOSS_STEP1', '-2.0')) # 亏损加仓阈值 1
    LOSS_ADD1 = float(os.getenv('LOSS_ADD1', '15'))  # 亏损加仓额度 1
    LOSS_STEP2 = float(os.getenv('LOSS_STEP2', '-4.0'))  # 亏损加仓阈值 2
    LOSS_ADD2 = float(os.getenv('LOSS_ADD2', '25'))  # 亏损加仓额度 2
    LOSS_STEP3 = float(os.getenv('LOSS_STEP3', '-6.0'))  # 亏损加仓阈值 3
    LOSS_ADD3 = float(os.getenv('LOSS_ADD3', '30'))  # 亏损加仓额度 3

    PROFIT_STEP1 = float(os.getenv('PROFIT_STEP1', '2.0'))  # 盈利加仓阈值 1
    PROFIT_ADD1 = float(os.getenv('PROFIT_ADD1', '10'))  # 盈利加仓额度 1
    PROFIT_STEP2 = float(os.getenv('PROFIT_STEP2', '4.0'))  # 盈利加仓阈值 2
    PROFIT_ADD2 = float(os.getenv('PROFIT_ADD2', '15'))  # 盈利加仓额度 2
    PROFIT_STEP3 = float(os.getenv('PROFIT_STEP3', '6.0'))  # 盈利加仓阈值 3
    PROFIT_ADD3 = float(os.getenv('PROFIT_ADD3', '20'))  # 盈利加仓额度 3

    # 建仓完成判断参数
    POSITION_COMPLETE_PROFIT_RISE = float(os.getenv('POSITION_COMPLETE_PROFIT_RISE', '10.0'))  # 建仓完成的利润上涨阈值
    POSITION_COMPLETE_LOSS_FALL = float(os.getenv('POSITION_COMPLETE_LOSS_FALL', '1.0'))  # 建仓完成的亏损回落阈值

    # ==================== 止盈参数（盈利率包含杠杆倍数） ====================
    # 例如：20x杠杆下，价格涨1% → 盈利率=1%×20=20%
    # 所有盈利率参数都是含杠杆的数值

    # 【高盈利止盈】三级分段止盈策略（适用于高盈利场景）
    # 先读策略私有配置，没有则用全局参数
    HIGH_PROFIT_THRESHOLD = float(
        os.getenv('HIGH_PROFIT_THRESHOLD', '60.0')
    )
    HIGH_PROFIT_DRAWBACK1 = float(
        os.getenv('15MTUPO_DRAWDOWN_TIER1_PCT') or
        os.getenv('HIGH_PROFIT_DRAWBACK1', '20.0')
    )
    HIGH_PROFIT_CLOSE1 = float(
        os.getenv('15MTUPO_TP_HIGH_CLOSE_PCT') or
        os.getenv('HIGH_PROFIT_CLOSE1', '50')
    )
    HIGH_PROFIT_DRAWBACK2 = float(os.getenv('HIGH_PROFIT_DRAWBACK2', '20.0'))
    HIGH_PROFIT_CLOSE2 = float(os.getenv('HIGH_PROFIT_CLOSE2', '50'))

    # 【低盈利止盈】较低盈利的分段止盈策略
    # 触发条件：最高盈利在 [LOW_PROFIT_THRESHOLD, HIGH_PROFIT_THRESHOLD) 区间（例：50%-60%）
    # - 第一级止盈：从最高盈利回撤达到 DRAWBACK1 时（例：回撤20%），平仓 CLOSE1%（例：50%）
    LOW_PROFIT_THRESHOLD = float(
        os.getenv('LOW_PROFIT_THRESHOLD', '50.0')
    )
    LOW_PROFIT_DRAWBACK1 = float(
        os.getenv('LOW_PROFIT_DRAWBACK1', '20.0')
    )
    LOW_PROFIT_CLOSE1 = float(
        os.getenv('LOW_PROFIT_CLOSE1', '50')
    )

    # 【盈利回撤保护】最高盈利保护机制（安全网）
    # 触发条件：最高盈利曾经超过 BREAKEVEN_THRESHOLD，且当前盈利 <= BREAKEVEN_THRESHOLD
    #
    # 重要说明：
    #   - 这是一个"安全网"机制，防止盈利大回撤
    #   - 不需要之前触发过高盈利或低盈利止盈（无论之前是否部分平仓，都会全部平仓）
    #   - 最高盈利60%，当前盈利8%，已回撤52%，触发全部平仓
    #   - 最高盈利15%，当前盈利8%（未超过阈值10%），不触发
    #
    # 实际示例（当前配置：10%阈值）：
    #   场景1：最高盈利50% → 跌到8%（回撤42%）→ 触发100%平仓 ✓
    #   场景2：最高盈利21% → 跌到8%（回撤13%）→ 触发100%平仓 ✓
    #   场景3：最高盈利8%（从未超过10%）→ 跌到5% → 不触发 ✗
    _breakeven = os.getenv('15MTUPO_BREAKEVEN_THRESHOLD') or os.getenv('BREAKEVEN_THRESHOLD', '10.0')
    BREAKEVEN_THRESHOLD = float(_breakeven)
    
    # ==================== 重新进场参数 ====================
    REENTER_ENABLED = os.getenv('REENTER_ENABLED', 'true').lower() == 'true'  # 重新进场开关
    PROFIT_REENTER_THRESHOLD = float(os.getenv('PROFIT_REENTER_THRESHOLD', '5.0'))  # 重新进场阈值（%）
    REENTER_MIN_INTERVAL = int(float(os.getenv('REENTER_MIN_INTERVAL', '300')))  # 重新进场最小时间间隔（秒）

    # ==================== 止损参数（盈亏率包含杠杆倍数） ====================
    # 例如：20x杠杆下，价格跌1% → 亏损率=-1%×20=-20%
    # 所有盈亏率参数都是含杠杆的数值

    # 【三级止损】逐步平仓止损策略
    # 止损单在建仓完成后立即创建（使用币安API的止损限价单）
    #
    # 第一级止损：
    #   - 触发条件：当前盈亏率 <= STOPLOSS_TRIGGER1（例：-35%）
    #   - 平仓比例：STOPLOSS_CLOSE1%（例：100%，即全部平仓）
    #   - 说明：这是第一层防线，通常设置较宽松
    #
    # 第二级止损：
    #   - 触发条件：当前盈亏率 <= STOPLOSS_TRIGGER2（例：-40%）且第一级已触发
    #   - 平仓比例：STOPLOSS_CLOSE2%（例：50%，即剩余仓位的50%）
    #   - 说明：只在第一级止损触发后才会生效
    #
    # 第三级止损（最后一级，不为0时）：
    #   - 触发条件：当前盈亏率 <= STOPLOSS_TRIGGER3（例：-50%）
    #   - 平仓比例：100%（立即市价清空）
    #   - 说明：这是紧急市价止损，立即执行全部平仓，使用市价单而非限价单
    #
    # 【亏损回撤保护】与止盈保本逻辑一致
    # 如果STOPLOSS_TRIGGER3不为0，则使用与止盈相同的逻辑：
    #   - 触发条件：最高盈利曾经超过-brebreakeven，且当前盈亏率 <= abs(STOPLOSS_TRIGGER3)的反向值
    #   - 但实际止损还是按逐级执行
    #   - 这里与BREAKEVEN_THRESHOLD（止盈保本）对称：止盈保本防止盈利大回撤，止损防止亏损扩大
    # ==================== 止损参数 ====================
    # 先尝试读策略私有配置（15MTUPO_ 前缀），没有则用全局
    # 以下中间变量用于 env → class attr 的转换链，下划线前缀避免与类属性名冲突
    _stop_loss_trigger1 = os.getenv('15MTUPO_STOP_LOSS_PCT') or os.getenv('STOPLOSS_TRIGGER1', '-35.0')
    STOPLOSS_TRIGGER1 = float(_stop_loss_trigger1)
    _stop_loss_trigger2 = os.getenv('STOPLOSS_TRIGGER2', '-40.0')
    STOPLOSS_TRIGGER2 = float(_stop_loss_trigger2)
    STOPLOSS_CLOSE1 = float(os.getenv('STOPLOSS_CLOSE1', '100'))  # 第一级止损平仓比例（总仓位的%，例：100%）
    STOPLOSS_CLOSE2 = float(os.getenv('STOPLOSS_CLOSE2', '50'))  # 第二级止损平仓比例（剩余仓位的%，例：50%）
    STOPLOSS_TRIGGER3 = float(os.getenv('STOPLOSS_TRIGGER3', '-50.0'))  # 第三级止损触发点（含杠杆亏损率%，负数，例：-50%，不为0时生效）
    # STOPLOSS_LIMIT_OFFSET 已移除，使用 DELAY_RATIO 统一控制

# 黑天鹅防护参数
    BLACK_SWAN_DROP_THRESHOLD_1 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_1', '-10.0'))  # 第一级黑天鹅：60秒内下跌10%
    BLACK_SWAN_DROP_THRESHOLD_2 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_2', '-15.0'))  # 第二级黑天鹅：60秒内下跌15%
    BLACK_SWAN_DROP_THRESHOLD_3 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_3', '-20.0'))  # 第三级黑天鹅：60秒内下跌20%

    # 熔断参数
    EMERGENCY_DAILY_LOSS_PERCENT = float(os.getenv('EMERGENCY_DAILY_LOSS_PERCENT', '10.0'))  # 单日亏损超过10%触发熔断
    EMERGENCY_CONTINUOUS_LOSS = int(float(os.getenv('EMERGENCY_CONTINUOUS_LOSS', '3')))  # 连续亏损3次触发熔断
    EMERGENCY_LIQUIDATION_BUFFER = float(os.getenv('EMERGENCY_LIQUIDATION_BUFFER', '0.1'))  # 强平缓冲低于10%触发熔断
    EMERGENCY_PAUSE_COOLDOWN = int(float(os.getenv('EMERGENCY_PAUSE_COOLDOWN', '1800')))  # 熔断后冷却时间（秒）

    # 紧急平仓
    EMERGENCY_STOP_ENABLED = os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() == 'true'  # 是否启用紧急熔断
    EMERGENCY_CLOSE_ON_PAUSE = os.getenv('EMERGENCY_CLOSE_ON_PAUSE', 'true').lower() == 'true'  # 熔断时是否紧急平仓所有持仓

     # 回测参数
    BACKTEST_AUTO_SAVE = os.getenv('BACKTEST_AUTO_SAVE', 'true').lower() == 'true'  # 回测自动保存
    BACKTEST_PLOT_ENABLED = os.getenv('BACKTEST_PLOT_ENABLED', 'true').lower() == 'true'  # 回测图表功能
    BACKTEST_FORCED_CLOSE_INCLUDE_STATS = os.getenv('BACKTEST_FORCED_CLOSE_INCLUDE_STATS', 'false').lower() == 'true'  # 强制平仓是否计入统计

    # ==================== 技术指标过滤参数（入场信号优化） ====================
    # 基于历史交易数据分析结果：
    # - LONG最佳信号：RSI>70(+353%), BB>80%(+144%), 2h涨>3%+RSI>70(+602%)
    # - SHORT最佳信号：RSI<30(+136%), BB<20%(+26%), 2h跌>3%+RSI<30(+166%)
    # - 应避免：LONG时BB 40-60%(-54%), RSI 45-55(-29%)；SHORT时BB>80%(-63%)

    # 技术指标过滤开关
    BACKTEST_INDICATOR_FILTER_ENABLED = os.getenv('BACKTEST_INDICATOR_FILTER_ENABLED', 'false').lower() == 'true'  # 启用技术指标过滤

    # ==================== 回测专用参数（兼容性添加） ====================
    # 监控K线数量
    MONITOR_KLINES_COUNT = int(os.getenv('MONITOR_KLINES_COUNT', '100'))
    # ADX指标
    ADX_ENABLED = os.getenv('ADX_ENABLED', 'false').lower() == 'true'
    ADX_THRESHOLD = int(os.getenv('15MTUPO_ADX_THRESHOLD') or os.getenv('ADX_THRESHOLD', '20'))
    # 趋势过滤
    TREND_FILTER_ENABLED = os.getenv('TREND_FILTER_ENABLED', 'false').lower() == 'true'
    ENTRY_TREND_FILTER_ENABLED = os.getenv('ENTRY_TREND_FILTER_ENABLED', 'false').lower() == 'true'
    # 趋势延续
    TREND_CONTINUATION_ENABLED = os.getenv('TREND_CONTINUATION_ENABLED', 'false').lower() == 'true'
    TREND_CONTINUATION_CHECK_INTERVAL = int(os.getenv('TREND_CONTINUATION_CHECK_INTERVAL', '60'))
    TREND_CONTINUATION_LOOKBACK = int(os.getenv('TREND_CONTINUATION_LOOKBACK', '20'))
    TREND_CONTINUATION_MIN_PROFIT = float(os.getenv('TREND_CONTINUATION_MIN_PROFIT', '5.0'))
    # 建仓回撤容忍度
    BUILDING_DRAWDOWN_TOLERANCE = float(os.getenv('BUILDING_DRAWDOWN_TOLERANCE', '10.0'))

    # 做多(LONG)过滤条件
    LONG_FILTER_RSI_MIN = float(os.getenv('LONG_FILTER_RSI_MIN', '55.0'))  # RSI最小值（避免45-55中性区）
    LONG_FILTER_BB_MIN = float(os.getenv('LONG_FILTER_BB_MIN', '0.45'))  # BB位置最小值（避免40-60%中性区）

    # 做空(SHORT)过滤条件
    SHORT_FILTER_RSI_MAX = float(os.getenv('SHORT_FILTER_RSI_MAX', '50.0'))  # RSI最大值
    SHORT_FILTER_BB_MAX = float(os.getenv('SHORT_FILTER_BB_MAX', '0.60'))  # BB位置最大值（避免>80%）

    # 波动率过滤
    ATR_MAX_PCT = float(os.getenv('ATR_MAX_PCT', '7.0'))  # ATR%最大值（避免高波动）

    # 多周期趋势过滤
    TREND_FILTER_2H_MIN = float(os.getenv('TREND_FILTER_2H_MIN', '-3.0'))  # 2h最小涨幅%（LONG时避免大跌）
    TREND_FILTER_4H_MIN = float(os.getenv('TREND_FILTER_4H_MIN', '-2.0'))  # 4h最小涨幅%（SHORT条件）

    # ==================== V23策略参数 ====================
    # 趋势判断参数（由 reload 路径统一处理 15MTUPO_* 前缀）
    
    # 分批建仓参数
    ADD_POSITION_RATIO = float(os.getenv('ADD_POSITION_RATIO', '50'))  # 加仓比例%
    ADD_POSITION_BARS_LIMIT = int(os.getenv('ADD_POSITION_BARS_LIMIT', '10'))  # 加仓等待K线数
    RESERVE_FOR_ADD_POSITION = os.getenv('RESERVE_FOR_ADD_POSITION', 'true').lower() == 'true'

    # 交易成本
    TRADING_FEE_RATE = float(os.getenv('TRADING_FEE_RATE', '0.0004'))  # 手续费率 0.04%
    SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', '0.001'))  # 滑点 0.1%

    # 实时判断参数
    REALTIME_JUDGE_ENABLED = os.getenv('REALTIME_JUDGE_ENABLED', 'true').lower() == 'true'
    MIN_TICK_COUNT_FOR_JUDGE = int(os.getenv('MIN_TICK_COUNT_FOR_JUDGE', '5'))
    MIN_BREAKOUT_PCT = float(os.getenv('MIN_BREAKOUT_PCT', '0.3'))

    # ATR止损参数
    STOP_LOSS_ATR_MULTIPLIER = float(os.getenv('STOP_LOSS_ATR_MULTIPLIER', '2.0'))
    USE_DYNAMIC_STOP_LOSS = os.getenv('USE_DYNAMIC_STOP_LOSS', 'true').lower() == 'true'

    # 性能参数
    MAX_SYMBOLS_MONITOR = int(os.getenv('MAX_SYMBOLS_MONITOR', '50'))
    TICK_QUEUE_SIZE = int(os.getenv('TICK_QUEUE_SIZE', '1000'))
    BATCH_PROCESS_SIZE = int(os.getenv('BATCH_PROCESS_SIZE', '10'))
    ASYNC_WORKERS = int(os.getenv('ASYNC_WORKERS', '4'))

    # 动态阈值
    MIN_PRICE_CHANGE_5M = float(os.getenv('MIN_PRICE_CHANGE_5M', '0.5'))
    MIN_PRICE_CHANGE_15M = float(os.getenv('MIN_PRICE_CHANGE_15M', '1.0'))
    MIN_PRICE_CHANGE_1H = float(os.getenv('MIN_PRICE_CHANGE_1H', '2.0'))
    MIN_PRICE_CHANGE_4H = float(os.getenv('MIN_PRICE_CHANGE_4H', '3.0'))

    # 杠杆范围
    MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', '20'))
    MIN_LEVERAGE = int(os.getenv('MIN_LEVERAGE', '1'))
    MAX_LOSS_PCT = float(os.getenv('MAX_LOSS_PCT', '100'))  # 硬止损100%

    # 持仓管理
    MAX_SINGLE_POSITION_PCT = float(os.getenv('MAX_SINGLE_POSITION_PCT', '20'))
    MAX_TOTAL_POSITION_PCT = float(os.getenv('MAX_TOTAL_POSITION_PCT', '80'))
    # ==================== V23策略参数结束 ====================

    # 其他系统参数
    MAX_RETRIES = int(float(os.getenv('MAX_RETRIES', '3')))  # 全局重试次数
    RETRY_DELAY = int(float(os.getenv('RETRY_DELAY', '1')))  # 重试间隔（秒）
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # 日志等级

    # API平台返回参数（用于server API）
    CHART_ENABLED = os.getenv('CHART_ENABLED', 'true').lower() == 'true'  # 图表功能

    @classmethod
    def validate(cls):
        """验证关键配置项是否合理，检测环境变量冲突"""
        errors = []
        warnings = []

        # 基础配置检查
        if not cls.BINANCE_API_KEY or not cls.BINANCE_API_SECRET:
            errors.append("未配置币安实盘 API 密钥")
        if not cls.TESTNET_API_KEY or not cls.TESTNET_API_SECRET:
            # 测试网可选，若未配置请在实盘模式启用时确保有效密钥
            pass
        if not cls.TELEGRAM_BOT_TOKEN_ALERT or not cls.TELEGRAM_CHAT_ID:
            errors.append("未配置 Telegram 警报机器人 Token 或 Chat ID")
        if cls.MONITOR_INTERVAL <= 0:
            errors.append("MONITOR_INTERVAL 必须大于 0")
        if cls.MONITOR_SYMBOLS_COUNT <= 0:
            errors.append("MONITOR_SYMBOLS_COUNT 必须大于 0")

        # ==================== 止盈参数冲突检测 ====================
        # 止盈阈值应该是：HIGH > LOW > BREAKEVEN（数值递减）
        if cls.HIGH_PROFIT_THRESHOLD <= cls.LOW_PROFIT_THRESHOLD:
            errors.append(f"Take Profit Threshold Conflict: HIGH_PROFIT_THRESHOLD({cls.HIGH_PROFIT_THRESHOLD}%) must be > LOW_PROFIT_THRESHOLD({cls.LOW_PROFIT_THRESHOLD}%)")
        if cls.LOW_PROFIT_THRESHOLD <= cls.BREAKEVEN_THRESHOLD:
            errors.append(f"Take Profit Threshold Conflict: LOW_PROFIT_THRESHOLD({cls.LOW_PROFIT_THRESHOLD}%) must be > BREAKEVEN_THRESHOLD({cls.BREAKEVEN_THRESHOLD}%)")
        if cls.BREAKEVEN_THRESHOLD <= 0:
            # BREAKEVEN_THRESHOLD should be positive (profit threshold)
            warnings.append(f"BREAKEVEN_THRESHOLD({cls.BREAKEVEN_THRESHOLD}%) should be > 0, negative means loss (handled by stop loss)")

        # Check take profit close percentage ranges
        if not (0 < cls.HIGH_PROFIT_CLOSE1 <= 100):
            errors.append(f"HIGH_PROFIT_CLOSE1({cls.HIGH_PROFIT_CLOSE1}%) must be in (0, 100]")
        if not (0 < cls.HIGH_PROFIT_CLOSE2 <= 100):
            errors.append(f"HIGH_PROFIT_CLOSE2({cls.HIGH_PROFIT_CLOSE2}%) must be in (0, 100]")
        if not (0 < cls.LOW_PROFIT_CLOSE1 <= 100):
            errors.append(f"LOW_PROFIT_CLOSE1({cls.LOW_PROFIT_CLOSE1}%) must be in (0, 100]")

        # Check take profit setback thresholds (must be positive)
        if cls.HIGH_PROFIT_DRAWBACK1 <= 0:
            errors.append(f"HIGH_PROFIT_DRAWBACK1({cls.HIGH_PROFIT_DRAWBACK1}%) must be > 0")
        if cls.HIGH_PROFIT_DRAWBACK2 <= 0:
            errors.append(f"HIGH_PROFIT_DRAWBACK2({cls.HIGH_PROFIT_DRAWBACK2}%) must be > 0")
        if cls.LOW_PROFIT_DRAWBACK1 <= 0:
            errors.append(f"LOW_PROFIT_DRAWBACK1({cls.LOW_PROFIT_DRAWBACK1}%) must be > 0")

        # Check total take profit close percentage (avoid exceeding 100%)
        # HIGH_PROFIT_CLOSE1 is % of total position, HIGH_PROFIT_CLOSE2 is % of remaining position
        # Max possible close = CLOSE1 + (100-CLOSE1) * CLOSE2 / 100
        max_high_profit_close = cls.HIGH_PROFIT_CLOSE1 + (100 - cls.HIGH_PROFIT_CLOSE1) * cls.HIGH_PROFIT_CLOSE2 / 100
        if max_high_profit_close > 100:
            warnings.append(f"High profit total close percentage({max_high_profit_close:.1f}%) > 100%, may exceed full position")

        # ==================== 止损参数冲突检测 ====================
        # 止损阈值应该是：TRIGGER1 > TRIGGER2 > TRIGGER3（都是负数，数值越大越接近0，如-10% > -20% > -30%）
        # 例如：-35% > -40% > -50% 是正确的顺序（亏损逐渐增加）
        if cls.STOPLOSS_TRIGGER1 <= cls.STOPLOSS_TRIGGER2:
            errors.append(f"Stop Loss Trigger Conflict: STOPLOSS_TRIGGER1({cls.STOPLOSS_TRIGGER1}%) must be > STOPLOSS_TRIGGER2({cls.STOPLOSS_TRIGGER2}%) (e.g., -35% > -40%)")
        if cls.STOPLOSS_TRIGGER2 <= cls.STOPLOSS_TRIGGER3:
            errors.append(f"Stop Loss Trigger Conflict: STOPLOSS_TRIGGER2({cls.STOPLOSS_TRIGGER2}%) must be > STOPLOSS_TRIGGER3({cls.STOPLOSS_TRIGGER3}%) (e.g., -40% > -50%)")

        # Stop loss triggers should be negative
        if cls.STOPLOSS_TRIGGER1 >= 0:
            errors.append(f"STOPLOSS_TRIGGER1({cls.STOPLOSS_TRIGGER1}%) must be < 0, positive means profit (handled by take profit)")
        if cls.STOPLOSS_TRIGGER2 >= 0:
            errors.append(f"STOPLOSS_TRIGGER2({cls.STOPLOSS_TRIGGER2}%) must be < 0")
        if cls.STOPLOSS_TRIGGER3 >= 0:
            errors.append(f"STOPLOSS_TRIGGER3({cls.STOPLOSS_TRIGGER3}%) must be < 0")

        # Check stop loss close percentage ranges
        if not (0 < cls.STOPLOSS_CLOSE1 <= 100):
            errors.append(f"STOPLOSS_CLOSE1({cls.STOPLOSS_CLOSE1}%) must be in (0, 100]")
        if not (0 < cls.STOPLOSS_CLOSE2 <= 100):
            errors.append(f"STOPLOSS_CLOSE2({cls.STOPLOSS_CLOSE2}%) must be in (0, 100]")

        # Check total stop loss close percentage (avoid exceeding 100%)
        # STOPLOSS_CLOSE2 is % of remaining position
        max_stop_loss_close = cls.STOPLOSS_CLOSE1 + (100 - cls.STOPLOSS_CLOSE1) * cls.STOPLOSS_CLOSE2 / 100
        if max_stop_loss_close > 100:
            warnings.append(f"Stop loss total close percentage({max_stop_loss_close:.1f}%) > 100%, may exceed full position")

        # ==================== 交易参数冲突检测 ====================
        if cls.LEVERAGE <= 0:
            errors.append(f"LEVERAGE({cls.LEVERAGE}) must be > 0")
        if cls.INITIAL_POSITION <= 0 or cls.INITIAL_POSITION > 100:
            errors.append(f"INITIAL_POSITION({cls.INITIAL_POSITION}%) must be in (0, 100]")

        # Warning: high leverage with high initial position
        if cls.LEVERAGE >= 20 and cls.INITIAL_POSITION > 50:
            warnings.append(f"High leverage({cls.LEVERAGE}x) with INITIAL_POSITION({cls.INITIAL_POSITION}%) may be risky,建议降低到50%以下")

        # Position quantity limits
        if cls.MAX_POSITIONS <= 0:
            errors.append(f"MAX_POSITIONS({cls.MAX_POSITIONS}) must be > 0")
        if cls.MAX_POSITIONS_PER_SYMBOL <= 0:
            errors.append(f"MAX_POSITIONS_PER_SYMBOL({cls.MAX_POSITIONS_PER_SYMBOL}) must be > 0")

        # ==================== 加仓参数冲突检测 ====================
        # Check position building thresholds
        # 注意：0 表示禁用该级别，不应报错
        loss_steps = [cls.LOSS_STEP1, cls.LOSS_STEP2, cls.LOSS_STEP3]
        for i, step in enumerate(loss_steps):
            # 跳过0值（表示禁用该级别）
            if step == 0:
                continue
            if step > 0:
                # Loss building threshold should be negative (or 0 to disable)
                errors.append(f"LOSS_STEP{i+1}({step}%) represents loss building, must be < 0, use PROFIT_STEP for positive values")

        profit_steps = [cls.PROFIT_STEP1, cls.PROFIT_STEP2, cls.PROFIT_STEP3]
        for i, step in enumerate(profit_steps):
            # 跳过0值（表示禁用该级别）
            if step == 0:
                continue
            if step < 0:
                # Profit building threshold should be positive (or 0 to disable)
                errors.append(f"PROFIT_STEP{i+1}({step}%) represents profit building, must be > 0, use LOSS_STEP for negative values")

        # 输出警告
        if warnings:
            print("[settings] 配置警告:")
            for warning in warnings:
                print(f"  [WARNING] {warning}")

        # 抛出错误（如果有）
        if errors:
            error_messages = "\n".join([f"[ERROR] {error}" for error in errors])
            raise ValueError(error_messages)

        print("[settings] 配置验证通过，没有发现冲突")
        return True

    @classmethod
    def _get_latest_env_mtime(cls) -> float:
        """获取所有受监控 .env 文件的最新修改时间"""
        mtimes = []
        if env_path.exists():
            mtimes.append(os.path.getmtime(env_path))
        if hf_env.exists():
            mtimes.append(os.path.getmtime(hf_env))
        if tupo_env.exists():
            mtimes.append(os.path.getmtime(tupo_env))
        return max(mtimes) if mtimes else 0.0

    @classmethod
    def _reload_env_vars(cls):
        """重新加载环境变量到类属性（包括策略私有配置）"""
        with cls._reload_lock:  # 使用锁保护热重载过程
            # 加载主配置（覆盖已有以便获取最新值）
            load_dotenv(env_path, override=True)
            
            # 加载策略私有配置
            if hf_env.exists():
                load_dotenv(hf_env, override=True)
            if tupo_env.exists():
                load_dotenv(tupo_env, override=True)
            
            cls._last_modified = cls._get_latest_env_mtime()

            # 重新设置所有从环境变量读取的属性
            # API 配置
            cls.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
            cls.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
            cls.TESTNET_API_KEY = os.getenv('TESTNET_API_KEY', '')
            cls.TESTNET_API_SECRET = os.getenv('TESTNET_API_SECRET', '')

            # Telegram 配置
            cls.TELEGRAM_BOT_TOKEN_ALERT = os.getenv('TELEGRAM_BOT_TOKEN_ALERT', '')
            cls.TELEGRAM_BOT_TOKEN_TRADE = os.getenv('TELEGRAM_BOT_TOKEN_TRADE', '')
            cls.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

            # 警报参数（使用安全转换）
            cls.MONITOR_INTERVAL = safe_int(os.getenv('MONITOR_INTERVAL', '120'), 1)
            cls.MONITOR_SYMBOLS_COUNT = safe_int(os.getenv('MONITOR_SYMBOLS_COUNT', '100'), 1, min_val=1)
            cls.SYMBOLS_UPDATE_INTERVAL = safe_int(os.getenv('SYMBOLS_UPDATE_INTERVAL', '60'), 1)
            cls.PRICE_CHANGE_THRESHOLD = safe_float(os.getenv('PRICE_CHANGE_THRESHOLD', '3.0'), 0.0, min_val=0.0)
            cls.VOLUME_THRESHOLD = safe_float(os.getenv('VOLUME_THRESHOLD', '1.5'), 0.0, min_val=0.0)
            cls.VOLUME_COMPARE_PERIODS = safe_int(os.getenv('VOLUME_COMPARE_PERIODS', '10'), 1, min_val=1)
            
            # BNFF 策略参数（使用安全转换）
            cls.SHORT_DROP_COUNT = safe_int(os.getenv('SHORT_DROP_COUNT', '5'), 0, min_val=0)
            cls.SHORT_DROP_THRESHOLD = safe_float(os.getenv('SHORT_DROP_THRESHOLD', '3.0'), 0.0, min_val=0.0)
            cls.LONG_BOUNCE_COUNT = safe_int(os.getenv('LONG_BOUNCE_COUNT', '3'), 0, min_val=0)
            cls.LONG_BOUNCE_THRESHOLD = safe_float(os.getenv('LONG_BOUNCE_THRESHOLD', '0.2'), 0.0, min_val=0.0)
            cls.LONG_SURGE_THRESHOLD = safe_float(os.getenv('LONG_SURGE_THRESHOLD', '5.0'), 0.0, min_val=0.0)
            cls.ALERT_COOLDOWN = safe_int(os.getenv('ALERT_COOLDOWN', '60'), 0, min_val=0)
            cls.EXIT_CLOSE_ALL_POSITIONS = os.getenv('EXIT_CLOSE_ALL_POSITIONS', 'true').lower() == 'true'
            cls.OPEN_INTEREST_MONITOR_ENABLED = os.getenv('OPEN_INTEREST_MONITOR_ENABLED', 'false').lower() == 'true'
            cls.OPEN_INTEREST_INCREASE_THRESHOLD = safe_float(os.getenv('OPEN_INTEREST_INCREASE_THRESHOLD', '5.0'), 0.0, min_val=0.0)
            cls.LONG_BOUNCE_OI_REQUIRED = os.getenv('LONG_BOUNCE_OI_REQUIRED', 'true').lower() == 'true'
            cls.SHORT_DROP_OI_REQUIRED = os.getenv('SHORT_DROP_OI_REQUIRED', 'false').lower() == 'true'
            cls.SHORT_DROP_VOLUME_REQUIRED = os.getenv('SHORT_DROP_VOLUME_REQUIRED', 'false').lower() == 'true'
            cls.SHORT_DROP_VOLUME_THRESHOLD = float(os.getenv('SHORT_DROP_VOLUME_THRESHOLD', '1.5'))
            cls.LONG_BOUNCE_VOLUME_REQUIRED = os.getenv('LONG_BOUNCE_VOLUME_REQUIRED', 'true').lower() == 'true'
            cls.LONG_BOUNCE_VOLUME_THRESHOLD = float(os.getenv('LONG_BOUNCE_VOLUME_THRESHOLD', '1.5'))
            cls.LONG_SURGE_VOLUME_REQUIRED = os.getenv('LONG_SURGE_VOLUME_REQUIRED', 'true').lower() == 'true'
            cls.LONG_SURGE_VOLUME_THRESHOLD = float(os.getenv('LONG_SURGE_VOLUME_THRESHOLD', '2.0'))
            
            # 策略开关
            _tmp_str_enabled = os.getenv('15MTUPO_ENABLED') or os.getenv('V23_ENABLED') or 'true'
            cls.STRAT15_ENABLED = _tmp_str_enabled.lower() == 'true'
            cls.V23_ENABLED = cls.STRAT15_ENABLED
            cls.HF_ENABLED = os.getenv('HF_ENABLED', 'false').lower() == 'true'
            cls.STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'SEPARATE')
            
            # V23策略参数
            cls.V23_MAX_POSITIONS = int(float(os.getenv('V23_MAX_POSITIONS', '5')))
            cls.V23_SINGLE_SYMBOL_MAX = float(os.getenv('V23_SINGLE_SYMBOL_MAX', '5000'))
            cls.V23_INITIAL_POSITION = float(os.getenv('V23_INITIAL_POSITION', '15'))
            
            # V23信号参数
            cls.V23_SIGNAL_VOLUME_THRESHOLD = float(os.getenv('V23_SIGNAL_VOLUME_THRESHOLD', '1.5'))
            cls.V23_SIGNAL_BODY_RATIO = float(os.getenv('V23_SIGNAL_BODY_RATIO', '0.5'))
            cls.V23_SIGNAL_CLOSE_POSITION = float(os.getenv('V23_SIGNAL_CLOSE_POSITION', '0.6'))
            cls.V23_SIGNAL_BREAKOUT_THRESHOLD = float(os.getenv('V23_SIGNAL_BREAKOUT_THRESHOLD', '0.3'))
            cls.V23_TREND_PRICE_CHANGE = float(os.getenv('V23_TREND_PRICE_CHANGE', '2.0'))
            
            # 监控币种数量
            cls.V23_MONITOR_SYMBOLS = int(float(os.getenv('V23_MONITOR_SYMBOLS', '50')))
            cls.V23_LOAD_KLINES_COUNT = int(float(os.getenv('V23_LOAD_KLINES_COUNT', '0')))
            
            # 数据下载参数
            cls.MIN_HISTORY_BARS = int(float(os.getenv('MIN_HISTORY_BARS', '1500')))
            cls.DOWNLOAD_DAYS = int(float(os.getenv('DOWNLOAD_DAYS', '90')))
            cls.AUTO_DOWNLOAD_ON_START = os.getenv('AUTO_DOWNLOAD_ON_START', 'true').lower() == 'true'
            
            # 组合模式资金分配
            cls.V23_CAPITAL_RATIO = float(os.getenv('V23_CAPITAL_RATIO', '70'))
            cls.HF_CAPITAL_RATIO = float(os.getenv('HF_CAPITAL_RATIO', '30'))
            cls.LONG_SURGE_OI_REQUIRED = os.getenv('LONG_SURGE_OI_REQUIRED', 'true').lower() == 'true'

            # 持仓监控参数
            cls.POSITION_MONITOR_SLEEP_TIME = int(float(os.getenv('POSITION_MONITOR_SLEEP_TIME', '15')))
            
            # 持仓同步参数
            cls.POSITION_SYNC_INTERVAL = int(os.getenv('POSITION_SYNC_INTERVAL', '10'))

            # 交易参数（使用安全转换）
            cls.MARGIN_MODE = os.getenv('MARGIN_MODE', 'ISOLATED')
            cls.LEVERAGE = safe_int(os.getenv('LEVERAGE', '20'), 1, min_val=1, max_val=125)
            cls.MAX_POSITIONS = safe_int(os.getenv('MAX_POSITIONS', '5'), 1, min_val=1)
            cls.MAX_POSITIONS_PER_SYMBOL = safe_int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3'), 1, min_val=1)
            cls.SINGLE_SYMBOL_MAX_INVESTMENT = safe_float(os.getenv('SINGLE_SYMBOL_MAX_INVESTMENT', '5000'), 0.0, min_val=0.0)
            cls.POSITION_ALLOCATION_MODE = os.getenv('POSITION_ALLOCATION_MODE', 'EQUAL')
            cls.ORDER_TYPE = os.getenv('ORDER_TYPE', 'MARKET')
            cls.INITIAL_POSITION = safe_float(os.getenv('INITIAL_POSITION', '15'), 0.0, min_val=0.0, max_val=100.0)
            cls.DELAY_RATIO = safe_float(os.getenv('DELAY_RATIO', '0.003'), 0.0, min_val=0.0)
            cls.MAX_MARGIN_PER_SYMBOL = safe_float(os.getenv('MAX_MARGIN_PER_SYMBOL', '25000'), 0.0, min_val=0.0)
            cls.ORDER_LIMIT_TIMEOUT = safe_int(os.getenv('ORDER_LIMIT_TIMEOUT', '5'), 1, min_val=1)

            # 订单类型判断参数
            cls.ORDER_TYPE_MARKET_THRESHOLD = safe_float(os.getenv('ORDER_TYPE_MARKET_THRESHOLD', '1000'), 0.0, min_val=0.0)
            cls.ORDER_TYPE_LOW_VOLUME_THRESHOLD = safe_float(os.getenv('ORDER_TYPE_LOW_VOLUME_THRESHOLD', '100000'), 0.0, min_val=0.0)
            cls.ORDER_TYPE_LIMIT_PRICE_BUY_BIAS = safe_float(os.getenv('ORDER_TYPE_LIMIT_PRICE_BUY_BIAS', '1.001'), 1.0, min_val=1.0)
            cls.ORDER_TYPE_LIMIT_PRICE_SELL_BIAS = safe_float(os.getenv('ORDER_TYPE_LIMIT_PRICE_SELL_BIAS', '0.999'), 0.0, min_val=0.0)

            # 限价单转市价阈值
            cls.LIMIT_TO_MARKET_PRICE_DEVIATION_THRESHOLD = safe_float(os.getenv('LIMIT_TO_MARKET_PRICE_DEVIATION_THRESHOLD', '0.5'), 0.0, min_val=0.0)
            cls.LIMIT_TO_MARKET_TIME_THRESHOLD = safe_float(os.getenv('LIMIT_TO_MARKET_TIME_THRESHOLD', '8'), 0.0, min_val=0.0)
            cls.LIMIT_TO_MARKET_PNL_DECLINE_THRESHOLD = safe_float(os.getenv('LIMIT_TO_MARKET_PNL_DECLINE_THRESHOLD', '2.0'), 0.0, min_val=0.0)

            # 亏损加仓参数
            cls.LOSS_STEP1 = safe_float(os.getenv('LOSS_STEP1', '-2.0'), 0.0, min_val=-100.0, max_val=0.0)
            cls.LOSS_ADD1 = safe_float(os.getenv('LOSS_ADD1', '15'), 0.0, min_val=0.0, max_val=100.0)
            cls.LOSS_STEP2 = safe_float(os.getenv('LOSS_STEP2', '-4.0'), 0.0, min_val=-100.0, max_val=0.0)
            cls.LOSS_ADD2 = safe_float(os.getenv('LOSS_ADD2', '25'), 0.0, min_val=0.0, max_val=100.0)
            cls.LOSS_STEP3 = safe_float(os.getenv('LOSS_STEP3', '-6.0'), 0.0, min_val=-100.0, max_val=0.0)
            cls.LOSS_ADD3 = safe_float(os.getenv('LOSS_ADD3', '30'), 0.0, min_val=0.0, max_val=100.0)

            # 盈利加仓参数
            cls.PROFIT_STEP1 = safe_float(os.getenv('PROFIT_STEP1', '2.0'), 0.0, min_val=0.0)
            cls.PROFIT_ADD1 = safe_float(os.getenv('PROFIT_ADD1', '10'), 0.0, min_val=0.0, max_val=100.0)
            cls.PROFIT_STEP2 = safe_float(os.getenv('PROFIT_STEP2', '4.0'), 0.0, min_val=0.0)
            cls.PROFIT_ADD2 = safe_float(os.getenv('PROFIT_ADD2', '15'), 0.0, min_val=0.0, max_val=100.0)
            cls.PROFIT_STEP3 = safe_float(os.getenv('PROFIT_STEP3', '6.0'), 0.0, min_val=0.0)
            cls.PROFIT_ADD3 = safe_float(os.getenv('PROFIT_ADD3', '20'), 0.0, min_val=0.0, max_val=100.0)

            # 止盈参数
            cls.HIGH_PROFIT_THRESHOLD = safe_float(os.getenv('HIGH_PROFIT_THRESHOLD', '60.0'), 0.0, min_val=0.0)
            cls.HIGH_PROFIT_CLOSE1 = safe_float(os.getenv('15MTUPO_TP_HIGH_CLOSE_PCT') or os.getenv('HIGH_PROFIT_CLOSE1', '50.0'), 0.0, min_val=0.0, max_val=100.0)
            cls.HIGH_PROFIT_DRAWBACK1 = safe_float(os.getenv('15MTUPO_DRAWDOWN_TIER1_PCT') or os.getenv('HIGH_PROFIT_DRAWBACK1', '20.0'), 0.0, min_val=0.0)
            cls.HIGH_PROFIT_CLOSE2 = safe_float(os.getenv('HIGH_PROFIT_CLOSE2', '50'), 0.0, min_val=0.0, max_val=100.0)
            cls.HIGH_PROFIT_DRAWBACK2 = safe_float(os.getenv('HIGH_PROFIT_DRAWBACK2', '20.0'), 0.0, min_val=0.0)
            cls.BREAKEVEN_THRESHOLD = safe_float(os.getenv('15MTUPO_BREAKEVEN_THRESHOLD') or os.getenv('BREAKEVEN_THRESHOLD', '10.0'), 0.0, min_val=0.0)
            
            # 低盈利止盈参数
            cls.LOW_PROFIT_THRESHOLD = safe_float(os.getenv('LOW_PROFIT_THRESHOLD', '50.0'), 0.0, min_val=0.0)
            cls.LOW_PROFIT_DRAWBACK1 = safe_float(os.getenv('LOW_PROFIT_DRAWBACK1', '20.0'), 0.0, min_val=0.0)
            cls.LOW_PROFIT_CLOSE1 = safe_float(os.getenv('LOW_PROFIT_CLOSE1', '50'), 0.0, min_val=0.0, max_val=100.0)
            
            # 重新进场参数
            cls.REENTER_ENABLED = os.getenv('REENTER_ENABLED', 'true').lower() == 'true'
            cls.PROFIT_REENTER_THRESHOLD = float(os.getenv('PROFIT_REENTER_THRESHOLD', '5.0'))
            cls.REENTER_MIN_INTERVAL = int(float(os.getenv('REENTER_MIN_INTERVAL', '300')))

            # 止损参数
            cls.STOPLOSS_TRIGGER1 = safe_float(os.getenv('15MTUPO_STOP_LOSS_PCT') or os.getenv('STOPLOSS_TRIGGER1', '-35.0'), -100.0, min_val=-100.0, max_val=0.0)
            cls.STOPLOSS_CLOSE1 = safe_float(os.getenv('STOPLOSS_CLOSE1', '100'), 0.0, min_val=0.0, max_val=100.0)
            cls.STOPLOSS_TRIGGER2 = safe_float(os.getenv('STOPLOSS_TRIGGER2', '-40.0'), -100.0, min_val=-100.0, max_val=0.0)
            cls.STOPLOSS_CLOSE2 = safe_float(os.getenv('STOPLOSS_CLOSE2', '50'), 0.0, min_val=0.0, max_val=100.0)
            cls.STOPLOSS_TRIGGER3 = safe_float(os.getenv('STOPLOSS_TRIGGER3', '-50.0'), -100.0, min_val=-100.0, max_val=0.0)

            # 建仓参数
            cls.POSITION_COMPLETE_PROFIT_RISE = safe_float(os.getenv('POSITION_COMPLETE_PROFIT_RISE', '10.0'), 0.0, min_val=0.0)
            cls.POSITION_COMPLETE_STEP_LOSS = safe_float(os.getenv('POSITION_COMPLETE_STEP_LOSS', '-10.0'), 0.0, min_val=-100.0, max_val=0.0)

            # 黑天鹅防护参数
            cls.BLACK_SWAN_DROP_THRESHOLD_1 = safe_float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_1', '-10.0'), 0.0, min_val=-100.0, max_val=0.0)
            cls.BLACK_SWAN_DROP_THRESHOLD_2 = safe_float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_2', '-15.0'), 0.0, min_val=-100.0, max_val=0.0)
            cls.BLACK_SWAN_DROP_THRESHOLD_3 = safe_float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_3', '-20.0'), 0.0, min_val=-100.0, max_val=0.0)

            # 15mTupo策略参数（使用15MTUPO_前缀匹配.env文件）
            cls.TREND_PERIOD = safe_int(os.getenv('15MTUPO_TREND_PERIOD', os.getenv('TREND_PERIOD', '16')), 1, min_val=1)
            cls.ADX_THRESHOLD = safe_int(os.getenv('15MTUPO_ADX_THRESHOLD', os.getenv('ADX_THRESHOLD', '20')), 1, min_val=1)
            cls.CONSOLIDATION_PERIOD = safe_int(os.getenv('15MTUPO_CONSOLIDATION_PERIOD', os.getenv('CONSOLIDATION_PERIOD', '40')), 1, min_val=1)
            cls.SIGNAL_VOLUME_THRESHOLD = safe_float(os.getenv('15MTUPO_SIGNAL_VOLUME_THRESHOLD', os.getenv('SIGNAL_VOLUME_THRESHOLD', '1.5')), 0.0, min_val=0.0)
            cls.SIGNAL_BODY_RATIO = safe_float(os.getenv('15MTUPO_SIGNAL_BODY_RATIO', os.getenv('SIGNAL_BODY_RATIO', '0.5')), 0.0, min_val=0.0, max_val=1.0)
            cls.SIGNAL_CLOSE_POSITION = safe_float(os.getenv('15MTUPO_SIGNAL_CLOSE_POSITION', os.getenv('SIGNAL_CLOSE_POSITION', '0.6')), 0.0, min_val=0.0, max_val=1.0)
            cls.SIGNAL_BREAKOUT_THRESHOLD = safe_float(os.getenv('15MTUPO_SIGNAL_BREAKOUT_THRESHOLD', os.getenv('SIGNAL_BREAKOUT_THRESHOLD', '0.3')), 0.0, min_val=0.0)
            cls.TREND_PRICE_CHANGE = safe_float(os.getenv('15MTUPO_TREND_PRICE_CHANGE', os.getenv('TREND_PRICE_CHANGE', '2.0')), 0.0, min_val=0.0)

            # 分批建仓参数
            cls.ADD_POSITION_RATIO = float(os.getenv('ADD_POSITION_RATIO', '50'))
            cls.ADD_POSITION_BARS_LIMIT = int(os.getenv('ADD_POSITION_BARS_LIMIT', '10'))
            cls.RESERVE_FOR_ADD_POSITION = os.getenv('RESERVE_FOR_ADD_POSITION', 'true').lower() == 'true'

            # 交易成本
            cls.TRADING_FEE_RATE = float(os.getenv('TRADING_FEE_RATE', '0.0004'))
            cls.SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', '0.001'))

            # 实时判断参数
            cls.REALTIME_JUDGE_ENABLED = os.getenv('REALTIME_JUDGE_ENABLED', 'true').lower() == 'true'
            cls.MIN_TICK_COUNT_FOR_JUDGE = int(os.getenv('MIN_TICK_COUNT_FOR_JUDGE', '5'))
            cls.MIN_BREAKOUT_PCT = float(os.getenv('MIN_BREAKOUT_PCT', '0.3'))

            # ATR止损参数
            cls.STOP_LOSS_ATR_MULTIPLIER = float(os.getenv('STOP_LOSS_ATR_MULTIPLIER', '2.0'))
            cls.USE_DYNAMIC_STOP_LOSS = os.getenv('USE_DYNAMIC_STOP_LOSS', 'true').lower() == 'true'

            # 性能参数
            cls.MAX_SYMBOLS_MONITOR = int(os.getenv('MAX_SYMBOLS_MONITOR', '50'))
            cls.TICK_QUEUE_SIZE = int(os.getenv('TICK_QUEUE_SIZE', '1000'))
            cls.BATCH_PROCESS_SIZE = int(os.getenv('BATCH_PROCESS_SIZE', '10'))
            cls.ASYNC_WORKERS = int(os.getenv('ASYNC_WORKERS', '4'))

            # 动态阈值
            cls.MIN_PRICE_CHANGE_5M = float(os.getenv('MIN_PRICE_CHANGE_5M', '0.5'))
            cls.MIN_PRICE_CHANGE_15M = float(os.getenv('MIN_PRICE_CHANGE_15M', '1.0'))
            cls.MIN_PRICE_CHANGE_1H = float(os.getenv('MIN_PRICE_CHANGE_1H', '2.0'))
            cls.MIN_PRICE_CHANGE_4H = float(os.getenv('MIN_PRICE_CHANGE_4H', '3.0'))

            # 杠杆范围（使用安全转换）
            cls.MAX_LEVERAGE = safe_int(os.getenv('MAX_LEVERAGE', '20'), 1, min_val=1, max_val=125)
            cls.MIN_LEVERAGE = safe_int(os.getenv('MIN_LEVERAGE', '1'), 1, min_val=1, max_val=125)
            cls.MAX_LOSS_PCT = safe_float(os.getenv('MAX_LOSS_PCT', '100'), 0.0, min_val=0.0, max_val=100.0)

            # 持仓管理（使用安全转换）
            cls.MAX_SINGLE_POSITION_PCT = safe_float(os.getenv('MAX_SINGLE_POSITION_PCT', '20'), 0.0, min_val=0.0, max_val=100.0)
            cls.MAX_TOTAL_POSITION_PCT = safe_float(os.getenv('MAX_TOTAL_POSITION_PCT', '80'), 0.0, min_val=0.0, max_val=100.0)

            # ===== 以下参数仅在顶层类属性中定义，reload时被遗漏，现统一补充 =====
            cls.ADX_ENABLED = os.getenv('ADX_ENABLED', 'false').lower() == 'true'
            cls.ATR_MAX_PCT = float(os.getenv('ATR_MAX_PCT', '7.0'))
            cls.BACKTEST_AUTO_SAVE = os.getenv('BACKTEST_AUTO_SAVE', 'true').lower() == 'true'
            cls.BACKTEST_DATA_DIR = os.getenv('BACKTEST_DATA_DIR', 'data')
            cls.BACKTEST_FORCED_CLOSE_INCLUDE_STATS = os.getenv('BACKTEST_FORCED_CLOSE_INCLUDE_STATS', 'false').lower() == 'true'
            cls.BACKTEST_INDICATOR_FILTER_ENABLED = os.getenv('BACKTEST_INDICATOR_FILTER_ENABLED', 'false').lower() == 'true'
            cls.BACKTEST_INITIAL_CAPITAL = int(float(os.getenv('BACKTEST_INITIAL_CAPITAL', '10000')))
            cls.BACKTEST_PLOT_ENABLED = os.getenv('BACKTEST_PLOT_ENABLED', 'true').lower() == 'true'
            cls.BACKTEST_RESULT_DIR = os.getenv('BACKTEST_RESULT_DIR', 'results')
            cls.BUILDING_DRAWDOWN_TOLERANCE = float(os.getenv('BUILDING_DRAWDOWN_TOLERANCE', '10.0'))
            cls.CHART_ENABLED = os.getenv('CHART_ENABLED', 'true').lower() == 'true'
            cls.COUNTER_LOSS_10 = safe_float(os.getenv('COUNTER_LOSS_10', '10'), 10.0, min_val=0.0, max_val=100.0)
            cls.COUNTER_LOSS_20 = safe_float(os.getenv('COUNTER_LOSS_20', '20'), 20.0, min_val=0.0, max_val=100.0)
            cls.DRAWDOWN_30_THRESHOLD = safe_float(os.getenv('DRAWDOWN_30_THRESHOLD', '30'), 30.0, min_val=0.0, max_val=100.0)
            cls.DRAWDOWN_50_THRESHOLD = safe_float(os.getenv('DRAWDOWN_50_THRESHOLD', '50'), 50.0, min_val=0.0, max_val=100.0)
            cls.EMERGENCY_CLOSE_ON_PAUSE = os.getenv('EMERGENCY_CLOSE_ON_PAUSE', 'true').lower() == 'true'
            cls.EMERGENCY_CONTINUOUS_LOSS = int(float(os.getenv('EMERGENCY_CONTINUOUS_LOSS', '3')))
            cls.EMERGENCY_DAILY_LOSS_PERCENT = float(os.getenv('EMERGENCY_DAILY_LOSS_PERCENT', '10.0'))
            cls.EMERGENCY_LIQUIDATION_BUFFER = float(os.getenv('EMERGENCY_LIQUIDATION_BUFFER', '0.1'))
            cls.EMERGENCY_PAUSE_COOLDOWN = int(float(os.getenv('EMERGENCY_PAUSE_COOLDOWN', '1800')))
            cls.EMERGENCY_STOP_ENABLED = os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() == 'true'
            cls.ENTRY_STOP_LOSS_BARS = int(float(os.getenv('ENTRY_STOP_LOSS_BARS', '30')))
            cls.ENTRY_STOP_LOSS_PCT = safe_float(os.getenv('ENTRY_STOP_LOSS_PCT', '0'), 0.0, min_val=-100.0, max_val=100.0)
            cls.ENTRY_TREND_FILTER_ENABLED = os.getenv('ENTRY_TREND_FILTER_ENABLED', 'false').lower() == 'true'
            cls.EXIT_BREAKEVEN_PNL = int(float(os.getenv('EXIT_BREAKEVEN_PNL', '10')))
            cls.EXIT_DRAWDOWN_30 = int(float(os.getenv('EXIT_DRAWDOWN_30', '30')))
            cls.EXIT_DRAWDOWN_50 = int(float(os.getenv('EXIT_DRAWDOWN_50', '50')))
            cls.EXIT_HOLD_BARS_STOP = int(float(os.getenv('EXIT_HOLD_BARS_STOP', '30')))
            cls.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
            cls.LONG_FILTER_BB_MIN = float(os.getenv('LONG_FILTER_BB_MIN', '0.45'))
            cls.LONG_FILTER_RSI_MIN = float(os.getenv('LONG_FILTER_RSI_MIN', '55.0'))
            cls.MAX_RETRIES = int(float(os.getenv('MAX_RETRIES', '3')))
            cls.MONITOR_KLINES_COUNT = int(float(os.getenv('MONITOR_KLINES_COUNT', '100')))
            cls.ORDER_MODE = os.getenv('ORDER_MODE', 'LIMIT_TO_MARKET')
            cls.POSITION_COMPLETE_LOSS_FALL = float(os.getenv('POSITION_COMPLETE_LOSS_FALL', '1.0'))
            cls.PROFIT_PROTECT_THRESHOLD = safe_float(os.getenv('PROFIT_PROTECT_THRESHOLD', '10'), 10.0, min_val=0.0, max_val=100.0)
            cls.REBUILD_FIRST_RATIO = safe_float(os.getenv('REBUILD_FIRST_RATIO', '50'), 50.0, min_val=0.0, max_val=100.0)
            cls.REBUILD_SECOND_RATIO = safe_float(os.getenv('REBUILD_SECOND_RATIO', '50'), 50.0, min_val=0.0, max_val=100.0)
            cls.RETRY_DELAY = int(float(os.getenv('RETRY_DELAY', '1')))
            cls.SHORT_FILTER_BB_MAX = float(os.getenv('SHORT_FILTER_BB_MAX', '0.60'))
            cls.SHORT_FILTER_RSI_MAX = float(os.getenv('SHORT_FILTER_RSI_MAX', '50.0'))
            cls.TREND_CONTINUATION_CHECK_INTERVAL = int(float(os.getenv('TREND_CONTINUATION_CHECK_INTERVAL', '60')))
            cls.TREND_CONTINUATION_ENABLED = os.getenv('TREND_CONTINUATION_ENABLED', 'false').lower() == 'true'
            cls.TREND_CONTINUATION_LOOKBACK = int(float(os.getenv('TREND_CONTINUATION_LOOKBACK', '20')))
            cls.TREND_CONTINUATION_MIN_PROFIT = float(os.getenv('TREND_CONTINUATION_MIN_PROFIT', '5.0'))
            cls.TREND_FILTER_2H_MIN = float(os.getenv('TREND_FILTER_2H_MIN', '-3.0'))
            cls.TREND_FILTER_4H_MIN = float(os.getenv('TREND_FILTER_4H_MIN', '-2.0'))
            cls.TREND_FILTER_ENABLED = os.getenv('TREND_FILTER_ENABLED', 'false').lower() == 'true'
            cls.TREND_INTERVAL = os.getenv('TREND_INTERVAL', '2h')
            cls.TREND_REVERSAL_BARS = int(float(os.getenv('TREND_REVERSAL_BARS', '60')))
            cls.V23_FALLBACK_SYMBOLS = os.getenv('V23_FALLBACK_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT')
            cls.V23_KLINE_HISTORY_COUNT = int(float(os.getenv('V23_KLINE_HISTORY_COUNT', '100')))
            cls.V23_SIGNAL_CHECK_INTERVAL = int(float(os.getenv('V23_SIGNAL_CHECK_INTERVAL', '2')))
            cls.V23_SIGNAL_COOLDOWN = int(float(os.getenv('V23_SIGNAL_COOLDOWN', '300')))

        # 触发策略热重载（在锁外执行，避免死锁）
        try:
            from strategies.hf.strategy import get_hf_settings
            hf_settings = get_hf_settings()
            hf_settings.reload()
        except Exception:
            pass  # HF策略未导入或重载失败时忽略

        # 触发15mTupo退出参数缓存刷新
        try:
            import importlib
            _m = importlib.import_module('strategies.15mTupo.private.tupo_core')
            _m.reload_exit_cfg()
        except Exception:
            pass  # 15mTupo策略未导入时忽略

    @classmethod
    def enable_hot_reload(cls, interval=60):
        """启用热读取（在后台线程中定期检查文件变化）"""
        def reload_worker():
            """热读取工作线程"""
            while not cls._stop_reload.is_set():
                try:
                    current_modified = cls._get_latest_env_mtime()
                    if current_modified > cls._last_modified:
                        # ✅ 修复 #4: 使用锁保护配置重载，防止竞态条件
                        with cls._reload_lock:
                            print(f"[settings] 检测到 .env 文件修改，重新加载配置...")
                            cls._reload_env_vars()
                            print(f"[settings] 配置已重新加载，读取间隔: {interval}秒")
                except Exception as e:
                    print(f"[settings] 热读取失败: {str(e)}")

                # 使用 wait(timeout) 代替 sleep，可被 _stop_reload.set() 中断
                cls._stop_reload.wait(timeout=interval)

        # 启动后台线程
        cls._stop_reload.clear()
        cls._reload_interval = interval
        cls._reload_thread = threading.Thread(target=reload_worker, daemon=True)
        cls._reload_thread.start()
        print(f"[settings] 热读取已启用，检测间隔: {interval}秒")

    @classmethod
    def disable_hot_reload(cls):
        """禁用热读取"""
        cls._stop_reload.set()
        if cls._reload_thread and cls._reload_thread.is_alive():
            cls._reload_thread.join(timeout=5)
        print("[settings] 热读取已禁用")

    # ========== 动态配置解析 ==========
    # 共享参数：每次访问时从 os.environ 动态读取，确保与 ConfigManager/tupo_core 一致
    _SHARED_PARAMS = {
        'LEVERAGE':            ('LEVERAGE', int, 20),
        'MAX_POSITIONS':       ('MAX_POSITIONS', int, 5),
        'INITIAL_POSITION':    ('INITIAL_POSITION', float, 15.0),
        'MARGIN_MODE':         ('MARGIN_MODE', str, 'ISOLATED'),
        'STOPLOSS_TRIGGER1':   ('STOPLOSS_TRIGGER1', float, -35.0),
        'STOPLOSS_TRIGGER2':   ('STOPLOSS_TRIGGER2', float, -40.0),
        'STOPLOSS_TRIGGER3':   ('STOPLOSS_TRIGGER3', float, -50.0),
        'STOPLOSS_CLOSE1':     ('STOPLOSS_CLOSE1', float, 100.0),
        'STOPLOSS_CLOSE2':     ('STOPLOSS_CLOSE2', float, 50.0),
        'HIGH_PROFIT_THRESHOLD': ('HIGH_PROFIT_THRESHOLD', float, 60.0),
        'LOW_PROFIT_THRESHOLD':  ('LOW_PROFIT_THRESHOLD', float, 50.0),
        'BREAKEVEN_THRESHOLD':   ('BREAKEVEN_THRESHOLD', float, 10.0),
        'DELAY_RATIO':         ('DELAY_RATIO', float, 0.003),
        'ORDER_MODE':          ('ORDER_MODE', str, 'LIMIT_TO_MARKET'),
        'LOG_LEVEL':           ('LOG_LEVEL', str, 'INFO'),
        'OPEN_INTEREST_MONITOR_ENABLED': ('OPEN_INTEREST_MONITOR_ENABLED', lambda v: v.lower() == 'true', False),
        'EMERGENCY_STOP_ENABLED': ('EMERGENCY_STOP_ENABLED', lambda v: v.lower() == 'true', True),
        'STRATEGY_PRIORITY':    ('STRATEGY_PRIORITY', lambda v: [s.strip() for s in v.split(',') if s.strip()], ['15mTupo', 'HF']),
    }

    def __getattr__(self, name):
        """实例访问共享参数时，从 os.environ 动态读取"""
        if name in self._SHARED_PARAMS:
            env_key, converter, default = self._SHARED_PARAMS[name]
            val = os.environ.get(env_key)
            if val is not None and val != '':
                try:
                    return converter(val)
                except (ValueError, TypeError):
                    return default
            return default
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @classmethod
    def get(cls, name, default=None):
        """类方法：从 os.environ 动态读取配置值"""
        if name in cls._SHARED_PARAMS:
            env_key, converter, fallback = cls._SHARED_PARAMS[name]
            val = os.environ.get(env_key)
            if val is not None and val != '':
                try:
                    return converter(val)
                except (ValueError, TypeError):
                    return fallback
            return fallback
        return getattr(cls, name, default)


class SettingsValidator:
    """配置参数验证器"""

    @staticmethod
    def validate_leverage(leverage: int) -> bool:
        """验证杠杆倍数是否有效"""
        return 1 <= leverage <= 125

    @staticmethod
    def validate_percentage(value: float, min_val: float = 0, max_val: float = 100) -> bool:
        """验证百分比值是否有效"""
        return min_val <= value <= max_val

    @staticmethod
    def validate_order_size(quantity: float, price: float, leverage: int, min_notional: float = 5.0) -> bool:
        """验证订单大小是否合理"""
        if quantity <= 0 or price <= 0:
            return False
        notional_value = quantity * price
        if notional_value < min_notional:
            return False
        return True

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """验证币种符号是否有效"""
        if not symbol or len(symbol) < 6:
            return False
        return symbol.endswith('USDT')

    @classmethod
    def validate_all(cls, settings_obj) -> Dict[str, List[str]]:
        """验证所有配置参数，返回错误字典"""
        errors = []

        if not cls.validate_leverage(settings_obj.LEVERAGE):
            errors.append(f"LEVERAGE={settings_obj.LEVERAGE} 无效，应在1-125之间")

        if settings_obj.INITIAL_POSITION <= 0 or settings_obj.INITIAL_POSITION > 100:
            errors.append(f"INITIAL_POSITION={settings_obj.INITIAL_POSITION} 无效，应在0-100之间")

        if settings_obj.STOPLOSS_CLOSE1 <= 0 or settings_obj.STOPLOSS_CLOSE1 > 100:
            errors.append(f"STOPLOSS_CLOSE1={settings_obj.STOPLOSS_CLOSE1} 无效，应在0-100之间")

        if settings_obj.MAX_POSITIONS < 1:
            errors.append(f"MAX_POSITIONS={settings_obj.MAX_POSITIONS} 无效，应至少为1")

        return {'all_errors': errors}


# 创建 Settings 实例
settings = Settings()

# 初始化时验证配置
validator = SettingsValidator()
validation_results = validator.validate_all(settings)
if validation_results['all_errors']:
    print("[警告] 配置验证发现以下问题:")
    for error in validation_results['all_errors']:
        print(f"  - {error}")
else:
    print("[settings] 配置验证通过")
