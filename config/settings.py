import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import time
import threading

# 加载环境变量
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Settings:
    """系统配置类 - 所有可调参数均从环境变量读取，带中文注释，便于维护与分块实现

    支持参数热读取：修改 .env 文件后无需重启，下次读取时自动更新
    """
    # 最后修改时间（用于检测文件变化）
    _last_modified = 0
    _reload_interval = 60  # 热读取间隔（秒），默认60秒
    _reload_thread = None
    _stop_reload = False
    # API 配置
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')  # 实盘：币安API Key
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')  # 实盘：币安API Secret
    TESTNET_API_KEY = os.getenv('TESTNET_API_KEY', '')  # 测试网：API Key
    TESTNET_API_SECRET = os.getenv('TESTNET_API_SECRET', '')  # 测试网：API Secret

    # Telegram 配置
    TELEGRAM_BOT_TOKEN_ALERT = os.getenv('TELEGRAM_BOT_TOKEN_ALERT', '')  # 警报 Telegram 机器人 Token
    TELEGRAM_BOT_TOKEN_TRADE = os.getenv('TELEGRAM_BOT_TOKEN_TRADE', '')  # 交易 Telegram 机器人 Token
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')  # Telegram 聊天 ID

    # HuggingFace 配置 (Streamlit Cloud 持久化存储)
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')  # HuggingFace Access Token
    HUGGINGFACE_REPO_ID = os.getenv('HUGGINGFACE_REPO_ID', '')  # HuggingFace 仓库 ID (格式: username/repo-name)

    # 警报参数 - 监控阈值与频率（处理优化器传入的浮点数值）
    MONITOR_INTERVAL = int(float(os.getenv('MONITOR_INTERVAL', '5')))  # 监控周期（分钟）
    MONITOR_SYMBOLS_COUNT = int(float(os.getenv('MONITOR_SYMBOLS_COUNT', '100')))  # 按成交量排序前 N 名币种进行监控
    SYMBOLS_UPDATE_INTERVAL = int(float(os.getenv('SYMBOLS_UPDATE_INTERVAL', '60')))  # 监控币种排序更新间隔（秒）
    PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '3.0'))  # 涨跌阈值（%）
    VOLUME_THRESHOLD = float(os.getenv('VOLUME_THRESHOLD', '5.0'))  # 成交量阈值（倍率）
    VOLUME_COMPARE_PERIODS = int(float(os.getenv('VOLUME_COMPARE_PERIODS', '10')))  # 用于比较的历史周期数
    ALERT_COOLDOWN = int(float(os.getenv('ALERT_COOLDOWN', '60')))  # 警报冷却时间（分钟）
    OPEN_INTEREST_MONITOR_ENABLED = os.getenv('OPEN_INTEREST_MONITOR_ENABLED', 'false').lower() == 'true'  # 持仓增量监控开关
    OPEN_INTEREST_INCREASE_THRESHOLD = float(os.getenv('OPEN_INTEREST_INCREASE_THRESHOLD', '5.0'))  # 持仓增量阈值（%）

    # 持仓监控参数（根据杠杆调整刷新频率）
    POSITION_MONITOR_SLEEP_TIME = int(float(os.getenv('POSITION_MONITOR_SLEEP_TIME', '15')))  # 持仓监控刷新间隔（秒）
    # 推荐值：
    # - 5x杠杆: 30秒（价格波动影响小）
    # - 10x杠杆: 20秒（中等风险）
    # - 15-20x杠杆: 15秒（推荐，当前设置）
    # - 20-30x杠杆: 10秒（高风险）
    # - >30x杠杆: 5秒（极度危险，但API限流风险高）

    # 交易参数
    MARGIN_MODE = os.getenv('MARGIN_MODE', 'ISOLATED')  # 保证金模式（ISOLATED / CROSSED）
    LEVERAGE = int(float(os.getenv('LEVERAGE', '20')))  # 最大杠杆（在币种允许范围内取小值）
    MAX_POSITIONS = int(float(os.getenv('MAX_POSITIONS', '2')))  # 最大同时持仓数量（Streamlit Cloud建议2）
    MAX_POSITIONS_PER_SYMBOL = int(float(os.getenv('MAX_POSITIONS_PER_SYMBOL', '2')))  # 单币种最大持仓数（Streamlit Cloud建议2）
    SINGLE_SYMBOL_MAX_INVESTMENT = float(os.getenv('SINGLE_SYMBOL_MAX_INVESTMENT', '5000'))  # 单币种最大投资金额（USDT）
    POSITION_ALLOCATION_MODE = os.getenv('POSITION_ALLOCATION_MODE', 'EQUAL')  # 资金分配模式（EQUAL / DYNAMIC）
    ORDER_TYPE = os.getenv('ORDER_TYPE', 'MARKET')  # 下单类型（MARKET / LIMIT）
    INITIAL_POSITION = float(os.getenv('INITIAL_POSITION', '15'))  # 初始下单占总资金比例（%），降低初始开仓比例
    DELAY_RATIO = float(os.getenv('DELAY_RATIO', '0.1'))  # 下单延迟系数 YC%
    MAX_MARGIN_PER_SYMBOL = float(os.getenv('MAX_MARGIN_PER_SYMBOL', '25000'))  # 单币种最大保证金（USDT）

    # 回测与策略参数
    LOSS_STEP1 = float(os.getenv('LOSS_STEP1', '-2.0'))  # 亏损加仓阈值 1
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
    POSITION_COMPLETE_PROFIT_RISE = float(os.getenv('POSITION_COMPLETE_PROFIT_RISE', '2.0'))  # 建仓完成的利润上涨阈值
    POSITION_COMPLETE_LOSS_FALL = float(os.getenv('POSITION_COMPLETE_LOSS_FALL', '1.0'))  # 建仓完成的亏损回落阈值

    # 止盈参数
    HIGH_PROFIT_THRESHOLD = float(os.getenv('HIGH_PROFIT_THRESHOLD', '8.0'))  # 高利润触发止盈阈值
    HIGH_PROFIT_DRAWBACK1 = float(os.getenv('HIGH_PROFIT_DRAWBACK1', '2.0'))  # 第一次回撤阈值
    HIGH_PROFIT_CLOSE1 = float(os.getenv('HIGH_PROFIT_CLOSE1', '30'))  # 第一次止盈平仓阈值
    HIGH_PROFIT_DRAWBACK2 = float(os.getenv('HIGH_PROFIT_DRAWBACK2', '4.0'))  # 第二次回撤阈值
    HIGH_PROFIT_CLOSE2 = float(os.getenv('HIGH_PROFIT_CLOSE2', '50'))  # 第二次止盈平仓阈值

    LOW_PROFIT_THRESHOLD = float(os.getenv('LOW_PROFIT_THRESHOLD', '3.0'))  # 低利润触发止盈阈值
    LOW_PROFIT_DRAWBACK1 = float(os.getenv('LOW_PROFIT_DRAWBACK1', '1.5'))  # 低利润回撤阈值
    LOW_PROFIT_CLOSE1 = float(os.getenv('LOW_PROFIT_CLOSE1', '50'))  # 低利润止盈阈值

    BREAKEVEN_THRESHOLD = float(os.getenv('BREAKEVEN_THRESHOLD', '0.5'))  # 保本阈值

    # 止损参数
    STOPLOSS_TRIGGER1 = float(os.getenv('STOPLOSS_TRIGGER1', '-8.0'))  # 第一级止损触发点止损触发点
    STOPLOSS_CLOSE1 = float(os.getenv('STOPLOSS_CLOSE1', '100'))  # 第一级止损平仓比例
    STOPLOSS_TRIGGER2 = float(os.getenv('STOPLOSS_TRIGGER2', '-12.0'))  # 第二级止损触发点例
    STOPLOSS_CLOSE2 = float(os.getenv('STOPLOSS_CLOSE2', '50'))  # 第二级止损平仓比例
    STOPLOSS_TRIGGER3 = float(os.getenv('STOPLOSS_TRIGGER3', '-15.0'))  # 第三级止损触发点

    # 重新进场阈值
    PROFIT_REENTER_THRESHOLD = float(os.getenv('PROFIT_REENTER_THRESHOLD', '2.0'))  # 重新进场阈值
    REENTER_MIN_INTERVAL = int(float(os.getenv('REENTER_MIN_INTERVAL', '300')))  # 重新进场最小时间间隔（秒），默认5分钟

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
    EMERGENCY_CLOSE_ON_PAUSE = os.getenv('EMERGENCY_CLOSE_ON_PAUSE', 'true').lower() == 'true'  # 熔断时是否紧急平仓所有持仓

    # 回测参数
    BACKTEST_AUTO_SAVE = os.getenv('BACKTEST_AUTO_SAVE', 'true').lower() == 'true'  # 回测自动保存
    BACKTEST_PLOT_ENABLED = os.getenv('BACKTEST_PLOT_ENABLED', 'true').lower() == 'true'  # 回测图表功能

    # 其他系统参数
    MAX_RETRIES = int(float(os.getenv('MAX_RETRIES', '3')))  # 全局重试次数
    RETRY_DELAY = int(float(os.getenv('RETRY_DELAY', '1')))  # 重试间隔（秒）
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # 日志等级

    @classmethod
    def enable_hot_reload(cls, interval=60):
        """启用热读取（在后台线程中定期检查文件变化）"""
        def reload_worker():
            """热读取工作线程"""
            while not cls._stop_reload:
                try:
                    current_modified = os.path.getmtime(env_path)
                    if current_modified > cls._last_modified:
                        print(f"[settings] 检测到 .env 文件修改，重新加载配置...")
                        cls._reload_from_file()
                        cls._last_modified = current_modified
                        print(f"[settings] 配置已重新加载，读取间隔: {interval}秒")
                except Exception as e:
                    print(f"[settings] 热读取失败: {str(e)}")

                time.sleep(interval)

        # 启动后台线程
        cls._stop_reload = False
        cls._reload_interval = interval
        cls._reload_thread = threading.Thread(target=reload_worker, daemon=True)
        cls._reload_thread.start()
        print(f"[settings] 热读取已启用，检测间隔: {interval}秒")

    @classmethod
    def disable_hot_reload(cls):
        """禁用热读取"""
        cls._stop_reload = True
        if cls._reload_thread and cls._reload_thread.is_alive():
            cls._reload_thread.join(timeout=5)
        print("[settings] 热读取已禁用")

    @classmethod
    def _reload_from_file(cls):
        """从文件重新加载所有参数（内部方法）"""
        load_dotenv(env_path, override=True)
        cls._reload_all_parameters()

    @classmethod
    def reload(cls):
        """重新加载环境变量（热读取）"""
        load_dotenv(env_path, override=True)

        # 重新读取所有环境变量
        cls.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
        cls.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
        cls.TESTNET_API_KEY = os.getenv('TESTNET_API_KEY', '')
        cls.TESTNET_API_SECRET = os.getenv('TESTNET_API_SECRET', '')

        # 重新加载所有其他参数...
        cls._reload_all_parameters()

        return True
    
    @classmethod
    def _reload_all_parameters(cls):
        """重新加载所有参数"""
        # Telegram 配置
        cls.TELEGRAM_BOT_TOKEN_ALERT = os.getenv('TELEGRAM_BOT_TOKEN_ALERT', '')
        cls.TELEGRAM_BOT_TOKEN_TRADE = os.getenv('TELEGRAM_BOT_TOKEN_TRADE', '')
        cls.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

        # HuggingFace 配置
        cls.HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
        cls.HUGGINGFACE_REPO_ID = os.getenv('HUGGINGFACE_REPO_ID', '')

        # 警报参数（处理优化器传入的浮点数值）
        cls.MONITOR_INTERVAL = int(float(os.getenv('MONITOR_INTERVAL', '5')))
        cls.MONITOR_SYMBOLS_COUNT = int(float(os.getenv('MONITOR_SYMBOLS_COUNT', '100')))
        cls.SYMBOLS_UPDATE_INTERVAL = int(float(os.getenv('SYMBOLS_UPDATE_INTERVAL', '60')))
        cls.PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '3.0'))
        cls.VOLUME_THRESHOLD = float(os.getenv('VOLUME_THRESHOLD', '5.0'))
        cls.VOLUME_COMPARE_PERIODS = int(float(os.getenv('VOLUME_COMPARE_PERIODS', '10')))
        cls.ALERT_COOLDOWN = int(float(os.getenv('ALERT_COOLDOWN', '60')))
        cls.OPEN_INTEREST_MONITOR_ENABLED = os.getenv('OPEN_INTEREST_MONITOR_ENABLED', 'false').lower() == 'true'
        cls.OPEN_INTEREST_INCREASE_THRESHOLD = float(os.getenv('OPEN_INTEREST_INCREASE_THRESHOLD', '5.0'))
        
        # 交易参数
        cls.MARGIN_MODE = os.getenv('MARGIN_MODE', 'ISOLATED')
        cls.LEVERAGE = int(float(os.getenv('LEVERAGE', '20')))
        cls.MAX_POSITIONS = int(float(os.getenv('MAX_POSITIONS', '2')))
        cls.MAX_POSITIONS_PER_SYMBOL = int(float(os.getenv('MAX_POSITIONS_PER_SYMBOL', '2')))
        cls.SINGLE_SYMBOL_MAX_INVESTMENT = float(os.getenv('SINGLE_SYMBOL_MAX_INVESTMENT', '5000'))
        cls.POSITION_ALLOCATION_MODE = os.getenv('POSITION_ALLOCATION_MODE', 'EQUAL')
        cls.ORDER_TYPE = os.getenv('ORDER_TYPE', 'MARKET')
        cls.INITIAL_POSITION = float(os.getenv('INITIAL_POSITION', '15'))
        cls.DELAY_RATIO = float(os.getenv('DELAY_RATIO', '0.1'))
        cls.MAX_MARGIN_PER_SYMBOL = float(os.getenv('MAX_MARGIN_PER_SYMBOL', '25000'))
        
        # 回测与策略参数
        cls.LOSS_STEP1 = float(os.getenv('LOSS_STEP1', '-2.0'))
        cls.LOSS_ADD1 = float(os.getenv('LOSS_ADD1', '15'))
        cls.LOSS_STEP2 = float(os.getenv('LOSS_STEP2', '-4.0'))
        cls.LOSS_ADD2 = float(os.getenv('LOSS_ADD2', '25'))
        cls.LOSS_STEP3 = float(os.getenv('LOSS_STEP3', '-6.0'))
        cls.LOSS_ADD3 = float(os.getenv('LOSS_ADD3', '30'))
        
        cls.PROFIT_STEP1 = float(os.getenv('PROFIT_STEP1', '2.0'))
        cls.PROFIT_ADD1 = float(os.getenv('PROFIT_ADD1', '10'))
        cls.PROFIT_STEP2 = float(os.getenv('PROFIT_STEP2', '4.0'))
        cls.PROFIT_ADD2 = float(os.getenv('PROFIT_ADD2', '15'))
        cls.PROFIT_STEP3 = float(os.getenv('PROFIT_STEP3', '6.0'))
        cls.PROFIT_ADD3 = float(os.getenv('PROFIT_ADD3', '20'))
        
        cls.POSITION_COMPLETE_PROFIT_RISE = float(os.getenv('POSITION_COMPLETE_PROFIT_RISE', '2.0'))
        cls.POSITION_COMPLETE_LOSS_FALL = float(os.getenv('POSITION_COMPLETE_LOSS_FALL', '1.0'))
        
        # 止盈参数
        cls.HIGH_PROFIT_THRESHOLD = float(os.getenv('HIGH_PROFIT_THRESHOLD', '8.0'))
        cls.HIGH_PROFIT_DRAWBACK1 = float(os.getenv('HIGH_PROFIT_DRAWBACK1', '2.0'))
        cls.HIGH_PROFIT_CLOSE1 = float(os.getenv('HIGH_PROFIT_CLOSE1', '30'))
        cls.HIGH_PROFIT_DRAWBACK2 = float(os.getenv('HIGH_PROFIT_DRAWBACK2', '4.0'))
        cls.HIGH_PROFIT_CLOSE2 = float(os.getenv('HIGH_PROFIT_CLOSE2', '50'))
        
        cls.LOW_PROFIT_THRESHOLD = float(os.getenv('LOW_PROFIT_THRESHOLD', '3.0'))
        cls.LOW_PROFIT_DRAWBACK1 = float(os.getenv('LOW_PROFIT_DRAWBACK1', '1.5'))
        cls.LOW_PROFIT_CLOSE1 = float(os.getenv('LOW_PROFIT_CLOSE1', '50'))

        cls.BREAKEVEN_THRESHOLD = float(os.getenv('BREAKEVEN_THRESHOLD', '0.5'))

        # 止损参数
        cls.STOPLOSS_TRIGGER1 = float(os.getenv('STOPLOSS_TRIGGER1', '-8.0'))
        cls.STOPLOSS_CLOSE1 = float(os.getenv('STOPLOSS_CLOSE1', '100'))
        cls.STOPLOSS_TRIGGER2 = float(os.getenv('STOPLOSS_TRIGGER2', '-12.0'))
        cls.STOPLOSS_CLOSE2 = float(os.getenv('STOPLOSS_CLOSE2', '50'))
        cls.STOPLOSS_TRIGGER3 = float(os.getenv('STOPLOSS_TRIGGER3', '-15.0'))

        # 重新进场阈值
        cls.PROFIT_REENTER_THRESHOLD = float(os.getenv('PROFIT_REENTER_THRESHOLD', '2.0'))
        cls.REENTER_MIN_INTERVAL = int(float(os.getenv('REENTER_MIN_INTERVAL', '300')))
        
        # 黑天鹅防护参数
        cls.BLACK_SWAN_DROP_THRESHOLD_1 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_1', '-10.0'))
        cls.BLACK_SWAN_DROP_THRESHOLD_2 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_2', '-15.0'))
        cls.BLACK_SWAN_DROP_THRESHOLD_3 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_3', '-20.0'))
        
        # 熔断参数
        cls.EMERGENCY_DAILY_LOSS_PERCENT = float(os.getenv('EMERGENCY_DAILY_LOSS_PERCENT', '10.0'))
        cls.EMERGENCY_CONTINUOUS_LOSS = int(float(os.getenv('EMERGENCY_CONTINUOUS_LOSS', '3')))
        cls.EMERGENCY_LIQUIDATION_BUFFER = float(os.getenv('EMERGENCY_LIQUIDATION_BUFFER', '0.1'))
        cls.EMERGENCY_PAUSE_COOLDOWN = int(float(os.getenv('EMERGENCY_PAUSE_COOLDOWN', '1800')))
        
        # 紧急平仓
        cls.EMERGENCY_CLOSE_ON_PAUSE = os.getenv('EMERGENCY_CLOSE_ON_PAUSE', 'true').lower() == 'true'
        
        # 回测参数
        cls.BACKTEST_AUTO_SAVE = os.getenv('BACKTEST_AUTO_SAVE', 'true').lower() == 'true'
        cls.BACKTEST_PLOT_ENABLED = os.getenv('BACKTEST_PLOT_ENABLED', 'true').lower() == 'true'
        
        # 其他系统参数
        cls.MAX_RETRIES = int(float(os.getenv('MAX_RETRIES', '3')))
        cls.RETRY_DELAY = int(float(os.getenv('RETRY_DELAY', '1')))
        cls.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def validate(cls):
        """验证关键配置项是否合理"""
        errors = []
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
        if errors:
            raise ValueError("配置错误:\n" + "\n".join(errors))
        return True

settings = Settings()
