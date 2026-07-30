# -*- coding: utf-8 -*-
"""
统一配置管理

特性：
1. 主配置（.env）- 管理公用参数
2. 策略配置（strategies/{策略名}/.env）- 策略独立参数
3. 配置热重载
4. 配置验证和冲突检测
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
from dotenv import load_dotenv
from framework.core.logger import get_logger

logger = get_logger('config_manager')


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    enabled: bool = True
    config_path: Optional[Path] = None
    
    # 策略特有参数（从.env加载）
    max_positions: int = 5
    single_symbol_max: float = 5000
    leverage: int = 20
    monitor_symbols: int = 30
    
    # 资金分配
    capital_ratio: float = 100.0  # 占用资金比例（%）
    allocation: float = 0.0  # 策略实际分配资金（根据capital_ratio计算）
    
    def load_from_env(self, env_path: Path):
        """从.env文件加载配置"""
        if not env_path.exists():
            logger.warning(f"策略配置文件不存在: {env_path}")
            return
            
        load_dotenv(env_path, override=True)
        
        # 加载策略参数（使用策略名前缀）
        prefix = self.name.upper()
        
        # 基础参数
        self.enabled = os.getenv(f'{prefix}_ENABLED', 'true').lower() == 'true'
        self.max_positions = int(os.getenv(f'{prefix}_MAX_POSITIONS', '5'))
        self.single_symbol_max = float(os.getenv(f'{prefix}_SINGLE_SYMBOL_MAX', '5000'))
        self.leverage = int(os.getenv(f'{prefix}_LEVERAGE', '20'))
        self.monitor_symbols = int(os.getenv(f'{prefix}_MONITOR_SYMBOLS', '30'))
        self.capital_ratio = float(os.getenv(f'{prefix}_CAPITAL_RATIO', '100'))
        self.allocation = 0.0  # 先默认0，后续根据主配置和capital_ratio计算
        
        logger.info(f"策略配置已加载: {self.name}", extra={
            'context': {
                'strategy': self.name,
                'enabled': self.enabled,
                'max_positions': self.max_positions,
                'single_symbol_max': self.single_symbol_max,
                'leverage': self.leverage,
                'capital_ratio': self.capital_ratio,
            }
        })
    
    def calculate_allocation(self, total_balance: float, max_positions: int = 5):
        """
        计算策略实际分配资金
        
        Args:
            total_balance: 账户总可用余额
            max_positions: 全局最大持仓数
            
        Returns:
            float: 策略实际分配资金
        """
        if self.allocation > 0:
            # 如果已经手动设置，直接返回
            return self.allocation
        
        # 根据capital_ratio计算
        strategy_allocation = total_balance * (self.capital_ratio / 100)
        
        # 单币种平均分配 = 策略分配 / 策略最大持仓数
        avg_per_symbol = strategy_allocation / self.max_positions if self.max_positions > 0 else strategy_allocation / max_positions
        
        # 保证金限制（单币最大投资）
        self.allocation = min(avg_per_symbol, self.single_symbol_max)
        
        return self.allocation


@dataclass  
class MainConfig:
    """主配置（框架级公用参数）"""
    
    # API配置
    binance_api_key: str = ''
    binance_api_secret: str = ''
    testnet_api_key: str = ''
    testnet_api_secret: str = ''
    
    # Telegram配置
    telegram_bot_token_alert: str = ''
    telegram_bot_token_trade: str = ''
    telegram_chat_id: str = ''
    
    # 全局风控参数
    emergency_daily_loss_percent: float = 10.0
    emergency_continuous_loss: int = 3
    emergency_stop_enabled: bool = True
    
    # 全局交易参数
    margin_mode: str = 'ISOLATED'
    max_positions: int = 5
    leverage: int = 20
    initial_position: float = 15.0
    
    # 止损参数
    stoploss_trigger1: float = -35.0
    stoploss_trigger2: float = -40.0
    stoploss_trigger3: float = -50.0
    
    # 止盈参数
    high_profit_threshold: float = 60.0
    breakeven_threshold: float = 10.0
    
    # 日志配置
    log_level: str = 'INFO'
    
    # 运行模式
    mode: str = 'testnet'  # live / testnet / backtest
    
    # 订单参数
    delay_ratio: float = 0.003  # 下单延迟系数 (0.3%)
    order_limit_timeout: int = 5  # 限价单转市价超时时间(秒)
    order_mode: str = 'LIMIT_TO_MARKET'  # 订单执行模式

    # HF 追踪止损(交易所预挂, 宕机也有尾随保护)
    hf_trail_activate: float = 0.2                  # 盈利达此 pnl% 激活追踪
    hf_trail_step: float = 0.05                     # 从峰值回撤此 pnl% 触发
    hf_trail_exchange_min_step_pct: float = 0.05    # 交易所止损单最小上移步长(价格%)
    hf_trail_exchange_min_interval: int = 15        # 交易所止损单最小改单间隔(秒)
    
    # 加仓参数
    add_position_ratio: float = 50.0  # 加仓比例%
    add_position_bars_limit: int = 10  # 加仓等待K线数
    
    # 风控参数
    emergency_liquidation_buffer: float = 0.1  # 强平缓冲
    emergency_pause_cooldown: int = 1800  # 熔断后冷却时间(秒)
    emergency_close_on_pause: bool = True  # 熔断时是否紧急平仓
    
    # 黑天鹅参数
    black_swan_drop_threshold_1: float = -10.0  # 一级黑天鹅下跌阈值
    black_swan_drop_threshold_2: float = -15.0  # 二级
    black_swan_drop_threshold_3: float = -20.0  # 三级

    # 外仓检测参数
    external_pos_cutoff_hours: int = 168  # 外部持仓检测窗口（小时），默认7天


class ConfigManager:
    """
    配置管理器
    
    管理主配置和策略配置，支持热重载
    """
    
    _instance = None
    _singleton_lock = threading.Lock()
    _init_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            
            self.main_config = MainConfig()
            self.strategy_configs: Dict[str, StrategyConfig] = {}
            self._config_dir = Path(__file__).resolve().parent.parent.parent / 'config'
            self._strategies_dir = Path(__file__).resolve().parent.parent.parent / 'strategies'
            self._env_path = Path(__file__).resolve().parent.parent.parent / '.env'
            self._last_modified = 0
            self._watch_thread = None
            self._stop_watch = False
            
            self._initialized = True
        
    def load_main_config(self, env_path: Path = None) -> bool:
        """
        加载主配置
        
        Args:
            env_path: .env文件路径，默认为项目根目录
            
        Returns:
            是否加载成功
        """
        env_path = env_path or self._env_path
        
        # 尝试多个可能的路径
        if not env_path.exists():
            # 尝试当前目录
            for check_path in [Path('.'), Path(__file__).parent]:
                test_path = check_path / '.env'
                if test_path.exists():
                    env_path = test_path
                    break
        
        if not env_path.exists():
            logger.error(f"主配置文件不存在: {env_path}")
            return False
        
        try:
            load_dotenv(env_path, override=True)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False
        
        try:
            # API配置
            self.main_config.binance_api_key = os.getenv('BINANCE_API_KEY', '')
            self.main_config.binance_api_secret = os.getenv('BINANCE_API_SECRET', '')
            self.main_config.testnet_api_key = os.getenv('TESTNET_API_KEY', '')
            self.main_config.testnet_api_secret = os.getenv('TESTNET_API_SECRET', '')
            
            # Telegram配置
            self.main_config.telegram_bot_token_alert = os.getenv('TELEGRAM_BOT_TOKEN_ALERT', '')
            self.main_config.telegram_bot_token_trade = os.getenv('TELEGRAM_BOT_TOKEN_TRADE', '')
            self.main_config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
            
            # 风控参数
            self.main_config.emergency_daily_loss_percent = float(os.getenv('EMERGENCY_DAILY_LOSS_PERCENT', '10.0'))
            self.main_config.emergency_continuous_loss = int(os.getenv('EMERGENCY_CONTINUOUS_LOSS', '3'))
            self.main_config.emergency_stop_enabled = os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() == 'true'
            
            # 交易参数
            self.main_config.margin_mode = os.getenv('MARGIN_MODE', 'ISOLATED')
            self.main_config.max_positions = int(os.getenv('MAX_POSITIONS', '5'))
            self.main_config.leverage = int(os.getenv('LEVERAGE', '20'))
            self.main_config.initial_position = float(os.getenv('INITIAL_POSITION', '15.0'))
            
            # 止损参数
            self.main_config.stoploss_trigger1 = float(os.getenv('STOPLOSS_TRIGGER1', '-35.0'))
            self.main_config.stoploss_trigger2 = float(os.getenv('STOPLOSS_TRIGGER2', '-40.0'))
            self.main_config.stoploss_trigger3 = float(os.getenv('STOPLOSS_TRIGGER3', '-50.0'))
            
            # 止盈参数
            self.main_config.high_profit_threshold = float(os.getenv('HIGH_PROFIT_THRESHOLD', '60.0'))
            self.main_config.breakeven_threshold = float(os.getenv('BREAKEVEN_THRESHOLD', '10.0'))
            
            # 日志
            self.main_config.log_level = os.getenv('LOG_LEVEL', 'INFO')
            
            # 订单参数
            self.main_config.delay_ratio = float(os.getenv('DELAY_RATIO', '0.003'))
            self.main_config.order_limit_timeout = int(os.getenv('ORDER_LIMIT_TIMEOUT', '5'))
            self.main_config.order_mode = os.getenv('ORDER_MODE', 'LIMIT_TO_MARKET')

            # HF 追踪止损(交易所预挂)
            self.main_config.hf_trail_activate = float(os.getenv('HF_TRAIL_ACTIVATE', '0.2'))
            self.main_config.hf_trail_step = float(os.getenv('HF_TRAIL_STEP', '0.05'))
            self.main_config.hf_trail_exchange_min_step_pct = float(os.getenv('HF_TRAIL_EXCHANGE_MIN_STEP_PCT', '0.05'))
            self.main_config.hf_trail_exchange_min_interval = int(os.getenv('HF_TRAIL_EXCHANGE_MIN_INTERVAL', '15'))
            
            # 加仓参数
            self.main_config.add_position_ratio = float(os.getenv('ADD_POSITION_RATIO', '50'))
            self.main_config.add_position_bars_limit = int(os.getenv('ADD_POSITION_BARS_LIMIT', '10'))
            
            # 风控参数
            self.main_config.emergency_liquidation_buffer = float(os.getenv('EMERGENCY_LIQUIDATION_BUFFER', '0.1'))
            self.main_config.emergency_pause_cooldown = int(os.getenv('EMERGENCY_PAUSE_COOLDOWN', '1800'))
            self.main_config.emergency_close_on_pause = os.getenv('EMERGENCY_CLOSE_ON_PAUSE', 'true').lower() == 'true'
            
            # 黑天鹅参数
            self.main_config.black_swan_drop_threshold_1 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_1', '-10.0'))
            self.main_config.black_swan_drop_threshold_2 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_2', '-15.0'))
            self.main_config.black_swan_drop_threshold_3 = float(os.getenv('BLACK_SWAN_DROP_THRESHOLD_3', '-20.0'))

            # 外仓检测参数
            self.main_config.external_pos_cutoff_hours = int(os.getenv('EXTERNAL_POS_CUTOFF_HOURS', '168'))
            
            self._last_modified = env_path.stat().st_mtime
            
            logger.info("主配置已加载", extra={
                'context': {
                    'leverage': self.main_config.leverage,
                    'max_positions': self.main_config.max_positions,
                    'margin_mode': self.main_config.margin_mode,
                }
            })
            
            return True
        
        except Exception as e:
            logger.exception(f"加载主配置失败: {e}")
            return False
    
    def load_strategy_config(self, strategy_name: str, env_path: Path = None) -> StrategyConfig:
        """
        加载策略配置
        
        Args:
            strategy_name: 策略名称（如 '15mTupo', 'hf'）
            env_path: 策略配置文件路径，默认为 strategies/{策略名}/.env
            
        Returns:
            StrategyConfig实例
        """
        # 默认路径：strategies/{策略名}/.env
        if env_path is None:
            env_path = Path('strategies') / strategy_name / '.env'
            
        config = StrategyConfig(name=strategy_name, config_path=env_path)
        config.load_from_env(env_path)
        
        self.strategy_configs[strategy_name] = config
        
        return config
        
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """获取策略配置"""
        return self.strategy_configs.get(strategy_name)
    
    def load_strategy_configs(self) -> Dict[str, StrategyConfig]:
        """
        加载所有策略配置
        
        Returns:
            策略配置字典
        """
        if not self._strategies_dir.exists():
            self._strategies_dir.mkdir(parents=True, exist_ok=True)
            return {}
            
        # 从strategies目录加载配置
        strategy_dirs = [d for d in self._strategies_dir.iterdir() if d.is_dir()]
        
        for strategy_dir in strategy_dirs:
            strategy_name = strategy_dir.name
            env_path = strategy_dir / '.env'
            
            if env_path.exists():
                self.load_strategy_config(strategy_name, env_path)
            else:
                # 如果没有.env文件，创建默认配置
                config = StrategyConfig(name=strategy_name)
                self.strategy_configs[strategy_name] = config
        
        return self.strategy_configs
        
    def get_strategy_param(self, strategy_name: str, param_name: str, default: Any = None) -> Any:
        """
        获取策略参数（优先使用策略配置，否则使用主配置）
        
        Args:
            strategy_name: 策略名称
            param_name: 参数名
            default: 默认值
            
        Returns:
            参数值
        """
        # 先查策略配置
        strategy_config = self.strategy_configs.get(strategy_name)
        if strategy_config and hasattr(strategy_config, param_name):
            return getattr(strategy_config, param_name)
            
        # 再查主配置
        if hasattr(self.main_config, param_name):
            return getattr(self.main_config, param_name)
            
        return default
        
    def validate(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误信息列表
        """
        errors = []
        
        # 检查API密钥
        if not self.main_config.binance_api_key:
            errors.append("缺少 BINANCE_API_KEY")
        if not self.main_config.binance_api_secret:
            errors.append("缺少 BINANCE_API_SECRET")
            
        # 检查风控参数
        if self.main_config.emergency_daily_loss_percent <= 0:
            errors.append("EMERGENCY_DAILY_LOSS_PERCENT 必须 > 0")
        if self.main_config.emergency_continuous_loss <= 0:
            errors.append("EMERGENCY_CONTINUOUS_LOSS 必须 > 0")
            
        # 检查交易参数
        if self.main_config.leverage <= 0 or self.main_config.leverage > 125:
            errors.append("LEVERAGE 必须在 1-125 之间")
        if self.main_config.max_positions <= 0:
            errors.append("MAX_POSITIONS 必须 > 0")
            
        # 检查止损参数顺序（止损值是负数，数值越大越接近0）
        # 例如: -35 > -40 > -50，所以 trigger1 > trigger2 > trigger3
        if self.main_config.stoploss_trigger1 <= self.main_config.stoploss_trigger2:
            errors.append(f"STOPLOSS_TRIGGER1({self.main_config.stoploss_trigger1}) must be > STOPLOSS_TRIGGER2({self.main_config.stoploss_trigger2})")
        if self.main_config.stoploss_trigger2 <= self.main_config.stoploss_trigger3:
            errors.append(f"STOPLOSS_TRIGGER2({self.main_config.stoploss_trigger2}) must be > STOPLOSS_TRIGGER3({self.main_config.stoploss_trigger3})")
            
        return errors
        
    def get_setting(self, name: str, default: Any = None) -> Any:
        """统一获取配置（先查 MainConfig.snake_case，再查 Settings.UPPER_CASE）

        Args:
            name: 配置名（大小写不敏感，自动转换）
            default: 默认值
        """
        # 尝试 MainConfig（snake_case）
        if hasattr(self.main_config, name):
            return getattr(self.main_config, name)
        lower_name = name.lower()
        for attr in dir(self.main_config):
            if attr.lower() == lower_name:
                return getattr(self.main_config, attr)
        # 尝试 Settings（UPPER_CASE）
        try:
            from config.settings import Settings as _S
            upper_name = name.upper()
            if hasattr(_S, upper_name):
                return getattr(_S, upper_name)
            for attr in dir(_S):
                if attr.upper() == upper_name:
                    return getattr(_S, attr)
        except ImportError:
            pass
        return default

    def set_setting(self, name: str, value: Any):
        """运行时设置配置值（同时写入 MainConfig 和 Settings）"""
        if hasattr(self.main_config, name):
            setattr(self.main_config, name, value)
        try:
            from config.settings import Settings as _S
            upper_name = name.upper()
            if hasattr(_S, upper_name):
                setattr(_S, upper_name, value)
        except ImportError:
            pass

    def enable_hot_reload(self, interval: int = 60):
        """
        启用配置热重载（同时刷新 MainConfig 和 Settings）
        
        Args:
            interval: 检查间隔（秒）
        """
        def watch():
            while not self._stop_watch:
                try:
                    if self._env_path.exists():
                        current_mtime = self._env_path.stat().st_mtime
                        if current_mtime > self._last_modified:
                            logger.info("检测到配置文件变更，重新加载...")
                            self.load_main_config()
                            # 重新加载所有策略配置
                            for name, config in self.strategy_configs.items():
                                if config.config_path and config.config_path.exists():
                                    config.load_from_env(config.config_path)
                            # 同步刷新 Settings 热重载
                            try:
                                from config.settings import Settings as _S
                                _S._reload_env_vars()
                            except (ImportError, AttributeError):
                                pass
                except Exception as e:
                    logger.error(f"配置热重载失败: {e}")

                time.sleep(interval)
                
        self._stop_watch = False
        self._watch_thread = threading.Thread(target=watch, daemon=True)
        self._watch_thread.start()
        logger.info(f"配置热重载已启用，检查间隔: {interval}秒")
        
    def disable_hot_reload(self):
        """禁用配置热重载"""
        self._stop_watch = True
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
        logger.info("配置热重载已禁用")
        
    def get_all_params(self) -> Dict[str, Any]:
        """获取所有配置参数（用于调试）"""
        return {
            'main': {
                'leverage': self.main_config.leverage,
                'max_positions': self.main_config.max_positions,
                'margin_mode': self.main_config.margin_mode,
                'daily_loss_limit': self.main_config.emergency_daily_loss_percent,
            },
            'strategies': {
                name: {
                    'enabled': config.enabled,
                    'max_positions': config.max_positions,
                    'leverage': config.leverage,
                    'capital_ratio': config.capital_ratio,
                }
                for name, config in self.strategy_configs.items()
            }
        }


# 便捷访问
_config_manager: Optional[ConfigManager] = None
_config_manager_lock = threading.Lock()


def get_config_manager() -> ConfigManager:
    """获取配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        with _config_manager_lock:
            if _config_manager is None:
                _config_manager = ConfigManager()
    return _config_manager


def get_main_config() -> MainConfig:
    """获取主配置"""
    return get_config_manager().main_config


def get_strategy_config(strategy_name: str) -> Optional[StrategyConfig]:
    """获取策略配置"""
    return get_config_manager().get_strategy_config(strategy_name)


def get_setting(name: str, default: Any = None) -> Any:
    """便捷获取配置（自动查找 MainConfig / Settings）"""
    return get_config_manager().get_setting(name, default)