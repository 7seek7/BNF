# -*- coding: utf-8 -*-
"""
统一日志系统 - 增强版

特性：
1. 分级日志策略（DEBUG/INFO/WARNING/ERROR/CRITICAL）
2. 上下文追踪（策略名、币种、订单ID等）
3. 性能监控（方法执行时间统计）
4. 结构化日志（JSON格式可选）
5. 日志轮转和归档
6. 多输出目标（控制台、文件、Telegram）
"""

import os
import sys
import json
import time
import logging
import logging.handlers
import threading
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from functools import wraps
from contextvars import ContextVar
from collections import defaultdict


# 上下文变量（线程安全）
_log_context: ContextVar[Dict[str, Any]] = ContextVar('log_context', default={})


@dataclass
class LogConfig:
    """日志配置"""
    name: str = "BNFRich"
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_dir: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 30
    console_output: bool = True
    file_output: bool = True
    json_format: bool = False
    include_context: bool = True
    performance_tracking: bool = True
    telegram_alert: bool = False  # ERROR以上发送Telegram


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器（Windows兼容）"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',  # 青色
        'INFO': '\033[32m',   # 绿色
        'WARNING': '\033[33m', # 黄色
        'ERROR': '\033[31m',   # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 检测是否支持颜色（Windows cmd 默认不支持）
        self._use_color = self._check_color_support()
    
    def _check_color_support(self) -> bool:
        """检查终端是否支持颜色"""
        # Windows Terminal 或已设置 ANSI 支持
        if sys.platform == 'win32':
            # 检查环境变量判断是否在 Windows Terminal 中
            import os
            if os.environ.get('WT_SESSION'):
                # Windows Terminal 支持 ANSI
                return True
            # 检查是否启用了 Virtual Terminal
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # 获取标准输出句柄
                h = kernel32.GetStdHandle(-11)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    # 检查是否已启用 VT 模式 (ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004)
                    if mode.value & 0x0004:
                        return True
                    # 尝试启用 VT 模式
                    if kernel32.SetConsoleMode(h, mode.value | 0x0004):
                        return True
            except Exception:
                pass  # Windows控制台检测失败，禁用颜色输出（预期行为）
        return False
    
    def format(self, record):
        # 添加颜色（仅在支持颜色的终端）
        if self._use_color and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.BOLD}{self.COLORS[record.levelname]}"
                f"[{record.levelname}]{self.RESET}"
            )
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加上下文
        if hasattr(record, 'context'):
            log_data['context'] = record.context
            
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        return json.dumps(log_data, ensure_ascii=False)


class ContextualLogger:
    """
    上下文日志器
    
    支持在日志中自动注入上下文信息（策略名、币种、订单ID等）
    """
    
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self._logger = logger
        self._context: Dict[str, Any] = {}
        
    def set_context(self, **kwargs):
        """设置日志上下文"""
        self._context.update(kwargs)
        return self
        
    def clear_context(self):
        """清除日志上下文"""
        self._context.clear()
        return self
        
    def _log(self, level: int, msg: str, *args, **kwargs):
        """带上下文的日志输出"""
        # 复制kwargs避免修改调用方的原始dict
        kwargs = dict(kwargs)
        # 合并上下文
        context = {**self._context}
        if 'extra' in kwargs:
            context.update(kwargs['extra'].get('context', {}))
            
        if context:
            kwargs['extra'] = dict(kwargs.get('extra', {}))
            kwargs['extra']['context'] = context
            
        # 格式化消息，添加上下文前缀
        if context:
            context_str = ' | '.join(f"{k}={v}" for k, v in context.items() if v)
            if context_str:
                msg = f"[{context_str}] {msg}"
                
        self._logger.log(level, msg, *args, **kwargs)
        
    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)
        
    def exception(self, msg: str, *args, **kwargs):
        """记录异常（自动包含堆栈）"""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, *args, **kwargs)
        
    def with_context(self, **kwargs) -> 'ContextualLogger':
        """返回带上下文的日志器实例（链式调用）"""
        new_logger = ContextualLogger(self.name, self._logger)
        new_logger._context = {**self._context, **kwargs}
        return new_logger


class PerformanceTracker:
    """
    性能追踪器
    
    记录方法执行时间，帮助发现性能瓶颈
    """
    
    def __init__(self, logger: ContextualLogger):
        self.logger = logger
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
    def track(self, name: str, duration: float):
        """记录执行时间"""
        with self._lock:
            self._metrics[name].append(duration)
            
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取统计数据"""
        with self._lock:
            if name not in self._metrics:
                return {}
            data = self._metrics[name]
            if not data:
                return {}
            return {
                'count': len(data),
                'total': sum(data),
                'avg': sum(data) / len(data),
                'min': min(data),
                'max': max(data),
            }
            
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计数据"""
        with self._lock:
            return {name: self.get_stats(name) for name in self._metrics}
            
    def report(self) -> str:
        """生成性能报告"""
        stats = self.get_all_stats()
        lines = ["=" * 60, "性能统计报告", "=" * 60]
        for name, data in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            lines.append(
                f"{name}: 调用{data['count']}次, "
                f"总耗时{data['total']:.2f}s, "
                f"平均{data['avg']*1000:.2f}ms, "
                f"最大{data['max']*1000:.2f}ms"
            )
        return "\n".join(lines)


def track_performance(name: Optional[str] = None):
    """
    性能追踪装饰器
    
    用法:
        @track_performance()
        def my_method(self):
            ...
            
        @track_performance("自定义名称")
        def my_method(self):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                # 获取logger
                logger = None
                if args and hasattr(args[0], 'logger'):
                    logger = args[0].logger
                elif args and hasattr(args[0], '_logger'):
                    logger = args[0]._logger
                    
                if logger:
                    method_name = name or func.__name__
                    logger.debug(f"⏱️ {method_name} 耗时: {duration*1000:.2f}ms")
                    
                    # 记录到性能追踪器
                    if hasattr(logger, '_tracker'):
                        logger._tracker.track(method_name, duration)
                        
        return wrapper
    return decorator


class LoggerManager:
    """
    日志管理器
    
    统一管理所有日志器，支持动态配置
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: LogConfig = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self, config: LogConfig = None):
        if getattr(self, '_initialized', False):
            return
        with self._lock:
            if getattr(self, '_initialized', False):
                return
            
        self.config = config or LogConfig()
        self._loggers: Dict[str, ContextualLogger] = {}
        self._trackers: Dict[str, PerformanceTracker] = {}
        self._root_logger = self._create_root_logger()
        self._initialized = True
        
    def _create_root_logger(self) -> logging.Logger:
        """创建根日志器"""
        logger = logging.getLogger(self.config.name)
        logger.setLevel(getattr(logging, self.config.level.upper()))
        logger.handlers.clear()
        
        # 控制台处理器
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = ColoredFormatter(
                '%(asctime)s %(levelname)s %(name)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
        # 文件处理器
        if self.config.file_output:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"{self.config.name}_{datetime.now():%Y%m%d}.log"
            
            # 使用自定义Handler实现日志轮转（Windows兼容）
            # 解决 WinError 32 文件锁定问题
            class WindowsSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
                """Windows兼容的日志轮转处理器，捕获 tell() / emit 异常
                """

                def emit(self, record):
                    try:
                        if self.shouldRollover(record):
                            self.doRollover()
                    except OSError as e:
                        import sys
                        if sys.platform == 'win32' and getattr(e, 'winerror', None) != 32:
                            raise
                    try:
                        logging.FileHandler.emit(self, record)
                    except OSError as e:
                        import sys
                        if sys.platform == 'win32' and getattr(e, 'winerror', None) != 32:
                            raise

                def doRollover(self):
                    """执行轮转，Windows兼容处理"""
                    try:
                        super().doRollover()
                    except OSError as e:
                        if hasattr(e, 'winerror') and e.winerror == 32:
                            import os
                            dfn = self.rotation_filename(self.baseFilename + ".1")
                            if os.path.exists(dfn):
                                try:
                                    os.remove(dfn)
                                except OSError:
                                    pass
                            super().doRollover()
                        else:
                            raise
            file_handler = WindowsSafeRotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            if self.config.json_format:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        return logger
        
    def get_logger(self, name: str, strategy: str = None) -> ContextualLogger:
        """
        获取日志器
        
        Args:
            name: 日志器名称
            strategy: 策略名称（可选）
            
        Returns:
            ContextualLogger实例
        """
        key = f"{name}_{strategy}" if strategy else name
        
        if key not in self._loggers:
            logger = logging.getLogger(f"{self.config.name}.{name}")
            contextual = ContextualLogger(name, logger)
            
            if strategy:
                contextual.set_context(strategy=strategy)
                
            self._loggers[key] = contextual
            
            # 创建性能追踪器
            if self.config.performance_tracking:
                self._trackers[key] = PerformanceTracker(contextual)
                contextual._tracker = self._trackers[key]
                
        return self._loggers[key]
        
    def get_tracker(self, name: str) -> Optional[PerformanceTracker]:
        """获取性能追踪器"""
        return self._trackers.get(name)
        
    def set_level(self, level: str):
        """动态设置日志级别"""
        self._root_logger.setLevel(getattr(logging, level.upper()))
        self.config.level = level
        
    def get_performance_report(self) -> str:
        """获取性能报告"""
        lines = []
        for name, tracker in self._trackers.items():
            stats = tracker.get_all_stats()
            if stats:
                lines.append(f"\n--- {name} ---")
                lines.append(tracker.report())
        return "\n".join(lines) if lines else "暂无性能数据"


# 便捷函数
_manager: Optional[LoggerManager] = None


def init_logging(config: LogConfig = None) -> LoggerManager:
    """初始化日志系统"""
    global _manager
    _manager = LoggerManager(config)
    return _manager


def get_logger(name: str, strategy: str = None) -> ContextualLogger:
    """
    获取日志器（便捷函数）
    
    Args:
        name: 日志器名称（通常是模块名）
        strategy: 策略名称（可选，用于区分不同策略）
        
    Returns:
        ContextualLogger实例
        
    用法:
        logger = get_logger('kline_manager')
        logger.info("K线数据加载完成", extra={'count': 100})
        
        # 带策略上下文
        logger = get_logger('order_engine', strategy='15mTupo')
        logger.info("下单成功", extra={'symbol': 'BTCUSDT', 'side': 'BUY'})
    """
    global _manager
    if _manager is None:
        _manager = LoggerManager()
    return _manager.get_logger(name, strategy)


def get_performance_report() -> str:
    """获取性能报告"""
    global _manager
    if _manager is None:
        return "日志系统未初始化"
    return _manager.get_performance_report()
