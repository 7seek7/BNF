"""
统一异常处理和重试策略
"""

import time
import threading
from functools import wraps
from framework.core.logger import get_logger

logger = get_logger('exception_handler')

# 可重试的异常类型
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
    ConnectionResetError,
    ConnectionAbortedError,
)

# Binance API特定异常
try:
    from binance.exceptions import (
        BinanceAPIException,
        BinanceRequestException,
        BinanceOrderException,
        BinanceConnectionException,
    )
    BINANCE_RETRYABLE = (
        BinanceConnectionException,
        BinanceRequestException,
    )
except ImportError:
    BINANCE_RETRYABLE = ()
    BinanceAPIException = Exception





def handle_api_errors(default_return=None, log_error=True):
    """
    API错误处理装饰器

    Args:
        default_return: 发生异常时的默认返回值
        log_error: 是否记录错误日志
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, BINANCE_RETRYABLE):
                    logger.error(f"API连接错误: {func.__name__} - {str(e)}")
                elif isinstance(e, BinanceOrderException):
                    # 修复：添加BinanceOrderException处理（BUG-H011）
                    logger.error(f"API订单错误: {func.__name__} - {str(e)}")
                elif isinstance(e, BinanceAPIException):
                    logger.error(f"API业务错误: {func.__name__} - {str(e)}")
                elif log_error:
                    logger.error(f"函数执行错误: {func.__name__} - {str(e)}", exc_info=True)

                return default_return
        return wrapper
    return decorator


def validate_input(*validators):
    """
    输入验证装饰器链

    Args:
        validators: 验证函数列表，每个函数接收参数应返回True/False
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for validator in validators:
                if not validator(*args, **kwargs):
                    raise ValueError(f"输入验证失败: {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_function_call(include_args=False, include_result=False):
    """
    函数调用日志装饰器

    Args:
        include_args: 是否记录参数
        include_result: 是否记录返回值
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            call_info = f"调用 {func.__name__}"
            if include_args:
                call_info += f" args={args}, kwargs={kwargs}"

            logger.debug(call_info)
            result = func(*args, **kwargs)

            if include_result:
                logger.debug(f"返回 {func.__name__}: {result}")

            return result
        return wrapper
    return decorator


class CircuitBreaker:
    """熔断器 - 防止级联失败"""

    def __init__(self, failure_threshold=5, timeout=60):
        """
        Args:
            failure_threshold: 失败次数阈值
            timeout: 熔断超时时间（秒）
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """
        通过熔断器调用函数

        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 关键字参数

        Returns:
            函数返回值
        """
        with self._lock:
            if self.is_open:
                if self._is_timeout_expired():
                    self._reset()
                else:
                    raise Exception("熔断器已打开，拒绝调用")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """成功回调"""
        with self._lock:
            self.failure_count = 0

    def _on_failure(self):
        """失败回调"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning(f"熔断器已打开（失败次数: {self.failure_count}）")

    def _reset(self):
        """重置熔断器"""
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None
        logger.info("熔断器已重置")

    def _is_timeout_expired(self):
        """检查超时是否已过期"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.timeout


# 全局熔断器实例
api_circuit_breaker = CircuitBreaker(failure_threshold=10, timeout=60)
