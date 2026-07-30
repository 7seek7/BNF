# -*- coding: utf-8 -*-
"""
统一重试机制
"""
import time
import functools
from typing import Callable, Optional, Tuple, Type
from functools import wraps
from framework.core.logger import get_logger

logger = get_logger('retry')

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
    ConnectionResetError,
    ConnectionAbortedError,
)

try:
    from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceConnectionException
    BINANCE_RETRYABLE = (BinanceConnectionException, BinanceRequestException)
except ImportError:
    BINANCE_RETRYABLE = ()
    BinanceAPIException = Exception

def _is_network_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return any(k in error_str for k in ['name resolution', 'getaddrinfo failed', 'timed out', 'connectionerror', 'timeout', '104', 'econnrefused', 'econnreset'])

def _is_fatal_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return any(k in error_str for k in ['invalid api key', 'unauthorized', 'forbidden', 'signature'])

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True, initial_delay: float = 1.0, max_delay: float = 30.0):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if _is_fatal_error(e):
                        logger.error(f"{func.__name__} 致命错误: {str(e)[:150]}")
                        raise
                    if attempt < max_retries - 1:
                        retry_delay = min(initial_delay * (2 ** attempt), max_delay) if exponential_backoff else delay * (attempt + 1)
                        if _is_network_error(e):
                            retry_delay = min(retry_delay * 2, max_delay)
                        logger.warning(f"{func.__name__} 重试 {attempt + 1}/{max_retries}: {str(e)[:150]}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{func.__name__} 达最大重试: {str(e)[:150]}")
            raise last_exception
        return wrapper
    return decorator

def api_retry(max_attempts: int = 3, min_wait: int = 1, max_wait: int = 10):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS + BINANCE_RETRYABLE as e:
                    last_exception = e
                    if _is_fatal_error(e): raise
                    if attempt < max_attempts - 1:
                        wait_time = min(min_wait * (2 ** attempt), max_wait)
                        logger.warning(f"重试 {func.__name__} (第{attempt+1}次): {str(e)[:100]}")
                        time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator

def safe_api_call(max_retries: int = 3, retry_delay: float = 1.0, exceptions: Tuple[Type[Exception], ...] = (Exception,), on_failure: Optional[Callable] = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if _is_fatal_error(e): break
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt)
                        logger.warning(f"API失败 ({attempt+1}/{max_retries+1}): {str(e)}, {delay:.1f}秒后重试")
                        time.sleep(delay)
            if on_failure:
                try: return on_failure(last_exception, *args, **kwargs)
                except Exception as cb: logger.error(f"失败回调出错: {str(cb)}")
            raise last_exception
        return wrapper
    return decorator

__all__ = ['retry_on_failure', 'api_retry', 'safe_api_call', 'RETRYABLE_EXCEPTIONS', 'BINANCE_RETRYABLE']