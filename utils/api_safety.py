"""
API安全装饰器 - 统一的API调用保护和错误处理

修复：safe_api_call 已移至 utils/retry.py，此处仅保留验证函数
"""
import functools
import time
from typing import Callable, Any
from framework.core.logger import get_logger

logger = get_logger('api_safety')

# 修复：从统一的 utils.retry 导入
from utils.retry import safe_api_call


def validate_symbol(func: Callable) -> Callable:
    """验证币种符号的装饰器"""
    @functools.wraps(func)
    def wrapper(self, symbol, *args, **kwargs):
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"无效的币种符号: {symbol}")
        
        # 标准化币种符号
        symbol = symbol.upper().strip()
        
        # 检查格式
        if not symbol.endswith('USDT'):
            logger.warning(f"币种符号可能不正确: {symbol}")
        
        return func(self, symbol, *args, **kwargs)
    return wrapper


def validate_quantity(func: Callable) -> Callable:
    """验证数量的装饰器"""
    @functools.wraps(func)
    def wrapper(self, symbol, quantity, *args, **kwargs):
        try:
            qty = float(quantity)
            if qty <= 0:
                raise ValueError(f"数量必须大于0: {qty}")
            if qty > 1e9:  # 不合理的巨大数量
                raise ValueError(f"数量异常巨大: {qty}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"无效的数量: {quantity}, 错误: {str(e)}")
        
        return func(self, symbol, quantity, *args, **kwargs)
    return wrapper


def validate_price(func: Callable) -> Callable:
    """验证价格的装饰器"""
    @functools.wraps(func)
    def wrapper(self, symbol, price, *args, **kwargs):
        try:
            p = float(price)
            if p <= 0:
                raise ValueError(f"价格必须大于0: {p}")
            if p > 1e9:  # 不合理的高价格
                logger.warning(f"{symbol} 价格异常高: {p}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"无效的价格: {price}, 错误: {str(e)}")
        
        return func(self, symbol, price, *args, **kwargs)
    return wrapper


def log_api_call(func: Callable) -> Callable:
    """记录API调用的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__

        try:
            symbol = args[1] if len(args) > 1 else kwargs.get('symbol', 'unknown')

            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            logger.debug(f"API调用成功: {func_name}({symbol}), 耗时: {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"API调用失败: {func_name}, 耗时: {elapsed:.3f}s, 错误: {str(e)}")
            raise e from e

    return wrapper
