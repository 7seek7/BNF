from typing import Optional

import time
from datetime import datetime, timezone
from functools import wraps
from framework.core.config import get_main_config
from config.settings import Settings
from framework.core.logger import get_logger
from decimal import Decimal, ROUND_HALF_UP

logger = get_logger('helpers')

# 延迟初始化Settings实例
_settings_instance = None

def _get_settings():
    """获取Settings实例（延迟初始化）"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

def retry_on_failure(max_retries=None, delay=None, exponential_backoff=True, initial_delay=1, max_delay=30):
    """失败重试装饰器（委托至 utils.retry）
    
    从 Settings 读取默认值，然后交由 utils.retry.retry_on_failure 执行。
    Args:
        max_retries: 最大重试次数
        delay: 基础延迟（秒）
        exponential_backoff: 是否使用指数退避
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
    """
    if max_retries is None:
        max_retries = _get_settings().MAX_RETRIES
    if delay is None:
        delay = _get_settings().RETRY_DELAY
    from utils.retry import retry_on_failure as _retry
    return _retry(max_retries=max_retries, delay=delay,
                  exponential_backoff=exponential_backoff,
                  initial_delay=initial_delay, max_delay=max_delay)

def format_number(num, decimals=2):
    """格式化数字"""
    try:
        if abs(num) >= 1e9:
            return f"{num/1e9:.{decimals}f}B"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.{decimals}f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.{decimals}f}K"
        else:
            return f"{num:.{decimals}f}"
    except (TypeError, ValueError, KeyError):
        return str(num)

def get_timestamp():
    """获取当前时间戳（毫秒）"""
    return int(time.time() * 1000)

def timestamp_to_datetime(timestamp):
    """时间戳转日期时间"""
    if timestamp > 1e12:  # 毫秒
        timestamp = timestamp / 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

def _round_to_step(value: float, step: float, min_value: float | None = None) -> float:
    """
    通用函数：将数值调整为步长的整数倍（使用Decimal避免浮点精度问题）
    
    Args:
        value: 原始值
        step: 步长
        min_value: 最小值（可选）
    
    Returns:
        float: 调整后的值
    """
    try:
        if step <= 0:
            return value
        
        value_dec = Decimal(str(value))
        step_dec = Decimal(str(step))
        
        # 计算倍数并四舍五入
        multiplier = (value_dec / step_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        adjusted = multiplier * step_dec
        
        # 确保不小于最小值
        if min_value is not None and adjusted < Decimal(str(min_value)):
            adjusted = Decimal(str(min_value))
        elif adjusted < step_dec:
            adjusted = step_dec
        
        return float(adjusted)
    except Exception as e:
        logger.error(f"调整步长失败: {str(e)}")
        return value


def round_step_size(quantity, step_size):
    """将数量调整为step_size的整数倍"""
    return _round_to_step(quantity, step_size, step_size)


def round_tick_size(price, tick_size):
    """将价格调整为tick_size的整数倍"""
    return _round_to_step(price, tick_size, tick_size)

def adjust_quantity_precision(symbol_info, quantity):
    """根据交易规则调整数量精度（正确实现）"""
    try:
        if not symbol_info:
            return quantity

        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f.get('minQty', 0))
                step_size = float(f.get('stepSize', 0.001))

                # 如果数量小于最小值，返回最小值
                if quantity < min_qty:
                    logger.warning(f"数量 {quantity} 小于最小值 {min_qty}，返回最小值")
                    return min_qty

                # 按步长调整（确保是step_size的整数倍）
                quantity = round_step_size(quantity, step_size)

                # 再次检查是否小于最小值
                if quantity < min_qty:
                    quantity = min_qty

                return quantity

        return quantity
    except Exception as e:
        logger.error(f"调整数量精度失败: {str(e)}")
        return quantity

def adjust_price_precision(symbol_info, price):
    """根据交易规则调整价格精度"""
    try:
        if not symbol_info:
            return price

        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'PRICE_FILTER':
                tick_size = float(f['tickSize'])

                # 按步长调整
                price = round_tick_size(price, tick_size)

                return price

        return price
    except Exception as e:
        logger.error(f"调整价格精度失败: {str(e)}")
        return price

def calculate_position_size(available_balance, position_count, price, leverage):
    """计算仓位大小"""
    try:
        if position_count <= 0:
            position_count = 1
        
        # 每个币种可用金额
        per_symbol_balance = available_balance / position_count
        
        # 计算数量
        quantity = (per_symbol_balance * leverage) / price
        
        return quantity, per_symbol_balance
    except Exception as e:
        logger.error(f"计算仓位大小失败: {str(e)}")
        return 0, 0

def align_to_interval(interval_minutes):
    """对齐到K线时间戳"""
    try:
        current_time = time.time()
        interval_seconds = interval_minutes * 60
        
        # 计算下一个整点时间
        next_time = ((current_time // interval_seconds) + 1) * interval_seconds
        wait_seconds = next_time - current_time
        
        if wait_seconds > 0:
            logger.info(f"等待 {wait_seconds:.1f} 秒对齐到下一个 {interval_minutes} 分钟K线")
            time.sleep(wait_seconds)
        
        return True
    except Exception as e:
        logger.error(f"时间对齐失败: {str(e)}")
        return False