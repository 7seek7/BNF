import time
from datetime import datetime, timezone
from functools import wraps
from config.settings import settings
from utils.logger import Logger

logger = Logger.get_logger('helpers')

def retry_on_failure(max_retries=None, delay=None, exponential_backoff=True, initial_delay=1, max_delay=30):
    """失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 基础延迟（秒）
        exponential_backoff: 是否使用指数退避
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
    """
    if max_retries is None:
        max_retries = settings.MAX_RETRIES
    if delay is None:
        delay = settings.RETRY_DELAY
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 检查是否是网络错误（DNS解析失败、超时、连接错误）
                    error_str = str(e)
                    is_network_error = (
                        'NameResolutionError' in error_str or  # DNS解析失败
                        'getaddrinfo failed' in error_str or
                        'timed out' in error_str.lower() or  # 超时
                        'ConnectionError' in error_str or  # 连接错误
                        'Max retries exceeded' in error_str or
                        'timeout' in error_str.lower() or  # 其他超时
                        '104' in error_str or  # 104 Connection reset
                        'ECONNREFUSED' in error_str or
                        'ECONNRESET' in error_str
                    )
                    
                    if attempt < max_retries - 1:
                        # 计算重试延迟（指数退避）
                        if exponential_backoff:
                            retry_delay = min(initial_delay * (2 ** attempt), max_delay)
                        else:
                            retry_delay = delay * (attempt + 1)
                        
                        # 网络错误使用更长的延迟
                        if is_network_error:
                            retry_delay = min(retry_delay * 2, max_delay)
                        
                        logger.warning(f"{func.__name__} 失败，重试 {attempt + 1}/{max_retries} (延迟{retry_delay:.1f}s): {str(e)[:150]}")
                        time.sleep(retry_delay)
                    else:
                        # 网络错误给出更友好的提示
                        if is_network_error:
                            logger.error(f"{func.__name__} 失败，网络连接问题，已达最大重试次数: {str(e)[:150]}")
                        else:
                            logger.error(f"{func.__name__} 失败，已达最大重试次数: {str(e)[:150]}")
                        raise
            return None
        return wrapper
    return decorator

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
    except:
        return str(num)

def get_timestamp():
    """获取当前时间戳（毫秒）"""
    return int(time.time() * 1000)

def timestamp_to_datetime(timestamp):
    """时间戳转日期时间"""
    if timestamp > 1e12:  # 毫秒
        timestamp = timestamp / 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

def round_step_size(quantity, step_size):
    """将数量调整为step_size的整数倍（使用更精确的Decimal方法）"""
    try:
        if step_size <= 0:
            return quantity

        # 使用Decimal避免浮点数精度问题
        import decimal
        from decimal import Decimal, ROUND_HALF_UP

        quantity_dec = Decimal(str(quantity))
        step_dec = Decimal(str(step_size))

        # 计算倍数并四舍五入到最接近的整数
        multiplier = (quantity_dec / step_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP)

        # 计算调整后的数量
        adjusted_quantity = multiplier * step_dec

        # 确保不少于最小值
        if adjusted_quantity < step_dec:
            adjusted_quantity = step_dec

        return float(adjusted_quantity)
    except Exception as e:
        logger.error(f"调整数量步长失败: {str(e)}")
        return quantity

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

def round_tick_size(price, tick_size):
    """将价格调整为tick_size的整数倍"""
    try:
        if tick_size <= 0:
            return price

        # 使用更精确的方法：避免浮点数精度问题
        # 计算倍数时使用Decimal或者更精确的计算
        import decimal
        from decimal import Decimal, ROUND_HALF_UP

        # 将参数转换为Decimal以避免浮点数精度问题
        price_dec = Decimal(str(price))
        tick_dec = Decimal(str(tick_size))

        # 计算倍数并四舍五入到最接近的整数
        multiplier = (price_dec / tick_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP)

        # 计算调整后的价格
        adjusted_price = multiplier * tick_dec

        # 确保价格为正数
        if adjusted_price <= 0:
            adjusted_price = tick_dec

        return float(adjusted_price)
    except Exception as e:
        logger.error(f"调整价格步长失败: {str(e)}")
        return price

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