"""
输入验证模块 - 防止无效输入导致的交易错误
"""
import re
from typing import Tuple, Optional, Any
from framework.core.logger import get_logger

logger = get_logger('validation')


class InputValidator:
    """交易输入验证器"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """
        验证币种符号格式
        
        Returns:
            (is_valid, error_message)
        """
        if not symbol:
            return False, "币种符号不能为空"
        
        if not isinstance(symbol, str):
            return False, f"币种符号必须是字符串，当前类型: {type(symbol)}"
        
        # 币安格式: BTCUSDT, ETHUSDT, etc.
        if not re.match(r'^[A-Z0-9]{2,15}USDT$', symbol.upper()):
            return False, f"币种符号格式错误: {symbol}，应为如 BTCUSDT 格式"
        
        return True, ""
    
    @staticmethod
    def validate_quantity(quantity: float, min_qty: float = 0, max_qty: float = None) -> Tuple[bool, str]:
        """
        验证交易数量
        
        Returns:
            (is_valid, error_message)
        """
        if quantity is None:
            return False, "数量不能为空"
        
        try:
            qty = float(quantity)
        except (TypeError, ValueError):
            return False, f"数量必须是数字，当前值: {quantity}"
        
        if qty <= 0:
            return False, f"数量必须大于0，当前值: {qty}"
        
        if qty < min_qty:
            return False, f"数量 {qty} 小于最小值 {min_qty}"
        
        if max_qty is not None and qty > max_qty:
            return False, f"数量 {qty} 超过最大值 {max_qty}"
        
        return True, ""
    
    @staticmethod
    def validate_price(price: float, min_price: float = 0) -> Tuple[bool, str]:
        """
        验证价格
        
        Returns:
            (is_valid, error_message)
        """
        if price is None:
            return False, "价格不能为空"
        
        try:
            p = float(price)
        except (TypeError, ValueError):
            return False, f"价格必须是数字，当前值: {price}"
        
        if p <= min_price:
            return False, f"价格必须大于{min_price}，当前值: {p}"
        
        return True, ""
    
    @staticmethod
    def validate_leverage(leverage: int, max_leverage: int = 125) -> Tuple[bool, str]:
        """
        验证杠杆倍数
        
        Returns:
            (is_valid, error_message)
        """
        if leverage is None:
            return False, "杠杆不能为空"
        
        try:
            lev = int(leverage)
        except (TypeError, ValueError):
            return False, f"杠杆必须是整数，当前值: {leverage}"
        
        if lev <= 0:
            return False, f"杠杆必须大于0，当前值: {lev}"
        
        if lev > max_leverage:
            return False, f"杠杆 {lev}x 超过最大值 {max_leverage}x"
        
        # 检查是否为标准档位
        standard_leverages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                             25, 30, 35, 40, 45, 50, 75, 100, 125]
        if lev not in standard_leverages:
            logger.warning(f"杠杆 {lev}x 不是标准档位，可能需要调整")
        
        return True, ""
    
    @staticmethod
    def validate_margin(margin: float, available_funds: float = None) -> Tuple[bool, str]:
        """
        验证保证金
        
        Returns:
            (is_valid, error_message)
        """
        if margin is None:
            return False, "保证金不能为空"
        
        try:
            m = float(margin)
        except (TypeError, ValueError):
            return False, f"保证金必须是数字，当前值: {margin}"
        
        if m <= 0:
            return False, f"保证金必须大于0，当前值: {m}"
        
        if available_funds is not None and m > available_funds:
            return False, f"保证金 {m:.2f} 超过可用资金 {available_funds:.2f}"
        
        return True, ""
    
    @staticmethod
    def validate_order_params(symbol: str, quantity: float, price: float = None, 
                            leverage: int = None, side: str = None) -> Tuple[bool, str]:
        """
        验证订单参数完整性
        
        Returns:
            (is_valid, error_message)
        """
        # 验证币种
        valid, msg = InputValidator.validate_symbol(symbol)
        if not valid:
            return False, msg
        
        # 验证数量
        valid, msg = InputValidator.validate_quantity(quantity)
        if not valid:
            return False, msg
        
        # 验证价格（如果提供）
        if price is not None:
            valid, msg = InputValidator.validate_price(price)
            if not valid:
                return False, msg
        
        # 验证杠杆（如果提供）
        if leverage is not None:
            valid, msg = InputValidator.validate_leverage(leverage)
            if not valid:
                return False, msg
        
        # 验证方向
        if side is not None:
            if side not in ['BUY', 'SELL', 'LONG', 'SHORT']:
                return False, f"交易方向必须是 BUY/SELL/LONG/SHORT，当前值: {side}"
        
        return True, ""


# ==================== 持仓方向标准化 ====================


def normalize_position_side(side: Any, position_amt: float = None, default: str = 'LONG') -> str:
    """
    标准化持仓方向，支持中英文、买/卖/LONG/SHORT 及缺失推断

    Args:
        side: 原始方向值（任意类型）
        position_amt: 持仓数量（用于推断）
        default: 转换失败时的默认方向

    Returns:
        'LONG' 或 'SHORT'
    """
    if side is None and position_amt is not None:
        # 缺失方向时按正负持仓量推断
        return 'LONG' if position_amt > 0 else 'SHORT'

    if side is None:
        return default

    if isinstance(side, str):
        s = side.strip().upper()
        if s in ('LONG', '做多', '买', 'BUY'):
            return 'LONG'
        if s in ('SHORT', '做空', '卖', 'SELL'):
            return 'SHORT'
        # positionSide 字段可能返回 'BOTH' 等
        if s in ('BOTH', 'NET'):
            return default
        # 尝试按正负推断
        try:
            amt = float(side)
            return 'LONG' if amt > 0 else 'SHORT'
        except (ValueError, TypeError):
            pass
        return default

    if isinstance(side, bool):
        return 'LONG' if side else 'SHORT'

    # 其他类型尝试数值推断
    try:
        amt = float(side)
        return 'LONG' if amt > 0 else 'SHORT'
    except (ValueError, TypeError):
        return default


def normalize_order_side(side: Any, position_side: str = None, position_amt: float = None,
                      default: str = 'BUY') -> str:
    """
    标准化订单方向（平仓方向），支持中英文并设置默认值

    平仓原则：
    - LONG 持仓 → SELL 平仓
    - SHORT 持仓 → BUY 平仓
    - 无持仓信息时默认 BUY

    Args:
        side: 原始方向值（任意类型）
        position_side: 持仓方向（LONG/SHORT）
        position_amt: 持仓数量（用于推断方向）
        default: 转换失败时的默认方向

    Returns:
        'BUY' 或 'SELL'
    """
    if side is not None and isinstance(side, str):
        s = side.strip().upper()
        if s in ('BUY', 'SELL', '买', '卖', '多', '空'):
            return s if s in ('BUY', 'SELL') else ('BUY' if s in ('买', '多') else 'SELL')

    # 无显式方向时，按持仓方向推断平仓方向
    ps = normalize_position_side(position_side, position_amt, default=None)
    if ps == 'LONG':
        return 'SELL'
    if ps == 'SHORT':
        return 'BUY'

    return default


def get_position_direction(side: Any, position_amt: float = None) -> Tuple[str, str]:
    """
    获取并补全持仓方向字段

    Returns:
        (position_side, order_side) 元组
        - position_side: 'LONG'/'SHORT'
        - order_side: 'BUY'/'SELL'
    """
    position_side = normalize_position_side(side, position_amt, default='LONG')
    order_side = normalize_order_side(None, position_side, position_amt, default='BUY')
    return position_side, order_side


def safe_divide(numerator: float, denominator: float, default: float = 0.0, 
                context: str = "") -> float:
    """
    安全的除法操作，防止除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        context: 错误上下文信息
    
    Returns:
        除法结果或默认值
    """
    try:
        if denominator == 0 or denominator is None:
            if context:
                logger.error(f"除零错误: {context}, 分子={numerator}, 分母={denominator}")
            return default
        return numerator / denominator
    except Exception as e:
        if context:
            logger.error(f"除法错误: {context}, {str(e)}")
        return default


def safe_calculate_pnl(current_price: float, entry_price: float,
                      leverage: int, direction: str, symbol: str = "") -> float:
    """
    安全计算盈亏率

    Returns:
        盈亏率（百分比）或0.0
    """
    if entry_price <= 0:
        if symbol:
            logger.error(f"{symbol} 入场价无效: {entry_price}")
        return 0.0

    if leverage <= 0:
        if symbol:
            logger.error(f"{symbol} 杠杆无效: {leverage}")
        return 0.0

    try:
        if direction == 'LONG':
            pnl = ((current_price - entry_price) / entry_price) * leverage * 100
        else:  # SHORT
            pnl = ((entry_price - current_price) / entry_price) * leverage * 100
        return pnl
    except Exception as e:
        if symbol:
            logger.error(f"{symbol} 盈亏计算错误: {str(e)}")
        return 0.0


# ==================== 防御性类型转换 ====================


def safe_float(value: Any, default: float = 0.0, context: str = "") -> float:
    """
    安全的浮点数转换，处理 None 和非法字符串

    Args:
        value: 待转换的值（任意类型）
        default: 转换失败时的默认值
        context: 错误上下文信息

    Returns:
        转换后的浮点数或默认值
    """
    if value is None:
        return default
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ('none', 'null', 'nan', ''):
            return default
        # 处理带符号的字符串如 "+1.5" 或 "-0.5"
        try:
            return float(value)
        except ValueError:
            if context:
                logger.warning(f"safe_float 转换失败: {context}, 值='{value}'")
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if context:
            logger.warning(f"safe_float 转换失败: {context}, 类型={type(value)}, 值={value}")
        return default


def safe_int(value: Any, default: int = 0, context: str = "") -> int:
    """
    安全的整数转换，处理 None 和非法字符串

    Args:
        value: 待转换的值（任意类型）
        default: 转换失败时的默认值
        context: 错误上下文信息

    Returns:
        转换后的整数或默认值
    """
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # 浮点数转整数：有小数部分时四舍五入
        if value != int(value):
            if context:
                logger.warning(f"safe_int: 浮点数 {value} 将四舍五入")
            return int(round(value))
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ('none', 'null', 'nan', ''):
            return default
        # 处理十六进制
        if value.startswith('0x'):
            try:
                return int(value, 16)
            except ValueError:
                if context:
                    logger.warning(f"safe_int 转换失败: {context}, 值='{value}'")
                return default
        try:
            return int(float(value))  # 处理 "1.0" 这种形式
        except ValueError:
            if context:
                logger.warning(f"safe_int 转换失败: {context}, 值='{value}'")
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        if context:
            logger.warning(f"safe_int 转换失败: {context}, 类型={type(value)}, 值={value}")
        return default


def safe_bool(value: Any, default: bool = False, context: str = "") -> bool:
    """
    安全的布尔转换，处理 None 和各种字符串形式

    支持的值（不区分大小写）：
    - True类: True, "true", "1", "yes", "on", "enabled"
    - False类: False, "false", "0", "no", "off", "disabled", ""

    Args:
        value: 待转换的值（任意类型）
        default: 转换失败时的默认值
        context: 错误上下文信息

    Returns:
        转换后的布尔值或默认值
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, float):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('true', '1', 'yes', 'on', 'enabled'):
            return True
        if value in ('false', '0', 'no', 'off', 'disabled', '', 'none', 'null', 'nan'):
            return False
        if context:
            logger.warning(f"safe_bool 未知值: {context}, 值='{value}'")
        return default
    try:
        return bool(value)
    except (TypeError, ValueError):
        if context:
            logger.warning(f"safe_bool 转换失败: {context}, 类型={type(value)}, 值={value}")
        return default
