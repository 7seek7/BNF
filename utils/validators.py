"""
价格数据验证器

功能：
- 验证价格数据合理性
- 检测异常数据（坏数据 vs 真实市场）
- 价格范围验证
"""

from typing import Dict, Optional, Tuple, Any
from config.settings import Settings
from framework.core.logger import get_logger

logger = get_logger('price_validator')


class PriceValidator:
    """价格数据验证器"""
    
    # 币种价格范围参考（美元）
    PRICE_RANGES = {
        'BTCUSDT': (10_000, 200_000),
        'ETHUSDT': (500, 10_000),
        'BNBUSDT': (200, 2_000),
        'ADAUSDT': (0.2, 5),
        'SOLUSDT': (10, 300),
        'XRPUSDT': (0.2, 5),
        'DOGEUSDT': (0.05, 2),
        'DOTUSDT': (2, 50),
        'LTCUSDT': (50, 300),
        # 添加更多币种...
    }
    
    def __init__(self):
        """初始化价格验证器"""
        self.price_history = {}  # {symbol: last_valid_price}
        self.price_history_count = {}  # {symbol: count}
        # 修复：使用实例变量避免线程安全问题（BUG-M003）
        self.PRICE_RANGES = dict(PriceValidator.PRICE_RANGES)
    
    def validate(self, symbol: str, price: float, 
                 previous_price: Optional[float] = None) -> Tuple[bool, str]:
        """
        验证价格数据
        
        Args:
            symbol: 币种
            price: 当前价格
            previous_price: 前一次价格（可选）
        
        Returns:
            (is_valid, reason) 验证结果和原因
        """
        # 1. 基础验证
        if not isinstance(price, (int, float)):
            return False, "价格必须是数字"
        
        if price <= 0:
            return False, "价格必须大于0"
        
        # 2. 检查是否为有限数值
        if not self._is_finite(price):
            return False, "价格不是有效数值（可能是NaN或Infinity）"
        
        # 3. 价格范围验证
        is_valid, reason = self.validate_range(symbol, price)
        if not is_valid:
            return False, reason
        
        # 4. 价格变化验证（如果有历史价格）
        if previous_price and previous_price > 0:
            is_valid, reason = self._validate_change(symbol, previous_price, price)
            if not is_valid:
                return False, reason
        
        # 5. 更新价格历史
        self.price_history[symbol] = price
        self.price_history_count[symbol] = self.price_history_count.get(symbol, 0) + 1
        
        return True, "OK"
    
    def _is_finite(self, price: float) -> bool:
        """检查数值是否有限"""
        try:
            # 检查NaN和Infinity
            import math
            return math.isfinite(price)
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.debug(f"检查数值有限性时发生异常: {e}")
            return False
    
    def validate_range(self, symbol: str, price: float) -> Tuple[bool, str]:
        """验证价格在合理范围内"""
        # 如果没有配置该币种的范围，使用默认范围
        min_price, max_price = self.PRICE_RANGES.get(symbol.upper(), (0.0001, 1_000_000))
        
        # 修复：限制范围膨胀次数，防止无限增长
        # 允许最多膨胀10次（100^10 = 10^20 倍）
        current_expansion_count = getattr(self, '_price_range_expansion_count', {}).get(symbol.upper(), 0)
        
        if price > max_price and current_expansion_count < 10:
            new_max = max_price * 100
            self.PRICE_RANGES[symbol.upper()] = (min_price, new_max)
            
            # 记录膨胀次数
            if not hasattr(self, '_price_range_expansion_count'):
                self._price_range_expansion_count = {}
            self._price_range_expansion_count[symbol.upper()] = current_expansion_count + 1
            
            max_price = new_max
        elif price > max_price:
            # 超过膨胀限制，记录警告但仍然允许
            logger.warning(f"{symbol} 价格范围已膨胀至上限，价格{price}超过{max_price}")
        
        # 检查是否在范围内
        if price < min_price or price > max_price:
            return False, f"价格{price}超出合理范围[{min_price}, {max_price}]"
        
        return True, "OK"
    
    def validate_change(self, symbol: str, previous_price: float, current_price: float) -> Tuple[bool, str]:
        """验证价格变化是否合理"""
        change_pct = abs((current_price - previous_price) / previous_price * 100)
        threshold = getattr(Settings, 'PRICE_CHANGE_THRESHOLD', 50)

        if change_pct > threshold:
            logger.warning(
                f"{symbol} 价格异常波动: {previous_price} -> {current_price} "
                f"({change_pct:.1f}%), 阈值: {threshold}%"
            )

        if change_pct > 90:
            return False, f"价格波动过大: {change_pct:.1f}%，可能存在坏数据"

        return True, "OK"
    
    def get_last_valid_price(self, symbol: str) -> Optional[float]:
        """获取最后一次验证通过的价格"""
        return self.price_history.get(symbol)
    
    def get_price_history_count(self, symbol: str) -> int:
        """获取价格历史记录数量"""
        return self.price_history_count.get(symbol, 0)


class VolumeValidator:
    """成交量数据验证器"""
    
    def validate(self, symbol: str, volume: float, 
                 previous_volume: Optional[float] = None) -> Tuple[bool, str]:
        """
        验证成交量数据
        
        Args:
            symbol: 币种
            volume: 成交量
            previous_volume: 前一次成交量（可选）
        
        Returns:
            (is_valid, reason) 验证结果和原因
        """
        # 基础验证
        if volume <= 0:
            return False, "成交量必须大于0"
        
        if not isinstance(volume, (int, float)):
            return False, "成交量必须是数字"
        
        # 检查是否为有限数值
        if not self._is_finite(volume):
            return False, "成交量不是有效数值"
        
        # 合理性检查（成交量不应突然暴增暴减超过100倍）
        if previous_volume and previous_volume > 0:
            ratio = volume / previous_volume
            if ratio > 100 or ratio < 0.01:
                logger.warning(
                    f"{symbol} 成交量异常波动: {previous_volume} -> {volume} "
                    f"(倍数: {ratio:.2f})"
                )
        
        return True, "OK"
    
    def _is_finite(self, volume: float) -> bool:
        """检查数值是否有限"""
        try:
            import math
            return math.isfinite(volume)
        except (TypeError, ValueError):
            return False


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        """初始化数据质量检查器"""
        self.price_validator = PriceValidator()
        self.volume_validator = VolumeValidator()
        logger.info("数据质量检查器已初始化")
    
    def check_kline_data(self, kline: Dict[str, any]) -> Tuple[bool, str]:
        """
        检查K线数据质量
        
        Args:
            kline: K线数据字典，应包含: open, high, low, close, volume
        
        Returns:
            (is_valid, reason) 验证结果和原因
        """
        # 提取K线数据
        open_price = float(kline.get('open', 0))
        high_price = float(kline.get('high', 0))
        low_price = float(kline.get('low', 0))
        close_price = float(kline.get('close', 0))
        volume = float(kline.get('volume', 0))
        symbol = kline.get('symbol', '')
        
        # 1. 验证基本数值
        for name, value in [('open', open_price), ('high', high_price), 
                             ('low', low_price), ('close', close_price), ('volume', volume)]:
            if value <= 0:
                return False, f"{name}价格必须大于0 (实际值: {value})"
        
        # 2. 验证OHLC逻辑 (high >= low, close在[low, high]之间)
        if high_price < low_price:
            return False, f"最高价{high_price}不能低于最低价{low_price}"

        if close_price < low_price or close_price > high_price:
            logger.warning(
                f"{symbol} K线数据异常: close_price={close_price}, "
                f"范围应为[{low_price}, {high_price}]"
            )
            return False, f"收盘价{close_price}超出合理范围[{low_price}, {high_price}]"
        
        # 3. 验证价格数据
        is_valid, reason = self.price_validator.validate(symbol, close_price)
        if not is_valid:
            return False, f"价格验证失败: {reason}"
        
        # 4. 验证成交量
        is_valid, reason = self.volume_validator.validate(symbol, volume)
        if not is_valid:
            return False, f"成交量验证失败: {reason}"
        
        return True, "OK"
    
    def check_tick_data(self, symbol: str, price: float,
                      previous_price: float) -> Tuple[bool, str]:
        """
        检查tick数据质量

        Args:
            symbol: 币种
            price: 当前价格
            previous_price: 前一次价格

        Returns:
            (is_valid, reason) 验证结果和原因
        """
        # 1. 检查价格是否有限数值
        if not self._is_finite(price):
            return False, "价格不是有效数值"

        # 2. 检查价格变化是否合理
        if previous_price and previous_price > 0:
            is_valid, reason = self.price_validator.validate_change(symbol, previous_price, price)
            if not is_valid:
                return False, reason

        # 3. 检查价格是否在合理范围内
        is_valid, reason = self.price_validator.validate_range(symbol, price)
        if not is_valid:
            return False, reason

        return True, "OK"


class PositionValidator:
    """位置字段验证器"""

    @staticmethod
    def validate_quantity(quantity: float) -> bool:
        """验证持仓数量是否有效"""
        return quantity > 0 and quantity < 1_000_000  # 防止极端值

    @staticmethod
    def validate_entry_price(price: float) -> bool:
        """验证入场价格是否有效"""
        return price > 0.00000001 and price < 1_000_000

    @staticmethod
    def validate_profit_pct(pct: float) -> bool:
        """验证盈亏百分比是否在合理范围内"""
        return -100 <= pct <= 10000  # -100%到10000%

    @staticmethod
    def validate_leverage(leverage: int) -> bool:
        """验证杠杆倍数是否有效"""
        return 1 <= leverage <= 125

    @staticmethod
    def clamp_value(value: float, min_val: float, max_val: float) -> float:
        """将值限制在指定范围内"""
        return max(min_val, min(value, max_val))

    @classmethod
    def sanitize_position_field(cls, field_name: str, value: Any) -> Any:
        """清理和验证位置字段值"""
        if field_name == 'total_quantity':
            if isinstance(value, (int, float)):
                return cls.clamp_value(value, 0, 1_000_000)
        elif field_name == 'entry_price':
            if isinstance(value, (int, float)):
                return cls.clamp_value(value, 0.00000001, 1_000_000)
        elif field_name == 'profit_pct':
            if isinstance(value, (int, float)):
                return cls.clamp_value(value, -100, 10000)
        elif field_name == 'leverage':
            if isinstance(value, int):
                return max(1, min(value, 125))
        return value


# 全局实例（延迟初始化）
_data_quality_checker = None

def get_data_quality_checker() -> DataQualityChecker:
    """获取数据质量检查器实例（单例模式）"""
    global _data_quality_checker
    if _data_quality_checker is None:
        _data_quality_checker = DataQualityChecker()
    return _data_quality_checker
