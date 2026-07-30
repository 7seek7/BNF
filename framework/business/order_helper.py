# -*- coding: utf-8 -*-
"""
公共订单处理模块 - 统一处理订单参数调整

功能：
1. 从币安API获取各币种的限制信息（杠杆档位，数量精度等）
2. 根据限制调整订单参数
3. 返回调整后的参数给调用方

所有策略的订单都通过此模块处理，确保符合币安规则
"""

from typing import Dict, Optional
from dataclasses import dataclass
from framework.core.logger import get_logger

logger = get_logger('order_helper')


@dataclass
class OrderAdjustment:
    """调整后的订单参数"""
    leverage: int
    quantity: float
    price: Optional[float] = None
    
    def __post_init__(self):
        """确保返回的是原始类型的字符串表示"""
        self.leverage = int(self.leverage)
        self.quantity = float(self.quantity)
        if self.price is not None:
            self.price = float(self.price)


class OrderHelper:
    """
    公共订单处理器
    
    使用方式：
        helper = OrderHelper(client)
        
        # 获取并调整订单参数
        adjusted = helper.adjust_order(
            symbol='BTCUSDT',
            leverage=20,
            quantity=0.1,
            price=50000.0
        )
        
        # 使用调整后的参数下单
        client.set_leverage(symbol, adjusted.leverage)
        client.create_order(symbol, quantity=adjusted.quantity, ...)
    """
    
    def __init__(self, client):
        """
        初始化
        
        Args:
            client: BinanceClient 实例
        """
        self._client = client
        
        # 缓存
        self._leverage_cache: Dict[str, int] = {}  # symbol -> max_leverage
        self._filters_cache: Dict[str, Dict] = {}  # symbol -> filters
        self._cache_time: Dict[str, float] = {}
        
        # 缓存有效期（秒）
        self._cache_ttl = 300  # 5分钟
    
    def get_max_leverage(self, symbol: str) -> int:
        """
        获取币种允许的最大杠杆倍数
        
        Args:
            symbol: 币种符号
            
        Returns:
            int: 最大杠杆倍数
        """
        import time
        
        # 检查缓存
        current_time = time.time()
        if symbol in self._leverage_cache:
            if current_time - self._cache_time.get(symbol, 0) < self._cache_ttl:
                return self._leverage_cache[symbol]
        
        # 从API获取
        try:
            max_leverage = self._client.get_leverage_bracket(symbol)
            self._leverage_cache[symbol] = max_leverage
            self._cache_time[symbol] = current_time
            logger.debug(f"{symbol} 最大杠杆: {max_leverage}x")
            return max_leverage
        except Exception as e:
            logger.warning(f"获取杠杆限制失败 {symbol}: {e}，使用默认值20")
            return 20
    
    def get_symbol_filters(self, symbol: str) -> Dict:
        """
        获取币种的交易规则
        
        Args:
            symbol: 币种符号
            
        Returns:
            Dict: 包含 minQty, stepSize, tickSize 等
        """
        import time
        
        # 检查缓存
        current_time = time.time()
        if symbol in self._filters_cache:
            if current_time - self._cache_time.get(symbol, 0) < self._cache_ttl:
                return self._filters_cache[symbol]
        
        # 从API获取
        try:
            filters = self._client.get_symbol_filters(symbol)
            self._filters_cache[symbol] = filters
            self._cache_time[symbol] = current_time
            return filters
        except Exception as e:
            logger.warning(f"获取交易规则失败 {symbol}: {e}")
            return {}
    
    def adjust_leverage(self, symbol: str, requested_leverage: int) -> int:
        """
        调整杠杆到币种允许的范围
        
        Args:
            symbol: 币种符号
            requested_leverage: 请求的杠杆
            
        Returns:
            int: 调整后的杠杆
        """
        max_leverage = self.get_max_leverage(symbol)
        
        if requested_leverage > max_leverage:
            logger.warning(
                f"{symbol} 请求杠杆{requested_leverage}x超过限制{max_leverage}x，"
                f"已调整为{max_leverage}x"
            )
            return max_leverage
        
        if requested_leverage < 1:
            logger.warning(f"{symbol} 请求杠杆{requested_leverage}x小于1，使用1x")
            return 1
        
        return requested_leverage
    
    def adjust_quantity(self, symbol: str, quantity: float) -> float:
        """
        调整数量到币种允许的精度和范围
        
        Args:
            symbol: 币种符号
            quantity: 请求的数量
            
        Returns:
            float: 调整后的数量
        """
        filters = self.get_symbol_filters(symbol)
        
        if not filters:
            # 无法获取规则，使用默认取整
            return round(quantity, 3)
        
        try:
            # 获取步长
            step_size = float(filters.get('stepSize', 0.001))
            min_qty = float(filters.get('minQty', 0.001))
            max_qty = float(filters.get('maxQty', 1000000))
            
            # 防御：stepSize为0或负数时使用默认值
            if step_size <= 0:
                logger.warning(f"{symbol} stepSize={filters.get('stepSize')} 无效，使用默认值0.001")
                step_size = 0.001
            
            # 确保不小于最小值
            if quantity < min_qty:
                logger.warning(
                    f"{symbol} 数量{quantity}小于最小值{min_qty}，已调整为{min_qty}"
                )
                return min_qty
            
            # 确保不超过最大值
            if quantity > max_qty:
                logger.warning(
                    f"{symbol} 数量{quantity}超过最大值{max_qty}，已调整为{max_qty}"
                )
                return max_qty
            
            # 按步长取整
            adjusted = round(quantity / step_size) * step_size
            
            # 确保调整后仍然大于最小值
            if adjusted < min_qty:
                adjusted = min_qty
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"调整数量失败 {symbol}: {e}，使用原始值")
            return quantity
    
    def adjust_price(self, symbol: str, price: float) -> float:
        """
        调整价格到币种允许的精度
        
        Args:
            symbol: 币种符号
            price: 请求的价格
            
        Returns:
            float: 调整后的价格
        """
        filters = self.get_symbol_filters(symbol)
        
        if not filters:
            return price
        
        try:
            tick_size = float(filters.get('tickSize', 0.01))
            
            # 防御：tick_size为0或负数时使用默认值
            if tick_size <= 0:
                logger.warning(f"{symbol} tick_size={filters.get('tickSize')} 无效，使用默认值0.01")
                tick_size = 0.01
            
            adjusted = round(price / tick_size) * tick_size
            return adjusted
        except Exception as e:
            logger.warning(f"调整价格失败 {symbol}: {e}")
            return price
    
    def adjust_order(
        self,
        symbol: str,
        leverage: int,
        quantity: float,
        price: Optional[float] = None
    ) -> OrderAdjustment:
        """
        统一调整订单参数
        
        一次性调整杠杆、数量、价格，符合币安规则
        
        Args:
            symbol: 币种符号
            leverage: 请求的杠杆
            quantity: 请求的数量
            price: 请求的价格（可选）
            
        Returns:
            OrderAdjustment: 调整后的参数
        """
        # 调整杠杆
        adjusted_leverage = self.adjust_leverage(symbol, leverage)
        
        # 调整数量
        adjusted_quantity = self.adjust_quantity(symbol, quantity)
        
        # 调整价格
        adjusted_price = None
        if price is not None:
            adjusted_price = self.adjust_price(symbol, price)
        
        # 日志记录
        changes = []
        if adjusted_leverage != leverage:
            changes.append(f"杠杆{leverage}→{adjusted_leverage}")
        if abs(adjusted_quantity - quantity) > 0.0001:
            changes.append(f"数量{quantity}→{adjusted_quantity}")
        if adjusted_price is not None and price is not None:
            if abs(adjusted_price - price) > 0.01:
                changes.append(f"价格{price}→{adjusted_price}")
        
        if changes:
            logger.info(f"{symbol} 订单参数调整: {', '.join(changes)}")
        
        return OrderAdjustment(
            leverage=adjusted_leverage,
            quantity=adjusted_quantity,
            price=adjusted_price
        )
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        清除缓存
        
        Args:
            symbol: 币种符号，为None则清除所有
        """
        if symbol:
            self._leverage_cache.pop(symbol, None)
            self._filters_cache.pop(symbol, None)
            self._cache_time.pop(symbol, None)
        else:
            self._leverage_cache.clear()
            self._filters_cache.clear()
            self._cache_time.clear()
