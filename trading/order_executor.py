"""
订单执行器接口 - 定义实盘和回测的统一接口

职责：
- 定义订单执行的标准接口
- 实盘和回测都实现此接口
- 交易策略类通过此接口执行订单
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

class OrderExecutor(ABC):
    """
    订单执行器基类
    
    实盘（LiveExecutor）和回测（BacktestExecutor）都继承此类
    这样可以确保交易策略在两种模式下使用相同的逻辑
    """
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Dict:
        """获取账户余额"""
        pass
    
    @abstractmethod
    def get_all_positions(self) -> List[Dict]:
        """获取所有持仓"""
        pass
    
    @abstractmethod
    def execute_market_order(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float,
        reason: str
    ) -> Optional[Dict]:
        """
        执行市价单
        
        Returns:
            包含 order_id, executed_price, executed_quantity, status 的字典
        """
        pass
    
    @abstractmethod
    def execute_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reason: str
    ) -> Optional[Dict]:
        """
        执行限价单
        
        Returns:
            包含 order_id, status 的字典
        """
        pass
    
    @abstractmethod
    def execute_stop_order(
        self,
        symbol: str,
        side: str,
        order_type: str,  # 'STOP' or 'STOP_MARKET'
        quantity: float,
        price: Optional[float],  # 限价单需要价格
        stop_price: float,
        reason: str
    ) -> Optional[Dict]:
        """
        执行止损单
        
        Returns:
            包含 order_id, status 的字典
        """
        pass
    
    @abstractmethod
    def execute_trailing_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        activation_price: float,
        callback_rate: float,  # 百分比，例如 1.5 表示 1.5%
        reason: str
    ) -> Optional[Dict]:
        """
        执行跟踪止损单
        
        Returns:
            包含 order_id, status 的字典
        """
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> bool:
        """取消某币种的所有订单"""
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """获取未成交订单"""
        pass
    
    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict:
        """获取币种信息（含filters）"""
        pass
    
    @abstractmethod
    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """调整数量精度"""
        pass
    
    @abstractmethod
    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """调整价格精度"""
        pass


class OrderResult:
    """订单执行结果"""
    
    def __init__(
        self,
        success: bool,
        order_id: Optional[str] = None,
        executed_price: Optional[float] = None,
        executed_quantity: Optional[float] = None,
        status: str = 'FAILED',
        error_message: Optional[str] = None
    ):
        self.success = success
        self.order_id = order_id
        self.executed_price = executed_price
        self.executed_quantity = executed_quantity
        self.status = status  # 'PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'FAILED'
        self.error_message = error_message
