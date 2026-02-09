"""
实盘订单执行器 - 包装 BinanceClient

职责：
- 实现 OrderExecutor 接口
- 调用币安API执行订单
- 满足实盘的特殊需求（API限流、重试等）
"""

from typing import Dict, List, Optional
from trading.order_executor import OrderExecutor, OrderResult
from config.settings import settings
from utils.logger import Logger

logger = Logger.get_logger('executor')

class LiveExecutor(OrderExecutor):
    """
    实盘订单执行器
    
    包装 BinanceClient，提供统一的订单执行接口
    """
    
    def __init__(self, client):
        """
        初始化实盘执行器
        
        Args:
            client: BinanceClient 实例
        """
        self.client = client
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        return self.client.get_ticker_price(symbol)
    
    def get_account_balance(self) -> Dict:
        """获取账户余额"""
        return self.client.get_account_balance()
    
    def get_all_positions(self) -> List[Dict]:
        """获取所有持仓"""
        positions = self.client.get_positions()
        # 只返回有持仓的（abs(amount) > 0）
        return [p for p in positions if abs(p.get('position_amt', 0)) > 0]
    
    def execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reason: str
    ) -> Optional[Dict]:
        """执行市价单"""
        try:
            result = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity,
                newClientOrderId=f"{symbol}_{side}_{int(__import__('time').time() * 1000)}"
            )
            return result
        except Exception as e:
            logger.error(f"市价单失败 {symbol}: {str(e)}")
            return None
    
    def execute_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reason: str
    ) -> Optional[Dict]:
        """执行限价单"""
        try:
            result = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='LIMIT',
                quantity=quantity,
                price=price,
                timeInForce='GTC',
                newClientOrderId=f"{symbol}_{side}_limit_{int(__import__('time').time() * 1000)}"
            )
            return result
        except Exception as e:
            logger.error(f"限价单失败 {symbol}: {str(e)}")
            return None
    
    def execute_stop_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        stop_price: float,
        reason: str
    ) -> Optional[Dict]:
        """
        执行止损单
        
        Args:
            symbol: 币种
            side: 方向 BUY/SELL
            order_type: 订单类型 STOP 或 STOP_MARKET
            quantity: 数量
            price: 限价（仅STOP类型需要）
            stop_price: 触发价格
            reason: 原因
        """
        try:
            # 根据文档，使用 stop_price 参数而不是 stopPrice kwargs
            result = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price if order_type == 'STOP' else None,
                stop_price=stop_price,  # 使用命名参数
                timeInForce='GTC',
                reduceOnly='true' if reason != 'ADD' else 'false',
                newClientOrderId=f"{symbol}_{order_type}_{int(__import__('time').time() * 1000)}"
            )
            return result
        except Exception as e:
            logger.error(f"止损单失败 {symbol}: {str(e)}")
            return None
    
    def execute_trailing_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        activation_price: float,
        callback_rate: float,
        reason: str
    ) -> Optional[Dict]:
        """
        执行跟踪止损单
        
        Args:
            symbol: 币种
            side: 方向 BUY/SELL
            quantity: 数量
            activation_price: 激活价格
            callback_rate: 回调百分比（例如1.5表示1.5%）
            reason: 原因
        """
        try:
            result = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='TRAILING_STOP_MARKET',
                quantity=quantity,
                activationPrice=activation_price,
                callbackRate=callback_rate,
                reduceOnly='true',
                workingType='MARK_PRICE',
                priceProtect='true',
                newClientOrderId=f"{symbol}_trailing_tp_{int(__import__('time').time() * 1000)}"
            )
            return result
        except Exception as e:
            logger.error(f"跟踪止损单失败 {symbol}: {str(e)}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        try:
            self.client.cancel_order(symbol, order_id)
            return True
        except Exception as e:
            logger.warning(f"取消订单失败 {symbol} {order_id}: {str(e)}")
            return False
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """取消某币种的所有订单"""
        try:
            self.client.cancel_all_orders(symbol)
            return True
        except Exception as e:
            logger.warning(f"取消所有订单失败 {symbol}: {str(e)}")
            return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """获取未成交订单"""
        return self.client.get_open_orders(symbol)
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """获取币种信息（含filters）"""
        return self.client.get_symbol_info(symbol)
    
    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """调整数量精度"""
        return self.client.adjust_quantity_precision(symbol, quantity)
    
    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """调整价格精度"""
        return self.client.adjust_price_precision(symbol, price)
    
    def send_trade_message(self, message: str):
        """发送交易消息（通过Telegram）"""
        # 这个方法需要访问 telegram_bot，暂时跳过
        # 后续可以通过依赖注入的方式添加
        pass
