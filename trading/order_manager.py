from utils.logger import Logger
from utils.helpers import retry_on_failure

logger = Logger.get_logger('order_manager')

class OrderManager:
    """订单管理器"""
    
    def __init__(self, client):
        self.client = client
    
    @retry_on_failure()
    def create_market_order(self, symbol, side, quantity):
        """创建市价单"""
        try:
            order = self.client.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"市价单成交: {symbol} {side} {quantity} @ {order.get('avgPrice', 'N/A')}")
            return order
            
        except Exception as e:
            logger.error(f"创建市价单失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def create_limit_order(self, symbol, side, quantity, price):
        """创建限价单"""
        try:
            order = self.client.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                quantity=quantity,
                price=price,
                timeInForce='GTC'
            )
            
            logger.info(f"限价单已下: {symbol} {side} {quantity} @ {price}")
            return order
            
        except Exception as e:
            logger.error(f"创建限价单失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def create_stop_market_order(self, symbol, side, quantity, stop_price):
        """创建止损市价单"""
        try:
            order = self.client.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price
            )
            
            logger.info(f"止损单已下: {symbol} {side} {quantity} @ {stop_price}")
            return order
            
        except Exception as e:
            logger.error(f"创建止损单失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def cancel_order(self, symbol, order_id):
        """取消订单"""
        try:
            result = self.client.client.futures_cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            
            logger.info(f"订单已取消: {symbol} {order_id}")
            return result
            
        except Exception as e:
            logger.error(f"取消订单失败 {symbol} {order_id}: {str(e)}")
            raise
    
    @retry_on_failure()
    def cancel_all_orders(self, symbol):
        """取消所有订单"""
        try:
            result = self.client.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"已取消所有订单: {symbol}")
            return result
        except Exception as e:
            logger.error(f"取消所有订单失败 {symbol}: {str(e)}")
            raise
    
    def get_open_orders(self, symbol):
        """获取未成交订单"""
        try:
            orders = self.client.client.futures_get_open_orders(symbol=symbol)
            return orders
        except Exception as e:
            logger.error(f"获取未成交订单失败 {symbol}: {str(e)}")
            return []