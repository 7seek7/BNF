"""
止损单管理器 (Stop Loss Order Manager)

功能：
- 创建交易所级别的止损单（网络断开也能触发）
- 止损单跟踪和管理
- 自动取消或更新止损单
"""

import time
import threading
from typing import Dict, Optional, List, Any
from utils.logger import Logger

logger = Logger.get_logger('stop_loss_manager')


class StopLossOrderManager:
    """
    止损单管理器
    
    使用交易所止损单（STOP_MARKET）而不是代码监控：
    - 网络断开时也能触发止损
    - 降低延迟
    - 可靠性更高
    """
    
    def __init__(self, client):
        """
        初始化止损单管理器
        
        Args:
            client: 币安客户端实例
        """
        self.client = client
        
        # {position_key: stop_loss_order_id}
        # position_key 形如: {symbol}_{position_side}
        self.stop_loss_orders: Dict[str, str] = {}
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("止损单管理器已初始化")
    
    def _get_position_key(self, symbol: str, side: str) -> str:
        """生成仓位唯一键"""
        return f"{symbol}_{side}"
    
    def create_stop_loss(self, symbol: str, side: str, quantity: float, 
                        stop_price: float, client_order_id: Optional[str] = None) -> Optional[str]:
        """
        创建止损单
        
        Args:
            symbol: 币种
            side: 方向(BUY买入时止损是SELL SELL卖出时止损是BUY) -> 止损订单方向
            quantity: 数量
            stop_price: 触发价格
            client_order_id: 客户端订单ID（可选）
        
        Returns:
            订单ID（创建失败返回None）
        """
        try:
            position_key = self._get_position_key(symbol, side)
            
            # 如果已有止损单，先取消
            if position_key in self.stop_loss_orders:
                existing_order_id = self.stop_loss_orders[position_key]
                logger.info(f"{symbol} 取消旧止损单: {existing_order_id}")
                self.cancel_stop_loss(symbol, existing_order_id)
            
            # 生成客户端订单ID
            if client_order_id is None:
                import time
                client_order_id = f"{symbol[:6]}_SL_{int(time.time())}"
            
            # 创建止损市价单（STOP_MARKET）
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                clientOrderId=client_order_id,
                timeInForce='GTC'  # Good Till Canceled
            )
            
            order_id = order.get('orderId')
            if order_id:
                # 保存止损单映射
                with self.lock:
                    self.stop_loss_orders[position_key] = order_id
                
                logger.info(
                    f"{symbol} 止损单已创建: "
                    f"OrderID={order_id}, StopPrice={stop_price}, Qty={quantity}"
                )
                return order_id
            else:
                logger.error(f"{symbol} 创建止损单失败: 未返回订单ID")
                return None
                
        except Exception as e:
            logger.error(f"{symbol} 创建止损单失败: {e}")
            return None
    
    def cancel_stop_loss(self, symbol: str, stop_loss_order_id: str) -> bool:
        """
        取消止损单
        
        Args:
            symbol: 币种
            stop_loss_order_id: 止损单ID
        
        Returns:
            是否成功取消
        """
        try:
            # 取消订单
            result = self.client.cancel_order(symbol, stop_loss_order_id)
            
            # 清除映射
            position_key = None
            with self.lock:
                for key, order_id in self.stop_loss_orders.items():
                    if order_id == stop_loss_order_id:
                        position_key = key
                        del self.stop_loss_orders[key]
                        break
            
            if result:
                logger.info(f"{symbol} 止损单已取消: {stop_loss_order_id}")
                return True
            else:
                logger.warning(f"{symbol} 止损单取消失败: {stop_loss_order_id}")
                return False
            
        except Exception as e:
            logger.error(f"{symbol} 取消止损单失败 {stop_loss_order_id}: {e}")
            return False
    
    def cancel_stop_loss_by_position(self, symbol: str, side: str) -> bool:
        """
        根据仓位取消止损单
        
        Args:
            symbol: 币种
            side: 仓位方向
        
        Returns:
            是否成功取消
        """
        try:
            position_key = self._get_position_key(symbol, side)
            
            with self.lock:
                if position_key in self.stop_loss_orders:
                    order_id = self.stop_loss_orders[position_key]
                    return self.cancel_stop_loss(symbol, order_id)
            
            # 没有找到止损单，认为已取消
            return True
            
        except Exception as e:
            logger.error(f"{symbol} 根据仓位取消止损单失败: {e}")
            return False
    
    def update_stop_loss(self, symbol: str, side: str, 
                         new_stop_price: float, new_quantity: Optional[float] = None) -> Optional[str]:
        """
        更新止损单（先取消旧的，再创建新的）
        
        Args:
            symbol: 币种
            side: 方向
            new_stop_price: 新的止损价格
            new_quantity: 新的数量（可选，不传则保持不变）
        
        Returns:
            新订单ID（失败返回None）
        """
        try:
            position_key = self._get_position_key(symbol, side)
            
            with self.lock:
                if position_key not in self.stop_loss_orders:
                    logger.warning(f"{symbol} 没有找到止损单，无法更新")
                    return None
                
                old_order_id = self.stop_loss_orders[position_key]
            
            # 获取止损单信息
            old_order = self.client.client.futures_get_order(symbol=symbol, orderId=old_order_id)
            
            # 使用旧的订单信息（数量等）
            quantity = float(old_order.get('origQty')) if new_quantity is None else new_quantity
            # 止损单方向：买入仓位的止损是卖出，卖出仓位的止损是买入
            # 根据原订单保持方向
            sl_side = old_order.get('side')
            
            # 取消旧订单
            if not self.cancel_stop_loss(symbol, old_order_id):
                logger.error(f"{symbol} 取消旧止损单失败，无法更新")
                return None
            
            # 创建新止损单
            return self.create_stop_loss(symbol, sl_side, quantity, new_stop_price)
            
        except Exception as e:
            logger.error(f"{symbol} 更新止损单失败: {e}")
            return None
    
    def get_stop_loss_status(self, symbol: str, side: str) -> Optional[Dict[str, Any]]:
        """
        获取止损单状态
        
        Args:
            symbol: 币种
            side: 仓位方向
        
        Returns:
            止损单信息（不存在返回None）
        """
        try:
            position_key = self._get_position_key(symbol, side)
            
            with self.lock:
                if position_key not in self.stop_loss_orders:
                    return None
                
                order_id = self.stop_loss_orders[position_key]
            
            # 查询订单状态
            order = self.client.client.futures_get_order(symbol=symbol, orderId=order_id)
            
            return {
                'order_id': order_id,
                'status': order.get('status'),
                'stop_price': order.get('stopPrice'),
                'quantity': order.get('origQty'),
                'symbol': order.get('symbol')
            }
            
        except Exception as e:
            logger.error(f"{symbol} 查询止损单状态失败: {e}")
            return None
    
    def cancel_all_stop_losses(self) -> int:
        """
        取消所有止损单
        
        Returns:
            成功取消的数量
        """
        cancelled_count = 0
        
        try:
            with self.lock:
                position_keys = list(self.stop_loss_orders.keys())
            
            for position_key in position_keys:
                order_id = self.stop_loss_orders[position_key]
                # 从position_key解析symbol
                parts = position_key.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    if self.cancel_stop_loss(symbol, order_id):
                        cancelled_count += 1
            
            logger.info(f"已取消所有止损单: {cancelled_count}个")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"取消所有止损单失败: {e}")
            return cancelled_count
    
    def get_active_stop_loss_count(self) -> int:
        """获取活跃止损单数量"""
        with self.lock:
            return len(self.stop_loss_orders)
    
    def cleanup_expired_stop_losses(self, max_age_hours: int = 24) -> int:
        """
        清理过期的止损单信息
        
        Args:
            max_age_hours: 最大保留时间（小时）
        
        Returns:
            清理数量
        """
        # 止损单本身由交易所管理，这里只清理跟踪信息
        # 需要查询每个止损单的实际状态
        cleaned_count = 0
        
        try:
            with self.lock:
                position_keys = list(self.stop_loss_orders.items())
            
            for (position_key, order_id), (symbol, side) in zip(self.stop_loss_orders.items(), 
                                                                      [(k.split('_')[0], 
                                                                       k.split('_')[1]) 
                                                                      for k in self.stop_loss_orders.keys()]):
                try:
                    # 查询订单状态
                    order = self.client.client.futures_get_order(symbol=symbol, orderId=order_id)
                    status = order.get('status', '')
                    
                    # 如果订单已完成（成交、取消、拒绝），清理跟踪信息
                    if status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        logger.info(
                            f"{symbol} 清理已完成的止损单: "
                            f"OrderID={order_id}, Status={status}"
                        )
                        del self.stop_loss_orders[position_key]
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"查询止损单状态失败 {order_id}: {e}")
                    # 查询失败，保留记录
                    continue
            
            logger.info(f"清理了 {cleaned_count} 个已完成的止损单记录")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理过期止损单失败: {e}")
            return cleaned_count


# 全局实例（延迟初始化）
_stop_loss_manager = None

def get_stop_loss_manager(client) -> StopLossOrderManager:
    """获取止损单管理器实例（单例模式）"""
    global _stop_loss_manager
    if _stop_loss_manager is None:
        _stop_loss_manager = StopLossOrderManager(client)
    return _stop_loss_manager
