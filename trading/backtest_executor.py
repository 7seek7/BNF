"""
回测订单执行器 - 模拟订单执行

职责：
- 实现 OrderExecutor 接口
- 在历史数据上模拟订单执行
- 处理回测的特殊需求（K线价格范围、立即成交等）
"""

from typing import Dict, List, Optional
from trading.order_executor import OrderExecutor
from config.settings import settings
from utils.logger import Logger

logger = Logger.get_logger('backtest_executor')

class BacktestExecutor(OrderExecutor):
    """
    回测订单执行器
    
    在历史数据上模拟订单执行，处理K线价格范围
    """
    
    def __init__(self, backtest_engine):
        """
        初始化回测执行器
        
        Args:
            backtest_engine: 回测引擎实例（包含价格数据、余额等）
        """
        self.engine = backtest_engine
        self.timestamp = None  # 当前回测时间戳
        self.kline_data = None  # 当前K线数据
        self.pending_orders = {}  # {order_id: order_info}
        self.next_order_id = 1
        self.trade_history = []  # 交易历史
    
    def set_context(self, timestamp, kline_data):
        """
        设置当前回测上下文
        
        Args:
            timestamp: 当前时间戳
            kline_data: 当前K线数据
        """
        self.timestamp = timestamp
        self.kline_data = kline_data
        
        # 处理挂单成交
        self._process_pending_orders()
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格（使用收盘价）"""
        if self.kline_data:
            return float(self.kline_data.get('close', 0))
        return 0.0
    
    def get_account_balance(self) -> Dict:
        """获取账户余额"""
        return {
            'total_balance': self.engine.balance + self.engine.get_total_position_value(),
            'available_balance': self.engine.balance
        }
    
    def get_all_positions(self) -> List[Dict]:
        """获取所有持仓"""
        positions = []
        for symbol, pos in self.engine.positions.items():
            if pos.get('status') == 'active':
                positions.append({
                    'symbol': symbol,
                    'position_amt': pos['total_quantity'] if pos['direction'] == 'LONG' else -pos['total_quantity'],
                    'entry_price': pos['average_price'],
                    'notional': pos['total_quantity'] * pos['current_price'],
                    'unrealized_pnl': pos['profit'],
                    'leverage': pos.get('leverage', settings.LEVERAGE)
                })
        return positions
    
    def execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reason: str
    ) -> Optional[Dict]:
        """
        执行市价单（立即成交）
        
        回测特性：市价单立即成交，使用收盘价
        """
        price = self.get_current_price(symbol)
        if price <= 0:
            logger.error(f"市价单失败：价格异常 {symbol}")
            return None
        
        # 检查余额并扣除（开仓时）
        if 'OPEN' in reason or 'ADD' in reason:
            margin = (price * quantity) / settings.LEVERAGE
            if margin > self.engine.balance:
                logger.warning(f"余额不足，跳过市价单 {symbol}: 需要{margin:.2f}, 余额{self.engine.balance:.2f}")
                return None
            # 扣除余额（移到持仓中）
            self.engine.balance -= margin
            logger.info(f"  [资金] 开仓扣除保证金: -{margin:.2f} USDT，余额: {self.engine.balance:.2f} USDT")
        
        # 平仓时增加余额（在 TradeStrategy 的 close_position 中处理，这里只记录）
        elif 'CLOSE' in reason:
            profit = self._calculate_close_profit(symbol, quantity, price)
            if profit is not None:
                # 这里不直接加到 balance，因为 close_position 会处理
                logger.info(f"  [资金] 平仓盈亏: {profit:+.2f} USDT")
        
        # 创建订单
        order_id = f"bk_{self.next_order_id}"
        self.next_order_id += 1
        
        order = {
            'orderId': order_id,
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'price': price,
            'status': 'FILLED',
            'executedQty': quantity,
            'avgPrice': price,
            'reason': reason,
            'timestamp': self.timestamp
        }
        
        # 记录交易
        actual_amount = quantity * price
        trade_record = {
            'timestamp': self.timestamp,
            'time': __import__('datetime').datetime.fromtimestamp(self.timestamp/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'action': reason.split()[0] if ' ' in reason else 'UNKNOWN',  # OPEN, ADD, CLOSE
            'price': price,
            'quantity': quantity,
            'amount': actual_amount / settings.LEVERAGE if 'OPEN' in reason or 'ADD' in reason else actual_amount,
            'direction': 'LONG' if side == 'BUY' else 'SHORT',
            'balance': self.engine.balance,
            'profit': 0.0,
            'profit_pct': 0.0,
            'reason': reason
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"[回测市价单] {symbol} {side} 数量:{quantity:.4f} 价格:{price:.4f} 原因:{reason}")
        
        return order
    
    def _calculate_close_profit(self, symbol: str, quantity: float, close_price: float) -> float:
        """计算平仓盈亏"""
        if symbol not in self.engine.strategy.positions:
            return None
        
        pos = self.engine.strategy.positions[symbol]
        if pos.get('status') != 'active':
            return None
        
        # 根据持仓计算盈亏
        entry_price = pos['entry_price']
        total_investment = pos['total_investment']
        leverage = pos['leverage']
        
        # 计算盈亏率
        if pos['direction'] == 'LONG':
            profit_pct = ((close_price - entry_price) / entry_price) * leverage * 100
        else:
            profit_pct = ((entry_price - close_price) / entry_price) * leverage * 100
        
        # 计算盈亏金额
        profit = total_investment * (profit_pct / 100)
        
        return profit
    
    def execute_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reason: str
    ) -> Optional[Dict]:
        """
        执行限价单（模拟成交检测）
        
        回测特性：
        - 检查当前K线的 high 和 low
        - 如果限价在K线价格范围内，立即成交
        - 否则添加到挂单列表
        """
        if not self.kline_data:
            return None
        
        low = float(self.kline_data.get('low', 0))
        high = float(self.kline_data.get('high', 0))
        
        # 检查是否触及限价
        if self._is_price_in_range(price, side, low, high):
            # 触及限价，立即成交
            order_id = f"bk_{self.next_order_id}"
            self.next_order_id += 1
            
            order = {
                'orderId': order_id,
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'quantity': quantity,
                'price': price,
                'status': 'FILLED',
                'executedQty': quantity,
                'avgPrice': price,
                'reason': reason,
                'timestamp': self.timestamp
            }
            
            # 记录交易
            actual_amount = quantity * price
            trade_record = {
                'timestamp': self.timestamp,
                'time': __import__('datetime').datetime.fromtimestamp(self.timestamp/1000).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'action': reason.split()[0] if ' ' in reason else 'UNKNOWN',
                'price': price,
                'quantity': quantity,
                'amount': actual_amount / settings.LEVERAGE if 'OPEN' in reason or 'ADD' in reason else actual_amount,
                'direction': 'LONG' if side == 'BUY' else 'SHORT',
                'balance': self.engine.balance,
                'profit': 0.0,
                'profit_pct': 0.0,
                'reason': reason
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"[回测限价单成交] {symbol} {side} 数量:{quantity:.4f} 限价:{price:.4f} 原因:{reason}")
            
            return order
        else:
            # 未触及，添加到挂单
            order_id = f"bk_{self.next_order_id}"
            self.next_order_id += 1
            
            order = {
                'orderId': order_id,
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'quantity': quantity,
                'price': price,
                'status': 'NEW',
                'reason': reason,
                'timestamp': self.timestamp
            }
            
            self.pending_orders[order_id] = order
            logger.info(f"[回测限价单挂起] {symbol} {side} 数量:{quantity:.4f} 限价:{price:.4f} 原因:{reason}")
            
            return order
    
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
        执行止损单（模拟）
        
        回测特性：
        - STOP 订单：价格必须在 stop_price 和限价之间
        - STOP_MARKET：价格触及 stop_price 立即市价成交
        """
        if not self.kline_data:
            return None
        
        low = float(self.kline_data.get('low', 0))
        high = float(self.kline_data.get('high', 0))
        
        # 检查是否触及止损价
        stop_triggered = False
        
        if order_type == 'STOP_MARKET':
            # 市价止损：只要触及 stop_price 就触发
            if side == 'SELL':  # 做多止损
                stop_triggered = low <= stop_price <= high
            else:  # BUY 做空止损
                stop_triggered = low <= stop_price <= high
        else:  # STOP 限价止损
            # 检查是否在触发价格范围内，且可以按限价成交
            if side == 'SELL':  # 做多止损
                stop_triggered = low <= stop_price <= high and price >= low
            else:  # BUY 做空止损
                stop_triggered = low <= stop_price <= high and price <= high
        
        if stop_triggered:
            # 触发止损，立即成交
            order_id = f"bk_{self.next_order_id}"
            self.next_order_id += 1
            
            # 使用收盘价成交
            price = self.get_current_price(symbol) if order_type == 'STOP_MARKET' else price
            
            order = {
                'orderId': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'stopPrice': stop_price,
                'price': price,
                'status': 'FILLED',
                'executedQty': quantity,
                'avgPrice': price,
                'reason': reason,
                'timestamp': self.timestamp
            }
            
            # 记录交易
            actual_amount = quantity * price
            trade_record = {
                'timestamp': self.timestamp,
                'time': __import__('datetime').datetime.fromtimestamp(self.timestamp/1000).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'action': 'CLOSE',
                'price': price,
                'quantity': quantity,
                'amount': actual_amount,
                'direction': 'LONG' if side == 'SELL' else 'SHORT',
                'balance': self.engine.balance,
                'profit': 0.0,
                'profit_pct': 0.0,
                'reason': reason
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"[回测止损单成交] {symbol} {side} 数量:{quantity:.4f} 止损价:{stop_price:.4f} 原因:{reason}")
            
            return order
        
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
        执行跟踪止损单（简化实现）
        
        回测特性：
        - 跟踪止损简化为基于回撤的止盈逻辑
        - 在交易策略中检查回撤触发，这里只创建订单记录
        """
        # 回测中跟踪止损由策略逻辑直接处理，不需要挂单
        # 返回一个虚拟订单ID
        order_id = f"ts_{self.next_order_id}"
        self.next_order_id += 1
        
        order = {
            'orderId': order_id,
            'symbol': symbol,
            'side': side,
            'type': 'TRAILING_STOP_MARKET',
            'quantity': quantity,
            'activation_price': activation_price,
            'callback_rate': callback_rate,
            'status': 'ACTIVE',  # 跟踪止损始终活跃
            'timestamp': self.timestamp
        }
        
        logger.info(f"[回测跟踪止损] {symbol} 激活价:{activation_price:.4f} 回调率:{callback_rate}%")
        
        return order
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            logger.info(f"[回测取消订单] {symbol} {order_id}")
            return True
        return False
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """取消某币种的所有订单"""
        cancelled = False
        for order_id, order in list(self.pending_orders.items()):
            if order['symbol'] == symbol:
                del self.pending_orders[order_id]
                cancelled = True
        if cancelled:
            logger.info(f"[回测取消所有订单] {symbol}")
        return cancelled
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """获取未成交订单"""
        orders = list(self.pending_orders.values())
        if symbol:
            orders = [o for o in orders if o['symbol'] == symbol]
        return orders
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        获取币种信息（模拟）
        
        回测特性：返回空 filters，表示无需验证 minNotional
        """
        return {
            'symbol': symbol,
            'filters': []  # 空filters表示忽略限制
        }
    
    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """调整数量精度（回测中不做处理）"""
        return quantity
    
    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """调整价格精度（回测中不做处理）"""
        return price
    
    def _is_price_in_range(self, price: float, side: str, low: float, high: float) -> bool:
        """
        检查价格是否在K线范围内可以成交
        
        Args:
            price: 限价
            side: 方向 BUY/SELL
            low: K线最低价
            high: K线最高价
        
        Returns:
            True 可以成交，False 不能成交
        
        规则：
        - BUY 订单：限价在 [low, high] 范围内时成交
        - SELL 订单：限价在 [low, high] 范围内时成交
        """
        return low <= price <= high
    
    def _process_pending_orders(self):
        """
        处理挂单成交
        
        在每根K线处理时调用，检查挂单是否触及价格范围
        """
        if not self.kline_data:
            return
        
        low = float(self.kline_data.get('low', 0))
        high = float(self.kline_data.get('high', 0))
        
        filled_orders = []
        
        for order_id, order in self.pending_orders.items():
            symbol = order['symbol']
            side = order['side']
            price = order['price']
            
            # 检查是否触及限价
            if self._is_price_in_range(price, side, low, high):
                # 触及限价，成交
                order['status'] = 'FILLED'
                order['executedQty'] = order['quantity']
                order['avgPrice'] = price
                order['timestamp'] = self.timestamp
                
                # 记录交易
                actual_amount = order['quantity'] * price
                trade_record = {
                    'timestamp': self.timestamp,
                    'time': __import__('datetime').datetime.fromtimestamp(self.timestamp/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'action': order.get('reason', 'UNKNOWN').split()[0] if ' ' in order.get('reason', '') else 'UNKNOWN',
                    'price': price,
                    'quantity': order['quantity'],
                    'amount': actual_amount / settings.LEVERAGE if 'OPEN' in order.get('reason', '') or 'ADD' in order.get('reason', '') else actual_amount,
                    'direction': 'LONG' if side == 'BUY' else 'SHORT',
                    'balance': self.engine.balance,
                    'profit': 0.0,
                    'profit_pct': 0.0,
                    'reason': order.get('reason', '限价单成交')
                }
                self.trade_history.append(trade_record)
                
                logger.info(f"[回测挂单成交] {symbol} 订单{order_id} 价格:{price:.4f}")
                filled_orders.append(order_id)
        
        # 移除已成交订单
        for order_id in filled_orders:
            del self.pending_orders[order_id]
