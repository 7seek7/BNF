"""
交易策略类 - 平台无关的核心交易逻辑

职责：
- 实现所有交易策略逻辑（加仓、止盈、止损）
- 通过执行器执行订单，不关心是实盘还是回测
- 确保实盘和回测使用相同的交易策略
"""

from typing import Dict, List, Optional, Any, Tuple
import time
from config.settings import settings
from utils.logger import Logger
from trading.order_executor import OrderExecutor

logger = Logger.get_logger('strategy')

class TradeStrategy:
    """
    交易策略类
    
    包含所有平台无关的交易逻辑：
    - 资金管理
    - 持仓管理
    - 加仓逻辑
    - 止盈逻辑
    - 止损逻辑
    - 建仓完成判断
    
    通过 OrderExecutor 执行订单，实盘和回测可以共用此策略
    """
    
    def __init__(self, executor: OrderExecutor):
        """
        初始化交易策略
        
        Args:
            executor: 订单执行器（实盘或回测）
        """
        self.executor = executor
        
        # 持仓数据：{symbol: position_info}
        self.positions: Dict[str, Dict] = {}
        
        # 账户资金缓存
        self._balance_cache = None
        self._balance_cache_time = 0
    
    def handle_alert(self, alert_data: Dict[str, Any], timestamp: Optional[int] = None) -> bool:
        """
        处理警报并开仓
        
        Args:
            alert_data: 警报数据
            timestamp: 时间戳（回测时需要）
        
        Returns:
            是否成功开仓
        """
        try:
            symbol = alert_data['symbol']
            direction = alert_data['direction']
            current_price = alert_data['current_price']
            
            # 检查是否达到最大持仓数
            active_positions = sum(1 for p in self.positions.values() 
                                 if p.get('status') == 'active')
            
            if active_positions >= settings.MAX_POSITIONS:
                logger.debug(f"已达到最大持仓数 {settings.MAX_POSITIONS}，跳过 {symbol}")
                return False
            
            # 检查是否已有该币种的仓位
            if symbol in self.positions:
                logger.debug(f"{symbol} 已有仓位，跳过")
                return False
            
            # 执行开仓
            return self.open_position(alert_data, timestamp)
            
        except Exception as e:
            logger.error(f"处理警报失败: {str(e)}")
            return False
    
    def open_position(self, alert_data: Dict[str, Any], timestamp: Optional[int] = None) -> bool:
        """
        开仓
        
        Args:
            alert_data: 警报数据
            timestamp: 时间戳（回测时需要）
        
        Returns:
            是否成功开仓
        """
        try:
            symbol = alert_data['symbol']
            direction = alert_data['direction']
            current_price = alert_data['current_price']
            leverage = settings.LEVERAGE
            
            logger.info(f"{symbol} 准备开仓: 方向={direction}, 价格={current_price:.6f}")
            
            # 计算保证金限制
            max_margin_for_symbol = self._get_max_margin_for_symbol(symbol, leverage)
            
            # 计算可用资金
            available_funds = self.get_available_funds()
            
            # 计算分配金额
            position_size, error_msg = self._calculate_position_size(available_funds)
            if position_size <= 0:
                logger.error(f"{symbol} {error_msg}")
                return False
            
            # 检查单币种最大投资金额
            if position_size > settings.SINGLE_SYMBOL_MAX_INVESTMENT:
                position_size = settings.SINGLE_SYMBOL_MAX_INVESTMENT
                logger.info(f"{symbol} 开仓金额调整为最大值: {position_size:.2f} USDT")
            
            # 验证并调整数量
            quantity, is_valid, validation_info = self._validate_and_adjust_quantity(
                symbol, position_size, current_price, leverage, '开仓'
            )
            
            if not is_valid:
                logger.error(f"{symbol} 数量验证失败: {validation_info}")
                return False
            
            # 检查保证金限制
            if position_size > max_margin_for_symbol:
                logger.error(f"{symbol} 保证金 {position_size:.2f} USDT 超过最大值 {max_margin_for_symbol:.2f} USDT")
                return False
            
            # 确定订单方向
            side = 'BUY' if direction == 'LONG' else 'SELL'
            
            # 执行开仓订单（市价单）
            order_result = self.executor.execute_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reason=f"OPEN {direction}"
            )
            
            if not order_result or order_result.get('status') != 'FILLED':
                logger.error(f"{symbol} 开仓订单失败")
                return False
            
            # 创建持仓信息
            position_info = {
                'symbol': symbol,
                'entry_price': current_price,
                'current_price': current_price,
                'direction': direction,
                'total_quantity': quantity,
                'total_investment': position_size,
                'initial_margin': position_size,
                'completed_investment': position_size,
                'leverage': leverage,
                'status': 'active',
                'profit': 0.0,
                'profit_pct': 0.0,
                'max_profit_pct': 0.0,
                # 加仓相关
                'added_levels': [],
                'pending_orders': [],
                'is_closing': False,
                'last_action_time': timestamp or int(time.time() * 1000),
                # 止盈止损相关
                'take_profit_levels': {},
                'stop_loss_levels': {},
                'trailing_take_profit_order_id': None,
                'trailing_take_profit_callback': None,
                # 建仓完成标记
                'position_complete': False
            }
            
            self.positions[symbol] = position_info
            
            logger.info(f"✅ {symbol} 开仓成功: 方向={direction}, "
                       f"数量={quantity:.4f}, 价格={current_price:.6f}, "
                       f"保证金={position_size:.2f} USDT, 杠杆={leverage}x")
            
            return True
            
        except Exception as e:
            logger.error(f"开仓失败: {str(e)}", exc_info=True)
            return False
    
    def update_positions(
        self,
        current_prices: Dict[str, float],
        timestamp: Optional[int] = None
    ) -> List[Tuple[str, int, float]]:
        """
        更新持仓状态（与实盘保持一致的执行顺序）

        执行顺序（与 trading/trader.py _monitor_positions 保持一致）：
        1. 计算盈亏率
        2. 更新最高盈利率
        3. 执行加仓（如果持仓未平仓）
        4. 检查建仓完成条件
        5. 执行止盈逻辑
        6. 执行止损逻辑

        Args:
            current_prices: 当前价格字典 {symbol: price}
            timestamp: 时间戳

        Returns:
            需要处理的持仓列表 [(symbol, action, pct), ...]
            action: 0=None, 1=平仓
        """
        actions_needed = []

        for symbol, position_info in list(self.positions.items()):
            # 跳过非活跃持仓
            if position_info.get('status') != 'active':
                continue

            current_price = current_prices.get(symbol, 0)
            if current_price <= 0:
                continue

            # 更新当前价格
            position_info['current_price'] = current_price

            # 1. 计算盈亏率
            self._calculate_profit(position_info)

            pnl_rate = position_info['profit_pct']

            # 2. 更新最高盈利率
            if pnl_rate > position_info.get('max_profit_pct', -float('inf')):
                position_info['max_profit_pct'] = pnl_rate

            # 3. 执行加仓逻辑
            # 注意：回测中每根K线只执行一次，不需要检查is_closing实盘标记防止重复执行
            self.execute_position_building(symbol, position_info, current_price)

            # 4. 检查建仓完成条件
            if self._check_position_complete(symbol, position_info, pnl_rate):
                self._finalize_position_building(symbol, position_info, "达到建仓完成条件")

            # 5. 执行止盈逻辑
            self._execute_take_profit(symbol, position_info, pnl_rate, current_price)

            # 6. 执行止损逻辑
            self._execute_stop_loss(symbol, position_info, pnl_rate)

            # 7. 检查是否已平仓
            if position_info.get('total_quantity', 0) <= 0:
                actions_needed.append((symbol, 1, 100))

        return actions_needed
    
    def execute_position_building(self, symbol: str, position_info: Dict, current_price: float):
        """
        执行加仓逻辑（每个级别只执行一次）
        
        Args:
            symbol: 币种
            position_info: 持仓信息
            current_price: 当前价格
        """
        try:
            pending_orders = position_info.get('pending_orders', [])
            
            for pending_order in pending_orders:
                if pending_order['status'] != 'pending':
                    continue
                
                trigger_rate = pending_order['trigger_rate']
                pnl_rate = position_info['profit_pct']
                
                # 检查触发条件
                should_trigger = False
                if trigger_rate < 0 and pnl_rate <= trigger_rate:  # 亏损加仓
                    should_trigger = True
                elif trigger_rate > 0 and pnl_rate >= trigger_rate:  # 盈利加仓
                    should_trigger = True
                
                if should_trigger:
                    # 检查是否已尝试过（防止重复）
                    if pending_order.get('attempted', False):
                        continue
                    
                    # 标记为已尝试
                    pending_order['attempted'] = True
                    
                    logger.info(f"{symbol} 触发加仓: 盈亏率{pnl_rate:.2f}% >= 触发率{trigger_rate}%")
                    
                    # 执行加仓
                    success = self._add_position(position_info, pending_order, current_price)
                    
                    if success:
                        pending_order['status'] = 'executed'
                        position_info['added_levels'].append(pending_order['index'])
                        logger.info(f"{symbol} 加仓级别 {pending_order['index']} 成功")
                    
        except Exception as e:
            logger.error(f"{symbol} 执行加仓失败: {str(e)}")
    
    def check_and_cancel_all_orders(self):
        """检查并取消所有挂单（实盘启动时调用）"""
        try:
            all_orders = self.executor.get_open_orders()
            
            if not all_orders:
                logger.info("启动时检查完毕：没有检测到挂单")
                return
            
            # 统计挂单
            orders_by_symbol = {}
            for order in all_orders:
                symbol = order.get('symbol', '')
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)
            
            total_count = len(all_orders)
            logger.warning(f"启动时检测到 {total_count} 个挂单！")
            
            # 取消所有挂单
            for symbol in orders_by_symbol:
                self.executor.cancel_all_orders(symbol)
                logger.info(f"✓ 已取消 {symbol} 的所有挂单")
            
            logger.info(f"✓ 挂单清理完成: 取消了 {len(orders_by_symbol)} 个币种的挂单")
            
        except Exception as e:
            logger.error(f"启动时清理挂单失败: {str(e)}")
    
    def close_all_positions(self, reason: str):
        """平仓所有持仓"""
        try:
            for symbol, position_info in list(self.positions.items()):
                if position_info.get('status') == 'active':
                    # 取消跟踪止损
                    self._cancel_trailing_take_profit(symbol, position_info)
                    
                    # 取消所有挂单
                    self.executor.cancel_all_orders(symbol)
                    
                    # 平仓
                    self._execute_close_position(
                        symbol, position_info, 100, 
                        reason or "全部平仓"
                    )
        except Exception as e:
            logger.error(f"平仓所有持仓失败: {str(e)}")
    
    def get_available_funds(self) -> float:
        """
        计算可用资金
        
        公式：可用资金 = 总余额 - 已用保证金 - 未平仓亏损（绝对值）
        """
        if self._balance_cache and (time.time() - self._balance_cache_time < 5):
            return self._balance_cache
        
        try:
            balance_info = self.executor.get_account_balance()
            total_balance = balance_info.get('total_balance', 0)
            
            # 计算已用保证金和未平仓亏损
            used_margin = 0
            unrealized_loss = 0
            
            for position_info in self.positions.values():
                if position_info.get('status') == 'active':
                    used_margin += position_info.get('total_investment', 0)
                    profit = position_info.get('profit', 0)
                    if profit < 0:
                        unrealized_loss += abs(profit)
            
            available_funds = total_balance - used_margin - unrealized_loss
            
            self._balance_cache = max(available_funds, 0)
            self._balance_cache_time = time.time()
            
            return self._balance_cache
            
        except Exception as e:
            logger.error(f"计算可用资金失败: {str(e)}")
            return 0.0
    
    def _calculate_position_size(self, available_funds: float) -> Tuple[float, str]:
        """
        计算开仓金额
        
        Args:
            available_funds: 可用资金
        
        Returns:
            (开仓金额, 错误信息)
        """
        # 计算剩余槽位
        active_positions = sum(1 for p in self.positions.values() 
                             if p.get('status') == 'active')
        remaining_slots = max(1, settings.MAX_POSITIONS - active_positions)
        
        # 基础分配金额
        base_allocation = available_funds / remaining_slots
        
        # 应用 INITIAL_INITIAL_POSITION 比例
        position_size = base_allocation * (settings.INITIAL_POSITION / 100)
        
        # 单币种最大限制
        if position_size > settings.SINGLE_SYMBOL_MAX_INVESTMENT:
            position_size = settings.SINGLE_SYMBOL_MAX_INVESTMENT
        
        if position_size <= 0:
            return 0.0, f"可用资金不足: {available_funds:.2f} USDT"
        
        return position_size, ""
    
    def _validate_and_adjust_quantity(
        self,
        symbol: str,
        position_size: float,
        price: float,
        leverage: int,
        description: str
    ) -> Tuple[float, bool, Dict]:
        """
        验证并调整数量（使用统一验证逻辑）
        """
        try:
            # 计算名义价值
            notional = position_size * leverage  # 实际交易金额
            
            # 计算数量
            quantity = notional / price if price > 0 else 0
            
            # 获取币种信息
            symbol_info = self.executor.get_symbol_info(symbol)
            
            # 解析filters
            min_qty = 0.001
            max_qty = float('inf')
            step_size = 0.001
            min_notional = 0.0
            
            for f in symbol_info.get('filters', []):
                filter_type = f.get('filterType')
                if filter_type == 'LOT_SIZE':
                    min_qty = float(f.get('minQty', 0.001))
                    max_qty = float(f.get('maxQty', float('inf')))
                    step_size = float(f.get('stepSize', 0.001))
                elif filter_type == 'NOTIONAL':
                    min_notional = float(f.get('minNotional', 0.0))
            
            # 应用精度调整
            quantity = self.executor.adjust_quantity_precision(symbol, quantity)
            
            # LOT_SIZE最小值检查
            adjustment_info = {'adjustments': [], 'adjusted': False}
            
            if quantity < min_qty:
                adjustment_info['adjustments'].append(f'LOT_SIZE最小值调整')
                adjustment_info['adjusted'] = True
                quantity = min_qty
            
            # 精度调整后重新应用
            quantity = self.executor.adjust_quantity_precision(symbol, quantity)
            
            # LOT_SIZE最大值检查
            if quantity > max_qty:
                return 0, False, {'error': f'数量超过最大值 {max_qty}'}
            
            # NOTIONAL检查（如果有min_notional）
            if min_notional > 0:
                actual_notional = quantity * price
                if actual_notional < min_notional:
                    # 增加数量以满足minNotional
                    min_quantity_for_notional = (min_notional * 1.01) / price
                    min_quantity_for_notional = self.executor.adjust_quantity_precision(symbol, min_quantity_for_notional)
                    
                    if quantity < min_quantity_for_notional:
                        adjustment_info['adjustments'].append(f'minNotional调整')
                        adjustment_info['adjusted'] = True
                        quantity = min_quantity_for_notional
            
            return quantity, True, adjustment_info
            
        except Exception as e:
            logger.error(f"{symbol} {description}: 数量验证失败 - {str(e)}")
            return 0, False, {'error': str(e)}
    
    def _add_position(self, position_info: Dict, pending_order: Dict, current_price: float) -> bool:
        """
        执行加仓（回测简化版本）

        ⚠️ 与实盘的差异（trading/trader.py）：
        1. 订单类型：
           - 实盘：使用 LIMIT 限价单，等待成交
           - 回测：使用 MARKET 市价单，立即成交

        2. 价格计算：
           - 实盘：entry_price +/- DELAY_RATIO（考虑决策到成交的时间差）
           - 回测：current_price（K线收盘价就是"成交价格"）

        3. DELAY_RATIO：
           - 实盘：考虑约0.5-2秒的决策延迟，补偿价格滑动
           - 回测：不需要（K线决策时即为"成交时刻"）

        4. 重试逻辑：
           - 实盘：-2019错误重试2次，每次减少30%数量
           - 回测：不需要重试（验证了余额就一定能成交）

        5. 余额检查：
           - 实盘：检查可用余额 ≥ 实际保证金 + 10%安全余量
           - 回测：简化检查可用余额 ≥ add_margin

        理由：
        - 回测无法模拟限价单等待成交的过程（基于历史K线，无法"等待"未来）
        - 回测的 current_price 就是"立即成交的价格"
        - 回测主要验证策略方向，而非精确模拟执行细节
        """
        try:
            symbol = position_info['symbol']
            add_margin = pending_order['add_margin']
            leverage = position_info['leverage']

            # 检查可用资金
            available_funds = self.get_available_funds()

            if add_margin > available_funds:
                logger.warning(f"{symbol} 保证金不足: 需要 {add_margin:.2f}, 可用 {available_funds:.2f}")
                return False

            # 计算数量
            add_notional = add_margin * leverage
            add_quantity = add_notional / current_price

            # 验证数量
            add_quantity, is_valid, _ = self._validate_and_adjust_quantity(
                symbol, add_margin, current_price, leverage, '加仓'
            )

            if not is_valid or add_quantity <= 0:
                return False

            # 确定订单方向
            side = 'BUY' if position_info['direction'] == 'LONG' else 'SELL'

            # 执行加仓订单（市价单，简化处理）
            order_result = self.executor.execute_market_order(
                symbol=symbol,
                side=side,
                quantity=add_quantity,
                reason=f"ADD {symbol}"
            )
            
            if not order_result or order_result.get('status') != 'FILLED':
                logger.error(f"{symbol} 加仓订单失败")
                return False
            
            # 更新持仓信息
            old_quantity = position_info['total_quantity']
            old_investment = position_info['total_investment']
            old_avg_price = position_info['entry_price']  # 使用entry_price作为均价
            
            new_quantity = old_quantity + add_quantity
            new_investment = old_investment + add_margin
            new_avg_price = new_investment / (new_quantity / leverage)
            
            position_info['total_quantity'] = new_quantity
            position_info['total_investment'] = new_investment
            position_info['entry_price'] = new_avg_price  # 更新均价
            
            logger.info(f"✅ {symbol} 加仓成功: 保证金={add_margin:.2f}, 数量={add_quantity:.4f}, "
                       f"新均价={new_avg_price:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"{symbol} 加仓失败: {str(e)}")
            return False
    
    def _calculate_profit(self, position_info: Dict):
        """计算持仓盈亏"""
        try:
            symbol = position_info['symbol']
            direction = position_info['direction']
            entry_price = position_info['entry_price']
            current_price = position_info['current_price']
            total_quantity = position_info['total_quantity']
            total_investment = position_info['total_investment']
            leverage = position_info['leverage']
            
            if total_quantity <= 0 or entry_price <= 0 or current_price <= 0:
                logger.warning(f"{symbol} 参数无效: quantity={total_quantity}, entry={entry_price}, current={current_price}")
                return
            
            # 计算盈亏率（含杠杆）
            if direction == 'LONG':
                profit_pct = ((current_price - entry_price) / entry_price) * leverage * 100
                profit = total_investment * (profit_pct / 100)
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * leverage * 100
                profit = total_investment * (profit_pct / 100)
            
            position_info['profit'] = profit
            position_info['profit_pct'] = profit_pct
            
            # 更新最高盈利率
            if profit_pct > position_info.get('max_profit_pct', -999):
                position_info['max_profit_pct'] = profit_pct
            
        except Exception as e:
            logger.error(f"{symbol} 计算盈亏失败: {str(e)}")
    
    def _execute_take_profit(self, symbol: str, position_info: Dict, pnl_rate: float, current_price: float):
        """执行止盈逻辑"""
        try:
            # 只在建仓完成后触发止盈
            if not position_info.get('position_complete', False):
                return
            
            max_profit_pct = position_info['max_profit_pct']
            profit_drawback = max_profit_pct - pnl_rate
            
            close_percentage = 0
            reason = ""
            
            # 高盈利止盈
            if pnl_rate >= settings.HIGH_PROFIT_THRESHOLD:
                if profit_drawback >= settings.HIGH_PROFIT_DRAWBACK2:
                    close_percentage = settings.HIGH_PROFIT_CLOSE2
                    reason = f"高盈利回撤{settings.HIGH_PROFIT_DRAWBACK2}%"
                elif profit_drawback >= settings.HIGH_PROFIT_DRAWBACK1:
                    close_percentage = settings.HIGH_PROFIT_CLOSE1
                    reason = f"高盈利回撤{settings.HIGH_PROFIT_DRAWBACK1}%"
            
            # 低盈利止盈
            elif settings.LOW_PROFIT_THRESHOLD <= pnl_rate < settings.HIGH_PROFIT_THRESHOLD:
                if profit_drawback >= settings.LOW_PROFIT_DRAWBACK1:
                    close_percentage = settings.LOW_PROFIT_CLOSE1
                    reason = f"低盈利回撤{settings.LOW_PROFIT_DRAWBACK1}%"
            
            # 保本止盈（已到达高利润后，利润跌回保本范围内）
            if max_profit_pct > 0 and 0 <= pnl_rate <= settings.BREAKEVEN_THRESHOLD:
                # 移除 'last_take_profit_pct in position_info' 条件
                # 只要曾经盈利过（max_profit_pct > 0）且现在接近保本，就应该检查
                # 避免首次达到高利润但没有触发回撤时无法止盈的问题
                # 修复：增加条件 'max_profit_pct >= settings.LOW_PROFIT_THRESHOLD'
                # 确保至少达到了低利润阈值才开始检查保本
                if max_profit_pct >= settings.LOW_PROFIT_THRESHOLD:
                    close_percentage = 100
                    reason = f"保本止盈（最高利润{max_profit_pct:.2f}%，当前利润{pnl_rate:.2f}%低于{settings.BREAKEVEN_THRESHOLD}%）"
                    logger.info(f"{symbol} {reason}")
            
            # 执行止盈
            if close_percentage > 0:
                logger.info(f"{symbol} 触发止盈: {reason}, 平仓比例={close_percentage}%")
                
                # 取消跟踪止损
                self._cancel_trailing_take_profit(symbol, position_info)
                
                # 不存储详细信息到 position_info，只记录必要的日志
                # 避免在回测循环中不断累积数据导致内存问题
                
                self._execute_close_position(symbol, position_info, close_percentage, reason)
            
            # 跟踪止盈（基于环境变量）
            self._setup_trailing_take_profit(symbol, position_info, current_price, pnl_rate)
            
        except Exception as e:
            logger.error(f"{symbol} 执行止盈失败: {str(e)}")
    
    def _setup_trailing_take_profit(self, symbol: str, position_info: Dict, current_price: float, pnl_rate: float):
        """
        设置跟踪止盈（回测不实现此功能）

        ⚠️ 回测限制：
        1. TRAILING_STOP_MARKET 是币安API的特殊订单类型
          - 服务器端自动计算最高价和回撤
          - 回测无法精确模拟服务器端的动态计算

        2. 离散K线 vs 连续监控
          - 实盘：实时监控价格变化（每秒都在计算）
          - 回测：基于离散K线（如1分钟一根），可能错过真实的触发点

        3. 重新激活逻辑
          - 实盘：价格回调后会自动重新设置止损价
          - 回测：难以模拟这种动态调整

        处理方式：
        - 回测不实现跟踪止损（此函数为空实现或仅做简要说明）
        - 仅依赖市价止盈（回撤止盈）部分（_execute_take_profit函数）
        - 跟踪止损的细节优化在实盘中进行

        理由：
        - 回测主要验证策略止盈方向（是否在高盈利后回撤时止盈）
        - 跟踪止损的细微参数调整（回调率、激活价）在实盘优化更合适
        """
        # 回测模式不实现跟踪止损
        return
    
    def _cancel_trailing_take_profit(self, symbol: str, position_info: Dict):
        """取消跟踪止盈"""
        try:
            order_id = position_info.get('trailing_take_profit_order_id')
            if order_id:
                self.executor.cancel_order(symbol, order_id)
                position_info.pop('trailing_take_profit_order_id', None)
                logger.info(f"{symbol} 取消跟踪止盈: {order_id}")
        except Exception as e:
            logger.warning(f"{symbol} 取消跟踪止盈失败: {str(e)}")
    
    def _execute_stop_loss(self, symbol: str, position_info: Dict, pnl_rate: float):
        """
        执行止损逻辑（回测简化版本）

        ⚠️ 与实盘的差异（trading/trader.py）：
        1. 订单类型：
           - 实盘：使用 STOP 限价单（两阶段：触发 → 变为限价成交）
           - 回测：简化为市价平仓（触发条件满足 → 直接成交）

        2. 止损级别：
           - 实盘：3级（STOPLOSS_TRIGGER1/2/3）
           - 回测：2级（STOPLOSS_TRIGGER1/2）

        3. 止损价格计算：
           - 实盘：entry_price +/- DELAY_RATIO（考虑价格滑动）
           - 回测：不使用 DELAY_RATIO

        4. 建仓完成检查：
           - 实盘：✅ 必须建仓完成后才触发止损
           - 回测：✅ 必须建仓完成后才触发止损（已添加）

        5. 第三级紧急止损：
           - 实盘：有第3级紧急市价止损（STOPLOSS_TRIGGER3）
           - 回测：目前未实现

        理由：
        - STOP 限价单有两阶段触发（触及stop_price → 变为limit订单 → 触及limit_price成交）
        - 回测基于K线的high/low/close，无法精确区分触发和执行阶段
        - 回测简化为"触发条件满足 → 直接市价平仓"，更简单且满足策略验证需求
        """
        try:
            # 只在建仓完成后才触发止损（与实盘一致）
            if not position_info.get('position_complete', False):
                return

            stop_loss_levels = position_info.get('stop_loss_levels', {})

            stop_orders = [
                (settings.STOPLOSS_TRIGGER1, settings.STOPLOSS_CLOSE1, 'SL1', 100),
                (settings.STOPLOSS_TRIGGER2, settings.STOPLOSS_CLOSE2, 'SL2', 100)
            ]

            for trigger_rate, close_pct, level_name, _ in stop_orders:
                if level_name not in stop_loss_levels:
                    if pnl_rate <= trigger_rate:
                        self._execute_close_position(symbol, position_info, close_pct, f"止损{level_name}")
                        stop_loss_levels[level_name] = True
                        position_info['stop_loss_levels'] = stop_loss_levels
                        return

        except Exception as e:
            logger.error(f"{symbol} 执行止损失败: {str(e)}")
    
    def _check_position_complete(self, symbol: str, position_info: Dict, pnl_rate: float) -> bool:
        """检查是否完成建仓"""
        try:
            # 达到单币种最大投资金额
            if position_info['total_investment'] >= settings.SINGLE_SYMBOL_MAX_INVESTMENT:
                return True
            
            # 盈利上涨达到阈值
            if pnl_rate >= settings.PROFIT_STEP3:
                max_profit_pct = position_info['max_profit_pct']
                profit_rise = max_profit_pct - pnl_rate
                if profit_rise >= settings.POSITION_COMPLETE_PROFIT_RISE:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"{symbol} 检查建仓完成失败: {str(e)}")
            return False
    
    def _finalize_position_building(self, symbol: str, position_info: Dict, reason: str):
        """完成建仓"""
        try:
            # 取消所有未成交的加仓订单
            pending_orders = position_info.get('pending_orders', [])
            for pending_order in pending_orders:
                if pending_order.get('order_id') and pending_order['status'] == 'submitted':
                    self.executor.cancel_order(symbol, pending_order['order_id'])
            
            # 清空待执行订单
            position_info['pending_orders'] = []
            position_info['completed_investment'] = position_info.get('total_investment', 0)
            
            # 标记建仓完成
            position_info['position_complete'] = True
            
            logger.info(f"{symbol} 建仓完成: {reason}")
            
        except Exception as e:
            logger.error(f"{symbol} 完成建仓失败: {str(e)}")
    
    def _execute_close_position(self, symbol: str, position_info: Dict, close_pct: float, reason: str):
        """执行平仓"""
        try:
            direction = position_info['direction']
            total_quantity = position_info['total_quantity']
            total_investment = position_info['total_investment']
            current_price = position_info['current_price']
            leverage = position_info.get('leverage', settings.LEVERAGE)
            
            # 计算平仓前的总价值
            close_value_before = (total_investment + position_info.get('profit', 0)) * (close_pct / 100)
            
            if close_pct >= 100:
                # 取消所有挂单
                self.executor.cancel_all_orders(symbol)
            
            # 计算平仓数量
            close_quantity = total_quantity * (close_pct / 100)
            
            # 验证数量
            close_quantity, is_valid, _ = self._validate_and_adjust_quantity(
                symbol, close_quantity * current_price / settings.LEVERAGE, 
                current_price, leverage, '平仓'
            )
            
            if not is_valid or close_quantity <= 0:
                return
            
            # 确定平仓方向
            side = 'SELL' if direction == 'LONG' else 'BUY'
            close_investment = total_investment * (close_pct / 100)
            
            # 执行平仓订单（市价单）
            order_result = self.executor.execute_market_order(
                symbol=symbol,
                side=side,
                quantity=close_quantity,
                reason=f"CLOSE {reason}"
            )
            
            if order_result and order_result.get('status') == 'FILLED':
                # 回测特殊处理：更新余额
                if hasattr(self.executor, 'engine'):
                    # 计算这部分的盈亏
                    if direction == 'LONG':
                        profit_pct = ((current_price - position_info['entry_price']) / position_info['entry_price']) * leverage * 100
                    else:
                        profit_pct = ((position_info['entry_price'] - current_price) / position_info['entry_price']) * leverage * 100
                    
                    profit = close_investment * (profit_pct / 100)
                    realized_value = close_investment + profit
                    
                    # 更新持仓
                    position_info['total_quantity'] -= close_quantity
                    position_info['total_investment'] -= close_investment
                    
                    # 向账户余额释放价值（保证金 + 盈亏）
                    self.executor.engine.balance += realized_value
                    logger.info(f"  [资金] {symbol} 平仓释放: +{realized_value:.2f} USDT (包括盈亏 {profit:+.2f} USDT)")
                
                # 更新持仓状态
                if position_info['total_quantity'] <= 0:
                    position_info['status'] = 'closed'
                    position_info['close_time'] = int(time.time() * 1000 if hasattr(time, 'time') else time.time())
                else:
                    # 部分平仓：更新entry_price为加权平均
                    remaining_pct = position_info['total_quantity'] / (position_info['total_quantity'] + close_quantity)
                    old_entry = position_info['entry_price']
                    position_info['entry_price'] = old_entry * remaining_pct + current_price * (1 - remaining_pct)
                
                logger.info(f"✅ {symbol} 平仓: {reason}, 平仓{close_pct}%, 数量={close_quantity:.4f}")
            
        except Exception as e:
            logger.error(f"{symbol} 平仓失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _get_max_margin_for_symbol(self, symbol: str, leverage: int) -> float:
        """获取单币种最大保证金限制"""
        try:
            # 简化：直接使用环境变量
            return settings.MAX_MARGIN_PER_SYMBOL
        except Exception as e:
            logger.warning(f"获取最大保证金失败 {symbol}: {str(e)}")
            return settings.MAX_MARGIN_PER_SYMBOL
