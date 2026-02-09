"""
交易核心模块 - 完整重写版

职责：
- 计算可用资金
- 处理警报并开仓
- 管理加仓逻辑
- 管理止盈止损
- 发送交易消息
"""

from typing import Dict, List, Optional, Any, Tuple
import time
import math
import threading
import json
from datetime import datetime
from collections import defaultdict
from binance.exceptions import BinanceAPIException
from config.settings import settings
from utils.logger import Logger
from utils.helpers import adjust_quantity_precision, adjust_price_precision
from utils.rate_limiter import get_adaptive_rate_limiter
from utils.validators import get_data_quality_checker
from alert.binance_client import BinanceClient
from alert.telegram_bot import TelegramBot
from trading.position_manager import PositionManager
from trading.trade_recorder import TradeRecorder
from trading.order_persistence import get_order_persistence, OrderStatus
from trading.duplicate_detector import get_duplicate_detector, get_conflict_detector
from trading.stop_loss_manager import get_stop_loss_manager

logger = Logger.get_logger('trader')


class TradeModule:
    """交易核心模块"""

    def __init__(self, client, telegram_bot=None):
        """
        初始化交易模块

        Args:
            client: 币安客户端实例
            telegram_bot: Telegram机器人实例（可选）
        """
        self.client = client
        self.telegram_bot = telegram_bot
        self.position_manager = PositionManager(client)
        self.trade_recorder = TradeRecorder()  # 创建交易记录器

        # 持仓数据：{symbol: position_info}
        self.positions: Dict[str, Dict] = {}

        # ✅ 生产工具 - 订单持久化
        self.order_persistence = get_order_persistence()
        logger.info("订单持久化管理器已初始化")

        # ✅ 生产工具 - 重复订单检测
        self.duplicate_detector = get_duplicate_detector(cooldown_seconds=60)
        logger.info("重复订单检测器已初始化 (60秒冷却)")

        # ✅ 生产工具 - 订单冲突检测
        self.conflict_detector = get_conflict_detector()
        logger.info("订单冲突检测器已初始化")

        # ✅ 生产工具 - 止损单管理
        self.stop_loss_manager = get_stop_loss_manager(client)
        logger.info("止损单管理器已初始化")

        # ✅ 生产工具 - API限流器
        self.rate_limiter = get_adaptive_rate_limiter()
        logger.info("API限流器已初始化")

        # ✅ 生产工具 - 数据验证器
        self.data_checker = get_data_quality_checker()
        logger.info("数据验证器已初始化")

        # 初始化时同步币安API上的现有持仓
        self._sync_existing_positions()

        # 账户资金缓存
        self._balance_cache = None
        self._balance_cache_time = 0

        # 运行控制
        self._running = False
        self._monitor_thread = None

    def _sync_existing_positions(self):
        """
        同步币安API上的现有持仓到本地positions字典
        防止程序重启后重复开仓或错过持仓管理
        """
        try:
            logger.info("正在同步币安API上的现有持仓...")

            # 获取币安API上的所有持仓
            existing_positions = self.position_manager.get_all_positions()

            if not existing_positions:
                logger.info("币安API上没有活跃持仓")
                return

            synced_count = 0
            for pos in existing_positions:
                symbol = pos['symbol']

                # 如果本地还没有这个持仓，添加到本地管理
                if symbol not in self.positions:
                    logger.info(f"发现新持仓: {symbol}")
                    
                    # ✅ 优先：尝试从订单历史恢复
                    position_info = self._recover_position_from_order_history(symbol, pos)
                    
                    # ✅ 保底：如果订单历史恢复失败，暂时跳过（不使用推算方法）
                    if position_info is None:
                        logger.info(f"{symbol} 订单历史恢复失败，暂时跳过")
                        return synced_count

                    # 创建建仓计划
                    self._create_position_building_plan(position_info, api_entry_price, pos['side'])

                    # 添加到本地持仓管理
                    self.positions[symbol] = position_info
                    synced_count += 1

                    # ✅ 持久化
                    import json
                    from pathlib import Path
                    try:
                        positions_file = Path('data/positions.json')
                        positions_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(positions_file, 'w', encoding='utf-8') as f:
                            json.dump(self.positions, f, indent=2, ensure_ascii=False)
                        logger.debug("持仓已持久化")
                    except Exception as e:
                        logger.warning(f"持���化失败: {str(e)}")

                else:
                    logger.debug(f"持仓 {symbol} 已在本地管理中")

            if synced_count > 0:
                logger.info(f"成功同步 {synced_count} 个现有持仓到本地管理")
            else:
                logger.info("所有现有持仓已在本地管理中")

        except Exception as e:
            logger.error(f"同步现有持仓失败: {str(e)}", exc_info=True)
    
    def _recover_position_from_order_history(self, symbol: str, api_position: Dict) -> Optional[Dict]:
        """
        从币安订单历史恢复完整的持仓状态
        
        Args:
            symbol: 币种符号
            api_position: 币安API返回的持仓信息
        
        Returns:
            Dict: 完整的position_info，如果无法恢复则返回None
        """
        try:
            logger.info(f"{symbol} 从订单历史恢复持仓...")
            
            # 1. 获取该币种的所有订单历史（最近24小时）
            orders = self.client.get_all_orders(symbol=symbol, limit=100)
            
            if not orders:
                logger.warning(f"{symbol} 没有找到订单历史")
                return None
            
            # 2. 过滤出已成交的市价单和限价单（与持仓方向匹配的）
            position_amt = api_position['position_amt']
            direction = 'LONG' if position_amt > 0 else 'SHORT'
            side = 'BUY' if direction == 'LONG' else 'SELL'
            
            filled_orders = [
                order for order in orders
                if order['status'] == 'FILLED'
                and order['side'] == side
                and order['type'] in ['MARKET', 'LIMIT']
            ]
            
            if not filled_orders:
                logger.warning(f"{symbol} 没有找到已成交订单")
                return None
            
            # 3. 获取成交历史（更精确的成交信息）
            trades = self.client.get_user_trades(symbol=symbol, limit=500)
            
            if trades:
                # 过滤与持仓方向相同的成交
                side_trades = [t for t in trades if t['side'] == side]
                
                if side_trades:
                    # 按时间排序
                    side_trades.sort(key=lambda t: t['time'])
                    
                    # 计算总成交数量和金额
                    total_qty = sum(float(t['qty']) for t in side_trades)
                    total_quote = sum(float(t['quoteQty']) for t in side_trades)
                    
                    # 精确的平均价格
                    if total_qty > 0:
                        avg_price = total_quote / total_qty
                        
                        logger.info(f"{symbol} 订单历史分析:")
                        logger.info(f"  - 订单数量: {len(filled_orders)}")
                        logger.info(f"  - 成交总量: {total_qty:.2f}")
                        logger.info(f"  - 平均价格: {avg_price:.6f}")
                        logger.info(f"  - 方向: {direction}")
                    else:
                        logger.warning(f"{symbol} 成交历史中没有有效记录")
                        return None
                else:
                    logger.warning(f"{symbol} 成交历史中没有匹配方向的交易")
                    return None
            else:
                # 使用订单数据作为保底
                total_executed_quantity = sum(float(order['executedQty']) for order in filled_orders)
                total_quote_qty = sum(float(order.get('cummulativeQuoteQty', 0)) for order in filled_orders)
                
                if total_executed_quantity > 0:
                    avg_price = total_quote_qty / total_executed_quantity
                    total_qty = total_executed_quantity
                else:
                    avg_price = api_position['entry_price']
                    total_qty = abs(position_amt)
            
            # 4. 推算加仓层级（基于订单时间戳和时间间隔）
            # 按时间排序订单
            filled_orders.sort(key=lambda o: o['time'])
            
            # 判断加仓层级（基于订单时间的分组）
            added_levels = []
            current_level = 0
            level_start_time = filled_orders[0]['time'] if filled_orders else int(time.time() * 1000)
            level_quantity = 0
            level_quote = 0
            
            # 根据订单分析加仓层级
            # 假设同一个监控周期（5分钟）内的订单属于同一层级
            monitor_window = settings.MONITOR_INTERVAL * 60 * 1000  # 监控周期的毫秒数
            
            for order in filled_orders:
                order_time = order['time']
                order_qty = float(order['executedQty'])
                order_quote = float(order.get('cummulativeQuoteQty', 0))
                
                # 如果超过监控周期，认为是新的加仓层级
                if order_time - level_start_time > monitor_window:
                    if level_quantity > 0:
                        # 保存上一层级
                        level_price = level_quote / level_quantity if level_quantity > 0 else avg_price
                        added_levels.append({
                            'level': current_level,
                            'price': level_price,
                            'quantity': level_quantity,
                            'investment': level_quote / api_position['leverage'],
                            'time': level_start_time
                        })
                        logger.debug(f"{symbol} 识别第{current_level}级加仓: {level_quantity:.2f} @ ${level_price:.6f}")
                    
                    current_level += 1
                    level_start_time = order_time
                    level_quantity = 0
                    level_quote = 0
                
                level_quantity += order_qty
                level_quote += order_quote
            
            # 保存最后一层级
            if level_quantity > 0:
                level_price = level_quote / level_quantity if level_quantity > 0 else avg_price
                added_levels.append({
                    'level': current_level,
                    'price': level_price,
                    'quantity': level_quantity,
                    'investment': level_quote / api_position['leverage'],
                    'time': level_start_time
                })
                logger.debug(f"{symbol} 识别第{current_level}级加仓: {level_quantity:.2f} @ ${level_price:.6f}")
            
            logger.info(f"{symbol} 识别到 {len(added_levels)} 个加仓层级:")
            for level in added_levels:
                logger.info(f"  - 第{level['level']}级: {level['quantity']:.2f} @ ${level['price']:.6f}, 投资=${level['investment']:.2f}")
            
            # 5. 构建完整的持仓信息
            api_notional = abs(position_amt) * api_position['entry_price']
            api_margin = api_notional / api_position['leverage']
            
            position_info = {
                'symbol': symbol,
                'entry_price': api_position['entry_price'],
                'current_price': self.client.get_ticker_price(symbol),
                'direction': direction,
                'total_quantity': abs(position_amt),
                'total_investment': api_margin,
                'initial_margin': added_levels[0]['investment'] if added_levels else api_margin,
                'completed_investment': api_margin,
                'leverage': api_position['leverage'],
                'added_levels': added_levels,
                'pending_orders': [],
                'is_closing': False,
                'last_action_time': api_position.get('updateTime', int(time.time() * 1000)),
                'status': 'active',
                'take_profit_levels': {},
                'stop_loss_levels': {},
                # 标记为从订单历史恢复
                '_recovered_order_history': True
            }
            
            # 6. 重新计算利润
            current_price = self.client.get_ticker_price(symbol)
            if direction == 'LONG':
                position_info['profit_pct'] = ((current_price - api_position['entry_price']) / api_position['entry_price']) * api_position['leverage'] * 100
            else:
                position_info['profit_pct'] = ((api_position['entry_price'] - current_price) / api_position['entry_price']) * api_position['leverage'] * 100
            
            position_info['profit'] = api_position['unrealized_pnl']
            position_info['max_profit_pct'] = position_info['profit_pct']
            
            logger.info(f"{symbol} ✅ 从订单历史成功恢复持仓: {len(added_levels)}个层级, 总投资={api_margin:.2f}")
            
            return position_info
            
        except Exception as e:
            logger.error(f"{symbol} 从订单历史恢复持仓失败: {str(e)}", exc_info=True)
            return None

    def _sync_position_states_from_api(self):
        """
        从币安API同步最新的持仓状态
        处理手动平仓、外部交易等情况
        """
        try:
            # 获取币安API上的最新持仓
            api_positions = self.position_manager.get_all_positions()

            # 创建API持仓的字典，便于查找
            api_positions_dict = {pos['symbol']: pos for pos in api_positions}

            # 检查本地持仓状态
            positions_to_remove = []
            positions_to_update = []

            for symbol, local_pos in list(self.positions.items()):
                api_pos = api_positions_dict.get(symbol)

                if api_pos is None:
                    # 币安API上没有这个持仓，说明已被平仓
                    logger.info(f"检测到持仓已平仓: {symbol} (API上不存在)")
                    positions_to_remove.append(symbol)

                    # 发送平仓通知（外部检测到平仓）
                    self._send_trade_message(
                        symbol=symbol,
                        action='CLOSE',
                        direction=local_pos['direction'],
                        current_investment=local_pos['total_investment'],
                        total_planned_investment=local_pos.get('allocated_funds', settings.SINGLE_SYMBOL_MAX_INVESTMENT),
                        quantity=local_pos['total_quantity'],
                        amount=local_pos['total_investment'],
                        price=local_pos['current_price'],
                        current_price=local_pos['current_price'],
                        leverage=local_pos['leverage'],
                        profit=local_pos['profit'],
                        profit_pct=local_pos['profit_pct'],
                        initial_investment=local_pos.get('initial_margin')
                    )

                    # ✅ 记录平仓到交易表格
                    try:
                        ratio_str = "0/100"  # 平仓后仓位为0
                        record = self.trade_recorder.record_trade(
                            symbol=symbol,
                            action='CLOSE',
                            direction=local_pos['direction'],
                            price=local_pos['current_price'],
                            quantity=local_pos['total_quantity'],
                            amount=local_pos['total_investment'],
                            leverage=local_pos['leverage'],
                            ratio=ratio_str
                        )
                        # 更新盈亏数据
                        record.profit = local_pos['profit']
                        record.profit_pct = local_pos['profit_pct']
                    except Exception as e:
                        logger.warning(f"{symbol} 记录平仓失败: {str(e)}")

                elif abs(api_pos['position_amt']) <= 0.000001:
                    # 币安API上的持仓数量为0，也视为已平仓
                    logger.info(f"检测到持仓已清空: {symbol} (API数量: {api_pos['position_amt']})")
                    positions_to_remove.append(symbol)

                else:
                    # 持仓仍然存在，检查是否需要更新
                    api_quantity = abs(api_pos['position_amt'])
                    local_quantity = local_pos['total_quantity']

                    # 如果数量差异较大，可能是外部操作导致的
                    quantity_diff = abs(api_quantity - local_quantity) / max(local_quantity, api_quantity)
                    if quantity_diff > 0.01:  # 1%的差异阈值
                        logger.warning(f"检测到持仓数量变化: {symbol} 本地{local_quantity:.6f} vs API{api_quantity:.6f}")
                        positions_to_update.append((symbol, api_pos))

            # 执行移除操作
            for symbol in positions_to_remove:
                del self.positions[symbol]
                logger.info(f"已从本地管理中移除持仓: {symbol}")

            # 执行更新操作
            for symbol, api_pos in positions_to_update:
                local_pos = self.positions[symbol]

                # 更新数量和价格
                old_quantity = local_pos['total_quantity']
                new_quantity = abs(api_pos['position_amt'])

                local_pos['total_quantity'] = new_quantity
                local_pos['entry_price'] = api_pos['entry_price']

                # 重新计算投资金额
                notional_value = new_quantity * api_pos['entry_price']
                local_pos['total_investment'] = notional_value / api_pos['leverage']
                local_pos['initial_margin'] = local_pos['total_investment']

                # 重新计算利润
                current_price = self.client.get_ticker_price(symbol)
                if api_pos['side'] == 'LONG':
                    local_pos['profit_pct'] = ((current_price - api_pos['entry_price']) / api_pos['entry_price']) * api_pos['leverage'] * 100
                else:
                    local_pos['profit_pct'] = ((api_pos['entry_price'] - current_price) / api_pos['entry_price']) * api_pos['leverage'] * 100

                local_pos['profit'] = local_pos['total_investment'] * local_pos['profit_pct'] / 100
                local_pos['current_price'] = current_price

                logger.info(f"已更新持仓状态: {symbol} 数量 {old_quantity:.6f} -> {new_quantity:.6f}")

            # 检查是否有新的持仓（在API上但不在本地）
            for symbol, api_pos in api_positions_dict.items():
                if symbol not in self.positions and abs(api_pos['position_amt']) > 0.000001:
                    logger.info(f"检测到新的外部持仓: {symbol}，尝试同步")

                    # 检查是否超过持仓上限
                    if len(self.positions) >= settings.MAX_POSITIONS:
                        logger.warning(f"外部持仓 {symbol} 同步失败: 持仓数量已达上限 ({settings.MAX_POSITIONS})")
                        continue

                    # 创建持仓信息（基于币安API数据）
                    position_info = {
                        'symbol': symbol,
                        'entry_price': api_pos['entry_price'],
                        'current_price': api_pos['entry_price'],  # 暂时使用开仓价
                        'direction': api_pos['side'],
                        'total_quantity': abs(api_pos['position_amt']),
                        'total_investment': 0.0,  # 需要估算，无法从API直接获取
                        'initial_margin': 0.0,
                        'completed_investment': 0.0,
                        'profit': api_pos['unrealized_pnl'],
                        'profit_pct': 0.0,  # 需要重新计算
                        'max_profit_pct': 0.0,
                        'leverage': api_pos['leverage'],
                        'added_levels': [],
                        'pending_orders': [],  # 外部持仓没有本地挂单
                        'is_closing': False,
                        'last_action_time': int(time.time()),
                        'status': 'active',
                        'take_profit_levels': {},
                        'stop_loss_levels': {},
                        'position_complete': False  # 标记建仓是否完成
                    }

                    # 估算投资金额（名义价值 / 杠杆）
                    notional_value = abs(api_pos['position_amt']) * api_pos['entry_price']
                    position_info['total_investment'] = notional_value / api_pos['leverage']
                    position_info['initial_margin'] = position_info['total_investment']

                    # 获取当前价格并重新计算利润率
                    current_price = self.client.get_ticker_price(symbol)
                    position_info['current_price'] = current_price

                    if api_pos['side'] == 'LONG':
                        position_info['profit_pct'] = ((current_price - api_pos['entry_price']) / api_pos['entry_price']) * api_pos['leverage'] * 100
                    else:
                        position_info['profit_pct'] = ((api_pos['entry_price'] - current_price) / api_pos['entry_price']) * api_pos['leverage'] * 100

                    position_info['max_profit_pct'] = position_info['profit_pct']  # 初始设置为当前值

                    # 添加到本地持仓管理
                    self.positions[symbol] = position_info
                    logger.info(f"外部持仓 {symbol} 已同步到本地管理: {api_pos['side']} {abs(api_pos['position_amt'])} @ {api_pos['entry_price']}, 杠杆{api_pos['leverage']}x")

        except Exception as e:
            logger.error(f"同步持仓状态失败: {str(e)}", exc_info=True)

    def _choose_order_type(self, symbol: str, quantity: float, current_price: float) -> str:
        """
        智能选择订单类型

        Args:
            symbol: 币种
            quantity: 订单数量
            current_price: 当前价格

        Returns:
            str: 'MARKET' 或 'LIMIT'
        """
        try:
            # 计算名义价值
            notional_value = quantity * current_price

            # 小额订单（<1000 USDT）使用市价单
            if notional_value < 1000:
                return 'MARKET'

            # 检查24小时交易量
            try:
                ticker_24h = self.client.get_ticker_24h(symbol)
                volume = float(ticker_24h.get('volume', 0)) * float(ticker_24h.get('lastPrice', current_price))

                # 低流动性币种使用限价单
                if volume < 100000:  # 日交易量小于10万美元
                    logger.info(f"{symbol} 流动性较低，使用限价单 (日交易量: {volume:.0f} USDT)")
                    return 'LIMIT'

            except Exception as e:
                logger.debug(f"{symbol} 无法获取交易量信息: {str(e)}")

            # 默认使用市价单
            return 'MARKET'

        except Exception as e:
            logger.warning(f"{symbol} 选择订单类型失败: {str(e)}")
            return 'MARKET'

    def _calculate_limit_price(self, symbol: str, side: str, current_price: float) -> float:
        """
        计算限价单价格

        Args:
            symbol: 币种
            side: 买卖方向
            current_price: 当前价格

        Returns:
            float: 限价单价格
        """
        try:
            # 获取价格精度
            precision_info = self.client.get_tick_size_and_precision(symbol)
            tick_size = precision_info['tick_size']

            if side == 'BUY':
                # 买入时稍微高于当前价格
                limit_price = current_price * 1.001  # 高1%
            else:
                # 卖出时稍微低于当前价格
                limit_price = current_price * 0.999  # 低1%

            # 调整到tick_size
            limit_price = round(limit_price / tick_size) * tick_size

            logger.debug(f"{symbol} 限价单价格: {current_price:.6f} -> {limit_price:.6f}")
            return limit_price

        except Exception as e:
            logger.warning(f"{symbol} 计算限价失败: {str(e)}")
            return current_price

    # ==================== 警报处理 ====================

    def handle_alert(self, alert_data: Dict[str, Any]):
        """
        处理警报（从alert模块调用）

        Args:
            alert_data: 警报数据
        """
        try:
            symbol = alert_data.get('symbol')
            logger.info(f"收到警报: {symbol}")

            # 尝试开仓
            if self.open_position(alert_data):
                logger.info(f"{symbol} 开仓成功")
            else:
                logger.info(f"{symbol} 开仓失败或跳过")

        except Exception as e:
            logger.error(f"处理警报失败: {str(e)}", exc_info=True)

    # ==================== 资金管理 ====================

    def get_available_funds(self) -> float:
        """
        获取可用资金

        Returns:
            float: 可用资金（USDT）
        """
        try:
            # 使用缓存，避免频繁API调用（5秒缓存）
            current_time = time.time()
            if self._balance_cache and (current_time - self._balance_cache_time) < 5:
                return self._balance_cache

            # 获取账户余额
            account_info = self.client.get_account_balance()

            # 可用资金 = 总余额 - 已用保证金 - 未平仓亏损的绝对值
            total_balance = account_info['total_balance']
            used_margin = account_info['total_margin']
            unrealized_pnl = account_info['unrealized_pnl']

            # 如果未平仓亏损为负数（亏损），取绝对值
            loss_amount = abs(unrealized_pnl) if unrealized_pnl < 0 else 0

            available_funds = total_balance - used_margin - loss_amount

            # 缓存结果
            self._balance_cache = available_funds
            self._balance_cache_time = current_time

            logger.debug(f"可用资金计算: 总余额={total_balance:.2f}, "
                       f"已用保证金={used_margin:.2f}, "
                       f"未平仓亏损={unrealized_pnl:.2f}, "
                       f"可用={available_funds:.2f}")

            return available_funds

        except Exception as e:
            logger.error(f"获取可用资金失败: {str(e)}")
            return 0

    def _get_max_order_quantity(self, symbol: str, price: float) -> float:
        """
        获取币种的最大订单数量限制

        Args:
            symbol: 币种
            price: 当前价格

        Returns:
            float: 最大订单数量
        """
        try:
            # 获取币种信息
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info or 'filters' not in symbol_info:
                return float('inf')  # 如果无法获取，返回无限大

            max_qty = float('inf')
            for f in symbol_info['filters']:
                if f.get('filterType') == 'LOT_SIZE':
                    max_qty = float(f.get('maxQty', float('inf')))

            # 也可以考虑最大名义价值限制
            # 但这里主要关注数量限制

            return max_qty

        except Exception as e:
            logger.warning(f"{symbol} 获取最大订单数量失败: {str(e)}")
            return float('inf')

    def _calculate_safe_order_params(self, symbol: str, target_margin: float, target_leverage: int, current_price: float):
        """
        计算安全的下单参数，先读取币安限制再进行修改

        Args:
            symbol: 币种
            target_margin: 目标保证金
            target_leverage: 目标杠杆
            current_price: 当前价格

        Returns:
            dict: 安全的下单参数
        """
        try:
            # ===== 第一步：从币安读取所有限制条件 =====
            
            # 1.1 获取杠杆限制
            max_leverage = self.client.get_leverage_bracket(symbol)
            logger.debug(f"{symbol} 币安最大杠杆: {max_leverage}x")

            # 1.2 获取最大保证金限制（统一使用_get_max_margin_for_symbol方法，P1修复）
            max_margin = self._get_max_margin_for_symbol(symbol, target_leverage)
            logger.debug(f"{symbol} 杠杆{target_leverage}x 最大保证金: {max_margin:.2f} USDT")
            max_notional = max_margin * target_leverage
            logger.debug(f"{symbol} 杠杆{target_leverage}x 最大名义价值: {max_notional:.2f} USDT")

            # 1.3 获取币种的LOT_SIZE和价格过滤器
            symbol_info = self.client.get_symbol_info(symbol)
            min_qty = 0.001
            max_qty = float('inf')
            min_notional_filter = 0
            
            if symbol_info and 'filters' in symbol_info:
                for f in symbol_info['filters']:
                    if f.get('filterType') == 'LOT_SIZE':
                        min_qty = float(f.get('minQty', 0.001))
                        max_qty = float(f.get('maxQty', float('inf')))
                    elif f.get('filterType') == 'NOTIONAL':
                        min_notional_filter = float(f.get('minNotional', 0))

            logger.debug(f"{symbol} LOT_SIZE: [{min_qty}, {max_qty}], minNotional: {min_notional_filter}")

            # ===== 第二步：根据读取的限制调整杠杆 =====
            safe_leverage = min(target_leverage, max_leverage)
            if safe_leverage != target_leverage:
                logger.warning(f"{symbol} 杠杆从{target_leverage}x调整为{safe_leverage}x（受币安限制）")

            # ===== 第三步：根据读取的限制调整保证金和名义价值 =====

            # 3.1 检查名义价值限制
            target_notional = target_margin * safe_leverage
            safe_notional = min(target_notional, max_notional)

            if safe_notional < target_notional:
                safe_margin = safe_notional / safe_leverage
                logger.warning(f"{symbol} 保证金从{target_margin:.2f}调整为{safe_margin:.2f}（受名义价值限制）")
            else:
                safe_margin = target_margin

            # 3.2 检查是否会超出最大持仓限制（提前预防）
            if safe_notional > max_notional * 0.95:  # 如果接近最大持仓的95%
                # 降低杠杆或保证金nicai
                if safe_leverage > 5:
                    safe_leverage = max(5, safe_leverage - 5)
                    safe_notional = safe_margin * safe_leverage
                    logger.warning(f"{symbol} 降低杠杆到{safe_leverage}x以避免超出最大持仓限制")
                else:
                    safe_margin = safe_margin * 0.8  # 减少20%保证金
                    safe_notional = safe_margin * safe_leverage
                    logger.warning(f"{symbol} 减少保证金到{safe_margin:.2f}以避免超出最大持仓限制")

            # ===== 第四步：计算数量并根据LOT_SIZE限制调整 =====

            # 4.1 初始计算数量
            quantity = (safe_margin * safe_leverage) / current_price

            # 4.2 使用全面的数量验证和调整
            quantity, is_valid = self._validate_and_adjust_quantity(symbol, quantity, current_price, safe_leverage)

            if not is_valid:
                logger.error(f"{symbol} 数量验证失败，使用保守参数")
                # 使用最小允许数量
                quantity = min_qty
                quantity = self.client.adjust_quantity_precision(symbol, quantity)

            # 4.3 重新计算保证金（基于调整后的数量）
            final_notional = quantity * current_price
            safe_margin = final_notional / safe_leverage

            logger.info(f"{symbol} 最终下单参数: 杠杆={safe_leverage}x, 保证金={safe_margin:.2f} USDT, "
                       f"数量={quantity:.8f}, 名义价值={final_notional:.2f} USDT, 价格={current_price:.6f}")

            return {
                'leverage': safe_leverage,
                'margin': safe_margin,
                'quantity': quantity,
                'notional': final_notional,
                'price': current_price
            }

        except Exception as e:
            logger.error(f"{symbol} 计算安全下单参数失败: {str(e)}")
            # 返回保守的默认值
            safe_leverage = min(target_leverage, 20)
            quantity = (target_margin * safe_leverage) / current_price
            return {
                'leverage': safe_leverage,
                'margin': target_margin,
                'quantity': quantity,
                'notional': target_margin * safe_leverage,
                'price': current_price
            }

    def _validate_and_adjust_quantity(self, symbol: str, quantity: float, price: float, leverage: int) -> tuple[float, bool]:
        """
        全面验证和调整数量参数

        Args:
            symbol: 币种
            quantity: 原始数量
            price: 价格
            leverage: 杠杆倍数

        Returns:
            tuple: (调整后的数量, 是否有效)
        """
        try:
            # 获取交易对信息
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"{symbol} 无法获取交易对信息，使用原始数量")
                return quantity, True

            # 初始化过滤器值
            lot_size_min_qty = 0.001
            lot_size_max_qty = float('inf')
            lot_size_step_size = 0.001
            min_notional = 0
            max_notional = float('inf')

            # 解析所有相关过滤器
            for f in symbol_info.get('filters', []):
                filter_type = f.get('filterType')
                if filter_type == 'LOT_SIZE':
                    lot_size_min_qty = float(f.get('minQty', 0.001))
                    lot_size_max_qty = float(f.get('maxQty', float('inf')))
                    lot_size_step_size = float(f.get('stepSize', 0.001))
                elif filter_type == 'NOTIONAL':
                    min_notional = float(f.get('minNotional', 0))
                elif filter_type == 'MAX_POSITION':
                    max_notional = float(f.get('maxPosition', float('inf')))

            logger.debug(f"{symbol} 过滤器: LOT_SIZE=[{lot_size_min_qty}, {lot_size_max_qty}, step={lot_size_step_size}], "
                        f"NOTIONAL=[{min_notional}, {max_notional}]")

            # 1. 检查并调整数量精度
            original_quantity = quantity

            # 先检查LOT_SIZE最小值，如果太小就直接设置为最小值
            if quantity < lot_size_min_qty:
                logger.warning(f"{symbol} 数量 {quantity:.8f} 小于LOT_SIZE最小值 {lot_size_min_qty}")
                quantity = lot_size_min_qty
                quantity = self.client.adjust_quantity_precision(symbol, quantity)
                logger.info(f"{symbol} 调整数量到最小值: {quantity:.8f}")
            else:
                # 正常调整精度
                quantity = self.client.adjust_quantity_precision(symbol, quantity)
                if abs(quantity - original_quantity) > 1e-8:
                    logger.debug(f"{symbol} 数量精度调整: {original_quantity:.8f} -> {quantity:.8f}")

            # 2. 检查LOT_SIZE最大值
            if quantity > lot_size_max_qty:
                logger.warning(f"{symbol} 数量 {quantity:.8f} 超过LOT_SIZE最大值 {lot_size_max_qty}")
                quantity = lot_size_max_qty
                quantity = self.client.adjust_quantity_precision(symbol, quantity)
                logger.info(f"{symbol} 调整数量到最大值: {quantity:.8f}")

            # 3. 检查名义价值最小值
            # 计算满足minNotional的最小数量（向上取整到step_size倍数）
            # ✅ 修复：使用更精确的向上取整方法，避免四舍五入破坏ceil结果
            min_qty_calc = min_notional / price
            min_qty_for_notional = math.ceil(min_qty_calc / lot_size_step_size) * lot_size_step_size

            # ✅ 修复：手动确保数量是step_size的整数倍（向上取整）
            min_qty_for_notional = math.ceil(min_qty_for_notional / lot_size_step_size) * lot_size_step_size

            # ✅ 修复：不调用adjust_quantity_precision，避免四舍五入回退
            # 改为手动确保满足minNotional
            while (min_qty_for_notional * price) < min_notional:
                min_qty_for_notional += lot_size_step_size

            # 取较大值（用户期望数量 或 满足minNotional的最小数量）
            notional_value = quantity * price
            if notional_value < min_notional:
                logger.warning(f"{symbol} 名义价值 {notional_value:.2f} USDT 小于最小要求 {min_notional} USDT")
                quantity = max(quantity, min_qty_for_notional)

                # ✅ 修复：确保调整后的数量满足minNotional
                while (quantity * price) < min_notional:
                    quantity = math.ceil((min_notional / price + 0.000000001) / lot_size_step_size) * lot_size_step_size

                notional_value = quantity * price
                logger.info(f"{symbol} 调整数量到满足minNotional: {quantity:.8f} (名义价值: {notional_value:.2f} USDT)")

            # 5. 检查名义价值最大值（基于杠杆）
            max_notional_for_leverage = self.client.get_max_notional_for_leverage(symbol, leverage)
            if max_notional_for_leverage > 0 and notional_value > max_notional_for_leverage:
                logger.warning(f"{symbol} 名义价值 {notional_value:.2f} 超过杠杆{leverage}x最大值 {max_notional_for_leverage:.2f}")
                max_quantity = max_notional_for_leverage / price
                quantity = self.client.adjust_quantity_precision(symbol, max_quantity)
                notional_value = quantity * price
                logger.info(f"{symbol} 调整数量以符合杠杆限制: {quantity:.8f} (名义价值: {notional_value:.2f})")

            # 6. 最终验证
            final_notional = quantity * price
            is_valid = (quantity >= lot_size_min_qty and
                       quantity <= lot_size_max_qty and
                       final_notional >= min_notional and
                       (max_notional_for_leverage <= 0 or final_notional <= max_notional_for_leverage))

            if is_valid:
                logger.debug(f"{symbol} 数量验证通过: {quantity:.8f} (名义价值: {final_notional:.2f})")
            else:
                logger.error(f"{symbol} 数量验证失败: {quantity:.8f} (名义价值: {final_notional:.2f})")

            return quantity, is_valid

        except Exception as e:
            logger.error(f"{symbol} 数量验证失败: {str(e)}")
            return quantity, False

    def _validate_order_quantity_unified(
        self,
        symbol: str,
        quantity: float,
        price: float,
        order_type: str = 'MARKET',
        ensure_min_notional: bool = True,
        max_quantity_limit: Optional[float] = None,
        description: str = '订单'
    ) -> Tuple[float, bool, Dict]:
        """
        统一的订单数量验证函数 - 适用于所有订单类型（开仓、加仓、止损、止盈）
        
        Args:
            symbol: 币种符号
            quantity: 待验证的数量
            price: 价格（用于计算名义价值）
            order_type: 订单类型（MARKET, LIMIT, STOP等）
            ensure_min_notional: 是否确保满足最小名义价值（止损单通常需要）
            max_quantity_limit: 最大数量限制（例如止损单限制平仓比例）
            description: 订单描述（用于日志）
        
        Returns:
            tuple: (验证后的数量, 是否有效, 调整信息)
        """
        try:
            adjustment_info = {
                'original_quantity': quantity,
                'adjusted': False,
                'adjustments': []
            }
            
            # 获取交易对信息
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"{symbol} {description}: 无法获取交易对信息，使用原始数量")
                return quantity, False, adjustment_info
            
            # 解析过滤器
            min_qty = 0.001
            max_qty = float('inf')
            step_size = 0.001
            min_notional = 0.0  # 从币安API获取，不使用默认值
            max_notional = float('inf')

            for f in symbol_info.get('filters', []):
                filter_type = f.get('filterType')
                if filter_type == 'LOT_SIZE':
                    min_qty = float(f.get('minQty', 0.001))
                    max_qty = float(f.get('maxQty', float('inf')))
                    step_size = float(f.get('stepSize', 0.001))
                elif filter_type == 'NOTIONAL':
                    min_notional = float(f.get('minNotional', 0.0))
                elif filter_type == 'MAX_POSITION':
                    max_notional = float(f.get('maxPosition', float('inf')))
            
            # 1. LOT_SIZE最小值检查
            if quantity < min_qty:
                old_qty = quantity
                quantity = min_qty
                adjustment_info['adjustments'].append(f'LOT_SIZE最小值: {old_qty:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.warning(f"{symbol} {description}: 数量 {old_qty:.8f} 小于最小值 {min_qty}，调整到 {quantity:.8f}")
            
            # 2. 调整数量精度（使用step_size）
            original = quantity
            quantity = round(quantity / step_size) * step_size
            if abs(quantity - original) > 1e-8:
                adjustment_info['adjustments'].append(f'精度调整: {original:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.debug(f"{symbol} {description}: 数量精度调整 {original:.8f} -> {quantity:.8f}")
            
            # 3. LOT_SIZE最大值检查
            if quantity > max_qty:
                old_qty = quantity
                quantity = max_qty
                quantity = round(quantity / step_size) * step_size
                adjustment_info['adjustments'].append(f'LOT_SIZE最大值: {old_qty:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.warning(f"{symbol} {description}: 数量 {old_qty:.8f} 超过最大值 {max_qty}，调整到 {quantity:.8f}")
            
            # 4. 最大数量限制检查（例如止损单限制平仓比例）
            if max_quantity_limit is not None and quantity > max_quantity_limit:
                old_qty = quantity
                quantity = max_quantity_limit
                quantity = round(quantity / step_size) * step_size
                adjustment_info['adjustments'].append(f'最大数量限制: {old_qty:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.warning(f"{symbol} {description}: 数量 {old_qty:.8f} 超过限制 {max_quantity_limit:.8f}，调整到 {quantity:.8f}")
            
            # 5. ✅ 关键：minNotional验证和调整
            notional_value = quantity * price
            
            if ensure_min_notional and notional_value < min_notional:
                # 计算满足minNotional的最小数量（向上取整）
                min_qty_calc = min_notional / price
                min_qty_for_notional = math.ceil(min_qty_calc / step_size) * step_size
                
                # 确保是step_size的整数倍
                min_qty_for_notional = math.ceil(min_qty_for_notional / step_size) * step_size
                
                # 确保满足minNotional（向上取整）
                while (min_qty_for_notional * price) < min_notional:
                    min_qty_for_notional += step_size
                
                # 取较大值（当前数量 或 满足minNotional的最小数量）
                old_qty = quantity
                quantity = max(quantity, min_qty_for_notional)
                
                # 调整到step_size的整数倍
                quantity = round(quantity / step_size) * step_size
                
                notional_value = quantity * price
                adjustment_info['adjustments'].append(f'minNotional: {old_qty:.8f} -> {quantity:.8f} (名义价值: {notional_value:.2f} USDT)')
                adjustment_info['adjusted'] = True
                
                logger.warning(f"{symbol} {description}: 名义价值 {old_qty * price:.2f} USDT < 最小要求 {min_notional} USDT")
                logger.info(f"{symbol} {description}: 调整数量到 {quantity:.8f} (名义价值: {notional_value:.2f} USDT)")
            
            # 6. 最终验证
            final_notional = quantity * price
            is_valid = (
                quantity >= min_qty and
                quantity <= max_qty and
                final_notional >= min_notional and
                (max_notional <= 0 or final_notional <= max_notional)
            )
            
            if not is_valid:
                logger.error(f"{symbol} {description}: 最终验证失败 - 数量: {quantity:.8f}, 名义价值: {final_notional:.2f}")
            
            adjustment_info['final_quantity'] = quantity
            adjustment_info['final_notional'] = final_notional
            adjustment_info['actual_min_notional'] = min_notional  # 保存从API获取的真实min_notional

            return quantity, is_valid, adjustment_info
            
        except Exception as e:
            logger.error(f"{symbol} {description}: 数量验证失败 - {str(e)}", exc_info=True)
            return quantity, False, adjustment_info

    def _get_max_margin_for_symbol(self, symbol: str, leverage: int) -> float:
        """
        获取币种的最大保证金限制

        Args:
            symbol: 币种
            leverage: 杠杆倍数

        Returns:
            float: 最大保证金（USDT）
        """
        try:
            # 调用币安API获取最大持仓价值
            max_notional = self.client.get_max_notional_for_leverage(symbol, leverage)

            if max_notional > 0:
                # 计算最大保证金
                max_margin = max_notional / leverage
                logger.debug(f"{symbol} 杠杆{leverage}x: 最大持仓价值={max_notional:.2f}, "
                            f"最大保证金={max_margin:.2f}")
                return max_margin
            else:
                # API调用失败，使用配置值
                logger.debug(f"{symbol} 无法获取最大持仓价值，使用配置值")
            return settings.MAX_MARGIN_PER_SYMBOL

        except Exception as e:
            logger.warning(f"{symbol} 获取最大保证金失败: {str(e)}，使用配置值")
            return settings.MAX_MARGIN_PER_SYMBOL

    def _set_margin_mode_and_leverage(self, symbol: str, leverage: int, margin_mode: str):
        """
        设置保证金模式和杠杆（带智能降级重试）

        Args:
            symbol: 币种
            leverage: 杠杆倍数
            margin_mode: 保证金模式（ISOLATED/CROSSED）
        """
        try:
            # ===== 第一步：获取完整的杠杆档位信息 =====
            try:
                leverage_brackets = self.client.get_leverage_brackets(symbol)
                if leverage_brackets and len(leverage_brackets) > 0:
                    # 获取所有支持的杠杆倍数
                    available_leverages = sorted([b['initialLeverage'] for b in leverage_brackets], reverse=True)
                    max_leverage = max(available_leverages)
                    logger.info(f"{symbol} 支持的杠杆档位: {available_leverages}")
                else:
                    # 如果获取失败，使用保守的默认逻辑
                    raise Exception("无法获取杠杆档位信息")
            except Exception as e:
                logger.warning(f"{symbol} 无法获取杠杆档位信息: {str(e)}，使用默认逻辑")
                # 获取该币种支持的最大杠杆
                max_leverage = self.client.get_leverage_bracket(symbol)
                available_leverages = []
                current = max_leverage
                while current >= 5:  # 最低5倍杠杆
                    available_leverages.append(current)
                    current -= 5
                if not available_leverages:
                    available_leverages = [5]

            # ===== 第二步：根据请求杠杆生成设置序列 =====
            if leverage > max_leverage:
                logger.warning(f"{symbol} 请求杠杆 {leverage}x 超过最大支持杠杆 {max_leverage}x，自动降低到 {max_leverage}x")
                leverage = max_leverage

            # 从请求杠杆开始，逐步降低到支持的杠杆
            leverage_sequence = []
            current = leverage
            while current >= 5:
                if current in available_leverages:
                    leverage_sequence.append(current)
                current -= 1  # 每次减1，更精确

            # 如果没有找到合适的杠杆，使用最低的可用杠杆
            if not leverage_sequence and available_leverages:
                leverage_sequence = [min(available_leverages)]

            # 确保至少有5x杠杆
            if not leverage_sequence:
                leverage_sequence = [5]

            logger.info(f"{symbol} 杠杆设置序列: {leverage_sequence}")

            # 使用重试机制
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                for try_leverage in leverage_sequence:
                    try:
                        # 设置保证金模式
                        self.client.change_margin_type(symbol, margin_mode)
                        logger.info(f"{symbol} 保证金模式已设置为: {margin_mode}")

                        # 设置杠杆
                        result = self.client.change_leverage(symbol, try_leverage)
                        if result:
                            logger.info(f"{symbol} 杠杆已设置为: {try_leverage}x")

                            # 验证设置
                            try:
                                pos_info = self.client.get_position(symbol)
                                if pos_info:
                                    current_leverage = pos_info.get('leverage', 1)
                                    if current_leverage == try_leverage:
                                        logger.info(f"{symbol} 杠杆验证成功: 当前杠杆={current_leverage}x")
                                        return
                                    else:
                                        logger.warning(f"{symbol} 杠杆设置可能未生效，API返回: {current_leverage}x")
                                else:
                                    logger.debug(f"{symbol} 无法获取持仓信息进行验证")
                            except Exception as e:
                                logger.debug(f"{symbol} 杠杆验证失败: {str(e)}")

                            # 如果验证失败但API调用成功，仍然认为成功
                            return
                        else:
                            logger.warning(f"{symbol} 杠杆 {try_leverage}x 设置返回失败")

                    except BinanceAPIException as e:
                        error_code = e.code if hasattr(e, 'code') else None

                        # 杠杆相关错误，继续尝试下一个杠杆
                        if error_code in [-4004, -4005, -1111]:  # Leverage not valid, Too many requests, etc.
                            logger.warning(f"{symbol} 杠杆 {try_leverage}x 设置失败 (错误码: {error_code})，尝试下一个杠杆")
                            continue
                        else:
                            # 其他错误，记录但继续尝试
                            logger.warning(f"{symbol} 设置杠杆时遇到错误 (错误码: {error_code}): {e.message}")
                            continue

                    except Exception as e:
                        logger.warning(f"{symbol} 设置杠杆 {try_leverage}x 时异常: {str(e)}")
                        continue

                # 如果所有杠杆都尝试失败，等待后重试
                if attempt < max_retries - 1:
                    logger.warning(f"{symbol} 第{attempt+1}次尝试全部失败，等待{retry_delay}秒后重试")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    logger.error(f"{symbol} 所有杠杆设置尝试均失败")
                    raise Exception(f"无法为{symbol}设置任何有效杠杆")

        except Exception as e:
            logger.error(f"{symbol} 设置保证金模式和杠杆失败: {str(e)}")
            raise

    def _sync_position_state(self, symbol: str, position_info: Dict, all_positions: List[Dict]):
        """
        同步本地持仓状态与币安API持仓状态

        Args:
            symbol: 币种
            position_info: 本地持仓信息
            all_positions: 币安API返回的所有持仓列表
        """
        try:
            # 检查币安API上是否有该币种的持仓
            api_position = None
            for pos in all_positions:
                if pos.get('symbol') == symbol:
                    api_position = pos
                    break

            if not api_position:
                # 没有持仓
                return

            # 更新开仓均价（使用币安API的实际数据）
            api_entry_price = api_position.get('entry_price', position_info['entry_price'])
            if api_entry_price:
                position_info['entry_price'] = api_entry_price
                logger.debug(f"{symbol} 更新开仓均价: {api_entry_price:.8f}")

            # 更新数量
            api_position_amt = api_position.get('positionAmt', 0)
            if abs(api_position_amt) > 0:
                old_quantity = position_info.get('total_quantity', 0)
                new_quantity = abs(api_position_amt)
                position_info['total_quantity'] = new_quantity

                # ✅ 如果持仓数量显著增加（超过10%），清除止损验证失败标记，尝试重新创建止损单
                if (position_info.get('_stop_loss_validation_failed', False) and
                    new_quantity > old_quantity * 1.1):  # 数量增加超过10%
                    logger.info(f"{symbol} 持仓数量显著增加({old_quantity:.2f} → {new_quantity:.2f})，清除止损验证失败标记")
                    position_info['_stop_loss_validation_failed'] = False
                    position_info.pop('_stop_loss_validation_reason', None)
                    position_info.pop('_stop_loss_validation_time', None)

                logger.debug(f"{symbol} 更新数量: {new_quantity:.8f}")

        except Exception as e:
            logger.error(f"{symbol} 同步持仓状态失败: {str(e)}")

    def _add_position(self, position_info: Dict, pending_order: Dict, current_price: float):
        """
        执行加仓（动态计算限价单价格，先读取币安限制再修改）

        Args:
            position_info: 持仓信息
            pending_order: 待执行订单
            current_price: 当前价格
        """
        try:
            symbol = position_info['symbol']
            direction = position_info['direction']
            entry_price = position_info['entry_price']
            total_investment = position_info['total_investment']
            leverage = position_info['leverage']

            # ===== 第一步：获取币种的最小订单信息（固定的minQty和minNotional）=====
            # ✅ 根据官方文档，这些值是固定的，可以缓存
            min_order_info = self.client.get_min_order_info(symbol, leverage)
            min_qty = min_order_info['min_qty']
            step_size = min_order_info['step_size']
            max_qty = min_order_info['max_qty']
            min_notional = min_order_info['min_notional']
            
            logger.debug(f"{symbol} 最小订单信息 - minQty={min_qty}, stepSize={step_size}, "
                        f"maxQty={max_qty}, minNotional={min_notional}")

            # ===== 第二步：计算限价单价格（根据方向和DELAY_RATIO）=====
            # ✅ 优先计算limit_price，因为订单实际使用这个价格
            trigger_rate = pending_order['trigger_rate'] / 100  # 转换为小数

            if direction == 'LONG':
                if trigger_rate < 0:  # 亏损加仓
                    # 开仓均价 × (1 + (触发率 + DELAY_RATIO) / 杠杆倍数)
                    limit_price = entry_price * (1 + (trigger_rate + settings.DELAY_RATIO / 100) / leverage)
                else:  # 盈利加仓
                    # 开仓均价 × (1 + (触发率 - DELAY_RATIO) / 杠杆倍数)
                    limit_price = entry_price * (1 + (trigger_rate - settings.DELAY_RATIO / 100) / leverage)
            else:  # SHORT
                if trigger_rate < 0:  # 亏损加仓
                    # 开仓均价 × (1 + (触发率 + DELAY_RATIO) / 杠杆倍数)
                    limit_price = entry_price * (1 + (trigger_rate + settings.DELAY_RATIO / 100) / leverage)
                else:  # 盈利加仓
                    # 开仓均价 × (1 + (触发率 - DELAY_RATIO) / 杠杆倍数)
                    limit_price = entry_price * (1 + (trigger_rate - settings.DELAY_RATIO / 100) / leverage)

            # 调整价格精度
            limit_price = self.client.adjust_price_precision(symbol, limit_price)
            logger.debug(f"{symbol} 计算限价单价格: {limit_price:.8f}")

            # ===== 第三步：基于limit_price和minQty计算最小保证金 =====
            # ✅ 新思路：minQty是固定的最小"币种单位"
            # 不管价格多少，至少要下单min_qty这么多单位
            # 最小保证金 = min_qty × limit_price / 杠杆
            
            min_margin_at_limit_price = (min_qty * limit_price) / leverage
            
            logger.info(f"{symbol} 基于minQty的最小保证金计算: minQty={min_qty:.8f}, "
                        f"limit_price={limit_price:.8f}, 最小保证金={min_margin_at_limit_price:.2f} USDT")

            # ===== 第四步：基于用户风险额度计算数量 =====
            add_percent = pending_order['add_percent']

            # ✅ 关键改进：基于limit_price而非current_price计算
            # 用户想要投入的保证金金额
            target_margin = total_investment * (add_percent / 100)

            # ✅ 直接反推数量：目标保证金 × 杠杆 ÷ 限价价格
            target_notional = target_margin * leverage
            quantity = target_notional / limit_price

            logger.info(f"{symbol} 加仓参数计算: 目标保证金={target_margin:.2f}, 目标名义价值={target_notional:.2f}, 限价={limit_price:.8f}, 初始数量={quantity:.8f}")

            # ===== 第四步：确保数量满足min_notional要求（基于limit_price）=====
            # min_notional 从币安API获取，如果API返回0则跳过此验证（避免硬编码默认值）
            if min_notional > 0:
                min_notional_filter = min_notional

                # 计算满足min_notional的最小数量（使用limit_price）
                min_quantity_for_notional = (min_notional_filter * 1.01) / limit_price  # 增加1%余量

                # 应用LOT_SIZE规则
                min_quantity_for_notional = self.client.adjust_quantity_precision(symbol, min_quantity_for_notional)

                logger.debug(f"{symbol} 最小名义价值数量: {min_quantity_for_notional:.8f} (基于limit_price, minNotional={min_notional_filter})")

                # 比较并取较大值
                if quantity < min_quantity_for_notional:
                    logger.info(f"{symbol} 数量不足，基于limit_price从{quantity:.8f}调整到{min_quantity_for_notional:.8f}")
                    quantity = min_quantity_for_notional
            else:
                min_notional_filter = 0.0
                logger.warning(f"{symbol} 未获取到 minNotional，跳过名义价值验证")

            # 再次应用精度调整
            quantity = self.client.adjust_quantity_precision(symbol, quantity)

            # ✅ 关键：基于limit_price重新计算实际保证金和名义价值
            actual_notional = quantity * limit_price
            actual_margin = actual_notional / leverage

            logger.info(f"{symbol} 实际加仓参数: 数量={quantity:.8f}, 限价={limit_price:.8f}, 名义价值={actual_notional:.2f}, 保证金={actual_margin:.2f}")

            # 检查是否超出用户预期的风险范围（允许±10%偏差）
            margin_deviation = abs(actual_margin - target_margin) / target_margin if target_margin > 0 else 0
            if margin_deviation > 0.1:  # 超过10%
                logger.warning(f"{symbol} 实际保证金{actual_margin:.2f}与目标{target_margin:.2f}偏差{margin_deviation*100:.1f}%，可能是强制调整到最小名义价值")

            # ===== 第五步：验证LOT_SIZE范围=====
            if quantity < min_qty or quantity > max_qty:
                logger.error(f"{symbol} 数量 {quantity:.8f} 超出允许范围 [{min_qty}, {max_qty}]")
                return False

            logger.info(f"{symbol} ✅ LOT_SIZE验证通过: {quantity:.8f} ∈ [{min_qty}, {max_qty}]")

            # ===== 第六步：最终验证名义价值（使用limit_price）=====
            if min_notional > 0 and actual_notional < min_notional:
                logger.error(f"{symbol} 最终名义价值{actual_notional:.2f} USDT小于最小要求{min_notional} USDT")
                logger.error(f"  数量={quantity:.8f}, 限价={limit_price:.8f}, 实际名义价值={actual_notional:.2f}")

                # 尝试逐步增加step_size
                for retry in range(10):  # 增加到10次尝试
                    quantity = quantity + step_size
                    quantity = self.client.adjust_quantity_precision(symbol, quantity)

                    # 检查是否超出max_qty
                    if quantity > max_qty:
                        logger.error(f"{symbol} 已达到最大数量{max_qty}，仍无法满足最小名义价值")
                        return False

                    actual_notional = quantity * limit_price
                    actual_margin = actual_notional / leverage

                    if actual_notional >= min_notional_filter:
                        logger.info(f"{symbol} 增加数量到{quantity:.8f}以满足最小名义价值")
                        break
                else:
                    logger.error(f"{symbol} 10次增加后仍无法满足最小名义价值，取消加仓")
                    return False

                logger.info(f"{symbol} 调整后实际保证金={actual_margin:.2f}, 名义价值={actual_notional:.2f}")

            logger.info(f"{symbol} ✅ 最终验证通过: 名义价值={actual_notional:.2f} USDT >= {min_notional_filter} USDT")

            # ===== 第七步：检查账户是否有足够保证金 =====
            # 获取账户信息，检查可用保证金
            try:
                account_info = self.client.get_account_balance()
                available_funds = account_info['available_balance']

                logger.info(f"{symbol} 账户资金检查: 可用余额={available_funds:.2f} USDT, 加仓保证金={actual_margin:.2f} USDT")

                if actual_margin > available_funds:
                    logger.error(f"{symbol} 加仓保证金 {actual_margin:.2f} USDT 超过可用余额 {available_funds:.2f} USDT，取消加仓")
                    logger.error(f"  总投资={total_investment:.2f} USDT, 加仓比例={add_percent}%")
                    logger.error(f"  加仓名义价值={actual_notional:.2f} USDT, 杠杆={leverage}x")
                    return False

                # 留10%余量，避免因价格波动导致保证金不足
                if actual_margin > available_funds * 0.9:
                    logger.warning(f"{symbol} 加仓保证金 {actual_margin:.2f} USDT 接近可用余额 {available_funds:.2f} USDT 的90%，可能存在风险")
                    logger.warning(f"  建议减少加仓比例或等待更多资金可用")

            except Exception as e:
                logger.warning(f"{symbol} 获取账户余额失败，跳过保证金检查: {str(e)}")
                # 继续执行，让API返回错误

            # 确定订单方向
            side = 'BUY' if direction == 'LONG' else 'SELL'

            # 再次确保参数精度（双重保险）
            quantity = self.client.adjust_quantity_precision(symbol, quantity)
            limit_price = self.client.adjust_price_precision(symbol, limit_price)

            logger.info(f"{symbol} 最终下单参数: 数量={quantity:.8f}, 价格={limit_price:.8f}, 方向={side}")

            # 保存加仓参数供后续使用
            add_margin = actual_margin
            add_notional = actual_notional
            pending_order['add_margin'] = add_margin
            pending_order['quantity'] = quantity

            # 发送限价单，带错误处理
            position_side = direction  # LONG or SHORT

            # 尝试下单，最多重试2次
            max_order_retries = 2
            order = None  # 初始化order变量，避免unbound错误
            
            for attempt in range(max_order_retries + 1):
                try:
                    # 修复 -4015错误：确保 newClientOrderId 长度 < 36 个字符
                    # 格式: symbol前6字符 + _A_ + 秒级时间戳 (总长度 < 20 字符)
                    order_params = {
                        'symbol': symbol,
                        'side': side,
                        'order_type': 'LIMIT',
                        'quantity': quantity,
                        'price': limit_price,
                        'timeInForce': 'GTC',
                        'newClientOrderId': f"{symbol[:6]}_A_{int(time.time())}"  # API合规性：添加idempotency
                    }

                    # 不发送positionSide参数 - 让币安API自动判断
                    # 这适用于ISOLATED + ONE-WAY模式
                    order = self.client.create_order(**order_params)
                    break  # 成功则跳出重试循环

                except BinanceAPIException as e:
                    error_code = e.code if hasattr(e, 'code') else None

                    if error_code == -2027 and attempt < max_order_retries:  # Exceeded max position
                        # 对于加仓，减少数量重试
                        quantity = quantity * 0.8  # 减少20%
                        quantity = self.client.adjust_quantity_precision(symbol, quantity)
                        logger.warning(f"{symbol} 加仓持仓量超限，减少数量到{quantity}重试")
                        continue

                    elif error_code == -4005 and attempt < max_order_retries:  # Quantity too large
                        # 减少数量重试
                        quantity = quantity * 0.8  # 减少20%
                        quantity = self.client.adjust_quantity_precision(symbol, quantity)
                        logger.warning(f"{symbol} 加仓数量超限，减少数量到{quantity}重试")
                        continue

                    elif error_code == -4164 and attempt < max_order_retries:  # Notional value too small
                        # 名义价值太小，调整到最小要求
                        logger.warning(f"{symbol} 订单名义价值不足，调整到最小值重试")

                        # 获取最小名义价值要求
                        min_notional = 0.0
                        symbol_info = self.client.get_symbol_info(symbol)
                        if symbol_info:
                            for f in symbol_info.get('filters', []):
                                if f.get('filterType') == 'NOTIONAL':
                                    min_notional = float(f.get('minNotional', 0.0))
                                    break

                        # 如果API返回minNotional为0，使用保守值（不使用硬编码）
                        if min_notional == 0.0:
                            logger.warning(f"{symbol} 未获取到minNotional，尝试将数量增加10%")
                            quantity = quantity * 1.1
                            quantity = self.client.adjust_quantity_precision(symbol, quantity)
                        else:
                            # 使用最小名义价值重新计算数量
                            quantity = min_notional / limit_price
                            quantity = self.client.adjust_quantity_precision(symbol, quantity)

                        # 重新计算保证金
                        add_margin = min_notional / leverage

                        logger.info(f"{symbol} 调整到最小名义价值: 数量={quantity:.8f}, 保证金={add_margin:.2f} USDT")

                        continue

                    elif error_code == -2019 and attempt < max_order_retries:  # Margin is insufficient
                        # 保证金不足，减少加仓比例重试
                        logger.warning(f"{symbol} 保证金不足，减少加仓数量重试 (尝试 {attempt + 1}/{max_order_retries + 1})")
                        logger.warning(f"  原数量={quantity:.8f}, 原保证金={actual_margin:.2f} USDT")

                        # 减少30%数量（比20%更激进，因为保证金不足通常是严重问题）
                        quantity = quantity * 0.7
                        quantity = self.client.adjust_quantity_precision(symbol, quantity)

                        # 重新计算保证金和名义价值
                        actual_notional = quantity * limit_price
                        actual_margin = actual_notional / leverage

                        logger.info(f"{symbol} 调整后: 数量={quantity:.8f}, 保证金={actual_margin:.2f} USDT, 名义价值={actual_notional:.2f} USDT")
                        continue

                    else:
                        # 其他错误，直接失败
                        logger.error(f"{symbol} 加仓订单创建失败 [API {error_code}]: {e.message}")
                        return False

                except Exception as e:
                    logger.error(f"{symbol} 加仓订单创建失败: {str(e)}")
                    return False

            # 检查订单是否成功创建
            if not order or 'orderId' not in order:
                logger.error(f"{symbol} 加仓订单创建失败，响应: {order}")
                return False

            order_id = order['orderId']
            logger.info(f"{symbol} 加仓限价单已提交，订单ID: {order_id}, 价格: {limit_price:.6f}")

            # 记录订单ID到pending_order
            pending_order['order_id'] = order_id
            pending_order['limit_price'] = limit_price
            pending_order['status'] = 'submitted'

            logger.info(f"{symbol} 加仓计划执行成功，等待成交")

            return True

        except Exception as e:
            logger.error(f"{symbol} 加仓失败: {str(e)}")
            return False

    # ==================== 开仓逻辑 ====================

    def open_position(self, alert_data: Dict[str, Any]) -> bool:
        """
        根据警报数据开仓

        Args:
            alert_data: 警报数据

        Returns:
            bool: 是否成功开仓
        """
        symbol = alert_data['symbol']
        direction = alert_data['direction']  # LONG 或 SHORT（基于MONITOR_INTERVAL周期，默认3分钟）
        hour_direction = alert_data.get('hour_direction')  # 2小时周期的方向

        logger.info(f"=== 开始开仓流程 === 币种: {symbol}, 方向: {direction}")

        try:
            # 0. 验证警报方向与2小时趋势方向是否一致
            if hour_direction and direction != hour_direction:
                logger.warning(f"{symbol} 警报方向({direction})与2小时趋势方向({hour_direction})不一致，跳过开仓")
                logger.warning(f"  当前监控周期({settings.MONITOR_INTERVAL}分钟)方向: {direction}")
                logger.warning(f"  2小时周期方向: {hour_direction}")
                logger.warning(f"  短期波动与长期趋势相反，避免逆势交易")
                return False
            elif hour_direction:
                logger.info(f"{symbol} 方向验证通过: {direction} (2小时趋势: {hour_direction})")
            else:
                logger.warning(f"{symbol} 无法获取2小时趋势数据，仅使用警报方向")

            # 1. 检查是否已有该币种持仓
            if symbol in self.positions:
                logger.warning(f"{symbol} 已有持仓，跳过开仓")
                return False

            # 2. 检查持仓数量限制
            if len(self.positions) >= settings.MAX_POSITIONS:
                logger.warning(f"持仓数量已达上限 ({settings.MAX_POSITIONS})，当前持仓: {len(self.positions)}/{settings.MAX_POSITIONS}")
                logger.warning(f"当前持仓列表: {list(self.positions.keys())}")
                return False

            # 3. 检查币安API上是否已有持仓（防止程序重启后重复开仓）
            existing_positions = self.position_manager.get_all_positions()
            has_existing = any(pos['symbol'] == symbol for pos in existing_positions)

            if has_existing:
                logger.warning(f"{symbol} 币安API上已存在持仓，跳过开仓（防止重复）")
                # 确保本地也记录了这个持仓
                if symbol not in self.positions:
                    logger.warning(f"{symbol} 在币安API上有持仓但本地未记录，尝试同步")
                    self._sync_existing_positions()
                return False

            # 4. 双重检查：确保持仓数量没有超过限制
            # 重新获取持仓数量，因为_sync_existing_positions可能添加了新的持仓
            current_position_count = len(self.positions)
            if current_position_count >= settings.MAX_POSITIONS:
                logger.warning(f"同步持仓后数量已达上限 ({current_position_count}/{settings.MAX_POSITIONS})，跳过开仓")
                return False

            # 4. 获取账户信息和当前价格
            account_info = self.client.get_account_balance()
            available_funds = account_info['available_balance']
            current_price = self.client.get_ticker_price(symbol)

            logger.info(f"{symbol} 账户资金: 总余额={account_info['total_balance']:.2f} USDT, "
                       f"可用余额={available_funds:.2f} USDT, 模式={self.client.mode}")

            # 5. 计算币种分配金额
            current_positions_count = len(self.positions)
            slots_available = settings.MAX_POSITIONS - current_positions_count

            if slots_available <= 0:
                logger.warning(f"没有可用持仓额度")
                return False

            symbol_amount = available_funds / slots_available

            # 6. 应用单币种最大投资金额限制
            symbol_amount = min(symbol_amount, settings.SINGLE_SYMBOL_MAX_INVESTMENT)

            logger.info(f"{symbol} 分配金额: {symbol_amount:.2f} USDT (可用余额={available_funds:.2f}, "
                       f"剩余额度={slots_available}, 最大投资={settings.SINGLE_SYMBOL_MAX_INVESTMENT:.2f})")

            # 7. 计算初始保证金
            initial_margin = symbol_amount * (settings.INITIAL_POSITION / 100)

            logger.info(f"{symbol} 初始保证金: {initial_margin:.2f} USDT (比例={settings.INITIAL_POSITION}%)")

            # 8. 计算安全的下单参数（考虑交易所所有限制）
            safe_params = self._calculate_safe_order_params(
                symbol=symbol,
                target_margin=initial_margin,
                target_leverage=settings.LEVERAGE,
                current_price=current_price
            )

            actual_leverage = safe_params['leverage']
            actual_margin = safe_params['margin']

            logger.info(f"{symbol} 安全参数计算完成: 杠杆={actual_leverage}x, 保证金={actual_margin:.2f} USDT")

            # 检查调整后的保证金是否超过可用资金
            if actual_margin > available_funds:
                logger.error(f"{symbol} 调整后保证金 {actual_margin:.2f} USDT 超过可用资金 {available_funds:.2f} USDT，跳过开仓")
                return False

            # 9.1. 检查调整后的保证金是否超过分配给该币种的最大投资金额（P0修复）
            if actual_margin > symbol_amount:
                logger.error(f"{symbol} 调整后保证金 {actual_margin:.2f} USDT 超过分配给该币种的最大投资金额 {symbol_amount:.2f} USDT，跳过开仓")
                logger.error(f"  分配金额={symbol_amount:.2f} USDT, 初始保证金={initial_margin:.2f} USDT")
                logger.error(f"  目标杠杆={settings.LEVERAGE}x, 实际杠杆={actual_leverage}x")
                return False

            # 10. 设置保证金模式和杠杆（带重试）
            self._set_margin_mode_and_leverage(symbol, actual_leverage, settings.MARGIN_MODE)
            
            # ===== 强制切换到ONE-WAY模式（修复-4061错误） =====
            try:
                logger.info("===== 强制切换到ONE-WAY模式开始 =====")
                
                # 第1次检查
                account = self.client.client.futures_account()
                position_side_mode = account.get('positionSide', 'ONE-WAY')
                logger.info(f"[第1次检查] 当前账户 positionSide 模式: {position_side_mode}")
                logger.info(f"[第1次检查] 完整账户响应: {account}")
                
                # 第2次检查（立即再次验证）
                account2 = self.client.client.futures_account()
                position_side_mode2 = account2.get('positionSide', 'ONE-WAY')
                logger.info(f"[第2次检查] 验证后账户 positionSide 模式: {position_side_mode2}")
                
                # 如果不是ONE-WAY，强制切换
                if position_side_mode != 'ONE-WAY' or position_side_mode2 != 'ONE-WAY':
                    logger.warning(f"账户模式不一致: {position_side_mode} vs {position_side_mode2}，强制切换到ONE-WAY")
                    self.client.client.futures_change_position_mode(dualSidePosition='false')
                    time.sleep(2)  # 等待切换生效
                    
                    # 第3次验证
                    account3 = self.client.client.futures_account()
                    position_side_mode3 = account3.get('positionSide', 'ONE-WAY')
                    logger.info(f"[第3次检查] 切换后账户 positionSide 模式: {position_side_mode3}")
                    
                    if position_side_mode3 != 'ONE-WAY':
                        logger.error(f"切换失败，账户仍为: {position_side_mode3}")
                    else:
                        logger.info("✓ 成功切换到ONE-WAY模式")
                else:
                    logger.info("✓ 账户已为ONE-WAY模式，无需切换")
                    
            except Exception as e:
                logger.error(f"强制切换positionSide模式失败: {str(e)}")
                # 不是致命错误，继续执行

            # 10. 使用安全参数中的数量（已经过限制检查）
            quantity = safe_params['quantity']
  
            # ✅ 使用统一的订单数量验证函数（确保满足所有币安限制）
            # 注意：这里先不指定order_type，稍后根据实际订单类型再验证
            # 因为现在还不知道会是MARKET还是LIMIT
            quantity, is_valid, adjustment_info = self._validate_order_quantity_unified(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                order_type='MARKET',  # 先用市价单验证（最宽松）
                ensure_min_notional=True,
                max_quantity_limit=None,
                description='开仓单'
            )
            
            if not is_valid:
                final_notional = adjustment_info.get('final_notional', 0)
                logger.error(f"{symbol} 开仓数量验证失败，取消开仓")
                logger.error(f"  最终数量: {adjustment_info.get('final_quantity')}")
                logger.error(f"  最终名义价值: {final_notional:.2f} USDT")

                # 如果是因为minNotional不满足，给出具体建议
                # 使用从API获取的min_notional，而不是硬编码5.0
                # 从adjustment_info获取实际使用的min_notional（0表示无限制）
                actual_min_notional = adjustment_info.get('actual_min_notional', 0)
                if actual_min_notional > 0 and final_notional < actual_min_notional:
                    final_qty = adjustment_info.get('final_quantity')
                    if final_qty:
                        actual_notional = final_qty * current_price
                    else:
                        actual_notional = final_notional

                    logger.error(f"  【问题】名义价值 {actual_notional:.2f} USDT 小于币安要求该币种的最低要求 {actual_min_notional} USDT")
                    logger.error(f"  【原因】该币种价格: {current_price:.6f} USDT，保证金: {actual_margin:.2f} USDT (INITIAL_POSITION={settings.INITIAL_POSITION}%)")
                    logger.error(f"  【测试网余额可用】{available_funds:.2f} USDT")
                    logger.error(f"  【建议1】检查环境变量 INITIAL_POSITION (当前={settings.INITIAL_POSITION}%)，建议增加到 100% 或更高")
                    logger.error(f"  【建议2】跳过价格过低的币种（当前价格 {current_price:.6f} USDT）")
                    logger.error(f"  【建议3】增加测试网账户余额到至少 {(actual_min_notional * 0.5):.0f} USDT ({actual_min_notional} USDT/2)")

                if '调整项' in adjustment_info:
                    logger.error(f"  调整详情: {adjustment_info['调整项']}")
                return False
            
            # 如果数量被调整了，更新日志
            if adjustment_info.get('adjusted', False):
                logger.info(f"{symbol} 订单数量已调整:")
                logger.info(f"  原始数量: {adjustment_info['original_quantity']:.8f}")
                logger.info(f"  最终数量: {adjustment_info['final_quantity']:.8f}")
                logger.info(f" 名义价值: {adjustment_info['final_notional']:.2f} USDT")
                if '调整项' in adjustment_info:
                    logger.debug(f" 调整详情: {adjustment_info['调整项']}")
            
            # 获取精度信息用于日志
            precision_info = self.client.get_tick_size_and_precision(symbol)
            qty_precision = precision_info['qty_precision']
            
            logger.info(f"{symbol} 订单数量: {quantity} (精度={qty_precision})")

            # 11. 智能选择订单类型
            side = 'BUY' if direction == 'LONG' else 'SELL'

            # 根据币种特征选择订单类型
            # 对于小币种或流动性不足的币种，使用限价单
            order_type = self._choose_order_type(symbol, quantity, current_price)

            # ✅ 生产工具 - 重复订单检测（在订单提交前检查）
            if not self.duplicate_detector.can_submit_order(symbol, 'open_position'):
                logger.warning(f"{symbol} 重复订单检测：60秒冷却期内禁止重复开仓")
                return False
            logger.info(f"{symbol} 重复订单检测通过，允许开仓")

            # ✅ 生产工具 - 检查持久化中是否存在未完成的相同操作
            recent_orders = self.order_persistence.get_orders_by_symbol(symbol)
            # 过滤出最近5分钟内未完成的订单
            import time
            cutoff_time = time.time() - 300  # 5分钟前
            pending_orders = [
                o for o in recent_orders
                if o.get('status') in ['PENDING', 'SUBMITTED', 'PARTIAL']
                and o.get('created_at') > cutoff_time
            ]
            if pending_orders:
                logger.warning(f"{symbol} 持久化中存在 {len(pending_orders)} 个近期未完成订单，跳过开仓")
                for order in pending_orders:
                    logger.warning(f"  - OrderId: {order['order_id']}, Status: {order['status']}, Created: {order.get('created_at')}")
                return False

            # 尝试下单，最多重试2次（处理交易所限制）
            max_order_retries = 2
            for attempt in range(max_order_retries + 1):
                try:
                    # 构建订单参数（必须使用create_order的参数名）
                    # binance_client.create_order 签名：create_order(symbol, side, order_type, quantity, price=None, stop_price=None, **kwargs)
                    # 修复 -4015错误：确保 newClientOrderId 长度 < 36 个字符
                    # 使用秒级时间戳，格式: symbol前6字符 + _O_ + 秒级时间戳 (总长度 < 20 字符)
                    order_params = {
                        'symbol': symbol,
                        'side': side,
                        'order_type': order_type,  # 使用正确的参数名 order_type
                        'quantity': quantity,
                        'newClientOrderId': f"{symbol[:6]}_O_{int(time.time())}"  # API合规性：添加idempotency
                    }

                    # 为限价单添加价格和timeInForce
                    if order_type == 'LIMIT':
                        # 为限价单计算合理的价格
                        limit_price = self._calculate_limit_price(symbol, side, current_price)
                        order_params['price'] = limit_price
                        order_params['timeInForce'] = 'GTC'
                    # MARKET类型不需要额外参数

                    # 打印订单参数用于调试
                    logger.info(f"{symbol} 下单参数: {order_params}")

                    # ===== 关键修复：正确设置 positionSide 参数 =====
                    # ===== 关键修复:完全移除positionSide参数 =====
                    # 币安期货API在单向模式下不需要positionSide参数
                    # 系统会自动根据方向(BUY/SELL)判断持仓方向
                    # 只在HEDGING模式下才需要positionSide,但我们统一使用ISOLATED+单向模式
                    
                    order = self.client.create_order(**order_params)
                    break  # 成功则跳出重试循环

                except BinanceAPIException as e:
                    error_code = e.code if hasattr(e, 'code') else None

                    if error_code == -2027 and attempt < max_order_retries:  # Exceeded max position
                        # 降低杠杆重试
                        if actual_leverage > 5:  # 最低降到5倍杠杆
                            new_leverage = max(actual_leverage - 5, 5)
                            logger.warning(f"{symbol} 持仓量超限，降低杠杆从{actual_leverage}x到{new_leverage}x重试")

                            # 重新设置杠杆
                            self._set_margin_mode_and_leverage(symbol, new_leverage, settings.MARGIN_MODE)

                            # 重新计算数量
                            notional_value = actual_margin * new_leverage
                            quantity = notional_value / current_price
                            quantity = round(quantity, qty_precision)
                            quantity = self.client.adjust_quantity_precision(symbol, quantity)

                            actual_leverage = new_leverage
                            continue
                        else:
                            logger.error(f"{symbol} 即使降低杠杆也无法开仓，跳过")
                            return False

                    elif error_code == -4005 and attempt < max_order_retries:  # Quantity too large
                        # 减少数量重试
                        quantity = quantity * 0.8  # 减少20%
                        quantity = round(quantity, qty_precision)
                        quantity = self.client.adjust_quantity_precision(symbol, quantity)
                        logger.warning(f"{symbol} 数量超限，减少数量到{quantity}重试")
                        continue

                    else:
                        # 其他错误，直接失败
                        logger.error(f"{symbol} 订单创建失败 [API {error_code}]: {e.message}")
                        return False

                except Exception as e:
                    logger.error(f"{symbol} 订单创建失败: {str(e)}")
                    return False

            # 12. 验证订单
            if not order or 'orderId' not in order:
                logger.error(f"{symbol} 订单创建失败，响应: {order}")
                return False

            order_id = order['orderId']
            logger.info(f"{symbol} 市价单已提交，订单ID: {order_id}, 状态: {order.get('status', 'UNKNOWN')}, "
                       f"成交数量: {order.get('executedQty', 0)}, 平均价格: {order.get('avgPrice', 0)}")

            # ✅ 生产工具 - 保存订单到持久化
            try:
                self.order_persistence.save_order({
                    'order_id': str(order_id),
                    'order_type': order_type,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': order_params.get('price'),
                    'executed_price': float(order.get('avgPrice', 0)),
                    'executed_quantity': float(order.get('executedQty', 0)),
                    'status': 'SUBMITTED',
                    'reason': '开仓',
                    'extra_info': json.dumps({
                        'direction': direction,
                        'leverage': actual_leverage,
                        'margin': actual_margin
                    })
                })
                logger.info(f"{symbol} 订单已保存到持久化数据库")
            except Exception as e:
                logger.warning(f"{symbol} 订单持久化保存失败: {str(e)} (不影响交易)")


            # 市价单可能返回'NEW'或'FILLED'状态，需要等待成交
            # 对于市价单，我们允许'NEW'状态，稍后会成交
            order_status = order.get('status', '')
            executed_qty = float(order.get('executedQty', 0))

            # 如果状态是'FILLED'或者有部分成交数量，视为成功
            if order_status == 'FILLED':
                logger.info(f"{symbol} 订单已完全成交")
                entry_price = float(order.get('avgPrice', order.get('price', current_price)))
                final_qty = float(order.get('executedQty', quantity))
                # ✅ 生产工具 - 更新订单持久化状态为成交
                try:
                    self.order_persistence.update_order_status(str(order_id), OrderStatus.FILLED)
                except Exception as e:
                    logger.warning(f"{symbol} 订单状态更新失败: {str(e)}")
            elif order_status == 'NEW' and executed_qty > 0:
                logger.info(f"{symbol} 订单部分成交，等待完全成交")
                # 等待订单完全成交，增加超时时间到60秒
                is_filled, filled_order = self.client.wait_order_filled(symbol, order_id, timeout=60)
                if is_filled:
                    entry_price = float(filled_order.get('avgPrice', current_price))
                    final_qty = float(filled_order.get('executedQty', quantity))
                    # ✅ 生产工具 - 更新订单持久化状态为成交
                    try:
                        self.order_persistence.update_order_status(str(order_id), OrderStatus.FILLED)
                    except Exception as e:
                        logger.warning(f"{symbol} 订单状态更新失败: {str(e)}")
                else:
                    logger.warning(f"{symbol} 订单部分成交但超时未完全成交，取消剩余部分")
                    # 尝试取消未成交部分
                    try:
                        self.client.cancel_order(symbol, order_id)
                        logger.info(f"{symbol} 已取消未成交的订单部分")
                        # ✅ 生产工具 - 更新订单持久化状态
                        try:
                            self.order_persistence.update_order_status(str(order_id), OrderStatus.CANCELLED)
                        except Exception as e:
                            logger.warning(f"{symbol} 订单状态更新失败: {str(e)}")
                    except Exception as e:
                        logger.warning(f"{symbol} 取消订单失败: {str(e)}")
                    return False
            else:
                # 等待订单成交，增加超时时间到60秒
                is_filled, filled_order = self.client.wait_order_filled(symbol, order_id, timeout=60)
                if is_filled:
                    entry_price = float(filled_order.get('avgPrice', current_price))
                    final_qty = float(filled_order.get('executedQty', quantity))
                    # ✅ 生产工具 - 更新订单持久化状态为成交
                    try:
                        self.order_persistence.update_order_status(str(order_id), OrderStatus.FILLED)
                    except Exception as e:
                        logger.warning(f"{symbol} 订单状态更新失败: {str(e)}")
                else:
                    logger.warning(f"{symbol} 订单超时未成交，取消订单")
                    # 尝试取消未成交的订单
                    try:
                        self.client.cancel_order(symbol, order_id)
                        logger.info(f"{symbol} 已取消未成交的订单")
                        # ✅ 生产工具 - 更新订单持久化状态
                        try:
                            self.order_persistence.update_order_status(str(order_id), OrderStatus.CANCELLED)
                        except Exception as e:
                            logger.warning(f"{symbol} 订单状态更新失败: {str(e)}")
                    except Exception as e:
                        logger.warning(f"{symbol} 取消订单失败: {str(e)}")
                    return False

            logger.info(f"{symbol} 开仓成功: 价格={entry_price:.4f}, 数量={final_qty}")

            # 验证entry_price和final_qty不为0（某些情况下API可能返回0值）
            if entry_price <= 0:
                logger.warning(f"{symbol} entry_price为0或负数，使用current_price={current_price}作为备用值")
                entry_price = current_price
            
            if final_qty <= 0:
                logger.warning(f"{symbol} final_qty为0或负数，使用原始quantity={quantity}作为备用值")
                final_qty = quantity

            # 13. 创建持仓数据和建仓计划
            position_info = {
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': entry_price,
                'direction': direction,
                'total_quantity': final_qty,
                'total_investment': actual_margin,
                'initial_margin': actual_margin,
                'completed_investment': actual_margin,  # 建仓完成时的总投资
                'allocated_funds': symbol_amount,  # 该币种分配的资金
                'profit': 0.0,
                'profit_pct': 0.0,
                'max_profit_pct': 0.0,
                'leverage': actual_leverage,
                'added_levels': [],  # 已执行的加仓级别索引
                'pending_orders': [],  # 待执行的加仓计划
                        'is_closing': False,
                        'last_action_time': int(time.time()),
                        'status': 'active',
                        'take_profit_levels': {},
                        'stop_loss_levels': {},
                        'position_complete': False  # 标记建仓是否完成
                    }

            # 创建6级建仓计划
            self._create_position_building_plan(position_info, entry_price, direction)

            # 添加到持仓列表
            self.positions[symbol] = position_info

            # ✅ 不在建仓计划创建时设置止损单，而是在建仓真正完成时设置（_finalize_position_building中）
            # 这里只创建止盈止损的监控逻辑，实际的止损限价单在position_complete=True时创建

            # 14. 发送交易通知
            # 开仓时：显示当前保证金占分配资金的比例（INITIAL_POSITION）
            # 预计总仓位 = 分配资金
            self._send_trade_message(
                symbol=symbol,
                action='OPEN',
                direction=direction,
                current_investment=actual_margin,
                total_planned_investment=symbol_amount,
                quantity=final_qty,
                amount=actual_margin,
                price=entry_price,
                current_price=entry_price,
                leverage=actual_leverage,
                profit=0.0,
                profit_pct=0.0
            )

            # ✅ 记录开仓到交易表格
            try:
                ratio_str = f"{actual_margin/symbol_amount*100:.0f}/{100}"
                self.trade_recorder.record_trade(
                    symbol=symbol,
                    action='OPEN',
                    direction=direction,
                    price=entry_price,
                    quantity=final_qty,
                    amount=actual_margin,
                    leverage=actual_leverage,
                    ratio=ratio_str
                )
            except Exception as e:
                logger.warning(f"{symbol} 记录开仓失败: {str(e)}")

            logger.info(f"{symbol} 开仓仓位比例: {(actual_margin/symbol_amount)*100:.0f}% (保证金: {actual_margin:.2f} USDT, 分配资金: {symbol_amount:.2f} USDT)")

            logger.info(f"=== 开仓流程完成 === 币种: {symbol}")
            return True

        except Exception as e:
            logger.error(f"{symbol} 开仓失败: {str(e)}", exc_info=True)
            return False

    def sync_positions_now(self):
        """
        立即同步持仓状态 - 供外部调用
        """
        logger.info("执行手动持仓同步...")
        self._sync_position_states_from_api()
        logger.info(f"同步完成，当前持仓数量: {len(self.positions)}")
        return len(self.positions)

    def _create_position_building_plan(self, position_info: Dict, entry_price: float, direction: str):
        """
        创建6级建仓计划（限价单价格动态计算）

        Args:
            position_info: 持仓信息
            entry_price: 开仓价格
            direction: 方向
        """
        try:
            symbol = position_info['symbol']
            total_investment = position_info['total_investment']
            leverage = position_info['leverage']

            # 验证entry_price是否有效
            if entry_price <= 0:
                logger.error(f"{symbol} entry_price无效({entry_price})，无法创建建仓计划")
                position_info['pending_orders'] = []
                return

            # 定义建仓级别
            # 格式: (触发率%, 加仓比例%, 原因)
            levels = [
                # 亏损加仓（限价单）
                (settings.LOSS_STEP1, settings.LOSS_ADD1, "亏损加仓1"),
                (settings.LOSS_STEP2, settings.LOSS_ADD2, "亏损加仓2"),
                (settings.LOSS_STEP3, settings.LOSS_ADD3, "亏损加仓3"),
                # 盈利加仓（限价单）
                (settings.PROFIT_STEP1, settings.PROFIT_ADD1, "盈利加仓1"),
                (settings.PROFIT_STEP2, settings.PROFIT_ADD2, "盈利加仓2"),
                (settings.PROFIT_STEP3, settings.PROFIT_ADD3, "盈利加仓3")
            ]

            pending_orders = []

            for idx, (trigger_rate, add_percent, reason) in enumerate(levels):
                # 计算加仓金额
                add_margin = total_investment * (add_percent / 100)

                # 计算加仓数量
                # 限价单价格将在触发时动态计算，这里使用entry_price作为占位
                add_notional = add_margin * leverage
                quantity = add_notional / entry_price

                # 记录加仓计划（限价单价格在触发时动态计算）
                pending_order = {
                    'index': idx,
                    'trigger_rate': trigger_rate,  # 触发盈利率（%）
                    'add_percent': add_percent,   # 加仓比例（%）
                    'add_margin': add_margin,     # 加仓保证金（USDT）
                    'limit_price': entry_price,    # 占位价格，实际触发时动态计算
                    'quantity': quantity,          # 数量
                    'order_id': None,             # 订单ID
                    'status': 'pending',          # 状态
                    'reason': reason
                }

                pending_orders.append(pending_order)
                logger.info(f"{symbol} 建仓计划{idx}: 触发盈利率={trigger_rate}%, "
                           f"加仓比例={add_percent}%, 数量={quantity:.6f}")

            # 将建仓计划分配给持仓信息（重要：这样才能在监控时执行加仓）
            position_info['pending_orders'] = pending_orders
            logger.info(f"{symbol} 建仓计划创建完成，共{len(pending_orders)}级已添加到持仓信息")

        except Exception as e:
            logger.error(f"{symbol} 创建建仓计划失败: {str(e)}")
            position_info['pending_orders'] = []

    def send_alert_message(self, message: str):
        """
        发送警报消息到Telegram

        Args:
            message: 消息内容
        """
        if not self.telegram_bot:
            return

        try:
            # 构建警报数据（按照telegram_bot.py的send_alert接口）
            from datetime import datetime
            alert_data = {
                'symbol': 'SYSTEM',
                'current_price': 0,
                'price_change': 0,
                'volume_usdt': 0,
                'volume_ratio': 0,
                'open_interest': 'N/A',
                'funding_rate': 'N/A',
                'price_change_1h': 0,
                'price_change_4h': 0,
                'alert_count': 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'direction': 'LONG',
                'hour_direction': 'LONG',
                'oi_change_pct': 0
            }

            # 发送到警报bot
            self.telegram_bot.send_alert('SYSTEM', alert_data)
            logger.info("警报消息已发送")
        except Exception as e:
            logger.error(f"发送警报消息失败: {str(e)}")

    def _send_trade_message(self, symbol: str, action: str, direction: str,
                          current_investment: float, total_planned_investment: float,
                          quantity: float, amount: float, price: float, current_price: float,
                          leverage: int, profit: float, profit_pct: float,
                          initial_investment: Optional[float] = None):
        """
        发送交易消息到Telegram
        
        Args:
            symbol: 币种
            action: 动作（OPEN/ADD/CLOSE）
            direction: 方向
            current_investment: 当前总投资（USDT）
            total_planned_investment: 预计总投资（USDT）
            quantity: 数量
            amount: 金额（USDT）
            price: 成交价格
            current_price: 当前价格
            leverage: 杠杆
            profit: 盈亏（USDT）
            profit_pct: 盈亏率（%）
            initial_investment: 初始总投资（USDT）- 用于计算平仓/止盈止损后的比例
        """
        if not self.telegram_bot:
            return

        try:
            # ✅ 用户要求的显示格式
            # 建仓阶段：【仓位】：150USDT（建仓至40%）
            # 止盈止损阶段：【仓位】：75USDT（平仓至50%）
            
            if action in ['OPEN', 'ADD']:
                # 建仓阶段：当前 / 预计完成
                denominator = total_planned_investment
                percentage = (current_investment / denominator * 100) if denominator > 0 else 0
                position_text = f"【仓位】：{current_investment:.0f}USDT（建仓至{percentage:.0f}%）"
                action_text = f"做多建仓" if direction == 'LONG' else "做空建仓"
                if action == 'ADD':
                    action_text = "加仓"
            else:
                # 平仓/止盈止损阶段：后续 / 初始总建仓
                denominator = initial_investment if initial_investment else current_investment
                percentage = (current_investment / denominator * 100) if denominator > 0 else 0
                position_text = f"【仓位】：{current_investment:.0f}USDT（平仓至{percentage:.0f}%）"
                action_text = '平仓'

            # 盈亏符号
            profit_symbol = "+" if profit >= 0 else ""
            profit_pct_symbol = "+" if profit_pct >= 0 else ""

            # 构建交易数据（按照telegram_bot.py的send_trade_message接口）
            trade_data = {
                'leverage': leverage,
                'side': direction,
                'status': action_text,
                'position_ratio': (current_investment / denominator * 100) if denominator > 0 else 0,
                'avg_entry_price': price,
                'current_price': current_price,
                'position_usdt': amount,
                'margin': amount,
                'pnl': profit,
                'pnl_percent': profit_pct
            }

            # 发送到交易bot
            self.telegram_bot.send_trade_message(symbol, trade_data)
            logger.info(f"{symbol} 交易消息已发送: {action_text}")

            # 记录到交易记录器
            try:
                ratio_str = f"{current_investment:.0f}/{denominator:.0f}" if denominator > 0 else "0/0"
                self.trade_recorder.record_trade(
                    symbol=symbol,
                    action=action,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    amount=amount,
                    leverage=leverage,
                    ratio=ratio_str
                )
            except Exception as e:
                logger.warning(f"{symbol} 记录交易失败: {str(e)}")

        except Exception as e:
            logger.error(f"{symbol} 发送交易消息失败: {str(e)}")

    # ==================== 启动/停止控制 ====================

    def start(self):
        """启动交易模块"""
        if self._running:
            logger.warning("交易模块已在运行")
            return

        self._running = True
        logger.info("交易模块启动中...")

        # ✅ 检测并取消所有挂单（防止程序重启后留有旧订单）
        self._check_and_cancel_all_orders()

        import threading
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("交易模块监控线程已启动")

    def _check_and_cancel_all_orders(self):
        """
        检测并取消所有挂单

        在启动交易模块时调用，防止程序重启后留有旧订单导致与本地持仓状态不一致
        """
        try:
            # 获取所有未成交订单
            all_orders = self.client.get_open_orders()

            if not all_orders:
                logger.info("启动时检查完毕：没有检测到挂单")
                return

            # 按symbol分组统计
            orders_by_symbol = {}
            for order in all_orders:
                symbol = order.get('symbol', '')
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)

            total_count = len(all_orders)
            logger.warning(f"启动时检测到 {total_count} 个挂单！")
            
            # 打印每个币种的挂单情况
            for symbol, orders in orders_by_symbol.items():
                logger.warning(f"  {symbol}: {len(orders)} 个挂单")
                for order in orders:
                    order_type = order.get('type', 'UNKNOWN')
                    side = order.get('side', 'UNKNOWN')
                    order_id = order.get('orderId', 'N/A')
                    logger.warning(f"    订单ID={order_id}, 类型={order_type}, 方向={side}")

            # 取消所有挂单
            logger.info("开始取消所有挂单...")
            cancelled_symbols = set()
            for symbol in orders_by_symbol.keys():
                try:
                    result = self.client.cancel_all_orders(symbol)
                    if result:
                        cancelled_symbols.add(symbol)
                        logger.info(f"  ✓ 已取消 {symbol} 的所有挂单")
                    else:
                        logger.warning(f"  ✗ 取消 {symbol} 挂单失败")
                except Exception as e:
                    logger.warning(f"  ✗ 取消 {symbol} 挂单异常: {str(e)}")

            logger.info(f"挂单清理完成: 成功取消 {len(cancelled_symbols)} 个币种的挂单")

        except Exception as e:
            logger.error(f"启动时检查挂单失败: {str(e)}", exc_info=True)

    def stop(self):
        """停止交易模块"""
        if not self._running:
            return

        logger.info("停止交易模块...")
        self._running = False

        # 平仓所有持仓（根据环境变量配置）
        try:
            if settings.EMERGENCY_CLOSE_ON_PAUSE:
                logger.info("配置为关闭时紧急平仓所有持仓")
                self.close_all_positions("系统停止")
            else:
                logger.info("配置为关闭时不平仓")
        except Exception as e:
            logger.error(f"平仓所有持仓失败: {str(e)}")

        # 等待监控线程结束
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
            logger.info("交易模块监控线程已停止")

        # 导出交易记录到CSV文件
        try:
            csv_file = self.trade_recorder.export_csv()
            if csv_file:
                logger.info(f"交易记录已导出到: {csv_file}")
            else:
                logger.info("没有交易记录需要导出")
        except Exception as e:
            logger.error(f"导出交易记录失败: {str(e)}")

        logger.info("交易模块已停止")

    def close_all_positions(self, reason: str):
        """平仓所有持仓"""
        try:
            if not self.positions:
                return

            logger.info(f"开始平仓所有持仓，原因: {reason}")

            for symbol, position_info in list(self.positions.items()):
                try:
                    if position_info.get('status') != 'closed':
                        self._execute_close_position(symbol, position_info, 100, reason)
                except Exception as e:
                    logger.error(f"{symbol} 平仓失败: {str(e)}")

            logger.info("所有持仓平仓完成")
        except Exception as e:
            logger.error(f"平仓所有持仓失败: {str(e)}")

    def _setup_trailing_take_profit(self, symbol: str, position_info: Dict, current_price: float, pnl_rate: float):
        """
        设置跟踪止盈订单（基于币安 TRAILING_STOP_MARKET）

        根据环境变量设置的止盈阈值和回撤参数，动态设置跟踪止损订单：
        - 利润达到 HIGH_PROFIT_THRESHOLD (8%) 时，设置回撤 2% 的跟踪止损
        - 利润达到更高阈值时，更新跟踪止损回撤参数
        - 如果利润下降到低利润区间，使用低利润参数

        Args:
            symbol: 币种
            position_info: 持仓信息
            current_price: 当前价格
            pnl_rate: 当前盈亏率（%）
        """
        try:
            # 只在建仓完成后启用跟踪止盈
            if not position_info.get('position_complete', False):
                return

            direction = position_info['direction']
            total_quantity = position_info['total_quantity']
            entry_price = position_info.get('entry_price', 0)

            if total_quantity <= 0:
                return

            # 根据利润水平确定跟踪止盈参数
            callback_rate = 0.0  # 回撤百分比（0表示不设置）
            reason = ""

            # 高利润区间：使用高利润参数
            if pnl_rate >= settings.HIGH_PROFIT_THRESHOLD:
                # 判断是否在两个回撤阈值之间
                max_profit_pct = position_info['max_profit_pct']
                profit_drawback = max_profit_pct - pnl_rate

                if profit_drawback < settings.HIGH_PROFIT_DRAWBACK1:
                    # 利润刚进入高利润区间，使用第一个回撤阈值
                    callback_rate = settings.HIGH_PROFIT_DRAWBACK1
                    reason = f"高利润跟踪止盈（阈值{settings.HIGH_PROFIT_THRESHOLD}%，回撤{callback_rate}%）"
                elif profit_drawback >= settings.HIGH_PROFIT_DRAWBACK1 and profit_drawback < settings.HIGH_PROFIT_DRAWBACK2:
                    # 已超过第一个回撤，使用第二个回撤阈值（更宽松）
                    callback_rate = settings.HIGH_PROFIT_DRAWBACK2
                    reason = f"高利润跟踪止盈（第二层级回撤{callback_rate}%）"

            # 低利润区间：使用低利润参数
            elif pnl_rate >= settings.LOW_PROFIT_THRESHOLD and pnl_rate < settings.HIGH_PROFIT_THRESHOLD:
                callback_rate = settings.LOW_PROFIT_DRAWBACK1
                reason = f"低利润跟踪止盈（阈值{settings.LOW_PROFIT_THRESHOLD}%，回撤{callback_rate}%）"

            # 利润不足，不需要设置跟踪止盈
            if callback_rate == 0:
                return

            # 检查是否已经有跟踪止盈订单
            existing_order_id = position_info.get('trailing_take_profit_order_id')
            if existing_order_id:
                # 已经有跟踪止盈订单，检查是否需要更新
                # 如果回调率变化不大（小于0.5%），不更新（避免频繁修改订单）
                existing_callback = position_info.get('trailing_take_profit_callback', 0)
                if abs(callback_rate - existing_callback) < 0.5:
                    logger.debug(f"{symbol} 跟踪止盈回调率变化不大（{existing_callback}% -> {callback_rate}%），不更新订单")
                    return

                # 需要更新：先取消旧订单
                try:
                    self.client.cancel_order(symbol, existing_order_id)
                    logger.info(f"{symbol} 取消旧的跟踪止盈订单: {existing_order_id}")
                except Exception as e:
                    logger.warning(f"{symbol} 取消跟踪止盈订单失败: {str(e)}")
                    # 如果取消失败，不创建新订单（避免重复）
                    return

            # 平仓方向与开仓相反
            side = 'SELL' if direction == 'LONG' else 'BUY'

            # 计算激活价格（基于当前价格）
            activation_price = current_price

            # 创建 TRAILING_STOP_MARKET 订单
            # 修复 -4015错误：确保 newClientOrderId 长度 < 36 个字符
            # 格式: symbol前6字符 + _tp_ + 秒级时间戳 (总长度 < 20 字符)
            order_params = {
                'symbol': symbol,
                'side': side,
                'order_type': 'TRAILING_STOP_MARKET',
                'quantity': total_quantity,
                'activationPrice': activation_price,
                'callbackRate': callback_rate,  # 基点，1 = 0.1%, 10 = 1%
                'reduceOnly': 'true',
                'workingType': 'MARK_PRICE',
                'newClientOrderId': f"{symbol[:6]}_tp_{int(time.time())}"
            }

            order = self.client.create_order(**order_params)

            if order and 'orderId' in order:
                logger.info(f"{symbol} 跟踪止盈订单已创建: ID={order['orderId']}, "
                           f"激活价={activation_price:.6f}, 回调率={callback_rate}%, 数量={total_quantity:.8f}, {reason}")

                # 记录跟踪止盈订单信息
                position_info['trailing_take_profit_order_id'] = order['orderId']
                position_info['trailing_take_profit_callback'] = callback_rate
                position_info['trailing_take_profit_time'] = time.time()
            else:
                logger.error(f"{symbol} 创建跟踪止盈订单失败: {order}")

        except Exception as e:
            logger.error(f"{symbol} 设置跟踪止盈失败: {str(e)}", exc_info=True)

    def _cancel_trailing_take_profit(self, symbol: str, position_info: Dict):
        """
        取消跟踪止盈订单（当手动止盈或平仓时调用）

        Args:
            symbol: 币种
            position_info: 持仓信息
        """
        try:
            order_id = position_info.get('trailing_take_profit_order_id')
            if order_id:
                self.client.cancel_order(symbol, order_id)
                logger.info(f"{symbol} 取消跟踪止盈订单: {order_id}")
                position_info.pop('trailing_take_profit_order_id', None)
                position_info.pop('trailing_take_profit_callback', None)
        except Exception as e:
            logger.warning(f"{symbol} 取消跟踪止盈订单失败: {str(e)}")

    def _execute_take_profit(self, symbol: str, position_info: Dict, pnl_rate: float):
        """
        执行止盈逻辑

        Args:
            symbol: 币种
            position_info: 持仓信息
            pnl_rate: 当前盈亏率（%）
        """
        try:
            # 关键：只有建仓完成后才触发止盈
            # 建仓未完成时不应该触发止盈，避免在建仓过程中被止盈
            if not position_info.get('position_complete', False):
                logger.debug(f"{symbol} 建仓未完成，跳过止盈检查")
                return

            current_price = position_info.get('current_price', 0)

            # ⚠️ 币安期货已不再支持 TRAILING_STOP_MARKET（2025-12-09）
            # 移除跟踪止损订单逻辑，只保留逻辑检查
            # logger.info(f"{symbol} 跟踪止损已禁用（API不支持）")

            max_profit_pct = position_info['max_profit_pct']
            total_investment = position_info['total_investment']
            direction = position_info['direction']
            leverage = position_info['leverage']

            # 利润回撤计算：最高回报率 - 当前回报率（两者都已包含杠杆，无需再乘杠杆）
            # 修复前错误: profit_drawback = (max_profit_pct - pnl_rate) * leverage / 100
            # 修复后正确: profit_drawback = max_profit_pct - pnl_rate
            # 例如：最高12%，当前10% → 回撤是2%，而不是0.4%
            profit_drawback = max_profit_pct - pnl_rate

            logger.debug(f"{symbol} 利润回撤: {profit_drawback:.2f}% (最高={max_profit_pct:.2f}%, 当前={pnl_rate:.2f}%)")

            # 修复: 回撤计算不应该再乘杠杆，因为max_profit_pct和pnl_rate已经包含杠杆
            # 原错误: profit_drawback = (max_profit_pct - pnl_rate) * leverage / 100
            # 修正后: profit_drawback = max_profit_pct - pnl_rate
            # 例如：最高12%，当前10% → 回撤应该是2%，而不是 (12-10)*20/100=0.4%

            close_percentage = 0
            reason = ""

            # 高盈利止盈 (优先于跟踪止盈执行止盈平仓)
            if pnl_rate >= settings.HIGH_PROFIT_THRESHOLD:
                if profit_drawback >= settings.HIGH_PROFIT_DRAWBACK1:
                    close_percentage = settings.HIGH_PROFIT_CLOSE1
                    reason = f"高盈利回撤{settings.HIGH_PROFIT_DRAWBACK1}%"
                elif profit_drawback >= settings.HIGH_PROFIT_DRAWBACK2:
                    close_percentage = settings.HIGH_PROFIT_CLOSE2
                    reason = f"高盈利回撤{settings.HIGH_PROFIT_DRAWBACK2}%"

            # 低盈利止盈
            elif pnl_rate >= settings.LOW_PROFIT_THRESHOLD and pnl_rate < settings.HIGH_PROFIT_THRESHOLD:
                if profit_drawback >= settings.LOW_PROFIT_DRAWBACK1:
                    close_percentage = settings.LOW_PROFIT_CLOSE1
                    reason = f"低盈利回撤{settings.LOW_PROFIT_DRAWBACK1}%"

            # 保本止盈：产生利润回撤并触发高盈利止盈或低盈利止盈之后，剩余仓位在回报率降到BREAKEVEN_THRESHOLD时全部平仓
            # 文档说明：回报率降到BREAKEVEN_THRESHOLD%（如0.5%）时触发
            # 关键：保本止盈不应在亏损时触发，只应在盈利降到很低时触发
            if 'last_take_profit_pct' in position_info and position_info['last_take_profit_pct'] is not None:
                # 前提：已经执行过高盈利或低盈利止盈（部分平仓）
                # 触发条件：盈利率在[0, BREAKEVEN_THRESHOLD]范围内（正值，表示略微盈利）
                if max_profit_pct > 0 and 0 <= pnl_rate <= settings.BREAKEVEN_THRESHOLD:
                    close_percentage = 100
                    reason = f"保本止盈（回报率{pnl_rate:.2f}%降到阈值{settings.BREAKEVEN_THRESHOLD}%以内）"

            if close_percentage > 0:
                logger.info(f"{symbol} 触发止盈: {reason}, 平仓比例={close_percentage}%")

                # ✅ 先取消跟踪止盈订单（如果存在）
                self._cancel_trailing_take_profit(symbol, position_info)

                # 标记止盈执行，用于重新进场
                position_info['last_take_profit_pct'] = pnl_rate
                position_info['last_take_profit_time'] = time.time()
                position_info['take_profit_closed_pct'] = close_percentage

                # 执行止盈平仓 - 创建市价单或限价单
                self._execute_close_position(symbol, position_info, close_percentage, reason)

        except Exception as e:
            logger.error(f"{symbol} 执行止盈失败: {str(e)}")

    def _execute_close_position(self, symbol: str, position_info: Dict, close_pct: float, reason: str):
        """
        执行平仓逻辑（市价单）

        Args:
            symbol: 币种
            position_info: 持仓信息
            close_pct: 平仓比例（%）
            reason: 平仓原因
        """
        try:
            direction = position_info['direction']
            total_quantity = position_info['total_quantity']
            current_price = position_info.get('current_price', 0)

            # 如果全部平仓，先取消所有挂单
            if close_pct >= 100:
                pending_orders = position_info.get('pending_orders', [])
                for pending_order in pending_orders:
                    if pending_order.get('order_id') and pending_order.get('status') == 'submitted':
                        try:
                            self.client.cancel_order(symbol, pending_order['order_id'])
                            logger.info(f"{symbol} 取消挂单: {pending_order['order_id']} (平仓前)")
                        except Exception as e:
                            logger.warning(f"{symbol} 取消挂单失败: {pending_order['order_id']}, {str(e)}")

            # ✅ 生产工具 - 如果全部平仓，取消止损单（平仓前）
            if close_pct >= 100:
                stop_loss_algo_ids = []
                client_algo_ids = []

                # 收集止损单ID
                if '_stop_loss_algo_id1' in position_info:
                    algo_id = position_info.pop('_stop_loss_algo_id1')
                    if algo_id:
                        stop_loss_algo_ids.append(algo_id)
                if '_stop_loss_algo_id2' in position_info:
                    algo_id = position_info.pop('_stop_loss_algo_id2')
                    if algo_id:
                        stop_loss_algo_ids.append(algo_id)

                # 取消所有止损单
                for algo_id in stop_loss_algo_ids:
                    try:
                        self.client.cancel_algo_order(symbol, algo_id=algo_id)
                        logger.info(f"{symbol} 止损单已取消: algoId={algo_id} (平仓前)")
                    except Exception as e:
                        logger.warning(f"{symbol} 取消止损单失败: algoId={algo_id}, {str(e)}")

            # 计算平仓数量
            if close_pct >= 100:
                close_quantity = total_quantity
            else:
                close_quantity = total_quantity * (close_pct / 100)

            # ✅ 新增：部分平仓时，调整数量以满足minNotional（学习开仓逻辑）
            if close_pct < 100 and current_price > 0:
                # 计算名义价值
                actual_notional = close_quantity * current_price

                # 获取minNotional
                min_notional = 0.0
                symbol_info = self.client.get_symbol_info(symbol)
                if symbol_info and 'filters' in symbol_info:
                    for f in symbol_info['filters']:
                        if f.get('filterType') == 'NOTIONAL':
                            min_notional = float(f.get('minNotional', 0.0))

                # 如果名义价值不足且有minNotional限制，调整数量
                if min_notional > 0 and actual_notional < min_notional:
                    # 获取step_size
                    step_size = 0.001
                    for f in symbol_info['filters']:
                        if f.get('filterType') == 'LOT_SIZE':
                            step_size = float(f.get('stepSize', 0.001))
                            break

                    # 向上取整到step_size倍数
                    min_quantity_raw = min_notional / current_price
                    min_quantity_for_notional = math.ceil(min_quantity_raw / step_size) * step_size
                    min_quantity_for_notional = self.client.adjust_quantity_precision(symbol, min_quantity_for_notional)
                    close_quantity = min_quantity_for_notional

                    logger.info(f"{symbol} 平仓调整: raw={min_quantity_raw:.8f}, step={step_size:.8f}, minNotional={min_notional} USDT, ceil后={close_quantity:.8f}")

                    # 检查是否超过总持仓
                    if close_quantity > total_quantity:
                        logger.warning(f"{symbol} 调整后数量{close_quantity:.8f}超过总持仓{total_quantity:.8f}，改为全平")
                        close_quantity = total_quantity
                        close_pct = 100

            # 验证平仓数量有效性
            if not self._validate_close_quantity(symbol, close_quantity):
                logger.error(f"{symbol} 平仓数量验证失败，取消平仓")
                return None

            # 确定平仓方向（与开仓方向相反）
            side = 'SELL' if direction == 'LONG' else 'BUY'

            # 修复 -4015错误：使用秒级时间戳，格式: symbol前6字符 + _C_ + 秒级时间戳 (总长度 < 20 字符)
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=close_quantity,
                newClientOrderId=f"{symbol[:6]}_C_{int(time.time())}"  # API合规性：添加idempotency
            )

            if order:
                logger.info(f"{symbol} 平仓订单创建成功: {reason}, 平仓数量={close_quantity:.8f}, 订单ID={order.get('orderId')}")

                # 如果全部平仓，更新持仓状态
                if close_pct >= 100:
                    position_info['total_quantity'] = 0
                    position_info['status'] = 'closed'
                    position_info['close_time'] = time.time()
                    # 清空挂单列表
                    position_info['pending_orders'] = []
                    # ✅ 清除止损验证失败标记（持仓已关闭，下次开仓重新计算）
                    position_info.pop('_stop_loss_validation_failed', None)
                    position_info.pop('_stop_loss_validation_reason', None)
                    position_info.pop('_stop_loss_validation_time', None)
                else:
                    position_info['total_quantity'] -= close_quantity

            return order

        except Exception as e:
            logger.error(f"{symbol} 执行平仓失败: {str(e)}")
            return None

    def _check_reenter_opportunity(self, symbol: str, position_info: Dict, current_price: float) -> bool:
        """
        检查重新进场机会

        Args:
            symbol: 币种
            position_info: 持仓信息
            current_price: 当前价格

        Returns:
            bool: 是否应该重新进场
        """
        try:
            # 检查是否执行过止盈
            if 'last_take_profit_pct' not in position_info:
                return False

            last_profit_pct = position_info['last_take_profit_pct']
            last_profit_time = position_info.get('last_take_profit_time', 0)
            entry_price = position_info.get('entry_price', 0)

            if entry_price <= 0:
                return False

            # 计算当前收益率
            if position_info['direction'] == 'LONG':
                current_pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                current_pnl = ((entry_price - current_price) / entry_price) * 100

            # 检查价格是否上涨至再次进场条件
            # 再次进场阈值：如果价格从止盈价格上涨超过 PROFIT_REENTER_THRESHOLD%
            reenter_price_threshold = last_profit_pct + settings.PROFIT_REENTER_THRESHOLD

            logger.debug(f"{symbol} 重新进场检查: 最后止盈={last_profit_pct:.2f}%, 当前={current_pnl:.2f}%, 再次进场阈值={reenter_price_threshold:.2f}%")

            if current_pnl >= reenter_price_threshold:
                # 检查最小时间间隔（避免频繁重新进场）
                min_reenter_interval = getattr(settings, 'REENTER_MIN_INTERVAL', 300)  # 默认5分钟
                if time.time() - last_profit_time >= min_reenter_interval:
                    logger.info(f"{symbol} 触发重新进场条件: 当前收益={current_pnl:.2f}% >= 再次进场阈值={reenter_price_threshold:.2f}%")
                    return True

            return False

        except Exception as e:
            logger.error(f"{symbol} 检查重新进场机会失败: {str(e)}")
            return False

    def _reenter_position(self, symbol: str, position_info: Dict):
        """
        重新进场：只重新设置止损和止盈，不增加头寸

        Args:
            symbol: 币种
            position_info: 持仓信息
        """
        try:
            logger.info(f"{symbol} 执行重新进场: 重置止损和止盈")

            # ✅ 修复: 重新进场前先取消旧的止损单（防止累积）
            stop_loss_algo_ids = []
            if '_stop_loss_algo_id1' in position_info:
                algo_id = position_info.pop('_stop_loss_algo_id1')
                if algo_id:
                    stop_loss_algo_ids.append(algo_id)
            if '_stop_loss_algo_id2' in position_info:
                algo_id = position_info.pop('_stop_loss_algo_id2')
                if algo_id:
                    stop_loss_algo_ids.append(algo_id)

            # 取消所有旧的止损单
            for algo_id in stop_loss_algo_ids:
                try:
                    self.client.cancel_algo_order(symbol, algo_id=algo_id)
                    logger.info(f"{symbol} 重新进场前取消旧止损单: algoId={algo_id}")
                except Exception as e:
                    logger.warning(f"{symbol} 取消旧止损单失败: algoId={algo_id}, {str(e)}")

            # 重置止盈止损状态
            position_info['last_take_profit_pct'] = None
            position_info['last_take_profit_time'] = None
            position_info['take_profit_closed_pct'] = 0
            position_info['max_profit_pct'] = 0  # 重置最高收益率

            # 重新设置止损单
            entry_price = position_info.get('entry_price', 0)
            direction = position_info.get('direction', 'LONG')
            leverage = position_info.get('leverage', settings.LEVERAGE)

            if entry_price > 0:
                self._setup_stop_loss_orders(symbol, position_info, entry_price, direction, leverage)
            else:
                logger.warning(f"{symbol} 缺少入场价格信息，无法重新设置止损")

            logger.info(f"{symbol} 重新进场完成: 止损和止盈已重新设置")

        except Exception as e:
            logger.error(f"{symbol} 重新进场失败: {str(e)}")

    def _check_position_complete(self, symbol: str, position_info: Dict, pnl_rate: float) -> bool:
        """
        检查建仓完成条件

        Args:
            symbol: 币种
            position_info: 持仓信息
            pnl_rate: 当前盈亏率

        Returns:
            bool: 是否建仓完成
        """
        try:
            max_profit_pct = position_info['max_profit_pct']
            total_investment = position_info['total_investment']
            completed_investment = position_info.get('completed_investment', position_info['initial_margin'])

            # 条件1: 盈利上涨回撤
            if pnl_rate >= settings.PROFIT_STEP3 and max_profit_pct > pnl_rate + settings.POSITION_COMPLETE_PROFIT_RISE:
                logger.info(f"{symbol} 建仓完成: 盈利上涨回撤{settings.POSITION_COMPLETE_PROFIT_RISE}%")
                self._finalize_position_building(symbol, position_info, "盈利上涨回撤")
                return True

            # 条件2: 亏损下跌
            if pnl_rate <= settings.LOSS_STEP3 - settings.POSITION_COMPLETE_LOSS_FALL:
                logger.info(f"{symbol} 建仓完成: 亏损下跌{settings.POSITION_COMPLETE_LOSS_FALL}%")
                self._finalize_position_building(symbol, position_info, "亏损下跌")
                return True

            # 条件3: 达到单币种最大投资金额
            if completed_investment >= settings.SINGLE_SYMBOL_MAX_INVESTMENT:
                logger.info(f"{symbol} 建仓完成: 达到最大投资金额{settings.SINGLE_SYMBOL_MAX_INVESTMENT} USDT")
                self._finalize_position_building(symbol, position_info, "达到最大投资金额")
                return True

            # 条件4: 保证金达到最大限制
            current_margin = position_info.get('total_investment', 0)
            if current_margin >= settings.MAX_MARGIN_PER_SYMBOL:
                logger.info(f"{symbol} 建仓完成: 达到最大保证金限制{settings.MAX_MARGIN_PER_SYMBOL} USDT")
                self._finalize_position_building(symbol, position_info, "达到最大保证金限制")
                return True

            return False

        except Exception as e:
            logger.error(f"{symbol} 检查建仓完成失败: {str(e)}")
            return False

    def _finalize_position_building(self, symbol: str, position_info: Dict, reason: str):
        """
        完成建仓，取消所有未成交的加仓订单

        Args:
            symbol: 币种
            position_info: 持仓信息
            reason: 完成原因
        """
        try:
            # 取消所有未成交的加仓订单
            pending_orders = position_info.get('pending_orders', [])
            for pending_order in pending_orders:
                if pending_order.get('order_id') and pending_order['status'] == 'submitted':
                    try:
                        self.client.cancel_order(symbol, pending_order['order_id'])
                        logger.info(f"{symbol} 取消加仓订单: {pending_order['order_id']}")
                    except Exception as e:
                        logger.warning(f"{symbol} 取消订单失败: {pending_order['order_id']}, {str(e)}")

            # 清空待执行订单
            position_info['pending_orders'] = []
            position_info['completed_investment'] = position_info.get('total_investment', 0)

            # 标记建仓完成（用于止损和止盈判断）
            position_info['position_complete'] = True

            logger.info(f"{symbol} 建仓完成: {reason}")

            # ✅ 建仓完成后立即设置止损条件单
            # 使用新的 Algo Order API (/fapi/v1/algoOrder) 创建止损挂单
            entry_price = position_info.get('entry_price', 0)
            direction = position_info.get('direction', 'LONG')
            leverage = position_info.get('leverage', settings.LEVERAGE)

            if entry_price > 0:
                self._setup_stop_loss_orders(symbol, position_info, entry_price, direction, leverage)
            else:
                logger.warning(f"{symbol} 缺少入场价格信息，无法设置止损单")

        except Exception as e:
            logger.error(f"{symbol} 完成建仓失败: {str(e)}")

    def _execute_position_building(self, symbol: str, position_info: Dict, current_price: float, pnl_rate: float):
        """
        执行建仓计划（加仓）

        Args:
            symbol: 币种
            position_info: 持仓信息
            current_price: 当前价格
            pnl_rate: 当前盈亏率
        """
        try:
            pending_orders = position_info.get('pending_orders', [])

            for pending_order in pending_orders:
                if pending_order['status'] != 'pending':
                    continue

                trigger_rate = pending_order['trigger_rate']

                # 检查是否触发加仓条件
                # 亏损加仓：trigger_rate < 0，触发条件是 pnl_rate <= trigger_rate
                # 盈利加仓：trigger_rate > 0，触发条件是 pnl_rate >= trigger_rate
                should_trigger = False
                if trigger_rate < 0 and pnl_rate <= trigger_rate:
                    # 亏损加仓
                    should_trigger = True
                elif trigger_rate > 0 and pnl_rate >= trigger_rate:
                    # 盈利加仓
                    should_trigger = True

                if should_trigger:
                    # 检查是否已经尝试过此级别的加仓（防止重复触发）
                    if pending_order.get('attempted', False):
                        # 如果订单已提交（status='submitted'）但未成交，继续等待
                        # 如果订单执行失败（返回False但status未变），则跳过（记录警告但不重试）
                        if pending_order.get('status') == 'submitted':
                            logger.debug(f"{symbol} 级别{pending_order['index']}的订单已提交，等待成交")
                        else:
                            logger.warning(f"{symbol} 级别{pending_order['index']}已尝试过但未成功，跳过重复触发")
                        continue

                    # 标记为已尝试（无论成功与否）
                    pending_order['attempted'] = True

                    logger.info(f"{symbol} 触发加仓条件: 盈亏率{pnl_rate:.2f}% {'<=' if trigger_rate < 0 else '>='} 触发率{trigger_rate}%")

                    # 执行加仓
                    success = self._add_position(position_info, pending_order, current_price)
                    if success:
                        pending_order['status'] = 'executed'
                        position_info['added_levels'].append(pending_order['index'])

                        # 发送加仓消息
                        add_amount = pending_order['add_margin']
                        new_total_investment = position_info['total_investment']
                        allocated_funds = position_info.get('allocated_funds', settings.SINGLE_SYMBOL_MAX_INVESTMENT)

                        # 加仓时显示当前总投资占分配资金的比例
                        self._send_trade_message(
                            symbol=symbol,
                            action='ADD',
                            direction=position_info['direction'],
                            current_investment=new_total_investment,
                            total_planned_investment=allocated_funds,
                            quantity=pending_order['quantity'],
                            amount=add_amount,
                            price=current_price,
                            current_price=current_price,
                            leverage=position_info['leverage'],
                            profit=position_info['profit'],
                            profit_pct=position_info['profit_pct']
                        )

                        # ✅ 记录加仓到交易表格
                        try:
                            ratio_str = f"{new_total_investment/allocated_funds*100:.0f}/{100}"
                            self.trade_recorder.record_trade(
                                symbol=symbol,
                                action='ADD',
                                direction=position_info['direction'],
                                price=current_price,
                                quantity=pending_order['quantity'],
                                amount=add_amount,
                                leverage=position_info['leverage'],
                                ratio=ratio_str
                            )
                        except Exception as e:
                            logger.warning(f"{symbol} 记录加仓失败: {str(e)}")

                        logger.info(f"{symbol} 加仓执行成功，当前仓位比例: {(new_total_investment/allocated_funds)*100:.0f}%")
                    else:
                        logger.warning(f"{symbol} 加仓执行失败")

        except Exception as e:
            logger.error(f"{symbol} 执行建仓计划失败: {str(e)}")

    def _execute_stop_loss(self, symbol: str, position_info: Dict, pnl_rate: float):
        """
        执行止损逻辑

        Args:
            symbol: 币种
            position_info: 持仓信息
            pnl_rate: 当前盈亏率（%）
        """
        try:
            # 关键：只有建仓完成后才触发止损
            # 建仓未完成时不应该触发止损，避免在建仓过程中被止损
            if not position_info.get('position_complete', False):
                logger.debug(f"{symbol} 建仓未完成，跳过止损检查")
                return

            entry_price = position_info['entry_price']
            direction = position_info['direction']
            leverage = position_info['leverage']

            stop_loss_triggered = False
            stop_price = 0
            close_percentage = 0
            reason = ""

            # 第一级止损
            if pnl_rate <= settings.STOPLOSS_TRIGGER1:
                # 动态计算止损价格
                if direction == 'LONG':
                    stop_price = entry_price * (1 + (settings.STOPLOSS_TRIGGER1 / 100 - settings.DELAY_RATIO / 100) / leverage)
                else:  # SHORT
                    stop_price = entry_price * (1 + (settings.STOPLOSS_TRIGGER1 / 100 + settings.DELAY_RATIO / 100) / leverage)

                close_percentage = settings.STOPLOSS_CLOSE1  # 从环境变量读取平仓比例
                reason = f"第一级止损（盈亏率{pnl_rate:.2f}% <= {settings.STOPLOSS_TRIGGER1}%）"

                # 设置止损限价单
                self._create_stop_limit_order(symbol, position_info, stop_price, close_percentage, reason)
                position_info['stop_loss_levels']['level1'] = True

            # 第二级止损（如果第一级已触发）
            elif pnl_rate <= settings.STOPLOSS_TRIGGER2 and position_info['stop_loss_levels'].get('level1', False):
                if direction == 'LONG':
                    stop_price = entry_price * (1 + (settings.STOPLOSS_TRIGGER2 / 100 - settings.DELAY_RATIO / 100) / leverage)
                else:  # SHORT
                    stop_price = entry_price * (1 + (settings.STOPLOSS_TRIGGER2 / 100 + settings.DELAY_RATIO / 100) / leverage)

                close_percentage = settings.STOPLOSS_CLOSE2
                reason = f"第二级止损（盈亏率{pnl_rate:.2f}% <= {settings.STOPLOSS_TRIGGER2}%）"

                # 设置止损限价单
                self._create_stop_limit_order(symbol, position_info, stop_price, close_percentage, reason)
                position_info['stop_loss_levels']['level2'] = True

            # 第三级止损（紧急市价止损）
            elif pnl_rate <= settings.STOPLOSS_TRIGGER3:
                logger.warning(f"{symbol} 触发第三级止损，立即市价清空所有剩余仓位")
                self._execute_close_position(symbol, position_info, 100, f"第三级止损（盈亏率{pnl_rate:.2f}% <= {settings.STOPLOSS_TRIGGER3}%）")

        except Exception as e:
            logger.error(f"{symbol} 执行止损失败: {str(e)}")

    def _validate_close_quantity(self, symbol: str, close_quantity: float) -> bool:
        """
        验证平仓数量是否符合币安要求

        Args:
            symbol: 币种
            close_quantity: 平仓数量

        Returns:
            bool: 数量是否有效
        """
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"{symbol} 无法获取币种信息，跳过最小数量验证")
                return True

            # 检查 LOT_SIZE 过滤器
            if 'filters' in symbol_info:
                for f in symbol_info['filters']:
                    if f.get('filterType') == 'LOT_SIZE':
                        min_qty = float(f.get('minQty', 0))
                        max_qty = float(f.get('maxQty', float('inf')))

                        if close_quantity < min_qty:
                            logger.warning(f"{symbol} 平仓数量 {close_quantity:.8f} 低于最小要求 {min_qty:.8f}，无法执行")
                            return False

                        if close_quantity > max_qty:
                            logger.warning(f"{symbol} 平仓数量 {close_quantity:.8f} 超过最大限制 {max_qty:.8f}，无法执行")
                            return False

                    elif f.get('filterType') == 'NOTIONAL':
                        # 币安要求订单名义价值不能太小
                        min_notional = float(f.get('minNotional', 0))
                        logger.debug(f"{symbol} NOTIONAL 最小要求: {min_notional}")

            return True

        except Exception as e:
            logger.warning(f"{symbol} 验证平仓数量失败: {str(e)}")
            return True

    def _create_stop_limit_order(self, symbol: str, position_info: Dict, stop_price: float, close_pct: float, reason: str):
        """
        创建止损限价单（LIMIT订单类型）
        改进版：使用统一的订单数量验证函数
        
        Args:
            symbol: 币种
            position_info: 持仓信息
            stop_price: 限价单价格
            close_pct: 平仓比例（%）
            reason: 原因
        
        Returns:
            dict or None: 订单信息
        """
        try:
            direction = position_info['direction']
            total_quantity = position_info['total_quantity']
            
            # 计算平仓数量
            if close_pct >= 100:
                close_quantity = total_quantity
            else:
                close_quantity = total_quantity * (close_pct / 100)
            
            # ✅ 使用统一的订单数量验证
            # 参数说明：
            # - symbol: 币种
            # - quantity: 待验证的数量（close_quantity）
            # - price: 价格用于计算名义价值（stop_price）
            # - order_type: 订单类型（'LIMIT'）
            # - ensure_min_notional: True（止损单必须满足minNotional）
            # - max_quantity_limit: 如果close_pct较小，限制最大平仓数量
            # - description: 用于日志
            close_quantity, is_valid, adjustment_info = self._validate_order_quantity_unified(
                symbol=symbol,
                quantity=close_quantity,
                price=stop_price,
                order_type='LIMIT',
                ensure_min_notional=True,
                max_quantity_limit=None,
                description=f'{reason}止损单'
            )
            
            if not is_valid:
                logger.error(f"{symbol} {reason}单数量验证失败: {adjustment_info.get('final_quantity', close_quantity)}")
                return None
            
            # 如果数量被调整了，记录日志
            if adjustment_info.get('adjusted', False):
                logger.info(f"{symbol} {reason}单数量已调整: 原始={adjustment_info['original_quantity']:.8f}, "
                           f"最终={adjustment_info['final_quantity']:.8f}, "
                           f"名义价值={adjustment_info['final_notional']:.2f} USDT")
                if '调整项' in adjustment_info:
                    logger.debug(f"{symbol} 调整详情: {adjustment_info['调整项']}")
            
            # 平仓方向与开仓相反
            side = 'SELL' if direction == 'LONG' else 'BUY'
            
            logger.info(f"{symbol} 创建{reason}限价单: 限价={stop_price:.6f}, 数量={close_quantity:.8f}, 比例={close_pct:.1f}%")
            
            # 创建限价止损单
            # 修复 -4015错误：使用秒级时间戳，格式: symbol前6字符 + _SL_ + 秒级时间戳 (总长度 < 20 字符)
            # API合规性：对于单向模式，不发送positionSide参数；添加newClientOrderId支持幂等性
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='LIMIT',
                quantity=close_quantity,
                price=stop_price,
                timeInForce='GTC',
                newClientOrderId=f"{symbol[:6]}_SL_{int(time.time())}"  # API合规性：添加idempotency
            )
            
            if order and 'orderId' in order:
                logger.info(f"{symbol} {reason}限价单已创建: ID={order['orderId']}")
            else:
                logger.error(f"{symbol} {reason}单创建失败: 返回订单信息无效")
                return None
            
            return order
            
        except Exception as e:
            logger.error(f"{symbol} 创建{reason}限价单失败: {str(e)}", exc_info=True)
            return None

    def _validate_stop_loss_quantity(self, symbol: str, quantity: float, limit_price: float, leverage: int,
                                     min_qty: float, max_qty: float, min_notional: float) -> float:
        """
        验证并调整止损单数量，确保满足币安的所有限制

        使用与加仓相同的验证逻辑：
        1. 基于minQty计算最小保证金
        2. 比较用户期望数量和最小数量
        3. 确保名义价值满足要求

        Args:
            symbol: 币种
            quantity: 基于比例计算的初步数量
            limit_price: 止损触发价格
            leverage: 杠杆
            min_qty: LOT_SIZE最小数量
            max_qty: LOT_SIZE最大数量
            min_notional: NOTIONAL最小名义价值

        Returns:
            float: 调整后的数量（0表示不满足要求）
        """
        try:
            # 只在API返回有效的min_notional时才进行验证
            if min_notional <= 0:
                logger.warning(f"{symbol} 未获取到有效的minNotional，跳过名义价值验证")
                # 仍然应用精度调整和LOT_SIZE验证
                quantity = self.client.adjust_quantity_precision(symbol, quantity)
                if quantity < min_qty or quantity > max_qty:
                    return 0
                return quantity

            min_notional_filter = min_notional

            # 应用精度调整
            quantity = self.client.adjust_quantity_precision(symbol, quantity)

            # 验证LOT_SIZE范围
            if quantity < min_qty:
                logger.info(f"{symbol} 止损数量{quantity:.8f}小于minQty={min_qty:.8f}，尝试调整")
                quantity = min_qty

            if quantity > max_qty:
                logger.warning(f"{symbol} 止损数量{quantity:.8f}超过maxQty={max_qty:.8f}，跳过")
                return 0

            quantity = self.client.adjust_quantity_precision(symbol, quantity)

            # 验证NOTIONAL（基于limit_price）
            actual_notional = quantity * limit_price

            if actual_notional < min_notional_filter:
                # 获取step_size
                symbol_info = self.client.get_symbol_info(symbol)
                step_size = 0.001
                if symbol_info and 'filters' in symbol_info:
                    for f in symbol_info['filters']:
                        if f.get('filterType') == 'LOT_SIZE':
                            step_size = float(f.get('stepSize', 0.001))
                            break

                # 向上取整到step_size倍数
                min_quantity_raw = min_notional_filter / limit_price
                min_quantity_for_notional = math.ceil(min_quantity_raw / step_size) * step_size
                min_quantity_for_notional = self.client.adjust_quantity_precision(symbol, min_quantity_for_notional)

                logger.info(f"{symbol} 止损数据: raw={min_quantity_raw:.8f}, step={step_size:.8f}, ceil后={min_quantity_for_notional:.8f}")
                quantity = min_quantity_for_notional
                actual_notional = quantity * limit_price

            # 最终验证
            if quantity < min_qty:
                logger.warning(f"{symbol} 止损数量{quantity:.8f}仍小于minQty={min_qty:.8f}，跳过")
                return 0

            actual_notional = quantity * limit_price
            # 最终验证：如果向上取整后仍不满足（极罕见），接受失败
            if actual_notional < min_notional_filter:
                logger.warning(f"{symbol} 止损名义价值{actual_notional:.2f}仍小于{min_notional_filter}，跳过")
                return 0

            logger.debug(f"{symbol} 止损数量验证通过: 数量={quantity:.8f}, 名义价值={actual_notional:.2f} USDT >= {min_notional_filter}")
            return quantity

        except Exception as e:
            logger.error(f"{symbol} 止损数量验证失败: {str(e)}")
            return 0

    def _setup_stop_loss_orders(self, symbol: str, position_info: Dict, entry_price: float, direction: str, leverage: int):
        """
        开仓完成后立即设置第一级和第二级止损条件单
        使用币安新的 Algo Order API（/fapi/v1/algoOrder），2025-12-09后要求

        Args:
            symbol: 币种
            position_info: 持仓信息
            entry_price: 开仓价格
            direction: 方向
            leverage: 杠杆倍数
        """
        try:
            total_quantity = position_info['total_quantity']

            # ✅ 获取币安限制（用于验证止损单参数）
            symbol_info = self.client.get_symbol_info(symbol)
            min_qty = 0.001
            max_qty = float('inf')
            min_notional = 0.0

            if symbol_info and 'filters' in symbol_info:
                for f in symbol_info['filters']:
                    if f.get('filterType') == 'LOT_SIZE':
                        min_qty = float(f.get('minQty', 0.001))
                        max_qty = float(f.get('maxQty', float('inf')))
                    elif f.get('filterType') == 'NOTIONAL':
                        min_notional = float(f.get('minNotional', 0))

            logger.debug(f"{symbol} 止损单币安限制 - LOT_SIZE: [{min_qty}, {max_qty}], NOTIONAL: {min_notional}")

            # 平仓方向与开仓相反
            side = 'SELL' if direction == 'LONG' else 'BUY'

            # ========== 第一级止损条件单 ==========
            stop_price1 = self._calculate_stop_loss_price(entry_price, settings.STOPLOSS_TRIGGER1, direction, leverage, symbol)

            # 基于比例计算初步数量
            close_quantity1_pre = total_quantity * (settings.STOPLOSS_CLOSE1 / 100)

            # ✅ 使用与加仓相同的验证逻辑：确保数量满足minQty和minNotional
            close_quantity1 = self._validate_stop_loss_quantity(
                symbol=symbol,
                quantity=close_quantity1_pre,
                limit_price=stop_price1,
                leverage=leverage,
                min_qty=min_qty,
                max_qty=max_qty,
                min_notional=min_notional
            )

            if close_quantity1 > 0:
                # ✅ 修复: 创建新止损单前检查并取消旧订单（防止重复）
                if '_stop_loss_algo_id1' in position_info:
                    old_algo_id = position_info['_stop_loss_algo_id1']
                    try:
                        self.client.cancel_algo_order(symbol, algo_id=old_algo_id)
                        logger.info(f"{symbol} 创建新第一级止损单前取消旧单: algoId={old_algo_id}")
                    except Exception as e:
                        logger.warning(f"{symbol} 取消旧第一级止损单失败: {str(e)}")

                # 使用秒级时间戳，格式: symbol前6字符 + _SL1_ + 秒级时间戳
                client_algo_id1 = f"{symbol[:6]}_SL1_{int(time.time())}"

                # ✅ 使用新的 Algo Order API
                algo_order = self.client.create_algo_order(
                    symbol=symbol,
                    side=side,
                    trigger_price=stop_price1,
                    quantity=close_quantity1,
                    order_type='STOP_MARKET',
                    working_type='MARK_PRICE',  # 使用标记价格触发，更准确
                    reduce_only='true',
                    client_algo_id=client_algo_id1
                )

                if algo_order and 'algoId' in algo_order:
                    logger.info(f"{symbol} 第一级止损条件单已创建: algoId={algo_order['algoId']}, 触发价={stop_price1:.4f}, 平仓比例={settings.STOPLOSS_CLOSE1}%, 数量={close_quantity1:.8f}")
                    # 保存 algoId 到 position_info，方便后续管理
                    if '_stop_loss_algo_id1' not in position_info:
                        position_info['_stop_loss_algo_id1'] = algo_order['algoId']
                    if '_stop_loss_client_algo_id1' not in position_info:
                        position_info['_stop_loss_client_algo_id1'] = client_algo_id1
            else:
                # ✅ 止损数量验证失败，标记以便避免重复尝试
                actual_notional = total_quantity * (settings.STOPLOSS_CLOSE1 / 100) * stop_price1
                if min_notional > 0:
                    logger.warning(f"{symbol} 第一级止损数量不足无法创建订单（持仓={total_quantity:.2f}, 价格={stop_price1:.4f}, 名义价值={actual_notional:.2f} USDT，需要≥{min_notional:.2f} USDT），将跳过")
                else:
                    logger.warning(f"{symbol} 第一级止损数量不足无法创建订单（持仓={total_quantity:.2f}, 价格={stop_price1:.4f}, 名义价值={actual_notional:.2f} USDT，minNotional=0，可能币种无限制），将跳过")
                position_info['_stop_loss_validation_failed'] = True
                position_info['_stop_loss_validation_time'] = time.time()
                position_info['_stop_loss_validation_reason'] = f"第一层止损{settings.STOPLOSS_CLOSE1}%名义价值不足"

            # ========== 第二级止损条件单 ==========
            # ✅ 只有第一层验证通过后才尝试第二层
            if not position_info.get('_stop_loss_validation_failed', False):
                stop_price2 = self._calculate_stop_loss_price(entry_price, settings.STOPLOSS_TRIGGER2, direction, leverage, symbol)

                close_quantity2_pre = total_quantity * (settings.STOPLOSS_CLOSE2 / 100)

                # ✅ 使用与加仓相同的验证逻辑
                close_quantity2 = self._validate_stop_loss_quantity(
                    symbol=symbol,
                    quantity=close_quantity2_pre,
                    limit_price=stop_price2,
                    leverage=leverage,
                    min_qty=min_qty,
                    max_qty=max_qty,
                    min_notional=min_notional
                )

                if close_quantity2 > 0:
                    # ✅ 修复: 创建新止损单前检查并取消旧订单（防止重复）
                    if '_stop_loss_algo_id2' in position_info:
                        old_algo_id = position_info['_stop_loss_algo_id2']
                        try:
                            self.client.cancel_algo_order(symbol, algo_id=old_algo_id)
                            logger.info(f"{symbol} 创建新第二级止损单前取消旧单: algoId={old_algo_id}")
                        except Exception as e:
                            logger.warning(f"{symbol} 取消旧第二级止损单失败: {str(e)}")

                    # 使用秒级时间戳
                    client_algo_id2 = f"{symbol[:6]}_SL2_{int(time.time())}"

                    # ✅ 使用新的 Algo Order API
                    algo_order = self.client.create_algo_order(
                        symbol=symbol,
                        side=side,
                        trigger_price=stop_price2,
                        quantity=close_quantity2,
                        order_type='STOP_MARKET',
                        working_type='MARK_PRICE',
                        reduce_only='true',
                        client_algo_id=client_algo_id2
                    )

                    if algo_order and 'algoId' in algo_order:
                        logger.info(f"{symbol} 第二级止损条件单已创建: algoId={algo_order['algoId']}, 触发价={stop_price2:.4f}, 平仓比例={settings.STOPLOSS_CLOSE2}%, 数量={close_quantity2:.8f}")
                        # 保存 algoId
                        if '_stop_loss_algo_id2' not in position_info:
                            position_info['_stop_loss_algo_id2'] = algo_order['algoId']
                        if '_stop_loss_client_algo_id2' not in position_info:
                            position_info['_stop_loss_client_algo_id2'] = client_algo_id2
                else:
                    # ✅ 第二层止损数量验证失败
                    actual_notional = total_quantity * (settings.STOPLOSS_CLOSE2 / 100) * stop_price2
                    if min_notional > 0:
                        logger.warning(f"{symbol} 第二级止损数量不足无法创建订单（持仓={total_quantity:.2f}, 价格={stop_price2:.4f}, 名义价值={actual_notional:.2f} USDT，需要≥{min_notional:.2f} USDT）")
                    else:
                        logger.warning(f"{symbol} 第二级止损数量不足无法创建订单（持仓={total_quantity:.2f}, 价格={stop_price2:.4f}, 名义价值={actual_notional:.2f} USDT，minNotional=0，可能币种无限制）")
                    # 更新失败原因（仅记录，不影响标记）
                    reason = position_info.get('_stop_loss_validation_reason', '')
                    position_info['_stop_loss_validation_reason'] = f"{reason}；第二层止损{settings.STOPLOSS_CLOSE2}%也不足" if reason else f"第二层止损{settings.STOPLOSS_CLOSE2}%名义价值不足"

        except Exception as e:
            logger.error(f"{symbol} 设置止损条件单失败: {str(e)}")

    def monitor_positions(self):
        """监控所有持仓"""
        try:
            current_count = len(self.positions)

            # 只在持仓数量变化或每10次监控时记录一次日志
            if not hasattr(self, '_monitor_count'):
                self._monitor_count = 0
            self._monitor_count += 1

            if current_count != getattr(self, '_last_position_count', 0) or self._monitor_count % 10 == 1:
                logger.info(f"开始监控持仓，当前持仓数量: {current_count}")
                self._last_position_count = current_count

            # 1. 定期同步币安API上的持仓状态（每30秒同步一次，避免过于频繁）
            current_time = time.time()
            if current_time - getattr(self, '_last_position_sync', 0) > 30:  # 30秒同步一次
                self._sync_position_states_from_api()
                self._last_position_sync = current_time
            else:
                logger.debug("跳过持仓同步（距离上次同步不足30秒）")

            # 检查持仓数量是否超过限制
            if current_count > settings.MAX_POSITIONS:
                logger.error(f"⚠️  持仓数量严重超限! 当前{current_count}个，限制{settings.MAX_POSITIONS}个")
                logger.error(f"当前持仓: {list(self.positions.keys())}")

                # 发送紧急通知
                if self.telegram_bot:
                    alert_data = {
                        'symbol': 'SYSTEM',
                        'direction': 'WARNING',
                        'message': f'持仓数量超限: {current_count}/{settings.MAX_POSITIONS}',
                        'positions': list(self.positions.keys())
                    }
                    try:
                        self.telegram_bot.send_alert('SYSTEM', alert_data)
                    except Exception as e:
                        logger.warning(f"发送Telegram告警失败: {e}")

            elif current_count >= settings.MAX_POSITIONS * 0.9:
                logger.warning(f"⚠️  持仓数量接近上限: {current_count}/{settings.MAX_POSITIONS}")

            # 清理余额缓存，强制刷新
            self._balance_cache = None

            # 获取币安API的实际持仓
            all_positions = self.position_manager.get_all_positions()

            for symbol, position_info in list(self.positions.items()):
                try:
                    logger.debug(f"{symbol} 监控中...")

                    # 1. 同步持仓状态
                    self._sync_position_state(symbol, position_info, all_positions)

                    # 如果持仓已关闭，跳过后续处理
                    if position_info['status'] == 'closed':
                        logger.debug(f"{symbol} 持仓已关闭，跳过")
                        continue

                    # 2. 更新价格和盈亏
                    current_price = self.client.get_ticker_price(symbol)
                    position_info['current_price'] = current_price

                    # 计算盈亏率（考虑杠杆倍数）
                    entry_price = position_info['entry_price']
                    direction = position_info['direction']
                    total_quantity = position_info['total_quantity']
                    total_investment = position_info['total_investment']
                    leverage = position_info['leverage']

                    if direction == 'LONG':
                        # 做多：盈亏率 = 价格变化率 × 杠杆倍数
                        pnl_rate = ((current_price - entry_price) / entry_price) * leverage * 100
                    else:
                        # 做空：盈亏率 = 价格变化率 × 杠杆倍数
                        pnl_rate = ((entry_price - current_price) / entry_price) * leverage * 100

                    # 计算盈亏金额
                    profit = total_investment * pnl_rate / 100
                    position_info['profit_pct'] = pnl_rate
                    position_info['profit'] = profit

                    # 更新最高盈利率
                    if pnl_rate > position_info['max_profit_pct']:
                        position_info['max_profit_pct'] = pnl_rate

                    # 只在重要价格变化或每20次监控时记录盈亏信息
                    if (abs(pnl_rate) > 5.0 or  # 盈亏超过5%
                        self._monitor_count % 20 == 0 or  # 每20次监控
                        position_info.get('_last_logged_pnl', 0) != pnl_rate):  # 盈亏率发生变化
                        logger.info(f"{symbol} 盈亏率={pnl_rate:.2f}%, 价格={current_price:.6f}, 数量={total_quantity:.6f}")
                        position_info['_last_logged_pnl'] = pnl_rate

                    if not position_info['is_closing']:
                        self._execute_position_building(symbol, position_info, current_price, pnl_rate)

                    # 4. 检查建仓完成条件（但不跳过止盈检查）
                    # 修复：去掉 continue，确保建仓完成后立即执行止盈检查
                    self._check_position_complete(symbol, position_info, pnl_rate)

                    # 5. 执行止盈逻辑
                    # 调试：记录止盈检查前的状态
                    position_complete = position_info.get('position_complete', False)
                    max_profit = position_info.get('max_profit_pct', 0)
                    logger.debug(f"{symbol} 止盈检查: position_complete={position_complete}, max_profit={max_profit:.2f}%, current={pnl_rate:.2f}%")
                    self._execute_take_profit(symbol, position_info, pnl_rate)

                    # 6. 检查重新进场机会
                    if self._check_reenter_opportunity(symbol, position_info, current_price):
                        self._reenter_position(symbol, position_info)

                    # 7. 执行止损逻辑
                    self._execute_stop_loss(symbol, position_info, pnl_rate)

                except Exception as e:
                    logger.error(f"{symbol} 监控持仓失败: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"监控持仓失败: {str(e)}")

    def _monitor_loop(self):
        """监控循环（在后台线程中运行）"""
        while self._running:
            try:
                # 监控所有持仓
                self.monitor_positions()
                
                # 使用配置的持仓监控刷新间隔
                # 可在 .env 中设置 POSITION_MONITOR_SLEEP_TIME
                # 根据杠杆调整：
                # - 5x杠杆: 30秒
                # - 10x杠杆: 20秒
                # - 15-20x杠杆: 15秒（推荐）
                # - 20-30x杠杆: 10秒
                monitor_sleep_time = settings.POSITION_MONITOR_SLEEP_TIME
                
                time.sleep(monitor_sleep_time)
                
                logger.debug(f"持仓监控刷新间隔: {monitor_sleep_time}秒")
                
            except Exception as e:
                logger.error(f"监控循环异常: {str(e)}", exc_info=True)
                # 异常后等待10秒再继续
                time.sleep(10)

    def _calculate_stop_loss_price(self, entry_price: float, trigger_rate: float, direction: str, leverage: int, symbol: str) -> float:
        """
        动态计算止损限价单价格

        Args:
            entry_price: 开仓价格
            trigger_rate: 止损触发率（%），负数
            direction: 方向
            leverage: 杠杆倍数
            symbol: 币种

        Returns:
            float: 限价单价格
        """
        try:
            # 获取tick size信息
            tick_info = self.client.get_tick_size_and_precision(symbol)
            tick_size = tick_info['tick_size']
            price_precision = tick_info['price_precision']

            # 计算基础止损价格
            if direction == 'LONG':
                # 多头止损：价格下跌
                raw_stop_price = entry_price * (1 + trigger_rate / 100 / leverage)
            else:  # SHORT
                # 空头止损：价格上涨
                raw_stop_price = entry_price * (1 - trigger_rate / 100 / leverage)

            # 调整到tick size的倍数，并确保精度符合要求
            adjusted_price = round(raw_stop_price / tick_size) * tick_size

            # 进一步确保价格精度不超过允许范围
            adjusted_price = round(adjusted_price, price_precision)

            # 验证价格合理性
            if adjusted_price <= 0:
                logger.error(f"{symbol} 止损价格计算异常: 调整后价格为非正数")
                return entry_price * 0.95 if direction == 'LONG' else entry_price * 1.05

            logger.info(f"{symbol} 止损价格计算: 入场价={entry_price:.6f}, 触发率={trigger_rate}%, "
                       f"杠杆={leverage}x, 原始价格={raw_stop_price:.6f}, 调整后={adjusted_price:.6f}")

            return adjusted_price

        except Exception as e:
            logger.error(f"{symbol} 计算止损价格失败: {str(e)}")
            # 返回保守的止损价格
            try:
                conservative_price = entry_price * (1 + abs(trigger_rate) / 100 / leverage) if direction == 'LONG' else entry_price * (1 - abs(trigger_rate) / 100 / leverage)
                # 确保保守价格也是有效的
                tick_info = self.client.get_tick_size_and_precision(symbol)
                tick_size = tick_info['tick_size']
                conservative_price = round(conservative_price / tick_size) * tick_size
                return conservative_price
            except:
                # 最后的兜底方案
                return round(entry_price * 0.95, 6) if direction == 'LONG' else round(entry_price * 1.05, 6)
