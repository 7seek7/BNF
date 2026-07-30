"""
持仓同步模块 (Position Sync Module)

职责：
- 同步币安API上的现有持仓
- 从订单历史恢复持仓状态
- 定期同步持仓状态
- 持久化持仓数据

独立原因：
- 持仓同步逻辑独立于交易决策
- 可复用于程序重启后恢复状态
- 减少主模块代码长度（~400行）

性能影响：
- 启动时一次性同步：~100-500ms（取决于持仓数量）
- 定期同步（每30秒）：~50-200ms
- 运行时无额外开销
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, Optional, List, Any
from framework.core.config import get_main_config
from config.settings import Settings

# 全局配置
main_config = get_main_config()
settings = Settings()  # 实例化 settings
from utils.logger import Logger

logger = Logger.get_logger('position_sync')


class PositionSyncManager:
    """
    持仓同步管理器

    负责与币安API同步持仓状态，处理程序重启后的恢复
    """

    def __init__(self, position_manager, client, trade_recorder=None, telegram_bot=None, strategy_type='15mTupo'):
        """
        初始化持仓同步管理器

        Args:
            position_manager: 持仓管理器实例
            client: 币安客户端实例
            trade_recorder: 交易记录器实例（可选）
            telegram_bot: Telegram机器人实例（可选）
            strategy_type: 策略类型 ('v23' 或 'hf')
        """
        self.position_manager = position_manager
        self.client = client
        self.trade_recorder = trade_recorder
        self.telegram_bot = telegram_bot
        self.strategy_type = strategy_type
        self.positions: Dict[str, Dict] = {}
        self._positions_lock = threading.Lock()  # 修复：初始化为实际锁，避免None导致的AttributeError
        self.main_config = get_main_config()

        # 持仓持久化文件路径
        self.positions_file = Path('data/positions.json')
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("持仓同步管理器已初始化")

    def set_positions_ref(self, positions: Dict[str, Dict]):
        """
        设置主模块的positions引用

        Args:
            positions: 主模块的positions字典
        """
        self.positions = positions

    def set_positions_lock_ref(self, lock: Any) -> None:
        """
        设置主模块的positions锁引用

        Args:
            lock: 主模块的positions锁
        """
        self._positions_lock = lock

    def set_telegram_bot(self, telegram_bot: Optional[Any] = None) -> None:
        """
        设置Telegram机器人

        Args:
            telegram_bot: Telegram机器人实例
        """
        self.telegram_bot = telegram_bot

    def sync_existing_positions(self) -> int:
        """
        同步币安API上的现有持仓到本地positions字典
        防止程序重启后重复开仓或错过持仓管理
        
        启动时会取消所有外部挂单，由程序重新创建。

        Returns:
            同步的持仓数量
        """
        try:
            logger.info("正在同步币安API上的现有持仓...")

            # ✅ 【关键】启动时取消所有外部挂单（包括没有持仓的挂单）
            self._cancel_all_external_orders_on_startup()

            # 获取币安API上的所有持仓
            existing_positions = self.position_manager.get_all_positions()

            if not existing_positions:
                logger.info("币安API上没有活跃持仓")
                return 0

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
                        continue  # 跳过当前币种，继续同步下一个

                    # 创建建仓计划（注意：使用 entry_price 而不是 entryPrice）
                    self._create_position_building_plan(position_info, pos['entry_price'], pos['side'])

                    # 添加到本地持仓管理（使用锁保护）
                    if self._positions_lock:
                        with self._positions_lock:
                            self.positions[symbol] = position_info
                    else:
                        self.positions[symbol] = position_info
                    synced_count += 1

                    # ✅ 重新设置止损单（如果建仓已完成）
                    # 注意：外部挂单已在 sync_existing_positions 开头全部取消
                    self._setup_stop_loss_for_external_position(symbol, position_info)

                    # ✅ 持久化
                    self._save_positions_to_cache()

                else:
                    logger.debug(f"持仓 {symbol} 已在本地管理中")

            if synced_count > 0:
                logger.info(f"成功同步 {synced_count} 个现有持仓到本地管理")
            else:
                logger.info("所有现有持仓已在本地管理中")

            return synced_count

        except Exception as e:
            logger.error(f"同步现有持仓失败: {str(e)}", exc_info=True)
            return 0

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
            filled_orders.sort(key=lambda o: o['time'])

            added_levels = []
            current_level = 0
            level_start_time = filled_orders[0]['time'] if filled_orders else int(time.time() * 1000)
            level_quantity = 0
            level_quote = 0

            monitor_window = settings.MONITOR_INTERVAL * 60 * 1000  # 监控周期的毫秒数

            for order in filled_orders:
                order_time = order['time']
                order_qty = float(order['executedQty'])
                order_quote = float(order.get('cummulativeQuoteQty', 0))

                if order_time - level_start_time > monitor_window:
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

                    current_level += 1
                    level_start_time = order_time
                    level_quantity = 0
                    level_quote = 0

                level_quantity += order_qty
                level_quote += order_quote

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

            # ✅ 修复：allocated_funds 应该设置为最大分配金额，而不是当前投资金额
            # 这样加仓逻辑才能正确判断是否需要继续加仓
            allocated_funds = settings.SINGLE_SYMBOL_MAX_INVESTMENT

            position_info = {
                'symbol': symbol,
                'entry_price': api_position['entry_price'],
                'current_price': self.client.get_ticker_price(symbol),
                'direction': direction,
                'total_quantity': abs(position_amt),
                'total_investment': api_margin,
                'allocated_funds': allocated_funds,  # ✅ 设置为最大分配金额
                'initial_margin': added_levels[0]['investment'] if added_levels else api_margin,
                'completed_investment': api_margin,
                'leverage': api_position['leverage'],
                'added_levels': added_levels,
                'is_closing': False,
                'last_action_time': api_position.get('updateTime', int(time.time() * 1000)),
                'status': 'active',
                'take_profit_levels': {},
                'stop_loss_levels': {},
                'take_profit_closed_pct': 0,  # 已止盈比例
                'last_take_profit_pct': None,  # 最后一次止盈时盈利百分比
                'last_take_profit_time': None,  # 最后一次止盈时间（ms）
                '_recovered_order_history': True
            }

            # ✅ 修复：为外部恢复的持仓创建加仓计划（pending_orders）
            # 这样才能继续执行加仓逻辑
            # 根据策略类型动态导入不同的position_builder
            if self.strategy_type == 'hf':
                from strategies.hf.private.position_builder import get_position_builder
            else:
                # v23策略已迁移到15mTupo
                import importlib
                module = importlib.import_module('strategies.15mTupo.private.position_builder')
                get_position_builder = module.get_position_builder
            
            if hasattr(self, 'position_manager'):
                position_builder = get_position_builder(
                    self.client, 
                    self.position_manager if hasattr(self, 'position_manager') else None
                )
                # 创建加仓计划
                pending_orders = position_builder._create_position_building_plan(
                    position_info,
                    api_position['entry_price'],
                    direction
                )
                # 过滤掉已完成的加仓层级（pending_orders 可能为 None）
                if pending_orders:
                    pending_orders = [
                        po for po in pending_orders 
                        if po.get('index', 0) > len(added_levels)
                    ]
                else:
                    pending_orders = []
                position_info['pending_orders'] = pending_orders or []
                
                if pending_orders:
                    logger.info(f"{symbol} 已为外部持仓创建{len(pending_orders)}个待执行加仓计划")
            else:
                position_info['pending_orders'] = []

            # 6. 重新计算利润
            current_price = self.client.get_ticker_price(symbol)
            if direction == 'LONG':
                position_info['profit_pct'] = ((current_price - api_position['entry_price']) / api_position['entry_price']) * api_position['leverage'] * 100
            else:
                position_info['profit_pct'] = ((api_position['entry_price'] - current_price) / api_position['entry_price']) * api_position['leverage'] * 100

            position_info['profit'] = api_position['unrealized_pnl']
            position_info['max_profit_pct'] = position_info['profit_pct']

            # ✅ 修复：使用正确的建仓完成条件判断
            # 条件：盈利超过最后一级 PROFIT_STEP / 杠杆 + POSITION_COMPLETE_PROFIT_RISE
            # 或者：亏损超过最后一级 LOSS_STEP / 杠杆 - POSITION_COMPLETE_LOSS_FALL
            leverage = api_position['leverage']
            pnl_rate = position_info['profit_pct']
            
            # 检查盈利条件
            profit_steps = [settings.PROFIT_STEP1, settings.PROFIT_STEP2, settings.PROFIT_STEP3]
            last_profit_step = 0
            for step in reversed(profit_steps):
                if step > 0:
                    last_profit_step = step
                    break
            
            position_complete = False
            complete_reason = ""
            
            if last_profit_step > 0:
                # ✅ 正确公式：pnl_rate(含杠杆) >= last_profit_step(含杠杆) + POSITION_COMPLETE_PROFIT_RISE * leverage
                profit_threshold = last_profit_step + settings.POSITION_COMPLETE_PROFIT_RISE * leverage
                if pnl_rate >= profit_threshold:
                    position_complete = True
                    complete_reason = f"盈利超过{last_profit_step}+{settings.POSITION_COMPLETE_PROFIT_RISE * leverage}%"
            
            # 检查亏损条件
            if not position_complete:
                loss_steps = [settings.LOSS_STEP1, settings.LOSS_STEP2, settings.LOSS_STEP3]
                last_loss_step = 0
                for step in reversed(loss_steps):
                    if step < 0:
                        last_loss_step = step
                        break
                
                if last_loss_step < 0:
                    # ✅ 正确公式：pnl_rate(含杠杆) <= last_loss_step(含杠杆) - POSITION_COMPLETE_LOSS_FALL * leverage
                    loss_threshold = last_loss_step - settings.POSITION_COMPLETE_LOSS_FALL * leverage
                    if pnl_rate <= loss_threshold:
                        position_complete = True
                        complete_reason = f"亏损超过{last_loss_step}-{settings.POSITION_COMPLETE_LOSS_FALL * leverage}%"
            
            # 检查是否已无待执行的加仓计划
            pending_orders = position_info.get('pending_orders', [])
            if not position_complete and len(pending_orders) == 0:
                position_complete = True
                complete_reason = "所有加仓计划已完成"
            
            position_info['position_complete'] = position_complete
            
            if position_complete:
                logger.info(f"{symbol} 建仓已完成: {complete_reason}")
            else:
                logger.info(f"{symbol} 建仓未完成: 盈利{pnl_rate:.2f}%, 剩余{len(pending_orders)}个加仓计划")

            logger.info(f"{symbol} ✅ 从订单历史成功恢复持仓: {len(added_levels)}个层级, 当前投资={api_margin:.2f}, 最大分配={allocated_funds:.2f}, 建仓完成={position_complete}")

            return position_info

        except Exception as e:
            logger.error(f"{symbol} 从订单历史恢复持仓失败: {str(e)}", exc_info=True)
            return None

    def sync_position_states_from_api(self):
        """
        从币安API同步最新的持仓状态
        处理手动平仓、外部交易等情况
        """
        try:
            api_positions = self.position_manager.get_all_positions()
            api_positions_dict = {pos['symbol']: pos for pos in api_positions}

            positions_to_remove = []
            positions_to_update = []

            for symbol, local_pos in list(self.positions.items()):
                api_pos = api_positions_dict.get(symbol)

                if api_pos is None:
                    # 币安API上没有这个持仓，说明已被平仓
                    logger.info(f"检测到持仓已平仓: {symbol} (API上不存在)")
                    positions_to_remove.append(symbol)

                    # 发送平仓通知
                    self._notify_position_closed(symbol, local_pos)

                    # 记录平仓
                    if self.trade_recorder:
                        self._record_position_closed(symbol, local_pos)

                elif abs(api_pos['position_amt']) <= 0.000001:
                    logger.info(f"检测到持仓已清空: {symbol} (API数量: {api_pos['position_amt']})")
                    positions_to_remove.append(symbol)

                else:
                    # 持仓仍然存在，检查是否需要更新
                    api_quantity = abs(api_pos['position_amt'])
                    local_quantity = local_pos['total_quantity']

                    quantity_diff = abs(api_quantity - local_quantity) / max(local_quantity, api_quantity, 0.000001)
                    if quantity_diff > 0.01:
                        logger.warning(f"检测到持仓数量变化: {symbol} 本地{local_quantity:.6f} vs API{api_quantity:.6f}")
                        positions_to_update.append((symbol, api_pos))

            # 执行移除操作（使用锁保护）
            for symbol in positions_to_remove:
                # ✅ 持仓被外部平仓时，先取消所有相关挂单
                try:
                    # 取消普通挂单
                    try:
                        open_orders = self.client.get_open_orders(symbol=symbol)
                        if open_orders:
                            for order in open_orders:
                                order_id = order.get('orderId')
                                if order_id:
                                    self.client.cancel_order(symbol, order_id)
                                    logger.info(f"{symbol} 取消平仓前残留挂单: ID={order_id}")
                    except Exception as e:
                        logger.warning(f"{symbol} 取消普通挂单失败: {str(e)}")
                    
                    # 取消Algo订单（止损单等）
                    try:
                        algo_orders = self.client.get_algo_open_orders(symbol=symbol)
                        if algo_orders:
                            for order in algo_orders:
                                algo_id = order.get('algoId')
                                if algo_id:
                                    self.client.cancel_algo_order(symbol, algo_id=algo_id)
                                    logger.info(f"{symbol} 取消平仓前残留Algo订单: ID={algo_id}")
                    except Exception as e:
                        logger.warning(f"{symbol} 取消Algo订单失败: {str(e)}")
                except Exception as e:
                    logger.warning(f"{symbol} 清理平仓前残留订单失败: {str(e)}")
                
                if self._positions_lock:
                    with self._positions_lock:
                        if symbol in self.positions:
                            del self.positions[symbol]
                else:
                    if symbol in self.positions:
                        del self.positions[symbol]
                logger.info(f"已从本地管理中移除持仓: {symbol}")

            # 执行更新操作
            for symbol, api_pos in positions_to_update:
                try:
                    # ✅ 添加锁保护
                    if self._positions_lock:
                        with self._positions_lock:
                            if symbol in self.positions:
                                old_qty = self.positions[symbol]['total_quantity']
                                self._update_position_from_api(symbol, self.positions[symbol], api_pos)
                                new_qty = self.positions[symbol]['total_quantity']
                            else:
                                logger.error(f"{symbol} 更新失败: 持仓不存在")
                                continue
                    else:
                        if symbol in self.positions:
                            old_qty = self.positions[symbol]['total_quantity']
                            self._update_position_from_api(symbol, self.positions[symbol], api_pos)
                            new_qty = self.positions[symbol]['total_quantity']
                        else:
                            logger.error(f"{symbol} 更新失败: 持仓不存在")
                            continue

                    logger.info(f"✓ 更新持仓 {symbol}: {old_qty:.6f} -> {new_qty:.6f}")
                except Exception as e:
                    logger.error(f"✗ 更新持仓失败 {symbol}: {str(e)}", exc_info=True)
                    # 尝试手动更新核心字段
                    try:
                        # ✅ 添加锁保护
                        if self._positions_lock:
                            with self._positions_lock:
                                if symbol in self.positions:
                                    self.positions[symbol]['total_quantity'] = abs(api_pos['position_amt'])
                                    self.positions[symbol]['current_price'] = self.client.get_ticker_price(symbol)
                                    logger.info(f"✓ 手动更新 {symbol}: 数量={self.positions[symbol]['total_quantity']:.6f}, 价格={self.positions[symbol]['current_price']:.6f}")
                        else:
                            if symbol in self.positions:
                                self.positions[symbol]['total_quantity'] = abs(api_pos['position_amt'])
                                self.positions[symbol]['current_price'] = self.client.get_ticker_price(symbol)
                                logger.info(f"✓ 手动更新 {symbol}: 数量={self.positions[symbol]['total_quantity']:.6f}, 价格={self.positions[symbol]['current_price']:.6f}")
                    except Exception as e2:
                        logger.error(f"✗ 手动更新也失败 {symbol}: {str(e2)}")

            # 检查是否有新的持仓（在API上但不在本地）
            for symbol, api_pos in api_positions_dict.items():
                if symbol not in self.positions and abs(api_pos['position_amt']) > 0.000001:
                    logger.info(f"检测到新的外部持仓: {symbol}，尝试同步")

                    if len(self.positions) >= self.main_config.max_positions:
                        logger.warning(f"外部持仓 {symbol} 同步失败: 持仓数量已达上限 ({self.main_config.max_positions})")
                        continue

                    # ✅ 添加锁保护：外部同步时保护 positions 字典
                    if self._positions_lock:
                        with self._positions_lock:
                            if symbol not in self.positions:
                                self._sync_external_position(symbol, api_pos)
                    else:
                        if symbol not in self.positions:
                            self._sync_external_position(symbol, api_pos)

            # ✅ 同步挂单状态（新增：处理外部挂单）
            self._sync_orders_from_api()

        except Exception as e:
            logger.error(f"同步持仓状态失败: {str(e)}", exc_info=True)

    def _sync_orders_from_api(self):
        """
        同步挂单状态：
        1. 取消外部挂单（不在本地记录中的订单）
        2. 检测本地记录的挂单是否被外部取消，如果是则重新挂单
        
        逻辑：
        1. 获取API上所有挂单
        2. 与本地持仓的pending_orders进行比较
        3. 取消所有不在本地记录中的外部挂单
        4. 检查本地记录的挂单是否在API上，如果不在说明被外部取消了，需要重新挂单
        5. 程序会在下次监控时自动重新创建需要的挂单
        """
        try:
            # 获取API上所有挂单（普通挂单和Algo订单）
            api_orders = self.client.get_open_orders()
            try:
                api_algo_orders = self.client.get_algo_open_orders()
            except Exception as e:
                logger.warning(f"获取Algo订单失败: {str(e)}")
                api_algo_orders = []
            
            # 合并所有挂单（普通挂单和Algo订单）
            all_api_orders = []
            if api_orders:
                for order in api_orders:
                    order['_order_source'] = 'normal'
                    all_api_orders.append(order)
            if api_algo_orders:
                for order in api_algo_orders:
                    order['_order_source'] = 'algo'
                    all_api_orders.append(order)
            
            if not all_api_orders:
                # 检查是否有本地挂单需要重新创建
                self._check_and_restore_local_orders()
                logger.debug("API上没有挂单")
                return

            logger.info(f"API上有 {len(all_api_orders)} 个挂单（普通: {len(api_orders) if api_orders else 0}, Algo: {len(api_algo_orders) if api_algo_orders else 0}），开始同步...")

            # 按symbol分组
            api_orders_by_symbol = {}
            for order in all_api_orders:
                symbol = order.get('symbol')
                if symbol:
                    if symbol not in api_orders_by_symbol:
                        api_orders_by_symbol[symbol] = []
                    api_orders_by_symbol[symbol].append(order)

            # 遍历本地持仓，处理每个持仓的挂单
            cancelled_count = 0
            restored_count = 0
            
            for symbol, position_info in list(self.positions.items()):
                # 1. 取消外部挂单
                if symbol in api_orders_by_symbol:
                    cancelled_for_symbol = self._cancel_external_orders(symbol, position_info, api_orders_by_symbol[symbol])
                    cancelled_count += cancelled_for_symbol
                
                # 2. 检测本地挂单是否被外部取消，如果是则重新挂单
                restored = self._check_and_restore_local_orders_for_symbol(symbol, position_info)
                restored_count += restored

            if cancelled_count > 0:
                logger.info(f"✓ 已取消 {cancelled_count} 个外部挂单，程序将重新创建挂单")
            
            if restored_count > 0:
                logger.info(f"✓ 已重新创建 {restored_count} 个被的本地取消挂单")

        except Exception as e:
            logger.error(f"同步挂单状态失败: {str(e)}", exc_info=True)

    def _check_and_restore_local_orders(self):
        """
        检查所有本地持仓的挂单，如果有被外部取消的则重新挂单
        """
        restored_count = 0
        for symbol, position_info in list(self.positions.items()):
            restored = self._check_and_restore_local_orders_for_symbol(symbol, position_info)
            restored_count += restored
        
        return restored_count

    def _check_and_restore_local_orders_for_symbol(self, symbol: str, position_info: Dict) -> int:
        """
        检查指定币种的本地挂单是否被外部取消，如果是则重新挂单

        Args:
            symbol: 币种符号
            position_info: 本地持仓信息

        Returns:
            int: 重新创建的挂单数量
        """
        try:
            restored_count = 0
            pending_orders = position_info.get('pending_orders', [])
            
            if not pending_orders:
                return 0
            
            # 获取API上该币种的所有挂单
            try:
                api_orders = self.client.get_open_orders(symbol=symbol)
            except Exception as e:
                logger.warning(f"{symbol} 获取API挂单失败: {str(e)}")
                api_orders = []
            
            try:
                api_algo_orders = self.client.get_algo_open_orders(symbol=symbol)
            except Exception as e:
                logger.warning(f"{symbol} 获取Algo订单失败: {str(e)}")
                api_algo_orders = []
            
            # 收集API上的所有订单ID
            api_order_ids = set()
            if api_orders:
                for order in api_orders:
                    order_id = str(order.get('orderId', ''))
                    if order_id:
                        api_order_ids.add(order_id)
            if api_algo_orders:
                for order in api_algo_orders:
                    algo_id = str(order.get('algoId', ''))
                    if algo_id:
                        api_order_ids.add(algo_id)
                    else:
                        # Algo订单缺少algoId可能是数据结构问题，不输出错误日志
                        logger.debug(f"{symbol} API Algo订单缺少algoId，跳过")
            
            # ✅ 修复缩进错误：检查每个本地挂单应该在 if api_algo_orders 块外面
            # 无论是否有Algo订单，都需要检查本地挂单状态
            for pending_order in pending_orders:
                order_id = pending_order.get('order_id')
                status = pending_order.get('status', '')
                
                # 只检查已提交但未成交的订单
                if not order_id or status != 'submitted':
                    continue
                
                order_id_str = str(order_id)
                
                # 如果订单ID不在API上，说明被外部取消了，需要重新挂单
                if order_id_str not in api_order_ids:
                    logger.warning(f"{symbol} 检测到本地挂单被外部取消: ID={order_id_str}, "
                                 f"类型={pending_order.get('reason', 'unknown')}，准备重新挂单")
                    
                    try:
                        # 重新挂单
                        result = self._recreate_pending_order(symbol, position_info, pending_order)
                        if result:
                            restored_count += 1
                            logger.info(f"{symbol} 成功重新创建挂单: ID={order_id_str}")
                        else:
                            # 重新挂单失败，清除本地记录
                            pending_order['order_id'] = None
                            pending_order['status'] = 'pending'
                            logger.warning(f"{symbol} 重新挂单失败，清除本地记录: ID={order_id_str}")
                    except Exception as e:
                        logger.error(f"{symbol} 重新挂单异常: ID={order_id_str}, {str(e)}")
                        # 清除本地记录，让下次监控重新处理
                        pending_order['order_id'] = None
                        pending_order['status'] = 'pending'
            
                # 检查止损单是否被外部取消（Algo订单）
                # 如果持仓已完成建仓，应该有止损单
                if position_info.get('position_complete', False):
                    # 获取本地记录的止损单ID
                    local_stop_loss_ids = []
                    if '_stop_loss_algo_id1' in position_info:
                        algo_id = position_info.get('_stop_loss_algo_id1')
                        if algo_id:
                            local_stop_loss_ids.append(algo_id)
                    if '_stop_loss_algo_id2' in position_info:
                        algo_id = position_info.get('_stop_loss_algo_id2')
                        if algo_id:
                            local_stop_loss_ids.append(algo_id)
                    
                    # 获取API上的Algo订单
                    try:
                        api_algo_orders = self.client.get_algo_open_orders(symbol=symbol)
                    except Exception as e:
                        logger.warning(f"{symbol} 获取Algo订单失败: {str(e)}")
                        api_algo_orders = []
                    
                    api_algo_ids = {str(order.get('algoId', '')) for order in api_algo_orders if order.get('algoId')}
                    
                    # 检查每个止损单是否还在API上
                    for algo_id in local_stop_loss_ids:
                        algo_id_str = str(algo_id)
                        if algo_id_str and algo_id_str not in api_algo_ids:
                            logger.warning(f"{symbol} 检测到止损单被外部取消: ID={algo_id_str}，清除本地记录，等待下次监控自动重新设置")
                            # 清除本地记录，让下次监控自动处理
                            if '_stop_loss_algo_id1' in position_info and position_info.get('_stop_loss_algo_id1') == algo_id:
                                position_info['_stop_loss_algo_id1'] = None
                            if '_stop_loss_algo_id2' in position_info and position_info.get('_stop_loss_algo_id2') == algo_id:
                                position_info['_stop_loss_algo_id2'] = None
                            restored_count += 1
            
            return restored_count

        except Exception as e:
            logger.error(f"{symbol} 检测并重新创建挂单失败: {str(e)}", exc_info=True)
            return 0

    def _recreate_pending_order(self, symbol: str, position_info: Dict, pending_order: Dict) -> bool:
        """
        重新创建挂单

        由于加仓逻辑较为复杂，这里采用简化方案：
        清除本地挂单记录，让持仓监控逻辑在下次监控时自动检测并重新挂单

        Args:
            symbol: 币种符号
            position_info: 持仓信息
            pending_order: 挂单信息

        Returns:
            bool: 是否成功处理（这里返回True表示已清除记录，等待下次监控处理）
        """
        try:
            # 清除本地挂单记录，状态改为pending
            # 同时清除 attempted 字段，否则 execute_position_building 会跳过这个订单
            pending_order['order_id'] = None
            pending_order['status'] = 'pending'
            pending_order['attempted'] = False  # 清除尝试标记，让下次监控可以重新尝试
            
            logger.info(f"{symbol} 已清除被取消的挂单记录（级别{pending_order.get('index', '?')}），等待下次监控自动重新挂单")
            
            # 返回True表示已处理（清除记录）
            return True
            
        except Exception as e:
            logger.error(f"{symbol} 处理被取消的挂单失败: {str(e)}", exc_info=True)
            return False

    def _cancel_external_orders(self, symbol: str, position_info: Dict, api_orders: List[Dict]) -> int:
        """
        取消外部挂单（不在本地记录中的订单）

        Args:
            symbol: 币种符号
            position_info: 本地持仓信息
            api_orders: API上的挂单列表

        Returns:
            int: 取消的订单数量
        """
        try:
            pending_orders = position_info.get('pending_orders', [])

            # 收集本地记录的订单ID
            local_order_ids = set()
            for po in pending_orders:
                order_id = po.get('order_id')
                if order_id:
                    local_order_ids.add(str(order_id))

            logger.debug(f"{symbol} 本地记录的订单ID: {local_order_ids}")

            # 取消不在本地记录中的订单
            cancelled_count = 0
            for api_order in api_orders:
                # ✅ 添加None保护 - 检查api_order是否为None
                if api_order is None:
                    logger.warning(f"{symbol} API订单为None，跳过")
                    continue

                api_order_id = str(api_order.get('orderId', ''))
                if not api_order_id:
                    logger.warning(f"{symbol} API订单缺少orderId，跳过")
                    continue

                api_type = api_order.get('type')
                api_price = api_order.get('price')
                api_qty = api_order.get('origQty')

                # 如果订单不在本地记录中，认为是外部挂单
                if api_order_id not in local_order_ids:
                    logger.warning(f"{symbol} 检测到外部挂单，准备取消: ID={api_order_id}, "
                                f"类型={api_type}, 价格={api_price}, 数量={api_qty}")

                    try:
                        # 取消订单
                        self.client.cancel_order(symbol, api_order_id)
                        logger.info(f"✓ {symbol} 已取消外部挂单: {api_order_id}")
                        cancelled_count += 1

                    except Exception as e:
                        logger.error(f"✗ {symbol} 取消外部挂单失败 {api_order_id}: {str(e)}")

            return cancelled_count

        except Exception as e:
            logger.error(f"{symbol} 取消外部挂单失败: {str(e)}", exc_info=True)
            return 0

    def _is_order_valid_for_logic(self, symbol: str, api_order: Dict, position_info: Dict) -> bool:
        """
        智能判断API挂单是否符合程序逻辑（已弃用，现在直接取消所有外部挂单）

        保留此方法以备将来需要更精细的订单验证逻辑

        Args:
            symbol: 币种符号
            api_order: API订单信息
            position_info: 本地持仓信息

        Returns:
            bool: 始终返回False，因为现在直接取消所有外部挂单
        """
        return False

    def _sync_position_state(self, symbol: str, position_info: Dict, all_positions: List[Dict]):
        """
        同步单个持仓的状态（预留接口）

        Args:
            symbol: 币种符号
            position_info: 本地持仓信息
            all_positions: 所有API持仓列表
        """
        # 占位方法，可在需要时实现单个持仓的精确同步
        pass

    def _save_positions_to_cache(self):
        """
        持久化持仓数据到文件
        """
        try:
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(self.positions, f, indent=2, ensure_ascii=False)
            logger.debug("持仓已持久化")
        except Exception as e:
            logger.warning(f"持���化失败: {str(e)}")

    def _load_positions_from_cache(self) -> Optional[Dict]:
        """
        从文件加载持仓数据

        Returns:
            Dict: 持仓数据，如果加载失败则返回None
        """
        try:
            if not self.positions_file.exists():
                return None

            with open(self.positions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"从缓存加载持仓失败: {str(e)}")
            return None

    def _create_position_building_plan(self, position_info: Dict, entry_price: float, direction: str):
        """
        创建6级建仓计划（限价单价格动态计算）

        Args:
            position_info: 持仓信息
            entry_price: 开仓价格
            direction: 方向
        """
        symbol = position_info.get('symbol', 'Unknown')
        try:
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
                # 【修复】跳过加仓比例为0的级别（不需要创建无效订单）
                if add_percent <= 0:
                    logger.info(f"{symbol} 建仓计划{idx}: 跳过（加仓比例={add_percent}%≤0）")
                    continue

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
            
            # 【关键修复】标记为建仓中（因为还有待加仓计划）
            position_info['position_complete'] = False
            
            logger.info(f"{symbol} 建仓计划创建完成，共{len(pending_orders)}级已添加到持仓信息")
            logger.info(f"{symbol} 当前处于建仓阶段 (position_complete=False)")

        except Exception as e:
            logger.error(f"{symbol} 创建建仓计划失败: {str(e)}")
            position_info['pending_orders'] = []
            # 失败时假设建仓已完成（避免无限循环）
            position_info['position_complete'] = True

    def _notify_position_closed(self, symbol: str, position_info: Dict):
        """
        通知持仓已平仓（发送Telegram消息）

        Args:
            symbol: 币种符号
            position_info: 持仓信息
        """
        if not self.telegram_bot:
            logger.debug(f"{symbol} 无Telegram机器人，跳过平仓通知")
            return

        try:
            # 总投资和持仓比例
            current_investment = position_info.get('total_investment', 0)
            total_planned_investment = position_info.get('allocated_funds', settings.SINGLE_SYMBOL_MAX_INVESTMENT)

            # 构建交易数据
            initial_margin = position_info.get('initial_margin', 0)
            position_ratio = (current_investment / initial_margin * 100) if initial_margin > 0 else 0

            trade_data = {
                'leverage': position_info.get('leverage', 1),
                'side': position_info.get('direction', 'LONG'),
                'status': '平仓',
                'position_ratio': position_ratio,
                'avg_entry_price': position_info.get('entry_price', 0),
                'current_price': position_info.get('current_price', 0),
                'position_usdt': current_investment,
                'margin': current_investment,
                'pnl': position_info.get('profit', 0),
                'pnl_percent': position_info.get('profit_pct', 0)
            }

            # 发送到交易bot
            self.telegram_bot.send_trade_message(symbol, trade_data)
            logger.info(f"{symbol} 平仓通知已发送")
        except Exception as e:
            logger.warning(f"{symbol} 发送平仓通知失败: {str(e)}")

    def _record_position_closed(self, symbol: str, position_info: Dict):
        """
        记录平仓交易

        Args:
            symbol: 币种符号
            position_info: 持仓信息
        """
        if not self.trade_recorder:
            logger.debug(f"{symbol} 无交易记录器，跳过平仓记录")
            return

        try:
            close_price = position_info.get('current_price', 0)
            reason = position_info.get('close_reason', '平仓')
            self.trade_recorder.record_close(symbol, close_price, reason)
            logger.info(f"{symbol} 平仓交易已记录")
        except Exception as e:
            logger.warning(f"{symbol} 记录平仓交易失败: {str(e)}")

    def _update_position_from_api(self, symbol: str, local_pos: Dict, api_pos: Dict):
        """
        从API更新本地持仓信息

        Args:
            symbol: 币种符号
            local_pos: 本地持仓信息
            api_pos: API持仓信息
        """
        old_quantity = local_pos['total_quantity']
        new_quantity = abs(api_pos['position_amt'])

        local_pos['total_quantity'] = new_quantity
        local_pos['entry_price'] = api_pos['entry_price']

        notional_value = new_quantity * api_pos['entry_price']
        local_pos['total_investment'] = notional_value / api_pos['leverage']
        local_pos['initial_margin'] = local_pos['total_investment']

        current_price = self.client.get_ticker_price(symbol)
        if api_pos['side'] == 'LONG':
            local_pos['profit_pct'] = ((current_price - api_pos['entry_price']) / api_pos['entry_price']) * api_pos['leverage'] * 100
        else:
            local_pos['profit_pct'] = ((api_pos['entry_price'] - current_price) / api_pos['entry_price']) * api_pos['leverage'] * 100

        local_pos['profit'] = local_pos['total_investment'] * local_pos['profit_pct'] / 100
        local_pos['current_price'] = current_price

        logger.info(f"已更新持仓状态: {symbol} 数量 {old_quantity:.6f} -> {new_quantity:.6f}")

    def _sync_external_position(self, symbol: str, api_pos: Dict):
        """
        同步外部持仓到本地（完整版，与启动时同步逻辑一致）n
        运行时检测到的新外部持仓，使用与启动时相同的完整恢复逻辑，n        包括：订单历史分析、建仓计划、止盈止损设置。

        Args:
            symbol: 币种符号
            api_pos: API持仓信息
        """
        logger.info(f"🔄 开始完整同步外部持仓: {symbol}")

        try:
            # 【关键修复】使用与启动时相同的恢复逻辑：从订单历史完整恢复
            position_info = self._recover_position_from_order_history(symbol, api_pos)

            if position_info is None:
                logger.warning(f"{symbol} 从订单历史恢复失败，使用基础信息同步")
                # 如果恢复失败，使用简化的基础信息（保底方案）
                position_info = self._create_basic_position_info(symbol, api_pos)

            # 【关键修复】创建建仓计划（如果持仓还在建仓阶段）
            if not position_info.get('position_complete', False):
                logger.info(f"{symbol} 持仓处于建仓阶段，创建建仓计划...")
                self._create_position_building_plan(position_info, api_pos['entry_price'], api_pos['side'])

            # 【关键修复】设置止盈水平（基于当前盈亏状态）
            self._initialize_take_profit_levels(symbol, position_info)

            # 添加到本地持仓管理（使用锁保护）
            if self._positions_lock:
                with self._positions_lock:
                    self.positions[symbol] = position_info
                    synced_count = len([p for p in self.positions.values() if p.get('_recovered_from_external')])
            else:
                self.positions[symbol] = position_info
                synced_count = len([p for p in self.positions.values() if p.get('_recovered_from_external')])

            logger.info(f"✅ 外部持仓 {symbol} 已完整同步到本地管理:")
            logger.info(f"  方向: {position_info['direction']}")
            logger.info(f"  数量: {position_info['total_quantity']}")
            logger.info(f"  入场价: {position_info['entry_price']}")
            logger.info(f"  杠杆: {position_info['leverage']}x")
            logger.info(f"  盈亏率: {position_info['profit_pct']:.2f}%")
            logger.info(f"  加仓层级: {len(position_info.get('added_levels', []))}级")
            logger.info(f"  建仓计划: {'已创建' if position_info.get('pending_orders') else '无'}")
            logger.info(f"  止盈设置: {'已初始化' if position_info.get('take_profit_levels') else '未设置'}")

            # 持久化
            self._save_positions_to_cache()

            # 【关键修复】取消外部挂单（止损单、止盈单等）
            self._cancel_all_orders_for_symbol(symbol)

            # 【关键修复】创建止损单（最后一步）
            self._setup_stop_loss_for_external_position(symbol, position_info)

            logger.info(f"🎉 外部持仓 {symbol} 完整同步完成，已按正常持仓管理")

        except Exception as e:
            logger.error(f"❌ 同步外部持仓 {symbol} 失败: {str(e)}", exc_info=True)
            # 失败时不阻止程序运行，但记录错误

    def _create_basic_position_info(self, symbol: str, api_pos: Dict) -> Dict:
        """
        创建基础持仓信息（当订单历史恢复失败时的保底方案）

        Args:
            symbol: 币种符号
            api_pos: API持仓信息

        Returns:
            Dict: 基础持仓信息
        """
        notional_value = abs(api_pos['position_amt']) * api_pos['entry_price']

        margin = notional_value / api_pos['leverage']
        return {
            'symbol': symbol,
            'entry_price': api_pos['entry_price'],
            'current_price': api_pos['entry_price'],
            'direction': api_pos['side'],
            'total_quantity': abs(api_pos['position_amt']),
            'total_investment': margin,
            'allocated_funds': margin,  # FIX: 设置为实际投资金额
            'initial_margin': margin,
            'completed_investment': margin,
            'profit': api_pos['unrealized_pnl'],
            'profit_pct': 0.0,
            'max_profit_pct': 0.0,
            'leverage': api_pos['leverage'],
            'added_levels': [],
            'pending_orders': [],
            'is_closing': False,
            'last_action_time': int(time.time()),
            'status': 'active',
            'take_profit_levels': {},
            'stop_loss_levels': {},
            'position_complete': False,  # 【修复】默认为建仓中（因为无法判断真实状态）
            '_recovered_from_external': True
        }

    def _initialize_take_profit_levels(self, symbol: str, position_info: Dict):
        """
        初始化止盈水平

        根据当前盈亏状态初始化止盈跟踪
        """
        try:
            pnl_rate = position_info.get('profit_pct', 0)

            # 初始化止盈层级跟踪
            position_info['take_profit_levels'] = {
                'high_profit_tp1_triggered': False,
                'high_profit_tp2_triggered': False,
                'low_profit_tp1_triggered': False,
                'breakeven_triggered': False,
                'max_profit_pct': pnl_rate,
                'last_take_profit_pct': None,
                'last_take_profit_time': None
            }

            logger.info(f"{symbol} 止盈水平已初始化，当前盈亏率: {pnl_rate:.2f}%")

        except Exception as e:
            logger.warning(f"{symbol} 初始化止盈水平失败: {str(e)}")

    def _cancel_all_external_orders_on_startup(self) -> int:
        """
        启动时取消所有外部挂单
        
        在同步持仓之前，先取消所有挂单（包括有持仓和没有持仓的挂单），
        然后由程序根据持仓状态重新创建需要的挂单（止损单、止盈单等）。
        
        注意：同时取消普通挂单和Algo订单（条件委托订单）
        
        Returns:
            int: 取消的订单数量
        """
        try:
            cancelled_count = 0
            
            # 1. 获取并取消所有 Algo 订单（条件委托订单，如止损单、止盈单）
            try:
                algo_orders = self.client.get_algo_open_orders()
                if algo_orders:
                    logger.info(f"🔄 启动时检测到 {len(algo_orders)} 个Algo订单，准备取消...")
                    for order in algo_orders:
                        if order is None:
                            continue
                        symbol = order.get('symbol', 'UNKNOWN')
                        algo_id = order.get('algoId')
                        order_type = order.get('orderType', 'UNKNOWN')
                        
                        try:
                            if algo_id:
                                self.client.cancel_algo_order(symbol, algo_id=algo_id)
                                logger.info(f"✓ 已取消Algo订单: {symbol} algoId={algo_id}, 类型={order_type}")
                                cancelled_count += 1
                        except Exception as e:
                            logger.warning(f"✗ 取消Algo订单失败: {symbol} algoId={algo_id}, {str(e)}")
            except Exception as e:
                logger.warning(f"获取Algo订单失败: {str(e)}")
            
            # 2. 获取并取消所有普通挂单
            try:
                api_orders = self.client.get_open_orders()
                if api_orders:
                    logger.info(f"🔄 启动时检测到 {len(api_orders)} 个普通挂单，准备取消...")
                    for order in api_orders:
                        if order is None:
                            continue
                            
                        symbol = order.get('symbol', 'UNKNOWN')
                        order_id = order.get('orderId')
                        order_type = order.get('type', 'UNKNOWN')
                        order_price = order.get('price', 0)
                        
                        try:
                            if order_id:
                                self.client.cancel_order(symbol, order_id)
                                logger.info(f"✓ 已取消普通挂单: {symbol} ID={order_id}, 类型={order_type}")
                                cancelled_count += 1
                        except Exception as e:
                            logger.warning(f"✗ 取消普通挂单失败: {symbol} ID={order_id}, {str(e)}")
            except Exception as e:
                logger.warning(f"获取普通挂单失败: {str(e)}")
            
            if cancelled_count > 0:
                logger.info(f"✅ 启动时已取消 {cancelled_count} 个外部挂单，程序将根据持仓状态重新设置")
            else:
                logger.info("启动时检查：没有检测到外部挂单")
            
            return cancelled_count
            
        except Exception as e:
            logger.error(f"启动时取消外部挂单失败: {str(e)}")
            return 0

    def _cancel_all_orders_for_symbol(self, symbol: str) -> int:
        """
        取消指定币种的所有挂单（用于外部持仓同步）
        
        当检测到外部持仓时，取消该持仓相关的所有挂单（包括止损单、止盈单等），
        由程序根据当前持仓状态重新设置。
        
        注意：同时取消Algo订单和普通挂单
        
        Args:
            symbol: 币种符号
            
        Returns:
            int: 取消的订单数量
        """
        try:
            cancelled_count = 0
            
            # 1. 取消该币种的所有 Algo 订单（条件委托订单）
            try:
                algo_orders = self.client.get_algo_open_orders(symbol=symbol)
                if algo_orders:
                    logger.info(f"{symbol} 检测到 {len(algo_orders)} 个Algo订单，准备取消...")
                    for order in algo_orders:
                        if order is None:
                            continue
                        algo_id = order.get('algoId')
                        order_type = order.get('orderType', 'UNKNOWN')
                        
                        try:
                            if algo_id:
                                self.client.cancel_algo_order(symbol, algo_id=algo_id)
                                logger.info(f"{symbol} 已取消Algo订单: algoId={algo_id}, 类型={order_type}")
                                cancelled_count += 1
                        except Exception as e:
                            logger.warning(f"{symbol} 取消Algo订单失败: algoId={algo_id}, {str(e)}")
            except Exception as e:
                logger.warning(f"{symbol} 获取Algo订单失败: {str(e)}")
            
            # 2. 取消该币种的所有普通挂单
            try:
                api_orders = self.client.get_open_orders(symbol=symbol)
                if api_orders:
                    logger.info(f"{symbol} 检测到 {len(api_orders)} 个普通挂单，准备取消...")
                    for order in api_orders:
                        if order is None:
                            continue
                            
                        order_id = order.get('orderId')
                        order_type = order.get('type', 'UNKNOWN')
                        order_price = order.get('price', 0)
                        
                        try:
                            self.client.cancel_order(symbol, order_id)
                            logger.info(f"{symbol} 已取消普通挂单: ID={order_id}, 类型={order_type}, 价格={order_price}")
                            cancelled_count += 1
                        except Exception as e:
                            logger.warning(f"{symbol} 取消普通挂单失败: ID={order_id}, {str(e)}")
            except Exception as e:
                logger.warning(f"{symbol} 获取普通挂单失败: {str(e)}")
            
            if cancelled_count > 0:
                logger.info(f"✓ {symbol} 已取消 {cancelled_count} 个外部挂单，程序将重新设置")
            
            return cancelled_count
            
        except Exception as e:
            logger.error(f"{symbol} 取消外部挂单失败: {str(e)}")
            return 0

    def _setup_stop_loss_for_external_position(self, symbol: str, position_info: Dict):
        """
        为外部持仓设置止损单（只在建仓完成后设置）
        
        【关键修复】止损单应该在建仓完成后才创建
        - 建仓中（position_complete=False）：不创建止损单，避免干扰建仓过程
        - 建仓完成（position_complete=True）：才创建止损单保护持仓
        """
        try:
            # 【关键修复】只在建仓完成后创建止损单
            if not position_info.get('position_complete', False):
                logger.info(f"{symbol} 持仓处于建仓阶段，暂不设置止损单（建仓完成后会自动设置）")
                return

            entry_price = position_info['entry_price']
            direction = position_info['direction']
            leverage = position_info['leverage']

            logger.info(f"{symbol} 建仓已完成，正在设置止损单...")

            # 计算止损价格
            stop_price = self._calculate_stop_price(entry_price, self.main_config.stoploss_trigger1, direction, leverage)

            # 【关键修复】精度调整
            stop_price = self.client.adjust_price_precision(symbol, stop_price)
            quantity = self.client.adjust_quantity_precision(symbol, position_info['total_quantity'])

            # 确定订单方向和 positionSide
            side = 'SELL' if direction == 'LONG' else 'BUY'
            position_side = direction.upper()  # HEDGING 模式需要：LONG/SHORT

            logger.info(f"{symbol} 创建止损单: 方向={side}, positionSide={position_side}, 触发价={stop_price}, 数量={quantity}")

            # 【关键修复】使用 create_algo_order 创建算法订单（限价止损单）
            # 不传递 reduce_only 参数，让 create_algo_order 内部根据 HEDGING/ONE-WAY 模式自动处理
            # 限价与触发价的偏移比例（使用 DELAY_RATIO）
            from framework.core.config import ConfigManager  # 保持兼容性
            delay_ratio = getattr(Settings, 'DELAY_RATIO', 0.003)  # 默认0.3%（已是小数形式）
            if direction == 'LONG':
                limit_price = stop_price * (1 - delay_ratio)
            else:
                limit_price = stop_price * (1 + delay_ratio)
            
            # ===== 检查并取消币安上已有的止损单 =====
            try:
                open_orders = self.client.get_algo_open_orders(symbol)
                for order in open_orders:
                    algo_id = order.get('algoId')
                    if algo_id:
                        try:
                            self.client.cancel_algo_order(symbol, algo_id=algo_id)
                            logger.info(f"已取消币安上的旧止损单: {symbol} algoId={algo_id}")
                        except Exception as e:
                            logger.debug(f"取消止损单失败（可能已成交）: {e}")
            except Exception as e:
                logger.debug(f"检查币安挂单失败: {e}")
            
            algo_order = self.client.create_algo_order(
                symbol=symbol,
                side=side,
                trigger_price=stop_price,
                quantity=quantity,
                order_type='STOP',
                price=limit_price,
                working_type='CONTRACT_PRICE',
                position_side=position_side
            )

            if algo_order and algo_order.get('algoId'):
                position_info['stop_loss_order_id'] = algo_order['algoId']
                logger.info(f"{symbol} ✅ 止损单已创建: 触发价={stop_price}, AlgoID={algo_order['algoId']}")
            else:
                logger.warning(f"{symbol} ⚠️ 止损单创建失败，无算法订单ID返回")

        except Exception as e:
            logger.error(f"{symbol} ❌ 创建止损单失败: {str(e)}")
            # 不阻止同步流程，但记录错误
    
    def _calculate_stop_price(self, entry_price: float, trigger_rate: float, direction: str, leverage: int) -> float:
        """
        计算止损价格
        """
        if direction == 'LONG':
            # 多头：价格下跌触发
            stop_price = entry_price * (1 + trigger_rate / 100 / leverage)
        else:  # SHORT
            # 空头：价格上涨触发
            stop_price = entry_price * (1 - trigger_rate / 100 / leverage)
        
        return stop_price


def get_position_sync_manager(position_manager, client, trade_recorder=None, telegram_bot=None, strategy_type='15mTupo') -> PositionSyncManager:
    """
    获取持仓同步管理器实例（单例模式）

    Args:
        position_manager: 持仓管理器实例
        client: 币安客户端实例
        trade_recorder: 交易记录器实例（可选）
        telegram_bot: Telegram机器人实例（可选）
        strategy_type: 策略类型 ('v23' 或 'hf')

    Returns:
        PositionSyncManager: 持仓同步管理器实例
    """
    return PositionSyncManager(position_manager, client, trade_recorder, telegram_bot, strategy_type)