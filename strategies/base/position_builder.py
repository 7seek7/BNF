"""
建仓策略模块 - 基类 (Position Builder Base)

提供 HF 和 15mTupo 共享的仓位构建框架。
子类可覆盖:
- close_position(): 添加 coin lock 释放
- _execute_open_position(): 添加 coin lock 生命周期管理
- _create_position_info(): 添加策略特有字段
"""

import time
import math
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING
from framework.core.config import get_main_config
from config.settings import Settings

if TYPE_CHECKING:
    from framework.business.exchange_client import ExchangeClient
    from framework.business.position_manager import PositionManager
from utils.logger import Logger
from framework.shared.strategy_dispatcher import get_strategy_dispatcher, StrategyType, StrategyDispatcher
from binance.exceptions import BinanceAPIException
from framework.business.trade_recorder import TradeRecorder

logger = Logger.get_logger('position_builder')
main_config = get_main_config()
settings = Settings()


class BasePositionBuilder:
    """
    建仓策略管理器基类

    负责处理开仓、加仓、平仓等所有仓位操作
    """

    def __init__(self, client: 'ExchangeClient', position_manager: 'PositionManager'):
        self.client = client
        self.position_manager = position_manager
        self.positions: Dict[str, Dict] = {}
        self.trade_recorder = TradeRecorder.get_instance()

        self._balance_cache = None
        self._balance_cache_time = 0

        if position_manager and hasattr(position_manager, '_positions'):
            self.positions = position_manager._positions
            logger.info(f"建仓策略管理器已绑定 position_manager，持仓计数: {len(self.positions)}")
        else:
            logger.warning("position_manager没有_positions属性，持仓计数可能不准确")

        logger.info("建仓策略管理器已初始化")

    def set_positions_ref(self, positions: Dict[str, Dict]):
        self.positions = positions

    def set_trader_ref(self, trader):
        self.trader = trader

    def open_position(self, alert_data: Dict[str, Any]) -> bool:
        symbol = alert_data['symbol']
        direction = alert_data['direction']
        hour_direction = alert_data.get('hour_direction')

        logger.info(f"=== 开始开仓流程 === 币种: {symbol}, 方向: {direction}")

        try:
            if hour_direction and direction != hour_direction:
                logger.warning(f"{symbol} 警报方向({direction})与2小时趋势方向({hour_direction})不一致，跳过开仓")
                return False

            if symbol in self.positions:
                logger.warning(f"{symbol} 已有持仓，跳过开仓")
                return False

            program_positions = {
                k: v for k, v in self.positions.items()
                if getattr(v, 'strategy', '') not in ('external', 'External', 'EXTERNAL')
            }
            program_count = len(program_positions)
            if program_count >= main_config.max_positions:
                external_count = len(self.positions) - program_count
                logger.warning(f"程序持仓数量已达上限 ({main_config.max_positions})，跳过开仓（外部持仓{external_count}个不计入限制）")
                logger.warning(f"程序持仓列表: {list(program_positions.keys())}")
                return False

            success = self._execute_open_position(alert_data)

            if success:
                logger.info(f"=== 开仓流程完成 === 币种: {symbol}")
            else:
                logger.warning(f"=== 开仓流程失败 === 币种: {symbol}")

            return success

        except Exception as e:
            logger.error(f"{symbol} 开仓失败: {str(e)}", exc_info=True)
            return False

    def close_position(self, symbol: str, reason: str = "手动平仓", close_pct: float = 100.0) -> bool:
        if symbol not in self.positions:
            logger.warning(f"持仓 {symbol} 不存在，无法平仓")
            return False

        position_info = self.positions[symbol]
        logger.info(f"关闭持仓: {symbol}, 原因: {reason}, 比例: {close_pct}%")

        try:
            success = self._execute_close_position(symbol, position_info, close_pct, reason)
            return success
        except Exception as e:
            logger.error(f"关闭持仓失败 {symbol}: {str(e)}", exc_info=True)
            return False

    def close_all_positions(self, reason: str = "紧急平仓") -> int:
        if not self.positions:
            logger.info("没有持仓需要关闭")
            return 0

        logger.info(f"关闭所有持仓: 原因: {reason}, 共 {len(self.positions)} 个")
        closed_count = 0

        for symbol in list(self.positions.keys()):
            if self.close_position(symbol, reason):
                closed_count += 1

        logger.info(f"关闭所有持仓完成: 成功 {closed_count}/{len(self.positions)}")
        return closed_count

    def add_to_position(self, symbol: str, current_price: float, pnl_rate: float) -> bool:
        if symbol not in self.positions:
            logger.warning(f"持仓 {symbol} 不存在，无法加仓")
            return False

        position_info = self.positions[symbol]
        logger.info(f"加仓: {symbol}, 当前价格: {current_price}, 盈利率: {pnl_rate:.2f}%")

        try:
            success = self._execute_add_to_position(symbol, position_info, current_price, pnl_rate)
            return success
        except Exception as e:
            logger.error(f"加仓失败 {symbol}: {str(e)}", exc_info=True)
            return False

    def _execute_open_position(self, alert_data: Dict[str, Any]) -> bool:
        symbol = alert_data['symbol']
        direction = alert_data['direction']
        hour_direction = alert_data.get('hour_direction')

        logger.info(f"=== 开始开仓流程 === 币种: {symbol}, 方向: {direction}")

        try:
            strategy_name = alert_data.get('strategy', '15mTupo')
            strategy_type = StrategyDispatcher.resolve_strategy(strategy_name)

            dispatcher = get_strategy_dispatcher()
            if not dispatcher.try_acquire_coin(symbol, strategy_type):
                logger.warning(f"{symbol} 已被其他策略持有，跳过开仓")
                return False

            coin_acquired = True
            logger.info(f"{symbol} 币种锁获取成功: {strategy_name}")

            existing_positions = self.position_manager.get_all_positions()
            has_existing = any(pos['symbol'] == symbol for pos in existing_positions)
            if has_existing:
                logger.warning(f"{symbol} 币安API上已存在持仓，跳过开仓（防止重复）")
                dispatcher.release_coin(symbol, strategy_type)
                return False

            account_info = self.client.get_account_balance()
            available_funds = account_info.get('available_balance', 0)
            total_balance = account_info.get('total_balance', 0)
            current_price = self.client.get_ticker_price(symbol)

            logger.info(f"{symbol} 账户资金: 总余额={total_balance:.2f} USDT, "
                       f"可用余额={available_funds:.2f} USDT, 模式={self.client.mode}")

            min_order_info = self.client.get_min_order_info(symbol, main_config.leverage)
            min_margin_required = min_order_info.get('min_margin_for_notional', 0.25)
            min_notional = min_order_info.get('min_notional', 5.0)

            logger.info(f"{symbol} 最小下单要求: 保证金>={min_margin_required:.2f} USDT, "
                       f"名义价值>={min_notional:.2f} USDT (杠杆{main_config.leverage}x)")

            current_positions_count = len(self.positions)
            max_slots = main_config.max_positions - current_positions_count

            if max_slots <= 0:
                logger.warning(f"没有可用持仓额度")
                dispatcher.release_coin(symbol, strategy_type)
                return False

            if available_funds < min_margin_required:
                logger.error(f"【开仓失败】{symbol} 资金不足！\n"
                            f"  可用资金: {available_funds:.2f} USDT\n"
                            f"  当前持仓: {current_positions_count}/{main_config.max_positions}\n"
                            f"  币种最小保证金要求: {min_margin_required:.2f} USDT\n"
                            f"  实际可用: {available_funds:.2f} USDT\n"
                            f"  建议: 充值更多资金")
                dispatcher.release_coin(symbol, strategy_type)
                return False

            min_symbol_amount_needed = min_margin_required / (main_config.initial_position / 100)

            optimal_slots = max_slots
            while optimal_slots > 1 and (available_funds / optimal_slots) < min_symbol_amount_needed:
                logger.info(f"{symbol} 资金不足以支持{optimal_slots}个仓位，调整为{optimal_slots-1}个，"
                           f"单份金额={available_funds/(optimal_slots-1):.2f} USDT")
                optimal_slots -= 1

            symbol_amount = available_funds / optimal_slots
            symbol_amount = min(symbol_amount, settings.SINGLE_SYMBOL_MAX_INVESTMENT)

            logger.info(f"{symbol} 最终分配: 持仓位={optimal_slots}, 单份金额={symbol_amount:.2f} USDT")

            initial_margin = symbol_amount * (main_config.initial_position / 100)

            if initial_margin < min_margin_required:
                logger.warning(f"{symbol} 初始保证金 {initial_margin:.2f} 小于最小要求 {min_margin_required:.2f}，"
                              f"自动调整为 {min_margin_required:.2f}")
                initial_margin = min_margin_required

            logger.info(f"{symbol} 初始保证金: {initial_margin:.2f} USDT (比例={main_config.initial_position}%)")

            safe_params = self._calculate_safe_order_params(
                symbol=symbol,
                target_margin=initial_margin,
                target_leverage=main_config.leverage,
                current_price=current_price
            )

            actual_leverage = safe_params['leverage']
            actual_margin = safe_params['margin']

            logger.info(f"{symbol} 安全参数计算完成: 杠杆={actual_leverage}x, 保证金={actual_margin:.2f} USDT")

            if actual_margin > available_funds:
                logger.error(f"{symbol} 调整后保证金 {actual_margin:.2f} USDT 超过可用资金 {available_funds:.2f} USDT，跳过开仓")
                dispatcher.release_coin(symbol, strategy_type)
                return False

            if actual_margin > symbol_amount:
                logger.error(f"{symbol} 调整后保证金 {actual_margin:.2f} USDT 超过分配给该币种的最大投资金额 {symbol_amount:.2f} USDT，跳过开仓")
                dispatcher.release_coin(symbol, strategy_type)
                return False

            self._set_margin_mode_and_leverage(symbol, actual_leverage, main_config.margin_mode)

            quantity = safe_params['quantity']

            quantity, is_valid, adjustment_info = self._validate_order_quantity_unified(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                order_type='MARKET',
                ensure_min_notional=True,
                max_quantity_limit=None,
                description='开仓单'
            )

            if not is_valid:
                final_notional = adjustment_info.get('final_notional', 0)
                logger.error(f"{symbol} 开仓数量验证失败，取消开仓")
                logger.error(f"  最终数量: {adjustment_info.get('final_quantity')}")
                logger.error(f"  最终名义价值: {final_notional:.2f} USDT")

                actual_min_notional = adjustment_info.get('actual_min_notional', 0)
                if actual_min_notional > 0 and final_notional < actual_min_notional:
                    logger.error(f"  【问题】名义价值 小于币安要求该币种的最低要求 {actual_min_notional} USDT")
                    logger.error(f"  【原因】该币种价格: {current_price:.6f} USDT，保证金: {actual_margin:.2f} USDT (INITIAL_POSITION={main_config.initial_position}%)")
                    logger.error(f"  【建议1】检查环境变量 INITIAL_POSITION (当前={main_config.initial_position}%)，建议增加到 100% 或更高")
                    logger.error(f"  【建议2】跳过价格过低的币种（当前价格 {current_price:.6f} USDT）")

                dispatcher.release_coin(symbol, strategy_type)
                return False

            if adjustment_info.get('adjusted', False):
                logger.info(f"{symbol} 订单数量已调整:")
                logger.info(f"  原始数量: {adjustment_info['original_quantity']:.8f}")
                logger.info(f"  最终数量: {adjustment_info['final_quantity']:.8f}")
                logger.info(f" 名义价值: {adjustment_info['final_notional']:.2f} USDT")

            precision_info = self.client.get_tick_size_and_precision(symbol)
            qty_precision = precision_info['qty_precision']

            logger.info(f"{symbol} 订单数量: {quantity} (精度={qty_precision})")

            side = 'BUY' if direction == 'LONG' else 'SELL'

            if self.trader:
                position_side_mode = self.trader.get_account_position_mode()
            else:
                position_side_mode = self._get_account_position_side_mode()

            order_type = 'LIMIT'
            order = None
            order_params = {}

            logger.info(f"{symbol} 尝试开仓: 优先使用限价单，价格={current_price}, 数量={quantity}")

            try:
                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'order_type': 'LIMIT',
                    'quantity': quantity,
                    'price': current_price,
                    'timeInForce': 'GTC',
                    'newClientOrderId': f"{symbol[:6]}_L_{int(time.time())}"
                }

                logger.info(f"{symbol} 限价单参数: {order_params}")

                order = self.client.create_order(**order_params)
                order_id = order.get('orderId')

                logger.info(f"{symbol} 限价单已创建: orderId={order_id}")

                max_wait_seconds = 5
                wait_interval = 1

                for wait_attempt in range(max_wait_seconds):
                    time.sleep(wait_interval)

                    order_status = self.client.get_order(symbol, order_id)
                    status = order_status.get('status')

                    logger.debug(f"{symbol} 订单状态查询 ({wait_attempt+1}/{max_wait_seconds}): {status}, 已成交: {order_status.get('executedQty', '0')}")

                    if status == 'FILLED':
                        logger.info(f"✅ {symbol} 限价单已完全成交")
                        break
                    elif status in ['CANCELED', 'EXPIRED', 'REJECTED', 'EXPIRED']:
                        logger.warning(f"{symbol} 限价单异常状态: {status}")
                        break

                order_status = self.client.get_order(symbol, order_id)
                status = order_status.get('status')

                if status != 'FILLED':
                    executed_qty = float(order_status.get('executedQty', 0))
                    remaining_qty = quantity - executed_qty

                    logger.info(f"{symbol} 限价单未完全成交，状态={status}, 已成交={executed_qty}, 剩余={remaining_qty}")

                    if status == 'NEW' or status == 'PARTIALLY_FILLED':
                        try:
                            self.client.cancel_order(symbol, order_id)
                            logger.info(f"{symbol} 限价单已取消")
                        except Exception as cancel_error:
                            logger.warning(f"{symbol} 取消限价单失败，可能已成交: {cancel_error}")

                    if remaining_qty > 0.001:
                        logger.info(f"{symbol} 使用市价单补足剩余 {remaining_qty:.6f}")

                        order_params = {
                            'symbol': symbol,
                            'side': side,
                            'order_type': 'MARKET',
                            'quantity': remaining_qty,
                            'newClientOrderId': f"{symbol[:6]}_M_{int(time.time())}"
                        }

                        limit_order_result = order
                        order = self.client.create_order(**order_params)
                        logger.info(f"✅ {symbol} 市价单执行成功")
                    else:
                        logger.info(f"✅ {symbol} 限价单已完成")
                        order = order_status
                else:
                    logger.info(f"✅ {symbol} 限价单完全成交")

            except BinanceAPIException as e:
                error_code = e.code if hasattr(e, 'code') else None

                logger.warning(f"{symbol} 限价单失败 [API {error_code}]: {e.message}, 改用市价单")

                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'order_type': 'MARKET',
                    'quantity': quantity,
                    'newClientOrderId': f"{symbol[:6]}_M_{int(time.time())}"
                }

                order = self.client.create_order(**order_params)
                logger.info(f"✅ {symbol} 市价单执行成功")

            position_info = self._create_position_info(
                symbol=symbol,
                strategy_name=strategy_name,
                current_price=current_price,
                direction=direction,
                quantity=quantity,
                actual_margin=actual_margin,
                initial_margin=initial_margin,
                actual_leverage=actual_leverage,
            )

            self.positions[symbol] = position_info

            ratio = f"{actual_margin:.0f}/{getattr(self.trader, 'account_balance', 0):.0f}" if hasattr(self, 'trader') and self.trader else ""
            capital_ratio = actual_margin / getattr(self.trader, 'account_balance', actual_margin) if hasattr(self, 'trader') and self.trader and getattr(self.trader, 'account_balance', 0) > 0 else 1.0
            self.trade_recorder.record_open(
                symbol=symbol,
                strategy=strategy_name,
                direction=direction,
                leverage=actual_leverage,
                quantity=quantity,
                margin=actual_margin,
                capital_ratio=capital_ratio,
                entry_price=current_price,
            )

            logger.info(f"✅ {symbol} 持仓已注册到本地管理: 数量={quantity}, 价格={current_price}, 方向={direction}, 保证金={actual_margin:.2f}")

            self._on_open_complete(symbol, position_info, strategy_type, dispatcher)
            return True

        except Exception as e:
            logger.error(f"{symbol} 开仓异常: {str(e)}", exc_info=True)
            if coin_acquired:
                dispatcher.release_coin(symbol, strategy_type)
            return False

    def _create_position_info(self, symbol: str, strategy_name: str, current_price: float,
                              direction: str, quantity: float, actual_margin: float,
                              initial_margin: float, actual_leverage: int) -> Dict:
        """创建持仓信息字典，子类可覆盖添加策略特有字段"""
        return {
            'symbol': symbol,
            'strategy': strategy_name,
            'entry_price': current_price,
            'current_price': current_price,
            'direction': direction,
            'total_quantity': quantity,
            'total_investment': actual_margin,
            'initial_margin': initial_margin,
            'completed_investment': actual_margin,
            'leverage': actual_leverage,
            'status': 'active',
            'profit': 0.0,
            'profit_pct': 0.0,
            'max_profit_pct': 0.0,
            'added_levels': [],
            'pending_orders': [],
            'is_closing': False,
            'last_action_time': int(time.time() * 1000),
            'take_profit_levels': {},
            'stop_loss_levels': {},
            'trailing_take_profit_order_id': None,
            'trailing_take_profit_callback': None,
            'position_complete': False,
            'is_internal': True
        }

    def _on_open_complete(self, symbol: str, position_info: Dict, strategy_type, dispatcher):
        """子类覆盖: 开仓完成后额外逻辑"""
        pass

    def _execute_add_to_position(self, symbol: str, position_info: Dict, current_price: float, pnl_rate: float) -> bool:
        logger.info(f"{symbol} 加仓逻辑暂未启用，跳过")
        return True

    def _execute_close_position(self, symbol: str, position_info: Dict, close_pct: float, reason: str) -> bool:
        """执行平仓操作"""
        try:
            pos_amt = float(position_info.get('position_amt', 0))
            if pos_amt == 0:
                logger.info(f"{symbol} 无持仓，跳过平仓")
                return True
            
            # 确定平仓方向：LONG仓位用SELL平，SHORT仓位用BUY平
            side = 'SELL' if pos_amt > 0 else 'BUY'
            
            # 计算平仓数量
            quantity = abs(pos_amt) * close_pct / 100.0
            if quantity <= 0:
                logger.info(f"{symbol} 平仓数量为0，跳过")
                return True
            
            # 精度调整
            quantity = self.client.adjust_quantity_precision(symbol, quantity)
            if quantity <= 0:
                logger.warning(f"{symbol} 精度调整后数量为0，跳过平仓")
                return True
            
            logger.info(f"{symbol} 执行平仓: {side} {quantity} ({close_pct}%), 原因={reason}")
            
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity,
            }
            
            result = self.client.create_order(**order_params)
            if result and result.get('orderId'):
                logger.info(f"{symbol} 平仓成功, orderId={result['orderId']}")
                return True
            else:
                logger.warning(f"{symbol} 平仓失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"{symbol} 平仓异常: {e}")
            return False

    def _choose_algo_order_type(self, direction: str) -> str:
        return 'STOP_MARKET' if direction == 'LONG' else 'TAKE_PROFIT_MARKET'

    def _calculate_stop_trigger_price(self, symbol: str, side: str, current_price: float) -> float:
        try:
            precision_info = self.client.get_tick_size_and_precision(symbol)
            tick_size = precision_info['tick_size']

            if side == 'BUY':
                trigger_price = current_price * 0.999
            else:
                trigger_price = current_price * 1.001

            adjusted_price = round(trigger_price / tick_size) * tick_size

            logger.debug(f"{symbol} {side} 方向止损触发价格: 当前价={current_price:.6f}, 触发价={adjusted_price:.6f}")
            return adjusted_price

        except Exception as e:
            logger.warning(f"{symbol} 计算止损触发价格失败: {str(e)}")
            return current_price

    def _calculate_take_profit_trigger_price(self, symbol: str, side: str, current_price: float) -> float:
        try:
            precision_info = self.client.get_tick_size_and_precision(symbol)
            tick_size = precision_info['tick_size']

            if side == 'BUY':
                trigger_price = current_price * 1.001
            else:
                trigger_price = current_price * 0.999

            adjusted_price = round(trigger_price / tick_size) * tick_size

            logger.debug(f"{symbol} {side} 方向止盈触发价格: 当前价={current_price:.6f}, 触发价={adjusted_price:.6f}")
            return adjusted_price

        except Exception as e:
            logger.warning(f"{symbol} 计算止盈触发价格失败: {str(e)}")
            return current_price

    def _choose_order_type(self, symbol: str, quantity: float, current_price: float) -> str:
        try:
            notional_value = quantity * current_price

            if notional_value < settings.ORDER_TYPE_MARKET_THRESHOLD:
                logger.debug(f"{symbol} 小额订单({notional_value:.2f} USDT < {settings.ORDER_TYPE_MARKET_THRESHOLD:.0f} USDT)，使用市价单")
                return 'MARKET'

            try:
                ticker_24h = self.client.get_ticker_24h(symbol)
                volume = float(ticker_24h.get('volume', 0)) * float(ticker_24h.get('lastPrice', current_price))

                if volume < settings.ORDER_TYPE_LOW_VOLUME_THRESHOLD:
                    logger.info(f"{symbol} 流动性较低，使用限价单 (日交易量: {volume:.0f} USDT < {settings.ORDER_TYPE_LOW_VOLUME_THRESHOLD:.0f} USDT)")
                    return 'LIMIT'

            except Exception as e:
                logger.debug(f"{symbol} 无法获取交易量信息: {str(e)}")

            return 'MARKET'

        except Exception as e:
            logger.warning(f"{symbol} 选择订单类型失败: {str(e)}")
            return 'MARKET'

    def _get_account_position_side_mode(self) -> str:
        try:
            account = self.client.get_account_balance()
            return account.get('position_side', 'ONE-WAY')
        except Exception as e:
            logger.debug(f"获取账户 positionSide 失败: {str(e)}")
            return 'ONE-WAY'

    def _calculate_limit_price(self, symbol: str, side: str, current_price: float) -> float:
        try:
            precision_info = self.client.get_tick_size_and_precision(symbol)
            tick_size = precision_info.get('tick_size', 0.0001)

            if side == 'BUY':
                limit_price = current_price * settings.ORDER_TYPE_LIMIT_PRICE_BUY_BIAS
            else:
                limit_price = current_price * settings.ORDER_TYPE_LIMIT_PRICE_SELL_BIAS

            if tick_size > 0:
                limit_price = round(limit_price / tick_size) * tick_size

            logger.debug(f"{symbol} 限价单价格: {current_price:.6f} -> {limit_price:.6f}")
            return limit_price

        except Exception as e:
            logger.warning(f"{symbol} 计算限价失败: {str(e)}")
            return current_price

    def get_available_funds(self) -> float:
        try:
            current_time = time.time()
            if hasattr(self, '_balance_cache') and self._balance_cache and (current_time - getattr(self, '_balance_cache_time', 0)) < 5:
                return self._balance_cache

            account_info = self.client.get_account_balance()

            total_balance = account_info.get('total_balance', 0)
            available_balance = account_info.get('available_balance', 0)
            unrealized_pnl = account_info.get('cross_unpnl', account_info.get('unrealized_pnl', 0))

            loss_amount = abs(unrealized_pnl) if unrealized_pnl < 0 else 0
            available_funds = available_balance - loss_amount

            self._balance_cache = available_funds
            self._balance_cache_time = current_time

            logger.debug(f"可用资金计算: 总余额={total_balance:.2f}, "
                        f"可用={available_funds:.2f}")

            return available_funds

        except Exception as e:
            logger.error(f"获取可用资金失败: {str(e)}")
            return 0

    def _get_max_order_quantity(self, symbol: str, price: float) -> float:
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info or 'filters' not in symbol_info:
                return float('inf')

            max_qty = float('inf')
            for f in symbol_info['filters']:
                if f.get('filterType') == 'LOT_SIZE':
                    max_qty = float(f.get('maxQty', float('inf')))

            return max_qty

        except Exception as e:
            logger.warning(f"{symbol} 获取最大订单数量失败: {str(e)}")
            return float('inf')

    def _calculate_safe_order_params(self, symbol: str, target_margin: float, target_leverage: int, current_price: float):
        try:
            max_leverage = self.client.get_leverage_bracket(symbol)
            logger.debug(f"{symbol} 币安最大杠杆: {max_leverage}x")

            effective_leverage = target_leverage

            volatility_enabled = getattr(settings, 'VOLATILITY_LEVERAGE_ENABLED', False)

            if volatility_enabled:
                try:
                    volatility_periods = getattr(settings, 'VOLATILITY_PERIODS', 20)
                    volatility_threshold = getattr(settings, 'VOLATILITY_AVG_THRESHOLD', 4.0)
                    volatility_reduction = getattr(settings, 'VOLATILITY_LEVERAGE_REDUCTION', 0.5)

                    monitor_interval = getattr(settings, 'MONITOR_INTERVAL', 60)

                    if monitor_interval <= 5:
                        volatility_kline = '1m'
                    elif monitor_interval <= 15:
                        volatility_kline = '5m'
                    elif monitor_interval <= 60:
                        volatility_kline = '15m'
                    else:
                        volatility_kline = '1h'

                    logger.debug(f"{symbol} 波动率计算: 监控周期={monitor_interval}分钟, K线周期={volatility_kline}")

                    volatility = self.client.calculate_volatility(symbol, volatility_periods, volatility_kline)

                    if volatility > 0:
                        if volatility > volatility_threshold * 1.5:
                            reduction_factor = max(0.25, volatility_reduction * 0.5)
                            effective_leverage = int(target_leverage * reduction_factor)
                            logger.warning(f"{symbol} 极高波动率({volatility:.2f}%)，杠杆从{target_leverage}x降到{effective_leverage}x")
                        elif volatility > volatility_threshold:
                            effective_leverage = int(target_leverage * volatility_reduction)
                            logger.warning(f"{symbol} 高波动率({volatility:.2f}%)，杠杆从{target_leverage}x降到{effective_leverage}x")
                        else:
                            logger.debug(f"{symbol} 正常波动率({volatility:.2f}%)，保持杠杆{target_leverage}x")

                except Exception as e:
                    logger.warning(f"{symbol} 波动率计算失败，使用原始杠杆: {str(e)}")
                    effective_leverage = target_leverage

            effective_leverage = min(effective_leverage, max_leverage)

            max_margin = self._get_max_margin_for_symbol(symbol, effective_leverage)
            logger.debug(f"{symbol} 杠杆{effective_leverage}x 最大保证金: {max_margin:.2f} USDT")
            max_notional = max_margin * effective_leverage
            logger.debug(f"{symbol} 杠杆{effective_leverage}x 最大名义价值: {max_notional:.2f} USDT")

            # 统一使用 get_symbol_info 获取过滤器
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

            safe_leverage = min(effective_leverage, max_leverage)
            if safe_leverage != target_leverage:
                logger.warning(f"{symbol} 杠杆从{target_leverage}x调整为{safe_leverage}x（波动率调整+币安限制）")

            target_notional = target_margin * safe_leverage
            safe_notional = min(target_notional, max_notional)

            if safe_notional < target_notional:
                safe_margin = safe_notional / safe_leverage
                logger.warning(f"{symbol} 保证金从{target_margin:.2f}调整为{safe_margin:.2f}（受名义价值限制）")
            else:
                safe_margin = target_margin

            if safe_notional > max_notional * 0.95:
                if safe_leverage > 5:
                    safe_leverage = max(5, safe_leverage - 5)
                    safe_notional = safe_margin * safe_leverage
                    logger.warning(f"{symbol} 降低杠杆到{safe_leverage}x以避免超出最大持仓限制")
                else:
                    safe_margin = safe_margin * 0.8
                    safe_notional = safe_margin * safe_leverage
                    logger.warning(f"{symbol} 减少保证金到{safe_margin:.2f}以避免超出最大持仓限制")

            quantity = (safe_margin * safe_leverage) / current_price

            quantity, is_valid = self._validate_and_adjust_quantity(symbol, quantity, current_price, safe_leverage)

            if not is_valid:
                logger.error(f"{symbol} 数量验证失败，使用保守参数")
                quantity = min_qty
                quantity = self.client.adjust_quantity_precision(symbol, quantity)

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
            safe_leverage = min(target_leverage, 20)
            quantity = (target_margin * safe_leverage) / current_price
            return {
                'leverage': safe_leverage,
                'margin': target_margin,
                'quantity': quantity,
                'notional': target_margin * safe_leverage,
                'price': current_price
            }

    def _validate_and_adjust_quantity(self, symbol: str, quantity: float, price: float, leverage: int) -> Tuple[float, bool]:
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"{symbol} 无法获取交易对信息，使用原始数量")
                return quantity, True

            lot_size_min_qty = 0.001
            lot_size_max_qty = float('inf')
            lot_size_step_size = 0.001
            min_notional = 0
            max_notional = float('inf')

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

            original_quantity = quantity

            if quantity < lot_size_min_qty:
                logger.warning(f"{symbol} 数量 {quantity:.8f} 小于LOT_SIZE最小值 {lot_size_min_qty}")
                quantity = lot_size_min_qty
                quantity = self.client.adjust_quantity_precision(symbol, quantity)
                logger.info(f"{symbol} 调整数量到最小值: {quantity:.8f}")
            else:
                quantity = self.client.adjust_quantity_precision(symbol, quantity)
                if abs(quantity - original_quantity) > 1e-8:
                    logger.debug(f"{symbol} 数量精度调整: {original_quantity:.8f} -> {quantity:.8f}")

            if quantity > lot_size_max_qty:
                logger.warning(f"{symbol} 数量 {quantity:.8f} 超过LOT_SIZE最大值 {lot_size_max_qty}")
                quantity = lot_size_max_qty
                quantity = self.client.adjust_quantity_precision(symbol, quantity)
                logger.info(f"{symbol} 调整数量到最大值: {quantity:.8f}")

            min_qty_calc = min_notional / price
            min_qty_for_notional = math.ceil(min_qty_calc / lot_size_step_size) * lot_size_step_size

            min_qty_for_notional = math.ceil(min_qty_for_notional / lot_size_step_size) * lot_size_step_size

            while (min_qty_for_notional * price) < min_notional:
                min_qty_for_notional += lot_size_step_size

            notional_value = quantity * price
            if notional_value < min_notional:
                logger.warning(f"{symbol} 名义价值 {notional_value:.2f} USDT 小于最小要求 {min_notional} USDT")
                quantity = max(quantity, min_qty_for_notional)

                while (quantity * price) < min_notional:
                    quantity = math.ceil((min_notional / price + 0.000000001) / lot_size_step_size) * lot_size_step_size

                notional_value = quantity * price
                logger.info(f"{symbol} 调整数量到满足minNotional: {quantity:.8f} (名义价值: {notional_value:.2f} USDT)")

            max_notional_for_leverage = self.client.get_max_notional_for_leverage(symbol, leverage)
            if max_notional_for_leverage > 0 and notional_value > max_notional_for_leverage:
                logger.warning(f"{symbol} 名义价值 {notional_value:.2f} 超过杠杆{leverage}x最大值 {max_notional_for_leverage:.2f}")
                max_quantity = max_notional_for_leverage / price
                quantity = self.client.adjust_quantity_precision(symbol, max_quantity)
                notional_value = quantity * price
                logger.info(f"{symbol} 调整数量以符合杠杆限制: {quantity:.8f} (名义价值: {notional_value:.2f})")
                # 重新校验 minNotional（杠杆调整可能使数量再次低于最小值）
                if notional_value < min_notional:
                    quantity = max(quantity, min_qty_for_notional)
                    while (quantity * price) < min_notional:
                        quantity = math.ceil((min_notional / price + 0.000000001) / lot_size_step_size) * lot_size_step_size
                    notional_value = quantity * price
                    logger.info(f"{symbol} 再次调整数量以保持minNotional: {quantity:.8f} (名义价值: {notional_value:.2f})")

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
        adjustment_info = {
            'original_quantity': quantity,
            'adjusted': False,
            'adjustments': []
        }

        try:
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"{symbol} {description}: 无法获取交易对信息，使用原始数量")
                return quantity, False, adjustment_info

            min_qty = 0.001
            max_qty = float('inf')
            step_size = 0.001
            min_notional = 0.0
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

            if quantity < min_qty:
                old_qty = quantity
                quantity = min_qty
                adjustment_info['adjustments'].append(f'LOT_SIZE最小值: {old_qty:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.warning(f"{symbol} {description}: 数量 {old_qty:.8f} 小于最小值 {min_qty}，调整到 {quantity:.8f}")

            original = quantity
            quantity = round(quantity / step_size) * step_size
            if abs(quantity - original) > 1e-8:
                adjustment_info['adjustments'].append(f'精度调整: {original:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.debug(f"{symbol} {description}: 数量精度调整 {original:.8f} -> {quantity:.8f}")

            if quantity > max_qty:
                old_qty = quantity
                quantity = max_qty
                quantity = round(quantity / step_size) * step_size
                adjustment_info['adjustments'].append(f'LOT_SIZE最大值: {old_qty:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.warning(f"{symbol} {description}: 数量 {old_qty:.8f} 超过最大值 {max_qty}，调整到 {quantity:.8f}")

            if max_quantity_limit is not None and quantity > max_quantity_limit:
                old_qty = quantity
                quantity = max_quantity_limit
                quantity = round(quantity / step_size) * step_size
                adjustment_info['adjustments'].append(f'最大数量限制: {old_qty:.8f} -> {quantity:.8f}')
                adjustment_info['adjusted'] = True
                logger.warning(f"{symbol} {description}: 数量 {old_qty:.8f} 超过限制 {max_quantity_limit:.8f}，调整到 {quantity:.8f}")

            notional_value = quantity * price

            if ensure_min_notional and notional_value < min_notional:
                min_qty_calc = min_notional / price
                min_qty_for_notional = math.ceil(min_qty_calc / step_size) * step_size

                min_qty_for_notional = math.ceil(min_qty_for_notional / step_size) * step_size

                while (min_qty_for_notional * price) < min_notional:
                    min_qty_for_notional += step_size

                old_qty = quantity
                quantity = max(quantity, min_qty_for_notional)

                quantity = round(quantity / step_size) * step_size

                notional_value = quantity * price
                adjustment_info['adjustments'].append(f'minNotional: {old_qty:.8f} -> {quantity:.8f} (名义价值: {notional_value:.2f} USDT)')
                adjustment_info['adjusted'] = True

                logger.warning(f"{symbol} {description}: 名义价值 {old_qty * price:.2f} USDT < 最小要求 {min_notional} USDT")
                logger.info(f"{symbol} {description}: 调整数量到 {quantity:.8f} (名义价值: {notional_value:.2f} USDT)")

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
            adjustment_info['actual_min_notional'] = min_notional

            return quantity, is_valid, adjustment_info

        except Exception as e:
            logger.error(f"{symbol} {description}: 数量验证失败 - {str(e)}", exc_info=True)
            return quantity, False, adjustment_info

    def _get_max_margin_for_symbol(self, symbol: str, leverage: int) -> float:
        try:
            max_notional = self.client.get_max_notional_for_leverage(symbol, leverage)

            if max_notional > 0:
                max_margin = max_notional / leverage
                logger.debug(f"{symbol} 杠杆{leverage}x: 最大持仓价值={max_notional:.2f}, "
                            f"最大保证金={max_margin:.2f}")
                return max_margin
            else:
                logger.debug(f"{symbol} 无法获取最大持仓价值，使用配置值")
            return settings.MAX_MARGIN_PER_SYMBOL

        except Exception as e:
            logger.warning(f"{symbol} 获取最大保证金失败: {str(e)}，使用配置值")
            return settings.MAX_MARGIN_PER_SYMBOL

    def _set_margin_mode_and_leverage(self, symbol: str, leverage: int, margin_mode: str):
        try:
            try:
                leverage_brackets = self.client.get_leverage_brackets(symbol)
                if leverage_brackets and len(leverage_brackets) > 0:
                    available_leverages = sorted([b['initialLeverage'] for b in leverage_brackets], reverse=True)
                    max_leverage = max(available_leverages)
                    logger.info(f"{symbol} 支持的杠杆档位: {available_leverages}")
                else:
                    raise Exception("无法获取杠杆档位信息")
            except Exception as e:
                logger.warning(f"{symbol} 无法获取杠杆档位信息: {str(e)}，使用默认逻辑")
                max_leverage = self.client.get_leverage_bracket(symbol)
                available_leverages = []
                current = max_leverage
                while current >= 5:
                    available_leverages.append(current)
                    current -= 5
                if not available_leverages:
                    available_leverages = [5]

            if leverage > max_leverage:
                logger.warning(f"{symbol} 请求杠杆 {leverage}x 超过最大支持杠杆 {max_leverage}x，自动降低到 {max_leverage}x")
                leverage = max_leverage

            leverage_sequence = []
            current = leverage
            while current >= 5:
                if current in available_leverages:
                    leverage_sequence.append(current)
                current -= 1

            if not leverage_sequence and available_leverages:
                leverage_sequence = [min(available_leverages)]

            if not leverage_sequence:
                leverage_sequence = [5]

            logger.info(f"{symbol} 杠杆设置序列: {leverage_sequence}")

            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                for try_leverage in leverage_sequence:
                    try:
                        self.client.change_margin_type(symbol, margin_mode)
                        logger.info(f"{symbol} 保证金模式已设置为: {margin_mode}")

                        result = self.client.change_leverage(symbol, try_leverage)
                        if result:
                            logger.info(f"{symbol} 杠杆已设置为: {try_leverage}x")

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

                            return
                        else:
                            logger.warning(f"{symbol} 杠杆 {try_leverage}x 设置返回失败")

                    except Exception:
                        logger.warning(f"{symbol} 杠杆 {try_leverage}x 设置失败，尝试下一个杠杆")
                        continue

                if attempt < max_retries - 1:
                    logger.warning(f"{symbol} 第{attempt+1}次尝试全部失败，等待{retry_delay}秒后重试")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"{symbol} 所有杠杆设置尝试均失败")
                    raise Exception(f"无法为{symbol}设置任何有效杠杆")

        except Exception as e:
            logger.error(f"{symbol} 设置保证金模式和杠杆失败: {str(e)}")
            raise

    def _create_position_building_plan(self, position_info: Dict, entry_price: float, direction: str):
        symbol = position_info.get('symbol', 'Unknown')
        try:
            total_investment = position_info['total_investment']
            leverage = position_info['leverage']

            if entry_price <= 0:
                logger.error(f"{symbol} entry_price无效({entry_price})，无法创建建仓计划")
                position_info['pending_orders'] = []
                return

            levels = [
                (settings.LOSS_STEP1, settings.LOSS_ADD1, "亏损加仓1"),
                (settings.LOSS_STEP2, settings.LOSS_ADD2, "亏损加仓2"),
                (settings.LOSS_STEP3, settings.LOSS_ADD3, "亏损加仓3"),
                (settings.PROFIT_STEP1, settings.PROFIT_ADD1, "盈利加仓1"),
                (settings.PROFIT_STEP2, settings.PROFIT_ADD2, "盈利加仓2"),
                (settings.PROFIT_STEP3, settings.PROFIT_ADD3, "盈利加仓3")
            ]

            pending_orders = []

            for idx, (trigger_rate, add_percent, reason) in enumerate(levels):
                add_margin = total_investment * (add_percent / 100)

                add_notional = add_margin * leverage
                quantity = add_notional / entry_price

                pending_order = {
                    'index': idx,
                    'trigger_rate': trigger_rate,
                    'add_percent': add_percent,
                    'add_margin': add_margin,
                    'limit_price': entry_price,
                    'quantity': quantity,
                    'order_id': None,
                    'status': 'pending',
                    'reason': reason
                }

                pending_orders.append(pending_order)
                logger.info(f"{symbol} 建仓计划{idx}: 触发盈利率={trigger_rate}%, "
                           f"加仓比例={add_percent}%, 数量={quantity:.6f}")

            position_info['pending_orders'] = pending_orders
            logger.info(f"{symbol} 建仓计划创建完成，共{len(pending_orders)}级已添加到持仓信息")

        except Exception as e:
            logger.error(f"{symbol} 创建建仓计划失败: {str(e)}")
            position_info['pending_orders'] = []

    def _check_position_complete(self, symbol: str, position_info: Dict, pnl_rate: float) -> bool:
        logger.debug(f"检查建仓完成: {symbol}, 盈利率: {pnl_rate}%")
        return False

    def _finalize_position_building(self, symbol: str, position_info: Dict, reason: str) -> bool:
        logger.debug(f"完成建仓: {symbol}, 原因: {reason}")
        return True


def get_position_builder(client, position_manager) -> BasePositionBuilder:
    return BasePositionBuilder(client, position_manager)
