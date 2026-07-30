# -*- coding: utf-8 -*-
"""
订单引擎

统一订单管理，提供下单、改单、撤单接口

特性：
1. 统一订单入口
2. 订单状态追踪
3. 重试机制
4. 订单持久化
5. 限价单转市价
"""

import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from framework.core.logger import get_logger, track_performance
from framework.core.events import EventBus, Event, EventType, OrderEvent
from framework.core.exceptions import OrderException, ErrorCode
# 修复：统一使用framework.shared.enums中的枚举定义
from framework.shared.enums import OrderSide, OrderType, OrderStatus, OrderMode

logger = get_logger('order_engine')


@dataclass
class OrderParams:
    """订单参数"""
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    leverage: int = 20
    margin_mode: str = 'ISOLATED'
    strategy: str = ''
    reduce_only: bool = False
    client_order_id: str = ''
    position_side: Optional[str] = None  # HEDGING 模式需要: 'LONG' 或 'SHORT'
    # 订单执行模式
    order_mode: OrderMode = OrderMode.LIMIT_TO_MARKET  # 默认限价转市价
    # 限价转市价超时时间（秒）
    limit_timeout: int = 5
    # 止盈止损（策略设置）
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    # 补充：开仓方向（用于校验订单方向是否匹配仓位）
    # 设置此值后，系统会自动校验：
    # - 若 open_direction='LONG'，则 BUY 订单应该对应多头开仓，SELL 应该对应平空
    # - 若 open_direction='SHORT'，则 SELL 订单应该对应空头开仓，BUY 应该对应平多
    open_direction: Optional[str] = None  # 'LONG' | 'SHORT' | None

    def __post_init__(self):
        if not self.client_order_id:
            # 生成唯一订单ID：策略名 + 时间戳 + UUID（确保唯一性）
            import time
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            unique_id = uuid.uuid4().hex[:12]  # 使用12个字符的UUID
            self.client_order_id = f"{self.strategy}_{timestamp}_{unique_id}"

        # 防御性校验：校验 position_side 的有效值
        if self.position_side is not None and self.position_side not in ('LONG', 'SHORT', None):
            raise ValueError(f"position_side 必须是 'LONG', 'SHORT' 或 None（对冲模式），当前值: {self.position_side}")


@dataclass
class OrderResult:
    """订单结果"""
    success: bool
    order_id: str = ''
    client_order_id: str = ''
    symbol: str = ''
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    message: str = ''
    timestamp: datetime = field(default_factory=datetime.now)
    params: Optional[OrderParams] = None
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]


@dataclass
class OrderInfo:
    """订单信息（内部追踪）"""
    params: OrderParams
    result: OrderResult
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3


class OrderEngine:
    """
    订单引擎
    
    统一管理所有订单
    
    使用方式：通过构造函数注入，勿使用单例模式
    """
    
    def __init__(self, client=None):
        self._client = client
        self._orders: Dict[str, OrderInfo] = {}  # client_order_id -> OrderInfo
        self._order_lock = threading.RLock()

        # 缓存已设置的杠杆和保证金模式，避免重复API调用（修复BUG-H005）
        self._symbol_settings: Dict[str, dict] = {}  # symbol -> {leverage, margin_mode}
        self._symbol_settings_lock = threading.RLock()  # 保护symbol_settings的并发访问

        # 事件总线 - 使用单例
        from framework.core.events import get_event_bus
        self._event_bus = get_event_bus()

        # 策略回调
        self._strategy_callbacks: Dict[str, Callable] = {}
        self._callback_lock = threading.Lock()

        # 公共订单处理器 - 用于调整参数符合币安限制
        from framework.business.order_helper import OrderHelper
        self._order_helper: OrderHelper = None

        logger.info("订单引擎已初始化")
        
    def set_client(self, client):
        """设置API客户端"""
        self._client = client
        
        # 初始化公共订单处理器
        from framework.business.order_helper import OrderHelper
        self._order_helper = OrderHelper(client)
        
    @property
    def client(self):
        """获取API客户端"""
        return self._client
        
    def register_strategy_callback(self, strategy_name: str, callback: Callable[[OrderResult], None]):
        """注册策略订单回调"""
        with self._callback_lock:
            self._strategy_callbacks[strategy_name] = callback
        
    @track_performance('create_order')
    def create_order(self, params: OrderParams) -> OrderResult:
        """
        创建订单
        
        Args:
            params: 订单参数
            
        Returns:
            OrderResult
        """
        # ========== 参数验证 ==========
        if not params.symbol or not isinstance(params.symbol, str):
            return OrderResult(
                success=False,
                message="symbol必须是非空字符串",
                params=params
            )
        
        if not params.side:
            return OrderResult(
                success=False,
                message="side必须指定",
                params=params
            )
        
        if not params.type:
            return OrderResult(
                success=False,
                message="type必须指定",
                params=params
            )
        
        if params.quantity is None or params.quantity <= 0:
            return OrderResult(
                success=False,
                message=f"quantity必须大于0，当前值: {params.quantity}",
                params=params
            )
        
        if params.type == OrderType.LIMIT and (params.price is None or params.price <= 0):
            return OrderResult(
                success=False,
                message=f"LIMIT订单必须指定有效的price，当前值: {params.price}",
                params=params
            )
        
        if params.type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT] and (params.stop_price is None or params.stop_price <= 0):
            return OrderResult(
                success=False,
                message=f"{params.type.value}订单必须指定有效的stop_price，当前值: {params.stop_price}",
                params=params
            )
        
        if not self._client:
            return OrderResult(
                success=False,
                message="未设置API客户端",
                params=params
            )
        
        # ========== 仓位方向校验 ==========
        # 如果指定了 open_direction，进行校验
        if params.open_direction:
            if params.open_direction not in ('LONG', 'SHORT'):
                return OrderResult(
                    success=False,
                    message=f"open_direction 必须是 'LONG' 或 'SHORT'，当前值: {params.open_direction}",
                    params=params
                )
            
            # 校验订单方向与开仓方向是否匹配
            if params.side == OrderSide.BUY:
                # 买入应该是:
                # - 开多单 (open_direction='LONG') 或
                # - 平空单 (open_direction='SHORT' with reduce_only=True)
                if params.open_direction == 'SHORT' and not params.reduce_only:
                    # 买入开空仓是错误的
                    logger.warning(f"订单方向不匹配: BUY 订单不能用于开空仓 (open_direction={params.open_direction})")
                    # 不直接拒绝，而是给出警告（可能有特殊情况）
            elif params.side == OrderSide.SELL:
                # 卖出应该是:
                # - 开空单 (open_direction='SHORT') 或
                # - 平多单 (open_direction='LONG' with reduce_only=True)
                if params.open_direction == 'LONG' and not params.reduce_only:
                    logger.warning(f"订单方向不匹配: SELL 订单不能用于开多仓 (open_direction={params.open_direction})")
            
            logger.debug(f"仓位方向校验通过: side={params.side.value}, open_direction={params.open_direction}, reduce_only={params.reduce_only}")

        effective_stop_price = params.stop_price if params.stop_price is not None else params.stop_loss_price

        try:
            # 使用OrderHelper调整参数（杠杆限制、数量精度等）
            if self._order_helper:
                adjusted = self._order_helper.adjust_order(
                    symbol=params.symbol,
                    leverage=params.leverage,
                    quantity=params.quantity,
                    price=params.price
                )
                params.leverage = adjusted.leverage
                params.quantity = adjusted.quantity
                if adjusted.price is not None:
                    params.price = adjusted.price
            
            # 设置杠杆和保证金模式（只在设置变化时才调用API）
            # 使用锁保护symbol_settings的并发访问
            with self._symbol_settings_lock:
                # 先确保symbol有字典存在
                if params.symbol not in self._symbol_settings:
                    self._symbol_settings[params.symbol] = {}
                
                current_settings = self._symbol_settings[params.symbol]
                
                # 检查并设置杠杆（只在杠杆有效且不同时才调用API）
                if params.leverage is not None and params.leverage > 0:
                    if current_settings.get('leverage') != params.leverage:
                        try:
                            self._client.set_leverage(params.symbol, params.leverage)
                            current_settings['leverage'] = params.leverage
                            logger.info(f"[杠杆设置] {params.symbol} 杠杆已设置为 {params.leverage}x")
                        except Exception as e:
                            logger.warning(f"[杠杆设置] {params.symbol} 设置杠杆失败: {e}，继续使用当前杠杆")
                else:
                    logger.warning(f"[杠杆设置] {params.symbol} 杠杆参数无效: {params.leverage}，跳过设置")

                # 检查并设置保证金模式（只在模式有效且不同时才调用API）
                if params.margin_mode and params.margin_mode in ['ISOLATED', 'CROSSED']:
                    if current_settings.get('margin_mode') != params.margin_mode:
                        try:
                            self._client.set_margin_type(params.symbol, params.margin_mode)
                            current_settings['margin_mode'] = params.margin_mode
                            logger.info(f"[保证金模式] {params.symbol} 保证金模式已设置为 {params.margin_mode}")
                        except Exception as e:
                            logger.warning(f"[保证金模式] {params.symbol} 设置保证金模式失败: {e}，继续使用当前模式")
                else:
                    logger.warning(f"[保证金模式] {params.symbol} 保证金模式参数无效: {params.margin_mode}，跳过设置")

            # 下单
            logger.info(f"[下单] {params.symbol} 准备下单: side={params.side.value}, type={params.type.value}, quantity={params.quantity}, leverage={params.leverage}")
            
            # 确保止损价格正确映射：优先使用stop_price，如果为None则使用stop_loss_price
            effective_stop_price = params.stop_price if params.stop_price is not None else params.stop_loss_price
            
            result = self._client.create_order(
                symbol=params.symbol,
                side=params.side.value,
                order_type=params.type.value,
                quantity=params.quantity,
                price=params.price,
                stop_price=effective_stop_price,
                reduce_only=params.reduce_only,
                position_side=params.position_side,
                leverage=params.leverage,
                **{'newClientOrderId': params.client_order_id} if params.client_order_id else {}
            )
            
            logger.info(f"[下单] {params.symbol} API返回: orderId={result.get('orderId')}, status={result.get('status')}, executedQty={result.get('executedQty')}, origQty={result.get('origQty')}")

            # 检查订单是否完全成交
            api_status = result.get('status', '')
            executed_qty = float(result.get('executedQty', 0))
            orig_qty = float(result.get('origQty', 0))

            # 对于市价单，如果executedQty > 0，则认为订单已成交
            if params.type == OrderType.MARKET and executed_qty > 0:
                order_status = OrderStatus.FILLED
                logger.info(f"[下单] {params.symbol} 市价单已成交: executedQty={executed_qty}/{orig_qty}")
            else:
                order_status = OrderStatus.from_string(api_status, OrderStatus.NEW)

            order_result = OrderResult(
                success=True,
                order_id=result.get('orderId', ''),
                client_order_id=params.client_order_id,
                symbol=params.symbol,
                status=order_status,
                filled_quantity=executed_qty,
                avg_price=float(result.get('avgPrice', 0)),
                message='订单创建成功',
                params=params
            )
            
            # 记录订单
            with self._order_lock:
                self._orders[params.client_order_id] = OrderInfo(
                    params=params,
                    result=order_result
                )
                
            # 发送事件
            self._event_bus.publish(OrderEvent(
                event_type=EventType.ORDER_CREATED,
                order_id=order_result.order_id,
                symbol=params.symbol,
                side=params.side.value,
                status=order_result.status.value,
                strategy=params.strategy,
                data={
                    'quantity': params.quantity,
                    'price': params.price,
                    'stop_loss_price': params.stop_loss_price,
                    'take_profit_price': params.take_profit_price,
                }
            ))

            # 如果订单已成交，发送 ORDER_FILLED 事件
            if order_result.status == OrderStatus.FILLED:
                self._event_bus.publish(OrderEvent(
                    event_type=EventType.ORDER_FILLED,
                    order_id=order_result.order_id,
                    symbol=params.symbol,
                    side=params.side.value,
                    status=OrderStatus.FILLED.value,
                    strategy=params.strategy,
                    data={
                        'quantity': order_result.filled_quantity,
                        'price': order_result.avg_price,
                        'filled_quantity': order_result.filled_quantity,
                        'stop_loss_price': params.stop_loss_price,
                        'take_profit_price': params.take_profit_price,
                    }
                ))

            # 调用策略回调
            with self._callback_lock:
                callback = self._strategy_callbacks.get(params.strategy) if params.strategy else None
            if callback:
                try:
                    callback(order_result)
                    logger.debug(f"策略回调已调用: {params.strategy}")
                except Exception as e:
                    logger.error(f"策略回调调用失败: {params.strategy}, {e}")

            logger.info(
                f"订单创建成功: {params.symbol} {params.side.value} {params.quantity}",
                extra={'context': {
                    'strategy': params.strategy,
                    'symbol': params.symbol,
                    'order_id': order_result.order_id,
                    'type': params.type.value,
                }}
            )
            
            return order_result
            
        except Exception as e:
            logger.exception(f"订单创建失败: {params.symbol}, {e}")
            
            # 实现重试机制
            retry_count = 0
            max_retries = 3
            retry_delay = 1.0  # 初始延迟1秒
            
            # 判断是否应该重试的错误
            should_retry = True
            error_msg = str(e).lower()
            
            # 以下错误不重试：余额不足、参数错误、订单已存在等
            no_retry_keywords = [
                'insufficient', 'balance', 'margin', '参数', 'parameter',
                'invalid', 'order already exists', 'duplicate'
            ]
            
            for keyword in no_retry_keywords:
                if keyword in error_msg:
                    should_retry = False
                    logger.warning(f"订单创建失败（不重试）: {params.symbol}, {e}")
                    break
            
            if should_retry:
                while retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"订单创建重试 {retry_count}/{max_retries}: {params.symbol}")
                    
                    try:
                        time.sleep(retry_delay)
                        
                        # 检查订单是否已存在（避免重复下单）
                        existing_order = None
                        if params.client_order_id:
                            try:
                                existing_order = self._client.get_order(
                                    symbol=params.symbol,
                                    orig_client_order_id=params.client_order_id
                                )
                            except Exception:
                                existing_order = None
                        
                        if existing_order:
                            existing_status = existing_order.get('status', '')
                            existing_filled = float(existing_order.get('executedQty', 0))
                            if existing_status in ['FILLED', 'CANCELED', 'EXPIRED', 'REJECTED']:
                                logger.info(f"[下单重试] {params.symbol} 订单已{existing_status}，跳过重试")
                                if existing_status == 'FILLED':
                                    return OrderResult(
                                        success=True,
                                        order_id=existing_order.get('orderId', ''),
                                        client_order_id=params.client_order_id,
                                        symbol=params.symbol,
                                        status=OrderStatus.FILLED,
                                        filled_quantity=existing_filled,
                                        avg_price=float(existing_order.get('avgPrice', 0)),
                                        message=f'订单已成交（重试发现）',
                                        params=params
                                    )
                                break
                            if existing_status == 'NEW':
                                # 先取消旧的NEW订单，再重试
                                try:
                                    self.cancel_order(params.symbol, client_order_id=params.client_order_id)
                                    logger.info(f"[下单重试] {params.symbol} 已取消旧NEW订单，准备重试")
                                except Exception as e:
                                    logger.warning(f"[下单重试] {params.symbol} 取消旧订单失败: {e}")
                            if existing_filled > 0:
                                params.quantity = max(params.quantity - existing_filled, 0)
                                logger.info(f"[下单重试] {params.symbol} 订单已部分成交 {existing_filled}，减少数量至 {params.quantity}")
                                if params.quantity <= 0:
                                    logger.info(f"[下单重试] {params.symbol} 订单已全部成交，跳过重试")
                                    break
                        
                        # 重新下单
                        result = self._client.create_order(
                            symbol=params.symbol,
                            side=params.side.value,
                            order_type=params.type.value,
                            quantity=params.quantity,
                            price=params.price,
                            stop_price=effective_stop_price,
                            reduce_only=params.reduce_only,
                            position_side=params.position_side,
                            leverage=params.leverage,
                            **{'newClientOrderId': params.client_order_id} if params.client_order_id else {}
                        )
                        
                        logger.info(f"[下单重试成功] {params.symbol} API返回: orderId={result.get('orderId')}, status={result.get('status')}")
                        
                        order_result = OrderResult(
                            success=True,
                            order_id=result.get('orderId', ''),
                            client_order_id=params.client_order_id,
                            symbol=params.symbol,
                            status=OrderStatus.from_string(result.get('status'), OrderStatus.NEW),
                            filled_quantity=float(result.get('executedQty', 0)),
                            avg_price=float(result.get('avgPrice', 0)),
                            message=f'订单创建成功（重试{retry_count}次）',
                            params=params
                        )
                        
                        # 记录订单
                        with self._order_lock:
                            self._orders[params.client_order_id] = OrderInfo(
                                params=params,
                                result=order_result,
                                retry_count=retry_count
                            )
                            
                        # 发送事件
                        self._event_bus.publish(OrderEvent(
                            event_type=EventType.ORDER_CREATED,
                            order_id=order_result.order_id,
                            symbol=params.symbol,
                            side=params.side.value,
                            status=order_result.status.value,
                            strategy=params.strategy,
                            data={
                                'quantity': params.quantity,
                                'price': params.price,
                                'stop_loss_price': params.stop_loss_price,
                                'take_profit_price': params.take_profit_price,
                                'retry_count': retry_count
                            }
                        ))
                        
                        return order_result
                        
                    except Exception as retry_e:
                        logger.warning(f"订单创建重试失败 {retry_count}/{max_retries}: {params.symbol}, {retry_e}")
                        retry_delay *= 2  # 指数退避
                
                # 重试次数用完
                logger.error(f"订单创建重试失败（已达最大重试次数）: {params.symbol}")
            
            # 记录失败的订单
            order_info = OrderInfo(
                params=params,
                result=OrderResult(
                    success=False,
                    message=str(e),
                    params=params
                ),
                retry_count=retry_count
            )
            
            with self._order_lock:
                self._orders[params.client_order_id] = order_info
                
            return OrderResult(
                success=False,
                message=f"订单创建失败: {str(e)}",
                params=params
            )
            
    def cancel_order(self, symbol: str, order_id: str = None, client_order_id: str = None) -> bool:
        """
        撤销订单

        Args:
            symbol: 币种
            order_id: 订单ID（二选一）
            client_order_id: 客户端订单ID（二选一）

        Returns:
            是否成功（统一返回bool类型）
        """
        if not self._client:
            logger.error("未设置API客户端")
            return False

        try:
            self._client.cancel_order(
                symbol=symbol,
                order_id=order_id,
                orig_client_order_id=client_order_id
            )

            logger.info(f"订单已撤销: {symbol} {order_id or client_order_id}")

            # 更新内部状态：从订单记录中移除
            with self._order_lock:
                # 使用client_order_id或order_id查找并移除订单
                key_to_remove = None
                if client_order_id and client_order_id in self._orders:
                    key_to_remove = client_order_id
                elif order_id:
                    # 如果只有order_id，需要遍历查找
                    for key, order_info in self._orders.items():
                        if order_info.result and order_info.result.order_id == order_id:
                            key_to_remove = key
                            break

                if key_to_remove:
                    del self._orders[key_to_remove]
                    logger.debug(f"已从内部状态移除订单: {key_to_remove}")

            return True

        except Exception as e:
            logger.exception(f"撤销订单失败: {symbol}, {e}")
            return False

    def execute_order(self, params: OrderParams) -> OrderResult:
        """
        执行订单（支持限价、市价、限价转市价三种模式）
        
        Args:
            params: 订单参数
            
        Returns:
            OrderResult
        """
        # 创建副本避免变异原始参数
        params = replace(params)
        order_mode = params.order_mode
        
        if order_mode == OrderMode.MARKET_ONLY:
            # 仅市价
            return self._execute_market_order(params)
        elif order_mode == OrderMode.LIMIT_ONLY:
            # 仅限价
            return self._execute_limit_order(params)
        elif order_mode == OrderMode.LIMIT_TO_MARKET:
            # 限价转市价
            return self._execute_limit_to_market(params)
        else:
            return OrderResult(success=False, message=f"未知的订单模式: {order_mode}", params=params)

    def _execute_market_order(self, params: OrderParams) -> OrderResult:
        """仅市价单"""
        params.type = OrderType.MARKET
        return self.create_order(params)

    def _execute_limit_order(self, params: OrderParams) -> OrderResult:
        """仅限价单"""
        params.type = OrderType.LIMIT
        if not params.price:
            return OrderResult(success=False, message="限价单需要指定价格", params=params)
        return self.create_order(params)

    def _execute_limit_to_market(self, params: OrderParams) -> OrderResult:
        """限价转市价"""
        # 第一步：尝试限价单
        params.type = OrderType.LIMIT
        if not params.price:
            return OrderResult(success=False, message="限价转市价需要指定价格", params=params)
        
        logger.info(f"[限价转市价] {params.symbol} 开始下单流程")
        logger.debug(f"[限价转市价] 参数: quantity={params.quantity}, price={params.price}, side={params.side.value}")
        
        result = self.create_order(params)
        
        if not result.success:
            # 限价单创建失败，直接改用市价
            logger.warning(f"[限价转市价] {params.symbol} 限价单创建失败，改用市价单")
            return self._execute_market_order(params)
        
        order_id = result.order_id
        limit_timeout = params.limit_timeout
        original_qty = params.quantity
        
        logger.info(f"[限价转市价] {params.symbol} 限价单已创建: orderId={order_id}")
        
        # 等待成交
        for i in range(limit_timeout):
            time.sleep(1)
            status = self.get_order_status(params.symbol, order_id=order_id)

            logger.debug(f"[限价转市价] {params.symbol} 第{i+1}/{limit_timeout}秒检查: status={status.status if status else 'N/A'}")

            if status and status.status == OrderStatus.FILLED:
                logger.info(f"[限价转市价] {params.symbol} 限价单已完全成交: orderId={order_id}")
                result.filled_quantity = original_qty
                result.avg_price = status.avg_price
                result.status = OrderStatus.FILLED
                # 限价单在create_order时为NEW状态，不会发布ORDER_FILLED，需在此补发
                self._event_bus.publish(OrderEvent(
                    event_type=EventType.ORDER_FILLED,
                    order_id=order_id,
                    symbol=params.symbol,
                    side=params.side.value,
                    status=OrderStatus.FILLED.value,
                    strategy=params.strategy,
                    data={
                        'quantity': original_qty,
                        'price': status.avg_price,
                        'filled_quantity': original_qty,
                        'stop_loss_price': params.stop_loss_price,
                        'take_profit_price': params.take_profit_price,
                    }
                ))
                return result
            elif status and status.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                logger.warning(f"[限价转市价] {params.symbol} 限价单状态异常: {status.status}")
                break
            elif status and status.status == OrderStatus.PARTIALLY_FILLED:
                logger.info(f"[限价转市价] {params.symbol} 部分成交: {status.filled_quantity}/{original_qty}")
            elif status and status.status == OrderStatus.NEW:
                # NEW状态：订单刚创建但未成交，继续等待
                logger.debug(f"[限价转市价] {params.symbol} 订单仍为NEW状态，继续等待")
                continue
        
        # 限价单未完全成交，取消并改用市价补足剩余
        executed_qty = 0
        if limit_timeout > 0 and status:
            executed_qty = status.filled_quantity
        
        remaining_qty = original_qty - executed_qty
        logger.info(f"[限价转市价] {params.symbol} 限价单未完全成交: 已成交={executed_qty}, 剩余={remaining_qty}")
        
        cancel_ok = self.cancel_order(params.symbol, order_id=order_id)
        if not cancel_ok:
            # 撤单失败：检查订单是否已成交
            try:
                final_status = self._client.get_order(symbol=params.symbol, order_id=order_id)
                final_status_str = final_status.get('status', '')
                final_filled = float(final_status.get('executedQty', 0))
                if final_status_str == 'FILLED':
                    logger.info(f"[限价转市价] {params.symbol} 限价单已成交: filled={final_filled}")
                    result.filled_quantity = final_filled
                    result.avg_price = float(final_status.get('avgPrice', 0))
                    result.message = '限价单已成交'
                    return result
                else:
                    # 未成交但撤单失败，限价单可能已过期/被其他线程处理，仍用市价补足
                    logger.warning(f"[限价转市价] {params.symbol} 取消限价单失败(status={final_status_str})，仍尝试市价补足")
            except Exception:
                logger.warning(f"[限价转市价] {params.symbol} 取消限价单失败且无法查询状态，仍尝试市价补足")
        else:
            logger.info(f"[限价转市价] {params.symbol} 限价单已取消")
        
        # 使用市价单补足剩余数量
        if remaining_qty > 0.001:
            logger.info(f"[限价转市价] {params.symbol} 使用市价单补足剩余: {remaining_qty}")
            params.quantity = remaining_qty
            market_result = self._execute_market_order(params)
            
            # 合并结果（加权混合均价）
            if market_result.success:
                limit_avg = status.avg_price if status else 0
                total_qty = executed_qty + market_result.filled_quantity
                blended_avg = (executed_qty * limit_avg + market_result.filled_quantity * market_result.avg_price) / total_qty if total_qty > 0 else 0
                market_result.filled_quantity = total_qty
                market_result.avg_price = blended_avg
                logger.info(f"[限价转市价] {params.symbol} 市价单成功，总成交: {market_result.filled_quantity}, 均价: {blended_avg:.2f}")
                return market_result
            else:
                # 市价单也失败，返回已成交部分
                result.filled_quantity = executed_qty
                result.message = f"市价单失败，已成交部分: {executed_qty}"
                return result
        else:
            logger.info(f"[限价转市价] {params.symbol} 限价单已完成")
            result.filled_quantity = original_qty
            return result

    def get_order_status(self, symbol: str, order_id: str = None, client_order_id: str = None) -> Optional[OrderResult]:
        """查询订单状态"""
        if not self._client:
            return None
            
        try:
            # 使用 order_id 参数（不是 orderId）
            result = self._client.get_order(
                symbol=symbol,
                order_id=order_id,
                orig_client_order_id=client_order_id
            )
            
            return OrderResult(
                success=True,
                order_id=result.get('orderId', ''),
                client_order_id=result.get('clientOrderId', ''),
                symbol=symbol,
                status=OrderStatus.from_string(result.get('status'), OrderStatus.NEW),
                filled_quantity=float(result.get('executedQty', 0)),
                avg_price=float(result.get('avgPrice', 0)),
            )
            
        except Exception as e:
            logger.exception(f"查询订单状态失败: {symbol}, {e}")
            return None
            
    def get_open_orders(self, symbol: str = None) -> List[OrderResult]:
        """获取未完成订单"""
        if not self._client:
            return []
            
        try:
            orders = self._client.get_open_orders(symbol=symbol)
            return [
                OrderResult(
                    success=True,
                    order_id=o.get('orderId', ''),
                    client_order_id=o.get('clientOrderId', ''),
                    symbol=o.get('symbol', ''),
                    status=OrderStatus.from_string(o.get('status'), OrderStatus.NEW),
                    filled_quantity=float(o.get('executedQty', 0)),
                )
                for o in orders
            ]
        except Exception as e:
            logger.exception(f"获取未完成订单失败: {e}")
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._order_lock:
            total = len(self._orders)
            active = sum(1 for o in self._orders.values() if o.result.is_active)
            
        return {
            'total_orders': total,
            'active_orders': active,
            'strategy_callbacks': list(self._strategy_callbacks.keys()),
        }


# 便捷工厂函数（向后兼容）
def get_order_engine(client=None) -> OrderEngine:
    """获取订单引擎实例（向后兼容函数）

    注意：新代码应直接通过构造函数注入。
    """
    return OrderEngine(client)
