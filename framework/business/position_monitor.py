# -*- coding: utf-8 -*-
"""
持仓监控

统一持仓状态管理

特性：
1. 实时持仓同步
2. 多策略持仓隔离
3. 盈亏计算
4. 止盈止损追踪
5. 持仓历史记录
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from framework.core.logger import get_logger, track_performance
from framework.core.events import EventBus, Event, EventType, PositionEvent
from framework.core.config import get_main_config
# 修复：统一使用framework.shared.enums中的枚举定义
from framework.shared.enums import PositionSide as EnumPositionSide, PositionStatus as EnumPositionStatus
from framework.business.order_engine import OrderEngine, OrderParams
from framework.business.order_engine import OrderSide, OrderType, OrderMode
from framework.business.trade_recorder import TradeRecorder

logger = get_logger('position_monitor')


# 为保持向后兼容，保留本地别名
PositionSide = EnumPositionSide
PositionStatus = EnumPositionStatus


@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    strategy: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    leverage: int
    margin: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_profit_pct: float = 0.0
    max_loss_pct: float = 0.0
    status: PositionStatus = PositionStatus.ACTIVE
    opened_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    close_reason: str = ''
    
    # 止盈止损
    stop_loss_price: Optional[float] = None
    stop_limit_price: Optional[float] = None  # 止损条件单的限价（创建限价单时的价格，穿越此价即市价兜底）
    take_profit_price: Optional[float] = None
    drawdown_threshold: float = 20.0  # 回撤止盈阈值，默认20%
    stop_loss_algo_id: Optional[int] = None  # 止损单AlgoID，用于取消
    static_stop_loss_price: Optional[float] = None  # L2静态地板止损价(追踪止损以此为底)
    trailing_last_update: float = 0.0  # 上次改交易所追踪止损单时间(节流用)

    # 趋势相关（用于趋势减弱离场和趋势反转止盈）
    entry_adx: Optional[float] = None  # 入场时的ADX值
    hold_bars: int = 0  # 持仓K线数
    entry_trend: Optional[str] = None  # 入场时的趋势方向 ('LONG' or 'SHORT')
    mae_pct: float = 0.0  # MAE（最大浮亏）

    # Trail_SL 状态（增量式出场追踪）
    trail_activated: bool = False  # Trail是否已激活
    trail_sl_price: float = 0.0  # Trail止损价格
    trail_step_pct: float = 3.0  # Trail步进（pnl%）
    tp_fired: bool = False  # TP drawback是否触发过（用于部分平仓后保本计算）
    remaining_pct: float = 100.0  # 剩余仓位比例（%）

    # 策略元数据（signal_type_str, entry_res, entry_sup, trend_type）
    metadata: Dict = field(default_factory=dict)

    # 分批建仓
    is_batch_entry: bool = False
    batch_level: int = 0  # 当前批次
    
    @property
    def position_value(self) -> float:
        """持仓价值"""
        return self.quantity * self.current_price
    
    @property
    def pnl_pct(self) -> Optional[float]:
        """盈亏比例（含杠杆）"""
        if self.entry_price <= 0:
            return None
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) / self.entry_price * self.leverage * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * self.leverage * 100
            
    @property
    def hold_time(self) -> float:
        """持仓时间（秒）"""
        return time.time() - self.opened_at
        
    @property
    def hold_hours(self) -> float:
        """持仓时间（小时）"""
        return self.hold_time / 3600
        
    def update_price(self, current_price: float):
        """更新价格"""
        self.current_price = current_price
        self.updated_at = time.time()
        
        # 更新盈亏
        if self.side == PositionSide.LONG:
            unrealized = (current_price - self.entry_price) * self.quantity
            # MAE for LONG: worst point = lowest price during holding
            # 记录相对于入场价的最低点
            if self.entry_price > 0:
                low_point_pct = (current_price - self.entry_price) / self.entry_price * self.leverage * 100
                if low_point_pct < self.mae_pct:
                    self.mae_pct = low_point_pct
        else:
            unrealized = (self.entry_price - current_price) * self.quantity
            # MAE for SHORT: worst point = highest price during holding
            if self.entry_price > 0:
                high_point_pct = (self.entry_price - current_price) / self.entry_price * self.leverage * 100
                if high_point_pct < self.mae_pct:
                    self.mae_pct = high_point_pct
            
        self.unrealized_pnl = unrealized
        
        # 更新最大盈利/亏损
        current_pct = self.pnl_pct
        if current_pct > self.max_profit_pct:
            self.max_profit_pct = current_pct
        if current_pct < self.max_loss_pct:
            self.max_loss_pct = current_pct
            
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'strategy': self.strategy,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'leverage': self.leverage,
            'pnl_pct': self.pnl_pct,
            'unrealized_pnl': self.unrealized_pnl,
            'hold_hours': self.hold_hours,
            'status': self.status.value,
        }


class PositionMonitor:
    """
    持仓监控
    
    统一管理所有持仓
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, client=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, client=None):
        if self._initialized:
            return
            
        self._client = client
        self.config = get_main_config()
        
        # 持仓数据: {symbol: PositionInfo}
        self._positions: Dict[str, PositionInfo] = {}
        self._position_lock = threading.RLock()
        
        # 策略持仓索引: {strategy: {symbol: PositionInfo}}
        self._strategy_positions: Dict[str, Dict[str, PositionInfo]] = {}
        
        # 事件总线 - 使用单例
        from framework.core.events import get_event_bus
        self._event_bus = get_event_bus()
        
        # 监控线程
        self._monitor_thread = None
        self._running = False
        self._monitor_interval = 15  # 秒
        
        # 价格回调
        self._price_callbacks: List[Callable] = []
        
        # 订单引擎
        self._order_engine = OrderEngine(client)

        # 交易记录器（共享实例）
        self._trade_recorder = TradeRecorder.get_instance()
        self._trade_recorder.set_client(client)

        # 策略调度器（用于调用策略的止盈止损逻辑）
        self._scheduler = None

        # 止损单失败重试追踪: {symbol: last_fail_timestamp}
        self._last_stop_fail_time: Dict[str, float] = {}

        self._initialized = True
        logger.info("持仓监控已初始化")

    def set_client(self, client):
        """设置API客户端"""
        self._client = client
        self._order_engine.set_client(client)

    def set_order_engine(self, order_engine):
        """设置订单引擎（使用框架共享的实例，避免双实例问题）"""
        self._order_engine = order_engine

    def set_scheduler(self, scheduler):
        """设置策略调度器"""
        self._scheduler = scheduler
        
    def start_monitor(self):
        """启动持仓监控"""
        if self._running:
            return
            
        self._running = True
        
        # 启动前先同步一次持仓（检测启动前的持仓）
        self.sync_from_api()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"持仓监控已启动: 价格检查每3秒, 持仓同步每15秒")
        
    def stop_monitor(self):
        """停止持仓监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("持仓监控已停止")
        
    def _monitor_loop(self):
        """监控循环：高频价格检查 + 低频完整同步
        
        高频（每3秒）：更新价格 + 止盈止损安全网
        低频（每15秒）：API持仓同步 + 止损单核对 + 币安持仓核对
        """
        FAST_INTERVAL = 3   # 高频价格检查间隔（秒）
        FULL_INTERVAL = 15  # 完整同步间隔（秒），必须是FAST_INTERVAL整数倍
        ticks_per_full = FULL_INTERVAL // FAST_INTERVAL  # 每几次tick做一次完整同步

        tick = 0
        verify_counter = 0
        while self._running:
            try:
                is_full_tick = (tick % ticks_per_full == 0)

                if is_full_tick:
                    # 完整同步：API持仓、止损单核对、币安持仓核对
                    self.sync_from_api()
                    self._verify_stop_loss_orders()
                    verify_counter += 1
                    if verify_counter >= 2:
                        self._verify_positions_with_binance()
                        verify_counter = 0

                # 每次tick：更新价格 + 安全网（轻量，无API调用）
                if self._positions:
                    self._update_all_prices()
                self._check_stop_loss_take_profit()

            except Exception as e:
                logger.exception(f"持仓监控异常: {e}")

            tick += 1
            time.sleep(FAST_INTERVAL)
    
    def _verify_positions_with_binance(self):
        """核对本地持仓与币安实际持仓"""
        if not self._client:
            return
        
        try:
            # 1. 获取币安实际持仓
            api_positions = self._client.get_positions()
            api_dict = {p['symbol']: p for p in api_positions if abs(float(p.get('position_amt', 0))) > 1e-8}
            
            logger.debug(f"【持仓核对】币安持仓数: {len(api_dict)}, 本地持仓数: {len(self._positions)}")
            
            # 2. 核对本地持仓 - 直接以币安数据为准
            with self._position_lock:
                # 构建币安持仓映射
                local_symbols = set(self._positions.keys())
                api_symbols = set(api_dict.keys())
                
                # 币安有、本地也有：更新为币安数据
                for symbol in local_symbols & api_symbols:
                    api_pos = api_dict[symbol]
                    pos = self._positions[symbol]
                    api_qty = abs(float(api_pos.get('position_amt', 0)))
                    
                    # 保护部分平仓的 _remaining_ratio：差异<5%时保留本地值
                    if hasattr(pos, '_remaining_ratio') and pos.quantity > 0:
                        ratio_diff = abs(pos.quantity - api_qty) / pos.quantity
                        if ratio_diff < 0.05:
                            api_qty = pos.quantity
                    
                    # 保存本地 mx/mn 防止币安同步时重置
                    local_mx = pos.max_profit_pct
                    local_mn = pos.mae_pct
                    
                    # 直接用币安数据覆盖
                    pos.quantity = api_qty
                    pos.entry_price = float(api_pos.get('entry_price', pos.entry_price))
                    pos.unrealized_pnl = float(api_pos.get('unrealized_pnl', 0))
                    pos.margin = float(api_pos.get('initialMargin', pos.margin))
                    pos.updated_at = time.time()
                    
                    # 恢复 mx/mn（取保守值：mx取较大，mn取较小）
                    pos.max_profit_pct = max(local_mx, pos.max_profit_pct)
                    pos.mae_pct = min(local_mn, pos.mae_pct)
                
                # 币安无、本地有：清理本地持仓
                for symbol in local_symbols - api_symbols:
                    pos = self._positions.pop(symbol, None)
                    if pos:
                        if pos.strategy in self._strategy_positions:
                            self._strategy_positions[pos.strategy].pop(symbol, None)
                        close_price = pos.current_price if pos.current_price > 0 else pos.entry_price
                        reason = "止损触发" if pos.pnl_pct < 0 else "止盈触发"
                        self._trade_recorder.record_close(symbol, close_price, reason)
                        if self._event_bus:
                            self._event_bus.publish(PositionEvent(
                                event_type=EventType.POSITION_CLOSED,
                                symbol=symbol,
                                strategy=pos.strategy,
                                side=pos.side.value,
                                quantity=pos.quantity,
                                priority=10,
                                data={'position': pos},
                            ))
                    logger.info(f"【同步清理】{symbol} 币安无持仓，本地已移除")
                
                # 币安有、本地无：检测遗漏的新持仓
                for symbol in api_symbols - local_symbols:
                    api_pos = api_dict[symbol]
                    qty = abs(float(api_pos.get('position_amt', 0)))
                    if qty > 1e-8:
                        entry_price = float(api_pos.get('entry_price', 0))
                        leverage = int(api_pos.get('leverage', 20))
                        position_amt = float(api_pos.get('position_amt', 0))
                        side = PositionSide.LONG if position_amt > 0 else PositionSide.SHORT
                        margin = qty * entry_price / leverage if leverage > 0 else 0

                        # 先检查是否为程序持仓（防止 race condition 导致刚开仓就被当作外部持仓）
                        is_program, prog_strategy = self._check_if_program_position(symbol)
                        if is_program:
                            logger.info(f"{symbol} 判定为程序持仓（订单前缀匹配），注册到本地监控")
                            self._register_synced_position(symbol, side, qty, entry_price, margin, leverage, strategy=prog_strategy if prog_strategy != 'unknown' else None)
                            if symbol in self._positions:
                                self._place_stop_loss_order(symbol, self._positions[symbol])
                            continue
                        logger.warning(f"【检测遗漏】{symbol} 币安有持仓但本地无记录")
                        self._handle_external_position(symbol, side, qty, entry_price, margin, leverage)
            
            # 4. 核对止损单状态
            self._verify_stop_loss_orders()
            
        except Exception as e:
            logger.error(f"【核对失败】核对持仓异常: {e}")
    
    def _verify_stop_loss_orders(self):
        """核对止损单是否仍在币安，被外部取消时自动补挂"""
        if not self._client or not self._positions:
            return
        
        try:
            open_orders = self._client.get_algo_open_orders(symbol='')
            # 统一转为字符串比较（API可能返回int或str）
            active_algo_ids = {str(order.get('algoId', '')) for order in open_orders if order.get('algoId')}
            
            with self._position_lock:
                for symbol, pos in list(self._positions.items()):
                    if not pos.stop_loss_algo_id or pos.status != PositionStatus.ACTIVE:
                        continue
                    # 跳过最近10秒内刚挂出的止损单（避免竞态条件：API尚未索引新订单）
                    last_update = getattr(pos, '_stop_loss_last_placed', 0)
                    if time.time() - last_update < 10:
                        logger.debug(f"{symbol} 止损单刚挂出，跳过本次核对")
                        continue
                    if str(pos.stop_loss_algo_id) not in active_algo_ids:
                        logger.warning(f"【止损单丢失】{symbol} 止损单已不在币安(algoId={pos.stop_loss_algo_id})，准备补挂")
                        old_algo_id = pos.stop_loss_algo_id  # 先保留旧algoId，补挂失败后恢复
                        pos.stop_loss_algo_id = None
                        try:
                            if not self._place_stop_loss_order(symbol, pos):
                                # 挂单失败（冷却期/API错误/参数校验不通过）
                                raise RuntimeError(f"{symbol} _place_stop_loss_order返回False")
                            logger.info(f"【止损单补挂】{symbol} 已重新挂出止损单(algoId={pos.stop_loss_algo_id})")
                        except Exception as e:
                            # 补挂失败，恢复旧algoId，让下次验证周期继续重试
                            if pos.stop_loss_algo_id is None:
                                pos.stop_loss_algo_id = old_algo_id
                            logger.error(f"【止损单补挂失败】{symbol}: {e}")
                            
        except Exception as e:
            logger.warning(f"【止损单核对】查询失败: {e}")
            
    @track_performance('sync_from_api')
    def sync_from_api(self) -> bool:
        """从API同步持仓"""
        if not self._client:
            logger.warning("sync_from_api: client未初始化")
            return False

        try:
            logger.debug("sync_from_api: 开始获取持仓")
            api_positions = self._client.get_positions()
            logger.info(f"sync_from_api: 获取到{len(api_positions)}个持仓")

            if not api_positions:
                logger.debug("sync_from_api: 无持仓")
                return True

            # 收集需要在锁外执行的网络操作
            external_positions = []
            program_positions_needing_sl = []

            with self._position_lock:
                # 标记所有持仓为待验证
                active_symbols = set()

                for api_pos in api_positions:
                    position_amt = float(api_pos.get('position_amt', 0))
                    if abs(position_amt) < 1e-8:
                        continue

                    symbol = api_pos.get('symbol')
                    if not symbol:
                        continue

                    # 调试日志：检查API返回的数量
                    logger.debug(f"API返回持仓: {symbol}, position_amt={position_amt}, entry_price={api_pos.get('entry_price')}, leverage={api_pos.get('leverage')}")

                    active_symbols.add(symbol)

                    # 检查本地是否有该持仓
                    if symbol in self._positions:
                        # 更新现有持仓，保留本地 mx/mn
                        pos = self._positions[symbol]
                        local_mx = pos.max_profit_pct
                        local_mn = pos.mae_pct
                        pos.quantity = abs(position_amt)
                        pos.entry_price = float(api_pos.get('entry_price', pos.entry_price))
                        pos.unrealized_pnl = float(api_pos.get('unrealized_pnl', 0))
                        pos.updated_at = time.time()
                        pos.max_profit_pct = max(local_mx, pos.max_profit_pct)
                        pos.mae_pct = min(local_mn, pos.mae_pct)
                    else:
                        # 新持仓 - 检测是否为外部持仓
                        logger.info(f"检测到新持仓: {symbol}")

                        # 判断是否为程序持仓
                        is_program, prog_strategy = self._check_if_program_position(symbol)

                        # 统一提取变量，供外部/程序两分支使用
                        entry_price = float(api_pos.get('entry_price', 0))
                        leverage = int(api_pos.get('leverage', 20))
                        margin = abs(position_amt) * entry_price / leverage if leverage > 0 else 0
                        side = PositionSide.LONG if position_amt > 0 else PositionSide.SHORT
                        qty = abs(position_amt)

                        if not is_program:
                            external_positions.append((symbol, side, qty, entry_price, margin, leverage, position_amt))
                        else:
                            logger.info(f"{symbol} 判定为程序持仓，正在注册本地监控...")
                            unrealized_pnl = float(api_pos.get('unrealized_pnl', 0))
                            self._register_synced_position(symbol, side, qty, entry_price, margin, leverage, unrealized_pnl, strategy=prog_strategy if prog_strategy != 'unknown' else None)
                            if symbol in self._positions:
                                program_positions_needing_sl.append((symbol, self._positions[symbol]))

                # 清理已平仓的持仓（币安algo单触发后持仓消失）
                stale_symbols = set(self._positions.keys()) - active_symbols
                for stale_symbol in stale_symbols:
                    if stale_symbol in self._positions:
                        pos = self._positions.pop(stale_symbol)
                        if pos.strategy in self._strategy_positions:
                            self._strategy_positions[pos.strategy].pop(stale_symbol, None)
                        # 推断平仓原因：PnL<0→止损，PnL>0→止盈
                        reason = "止损触发" if pos.pnl_pct < 0 else "止盈触发"
                        close_price = pos.current_price if pos.current_price > 0 else pos.entry_price
                        self._trade_recorder.record_close(stale_symbol, close_price, reason)
                        # 发送平仓事件
                        if self._event_bus:
                            self._event_bus.publish(PositionEvent(
                                event_type=EventType.POSITION_CLOSED,
                                symbol=stale_symbol,
                                strategy=pos.strategy,
                                side=pos.side.value,
                                quantity=pos.quantity,
                                priority=10,
                                data={'position': pos},
                            ))
                        logger.info(f"【持仓已平仓】{stale_symbol} AlgoOrder触发")

            # 锁外执行网络I/O（避免阻塞其他持仓操作）
            for symbol, side, qty, entry_price, margin, leverage, position_amt in external_positions:
                logger.info(f"{symbol} 判定为外部持仓，正在设置止盈止损...")
                logger.info(f"【外部持仓原始数据】{symbol}: position_amt={position_amt}, entry_price={entry_price}, leverage={leverage}")
                self._handle_external_position(symbol, side, qty, entry_price, margin, leverage)

            for symbol, pos in program_positions_needing_sl:
                self._place_stop_loss_order(symbol, pos)

            return True

        except Exception as e:
            logger.exception(f"同步持仓失败: {e}")
            return False
            
    def _update_all_prices(self):
        """更新所有持仓价格（批量取价，减少API调用）"""
        if not self._client or not self._positions:
            return

        try:
            # 一次批量获取所有币种价格（weight=5）
            prices = self._client.get_all_prices()
            if not prices:
                logger.debug("批量取价返回空")
                return
        except Exception as e:
            logger.warning(f"批量取价失败，回退到逐个取价: {e}")
            prices = {}

        for symbol, pos in list(self._positions.items()):
            try:
                current_price = prices.get(symbol)
                if current_price is None:
                    # 批量取价遗漏时回退到逐个取价
                    current_price = self._client.get_ticker_price(symbol)
                if current_price:
                    pos.update_price(current_price)
                    
                    # 通知价格回调
                    for callback in self._price_callbacks:
                        try:
                            callback(symbol, current_price, pos)
                        except Exception as e:
                            logger.exception(f"价格回调失败: {callback.__name__}")
                            
            except Exception as e:
                logger.debug(f"更新价格失败: {symbol}, {e}")

    def _detect_strategy_from_orders(self, symbol: str) -> str:
        """从最近的Binance订单clientOrderId识别策略名"""
        if not self._client or not self._scheduler:
            return 'unknown'
        try:
            from trading.binance_client import BNFF_ORDER_PREFIX
            orders = self._client.get_all_orders(symbol, limit=10)
            if not orders:
                return 'unknown'
            # 检查最近的订单，优先匹配策略名前缀
            for order in orders:
                cid = order.get('clientOrderId', '')
                for sname in self._scheduler._strategies:
                    if cid.startswith(f"{sname}_"):
                        return sname
            # 回退：BNFF_前缀无法识别策略
            return 'unknown'
        except Exception:
            return 'unknown'

    def _check_if_program_position(self, symbol: str) -> Tuple[bool, str]:
        """
        检查是否是程序开仓，并返回策略名称

        识别逻辑：
        1. 检查内存中的策略持仓记录
        2. 检查Binance订单的clientOrderId是否包含策略名前缀

        Returns:
            Tuple[bool, str]: (是否程序持仓, 策略名称)
        """
        # 方法1：检查策略执行记录（排除 external 策略）
        for strategy_name, positions in self._strategy_positions.items():
            if strategy_name == 'external':
                continue
            if symbol in positions:
                logger.debug(f"{symbol} 在策略 {strategy_name} 的持仓记录中，确认为程序持仓")
                return True, strategy_name

        # 方法2：检查Binance订单的clientOrderId前缀（最可靠）
        try:
            if self._client:
                from trading.binance_client import BNFF_ORDER_PREFIX
                orders = self._client.get_all_orders(symbol, limit=50)
                if orders:
                    for order in orders:
                        client_order_id = order.get('clientOrderId', '')
                        # 匹配 BNFF_ 前缀
                        if client_order_id.startswith(BNFF_ORDER_PREFIX):
                            logger.info(f"{symbol} 发现程序订单(clientOrderId={client_order_id})，确认为程序持仓")
                            return True, 'unknown'
                        # 匹配 策略名_ 前缀（如 hf_1712345678）
                        for sname in self._scheduler._strategies if self._scheduler else []:
                            if client_order_id.startswith(f"{sname}_"):
                                logger.info(f"{symbol} 发现策略{sname}订单(clientOrderId={client_order_id})")
                                return True, sname
        except Exception as e:
            logger.debug(f"检查程序持仓失败: {symbol}, {e}")

        return False, ''

    def _handle_external_position(self, symbol: str, side: PositionSide, quantity: float, entry_price: float, margin: float, leverage: int = 20):
        """处理外部持仓"""
        cfg = get_main_config()
        delay_ratio = getattr(cfg, 'delay_ratio', 0.003) * 100  # 转换为百分比

        # 外部持仓参数
        stop_loss_trigger = getattr(cfg, 'external_stop_loss_pct', -30.0)  # 亏损30%
        drawdown_threshold = getattr(cfg, 'external_drawdown_threshold', 15.0)  # 回撤15%
        breakeven_threshold = getattr(cfg, 'external_breakeven_threshold', 15.0)  # 保本15%

        logger.info(f"【外部持仓】{symbol} 使用特殊止盈止损规则: SL={stop_loss_trigger}%, TP回撤={drawdown_threshold}%, 保本={breakeven_threshold}%, DELAY_RATIO={delay_ratio}%")

        # 计算止损价格
        stop_loss_trigger_abs = abs(stop_loss_trigger)
        if side == PositionSide.LONG:
            stop_loss_price = entry_price * (1 - stop_loss_trigger_abs / 100 / leverage)  # 30%杠杆盈亏 = 1.5%价格跌幅
        else:
            stop_loss_price = entry_price * (1 + stop_loss_trigger_abs / 100 / leverage)

        logger.info(f"【自动设置止损】止损={stop_loss_price:.6f} ({stop_loss_trigger}%), 入场价={entry_price:.6f}")

        # 获取当前价格
        current_price = None
        if self._client:
            try:
                current_price = self._client.get_ticker_price(symbol)
            except Exception as e:
                logger.debug(f"获取当前价格失败: {e}")

        # 检查是否已经触及止损（亏损超过阈值）- 如果已超限，立即平仓
        if current_price and current_price > 0:
            if side == PositionSide.LONG:
                loss_pct = (entry_price - current_price) / entry_price * 100 * leverage
            else:
                loss_pct = (current_price - entry_price) / entry_price * 100 * leverage

            if loss_pct >= stop_loss_trigger_abs:
                logger.warning(f"⚠️ {symbol} 当前亏损{loss_pct:.1f}%已超过止损阈值{stop_loss_trigger_abs}%，立即平仓")
                # 在交易所上真正平仓
                if self.close_position_on_exchange(symbol, "外部持仓亏损超限"):
                    # 平仓成功后，从本地记录中移除
                    self.close_position(symbol, "外部持仓亏损超限")
                return

        # 创建持仓信息
        pos = PositionInfo(
            symbol=symbol,
            strategy='external',
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            leverage=leverage,
            margin=margin,
            stop_loss_price=stop_loss_price,
            take_profit_price=None,  # 外部持仓使用回撤止盈，不设置固定止盈价
            drawdown_threshold=drawdown_threshold,  # 回撤15%触发止盈
        )

        with self._position_lock:
            self._positions[symbol] = pos

            # 添加到策略索引
            if 'external' not in self._strategy_positions:
                self._strategy_positions['external'] = {}
            self._strategy_positions['external'][symbol] = pos

        # 自动挂止损单
        self._place_stop_loss_order(symbol, pos)

        logger.info(f"【外部持仓】{symbol} 已标记并挂止损单")

    def _register_synced_position(self, symbol: str, side: PositionSide, quantity: float, entry_price: float, margin: float, leverage: int, unrealized_pnl: float = 0, strategy: str = None):
        """注册同步发现的程序持仓到本地跟踪"""
        # 优先使用从 clientOrderId 提取的策略名
        detected_strategy = strategy if strategy else 'unknown'
        if detected_strategy == 'unknown' and self._scheduler:
            # 尝试从最近的订单clientOrderId识别策略（比盲目preferring hf更准确）
            detected_strategy = self._detect_strategy_from_orders(symbol)
            if detected_strategy == 'unknown':
                # 回退：遍历所有策略，匹配 _symbols（不preferring任何策略）
                for name, info in self._scheduler._strategies.items():
                    if info.instance and hasattr(info.instance, '_symbols') and symbol in info.instance._symbols:
                        detected_strategy = name
                        break

        # 检查max_positions限制（仍需注册，但发出警告）
        if detected_strategy != 'unknown' and self._scheduler:
            strategy_info = self._scheduler._strategies.get(detected_strategy)
            if strategy_info and strategy_info.config:
                current_count = len(self._strategy_positions.get(detected_strategy, {}))
                max_pos = strategy_info.config.max_positions
                if current_count >= max_pos:
                    logger.warning(f"⚠️ {symbol} 策略{detected_strategy}持仓数已达上限: {current_count}/{max_pos}，仍注册监控但阻止新开仓")

        pos = PositionInfo(
            symbol=symbol,
            strategy=detected_strategy,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            leverage=leverage,
            margin=margin,
            unrealized_pnl=unrealized_pnl,
        )
        with self._position_lock:
            self._positions[symbol] = pos
            if detected_strategy not in self._strategy_positions:
                self._strategy_positions[detected_strategy] = {}
            self._strategy_positions[detected_strategy][symbol] = pos
        logger.info(f"{symbol} 程序持仓已注册到本地监控 (策略={detected_strategy})")

        # 调用策略 on_position_opened 恢复策略属性（ExitState、signal_type等）
        if self._scheduler and detected_strategy != 'unknown':
            strategy_info = self._scheduler._strategies.get(detected_strategy)
            if strategy_info and strategy_info.instance:
                try:
                    strategy_info.instance.on_position_opened(pos)
                except Exception as e:
                    logger.warning(f"{symbol} on_position_opened恢复失败: {e}")

    def _place_stop_loss_order(self, symbol: str, pos: PositionInfo) -> bool:
        """挂止损单（先取消旧单再挂新单），失败时带重试退避与市价平仓

        Returns:
            True=挂单成功(或已市价平仓), False=跳过/失败
        """
        if not self._client or not pos.stop_loss_price:
            return False

        # ===== 重试退避：15秒内不重复尝试（避免API限流，同时不过长裸奔） =====
        last_fail = self._last_stop_fail_time.get(symbol, 0)
        if time.time() - last_fail < 15:
            logger.debug(f"{symbol} 止损单{int(time.time()-last_fail)}秒前刚失败过，跳过本次重试")
            return False

        try:
            # ===== 0. 检查数量合理性 =====
            if pos.quantity <= 0:
                logger.warning(f"⚠️ {symbol} 持仓数量异常，跳过挂单: {pos.quantity}")
                return False

            # ===== 1. 检查并取消币安上已有的止损单 =====
            try:
                open_orders = self._client.get_algo_open_orders(symbol)
                for order in open_orders:
                    algo_id = order.get('algoId')
                    if algo_id:
                        try:
                            self._client.cancel_algo_order(symbol, algo_id=algo_id)
                            logger.info(f"已取消币安上的旧止损单: {symbol} algoId={algo_id}")
                        except Exception as e:
                            logger.debug(f"取消止损单失败（可能已成交）: {e}")
            except Exception as e:
                logger.debug(f"检查币安挂单失败: {e}")

            # ===== 2. 先取消本地记录的旧止损单 =====
            if pos.stop_loss_algo_id:
                try:
                    self._client.cancel_algo_order(symbol, algo_id=pos.stop_loss_algo_id)
                    logger.info(f"已取消旧的止损单: {symbol} algoId={pos.stop_loss_algo_id}")
                except Exception as e:
                    logger.debug(f"取消旧止损单失败（可能已成交）: {e}")

            stop_side = 'SELL' if pos.side == PositionSide.LONG else 'BUY'
            position_side_str = 'LONG' if pos.side == PositionSide.LONG else 'SHORT'

            # 获取当前价格（用 mark_price 与 algo 单的 workingType=MARK_PRICE 保持一致）
            # 避免安检用 last_price 通过但 API 用 mark_price 拒绝（-2021）
            try:
                current_price = self._client.get_mark_price(symbol) if self._client else None
            except Exception:
                current_price = None
            if not current_price:
                current_price = self._client.get_price(symbol) if self._client else None
            if not current_price:
                current_price = pos.current_price

            # ===== 3. 检查止损触发价是否已被当前价格穿越 =====
            # 对于LONG：止损触发价应低于当前价格（价格下跌才触发）
            # 对于SHORT：止损触发价应高于当前价格上涨才触发）
            price_crossed = False
            if pos.side == PositionSide.LONG and current_price <= pos.stop_loss_price:
                price_crossed = True
            if pos.side == PositionSide.SHORT and current_price >= pos.stop_loss_price:
                price_crossed = True

            if price_crossed:
                logger.warning(f"⚠️ {symbol} 当前标记价{current_price:.6f}已穿越止损触发价{pos.stop_loss_price:.6f}，市价平仓")
                # 不挂止损单了，直接市价平仓（止损位已被穿越）
                self.close_position_on_exchange(symbol, "止损触发价已被穿越")
                self.close_position(symbol, "止损触发价已被穿越")
                return True

            # ===== 3.5 额外安全检查：确保止损方向正确（基于标记价） =====
            # 防止止损触发价设置错误导致API返回 -2021 Order would immediately trigger
            if pos.side == PositionSide.LONG and pos.stop_loss_price >= current_price:
                logger.error(f"❌ {symbol} LONG止损触发价{pos.stop_loss_price:.6f} >= 当前标记价{current_price:.6f}，跳过挂单")
                return False
            if pos.side == PositionSide.SHORT and pos.stop_loss_price <= current_price:
                logger.error(f"❌ {symbol} SHORT止损触发价{pos.stop_loss_price:.6f} <= 当前标记价{current_price:.6f}，跳过挂单")
                return False

            # ===== 4. 限价止损价格：比触发价偏移 =====
            cfg = get_main_config()
            delay_ratio = getattr(cfg, 'delay_ratio', 0.003)  # 默认0.3%
            if pos.side == PositionSide.LONG:
                limit_price = pos.stop_loss_price * (1 - delay_ratio)
            else:
                limit_price = pos.stop_loss_price * (1 + delay_ratio)
            limit_price = self._client.adjust_price_precision(symbol, limit_price) if self._client else limit_price
            pos.stop_limit_price = limit_price  # 保存限价供安全网判断价格是否穿越

            adjusted_qty = self._client.adjust_quantity_precision(symbol, pos.quantity)

            logger.info(f"{symbol} 挂止损单: 原始数量={pos.quantity}, 调整后数量={adjusted_qty}, 触发价={pos.stop_loss_price:.6f}, 限价={limit_price:.6f}, 触发价跌幅={((pos.entry_price-limit_price)/pos.entry_price*100):.2f}%")
            
            # ===== 4.5 API调用前再次检查价格（用标记价与 workingType=MARK_PRICE 一致） =====
            try:
                latest_price = self._client.get_mark_price(symbol) if self._client else None
                if latest_price:
                    if pos.side == PositionSide.LONG and latest_price <= pos.stop_loss_price:
                        logger.warning(f"⚠️ {symbol} API调用前再次检查：当前标记价{latest_price:.6f}已穿越止损触发价{pos.stop_loss_price:.6f}，市价平仓")
                        self.close_position_on_exchange(symbol, "止损触发价已被穿越")
                        self.close_position(symbol, "止损触发价已被穿越")
                        return True
                    if pos.side == PositionSide.SHORT and latest_price >= pos.stop_loss_price:
                        logger.warning(f"⚠️ {symbol} API调用前再次检查：当前标记价{latest_price:.6f}已穿越止损触发价{pos.stop_loss_price:.6f}，市价平仓")
                        self.close_position_on_exchange(symbol, "止损触发价已被穿越")
                        self.close_position(symbol, "止损触发价已被穿越")
                        return True
            except Exception as e:
                logger.debug(f"{symbol} 标记价查询失败，回退到成交价检查（继续挂单）: {e}")
            
            logger.info(f"【API下单参数】symbol={symbol}, side={stop_side}, type=STOP, quantity={adjusted_qty}, stop_price={pos.stop_loss_price}, price={limit_price}, position_side={position_side_str}")
            
            algo_result = self._client.create_algo_order(
                symbol=symbol,
                side=stop_side,
                order_type='STOP',
                quantity=adjusted_qty,
                stop_price=pos.stop_loss_price,
                price=limit_price,
                position_side=position_side_str,
                client_order_id=f"{pos.strategy}_SL_{int(time.time()*1000)}"
            )

            if algo_result and algo_result.get('algoId'):
                try:
                    algo_id = algo_result.get('algoId')
                    if algo_id is not None:
                        pos.stop_loss_algo_id = int(algo_id)
                        pos._stop_loss_last_placed = time.time()  # 记录挂单时间
                        # 成功后再验证：查询币安确认止损单真实存在
                        verified = False
                        try:
                            open_orders = self._client.get_algo_open_orders(symbol)
                            for order in open_orders:
                                if order.get('algoId') == algo_id:
                                    verified = True
                                    break
                        except Exception as e:
                            logger.debug(f"{symbol} 验证止损单失败（继续）: {e}")
                        if verified:
                            self._last_stop_fail_time.pop(symbol, None)
                            logger.info(f"✅ {symbol} 止损单已挂出并确认: algoId={algo_id}, 触发价={pos.stop_loss_price}")
                            return True
                        else:
                            logger.warning(f"⚠️ {symbol} 止损单API返回成功但币安上未找到（可能测试网限制），algoId={algo_id}")
                            # 已设置algoId和_last_placed，但不清空last_stop_fail_time，下一次验证会做最终确认
                            return True  # API已返回algoId，视为成功
                    else:
                        logger.warning(f"⚠️ {symbol} algoId为None")
                        self._last_stop_fail_time[symbol] = time.time()
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ {symbol} algoId转换失败: {algo_id}, {e}")
                    self._last_stop_fail_time[symbol] = time.time()
            else:
                logger.warning(f"⚠️ {symbol} 止损单创建未返回algoId: {algo_result}")
                self._last_stop_fail_time[symbol] = time.time()

        except Exception as e:
            self._last_stop_fail_time[symbol] = time.time()
            logger.error(f"❌ {symbol} 创建止损单失败: {str(e)}")
            raise  # 重新抛出异常，让调用者知道失败了
        return False

    def _close_position_by_market(self, symbol: str, pos: PositionInfo, reason: str):
        """限价转市价平仓（支持部分平仓，通过 pos._partial_close_ratio 控制）"""
        try:
            # 先验证交易所真实持仓，避免空操作
            try:
                exchange_positions = self._client.get_positions()
                exchange_pos = None
                for p in exchange_positions:
                    if p.get('symbol') == symbol:
                        exchange_pos = p
                        break
                if exchange_pos:
                    exchange_amt = float(exchange_pos.get('position_amt', 0))
                    if abs(exchange_amt) < 1e-8:
                        logger.info(f"{symbol} 交易所已无持仓，跳过平仓")
                        return True
            except Exception as e:
                logger.warning(f"{symbol} 查询交易所持仓失败（继续平仓）: {e}")

            close_side = 'SELL' if pos.side == PositionSide.LONG else 'BUY'
            
            # 检查是否部分平仓
            close_ratio = getattr(pos, '_partial_close_ratio', 0)
            if 0 < close_ratio < 1.0:
                close_qty = max(pos.quantity * close_ratio, 0)
            else:
                close_qty = pos.quantity
            
            adjusted_qty = self._client.adjust_quantity_precision(symbol, close_qty)

            if adjusted_qty <= 0:
                logger.error(f"❌ {symbol} 平仓数量无效: {adjusted_qty}")
                return False

            position_side = 'LONG' if pos.side == PositionSide.LONG else 'SHORT'
            # 使用策略计算的出场价（如有），否则用当前市价
            exit_price = getattr(pos, '_exit_price', 0)
            current_price = exit_price if exit_price > 0 else self._client.get_price(symbol)
            
            logger.info(f"正在平仓: {symbol} {close_side} {adjusted_qty} @ {current_price}")

            # 获取订单执行模式配置
            cfg = get_main_config()
            order_mode_str = getattr(cfg, 'order_mode', 'LIMIT_TO_MARKET')
            limit_timeout = getattr(cfg, 'order_limit_timeout', 5)
            
            # 转换OrderMode
            if order_mode_str == 'LIMIT_ONLY':
                order_mode = OrderMode.LIMIT_ONLY
            elif order_mode_str == 'MARKET_ONLY':
                order_mode = OrderMode.MARKET_ONLY
            else:
                order_mode = OrderMode.LIMIT_TO_MARKET
            
            params = OrderParams(
                symbol=symbol,
                side=OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=adjusted_qty,
                price=current_price,
                position_side=position_side,
                order_mode=order_mode,
                limit_timeout=limit_timeout,
                strategy='external',
                reduce_only=True,
            )

            result = self._order_engine.execute_order(params)

            if result.success:
                logger.info(f"✅ {symbol} 平仓成功: orderId={result.order_id}, 原因={reason}, 数量={adjusted_qty}")
                if 0 < close_ratio < 1.0:
                    # 部分平仓：减少持仓数量，不清除持仓
                    pos.quantity -= adjusted_qty
                    pos.margin *= (1 - close_ratio)
                    pos._partial_close_ratio = 0  # 重置
                    # 记录部分平仓历史
                    if not hasattr(pos, '_close_history'):
                        pos._close_history = []
                    pos._close_history.append({'time': time.time(), 'qty': adjusted_qty, 'reason': reason, 'price': current_price})
                    logger.info(f"📌 {symbol} 部分平仓: {close_ratio*100:.0f}%, 剩余={pos.quantity:.4f}")
                else:
                    self.close_position(symbol, reason)
                return True
            else:
                logger.error(f"❌ {symbol} 平仓失败: {result.message}")
                return False

        except Exception as e:
            logger.error(f"❌ {symbol} 平仓异常: {str(e)}")
            return False

    def _check_stop_loss_take_profit(self):
        """检查止盈止损（触发后实际执行平仓）"""
        if not self._positions:
            logger.debug("无持仓，跳过止盈止损检查")
            return

        logger.debug(f"检查止盈止损，持仓数: {len(self._positions)}")

        for symbol, pos in list(self._positions.items()):
            if pos.status != PositionStatus.ACTIVE:
                continue

            current_price = pos.current_price
            entry_price = pos.entry_price
            leverage = pos.leverage

            # 计算当前盈亏百分比（含杠杆）
            if entry_price > 0 and leverage > 0:
                if pos.side == PositionSide.LONG:
                    pnl_pct = (current_price - entry_price) / entry_price * leverage * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * leverage * 100
            else:
                pnl_pct = 0

            # 更新最高盈利
            if pnl_pct > pos.max_profit_pct:
                pos.max_profit_pct = pnl_pct

            # 更新MAE（最大浮亏）
            if pnl_pct < pos.mae_pct:
                pos.mae_pct = pnl_pct

            # 计算回撤
            drawdown = pos.max_profit_pct - pnl_pct if pos.max_profit_pct > 0 else 0

            # ===== 根据策略类型执行不同的止盈止损逻辑 =====
            if pos.strategy == 'external':
                self._check_external_position_stop_loss(symbol, pos, pnl_pct, drawdown)
            else:
                # 尝试调用策略的止盈止损逻辑
                close_reason = self._call_strategy_stop_loss(pos, current_price)
                if close_reason and close_reason != 'HOLD':
                    self._close_position_by_market(symbol, pos, close_reason)
                    continue

                # 硬止损安全网：交易所stop-limit限价单未成交时市价兜底
                # stop-limit触发后挂限价单，如果价格穿越限价说明限价单本应已成交但未填上（gap/插针）
                # 此时立即市价平仓，防止止损单挂在那里空等
                if pos.stop_limit_price and close_reason == 'HOLD':
                    if (pos.side == PositionSide.LONG and current_price <= pos.stop_limit_price) or \
                       (pos.side == PositionSide.SHORT and current_price >= pos.stop_limit_price):
                        logger.warning(f"【硬止损安全网】{symbol} {pos.side.value} 价格={current_price:.6f} 限价={pos.stop_limit_price:.6f} 止损触发价={pos.stop_loss_price:.6f} 盈亏={pnl_pct:.2f}%")
                        self._close_position_by_market(symbol, pos, "硬止损(价格穿越限价)")
                        continue

                # 如果策略没有实现止盈止损逻辑，使用通用逻辑
                # _call_strategy_stop_loss 返回 'HOLD' 表示策略主动持有，不应用通用兜底
                if close_reason != 'HOLD':
                    self._check_generic_stop_loss(symbol, pos, pnl_pct, drawdown)

            # 追踪止损上移交易所(仅HF类策略): 盈利后把止损单上移, 宕机也有尾随保护
            if pos.strategy != 'external':
                self._update_trailing_stop_on_exchange(symbol, pos)

    def _call_strategy_stop_loss(self, pos: PositionInfo, current_price: float) -> Optional[str]:
        """
        调用策略的止盈止损逻辑

        Args:
            pos: 持仓信息
            current_price: 当前价格

        Returns:
            平仓原因，如果不需要平仓则返回None
        """
        if not self._scheduler:
            return None

        try:
            # 获取策略实例
            strategy_info = self._scheduler._strategies.get(pos.strategy)

            # 'unknown'策略：遍历所有策略，找到监控该symbol的那个（仅查找，不修改归属）
            # 归属应在 _register_synced_position 阶段完成，此处只做只读查找
            if not strategy_info and pos.strategy == 'unknown':
                for name, info in self._scheduler._strategies.items():
                    if info.instance and hasattr(info.instance, '_symbols') and pos.symbol in info.instance._symbols:
                        strategy_info = info
                        break

            if not strategy_info or not strategy_info.instance:
                return None

            # 如果持仓没有止损价，尝试从策略获取静态止损价并挂 algo 单
            if not pos.stop_loss_price and hasattr(strategy_info.instance, 'get_static_stop_loss_price'):
                try:
                    static_sl = strategy_info.instance.get_static_stop_loss_price(pos)
                    if static_sl and static_sl > 0:
                        # 先查币安是否已有止损单(入口已挂的 ALGO STOP), 有则采用, 避免重复挂单
                        adopted = False
                        try:
                            open_orders = self._client.get_algo_open_orders(pos.symbol) if self._client else []
                            want_side = 'SELL' if pos.side == PositionSide.LONG else 'BUY'
                            want_ps = 'LONG' if pos.side == PositionSide.LONG else 'SHORT'
                            for o in open_orders:
                                if not o.get('algoId'):
                                    continue
                                if o.get('side') != want_side:
                                    continue
                                if o.get('positionSide') and o.get('positionSide') != want_ps:
                                    continue
                                sp_raw = o.get('stopPrice') or o.get('price')
                                try:
                                    sp = float(sp_raw) if sp_raw is not None else 0
                                except (TypeError, ValueError):
                                    sp = 0
                                if sp and abs(sp - static_sl) / static_sl < 0.02:
                                    pos.stop_loss_algo_id = int(o['algoId'])
                                    pos.stop_loss_price = static_sl
                                    pos.static_stop_loss_price = static_sl
                                    adopted = True
                                    logger.info(f"{pos.symbol} 采用入口已挂止损单 algoId={o['algoId']}, 触发价={static_sl}")
                                    break
                        except Exception as e:
                            logger.debug(f"{pos.symbol} 查询已有止损单失败(继续新建): {e}")
                        if not adopted:
                            pos.stop_loss_price = static_sl
                            pos.static_stop_loss_price = static_sl
                            logger.info(f"{pos.symbol} 策略提供静态止损价: {pos.stop_loss_price}")
                            self._place_stop_loss_order(pos.symbol, pos)
                except Exception as e:
                    logger.debug(f"{pos.symbol} 获取静态止损价失败: {e}")

            # floor 兜底：stop_loss_price 已设但 static_stop_loss_price 未设（如 execute_signal 注册时已带止损价）
            # 防止交易所追踪止损(_update_trailing_stop_on_exchange)因 floor 缺失而被禁用
            if pos.stop_loss_price and not pos.static_stop_loss_price and hasattr(strategy_info.instance, 'get_static_stop_loss_price'):
                try:
                    static_sl = strategy_info.instance.get_static_stop_loss_price(pos)
                    if static_sl and static_sl > 0:
                        pos.static_stop_loss_price = static_sl
                except Exception:
                    pass

            # 调用策略的止盈止损检查
            close_reason = strategy_info.instance.check_stop_loss_take_profit(pos, current_price)
            # 策略返回None表示主动持有（不触发退出），返回'HOLD'区分于'未找到策略'
            return close_reason if close_reason else 'HOLD'
        except Exception as e:
            logger.exception(f"调用策略止盈止损逻辑失败: {pos.strategy}, {e}")
            return None

    def _check_external_position_stop_loss(self, symbol: str, pos: PositionInfo, pnl_pct: float, drawdown: float):
        """外部持仓止盈止损逻辑"""
        # 获取外部持仓参数
        cfg = get_main_config()
        breakeven_threshold = getattr(cfg, 'external_breakeven_threshold', 15.0)

        # 保本止盈：最大盈利>15%，当前盈利回落到±2%以内
        if pos.max_profit_pct > breakeven_threshold and abs(pnl_pct) <= 2.0:
            logger.warning(f"【触发保本止盈】{symbol} {pos.side.value} 最高盈利={pos.max_profit_pct:.2f}% 当前盈利={pnl_pct:.2f}%")
            self._close_position_by_market(symbol, pos, "保本止盈触发")
            return

        # 回撤止盈：最大盈利>阈值且回撤超限即触发（不要求当前仍盈利，避免死区）
        if pos.max_profit_pct > 0:
            logger.info(f"【止盈监控】{symbol} 最高盈利={pos.max_profit_pct:.2f}% 当前盈利={pnl_pct:.2f}% 回撤={drawdown:.2f}% 阈值={pos.drawdown_threshold}%")

            if drawdown >= pos.drawdown_threshold:
                logger.warning(f"【触发止盈-回撤】{symbol} {pos.side.value} 最高盈利={pos.max_profit_pct:.2f}% 当前盈利={pnl_pct:.2f}% 回撤={drawdown:.2f}% 阈值={pos.drawdown_threshold}%")
                self._close_position_by_market(symbol, pos, "回撤止盈触发")
                return

        # 止损检查
        self._check_hard_stop_loss(symbol, pos, pnl_pct)

    def _check_generic_stop_loss(self, symbol: str, pos: PositionInfo, pnl_pct: float, drawdown: float):
        """通用止盈止损逻辑（用于其他策略）"""
        # 回撤止盈：最大盈利>阈值且回撤超限即触发（不要求当前仍盈利，避免死区）
        if pos.max_profit_pct > 0 and drawdown >= pos.drawdown_threshold:
            logger.warning(f"【触发回撤止盈】{symbol} {pos.side.value} 最高盈利={pos.max_profit_pct:.2f}% 当前盈利={pnl_pct:.2f}% 回撤={drawdown:.2f}%")
            self._close_position_by_market(symbol, pos, "回撤止盈触发")
            return

        # 止损检查
        self._check_hard_stop_loss(symbol, pos, pnl_pct)

    def _check_hard_stop_loss(self, symbol: str, pos: PositionInfo, pnl_pct: float):
        """硬止损检查（所有策略通用）"""
        current_price = pos.current_price

        # 止损检查
        if pos.stop_loss_price:
            if pos.side == PositionSide.LONG:
                logger.info(f"【止损监控】{symbol} LONG 当前价={current_price:.6f} 止损价={pos.stop_loss_price:.6f} 盈亏={pnl_pct:.2f}%")
                if current_price <= pos.stop_loss_price:
                    logger.warning(f"【触发止损】{symbol} LONG 价格={current_price:.6f} 止损价={pos.stop_loss_price:.6f} 盈亏={pnl_pct:.2f}%")
                    self._close_position_by_market(symbol, pos, "止损触发")
                    return
            elif pos.side == PositionSide.SHORT:
                logger.info(f"【止损监控】{symbol} SHORT 当前价={current_price:.6f} 止损价={pos.stop_loss_price:.6f} 盈亏={pnl_pct:.2f}%")
                if current_price >= pos.stop_loss_price:
                    logger.warning(f"【触发止损】{symbol} SHORT 价格={current_price:.6f} 止损价={pos.stop_loss_price:.6f} 盈亏={pnl_pct:.2f}%")
                    self._close_position_by_market(symbol, pos, "止损触发")
                    return

        # 普通止盈检查
        if pos.take_profit_price:
            if pos.side == PositionSide.LONG and current_price >= pos.take_profit_price:
                logger.warning(f"【触发止盈】{symbol} LONG 价格={current_price:.6f} 止盈价={pos.take_profit_price:.6f} 盈亏={pnl_pct:.2f}%")
                self._close_position_by_market(symbol, pos, "止盈触发")
                return
            elif pos.side == PositionSide.SHORT and current_price <= pos.take_profit_price:
                logger.warning(f"【触发止盈】{symbol} SHORT 价格={current_price:.6f} 止盈价={pos.take_profit_price:.6f} 盈亏={pnl_pct:.2f}%")
                self._close_position_by_market(symbol, pos, "止盈触发")
                return
                    
    def _update_trailing_stop_on_exchange(self, symbol: str, pos: PositionInfo):
        """将追踪止损同步到交易所: 盈利后把止损单上移, 宕机也有尾随保护。

        复用 _place_stop_loss_order (create_algo_order, order_type='STOP', price=limit_price),
        即限价转市价 STOP_LIMIT。带价格步长+时间间隔双重节流, 避免每tick刷API。
        仅上移(棘轮), 不回撤; 未激活追踪前保持 L2 地板不动。
        """
        if pos.status != PositionStatus.ACTIVE or pos.quantity <= 0 or not self._client:
            return
        floor = getattr(pos, 'static_stop_loss_price', None)
        if not floor:
            return  # 还没拿到L2地板价(首tick由_call_strategy_stop_loss设置)

        cfg = get_main_config()
        activate = getattr(cfg, 'hf_trail_activate', 0.2)
        step = getattr(cfg, 'hf_trail_step', 0.05)
        min_step_pct = getattr(cfg, 'hf_trail_exchange_min_step_pct', 0.05)
        min_interval = getattr(cfg, 'hf_trail_exchange_min_interval', 15)

        leverage = pos.leverage or 1
        if leverage <= 0:
            leverage = 1
        entry = pos.entry_price
        if not entry or entry <= 0:
            return
        pnl_mult = leverage * 100
        peak = pos.max_profit_pct

        if peak < activate:
            return  # 未激活追踪: 保持L2地板, 不改单

        trail_pnl = max(peak - step, 0.0)
        if pos.side == PositionSide.LONG:
            trail_price = entry * (1 + trail_pnl / pnl_mult)
        else:
            trail_price = entry * (1 - trail_pnl / pnl_mult)

        if pos.side == PositionSide.LONG:
            new_stop = max(floor, trail_price)
            old_stop = pos.stop_loss_price if pos.stop_loss_price else floor
            if new_stop <= old_stop + 1e-12:
                return  # LONG只上移
        else:
            new_stop = min(floor, trail_price)
            old_stop = pos.stop_loss_price if pos.stop_loss_price else floor
            if new_stop >= old_stop - 1e-12:
                return  # SHORT只下移

        # 节流1: 价格步长（用绝对值，兼容LONG上移/SHORT下移）
        move_pct = abs((new_stop - old_stop) / old_stop * 100) if old_stop else 0
        if move_pct < min_step_pct:
            return

        # 节流2: 时间间隔
        now = time.time()
        last = getattr(pos, 'trailing_last_update', 0.0)
        if now - last < min_interval:
            return

        pos.stop_loss_price = new_stop
        try:
            self._place_stop_loss_order(symbol, pos)
            pos.trailing_last_update = now
            logger.info(f"{symbol} 追踪止损上移交易所: {old_stop:.6f} -> {new_stop:.6f} (峰值{peak:.2f}%, 上移{move_pct:.3f}%)")
        except Exception as e:
            logger.error(f"{symbol} 追踪止损改交易所单失败: {e}")
            pos.stop_loss_price = old_stop  # 回滚, 下次重试

    def open_position(self, position: PositionInfo) -> bool:
        """
        记录开仓

        Args:
            position: 持仓信息

        Returns:
            是否成功
        """
        with self._position_lock:
            # 检查是否已存在
            if position.symbol in self._positions:
                existing = self._positions[position.symbol]
                # 如果是同步注册的'unknown'策略，升级为实际策略
                if existing.strategy == 'unknown' and position.strategy != 'unknown':
                    old_strategy = existing.strategy
                    existing.strategy = position.strategy
                    # 从 unknown 索引移除
                    if 'unknown' in self._strategy_positions and position.symbol in self._strategy_positions['unknown']:
                        del self._strategy_positions['unknown'][position.symbol]
                    # 添加到实际策略索引
                    if position.strategy not in self._strategy_positions:
                        self._strategy_positions[position.strategy] = {}
                    self._strategy_positions[position.strategy][position.symbol] = existing
                    # 更新其他字段（entry_price, quantity 等可能因加仓变化）
                    existing.entry_price = position.entry_price
                    existing.quantity = position.quantity
                    existing.leverage = position.leverage
                    existing.margin = position.margin
                    existing.current_price = position.current_price
                    existing.side = position.side
                    logger.info(f"{position.symbol} 从unknown升级到策略{position.strategy}")
                    return True
                logger.warning(f"持仓已存在: {position.symbol}")
                return False

            self._positions[position.symbol] = position

            # 添加到策略索引
            if position.strategy not in self._strategy_positions:
                self._strategy_positions[position.strategy] = {}
            self._strategy_positions[position.strategy][position.symbol] = position

        # 调用策略 on_position_opened 初始化 ExitState、signal_type 等策略属性
        if self._scheduler and position.strategy != 'external':
            strategy_info = self._scheduler._strategies.get(position.strategy)
            if strategy_info and strategy_info.instance:
                try:
                    strategy_info.instance.on_position_opened(position)
                except Exception as e:
                    logger.warning(f"{position.symbol} on_position_opened调用失败: {e}")

        # 挂止损单保护新开仓（锁外执行，避免持锁网络I/O）
        self._place_stop_loss_order(position.symbol, position)

        # 记录开仓交易
        side_str = 'LONG' if position.side == PositionSide.LONG else 'SHORT'
        total_margin = sum(p.margin for p in self._positions.values())
        capital_ratio = position.margin / total_margin if total_margin > 0 else 1.0
        self._trade_recorder.record_open(
            symbol=position.symbol,
            strategy=position.strategy,
            direction=side_str,
            leverage=position.leverage,
            quantity=position.quantity,
            margin=position.margin,
            capital_ratio=capital_ratio,
            entry_price=position.entry_price,
        )

        # 发送事件
        self._event_bus.publish(PositionEvent(
            event_type=EventType.POSITION_OPENED,
            symbol=position.symbol,
            strategy=position.strategy,
            side=position.side.value,
            quantity=position.quantity,
            data={'position': position},
        ))

        logger.info(
            f"持仓已开: {position.symbol} {position.side.value} {position.quantity}",
            extra={'context': {
                'strategy': position.strategy,
                'symbol': position.symbol,
                'entry_price': position.entry_price,
                'leverage': position.leverage,
            }}
        )

        return True
        
    def close_position_on_exchange(self, symbol: str, reason: str = '') -> bool:
        """
        在交易所上真正平仓（用于外部持仓超限等情况）
        
        Args:
            symbol: 币种
            reason: 平仓原因
            
        Returns:
            是否成功
        """
        if not self._client:
            logger.warning(f"close_position_on_exchange: client未初始化，无法平仓 {symbol}")
            return False
        
        try:
            # 获取当前持仓
            positions = self._client.get_positions()
            target_pos = None
            for pos in positions:
                if pos.get('symbol') == symbol:
                    target_pos = pos
                    break
            
            if not target_pos:
                logger.warning(f"close_position_on_exchange: 未找到持仓 {symbol}")
                return False
            
            position_amt = float(target_pos.get('position_amt', 0))
            if abs(position_amt) < 1e-8:
                logger.info(f"close_position_on_exchange: {symbol} 持仓数量为0，无需平仓")
                return True
            
            # 确定平仓方向
            side = 'SELL' if position_amt > 0 else 'BUY'
            
            # 使用市价单平仓
            logger.info(f"close_position_on_exchange: {symbol} 市价平仓，数量={abs(position_amt)}，方向={side}")
            result = self._client.create_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=abs(position_amt),
                reduce_only=True
            )
            
            if result and result.get('orderId'):
                logger.info(f"close_position_on_exchange: {symbol} 平仓成功，orderId={result.get('orderId')}")
                return True
            else:
                logger.warning(f"close_position_on_exchange: {symbol} 平仓失败，result={result}")
                return False
                
        except Exception as e:
            logger.error(f"close_position_on_exchange: {symbol} 平仓异常: {e}")
            return False

    def close_position(self, symbol: str, reason: str = '') -> Optional[PositionInfo]:
        """
        记录平仓（同时取消相关挂单）
        
        Args:
            symbol: 币种
            reason: 平仓原因
            
        Returns:
            平仓的持仓信息
        """
        with self._position_lock:
            if symbol not in self._positions:
                return None
                
            position = self._positions[symbol]
            
            # ===== 先取消止损单，再移除持仓（防止止损单残留成为幽灵单） =====
            if position.stop_loss_algo_id and self._client:
                try:
                    self._client.cancel_algo_order(symbol, algo_id=position.stop_loss_algo_id)
                    logger.info(f"已取消平仓持仓的止损单: {symbol} algoId={position.stop_loss_algo_id}")
                except Exception as e:
                    logger.warning(f"取消止损单失败（可能已成交或残留）: {symbol} algoId={position.stop_loss_algo_id} err={e}")
            
            self._positions.pop(symbol)
            position.status = PositionStatus.CLOSED
            position.closed_at = time.time()
            position.close_reason = reason
            
            # 从策略索引移除
            if position.strategy in self._strategy_positions:
                self._strategy_positions[position.strategy].pop(symbol, None)
                
        # 发送事件（高优先级，防止队列满时被丢弃导致币种锁死锁）
        self._event_bus.publish(PositionEvent(
            event_type=EventType.POSITION_CLOSED,
            symbol=symbol,
            strategy=position.strategy,
            side=position.side.value,
            quantity=position.quantity,
            priority=10,
            data={'position': position},
        ))
        
        logger.info(
            f"持仓已平: {symbol}, 原因: {reason}",
            extra={'context': {
                'strategy': position.strategy,
                'symbol': symbol,
                'pnl_pct': position.pnl_pct,
                'hold_hours': position.hold_hours,
            }}
        )

        # 记录平仓交易（更新开仓记录）
        close_price = position.current_price if position.current_price > 0 else position.entry_price
        self._trade_recorder.record_close(symbol, close_price, reason)

        return position
        
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """获取持仓"""
        return self._positions.get(symbol)
        
    def get_all_positions(self) -> List[PositionInfo]:
        """获取所有持仓"""
        return list(self._positions.values())
        
    def get_strategy_positions(self, strategy: str) -> List[PositionInfo]:
        """获取策略持仓"""
        return list(self._strategy_positions.get(strategy, {}).values())
        
    def get_position_count(self, strategy: str = None) -> int:
        """获取持仓数量"""
        if strategy:
            return len(self._strategy_positions.get(strategy, {}))
        return len(self._positions)
        
    def get_total_pnl(self, strategy: str = None) -> float:
        """获取总盈亏"""
        if strategy:
            positions = self._strategy_positions.get(strategy, {}).values()
        else:
            positions = self._positions.values()
        return sum(p.unrealized_pnl for p in positions)
        
    def add_price_callback(self, callback: Callable[[str, float, PositionInfo], None]):
        """添加价格回调"""
        self._price_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._position_lock:
            total_pnl = self.get_total_pnl()
            positions = list(self._positions.values())
            
        return {
            'total_positions': len(self._positions),
            'total_pnl': total_pnl,
            'strategies': {
                name: {
                    'positions': len(pos),
                    'pnl': sum(p.unrealized_pnl for p in pos.values()),
                }
                for name, pos in self._strategy_positions.items()
            },
            'positions': [p.to_dict() for p in positions],
        }


# 便捷函数
_position_monitor: Optional[PositionMonitor] = None


def get_position_monitor(client=None) -> PositionMonitor:
    """获取持仓监控单例"""
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitor(client)
    elif client:
        _position_monitor.set_client(client)
    return _position_monitor
