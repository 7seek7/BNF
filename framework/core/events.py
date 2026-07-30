# -*- coding: utf-8 -*-
"""
事件总线 - 模块间解耦通信

特性：
1. 发布/订阅模式
2. 类型安全的事件
3. 异步事件支持
4. 事件优先级
5. 事件历史记录
"""

import threading
import queue
import time
from collections import deque
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from framework.core.logger import get_logger

logger = get_logger('event_bus')


class EventType(Enum):
    """事件类型"""
    # K线事件
    KLINE_UPDATE = auto()
    KLINE_HISTORY_LOADED = auto()
    
    # 订单事件
    ORDER_CREATED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()
    ORDER_FAILED = auto()
    
    # 持仓事件
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()
    POSITION_LIQUIDATED = auto()
    
    # 信号事件
    SIGNAL_GENERATED = auto()
    SIGNAL_ACCEPTED = auto()
    SIGNAL_REJECTED = auto()
    
    # 风控事件
    RISK_WARNING = auto()
    RISK_LIMIT_REACHED = auto()
    EMERGENCY_STOP = auto()
    CIRCUIT_BREAKER_TRIGGERED = auto()
    
    # 策略事件
    STRATEGY_STARTED = auto()
    STRATEGY_STOPPED = auto()
    STRATEGY_ERROR = auto()
    
    # 系统事件
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    SYSTEM_ERROR = auto()
    CONFIG_RELOADED = auto()
    
    # 账户事件
    ACCOUNT_BALANCE_UPDATE = auto()
    ACCOUNT_MARGIN_UPDATE = auto()
    

@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ''  # 事件来源（模块名）
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 优先级（数字越大越优先，PriorityQueue为最小堆，故反转比较）
    handled: bool = False  # 是否已处理

    def __lt__(self, other):
        """
        用于优先级队列排序

        注意：Python的PriorityQueue是最小堆（小的先出），
        但我们的需求是priority数字越大越优先，
        所以使用>来反转比较逻辑。

        例如：
        - priority=10的事件应该比priority=5的事件先处理
        - 但PriorityQueue会先返回priority=5的事件
        - 所以用>来让priority=10的事件"小于"priority=5的事件
        """
        return self.priority > other.priority
        
    def __str__(self):
        return f"Event({self.event_type.name}, source={self.source}, data={self.data})"


@dataclass
class KlineEvent(Event):
    """K线事件"""
    symbol: str = ''
    interval: str = ''
    
    def __post_init__(self):
        self.event_type = EventType.KLINE_UPDATE
        

@dataclass
class OrderEvent(Event):
    """订单事件"""
    order_id: str = ''
    symbol: str = ''
    side: str = ''
    status: str = ''
    strategy: str = ''
    
    def __post_init__(self):
        if self.event_type not in [
            EventType.ORDER_CREATED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
            EventType.ORDER_FAILED,
        ]:
            self.event_type = EventType.ORDER_CREATED


@dataclass
class PositionEvent(Event):
    """持仓事件"""
    symbol: str = ''
    strategy: str = ''
    side: str = ''
    quantity: float = 0.0
    
    def __post_init__(self):
        if self.event_type not in [
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
            EventType.POSITION_UPDATED,
            EventType.POSITION_LIQUIDATED,
        ]:
            self.event_type = EventType.POSITION_OPENED


@dataclass
class SignalEvent(Event):
    """信号事件"""
    strategy: str = ''
    symbol: str = ''
    signal_type: str = ''
    direction: str = ''
    confidence: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL_GENERATED


@dataclass
class RiskEvent(Event):
    """风控事件"""
    risk_type: str = ''
    risk_level: str = 'WARNING'  # WARNING / CRITICAL
    message: str = ''
    
    def __post_init__(self):
        self.event_type = EventType.RISK_WARNING


# 事件处理器类型
EventHandler = Callable[[Event], None]


class EventBus:
    """
    事件总线
    
    模块间解耦通信的核心组件
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        with self._lock:
            if getattr(self, '_initialized', False):
                return
            
        # 订阅者: {EventType: [handler]}
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._subscriber_lock = threading.RLock()
        
        # 事件队列
        self._event_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._max_queue_size = 10000
        
        # 事件历史（用于调试）- 使用 deque 避免 list.pop(0) 的 O(n) 开销
        self._max_history = 1000
        self._event_history: deque = deque(maxlen=self._max_history)
        self._history_lock = threading.Lock()
        
        # 工作线程
        self._worker_thread = None
        self._running = False
        
        self._initialized = True
        logger.info("事件总线已初始化")
        
    def subscribe(self, event_type: EventType, handler: EventHandler):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        with self._subscriber_lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)
            logger.debug(f"订阅事件: {event_type.name} -> {handler.__name__}")
            
    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        """取消订阅"""
        with self._subscriber_lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                    logger.debug(f"取消订阅: {event_type.name} -> {handler.__name__}")
                except ValueError:
                    logger.debug(f"订阅者不在列表中: {handler.__name__}")
                    
    def publish(self, event: Event):
        """
        发布事件（异步）

        Args:
            event: 事件对象
        """
        # 发布事件（异步）
        try:
            self._event_queue.put_nowait(event)
            logger.debug(f"发布事件: {event.event_type.name}")
        except queue.Full:
            # 队列已满，根据优先级决定是否丢弃
            if event.priority >= 8:  # 高优先级事件
                logger.warning(f"事件队列已满({self._max_queue_size})，高优先级事件阻塞等待: {event}")
                try:
                    self._event_queue.put(event, timeout=5)
                    logger.debug(f"发布事件: {event.event_type.name}")
                except queue.Full:
                    logger.error(f"事件队列已满，高优先级事件丢弃: {event}")
            else:
                logger.warning(f"事件队列已满({self._max_queue_size})，丢弃低优先级事件: {event}")
        
    def publish_sync(self, event: Event):
        """
        同步发布事件（立即处理）
        
        Args:
            event: 事件对象
        """
        self._handle_event(event)
        
    def _handle_event(self, event: Event):
        """处理事件"""
        # 记录历史 - deque 自动限制大小，无需手动 pop(0)
        with self._history_lock:
            self._event_history.append(event)
                
        # 调用订阅者 - 在锁内迭代以避免竞态条件
        with self._subscriber_lock:
            handlers = list(self._subscribers.get(event.event_type, []))
            
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"事件处理失败: {handler.__name__}, 事件: {event}, 错误: {e}")
                
        event.handled = True
        
    def _worker(self):
        """事件处理工作线程"""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1)
                self._handle_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"事件工作线程异常: {e}")
                
    def start(self):
        """启动事件总线"""
        if self._running:
            return
            
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("事件总线已启动")
        
    def stop(self):
        """停止事件总线"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("事件总线已停止")
        
    def get_event_history(self, event_type: EventType = None, limit: int = 100) -> List[Event]:
        """
        获取事件历史
        
        Args:
            event_type: 过滤事件类型（可选）
            limit: 返回数量限制
            
        Returns:
            事件列表
        """
        with self._history_lock:
            events = list(self._event_history)
            
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        return events[-limit:]
        
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self._event_queue.qsize()
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._history_lock:
            history = list(self._event_history)
            
        stats = {
            'queue_size': self._event_queue.qsize(),
            'history_count': len(history),
            'subscriber_count': sum(len(h) for h in list(self._subscribers.values())),
            'event_types': {
                et.name: len([e for e in history if e.event_type == et])
                for et in EventType
            }
        }
        return stats


# 便捷函数
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取事件总线单例"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def subscribe(event_type: EventType, handler: EventHandler):
    """订阅事件（便捷函数）"""
    get_event_bus().subscribe(event_type, handler)
    

def publish(event: Event):
    """发布事件（便捷函数）"""
    get_event_bus().publish(event)


def publish_sync(event: Event):
    """同步发布事件（便捷函数）"""
    get_event_bus().publish_sync(event)
