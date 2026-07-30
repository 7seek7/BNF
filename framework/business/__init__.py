# -*- coding: utf-8 -*-
"""业务模块"""

from framework.business.kline_manager import KlineManager, KlineCache
from framework.business.order_engine import OrderEngine, OrderResult
from framework.business.risk_center import RiskCenter, RiskLevel
from framework.business.position_monitor import PositionMonitor, PositionInfo
from framework.business.position_sync import PositionSyncManager, get_position_sync_manager
from framework.business.fund_allocator import FundAllocator, get_fund_allocator
from framework.business.order_persistence import OrderPersistence, get_order_persistence, OrderStatus, OrderRecord
from framework.business.position_manager import PositionManager
from framework.business.risk_manager import RiskManager, RiskState, PositionSizer
from framework.business.circuit_breaker import CircuitBreaker, CircuitBreakerStatus, CircuitBreakerType
from framework.business.duplicate_detector import DuplicateDetector, ConflictDetector, OrderFingerprint
from framework.business.trade_recorder import TradeRecorder, TradeRecord

__all__ = [
    # 基础
    'KlineManager',
    'KlineCache',
    'OrderEngine',
    'OrderResult',
    # 风控
    'RiskCenter',
    'RiskLevel',
    'RiskManager',
    'RiskState',
    'PositionSizer',
    'CircuitBreaker',
    'CircuitBreakerStatus',
    'CircuitBreakerType',
    # 持仓
    'PositionMonitor',
    'PositionInfo',
    'PositionManager',
    'PositionSyncManager',
    'get_position_sync_manager',
    # 资金
    'FundAllocator',
    'get_fund_allocator',
    # 订单
    'OrderPersistence',
    'get_order_persistence',
    'OrderStatus',
    'OrderRecord',
    # 检测
    'DuplicateDetector',
    'ConflictDetector',
    'OrderFingerprint',
    # 记录
    'TradeRecorder',
    'TradeRecord',
]
