# -*- coding: utf-8 -*-
"""
订单持久化管理器（公共模块）

功能：
- 订单状态持久化存储
- 订单历史查询
- 订单状态追踪
"""

import os
import json
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pathlib import Path
from utils.logger import Logger

logger = Logger.get_logger('order_persistence')


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class OrderRecord:
    """订单记录"""
    def __init__(self, order_id: str, symbol: str, side: str, 
                 order_type: str, quantity: float, price: float = None,
                 strategy: str = None, status: OrderStatus = OrderStatus.PENDING):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.strategy = strategy
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.filled_quantity = 0.0
        self.filled_price = 0.0
        self.error_message = ""
        
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'strategy': self.strategy,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'error_message': self.error_message
        }


class OrderPersistence:
    """订单持久化管理器（公共模块）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._orders: Dict[str, OrderRecord] = {}
        self._orders_lock = threading.RLock()
        self._data_dir = Path("data/orders")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
    
    def save_order(self, order: OrderRecord) -> bool:
        """保存订单"""
        with self._orders_lock:
            self._orders[order.order_id] = order
            self._persist_to_file(order)
            return True
    
    def update_order_status(self, order_id: str, status: OrderStatus, 
                           filled_quantity: float = None, 
                           filled_price: float = None,
                           error_message: str = None) -> bool:
        """更新订单状态"""
        with self._orders_lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            order.status = status
            order.updated_at = datetime.now()
            
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
            if filled_price is not None:
                order.filled_price = filled_price
            if error_message is not None:
                order.error_message = error_message
            
            self._persist_to_file(order)
            return True
    
    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        """获取订单"""
        with self._orders_lock:
            return self._orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str, status: OrderStatus = None) -> List[OrderRecord]:
        """按币种获取订单"""
        with self._orders_lock:
            orders = [o for o in self._orders.values() if o.symbol == symbol]
            if status:
                orders = [o for o in orders if o.status == status]
            return orders
    
    def get_orders_by_strategy(self, strategy: str) -> List[OrderRecord]:
        """按策略获取订单"""
        with self._orders_lock:
            return [o for o in self._orders.values() if o.strategy == strategy]
    
    def get_pending_orders(self) -> List[OrderRecord]:
        """获取待处理订单"""
        with self._orders_lock:
            return [o for o in self._orders.values() 
                   if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)]
    
    def cleanup_old_orders(self, days: int = 7) -> int:
        """清理旧订单"""
        cutoff = datetime.now()
        count = 0
        with self._orders_lock:
            to_remove = []
            for order_id, order in self._orders.items():
                if (cutoff - order.created_at).days > days:
                    to_remove.append(order_id)
            
            for order_id in to_remove:
                del self._orders[order_id]
                count += 1
        
        return count
    
    def _persist_to_file(self, order: OrderRecord):
        """持久化到文件（原子写入，防止进程崩溃导致文件损坏）"""
        try:
            import tempfile
            import os
            
            filename = f"{order.symbol}_{order.order_id}.json".replace('/', '_').replace('\\', '_').replace('..', '_')
            filepath = self._data_dir / filename
            
            # 使用临时文件+原子重命名，确保写入安全
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                            dir=self._data_dir, delete=False) as tmp:
                json.dump(order.to_dict(), tmp, ensure_ascii=False, indent=2)
                tmp_path = tmp.name
            
            # 原子重命名
            os.replace(tmp_path, filepath)
        except Exception as e:
            logger.error(f"订单持久化失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._orders_lock:
            total = len(self._orders)
            by_status = {}
            for status in OrderStatus:
                by_status[status.value] = len([o for o in self._orders.values() if o.status == status])
            
            return {
                'total_orders': total,
                'by_status': by_status
            }


# 单例获取函数
def get_order_persistence() -> OrderPersistence:
    """获取订单持久化管理器单例"""
    return OrderPersistence()