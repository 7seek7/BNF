# -*- coding: utf-8 -*-
"""
重复订单检测器

功能：
- 检测短时间内重复的订单请求
- 防止因网络延迟导致的重复下单
- 订单冲突检测
"""

import time
import threading
import hashlib
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderFingerprint:
    """订单指纹"""
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    timestamp: float
    
    def to_hash(self) -> str:
        """生成哈希值（大小写不敏感）"""
        key = f"{self.symbol}_{self.side.upper()}_{self.order_type.upper()}_{self.quantity}_{self.price or 0}"
        return hashlib.md5(key.encode()).hexdigest()


class DuplicateDetector:
    """重复订单检测器"""
    
    _instance = None
    _singleton_lock = threading.Lock()
    
    def __new__(cls, cooldown_seconds: int = 60):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._init_params = cooldown_seconds  # 记录初始参数
        return cls._instance
    
    def __init__(self, cooldown_seconds: int = 60):
        if self._initialized:
            # 修复：允许更新cooldown_seconds，即使单例已初始化
            if hasattr(self, '_init_params') and cooldown_seconds != self._init_params:
                self._cooldown_seconds = cooldown_seconds
                self._init_params = cooldown_seconds
            return
        
        self._cooldown_seconds = cooldown_seconds
        self._fingerprints: Dict[str, float] = {}  # hash -> timestamp
        self._lock = threading.RLock()
        self._initialized = True
    
    def check_duplicate(self, symbol: str, side: str, order_type: str,
                       quantity: float, price: float = None) -> Tuple[bool, str]:
        """
        检查是否为重复订单
        
        Returns:
            (is_duplicate, reason)
        """
        fingerprint = OrderFingerprint(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=time.time()
        )
        
        hash_key = fingerprint.to_hash()
        
        with self._lock:
            now = time.time()
            
            # 清理过期的指纹
            expired = [k for k, v in self._fingerprints.items() 
                      if now - v > self._cooldown_seconds]
            for k in expired:
                del self._fingerprints[k]
            
            # 检查是否存在
            if hash_key in self._fingerprints:
                elapsed = now - self._fingerprints[hash_key]
                return True, f"重复订单（{elapsed:.1f}秒前已提交）"
            
            # 记录新指纹
            self._fingerprints[hash_key] = now
            return False, ""
    
    def record_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: float = None):
        """记录订单（手动记录）"""
        fingerprint = OrderFingerprint(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=time.time()
        )
        
        hash_key = fingerprint.to_hash()
        with self._lock:
            self._fingerprints[hash_key] = time.time()
    
    def clear_fingerprint(self, symbol: str, side: str, order_type: str,
                         quantity: float, price: float = None):
        """清除指纹（订单成功后可清除）"""
        fingerprint = OrderFingerprint(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=time.time()
        )
        
        hash_key = fingerprint.to_hash()
        with self._lock:
            if hash_key in self._fingerprints:
                del self._fingerprints[hash_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            now = time.time()
            active = len(self._fingerprints)
            return {
                'active_fingerprints': active,
                'cooldown_seconds': self._cooldown_seconds
            }


class ConflictDetector:
    """订单冲突检测器"""
    
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
        
        # 记录每个币种的待处理订单
        self._pending_orders: Dict[str, Dict] = {}  # symbol -> order_info
        self._lock = threading.RLock()
        self._initialized = True
    
    def check_conflict(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        检查是否存在订单冲突
        
        冲突条件：
        1. 同一币种已有反向待处理订单
        
        Returns:
            (has_conflict, reason)
        """
        with self._lock:
            if symbol not in self._pending_orders:
                return False, ""
            
            pending = self._pending_orders[symbol]
            pending_side = pending.get('side', '')
            
            # 反向订单检测
            if pending_side and pending_side != side:
                return True, f"存在反向待处理订单（{pending_side}）"
            
            return False, ""
    
    def register_pending(self, symbol: str, order_info: Dict):
        """注册待处理订单"""
        with self._lock:
            self._pending_orders[symbol] = {
                'side': order_info.get('side'),
                'order_id': order_info.get('order_id'),
                'timestamp': time.time()
            }
    
    def unregister_pending(self, symbol: str):
        """取消注册待处理订单"""
        with self._lock:
            if symbol in self._pending_orders:
                del self._pending_orders[symbol]
    
    def get_pending(self, symbol: str) -> Optional[Dict]:
        """获取待处理订单"""
        with self._lock:
            return self._pending_orders.get(symbol)
    
    def cleanup_expired(self, max_age_seconds: int = 300):
        """清理过期的待处理订单"""
        now = time.time()
        with self._lock:
            expired = [s for s, info in self._pending_orders.items()
                      if now - info.get('timestamp', 0) > max_age_seconds]
            for s in expired:
                del self._pending_orders[s]


# 单例获取函数
def get_duplicate_detector(cooldown_seconds: int = 60) -> DuplicateDetector:
    """获取重复订单检测器单例"""
    return DuplicateDetector(cooldown_seconds)


def get_conflict_detector() -> ConflictDetector:
    """获取订单冲突检测器单例"""
    return ConflictDetector()
