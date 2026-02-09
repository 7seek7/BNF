"""
重复订单检测器

功能：
- 防止在短时间内重复提交相同订单
- 支持订单冷却机制
- 多线程安全
"""

import time
import threading
from typing import Dict, Optional, Tuple
from collections import defaultdict
from utils.logger import Logger

logger = Logger.get_logger('duplicate_detector')


class DuplicateDetector:
    """重复订单检测器 - 防止短时间内重复下单"""
    
    def __init__(self, cooldown_seconds: int = 60):
        """
        初始化重复订单检测器
        
        Args:
            cooldown_seconds: 订单冷却时间（秒），默认60秒
        """
        self.cooldown_seconds = cooldown_seconds
        
        # {symbol: (last_order_time, order_type)}
        self.pending_orders: Dict[str, Tuple[float, str]] = defaultdict(lambda: (0, None))
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"重复订单检测器已初始化，冷却时间: {cooldown_seconds}秒")
    
    def can_submit_order(self, symbol: str, order_type: str = "open_position") -> bool:
        """
        检查是否可以提交订单
        
        Args:
            symbol: 币种
            order_type: 订单类型（open_position, close_position, add_position等）
        
        Returns:
            bool: True表示可以提交，False表示冷却中
        """
        with self.lock:
            now = time.time()
            last_time, last_type = self.pending_orders[symbol]
            
            # 计算距离上次下单的时间
            elapsed = now - last_time
            
            # 如果在冷却时间内
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                logger.warning(
                    f"{symbol} 订单冷却中，距离上次下单{last_type}仅{elapsed:.1f}秒，"
                    f"还需等待{remaining:.1f}秒"
                )
                return False
            
            # 如果相同币种但不同类型，缩短冷却时间（30秒）
            if last_type and last_type != order_type:
                if elapsed < 30:
                    logger.info(
                        f"{symbol} 订单类型切换冷却中，距离上次{last_type}仅{elapsed:.1f}秒，"
                        f"建议切换到{order_type}"
                    )
                    return False
            
            # 记录本次提交
            self.pending_orders[symbol] = (now, order_type)
            
            logger.debug(f"{symbol} 订单已通过检测: {order_type}")
            return True
    
    def reset_cooldown(self, symbol: str):
        """
        重置指定币种的冷却时间（用于特殊场景）
        
        Args:
            symbol: 币种
        """
        with self.lock:
            self.pending_orders[symbol] = (0, None)
            logger.info(f"{symbol} 订单冷却时间已重置")
    
    def get_cooldown_remaining(self, symbol: str) -> float:
        """
        获取剩余冷却时间
        
        Args:
            symbol: 币种
        
        Returns:
            剩余冷却时间（秒）
        """
        with self.lock:
            now = time.time()
            last_time, _ = self.pending_orders[symbol]
            elapsed = now - last_time
            remaining = self.cooldown_seconds - elapsed
            return max(0, remaining)
    
    def clear_all(self):
        """清除所有冷却记录（谨慎使用）"""
        with self.lock:
            count = len([v for v in self.pending_orders.values() if v[0] > 0])
            self.pending_orders.clear()
            logger.warning(f"已清除所有冷却记录（共{count}个币种）")
    
    def get_status(self) -> Dict[str, any]:
        """
        获取检测器状态（用于监控）
        
        Returns:
            状态字典
        """
        with self.lock:
            now = time.time()
            status = {
                'cooldown_seconds': self.cooldown_seconds,
                'pending_orders': {},
            }
            
            for symbol, (last_time, last_type) in self.pending_orders.items():
                if last_time > 0:
                    elapsed = now - last_time
                    remaining = max(0, self.cooldown_seconds - elapsed)
                    status['pending_orders'][symbol] = {
                        'last_order_time': last_time,
                        'last_order_type': last_type,
                        'elapsed_seconds': elapsed,
                        'remaining_seconds': remaining
                    }
            
            return status
    
    @property
    def cooldown_symbols(self) -> int:
        """当前有冷却记录的币种数量"""
        with self.lock:
            return len([k for k, v in self.pending_orders.items() if v[0] > 0])


class OrderConflictDetector:
    """订单冲突检测器 - 检测可能冲突的订单"""
    
    def __init__(self):
        """初始化订单冲突检测器"""
        self.active_orders: Dict[str, Dict] = {}  # {symbol: order_info}
        self.lock = threading.RLock()
        
        logger.info("订单冲突检测器已初始化")
    
    def register_order(self, symbol: str, order_id: str, side: str, quantity: float) -> bool:
        """
        注册订单并检查冲突
        
        Args:
            symbol: 币种
            order_id: 订单ID
            side: 方向（BUY/SELL）
            quantity: 数量
        
        Returns:
            bool: True表示无冲突，False表示有冲突
        """
        with self.lock:
            if symbol in self.active_orders:
                existing = self.active_orders[symbol]
                
                # 检查是否是同一方向
                if existing['side'] == side:
                    logger.warning(
                        f"{symbol} 订单冲突: 已存在{existing['side']}订单{existing['order_id']}, "
                        f"现在又提交{side}订单{order_id}"
                    )
                    return False
            
            # 注册订单
            self.active_orders[symbol] = {
                'order_id': order_id,
                'side': side,
                'quantity': quantity,
                'timestamp': time.time()
            }
            
            logger.debug(f"{symbol} 订单已注册: {order_id} {side}")
            return True
    
    def unregister_order(self, symbol: str, order_id: str):
        """
        注销订单
        
        Args:
            symbol: 币种
            order_id: 订单ID
        """
        with self.lock:
            if symbol in self.active_orders:
                if self.active_orders[symbol]['order_id'] == order_id:
                    del self.active_orders[symbol]
                    logger.debug(f"{symbol} 订单已注销: {order_id}")
                else:
                    logger.warning(f"{symbol} 活跃订单ID不匹配: {order_id} vs {self.active_orders[symbol]['order_id']}")
    
    def is_position_opening(self, symbol: str) -> bool:
        """
        检查是否正在开仓中
        
        Args:
            symbol: 币种
        
        Returns:
            True表示有未完成的开仓订单
        """
        with self.lock:
            if symbol in self.active_orders:
                order = self.active_orders[symbol]
                # 3分钟内的订单认为未完成
                if time.time() - order['timestamp'] < 180:
                    return True
                else:
                    # 超时，自动清理
                    del self.active_orders[symbol]
            return False
    
    def cleanup_stale_orders(self, max_age_seconds: int = 600):
        """
        清理过期的订单记录
        
        Args:
            max_age_seconds: 最大保留时间（秒），默认10分钟
        """
        with self.lock:
            now = time.time()
            to_remove = []
            
            for symbol, order in self.active_orders.items():
                if now - order['timestamp'] > max_age_seconds:
                    to_remove.append(symbol)
            
            for symbol in to_remove:
                del self.active_orders[symbol]
            
            if to_remove:
                logger.info(f"清理了{len(to_remove)}个过期订单记录")


# 全局实例（延迟初始化）
_duplicate_detector = None
_conflict_detector = None

def get_duplicate_detector(cooldown_seconds: int = 60) -> DuplicateDetector:
    """获取重复订单检测器实例（单例模式）"""
    global _duplicate_detector
    if _duplicate_detector is None:
        _duplicate_detector = DuplicateDetector(cooldown_seconds)
    return _duplicate_detector

def get_conflict_detector() -> OrderConflictDetector:
    """获取订单冲突检测器实例（单例模式）"""
    global _conflict_detector
    if _conflict_detector is None:
        _conflict_detector = OrderConflictDetector()
    return _conflict_detector
