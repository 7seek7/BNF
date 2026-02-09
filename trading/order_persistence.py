"""
订单持久化管理器

功能：
- 将所有订单信息持久化到SQLite数据库
- 系统崩溃重启后恢复订单状态
- 防止重复下单
- 支持订单历史查询
"""

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from utils.logger import Logger

logger = Logger.get_logger('order_persistence')


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"        # 待提交
    SUBMITTED = "submitted"    # 已提交到交易所
    PARTIAL = "partial"        # 部分成交
    FILLED = "filled"          # 全部成交
    CANCELLED = "cancelled"    # 已取消
    FAILED = "failed"          # 失败
    REJECTED = "rejected"      # 被交易所拒绝


class OrderPersistence:
    """订单持久化管理器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化订单持久化
        
        Args:
            db_path: 数据库文件路径，默认为 data/orders.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'data' / 'orders.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        
        # 初始化数据库
        self._init_db()
        
        logger.info(f"订单持久化已初始化: {self.db_path}")
    
    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Orders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    order_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    executed_price REAL,
                    executed_quantity REAL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extra_info TEXT
                )
            ''')
            
            # 状态变更历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_status_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id)
                )
            ''')
            
            # 索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at)')
            
            conn.commit()
    
    def save_order(self, order_data: Dict[str, Any]) -> bool:
        """
        保存订单信息
        
        Args:
            order_data: 订单数据字典，必须包含 order_id
            
        Returns:
            bool: 保存是否成功
        """
        try:
            order_id = order_data.get('order_id')
            if not order_id:
                logger.error("订单数据缺少order_id")
                return False
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 检查订单是否已存在
                    cursor.execute('SELECT id FROM orders WHERE order_id = ?', (order_id,))
                    if cursor.fetchone():
                        # 更新现有订单
                        cursor.execute('''
                            UPDATE orders 
                            SET status = ?, 
                                executed_price = ?,
                                executed_quantity = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE order_id = ?
                        ''', (
                            order_data.get('status', 'PENDING'),
                            order_data.get('executed_price'),
                            order_data.get('executed_quantity'),
                            order_id
                        ))
                    else:
                        # 插入新订单
                        cursor.execute('''
                            INSERT INTO orders (
                                order_id, order_type, symbol, side, quantity, 
                                price, executed_price, executed_quantity, 
                                status, reason, extra_info
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            order_id,
                            order_data.get('order_type', 'MARKET'),
                            order_data.get('symbol'),
                            order_data.get('side'),
                            order_data.get('quantity'),
                            order_data.get('price'),
                            order_data.get('executed_price'),
                            order_data.get('executed_quantity'),
                            order_data.get('status', 'PENDING'),
                            order_data.get('reason', ''),
                            json.dumps(order_data.get('extra_info', {}))
                        ))
                    
                    # 记录状态变更
                    status = order_data.get('status', 'PENDING')
                    if status != 'PENDING':
                        cursor.execute('''
                            INSERT INTO order_status_history (order_id, status, reason)
                            VALUES (?, ?, ?)
                        ''', (order_id, status, order_data.get('reason', '')))
                    
                    conn.commit()
                    logger.debug(f"订单已保存: {order_id} - {status}")
                    return True
            
        except Exception as e:
            logger.error(f"保存订单失败 {order_data.get('order_id')}: {e}")
            return False
    
    def update_order_status(self, order_id: str, status: OrderStatus, reason: str = "") -> bool:
        """
        更新订单状态
        
        Args:
            order_id: 订单ID
            status: 新状态
            reason: 状态变更原因
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 更新状态
                    cursor.execute('''
                        UPDATE orders 
                        SET status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE order_id = ?
                    ''', (status.value, order_id))
                    
                    # 记录历史
                    cursor.execute('''
                        INSERT INTO order_status_history (order_id, status, reason)
                        VALUES (?, ?, ?)
                    ''', (order_id, status.value, reason))
                    
                    conn.commit()
                    logger.info(f"订单状态已更新: {order_id} -> {status.value} ({reason})")
                    return True
            
        except Exception as e:
            logger.error(f"更新订单状态失败 {order_id}: {e}")
            return False
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """
        获取所有待处理的订单（启动时恢复用）
        
        Returns:
            订单列表
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 获取SUBMITTED和PARTIAL状态的订单
                    cursor.execute('''
                        SELECT * FROM orders 
                        WHERE status IN ('SUBMITTED', 'PARTIAL')
                        ORDER BY created_at
                    ''')
                    
                    columns = [desc[0] for desc in cursor.description]
                    orders = []
                    
                    for row in cursor.fetchall():
                        order = dict(zip(columns, row))
                        # 解析extra_info
                        if order['extra_info']:
                            order['extra_info'] = json.loads(order['extra_info'])
                        orders.append(order)
                    
                    logger.info(f"找到 {len(orders)} 个待恢复订单")
                    return orders
            
        except Exception as e:
            logger.error(f"获取待处理订单失败: {e}")
            return []
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        获取订单信息
        
        Args:
            order_id: 订单ID
            
        Returns:
            订单信息，不存在返回None
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT * FROM orders WHERE order_id = ?', (order_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        columns = [desc[0] for desc in cursor.description]
                        order = dict(zip(columns, row))
                        if order['extra_info']:
                            order['extra_info'] = json.loads(order['extra_info'])
                        return order
                    
                    return None
            
        except Exception as e:
            logger.error(f"获取订单失败 {order_id}: {e}")
            return None
    
    def get_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        获取指定币种的所有订单
        
        Args:
            symbol: 币种
            
        Returns:
            订单列表
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT * FROM orders 
                        WHERE symbol = ?
                        ORDER BY created_at DESC
                        LIMIT 100
                    ''', (symbol,))
                    
                    columns = [desc[0] for desc in cursor.description]
                    orders = []
                    
                    for row in cursor.fetchall():
                        order = dict(zip(columns, row))
                        if order['extra_info']:
                            order['extra_info'] = json.loads(order['extra_info'])
                        orders.append(order)
                    
                    return orders
            
        except Exception as e:
            logger.error(f"获取订单列表失败 {symbol}: {e}")
            return []
    
    def cleanup_old_orders(self, days: int = 7):
        """
        清理旧订单记录
        
        Args:
            days: 保留最近几天的数据
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 删除已完成且超过指定天数的订单
                    cursor.execute('''
                        DELETE FROM orders 
                        WHERE created_at < datetime('now', '-' || ? || ' days')
                        AND status IN ('FILLED', 'CANCELLED', 'FAILED', 'REJECTED')
                    ''', (days,))
                    
                    deleted = cursor.rowcount
                    conn.commit()
                    logger.info(f"清理了 {deleted} 条旧订单记录")
            
        except Exception as e:
            logger.error(f"清理旧订单失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取订单统计信息
        
        Returns:
            统计数据
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 按状态统计
                    cursor.execute('''
                        SELECT status, COUNT(*) as count
                        FROM orders
                        GROUP BY status
                    ''')
                    
                    by_status = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # 总数
                    cursor.execute('SELECT COUNT(*) FROM orders')
                    total = cursor.fetchone()[0]
                    
                    # 今日订单
                    cursor.execute('''
                        SELECT COUNT(*) FROM orders
                        WHERE DATE(created_at) = DATE('now')
                    ''')
                    today = cursor.fetchone()[0]
                    
                    return {
                        'total': total,
                        'today': today,
                        'by_status': by_status
                    }
            
        except Exception as e:
            logger.error(f"获取订单统计失败: {e}")
            return {}


# 全局实例（延迟初始化）
_persistence_instance = None

def get_order_persistence(db_path: str = None) -> OrderPersistence:
    """
    获取订单持久化实例（单例模式）
    
    Args:
        db_path: 数据库路径
        
    Returns:
        OrderPersistence实例
    """
    global _persistence_instance
    if _persistence_instance is None:
        _persistence_instance = OrderPersistence(db_path)
    return _persistence_instance
