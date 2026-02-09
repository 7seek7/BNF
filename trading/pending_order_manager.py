#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pending Orders 管理器
"""
import threading
from datetime import datetime, timedelta
from enum import Enum
from utils.logger import Logger

logger = Logger.get_logger('pending_orders')


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"        # 待提交
    SUBMITTED = "submitted"    # 已提交到交易所
    FILLED = "filled"        # 已成交
    CANCELLED = "cancelled"    # 已取消
    FAILED = "failed"         # 失败


class ErrorType(Enum):
    """错误类型"""
    RETRYABLE = "retryable"              # 可重试错误（网络、超时）
    NON_RETRYABLE = "non_retryable"      # 不可重试错误（余额不足、权限错误）
    RATE_LIMIT = "rate_limit"             # 限流错误
    POSITION_LIMIT = "position_limit"       # 仓位限制错误


class PendingOrder:
    """待处理订单数据结构"""

    def __init__(self, order_id: str, symbol: str, side: str, 
                 quantity: float, price: float, 
                 order_type: str = "LIMIT",
                 reason: str = "", 
                 add_percent: float = 0):
        """
        初始化待处理订单
        
        Args:
            order_id: 订单ID
            symbol: 币种
            side: 方向（BUY/SELL）
            quantity: 数量
            price: 价格
            order_type: 订单类型（MARKET/LIMIT）
            reason: 原因（加仓原因）
            add_percent: 加仓比例（%）
        """
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.reason = reason
        self.add_percent = add_percent

        # 状态管理
        self.status = OrderStatus.PENDING
        self.create_time = datetime.now()
        self.submit_time: datetime = None
        self.fill_time: datetime = None
        self.retry_count = 0

        # 错误信息
        self.last_error: str = None
        self.error_type: ErrorType = None


class PendingOrderManager:
    """Pending订单管理器"""

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 300,
                 retry_delay: float = 2.0, max_retry_delay: float = 60.0):
        """
        初始化Pending订单管理器

        Args:
            max_retries: 最大重试次数
            timeout_seconds: 超时时间（秒）
            retry_delay: 基础重试延迟（秒）
            max_retry_delay: 最大重试延迟（秒）
        """
        self._orders: dict[str, PendingOrder] = {}  # order_id -> PendingOrder
        self._lock = threading.Lock()
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._retry_delay = retry_delay
        self._max_retry_delay = max_retry_delay

        # 统计信息
        self._stats = {
            'total_added': 0,
            'total_filled': 0,
            'total_failed': 0,
            'total_cancelled': 0,
            'total_expired': 0,
            'retry_count': 0
        }

    def add_order(self, order_id: str, symbol: str, side: str,
                 quantity: float, price: float, 
                 order_type: str = "LIMIT",
                 reason: str = "", add_percent: float = 0) -> None:
        """
        添加订单到pending列表

        Args:
            order_id: 订单ID
            symbol: 币种
            side: 方向（BUY/SELL）
            quantity: 数量
            price: 价格
            order_type: 订单类型（MARKET/LIMIT）
            reason: 原因
            add_percent: 加仓比例（%）
        """
        with self._lock:
            self._orders[order_id] = PendingOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type=order_type,
                reason=reason,
                add_percent=add_percent
            )
            self._stats['total_added'] += 1

            logger.info(f"[PendingOrders] 添加订单: {order_id} ({symbol} {side} "
                       f"{quantity} @ {price})")

    def mark_submitted(self, order_id: str) -> None:
        """
        标记订单已提交到交易所

        Args:
            order_id: 订单ID
        """
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                order.status = OrderStatus.SUBMITTED
                order.submit_time = datetime.now()
                logger.info(f"[PendingOrders] 订单已提交: {order_id}")

    def mark_filled(self, order_id: str) -> None:
        """
        标记订单已成交

        Args:
            order_id: 订单ID
        """
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                order.status = OrderStatus.FILLED
                order.fill_time = datetime.now()
                self._stats['total_filled'] += 1
                logger.info(f"[PendingOrders] 订单已成交: {order_id}")

    def mark_failed(self, order_id: str, error: str) -> None:
        """
        标记订单失败

        Args:
            order_id: 订单ID
            error: 错误信息
        """
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                order.status = OrderStatus.FAILED
                order.last_error = error
                order.retry_count += 1
                order.error_type = self._classify_error(error)
                self._stats['total_failed'] += 1
                self._stats['retry_count'] += 1

                logger.error(f"[PendingOrders] 订单失败: {order_id}, "
                             f"重试次数: {order.retry_count}, "
                             f"错误类型: {order.error_type.value}, "
                             f"错误: {error}")

    def mark_cancelled(self, order_id: str) -> None:
        """
        标记订单已取消

        Args:
            order_id: 订单ID
        """
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                order.status = OrderStatus.CANCELLED
                self._stats['total_cancelled'] += 1
                logger.info(f"[PendingOrders] 订单已取消: {order_id}")

    def cleanup(self, order_id: str) -> None:
        """
        清理已完成的订单

        Args:
            order_id: 订单ID
        """
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    del self._orders[order_id]
                    logger.info(f"[PendingOrders] 清理订单: {order_id}")

    def cleanup_stale_orders(self) -> int:
        """
        清理超时的pending订单

        Returns:
            int: 清理的订单数量
        """
        with self._lock:
            current_time = datetime.now()
            cleaned_count = 0
            expired_orders = []

            for order_id, order in self._orders.items():
                time_diff = (current_time - order.create_time).total_seconds()

                # 超时订单
                if time_diff > self._timeout_seconds:
                    expired_orders.append(order_id)

            # 清理超时订单
            for order_id in expired_orders:
                order = self._orders[order_id]
                order.status = OrderStatus.CANCELLED  # 标记为取消
                self._stats['total_expired'] += 1
                del self._orders[order_id]
                cleaned_count += 1
                logger.warning(f"[PendingOrders] 清理超时订单: {order_id}, "
                                f"时长: {time_diff:.0f}秒")

            if cleaned_count > 0:
                logger.info(f"[PendingOrders] 共清理 {cleaned_count} 个超时订单")

            return cleaned_count

    def get_pending_orders(self, symbol: str = None) -> list[PendingOrder]:
        """
        获取待处理的订单

        Args:
            symbol: 币种（可选），如果指定则只返回该币种的订单

        Returns:
            list[PendingOrder]: 待处理订单列表
        """
        with self._lock:
            if symbol:
                return [order for order in self._orders.values()
                        if order.symbol == symbol and order.status == OrderStatus.PENDING]
            else:
                return [order for order in self._orders.values()
                        if order.status == OrderStatus.PENDING]

    def get_failed_orders(self, symbol: str = None) -> list[PendingOrder]:
        """
        获取失败的订单（可重试）

        Args:
            symbol: 币种（可选）

        Returns:
            list[PendingOrder]: 失败订单列表
        """
        with self._lock:
            if symbol:
                return [order for order in self._orders.values()
                        if order.symbol == symbol
                        and order.status == OrderStatus.FAILED
                        and order.retry_count < self._max_retries
                        and order.error_type in [ErrorType.RETRYABLE, ErrorType.RATE_LIMIT]]
            else:
                return [order for order in self._orders.values()
                        if order.status == OrderStatus.FAILED
                        and order.retry_count < self._max_retries
                        and order.error_type in [ErrorType.RETRYABLE, ErrorType.RATE_LIMIT]]

    def should_retry(self, order_id: str) -> bool:
        """
        检查订单是否应该重试

        Args:
            order_id: 订单ID

        Returns:
            bool: 是否应该重试
        """
        with self._lock:
            if order_id not in self._orders:
                return False

            order = self._orders[order_id]

            # 达到最大重试次数
            if order.retry_count >= self._max_retries:
                return False

            # 不可重试的错误
            if order.error_type in [ErrorType.NON_RETRYABLE, ErrorType.POSITION_LIMIT]:
                return False

            # 可重试的错误
            if order.error_type in [ErrorType.RETRYABLE, ErrorType.RATE_LIMIT]:
                return True

            return False

    def get_retry_delay(self, order_id: str) -> float:
        """
        获取重试延迟（指数退避）

        Args:
            order_id: 订单ID

        Returns:
            float: 重试延迟（秒）
        """
        with self._lock:
            if order_id not in self._orders:
                return self._retry_delay

            order = self._orders[order_id]

            # 限流错误使用指数退避
            if order.error_type == ErrorType.RATE_LIMIT:
                delay = min(self._retry_delay * (2 ** order.retry_count), self._max_retry_delay)
                return delay

            # 其他错误使用线性延迟
            delay = min(self._retry_delay * (order.retry_count + 1), self._max_retry_delay)
            return delay

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            dict: 统计信息
        """
        with self._lock:
            return self._stats.copy()

    def has_pending_order(self, symbol: str, side: str = None) -> bool:
        """
        检查是否有指定币种的pending订单

        Args:
            symbol: 币种
            side: 方向（可选）

        Returns:
            bool: 是否有pending订单
        """
        with self._lock:
            for order in self._orders.values():
                if order.symbol == symbol and order.status == OrderStatus.PENDING:
                    if side is None or order.side == side:
                        return True
            return False

    def _classify_error(self, error_msg: str) -> ErrorType:
        """
        分类错误类型

        Args:
            error_msg: 错误消息

        Returns:
            ErrorType: 错误类型
        """
        error_lower = error_msg.lower()

        # 仓位限制错误
        if 'exceeded maximum allowable position' in error_lower:
            return ErrorType.POSITION_LIMIT
        if 'position limit' in error_lower:
            return ErrorType.POSITION_LIMIT

        # 余额不足错误
        if 'insufficient' in error_lower or 'balance' in error_lower:
            return ErrorType.NON_RETRYABLE

        # 限流错误
        if 'rate limit' in error_lower or 'too many requests' in error_lower:
            return ErrorType.RATE_LIMIT

        # 网络错误
        if 'timeout' in error_lower or 'connection' in error_lower:
            return ErrorType.RETRYABLE

        # 默认可重试
        return ErrorType.RETRYABLE
