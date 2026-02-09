"""
订单执行辅助函数

功能：
- 处理部分成交
- 验证订单响应
- 订单安全操作工具
"""

import logging
from typing import Dict, Optional, Any, Tuple
from utils.logger import Logger

logger = Logger.get_logger('order_utils')


def validate_order_response(order: Dict[str, Any]) -> Tuple[bool, str]:
    """
    验证订单响应的有效性
    
    Args:
        order: 订单响应字典
        
    Returns:
        (is_valid, error_message) 验证结果和错误信息
    """
    # 检查必需字段
    required_fields = ['orderId', 'symbol', 'status']
    for field in required_fields:
        if field not in order:
            return False, f"订单响应缺少必需字段: {field}"
    
    # 检查订单ID
    order_id = order.get('orderId')
    if not order_id or not isinstance(order_id, (int, str)):
        return False, f"订单ID无效: {order_id}"
    
    # 检查订单状态
    status = order.get('status')
    valid_statuses = ['NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED', 'EXPIRED', 'EXPIRED_IN_MATCH']
    if status not in valid_statuses:
        return False, f"订单状态无效: {status}"
    
    return True, "OK"


def check_partial_fill(order: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    检查订单是否部分成交
    
    Args:
        order: 订单响应字典
        
    Returns:
        (is_partial, executed_qty, orig_qty) 是否部分成交、已成交数量、原始数量
    """
    try:
        order_status = order.get('status')
        executed_qty = float(order.get('executedQty', 0))
        orig_qty = float(order.get('origQty', 0))
        
        # 市价单可能没有origQty字段，用executedQty替代
        if orig_qty == 0 and executed_qty > 0:
            orig_qty = executed_qty
        
        # 部分成交状态
        is_partial = order_status in ['PARTIALLY_FILLED'] and (executed_qty < orig_qty)
        
        return is_partial, executed_qty, orig_qty
        
    except (ValueError, TypeError) as e:
        logger.error(f"检查部分成交失败: {e}")
        return False, 0, 0


def handle_partial_fill(order: Dict[str, Any], expected_quantity: float = None) -> Dict[str, Any]:
    """
    处理部分成交情况
    
    Args:
        order: 订单响应字典
        expected_quantity: 预期数量（可选，用于验证）
        
    Returns:
        处理信息字典，包含后续操作建议
    """
    is_valid, error_msg = validate_order_response(order)
    if not is_valid:
        logger.error(f"订单验证失败: {error_msg}")
        return {
            'success': False,
            'action': 'reject_order',
            'reason': error_msg
        }
    
    symbol = order.get('symbol')
    order_id = order.get('orderId')
    
    # 检查部分成交
    is_partial, executed_qty, orig_qty = check_partial_fill(order)
    
    if is_partial:
        # 部分成交处理
        canceled_qty = orig_qty - executed_qty
        
        logger.warning(
            f"{symbol} 订单部分成交: "
            f"OrderID={order_id}, 已成交={executed_qty}, 取消={canceled_qty}, "
            f"成交率={executed_qty/orig_qty*100:.1f}%"
        )
        
        # 如果部分成交比例过低（<50%），认为可能需要重新下单
        fill_ratio = executed_qty / orig_qty if orig_qty > 0 else 0
        action = 'retry_remaining' if fill_ratio < 0.5 else 'accept_partial'
        
        return {
            'success': True,
            'action': action,
            'executed_quantity': executed_qty,
            'original_quantity': orig_qty,
            'canceled_quantity': canceled_qty,
            'fill_ratio': fill_ratio,
            'reason': f'部分成交: {fill_ratio*100:.1f}%',
        }
    else:
        # 完全成交或未被拒绝
        status = order.get('status')
        
        if status == 'FILLED':
            return {
                'success': True,
                'action': 'accept_full',
                'executed_quantity': executed_qty,
                'reason': '全部成交'
            }
        elif status == 'REJECTED':
            rejected_reason = order.get('reject_reason', '未知原因')
            logger.error(f"{symbol} 订单被拒绝: {rejected_reason}")
            return {
                'success': False,
                'action': 'reject_order',
                'reason': f'订单被交易所拒绝: {rejected_reason}'
            }
        elif status == 'CANCELED':
            return {
                'success': False,
                'action': 'order_canceled',
                'reason': '订单已取消'
            }
        else:
            # NEW状态或其他
            return {
                'success': True,
                'action': 'wait_execution',
                'reason': f'订单状态: {status}'
            }


def sanitize_order_data_for_log(order: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理订单数据用于日志记录（保护敏感信息）
    
    Args:
        order: 原始订单数据
        
    Returns:
        清理后的订单数据
    """
    # 创建数据副本避免修改原数据
    sanitized = order.copy()
    
    # 脱敏处理
    if 'clientOrderId' in sanitized and sanitized['clientOrderId']:
        # 只保留前8位
        client_id = sanitized['clientOrderId']
        sanitized['clientOrderId'] = client_id[:8] + '***'
    
    # 移除可能泄露内部信息的字段
    sensitive_fields = ['apiKey', 'signature', 'recvWindow', 'timestamp']
    for field in sensitive_fields:
        sanitized.pop(field, None)
    
    return sanitized


def calculate_slippage(expected_price: float, actual_price: float, side: str) -> float:
    """
    计算滑点
    
    Args:
        expected_price: 预期价格
        actual_price: 实际价格
        side: 方向（BUY/SELL）
        
    Returns:
        滑点百分比（负数表示滑点，正数表示有利价格）
    """
    if expected_price == 0:
        return 0
    
    diff = actual_price - expected_price
    
    # 买入时，实际价格高于预期价格是滑点（负值）
    # 卖出时，实际价格低于预期价格是滑点（负值）
    if side == 'BUY':
        slippage_pct = (diff / expected_price) * 100
    else:  # SELL
        slippage_pct = -(diff / expected_price) * 100
    
    return slippage_pct


def validate_price_change(previous_price: float, current_price: float, 
                         max_change_pct: float = 50.0) -> Tuple[bool, str]:
    """
    验证价格变化是否合理
    
    Args:
        previous_price: 前一次价格
        current_price: 当前价格
        max_change_pct: 最大允许变化百分比（默认50%）
        
    Returns:
        (is_valid, reason) 验证结果和原因
    """
    # 检查价格是否有效
    if current_price <= 0:
        return False, "价格必须大于0"
    
    # 检查价格是否在合理范围内
    if current_price > 1_000_000 or current_price < 0.0001:
        return False, "价格超出合理范围"
    
    # 检查价格变化
    if previous_price > 0:
        change_pct = abs((current_price - previous_price) / previous_price * 100)
        
        if change_pct > max_change_pct:
            return False, f"价格异常波动: {change_pct:.1f}%"
    
    return True, "OK"


def get_order_summary(order: Dict[str, Any]) -> str:
    """
    生成订单摘要字符串
    
    Args:
        order: 订单数据
        
    Returns:
        订单摘要
    """
    symbol = order.get('symbol', 'UNKNOWN')
    order_id = str(order.get('orderId', ''))
    side = order.get('side', '')
    order_type = order.get('type', '')
    status = order.get('status', '')
    price = order.get('price', 'N/A')
    quantity = order.get('origQty', order.get('executedQty', 'N/A'))
    
    return f"{symbol} [{side} {order_type}] Order:{order_id} Qty:{quantity} Price:{price} Status:{status}"


class OrderTracker:
    """
    订单跟踪器
    
    功能：
    - 追踪订单状态变化
    - 检测订单超时
    - 提供订单执行统计
    """
    
    def __init__(self):
        """初始化订单跟踪器"""
        self.active_orders = {}  # {order_id: order_data}
        self.completed_orders = {}  # {order_id: order_data}
        logger.info("订单跟踪器已初始化")
    
    def add_order(self, order_id: str, symbol: str, order_type: str, 
                  expected_quantity: float):
        """
        添加订单到跟踪列表
        
        Args:
            order_id: 订单ID
            symbol: 币种
            order_type: 订单类型
            expected_quantity: 预期数量
        """
        import time
        self.active_orders[order_id] = {
            'symbol': symbol,
            'order_type': order_type,
            'expected_quantity': expected_quantity,
            'created_at': time.time(),
            'status': 'NEW'
        }
        logger.info(f"{symbol} 订单已添加到跟踪: {order_id}")
    
    def update_order(self, order_id: str, order_data: Dict[str, Any]):
        """
        更新订单状态
        
        Args:
            order_id: 订单ID
            order_data: 订单数据
        """
        if order_id in self.active_orders:
            self.active_orders[order_id].update(order_data)
            
            # 如果订单完成，移动到completed_orders
            status = order_data.get('status')
            if status in ['FILLED', 'CANCELED', 'REJECTED']:
                self.completed_orders[order_id] = self.active_orders.pop(order_id)
                logger.info(f"{order_id} 订单已完成: {status}")
    
    def check_timeout_orders(self, timeout_seconds: int = 300) -> list:
        """
        检查超时订单
        
        Args:
            timeout_seconds: 超时时间（秒）
        
        Returns:
            超时订单ID列表
        """
        import time
        now = time.time()
        timeout_orders = []
        
        for order_id, order_data in self.active_orders.items():
            created_at = order_data.get('created_at', 0)
            if now - created_at > timeout_seconds:
                timeout_orders.append(order_id)
                logger.warning(f"{order_id} 订单超时: {now - created_at:.1f}秒未完成")
        
        return timeout_orders
    
    def get_active_count(self) -> int:
        """获取活跃订单数量"""
        return len(self.active_orders)
    
    def get_completed_count(self) -> int:
        """获取已完成订单数量"""
        return len(self.completed_orders)
