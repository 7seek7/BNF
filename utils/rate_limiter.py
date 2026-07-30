"""
API限流器

功能：
- 防止触发币安API限流(429错误)
- 动态调整请求频率
- 请求权重管理
- 多线程安全

币安限流规则：
- 现货: 1200 请求/分钟
- 期货: 2400 请求/分钟
- 每个请求有不同权重(1-5)
- IP限流更严格
"""

import time
import threading
from typing import Dict, List, Optional
from collections import deque
from framework.core.logger import get_logger

logger = get_logger('rate_limiter')


class APIRequestWeight:
    """API请求权重参考
    
    来源: Binance API文档
    通用规则:
    - 简单查询: 权重1
    - 复杂查询: 权重5
    - 订单操作: 权重1-5
    """
    
    # 期货端点权重
    FUTURES_WEIGHTS = {
        # 查询类 (权重1)
        'get_open_orders': 1,
        'get_account': 5,  # 账户信息权重较高
        'get_position': 1,
        'get_klines': 1,
        'get_ticker_price': 1,
        'get_ticker_book': 1,
        
        # 订单操作 (权重1-5)
        'create_order': 1,
        'cancel_order': 1,
        'cancel_all_orders': 1,
        'modify_order': 1,
        
        # 杠杆操作 (权重10)
        'change_leverage': 10,
        'change_margin_type': 10,
    }


class RateLimiter:
    """
    API请求限流器
    
    功能：
    - 动态计算等待时间
    - 滑动窗口限流算法
    - 请求权重跟踪
    """
    
    def __init__(self, max_requests: int = 2400, period: int = 60):
        """
        初始化限流器
        
        Args:
            max_requests: 时间窗口内最大请求数（默认2400）
            period: 时间窗口（秒，默认60秒）
        """
        self.max_requests = max_requests
        self.period = period
        
        # 请求时间戳队列（滑动窗口）
        self.request_times: deque = deque()
        
        # 当前已使用的请求数
        self.used_requests = 0
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 统计信息
        self.total_requests = 0
        self.wait_time_total = 0
        self.rate_limit_hits = 0
        
        logger.info(f"限流器已初始化: {max_requests}请求/{period}秒")
    
    def wait_if_needed(self, weight: int = 1, force_wait: bool = True) -> float:
        """
        如果接近限流，等待可用配额
        
        Args:
            weight: 请求权重
            force_wait: 是否强制等待（False时仅返回剩余时间）
        
        Returns:
            实际等待时间（秒）
        """
        with self.lock:
            current_time = time.time()
            
            # 清理过期时间戳
            self._cleanup_old_requests(current_time)
            
            # 计算可用配额
            available = self.max_requests - self.used_requests
            
            # 如果配额不足
            if weight > available:
                self.rate_limit_hits += 1
                
                # 计算需要等待的时间
                wait_time = self._calculate_wait_time(weight)
                
                if force_wait:
                    if wait_time > 0:
                        logger.info(
                            f"接近限流: 当前使用{self.used_requests}/{self.max_requests}, "
                            f"请求权重{weight}, 等待{wait_time:.1f}秒..."
                        )
                        
                        # 修复：持锁sleep会导致线程饥饿
                        # 先释放锁，等待后再重新获取
                        self.lock.release()
                        try:
                            time.sleep(wait_time)
                        finally:
                            self.lock.acquire()
                        
                        self.wait_time_total += wait_time
                        
                        # 重新获取锁后重新清理并记录，防止竞态导致超额
                        current_time = time.time()
                        self._cleanup_old_requests(current_time)
                        
                        # 重新检查配额（释放锁期间其他线程可能已消耗配额）
                        while self.used_requests >= self.max_requests:
                            wait_time = self.request_times[0] + self.period - current_time
                            if wait_time > 0:
                                self.lock.release()
                                try:
                                    time.sleep(wait_time)
                                finally:
                                    self.lock.acquire()
                                current_time = time.time()
                                self._cleanup_old_requests(current_time)
                            else:
                                break
                        
                        # 记录请求（等待后的时间）
                        self.request_times.append(current_time)
                        self.used_requests += weight
                        self.total_requests += weight
                        
                        return wait_time
                else:
                    # 不强制等待，返回等待时间
                    return wait_time
            else:
                # 配额充足，直接记录请求（修复：total_requests应按权重累加）
                self.request_times.append(current_time)
                self.used_requests += weight
                self.total_requests += weight

                return 0
    
    def _cleanup_old_requests(self, current_time: float):
        """清理过期的请求记录"""
        cutoff_time = current_time - self.period
        
        # 超过窗口期的请求
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
            
        # 重新计算已使用请求数（请求计数，不含权重）
        self.used_requests = len(self.request_times)
    
    def _calculate_wait_time(self, weight: int) -> float:
        """
        计算等待时间
        
        Args:
            weight: 请求权重
        
        Returns:
            需要等待的秒数
        """
        if not self.request_times:
            return 0
        
        # 找到最早的请求时间
        oldest_time = self.request_times[0]
        current_time = time.time()
        
        # 如果最旧的请求还没超过周期，需要等待到它过期
        if current_time - oldest_time < self.period:
            # 需要等最旧的请求过期，再加上weight对应的等待
            time_to_expire = self.period - (current_time - oldest_time)
            
            # 如果还需额外配额，估计需要再等一个周期
            if weight > (self.max_requests - self.used_requests):
                return time_to_expire + (self.period * ((weight - (self.max_requests - self.used_requests)) / self.max_requests))
            else:
                return time_to_expire
        else:
            # 所有请求都已过期，但配额仍不足（因为清理了）
            # 简单估计：按比例等待
            excess = weight - (self.max_requests - self.used_requests)
            return (excess / self.max_requests) * self.period
    
    def get_available_capacity(self) -> int:
        """
        获取可用配额
        
        Returns:
            可用配额数
        """
        with self.lock:
            current_time = time.time()
            self._cleanup_old_requests(current_time)
            return max(0, self.max_requests - self.used_requests)
    
    def wait_until_available(self, weight: int = 1, max_wait_seconds: int = 60) -> bool:
        """
        等待直到可用配额足够
        
        Args:
            weight: 请求权重
            max_wait_seconds: 最大等待时间
        
        Returns:
            是否成功（超时返回False）
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            if self.get_available_capacity() >= weight:
                return True
            time.sleep(1)
        
        logger.warning(f"限流器等待超时：{max_wait_seconds}秒后仍无可用配额")
        return False
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取限流器统计信息
        
        Returns:
            统计数据
        """
        with self.lock:
            return {
                'max_requests': self.max_requests,
                'period_seconds': self.period,
                'used_requests': self.used_requests,
                'available': self.max_requests - self.used_requests,
                'total_requests': self.total_requests,
                'rate_limit_hits': self.rate_limit_hits,
                'total_wait_time': self.wait_time_total,
                'utilization': f"{(self.used_requests / self.max_requests * 100):.1f}%"
            }
    
    def reset(self):
        """重置限流器（谨慎使用）"""
        with self.lock:
            self.request_times.clear()
            self.used_requests = 0
            logger.warning("限流器已重置")


class AdaptiveRateLimiter(RateLimiter):
    """
    自适应限流器
    
    功能：
    - 根据响应时间动态调整限流
    - 检测API响应延迟上升
    - 429错误后临时降频
    """
    
    def __init__(self, max_requests: int = 2400, period: int = 60):
        """
        初始化自适应限流器
        
        Args:
            max_requests: 最大请求数
            period: 时间窗口
        """
        super().__init__(max_requests, period)
        
        # 自适应参数
        self.min_requests = max_requests * 0.5  # 最小请求率（触发429后）
        self.current_max = max_requests       # 当前最大请求数
        self.recovery_factor = 1.1             # 恢复因子
        
        # 检测历史
        self.latency_history = deque(maxlen=100)
        self.last_429_time = 0                 # 上次429错误时间
        
        # 状态
        self.backup_mode = False
        self.backup_mode_start_time = 0
        self.backup_duration = 300                # 降频模式持续时间（秒）
    
    def record_latency(self, latency_seconds: float):
        """
        记录API响应延迟
        
        Args:
            latency_seconds: 响应延迟（秒）
        """
        self.latency_history.append(latency_seconds)
    
    def record_429_error(self):
        """
        记录429错误（API限流）
        """
        self.last_429_time = time.time()
        
        # 进入降频模式
        if not self.backup_mode:
            self.backup_mode = True
            self.backup_mode_start_time = time.time()
            self.current_max = self.min_requests
            
            logger.warning(
                f"检测到API限流(429)，进入降频模式：{self.min_requests}请求/{self.period}秒"
            )
    
    def wait_if_needed(self, weight: int = 1, force_wait: bool = True) -> float:
        """
        等待（自适应版本）
        
        Args:
            weight: 请求权重
            force_wait: 是否强制等待（保持与父类签名兼容）
        
        Returns:
            实际等待时间
        """
        # 检查是否在降频模式
        if self.backup_mode:
            elapsed = time.time() - self.backup_mode_start_time
            if elapsed > self.backup_duration:
                # 退出降频模式，逐步恢复
                self.backup_mode = False
                self.current_max = min(
                    self.max_requests,
                    int(self.current_max * self.recovery_factor)
                )
                logger.info(f"退出降频模式，当前限制: {self.current_max}请求/{self.period}秒")
            else:
                logger.debug(f"降频模式中，剩余{self.backup_duration - elapsed:.1f}秒")
        
        # 使用当前最大请求数更新计算
        old_max = self.max_requests
        self.max_requests = self.current_max
        
        wait_time = super().wait_if_needed(weight, force_wait=False)
        
        # 恢复原设置
        self.max_requests = old_max
        
        # 如果需要等待，执行等待
        if wait_time > 0:
            # 修复：释放锁后再等待，避免线程饥饿
            self.lock.release()
            try:
                time.sleep(wait_time)
            finally:
                self.lock.acquire()
            
            self.wait_time_total += wait_time
            
            # 记录请求（修复：total_requests应按权重累加）
            self.request_times.append(time.time())
            self.used_requests += weight
            self.total_requests += weight
        
        return wait_time
    
    def get_availability_score(self) -> float:
        """
        获取API可用性评分(0-1)
        
        Returns:
            评分：1表示完全可用，0表示完全不可用
        """
        available = self.get_available_capacity()
        return available / self.max_requests


# 全局实例（延迟初始化）
_rate_limiter = None

def get_rate_limiter(max_requests: int = 2400, period: int = 60) -> RateLimiter:
    """获取限流器实例（单例模式）"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = AdaptiveRateLimiter(max_requests, period)
    return _rate_limiter

def get_adaptive_rate_limiter() -> AdaptiveRateLimiter:
    """获取自适应限流器实例"""
    global _rate_limiter
    if _rate_limiter is None or not isinstance(_rate_limiter, AdaptiveRateLimiter):
        _rate_limiter = AdaptiveRateLimiter()
    return _rate_limiter
