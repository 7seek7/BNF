# -*- coding: utf-8 -*-
"""
风控熔断机制 - 保护账户免受极端损失

熔断规则：
1. 单日亏损超过10%触发熔断
2. 连续亏损3次触发熔断
3. 强平缓冲低于10%触发熔断
4. 黑天鹅事件触发熔断
"""

import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from framework.core.config import get_main_config
from framework.core.logger import get_logger

logger = get_logger('circuit_breaker')

# 获取主配置（模块级别单例）
_main_config = None

def _get_main_config():
    """获取主配置单例"""
    global _main_config
    if _main_config is None:
        _main_config = get_main_config()
    return _main_config


class CircuitBreakerType(Enum):
    """熔断类型"""
    DAILY_LOSS = "DAILY_LOSS"  # 单日亏损
    CONTINUOUS_LOSS = "CONTINUOUS_LOSS"  # 连续亏损
    LIQUIDATION_BUFFER = "LIQUIDATION_BUFFER"  # 强平缓冲
    BLACK_SWAN = "BLACK_SWAN"  # 黑天鹅
    MANUAL = "MANUAL"  # 手动


@dataclass
class CircuitBreakerStatus:
    """熔断状态"""
    is_triggered: bool
    breaker_type: Optional[CircuitBreakerType]
    trigger_time: Optional[datetime]
    reason: str
    cooldown_remaining: int  # 剩余冷却时间（秒）


class CircuitBreaker:
    """
    风控熔断器
    
    监控账户风险，触发熔断保护
    """
    
    def __init__(self):
        """初始化熔断器"""
        # 从配置获取参数
        cfg = _get_main_config()
        self.daily_loss_limit = cfg.emergency_daily_loss_percent if hasattr(cfg, 'emergency_daily_loss_percent') else 10.0
        self.continuous_loss_limit = cfg.emergency_continuous_loss if hasattr(cfg, 'emergency_continuous_loss') else 3
        self.liquidation_buffer_limit = getattr(cfg, 'emergency_liquidation_buffer', 0.1) * 100  # 转换为百分比
        self.cooldown_seconds = getattr(cfg, 'emergency_pause_cooldown', 1800)
        self.close_on_pause = getattr(cfg, 'emergency_close_on_pause', True)
        
        # 黑天鹅阈值
        self.black_swan_1 = getattr(cfg, 'black_swan_drop_threshold_1', -10.0)
        self.black_swan_2 = getattr(cfg, 'black_swan_drop_threshold_2', -15.0)
        self.black_swan_3 = getattr(cfg, 'black_swan_drop_threshold_3', -20.0)
        
        # 状态
        self.is_triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_type: Optional[CircuitBreakerType] = None
        self.trigger_reason = ""
        
        # 统计
        self.trade_history: List[Dict] = []
        self.continuous_loss_count = 0
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        logger.info(f"熔断器初始化: 日亏损限制={self.daily_loss_limit}%")
    
    def check_daily_pnl(self, current_balance: float, initial_balance: float) -> CircuitBreakerStatus:
        """
        检查单日亏损
        
        Args:
            current_balance: 当前余额
            initial_balance: 初始余额
        
        Returns:
            CircuitBreakerStatus
        """
        with self._lock:
            if initial_balance <= 0:
                return self._get_status()
            
            pnl_pct = (current_balance - initial_balance) / initial_balance * 100
            
            if pnl_pct <= -self.daily_loss_limit:
                self._trigger(
                    CircuitBreakerType.DAILY_LOSS,
                    f"单日亏损{pnl_pct:.1f}%超过限制{self.daily_loss_limit}%"
                )
            
            return self._get_status()
    
    def check_continuous_loss(self, is_win: bool) -> CircuitBreakerStatus:
        """
        检查连续亏损
        
        Args:
            is_win: 最近交易是否盈利
        
        Returns:
            CircuitBreakerStatus
        """
        with self._lock:
            if is_win:
                self.continuous_loss_count = 0
            else:
                self.continuous_loss_count += 1
                
                if self.continuous_loss_count >= self.continuous_loss_limit:
                    self._trigger(
                        CircuitBreakerType.CONTINUOUS_LOSS,
                        f"连续亏损{self.continuous_loss_count}次"
                    )
            
            return self._get_status()
    
    def check_liquidation_buffer(self, margin_balance: float, 
                                  total_margin: float) -> CircuitBreakerStatus:
        """
        检查强平缓冲
        
        Args:
            margin_balance: 保证金余额
            total_margin: 已用保证金
        
        Returns:
            CircuitBreakerStatus
        """
        if total_margin <= 0:
            return self._get_status()
        
        buffer_pct = (margin_balance - total_margin) / margin_balance * 100 if margin_balance > 0 else 0
        
        if buffer_pct <= self.liquidation_buffer_limit:
            self._trigger(
                CircuitBreakerType.LIQUIDATION_BUFFER,
                f"强平缓冲{buffer_pct:.1f}%低于限制{self.liquidation_buffer_limit}%"
            )
        
        return self._get_status()
    
    def check_black_swan(self, price_change_pct: float) -> CircuitBreakerStatus:
        """
        检查黑天鹅事件
        
        Args:
            price_change_pct: 价格变化百分比（负数表示下跌）
        
        Returns:
            CircuitBreakerStatus
        """
        if price_change_pct <= self.black_swan_3:
            self._trigger(
                CircuitBreakerType.BLACK_SWAN,
                f"黑天鹅事件: 价格下跌{abs(price_change_pct):.1f}%"
            )
        elif price_change_pct <= self.black_swan_2:
            logger.warning(f"二级黑天鹅警告: 价格下跌{abs(price_change_pct):.1f}%")
        elif price_change_pct <= self.black_swan_1:
            logger.warning(f"一级黑天鹅警告: 价格下跌{abs(price_change_pct):.1f}%")
        
        return self._get_status()
    
    def _trigger(self, breaker_type: CircuitBreakerType, reason: str):
        """触发熔断"""
        with self._lock:
            self.is_triggered = True
            self.trigger_time = datetime.now()
            self.trigger_type = breaker_type
            self.trigger_reason = reason
            
        logger.error(f"[熔断触发] {breaker_type.value}: {reason}")
    
    def _get_status(self) -> CircuitBreakerStatus:
        """获取当前状态"""
        if not self.is_triggered:
            return CircuitBreakerStatus(
                is_triggered=False,
                breaker_type=None,
                trigger_time=None,
                reason="",
                cooldown_remaining=0
            )
        
        # 计算剩余冷却时间
        elapsed = (datetime.now() - self.trigger_time).total_seconds() if self.trigger_time else 0
        remaining = max(0, int(self.cooldown_seconds - elapsed))
        
        return CircuitBreakerStatus(
            is_triggered=True,
            breaker_type=self.trigger_type,
            trigger_time=self.trigger_time,
            reason=self.trigger_reason,
            cooldown_remaining=remaining
        )
    
    def reset(self):
        """重置熔断状态"""
        with self._lock:
            self.is_triggered = False
            self.trigger_time = None
            self.trigger_type = None
            self.trigger_reason = ""
            self.continuous_loss_count = 0
        logger.info("熔断状态已重置")
    
    def should_close_positions(self) -> bool:
        """是否应该平仓"""
        return self.is_triggered and self.close_on_pause
    
    def can_trade(self) -> bool:
        """是否可以交易"""
        with self._lock:
            if not self.is_triggered:
                return True
            
            # 检查冷却时间
            if self.trigger_time:
                elapsed = (datetime.now() - self.trigger_time).total_seconds()
                if elapsed >= self.cooldown_seconds:
                    self.is_triggered = False
                    self.trigger_time = None
                    self.trigger_type = None
                    self.trigger_reason = ""
                    logger.info("熔断冷却时间已过，自动恢复交易")
                    return True
        
        return False
    
    def get_status(self) -> CircuitBreakerStatus:
        """获取当前状态（公共方法）"""
        return self._get_status()
    
    def get_summary(self) -> Dict:
        """获取摘要"""
        status = self._get_status()
        return {
            'is_triggered': status.is_triggered,
            'breaker_type': status.breaker_type.value if status.breaker_type else None,
            'reason': status.reason,
            'cooldown_remaining': status.cooldown_remaining,
            'continuous_loss_count': self.continuous_loss_count,
            'can_trade': self.can_trade()
        }