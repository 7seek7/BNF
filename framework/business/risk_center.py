# -*- coding: utf-8 -*-
"""
风控中心

统一风险管理

特性：
1. 日亏损限制
2. 连续亏损限制
3. 最大回撤监控
4. 黑天鹅保护
5. 紧急熔断
6. 策略级风控隔离
"""

import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from framework.core.logger import get_logger
from framework.core.events import EventBus, Event, EventType, RiskEvent
from framework.core.config import get_main_config

logger = get_logger('risk_center')


class RiskLevel(Enum):
    """风险等级"""
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'
    CRITICAL = 'CRITICAL'


@dataclass
class RiskState:
    """风控状态"""
    # 全局状态
    daily_pnl: float = 0.0
    daily_start_balance: float = 0.0
    current_balance: float = 0.0
    peak_balance: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    
    # 风控状态
    emergency_stop: bool = False
    emergency_stop_reason: str = ''
    trading_paused: bool = False
    
    # 时间戳
    last_trade_time: float = 0.0
    last_reset_date: date = field(default_factory=date.today)
    
    @property
    def daily_pnl_pct(self) -> float:
        """日盈亏比例"""
        if self.daily_start_balance <= 0:
            return 0.0
        return (self.current_balance - self.daily_start_balance) / self.daily_start_balance * 100
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100
    
    @property
    def max_drawdown(self) -> float:
        """最大回撤（从峰值到当前值的百分比）"""
        if self.peak_balance <= 0:
            return 0.0
        # 计算回撤：峰值 - 当前值 / 峰值 * 100
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        return max(0.0, drawdown)  # 确保回撤不为负数


@dataclass
class StrategyRiskState:
    """策略风控状态（每个策略独立）"""
    strategy_name: str
    daily_pnl: float = 0.0
    daily_start_allocation: float = 0.0
    current_allocation: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0  # ✅ 修复：添加缺失的连续盈利计数
    total_trades: int = 0
    winning_trades: int = 0
    paused: bool = False
    pause_reason: str = ''


class RiskCenter:
    """
    风控中心
    
    统一管理全局和策略级风控
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
            
        self.config = get_main_config()
        self.global_state = RiskState()
        self.strategy_states: Dict[str, StrategyRiskState] = {}
        
        self._lock = threading.RLock()
        # 事件总线 - 使用单例
        from framework.core.events import get_event_bus
        self._event_bus = get_event_bus()
        
        # 价格监控（用于黑天鹅检测）
        self._price_history: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(timestamp, price)]
        self._price_lock = threading.Lock()
        
        # 黑天鹅阈值 - 从配置中读取，如果没有配置则使用默认值
        self._black_swan_thresholds = {
            'level1': getattr(self.config, 'black_swan_drop_threshold_1', -10.0),  # 60秒内跌10%
            'level2': getattr(self.config, 'black_swan_drop_threshold_2', -15.0),  # 60秒内跌15%
            'level3': getattr(self.config, 'black_swan_drop_threshold_3', -20.0),  # 60秒内跌20%
        }
        
        self._initialized = True
        logger.info("风控中心已初始化")
        
    def initialize(self, initial_balance: float):
        """
        初始化
        
        Args:
            initial_balance: 初始余额
        """
        with self._lock:
            self.global_state.daily_start_balance = initial_balance
            self.global_state.current_balance = initial_balance
            self.global_state.peak_balance = initial_balance
            self.global_state.last_reset_date = date.today()
            
        logger.info(f"风控中心已初始化，初始余额: {initial_balance:.2f} USDT")
        
    def initialize_strategy(self, strategy_name: str, allocation: float):
        """初始化策略风控"""
        with self._lock:
            self.strategy_states[strategy_name] = StrategyRiskState(
                strategy_name=strategy_name,
                daily_start_allocation=allocation,
                current_allocation=allocation,
            )
        logger.info(f"策略风控初始化: {strategy_name}, 分配资金: {allocation:.2f} USDT")
        
    def can_open_position(self, strategy_name: str = None, symbol: str = None) -> Tuple[bool, str]:
        """
        检查是否可以开仓
        
        Args:
            strategy_name: 策略名称（可选）
            symbol: 币种（可选）
            
        Returns:
            (can_open, reason)
        """
        self._check_daily_reset()
        
        with self._lock:
            # 检查全局熔断
            if self.global_state.emergency_stop:
                return False, f"全局熔断: {self.global_state.emergency_stop_reason}"
                
            if self.global_state.trading_paused:
                return False, "交易已暂停"
                
            # 检查全局日亏损
            daily_loss_limit = self.config.emergency_daily_loss_percent
            if self.global_state.daily_pnl_pct < -daily_loss_limit:
                self._trigger_emergency_stop(f"日亏损达到 {self.global_state.daily_pnl_pct:.2f}%")
                return False, f"日亏损超限: {self.global_state.daily_pnl_pct:.2f}%"
                
            # 检查连续亏损
            if self.global_state.consecutive_losses >= self.config.emergency_continuous_loss:
                self._trigger_emergency_stop(f"连续亏损 {self.global_state.consecutive_losses} 次")
                return False, f"连续亏损次数: {self.global_state.consecutive_losses}"
                
            # 检查策略级风控
            if strategy_name and strategy_name in self.strategy_states:
                strategy_state = self.strategy_states[strategy_name]
                if strategy_state.paused:
                    return False, f"策略暂停: {strategy_state.pause_reason}"
                    
            return True, "风控检查通过"
            
    def update_after_trade(
        self,
        pnl: float,
        strategy_name: str = None,
        symbol: str = None
    ):
        """
        交易后更新状态
        
        Args:
            pnl: 盈亏金额
            strategy_name: 策略名称
            symbol: 币种
        """
        self._check_daily_reset()
        
        with self._lock:
            # 更新全局状态
            self.global_state.daily_pnl += pnl
            self.global_state.current_balance += pnl
            self.global_state.total_trades += 1
            self.global_state.last_trade_time = time.time()
            
            if pnl > 0:
                self.global_state.consecutive_wins += 1
                self.global_state.consecutive_losses = 0
                self.global_state.winning_trades += 1
            else:
                self.global_state.consecutive_losses += 1
                self.global_state.consecutive_wins = 0
                
            # 更新峰值
            if self.global_state.current_balance > self.global_state.peak_balance:
                self.global_state.peak_balance = self.global_state.current_balance
                
            # 更新策略状态
            if strategy_name and strategy_name in self.strategy_states:
                strategy_state = self.strategy_states[strategy_name]
                strategy_state.daily_pnl += pnl
                strategy_state.current_allocation += pnl
                strategy_state.total_trades += 1
                
                if pnl > 0:
                    strategy_state.consecutive_losses = 0
                    strategy_state.consecutive_wins += 1  # ✅ 修复：添加连续盈利计数
                    strategy_state.winning_trades += 1
                else:
                    strategy_state.consecutive_losses += 1
                    strategy_state.consecutive_wins = 0  # ✅ 修复：亏损时重置连续盈利计数
                    
            # 检查是否需要触发风控
            self._check_risk_limits()
            
        logger.debug(
            f"风控状态更新: PnL={pnl:.2f}, 日PnL={self.global_state.daily_pnl:.2f}, "
            f"连续亏损={self.global_state.consecutive_losses}"
        )
        
    def update_price(self, symbol: str, price: float):
        """更新价格（用于黑天鹅检测）"""
        timestamp = time.time()
        
        with self._price_lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = []
                
            # 只保留最近60秒的数据
            cutoff = timestamp - 60
            self._price_history[symbol] = [
                (t, p) for t, p in self._price_history[symbol] if t > cutoff
            ]
            self._price_history[symbol].append((timestamp, price))
            
    def check_black_swan(self, symbol: str, current_price: float) -> Tuple[bool, int, float]:
        """
        检查黑天鹅
        
        Returns:
            (is_black_swan, level, drop_pct)
        """
        with self._price_lock:
            if symbol not in self._price_history:
                return False, 0, 0.0
                
            history = self._price_history[symbol]
            if len(history) < 2:
                return False, 0, 0.0
                
            # 找60秒前的价格
            cutoff = time.time() - 60
            old_prices = [(t, p) for t, p in history if t < cutoff]
            
            if not old_prices:
                return False, 0, 0.0
                
            old_price = old_prices[-1][1]
            if old_price <= 0:
                return False, 0, 0.0
                
            drop_pct = (current_price - old_price) / old_price * 100
            
            # 检查各级阈值
            if drop_pct <= self._black_swan_thresholds['level3']:
                return True, 3, drop_pct
            elif drop_pct <= self._black_swan_thresholds['level2']:
                return True, 2, drop_pct
            elif drop_pct <= self._black_swan_thresholds['level1']:
                return True, 1, drop_pct
                
        return False, 0, drop_pct
        
    def _check_daily_reset(self):
        """检查是否需要日重置"""
        current_date = date.today()
        with self._lock:
            if current_date > self.global_state.last_reset_date:
                self.global_state.daily_pnl = 0.0
                self.global_state.consecutive_losses = 0
                self.global_state.consecutive_wins = 0
                self.global_state.daily_start_balance = self.global_state.current_balance
                self.global_state.last_reset_date = current_date

                # 重置策略状态（在锁内操作）
                for state in self.strategy_states.values():
                    state.daily_pnl = 0.0
                    state.consecutive_losses = 0
                    state.consecutive_wins = 0  # 修复：同时重置连胜计数
                    state.daily_start_allocation = state.current_allocation

                logger.info("风控状态已日重置")
            
    def _check_risk_limits(self):
        """检查风控限制"""
        # 检查日亏损（使用配置的1.5倍作为紧急阈值）
        if self.global_state.daily_pnl_pct < -self.config.emergency_daily_loss_percent * 1.5:
            self._trigger_emergency_stop(
                f"日亏损严重: {self.global_state.daily_pnl_pct:.2f}%"
            )

        # 检查回撤（使用配置的回撤限制，而不是硬编码的50%）
        drawdown = self.global_state.max_drawdown
        max_drawdown_limit = getattr(self.config, 'max_drawdown_percent', 50)  # 默认50%
        if drawdown > max_drawdown_limit:
            self._trigger_emergency_stop(f"回撤过大: {drawdown:.2f}% (限制: {max_drawdown_limit}%)")
            
    def _trigger_emergency_stop(self, reason: str):
        """触发紧急熔断"""
        self.global_state.emergency_stop = True
        self.global_state.emergency_stop_reason = reason
        
        # 发送事件
        self._event_bus.publish(RiskEvent(
            event_type=EventType.EMERGENCY_STOP,
            risk_type='EMERGENCY_STOP',
            risk_level='CRITICAL',
            message=reason,
            data={'reason': reason}
        ))
        
        logger.critical(f"紧急熔断触发: {reason}")
        
    def pause_trading(self, reason: str = "手动暂停"):
        """暂停交易"""
        with self._lock:
            self.global_state.trading_paused = True
            
        logger.warning(f"交易已暂停: {reason}")
        
    def resume_trading(self):
        """恢复交易"""
        with self._lock:
            self.global_state.trading_paused = False
            self.global_state.emergency_stop = False
            self.global_state.emergency_stop_reason = ''
            
        logger.info("交易已恢复")
        
    def pause_strategy(self, strategy_name: str, reason: str):
        """暂停策略"""
        with self._lock:
            if strategy_name in self.strategy_states:
                self.strategy_states[strategy_name].paused = True
                self.strategy_states[strategy_name].pause_reason = reason
                
        logger.warning(f"策略已暂停: {strategy_name}, 原因: {reason}")
        
    def resume_strategy(self, strategy_name: str):
        """恢复策略"""
        with self._lock:
            if strategy_name in self.strategy_states:
                self.strategy_states[strategy_name].paused = False
                self.strategy_states[strategy_name].pause_reason = ''
                
        logger.info(f"策略已恢复: {strategy_name}")
        
    def get_global_state(self) -> RiskState:
        """获取全局风控状态"""
        return self.global_state
        
    def get_strategy_state(self, strategy_name: str) -> Optional[StrategyRiskState]:
        """获取策略风控状态"""
        return self.strategy_states.get(strategy_name)
        
    def get_all_states(self) -> Dict[str, Any]:
        """获取所有风控状态"""
        return {
            'global': {
                'daily_pnl': self.global_state.daily_pnl,
                'daily_pnl_pct': self.global_state.daily_pnl_pct,
                'current_balance': self.global_state.current_balance,
                'consecutive_losses': self.global_state.consecutive_losses,
                'win_rate': self.global_state.win_rate,
                'max_drawdown': self.global_state.max_drawdown,
                'emergency_stop': self.global_state.emergency_stop,
                'trading_paused': self.global_state.trading_paused,
            },
            'strategies': {
                name: {
                    'daily_pnl': state.daily_pnl,
                    'consecutive_losses': state.consecutive_losses,
                    'paused': state.paused,
                    'pause_reason': state.pause_reason,
                }
                for name, state in self.strategy_states.items()
            }
        }


# 便捷函数
_risk_center: Optional[RiskCenter] = None


def get_risk_center() -> RiskCenter:
    """获取风控中心单例"""
    global _risk_center
    if _risk_center is None:
        _risk_center = RiskCenter()
    return _risk_center
