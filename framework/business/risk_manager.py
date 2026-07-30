# -*- coding: utf-8 -*-
"""
风险控制模块 - 参考trading_GPT实现
包含：日亏损限制、连续亏损限制、黑天鹅保护、紧急熔断
"""
import time
import threading
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from config.settings import Settings

# 延迟初始化 settings，避免模块导入时污染环境变量
_settings_instance = None

def _get_settings():
    """获取 Settings 实例（延迟初始化）"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

@dataclass
class RiskState:
    """风控状态"""
    daily_pnl: float = 0.0
    daily_start_balance: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    last_trade_time: float = 0.0
    last_trade_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_balance: float = 0.0
    emergency_stop: bool = False
    emergency_stop_reason: str = ""


class RiskManager:
    """
    风险管理器
    
    功能：
    1. 日亏损限制
    2. 连续亏损限制
    3. 最大回撤监控
    4. 黑天鹅保护
    5. 紧急熔断
    """
    
    def __init__(self):
        from framework.core.config import get_main_config
        
        self.settings = _get_settings()
        self.state = RiskState()
        self.lock = threading.RLock()
        
        # 风控参数（使用正确的属性名）
        self.max_daily_loss_pct = getattr(self.settings, 'EMERGENCY_DAILY_LOSS_PERCENT', 10.0)
        self.max_consecutive_losses = getattr(self.settings, 'EMERGENCY_CONTINUOUS_LOSS', 3)
        
        # 黑天鹅参数
        self.black_swan_thresholds = {
            'drop_1': -10.0,  # 第一级：60秒内下跌10%
            'drop_2': -15.0,  # 第二级：60秒内下跌15%
            'drop_3': -20.0,  # 第三级：60秒内下跌20%
        }
        
        # 价格监控（用于黑天鹅检测）
        self._price_history: Dict[str, list] = {}
        self._price_history_lock = threading.Lock()
        
        # 初始化时间
        self._last_reset_date = datetime.now().date()
    
    def initialize(self, current_balance: float):
        """初始化"""
        with self.lock:
            self.state.daily_start_balance = current_balance
            self.state.peak_balance = current_balance
            self._last_reset_date = datetime.now().date()
    
    def _check_daily_reset(self, current_balance: float = None):
        """检查是否需要日重置"""
        current_date = datetime.now().date()
        if current_date > self._last_reset_date:
            with self.lock:
                self.state.daily_pnl = 0.0
                self.state.consecutive_losses = 0
                self.state.consecutive_wins = 0
                self._last_reset_date = current_date

                if current_balance and current_balance > 0:
                    self.state.daily_start_balance = current_balance
    
    def can_open_position(self, current_balance: float, symbol: str = None) -> tuple:
        """
        检查是否可以开仓
        
        Returns:
            (can_open: bool, reason: str)
        """
        self._check_daily_reset(current_balance)
        
        with self.lock:
            # 紧急熔断检查
            if self.state.emergency_stop:
                return False, f"紧急熔断: {self.state.emergency_stop_reason}"
            
            # 日亏损检查
            if self.state.daily_start_balance > 0:
                daily_pnl_pct = (current_balance - self.state.daily_start_balance) / self.state.daily_start_balance * 100
                if daily_pnl_pct < -self.max_daily_loss_pct:
                    return False, f"日亏损超限: {daily_pnl_pct:.2f}% < -{self.max_daily_loss_pct}%"
            
            # 连续亏损检查
            if self.state.consecutive_losses >= self.max_consecutive_losses:
                return False, f"连续亏损次数: {self.state.consecutive_losses}"
            
            return True, "风控检查通过"
    
    def update_after_trade(self, pnl: float, current_balance: float):
        """交易后更新状态"""
        self._check_daily_reset(current_balance)

        with self.lock:
            # 如果daily_start_balance未初始化，使用当前余额减去盈亏作为初始余额
            if self.state.daily_start_balance <= 0:
                self.state.daily_start_balance = current_balance - pnl

            # 更新日盈亏
            self.state.daily_pnl += pnl
            self.state.last_trade_pnl = pnl
            self.state.last_trade_time = time.time()
            self.state.total_trades += 1

            # 更新连续胜/负
            if pnl > 0:
                self.state.consecutive_wins += 1
                self.state.consecutive_losses = 0
                self.state.winning_trades += 1
            else:
                self.state.consecutive_losses += 1
                self.state.consecutive_wins = 0

            # 更新回撤
            if current_balance > self.state.peak_balance:
                self.state.peak_balance = current_balance

            if self.state.peak_balance > 0:
                drawdown = (self.state.peak_balance - current_balance) / self.state.peak_balance * 100
                self.state.current_drawdown = drawdown
                self.state.max_drawdown = max(self.state.max_drawdown, drawdown)

            # 检查是否需要紧急熔断
            self._check_emergency_stop(current_balance)
    
    def _check_emergency_stop(self, current_balance: float):
        """检查是否需要紧急熔断"""
        # 每次从settings读取，支持热重载
        emergency_stop_enabled = getattr(self.settings, 'EMERGENCY_STOP_ENABLED', True)
        if not emergency_stop_enabled:
            return
        
        # 日亏损超过限制的150%
        if self.state.daily_start_balance > 0:
            daily_pnl_pct = (current_balance - self.state.daily_start_balance) / self.state.daily_start_balance * 100
            if daily_pnl_pct < -self.max_daily_loss_pct * 1.5:
                self.state.emergency_stop = True
                self.state.emergency_stop_reason = f"日亏损严重: {daily_pnl_pct:.2f}%"
        
        # 回撤过大
        if self.state.current_drawdown > 50:
            self.state.emergency_stop = True
            self.state.emergency_stop_reason = f"回撤过大: {self.state.current_drawdown:.2f}%"
    
    def update_price(self, symbol: str, price: float, timestamp: float = None):
        """更新价格（用于黑天鹅检测）"""
        timestamp = timestamp or time.time()
        
        with self._price_history_lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            
            # 只保留最近60秒的数据
            cutoff = timestamp - 60
            self._price_history[symbol] = [
                (t, p) for t, p in self._price_history[symbol] if t > cutoff
            ]
            
            self._price_history[symbol].append((timestamp, price))
    
    def check_black_swan(self, symbol: str, current_price: float) -> tuple:
        """
        检查黑天鹅事件
        
        Returns:
            (is_black_swan: bool, level: int, drop_pct: float)
        """
        with self._price_history_lock:
            if symbol not in self._price_history:
                return False, 0, 0.0
            
            history = self._price_history[symbol]
            if len(history) < 2:
                return False, 0, 0.0
            
            # 找到60秒前的价格
            current_time = time.time()
            cutoff = current_time - 60
            
            old_prices = [p for t, p in history if t < cutoff]
            if not old_prices:
                return False, 0, 0.0
            
            old_price = old_prices[-1]  # 最接近60秒前的价格
            # Bug #25修复: 检查old_price是否为0避免除零
            if old_price <= 0:
                return False, 0, 0.0
            drop_pct = (current_price - old_price) / old_price * 100
            
            # 检查各级阈值
            if drop_pct <= self.black_swan_thresholds['drop_3']:
                return True, 3, drop_pct
            elif drop_pct <= self.black_swan_thresholds['drop_2']:
                return True, 2, drop_pct
            elif drop_pct <= self.black_swan_thresholds['drop_1']:
                return True, 1, drop_pct
            
            return False, 0, drop_pct
    
    def get_status(self) -> Dict:
        """获取风控状态"""
        with self.lock:
            return {
                'daily_pnl': self.state.daily_pnl,
                'daily_pnl_pct': (self.state.daily_pnl / self.state.daily_start_balance * 100) if self.state.daily_start_balance > 0 else 0,
                'consecutive_losses': self.state.consecutive_losses,
                'consecutive_wins': self.state.consecutive_wins,
                'total_trades': self.state.total_trades,
                'win_rate': (self.state.winning_trades / self.state.total_trades * 100) if self.state.total_trades > 0 else 0,
                'max_drawdown': self.state.max_drawdown,
                'current_drawdown': self.state.current_drawdown,
                'emergency_stop': self.state.emergency_stop,
                'emergency_stop_reason': self.state.emergency_stop_reason,
            }
    
    def reset_emergency_stop(self):
        """重置紧急熔断状态"""
        with self.lock:
            self.state.emergency_stop = False
            self.state.emergency_stop_reason = ""
    
    def should_reduce_position(self, current_pnl_pct: float) -> tuple:
        """
        检查是否需要减仓
        
        Returns:
            (should_reduce: bool, reason: str, reduce_ratio: float)
        """
        with self.lock:
            # 连续亏损达到限制后，降低仓位
            if self.state.consecutive_losses >= self.max_consecutive_losses:
                return True, "连续亏损减仓", 0.5
            
            # 当前亏损接近日限制
            if self.state.daily_start_balance > 0:
                daily_pnl_pct = self.state.daily_pnl / self.state.daily_start_balance * 100
                if daily_pnl_pct < -self.max_daily_loss_pct * 0.7:
                    return True, "接近日亏损限制", 0.5
            
            return False, "", 1.0


class PositionSizer:
    """
    仓位计算器
    
    根据账户余额、风险参数计算合适的仓位大小
    """
    
    def __init__(self):
        from framework.core.config import get_main_config
        self.settings = _get_settings()  # 使用延迟初始化的settings实例
    
    def calculate_position_size(self, 
                                account_balance: float,
                                entry_price: float,
                                stop_loss_price: float,
                                leverage: int,
                                risk_per_trade_pct: float = 2.0,
                                max_position_pct: float = 30.0) -> Dict:
        """
        计算仓位大小
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            stop_loss_price: 止损价格
            leverage: 杠杆倍数
            risk_per_trade_pct: 单笔风险比例（默认2%）
            max_position_pct: 最大仓位比例（默认30%）
        
        Returns:
            {
                'quantity': 下单数量,
                'margin': 保证金,
                'position_value': 仓位价值,
                'risk_amount': 风险金额,
                'risk_pct': 风险比例
            }
        """
        # 计算止损距离百分比
        if entry_price <= 0:
            return {'quantity': 0, 'margin': 0, 'error': '入场价格无效'}
        
        stop_loss_pct = abs(entry_price - stop_loss_price) / entry_price * 100
        
        if stop_loss_pct < 0.5:
            return {'quantity': 0, 'margin': 0, 'error': '止损距离过近'}
        
        # 基于风险的仓位计算
        # 风险金额 = 账户余额 * 风险比例
        risk_amount = account_balance * (risk_per_trade_pct / 100)
        
        # 止损时的亏损 = 仓位价值 * 杠杆 * 止损百分比
        # risk_amount = position_value * leverage * (stop_loss_pct / 100)
        # position_value = risk_amount / (leverage * stop_loss_pct / 100)
        
        if stop_loss_pct * leverage > 100:
            return {'quantity': 0, 'margin': 0, 'error': '止损距离或杠杆过大'}
        
        position_value = risk_amount / (stop_loss_pct / 100)
        
        # 保证金 = 仓位价值 / 杠杆
        margin = position_value / leverage
        
        # 检查最大仓位限制
        max_margin = account_balance * (max_position_pct / 100)
        if margin > max_margin:
            margin = max_margin
            position_value = margin * leverage
            # 正确公式：风险金额 = 仓位价值 * 止损比例
            risk_amount = position_value * (stop_loss_pct / 100)
        
        # 计算数量
        quantity = position_value / entry_price
        
        return {
            'quantity': quantity,
            'margin': margin,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_pct': risk_amount / account_balance * 100 if account_balance > 0 else 0,
            'stop_loss_pct': stop_loss_pct
        }
    
    def calculate_leverage_for_risk(self,
                                     entry_price: float,
                                     stop_loss_price: float,
                                     target_risk_pct: float = 50.0,
                                     max_leverage: int = 20) -> int:
        """
        根据风险目标计算合适的杠杆

        Args:
            entry_price: 入场价格
            stop_loss_price: 止损价格
            target_risk_pct: 目标风险百分比（触发止损时的亏损）
            max_leverage: 最大杠杆

        Returns:
            建议杠杆倍数
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return max_leverage // 2  # 返回保守值

        stop_loss_pct = abs(entry_price - stop_loss_price) / entry_price * 100

        if stop_loss_pct <= 0:
            return max_leverage // 2

        # 目标风险 = 杠杆 * 止损百分比
        # leverage = target_risk_pct / stop_loss_pct
        # 添加边界检查，防止中间值过大
        if stop_loss_pct < 0.1:  # 止损百分比小于0.1%时，使用保守杠杆
            return max_leverage // 2

        leverage = target_risk_pct / stop_loss_pct

        # 限制范围
        leverage = max(1, min(int(leverage), max_leverage))

        return leverage
