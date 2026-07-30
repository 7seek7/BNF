# -*- coding: utf-8 -*-
"""
V8.3 多币种风控 + 异常场景防护 (Section 11 & 12)
"""
import time
from typing import Dict, List, Optional, Tuple

EPS = 1e-12


class DailyLossManager:
    """日亏损管理 (Section 11.1)"""
    def __init__(self, warn_pct=3.0, hard_stop_pct=5.0):
        self.warn_pct = warn_pct
        self.hard_stop_pct = hard_stop_pct
        self.daily_pnl = 0.0
        self.current_date = ''
        self.is_restricted = False  # 预警状态
        self.is_halted = False      # 硬停状态
        self.halt_until = 0.0       # 硬停截止时间

    def reset_if_new_day(self, date_str: str):
        if date_str != self.current_date:
            self.daily_pnl = 0.0
            self.current_date = date_str
            self.is_restricted = False
            self.is_halted = False
            self.halt_until = 0.0

    def on_trade_result(self, pnl_pct: float, date_str: str):
        self.reset_if_new_day(date_str)
        self.daily_pnl += pnl_pct
        if self.daily_pnl <= -self.hard_stop_pct:
            self.is_halted = True
            self.halt_until = time.time() + 86400  # 当日剩余+次日0-8点
        elif self.daily_pnl <= -self.warn_pct:
            self.is_restricted = True

    def can_open_new_position(self) -> Tuple[bool, float]:
        """返回 (允许开仓, 风险系数)"""
        if self.is_halted:
            return False, 0.0
        if self.is_restricted:
            return True, 0.5  # 风险系数减半
        return True, 1.0


class CorrelationManager:
    """相关性管理 (Section 11.2)"""
    def __init__(self, max_correlated=3, ewma_halflife=10):
        self.max_correlated = max_correlated
        self.ewma_halflife = ewma_halflife
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}

    def can_open(self, symbol: str, direction: str, open_positions: List[Tuple[str, str]]) -> bool:
        """检查是否可开仓"""
        if direction == 'SHORT':
            return True  # 反向持仓完全豁免
        same_dir = [s for s, d in open_positions if d == 'LONG']
        return len(same_dir) < self.max_correlated


class ConcurrencyManager:
    """并发限制 (Section 11.3)"""
    def __init__(self, max_total=5, max_per_coin=2):
        self.max_total = max_total
        self.max_per_coin = max_per_coin

    def can_open(self, symbol: str, open_positions: List[Tuple[str, str]]) -> bool:
        if len(open_positions) >= self.max_total:
            return False
        coin_count = sum(1 for s, _ in open_positions if s == symbol)
        return coin_count < self.max_per_coin


class AnomalyDetector:
    """异常场景检测 (Section 12)"""
    def __init__(self):
        self.shock_mode = False
        self.shock_bar = 0
        self.shock_vsr_history: List[float] = []

    def check_alt_coin_spike(self, high, low, atr14, volume, avg_vol, idx):
        """山寨币暴击检测 (Section 12.1)"""
        if idx < 1:
            return False
        amp = (high - low) / max(high, EPS) * 100 if high > 0 else 0
        vol_ratio = volume / max(avg_vol, EPS) if avg_vol > 0 else 0
        return amp >= 4.0 * atr_pct and vol_ratio >= 5.0

    def check_btc_shock(self, vsr: float) -> bool:
        """BTC消息冲击检测 (Section 12.2)"""
        return vsr >= 2.5

    def check_data_spike(self, high, low, close, atr14, idx) -> bool:
        """数据插针检测 (Section 12.3)"""
        if idx < 1 or atr14 <= 0:
            return False
        amp = (high - low) / max(close, EPS) * 100
        return amp > 10 * (atr14 / max(close, EPS) * 100)

    def check_extreme_funding(self, funding_rate: float) -> Tuple[bool, str, str]:
        """极端资金费率检测 (Section 12.6)"""
        if funding_rate > 0.05:
            return True, 'LONG', '禁止开多'
        if funding_rate < -0.05:
            return True, 'SHORT', '禁止开空'
        return False, '', ''

    def is_weekend_low_liquidity(self, timestamp: int) -> bool:
        """周末低流动性检测 (Section 12.7)"""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp / 1000)
        return dt.weekday() >= 5  # Saturday=5, Sunday=6
