"""
持仓监控模块 (15mTupo)

继承 BasePositionMonitor，15mTupo 无需额外逻辑。
"""

from strategies.base.position_monitor import BasePositionMonitor, get_position_monitor

PositionMonitor = BasePositionMonitor

__all__ = ['PositionMonitor', 'get_position_monitor']
