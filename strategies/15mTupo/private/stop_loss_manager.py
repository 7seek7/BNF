"""
止损单管理器 (15mTupo)

继承 BaseStopLossOrderManager，15mTupo 无需额外逻辑。
"""

from strategies.base.stop_loss_manager import BaseStopLossOrderManager, get_stop_loss_manager

StopLossOrderManager = BaseStopLossOrderManager

__all__ = ['StopLossOrderManager', 'get_stop_loss_manager']
