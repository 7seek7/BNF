"""
建仓策略模块 (15mTupo)

继承 BasePositionBuilder，添加 15mTupo 特有逻辑:
- close_position 释放 coin lock
"""

import threading
from typing import Dict, TYPE_CHECKING

from strategies.base.position_builder import BasePositionBuilder, logger
from framework.shared.strategy_dispatcher import get_strategy_dispatcher, StrategyType, StrategyDispatcher

if TYPE_CHECKING:
    from framework.business.exchange_client import ExchangeClient
    from framework.business.position_manager import PositionManager


class PositionBuilder(BasePositionBuilder):
    """
    15mTupo 建仓策略管理器

    继承基类仓位构建逻辑，添加 coin lock 释放
    """

    _close_lock = threading.Lock()

    def close_position(self, symbol: str, reason: str = "手动平仓", close_pct: float = 100.0) -> bool:
        """15mTupo 特有: 平仓后释放 coin lock"""
        with self._close_lock:
            if symbol not in self.positions:
                logger.warning(f"持仓 {symbol} 不存在，无法平仓")
                return False

            position_info = self.positions[symbol]
            logger.info(f"关闭持仓: {symbol}, 原因: {reason}, 比例: {close_pct}%")

            try:
                success = self._execute_close_position(symbol, position_info, close_pct, reason)
                if success:
                    strategy_name = position_info.get('strategy', '15mTupo')
                    strategy_type = StrategyDispatcher.resolve_strategy(strategy_name)
                    dispatcher = get_strategy_dispatcher()
                    dispatcher.release_coin(symbol, strategy_type)
                return success
            except Exception as e:
                logger.error(f"关闭持仓失败 {symbol}: {str(e)}", exc_info=True)
                return False


def get_position_builder(client: 'ExchangeClient', position_manager: 'PositionManager') -> PositionBuilder:
    return PositionBuilder(client, position_manager)
