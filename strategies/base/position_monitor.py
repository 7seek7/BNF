"""
持仓监控模块 - 基类 (Position Monitor Base)

提供 HF 和 15mTupo 共享的监控循环框架。
子类可覆盖:
- _on_position_update(): 持仓更新后额外逻辑（如 MAE 追踪）
"""

import time
import threading
from typing import Dict, Optional, Any
from framework.core.config import get_main_config
from config.settings import Settings
from utils.logger import Logger

logger = Logger.get_logger('position_monitor')
settings = Settings()
main_config = get_main_config()


class BasePositionMonitor:
    """
    持仓监控管理器基类

    负责监控所有持仓状态，检测各种触发条件
    """

    def __init__(self, trader):
        self.trader = trader
        self.client = trader.client
        self.positions = trader.positions
        self.position_manager = trader.position_manager
        self.take_profit = trader.take_profit
        self.stop_loss_manager = trader.stop_loss_manager
        self.position_builder = trader.position_builder

        self._running = False
        self._monitor_thread = None
        self._monitor_count = 0
        self._last_position_sync = 0
        self._positions_lock = threading.RLock()

        logger.info("持仓监控管理器已初始化")

    def start(self):
        if self._running:
            logger.warning("监控线程已在运行中")
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("持仓监控线程已启动")

    def stop(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("持仓监控线程已停止")

    def _monitor_loop(self):
        while self._running:
            try:
                self.monitor_positions()
                monitor_sleep_time = settings.POSITION_MONITOR_SLEEP_TIME
                time.sleep(monitor_sleep_time)
                logger.debug(f"持仓监控刷新间隔: {monitor_sleep_time}秒")
            except Exception as e:
                logger.error(f"监控循环异常: {str(e)}", exc_info=True)
                time.sleep(10)

    def monitor_positions(self):
        try:
            current_time = time.time()
            with self._positions_lock:
                current_count = len(self.positions)
            self._monitor_count += 1

            if current_count != getattr(self, '_last_position_count', 0) or self._monitor_count % 10 == 1:
                logger.info(f"开始监控持仓，当前持仓数量: {current_count}")
                self._last_position_count = current_count

            if current_time - self._last_position_sync > settings.POSITION_SYNC_INTERVAL:
                self._last_position_sync = current_time
            else:
                logger.debug(f"跳过持仓同步（距离上次同步不足{settings.POSITION_SYNC_INTERVAL}秒）")

            if current_count > main_config.max_positions:
                logger.error(f"⚠️  持仓数量严重超限! 当前{current_count}个，限制{main_config.max_positions}个")
                logger.error(f"当前持仓: {list(self.positions.keys())}")
                if self.trader.telegram_bot:
                    alert_data = {
                        'symbol': 'SYSTEM',
                        'direction': 'WARNING',
                        'message': f'持仓数量超限: {current_count}/{main_config.max_positions}',
                        'positions': list(self.positions.keys())
                    }
                    try:
                        self.trader.telegram_bot.send_alert('SYSTEM', alert_data)
                    except Exception as e:
                        logger.warning(f"发送Telegram告警失败: {e}")
            elif current_count >= main_config.max_positions * 0.9:
                logger.warning(f"⚠️  持仓数量接近上限: {current_count}/{main_config.max_positions}")

            self.trader._balance_cache = None
            all_positions = self.position_manager.get_all_positions()

            with self._positions_lock:
                positions_snapshot = dict(self.positions)
            for symbol, position_info in positions_snapshot.items():
                try:
                    logger.debug(f"{symbol} 监控中...")

                    if position_info.get('status') == 'closed':
                        logger.debug(f"{symbol} 持仓已关闭，跳过")
                        continue

                    current_price = self.client.get_ticker_price(symbol)
                    position_info['current_price'] = current_price

                    entry_price = position_info['entry_price']
                    direction = position_info['direction']
                    total_quantity = position_info['total_quantity']
                    total_investment = position_info['total_investment']
                    leverage = position_info['leverage']

                    if direction == 'LONG':
                        pnl_rate = ((current_price - entry_price) / entry_price) * leverage * 100
                    else:
                        pnl_rate = ((entry_price - current_price) / entry_price) * leverage * 100

                    profit = total_investment * pnl_rate / 100
                    position_info['profit_pct'] = pnl_rate
                    position_info['profit'] = profit

                    if pnl_rate > position_info.get('max_profit_pct', 0):
                        position_info['max_profit_pct'] = pnl_rate

                    # 子类可覆盖: 如 MAE 追踪
                    self._on_position_update(symbol, position_info, pnl_rate)

                    if (abs(pnl_rate) > 5.0 or
                        self._monitor_count % 20 == 0 or
                        position_info.get('_last_logged_pnl', 0) != pnl_rate):
                        logger.info(f"{symbol} 盈亏率={pnl_rate:.2f}%, 价格={current_price:.6f}, 数量={total_quantity:.6f}")
                        position_info['_last_logged_pnl'] = pnl_rate

                    position_complete = position_info.get('position_complete', False)
                    max_profit = position_info.get('max_profit_pct', 0)
                    logger.info(f"{symbol} 止盈检查前状态: position_complete={position_complete}, max_profit={max_profit:.2f}%, current={pnl_rate:.2f}%")

                    tp_action = self.take_profit.check_take_profit(symbol, position_info, current_price, pnl_rate)
                    if tp_action:
                        logger.info(f"{symbol} 执行止盈操作: {tp_action}")

                    if self.take_profit.check_reenter_opportunity(symbol, position_info, current_price):
                        self.take_profit.reenter_position(symbol, position_info)

                    self.take_profit.monitor_take_profit_limit_order(symbol, position_info, current_price, pnl_rate)

                except Exception as e:
                    logger.error(f"{symbol} 监控持仓失败: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"监控循环失败: {str(e)}")

    def _on_position_update(self, symbol: str, position_info: Dict, pnl_rate: float):
        """子类覆盖: 持仓更新后额外逻辑（如 MAE 追踪）"""
        pass

_position_monitor_instance = None

def get_position_monitor(trader) -> BasePositionMonitor:
    global _position_monitor_instance
    if _position_monitor_instance is None:
        _position_monitor_instance = BasePositionMonitor(trader)
    return _position_monitor_instance
