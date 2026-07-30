"""
止盈管理模块 (15mTupo)

继承 BaseTakeProfitManager，添加 15mTupo 特有逻辑:
- 阶梯回撤止盈（按30/40/30比例部分平仓）
- coin lock 释放
"""

from typing import Dict, Optional

from strategies.base.take_profit_manager import BaseTakeProfitManager, logger
from framework.shared.strategy_dispatcher import get_strategy_dispatcher, StrategyType, StrategyDispatcher

# ── 命名常量 ──────────────────────────────────────────────
# 阶梯止盈参数
LADDER_RATIOS = (0.30, 0.40, 0.30)          # 3级阶梯平仓比例
REBOUND_LADDER_DD = (10, 15, 20)            # 反弹信号回撤阈值 (%)
TREND_LADDER_DD = (20, 30, 40)              # 趋势信号回撤阈值 (%)
REBOUND_MIN_PEAK = 10                       # 反弹信号最低峰值 (%)
TREND_MIN_PEAK = 20                         # 趋势信号最低峰值 (%)
LADDER_LEVELS = 3                           # 阶梯止盈级数
PNL_HISTORY_MAX = 10                        # PNL 历史记录上限
MIN_REMAINING_PCT = 0.01                    # 剩余仓位最小比例
# ──────────────────────────────────────────────────────────


class TakeProfitManager(BaseTakeProfitManager):
    """
    15mTupo 止盈管理器

    继承基类止盈逻辑，添加阶梯止盈和 coin lock 释放
    """

    def check_take_profit(self, symbol: str, position_info: Dict, current_price: float, pnl_rate: float) -> Optional[str]:
        """
        检查并执行止盈逻辑（含阶梯回撤止盈30/40/30）
        """
        try:
            if position_info.get('_take_profit_order_monitoring', False):
                logger.debug(f"{symbol} 已有待处理的止盈订单，跳过检查")
                return None

            # === 0. 阶梯回撤止盈（按30/40/30比例部分平仓）===
            max_profit = position_info.get('max_profit_pct', 0)
            sig_type = position_info.get('signal_type', '')

            # 初始化追踪变量
            if '_ladder_count' not in position_info:
                position_info['_ladder_count'] = 0
                position_info['_trailing_stop_pnl'] = 0.0
                position_info['_has_reached_tp'] = False
                position_info['_pnl_history'] = []

            # 更新PNL历史
            if '_pnl_history' not in position_info:
                position_info['_pnl_history'] = []
            position_info['_pnl_history'].append(pnl_rate)
            if len(position_info['_pnl_history']) > PNL_HISTORY_MAX:
                position_info['_pnl_history'].pop(0)

            # 更新追踪峰值
            if pnl_rate > position_info['_trailing_stop_pnl']:
                position_info['_trailing_stop_pnl'] = pnl_rate

            # 判断阈值类型
            is_rebound_tri = "REBOUND" in sig_type or "TRIANGLE" in sig_type
            min_peak = REBOUND_MIN_PEAK if is_rebound_tri else TREND_MIN_PEAK
            dd_thresholds = REBOUND_LADDER_DD if is_rebound_tri else TREND_LADDER_DD

            if position_info['_ladder_count'] < LADDER_LEVELS and position_info['_trailing_stop_pnl'] >= min_peak:
                drawdown = position_info['_trailing_stop_pnl'] - pnl_rate
                base_dd = dd_thresholds[position_info['_ladder_count']]

                if drawdown >= base_dd:
                    # 计算平仓比例：本步应平 ratio_this 占总仓位的比例
                    # 若之前有部分平仓(其他TP)，按当前剩余等比例缩放
                    step = position_info['_ladder_count']
                    ratio_this = LADDER_RATIOS[step]
                    init_qty = position_info.get('initial_quantity', position_info.get('total_quantity', 0))
                    cur_qty = position_info.get('total_quantity', 0)
                    remaining_pct = cur_qty / init_qty if init_qty > 0 else 1.0
                    cum_closed = sum(LADDER_RATIOS[:step])
                    target_remaining = 1.0 - cum_closed - ratio_this
                    close_pct = max(0, min((remaining_pct - target_remaining) / max(remaining_pct, MIN_REMAINING_PCT) * 100, 100))

                    logger.info(f"{symbol} 触发阶梯回撤止盈: 第{step+1}级, "
                               f"峰值={position_info['_trailing_stop_pnl']:.1f}%, 当前={pnl_rate:.1f}%, "
                               f"平仓比例={close_pct:.0f}%")

                    self.cancel_trailing_take_profit(symbol, position_info)
                    self.execute_close_position(symbol, position_info, close_pct, f"阶梯止盈-第{step+1}级", 'LIMIT')

                    position_info['_ladder_count'] = step + 1
                    position_info['_has_reached_tp'] = True
                    return f"ladder_tp_level{position_info['_ladder_count']}"

            # 1. 高盈利移动止盈
            result = self._check_take_profit_high(symbol, position_info, current_price, pnl_rate)
            if result:
                return result

            # 2. 低盈利移动止盈
            result = self._check_take_profit_low(symbol, position_info, current_price, pnl_rate)
            if result:
                return result

            # 3. 盈亏平衡止盈
            result = self._check_breakeven(symbol, position_info, current_price, pnl_rate)
            if result:
                return result

            return None

        except Exception as e:
            logger.error(f"止盈检查失败 {symbol}: {str(e)}", exc_info=True)
            return None

    def _on_close_complete(self, symbol: str, position_info: Dict) -> None:
        """15mTupo 特有: 平仓完成后释放 coin lock"""
        strategy_name = position_info.get('strategy', '15mTupo')
        strategy_type = StrategyDispatcher.resolve_strategy(strategy_name)
        dispatcher = get_strategy_dispatcher()
        dispatcher.release_coin(symbol, strategy_type)


__all__ = ['TakeProfitManager']
