from utils.logger import Logger
from framework.core.config import get_main_config
from utils.input_validator import safe_float, safe_int
import time
from typing import Optional

logger = Logger.get_logger('position_manager')

_settings_instance = None

def _get_settings():
    """延迟初始化 Settings"""
    global _settings_instance
    if _settings_instance is None:
        from config.settings import Settings
        _settings_instance = Settings()
    return _settings_instance


def determine_position_side(position_amt: float) -> str:
    """判断持仓方向（支持浮点数精度保护）
    
    Args:
        position_amt: 持仓数量，正数为多头，负数为空头
        
    Returns:
        'LONG' | 'SHORT' | 'NONE'
    """
    # 使用阈值判断，避免浮点数精度问题
    threshold = 0.000001  # 非常小的阈值
    if abs(position_amt) < threshold:
        return 'NONE'
    return 'LONG' if position_amt > 0 else 'SHORT'


class PositionManager:
    """仓位管理器"""

    def __init__(self, client):
        self.client = client
        # 修复：使用实例变量避免多实例共享问题（BUG-H014）
        self._last_time_sync = 0  # 实例变量，跟踪上次时间同步时间
    
    @property
    def _api_client(self):
        """获取底层API客户端（封装层访问）"""
        if hasattr(self.client, 'client'):
            return self.client.client
        return self.client
    
    def get_all_positions(self):
        """获取所有持仓"""
        try:
            # 定期同步时间戳，避免-1021错误（时间偏移错误）
            current_time = time.time()
            main_config = get_main_config()
            time_sync_interval = getattr(main_config, 'time_sync_interval', 300)  # 默认5分钟
            
            if current_time - self._last_time_sync > time_sync_interval:
                # 修复🟡-4: 使用_api_client确保正确调用
                if hasattr(self._api_client, '_sync_time'):
                    try:
                        self._api_client._sync_time()
                        self._last_time_sync = current_time
                    except Exception as sync_error:
                        logger.debug(f"时间同步失败，继续获取持仓: {str(sync_error)[:100]}")
            
            positions = self._api_client.futures_position_information()

            active_positions = []
            for pos in positions:
                try:
                    # 安全获取positionAmt
                    position_amt_raw = pos.get('positionAmt', 0)
                    if isinstance(position_amt_raw, dict):
                        logger.warning(f"持仓量数据格式异常: {type(position_amt_raw)} = {position_amt_raw}")
                        continue
                    
                    position_amt = safe_float(
                        position_amt_raw, default=0.0, context=f'{pos.get("symbol", "?")} position_amt')

                    # 修复：使用绝对值比较，避免精度问题
                    if abs(position_amt) > 0.000001:  # 使用很小的阈值而不是严格等于0
                        # 【关键修复】币安HEDGING模式可能返回不同的入场价字段名
                        # 依次尝试：entryPrice, avgPrice, breakEvenPrice
                        entry_price = 0
                        for field in ['entryPrice', 'avgPrice', 'breakEvenPrice']:
                            if field in pos and pos[field]:
                                entry_price = safe_float(pos[field], default=0.0,
                                                        context=f'{pos.get("symbol", "?")} {field}')
                                break

                        active_positions.append({
                            'symbol': pos.get('symbol', ''),
                            'position_amt': position_amt,
                            'entry_price': entry_price,
                            'unrealized_pnl': safe_float(pos.get('unRealizedProfit', 0),
                                                         default=0.0,
                                                         context=f'{pos.get("symbol", "?")} unrealized_pnl'),
                            'leverage': safe_int(pos.get('leverage', 1), default=1,
                                                context=f'{pos.get("symbol", "?")} leverage'),
                            # 使用防御性方向推断（兼容LONG/SHORT/买/卖等中英文）
                            'side': 'LONG' if position_amt > 0 else 'SHORT',
                            # HEDGING模式特有字段
                            'positionSide': pos.get('positionSide', 'BOTH')
                        })
                except (ValueError, TypeError) as e:
                    logger.warning(f"解析持仓数据失败: {pos}, 错误: {str(e)}")
                    continue

            logger.debug(f"获取到 {len(active_positions)} 个活跃持仓")
            return active_positions

        except Exception as e:
            logger.error(f"获取持仓列表失败: {str(e)}")
            return []
    
    def calculate_profit_rate(self, position_amt, entry_price, current_price, side, leverage=1):
        """计算收益率（含杠杆）"""
        try:
            # ✅ 修复 #2: 除零保护
            if entry_price is None or entry_price <= 0:
                logger.warning(f"计算收益率失败: 入场价格无效 (entry_price={entry_price})")
                return 0
            if current_price is None or current_price <= 0:
                logger.warning(f"计算收益率失败: 当前价格无效 (current_price={current_price})")
                return 0

            if side == 'LONG':
                pnl_rate = ((current_price - entry_price) / entry_price) * 100 * leverage
            else:
                pnl_rate = ((entry_price - current_price) / entry_price) * 100 * leverage

            return pnl_rate

        except Exception as e:
            logger.error(f"计算收益率失败: {str(e)}")
            return 0
    
    def get_position_value(self, symbol):
        """获取持仓价值"""
        try:
            pos_info = self.client.get_position(symbol)
            if not pos_info:
                return 0
            
            current_price = self.client.get_ticker_price(symbol)
            position_value = abs(pos_info['position_amt']) * current_price
            
            return position_value
            
        except Exception as e:
            logger.error(f"获取持仓价值失败: {str(e)}")
            return 0