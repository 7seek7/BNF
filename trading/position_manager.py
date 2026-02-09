from utils.logger import Logger

logger = Logger.get_logger('position_manager')

class PositionManager:
    """仓位管理器"""
    
    def __init__(self, client):
        self.client = client
    
    def get_all_positions(self):
        """获取所有持仓"""
        try:
            # 在调用API前同步时间戳，避免-1021错误（网络延迟修复）
            if hasattr(self.client, '_sync_time'):
                try:
                    self.client._sync_time()
                except Exception as sync_error:
                    # 时间同步失败不应该阻塞持仓获取
                    logger.debug(f"时间同步失败，继续获取持仓: {str(sync_error)[:100]}")
            
            positions = self.client.client.futures_position_information()

            active_positions = []
            for pos in positions:
                try:
                    # 安全获取positionAmt
                    position_amt_raw = pos.get('positionAmt', 0)
                    if isinstance(position_amt_raw, dict):
                        logger.warning(f"持仓量数据格式异常: {type(position_amt_raw)} = {position_amt_raw}")
                        continue
                    
                    position_amt = float(position_amt_raw)
                    
                    # 修复：使用绝对值比较，避免精度问题
                    if abs(position_amt) > 0.000001:  # 使用很小的阈值而不是严格等于0
                        active_positions.append({
                            'symbol': pos.get('symbol', ''),
                            'position_amt': position_amt,
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'unrealized_pnl': float(pos.get('unRealizedProfit', 0)),
                            'leverage': int(pos.get('leverage', 1)),
                            'side': 'LONG' if position_amt > 0 else 'SHORT'
                        })
                except (ValueError, TypeError) as e:
                    logger.warning(f"解析持仓数据失败: {pos}, 错误: {str(e)}")
                    continue

            logger.debug(f"获取到 {len(active_positions)} 个活跃持仓")
            return active_positions

        except Exception as e:
            logger.error(f"获取持仓列表失败: {str(e)}")
            return []
    
    def calculate_profit_rate(self, position_amt, entry_price, current_price, side):
        """计算收益率"""
        try:
            if side == 'LONG':
                pnl_rate = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_rate = ((entry_price - current_price) / entry_price) * 100
            
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