from config.settings import settings
from utils.logger import Logger

logger = Logger.get_logger('risk_manager')

class RiskManager:
    """风险管理器"""
    
    def __init__(self, client):
        self.client = client
    
    def check_position_limit(self, current_positions):
        """检查持仓数量限制"""
        try:
            if len(current_positions) >= settings.MAX_POSITIONS:
                logger.warning(f"已达持仓上限: {len(current_positions)}/{settings.MAX_POSITIONS}")
                return False
            return True
        except Exception as e:
            logger.error(f"检查持仓限制失败: {str(e)}")
            return False
    
    def calculate_risk_exposure(self):
        """计算风险敞口"""
        try:
            balance_info = self.client.get_account_balance()
            
            total_balance = balance_info['total_balance']
            total_margin = balance_info['total_margin']
            unrealized_pnl = balance_info['unrealized_pnl']
            
            # 风险敞口比例
            if total_balance > 0:
                exposure_ratio = (total_margin / total_balance) * 100
            else:
                exposure_ratio = 0
            
            return {
                'total_balance': total_balance,
                'total_margin': total_margin,
                'unrealized_pnl': unrealized_pnl,
                'exposure_ratio': exposure_ratio
            }
            
        except Exception as e:
            logger.error(f"计算风险敞口失败: {str(e)}")
            return None
    
    def should_stop_loss(self, profit_rate, entry_price, current_price):
        """判断是否应该止损"""
        try:
            # 第一级止损
            if profit_rate <= settings.STOPLOSS_TRIGGER1:
                logger.warning(f"触发第一级止损: 收益率 {profit_rate:.2f}%")
                return True, 'LEVEL1'
            
            # 第二级止损
            if profit_rate <= settings.STOPLOSS_TRIGGER2:
                logger.warning(f"触发第二级止损: 收益率 {profit_rate:.2f}%")
                return True, 'LEVEL2'
            
            # 第三级止损（紧急）
            if profit_rate <= settings.STOPLOSS_TRIGGER3:
                logger.error(f"触发紧急止损: 收益率 {profit_rate:.2f}%")
                return True, 'LEVEL3'
            
            return False, None
            
        except Exception as e:
            logger.error(f"判断止损失败: {str(e)}")
            return False, None
    
    def should_take_profit(self, profit_rate, max_profit_rate):
        """判断是否应该止盈"""
        try:
            # 计算利润回撤
            profit_drawback = max_profit_rate - profit_rate
            
            # 高盈利止盈
            if max_profit_rate >= settings.HIGH_PROFIT_THRESHOLD:
                if profit_drawback >= settings.HIGH_PROFIT_DRAWBACK2:
                    logger.info(f"触发高盈利止盈2: 回撤 {profit_drawback:.2f}%")
                    return True, 'HIGH2', settings.HIGH_PROFIT_CLOSE2
                elif profit_drawback >= settings.HIGH_PROFIT_DRAWBACK1:
                    logger.info(f"触发高盈利止盈1: 回撤 {profit_drawback:.2f}%")
                    return True, 'HIGH1', settings.HIGH_PROFIT_CLOSE1
            
            # 低盈利止盈
            if max_profit_rate >= settings.LOW_PROFIT_THRESHOLD:
                if profit_drawback >= settings.LOW_PROFIT_DRAWBACK1:
                    logger.info(f"触发低盈利止盈: 回撤 {profit_drawback:.2f}%")
                    return True, 'LOW', settings.LOW_PROFIT_CLOSE1
            
            # 保本止盈
            if profit_rate <= settings.BREAKEVEN_THRESHOLD and max_profit_rate > settings.BREAKEVEN_THRESHOLD:
                logger.info(f"触发保本止盈: 利润率 {profit_rate:.2f}%")
                return True, 'BREAKEVEN', 100
            
            return False, None, 0
            
        except Exception as e:
            logger.error(f"判断止盈失败: {str(e)}")
            return False, None, 0
    
    def reevaluate_risk_after_close(self, profit_rate, close_type):
        """
        止盈止损后的风险重新评估
        
        Args:
            profit_rate: 本次交易收益率
            close_type: 平仓类型 ('TAKE_PROFIT', 'STOP_LOSS', 'BREAKEVEN')
        
        Returns:
            risk_adjustment: 风险调整建议
                {
                    'should_reduce_positions': bool,  # 是否减少持仓数
                    'should_reduce_leverage': bool,    # 是否降低杠杆
                    'should_pause_trading': bool,       # 是否暂停交易
                    'reason': str                       # 调整原因
                }
        """
        try:
            risk_adjustment = {
                'should_reduce_positions': False,
                'should_reduce_leverage': False,
                'should_pause_trading': False,
                'reason': '风险正常'
            }
            
            # 获取当前风险敞口
            exposure = self.calculate_risk_exposure()
            if not exposure:
                return risk_adjustment
            
            current_exposure = exposure['exposure_ratio']
            total_balance = exposure['total_balance']
            
            # 止损后的风险重新评估
            if close_type == 'STOP_LOSS':
                # 如果亏损超过10%，建议暂停交易
                if profit_rate <= -10.0:
                    risk_adjustment['should_pause_trading'] = True
                    risk_adjustment['reason'] = f'大额止损({profit_rate:.2f}%)，建议暂停交易'
                    logger.warning(f"止损失败{profit_rate:.2f}%，建议暂停交易")
                
                # 如果亏损超过5%，减少持仓数
                elif profit_rate <= -5.0:
                    risk_adjustment['should_reduce_positions'] = True
                    risk_adjustment['reason'] = f'中等止损失({profit_rate:.2f}%)，建议减少持仓'
                    logger.warning(f"止损失败{profit_rate:.2f}%，建议减少持仓数")
                
                # 如果风险敞口超过80%，降低杠杆
                elif current_exposure > 80.0:
                    risk_adjustment['should_reduce_leverage'] = True
                    risk_adjustment['reason'] = f'风险敞口过高({current_exposure:.1f}%)，建议降低杠杆'
                    logger.warning(f"风险敞口{current_exposure:.1f}%，建议降低杠杆")
            
            # 止盈后的风险重新评估
            elif close_type == 'TAKE_PROFIT':
                # 如果盈利超过20%，可以适当增加风险（已通过连续亏损计数重置实现）
                if profit_rate >= 20.0:
                    logger.info(f"大额盈利{profit_rate:.2f}%，风险可控")
                # 如果盈利超过10%，风险正常
                elif profit_rate >= 10.0:
                    logger.info(f"盈利{profit_rate:.2f}%，风险正常")
            
            # 保本平仓后的风险重新评估
            elif close_type == 'BREAKEVEN':
                # 保本平仓说明市场波动大，建议减少持仓
                if current_exposure > 70.0:
                    risk_adjustment['should_reduce_positions'] = True
                    risk_adjustment['reason'] = f'保本平仓且风险敞口较高({current_exposure:.1f}%)，建议减少持仓'
                    logger.warning(f"保本平仓且风险敞口{current_exposure:.1f}%，建议减少持仓")
            
            logger.info(f"风险重新评估完成: {risk_adjustment}")
            return risk_adjustment
            
        except Exception as e:
            logger.error(f"风险重新评估失败: {str(e)}")
            return {
                'should_reduce_positions': False,
                'should_reduce_leverage': False,
                'should_pause_trading': False,
                'reason': '评估失败'
            }
    
    def apply_risk_adjustment(self, risk_adjustment):
        """
        应用风险调整建议
        
        Args:
            risk_adjustment: 风险调整建议
        """
        try:
            if risk_adjustment.get('should_pause_trading'):
                logger.warning(f"风险调整建议: {risk_adjustment['reason']}")
                # 具体暂停逻辑由调用方实现
            
            if risk_adjustment.get('should_reduce_positions'):
                logger.warning(f"风险调整建议: {risk_adjustment['reason']}")
                # 具体减少持仓逻辑由调用方实现
            
            if risk_adjustment.get('should_reduce_leverage'):
                logger.warning(f"风险调整建议: {risk_adjustment['reason']}")
                # 具体降低杠杆逻辑由调用方实现
            
        except Exception as e:
            logger.error(f"应用风险调整失败: {str(e)}")