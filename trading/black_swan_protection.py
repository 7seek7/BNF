import time
from collections import defaultdict
from config.settings import settings
from utils.logger import Logger

logger = Logger.get_logger('black_swan')

class BlackSwanDetector:
    """黑天鹅事件检测器"""
    
    def __init__(self):
        self.price_history = defaultdict(list)
        self.emergency_mode = False
        self.market_sentiment = 'normal'
    
    def check_price_crash(self, symbol, current_price):
        """
        检测单币种价格暴跌（黑天鹅）
        
        Args:
            symbol: 币种
            current_price: 当前价格
        
        Returns:
            (is_crash, crash_type, drop_pct): 是否崩溃、崩溃类型、跌幅
        """
        try:
            current_time = time.time()
            if symbol not in self.price_history:
                self.price_history[symbol].append((current_time, current_price))
                return False, None, 0
            
            # 获取最近60秒的价格
            history = self.price_history[symbol]
            cutoff_time = current_time - 60
            recent_prices = [(t, p) for t, p in history if t > cutoff_time]
            
            if len(recent_prices) < 2:
                self.price_history[symbol].append((current_time, current_price))
                return False, None, 0
            
            # 计算最大跌幅 - O(n)优化算法
            # 只需找到60秒内最高点和最低点，计算从高点到低点的最大跌幅
            max_price = max(p[1] for p in recent_prices)
            min_price = min(p[1] for p in recent_prices)

            # 找到最高点和最低点的时间
            max_price_time = next(p[0] for p in recent_prices if p[1] == max_price)
            min_price_time = next(p[0] for p in recent_prices if p[1] == min_price)

            # 计算最大跌幅（从高点到低点）
            max_drop_pct = 0
            if max_price > 0 and min_price_time > max_price_time:
                max_drop_pct = ((min_price - max_price) / max_price) * 100
            
            # 检测暴跌
            if max_drop_pct <= -20.0:  # 60秒内下跌20%
                logger.critical(f"{symbol} 黑天鹅事件: 60秒内下跌{abs(max_drop_pct):.1f}%")
                self.price_history[symbol].append((current_time, current_price))
                return True, 'BLACK_SWAN_DROP_20PCT', max_drop_pct
            
            elif max_drop_pct <= -15.0:  # 60秒内下跌15%
                logger.critical(f"{symbol} 黑天鹅事件: 60秒内下跌{abs(max_drop_pct):.1f}%")
                self.price_history[symbol].append((current_time, current_price))
                return True, 'BLACK_SWAN_DROP_15PCT', max_drop_pct
            
            elif max_drop_pct <= -10.0:  # 60秒内下跌10%
                logger.warning(f"{symbol} 价格快速下跌: 60秒内下跌{abs(max_drop_pct):.1f}%")
                self.price_history[symbol].append((current_time, current_price))
                return False, 'FAST_DROP_10PCT', max_drop_pct
            
            # 正常情况，更新历史
            self.price_history[symbol].append((current_time, current_price))
            # 保留最近100条
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            return False, None, 0
            
        except Exception as e:
            logger.error(f"检测价格暴跌失败 {symbol}: {str(e)}")
            return False, None, 0
    
    def check_market_crash(self, price_changes):
        """
        检测全市场崩溃
        
        Args:
            price_changes: {symbol: price_change_pct} 各币种价格变化
        
        Returns:
            is_crash: 是否市场崩溃
        """
        try:
            if not price_changes:
                return False
            
            # 计算下跌币种比例
            crash_count = sum(1 for change in price_changes.values() if change <= -5.0)
            total_count = len(price_changes)
            crash_ratio = crash_count / total_count if total_count > 0 else 0
            
            # 如果超过50%的币种下跌5%以上，触发市场崩溃
            if crash_ratio >= 0.5:
                logger.critical(f"全市场崩溃: {crash_count}/{total_count} 币种下跌超过5%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"检测全市场崩溃失败: {str(e)}")
            return False


class EmergencyCircuitBreaker:
    """紧急熔断器"""

    def __init__(self, initial_balance):
        self.initial_balance = initial_balance
        self.daily_loss = 0.0
        self.continuous_loss_count = 0
        self.max_drawdown = 0.0
        self.is_paused = False
        self.pause_reason = None
        self.emergency_mode = False
        self.paused_at = None
        self.last_check_time = time.time()

        # 新增：单日亏损警告冷却
        self.daily_loss_warning_cooldown = {}  # {symbol: last_warning_time}
        self.daily_loss_warning_interval = 600  # 10分钟冷却（秒）
    
    def check_daily_loss(self, current_balance):
        """
        检查单日亏损

        Args:
            current_balance: 当前余额

        Returns:
            should_pause: 是否应该暂停
        """
        try:
            self.last_check_time = time.time()

            # 计算单日亏损
            loss_pct = ((current_balance - self.initial_balance) / self.initial_balance) * 100
            self.daily_loss = loss_pct

            # 计算最大回撤
            if loss_pct < self.max_drawdown:
                self.max_drawdown = loss_pct

            # 检查冷却时间（每个币种独立冷却）
            global_symbol = "GLOBAL"  # 使用全局标记
            current_time = time.time()
            last_warning_time = self.daily_loss_warning_cooldown.get(global_symbol, 0)

            # 冷却时间内不重复警告
            cooldown_seconds = self.daily_loss_warning_interval
            if current_time - last_warning_time < cooldown_seconds:
                logger.debug(f"单日亏损警告冷却中，剩余{(cooldown_seconds - (current_time - last_warning_time)):.0f}秒")
                return False

            # 单日亏损超过10%，触发熔断
            if loss_pct <= -10.0:
                # 更新冷却时间
                self.daily_loss_warning_cooldown[global_symbol] = current_time

                self.trigger_emergency_pause(
                    f"单日亏损超过10%（当前{loss_pct:.2f}%），触发熔断"
                )
                return True

            # 单日亏损超过5%，发送警告（带冷却）
            elif loss_pct <= -5.0:
                # 更新冷却时间
                self.daily_loss_warning_cooldown[global_symbol] = current_time

                logger.warning(f"单日亏损警告: {loss_pct:.2f}%")

                # 发送Telegram警告
                self._send_warning_alert(f"⚠️ 单日亏损警告: {loss_pct:.2f}%")

            return False

        except Exception as e:
            logger.error(f"检查单日亏损失败: {str(e)}")
            return False
    
    def check_continuous_loss(self, profit_rate):
        """
        检查连续亏损
        
        Args:
            profit_rate: 本次交易收益率
        
        Returns:
            should_pause: 是否应该暂停
        """
        try:
            if profit_rate < 0:
                self.continuous_loss_count += 1
                logger.warning(f"连续亏损次数: {self.continuous_loss_count}")
                
                # 连续亏损3次，触发熔断
                if self.continuous_loss_count >= 3:
                    self.trigger_emergency_pause(
                        f"连续亏损{self.continuous_loss_count}次，触发熔断"
                    )
                    return True
            else:
                # 盈利，重置连续亏损计数
                if self.continuous_loss_count > 0:
                    logger.info(f"连续亏损结束，共{self.continuous_loss_count}次")
                self.continuous_loss_count = 0
            
            return False
            
        except Exception as e:
            logger.error(f"检查连续亏损失败: {str(e)}")
            return False
    
    def check_force_liquidation_risk(self, account_info):
        """
        检查强平风险
        
        Args:
            account_info: 账户信息
        
        Returns:
            should_pause: 是否应该暂停
        """
        try:
            total_wallet_balance = account_info.get('total_balance', 0) or 0
            if total_wallet_balance is None:
                total_wallet_balance = 0
            total_wallet_balance = float(total_wallet_balance)
            
            total_position_initial_margin = account_info.get('total_margin', 0) or 0
            if total_position_initial_margin is None:
                total_position_initial_margin = 0
            total_position_initial_margin = float(total_position_initial_margin)
            
            if total_position_initial_margin > 0:
                # 计算强平价格缓冲
                liquidation_buffer = (total_wallet_balance / total_position_initial_margin) - 1.0
                
                # 缓冲小于10%，立即强制平仓
                if liquidation_buffer < 0.1:
                    logger.critical(f"强平风险极高: 缓冲{liquidation_buffer*100:.1f}%")
                    self.trigger_emergency_pause(
                        f"强平风险极高（缓冲{liquidation_buffer*100:.1f}%），触发熔断"
                    )
                    return True
                
                # 缓冲小于30%，发送警告
                elif liquidation_buffer < 0.3:
                    logger.warning(f"强平风险警告: 缓冲{liquidation_buffer*100:.1f}%")
            else:
                # 无持仓时，无强平风险
                liquidation_buffer = 1.0
            
            return False
            
        except Exception as e:
            logger.error(f"检查强平风险失败: {str(e)}")
            return False
    
    def trigger_emergency_pause(self, reason):
        """触发紧急暂停"""
        logger.critical(f"触发紧急暂停: {reason}")
        self.is_paused = True
        self.pause_reason = reason
        self.emergency_mode = True
        self.paused_at = time.time()

        # 发送紧急告警到Telegram
        self._send_emergency_alert(f"🚨 紧急暂停: {reason}")
    
    def can_resume(self):
        """检查是否可以恢复交易"""
        if not self.is_paused:
            return True
        
        # 暂停后至少等待30分钟才能恢复
        if self.paused_at is not None and time.time() - self.paused_at < 1800:
            return False
        
        return True
    
    def resume(self):
        """恢复交易"""
        logger.info("恢复交易")
        self.is_paused = False
        self.pause_reason = None
        self.emergency_mode = False
        self.paused_at = None
        self.continuous_loss_count = 0

    def _send_warning_alert(self, message):
        """发送警告告警到Telegram"""
        try:
            from alert.telegram_bot import TelegramBot
            telegram = TelegramBot('alert')
            telegram.send_message(message)
            logger.info(f"警告告警已发送: {message}")
        except Exception as e:
            logger.error(f"发送警告告警失败: {str(e)}")

    def _send_emergency_alert(self, message):
        """发送紧急告警到Telegram"""
        try:
            from alert.telegram_bot import TelegramBot
            telegram = TelegramBot('trade')
            telegram.send_message(message)
            logger.info(f"紧急告警已发送: {message}")
        except Exception as e:
            logger.error(f"发送紧急告警失败: {str(e)}")


class EmergencyClose:
    """紧急平仓处理器"""
    
    def __init__(self, client):
        self.client = client
    
    def emergency_close_position(self, symbol, reason):
        """
        紧急平仓单个仓位
        
        Args:
            symbol: 币种
            reason: 平仓原因
        
        Returns:
            success: 是否成功
        """
        try:
            logger.critical(f"紧急平仓 {symbol}: {reason}")
            
            # 1. 取消所有挂单
            try:
                self.client.cancel_all_orders(symbol)
            except Exception as e:
                logger.warning(f"{symbol} 取消所有订单失败: {str(e)}")
            
            # 2. 获取持仓信息
            position = self.client.get_position(symbol)
            if not position:
                logger.warning(f"{symbol} 无持仓")
                return False
            
            position_amt = position['position_amt']
            if position_amt == 0:
                logger.warning(f"{symbol} 持仓量为0")
                return False
            
            # 3. 市价平仓
            side = 'SELL' if position_amt > 0 else 'BUY'
            quantity = abs(position_amt)
            
            # 使用市价单
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )
            
            logger.critical(f"{symbol} 紧急平仓成功: 原因={reason}, 订单={order.get('orderId')}")

            # 发送紧急告警到Telegram
            self._send_emergency_alert(f"🚨 紧急平仓 {symbol}: {reason}")

            return True
            
        except Exception as e:
            logger.critical(f"{symbol} 紧急平仓失败: {str(e)}")
            return False
    
    def emergency_close_all_positions(self, reason):
        """
        紧急平仓所有仓位
        
        Args:
            reason: 平仓原因
        
        Returns:
            success_count: 成功平仓数量
        """
        try:
            logger.critical(f"紧急平仓所有仓位: {reason}")
            
            # 获取所有持仓
            from trading.position_manager import PositionManager
            pos_manager = PositionManager(self.client)
            positions = pos_manager.get_all_positions()
            
            success_count = 0
            for position in positions:
                symbol = position['symbol']
                if self.emergency_close_position(symbol, reason):
                    success_count += 1
                time.sleep(0.1)  # 避免API限流
            
            logger.critical(f"紧急平仓完成: 成功{success_count}/{len(positions)}个仓位")
            
            return success_count
            
        except Exception as e:
            logger.critical(f"紧急平仓所有仓位失败: {str(e)}")
            return 0

    def _send_emergency_alert(self, message):
        """发送紧急告警到Telegram"""
        try:
            from alert.telegram_bot import TelegramBot
            telegram = TelegramBot('trade')
            telegram.send_message(message)
            logger.info(f"紧急告警已发送: {message}")
        except Exception as e:
            logger.error(f"发送紧急告警失败: {str(e)}")
