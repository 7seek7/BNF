"""
新架构回测引擎 - 使用 TradeStrategy 的简化版本

职责：
- 使用统一的 TradeStrategy 进行回测
- 与实盘使用相同的交易逻辑
- 适配回测的特殊需求（历史数据、模拟执行）
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from backtest.data_downloader import DataDownloader
from trading.trade_strategy import TradeStrategy
from trading.backtest_executor import BacktestExecutor
from config.settings import settings
from utils.logger import Logger

logger = Logger.get_logger('unified_backtest')

class UnifiedBacktester:
    """
    统一回测引擎（使用 TradeStrategy）
    
    特点：
    - 与实盘使用完全相同的交易逻辑
    - 通过 BacktestExecutor 模拟订单执行
    - 简化实现，专注于核心逻辑验证
    """
    
    def __init__(self, offline=False):
        """
        初始化回测引擎
        
        Args:
            offline: 是否为离线模式，True=只使用本地缓存，False=尝试下载
        """
        # 回测引擎状态
        self.balance = 300.0  # 初始资金
        self.initial_balance = self.balance
        self.positions = {}  # 持仓（由 TradeStrategy 管理）
        self.trade_history = []  # 交易历史
        self.results = {'balance_history': [], 'timestamp_history': []}
        
        # 创建回测执行器
        self.executor = BacktestExecutor(self)
        
        # 创建交易策略（与实盘相同）
        self.strategy = TradeStrategy(self.executor)
        
        # 数据下载器（传递offline参数）
        self.downloader = DataDownloader(offline=offline)
        
        # 回测时间
        self.current_timestamp = None
        
    def get_total_position_value(self):
        """计算所有持仓的价值"""
        total_value = 0.0
        # 使用 TradeStrategy 的持仓（因为它管理实际的仓位）
        for pos in self.strategy.positions.values():
            if pos.get('status') == 'active':
                total_value += pos.get('total_investment', 0) + pos.get('profit', 0)
        return total_value
    
    def run_backtest(self, symbols=None, days=None, interval='1m', start_date=None, end_date=None, interactive=False):
        """
        运行回测

        Args:
            symbols: 币种列表
            days: 回测天数（如果不指定start_date和end_date，则使用天数计算）
            interval: K线周期
            start_date: 开始日期 (YYYY-MM-DD 格式)
            end_date: 结束日期 (YYYY-MM-DD 格式)
            interactive: 是否为交互模式（会等待用户输入）
        """
        try:
            # 计算日期范围
            if start_date and end_date:
                date_range_str = f"{start_date} ~ {end_date}"
                date_start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                date_end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                delta = date_end_dt - date_start_dt
                calculated_days = delta.days + 1
            else:
                if not days:
                    days = 180  # 默认180天
                date_range_str = f"{days} 天"
                date_start_dt = datetime.now() - timedelta(days=days - 1)
                date_end_dt = datetime.now()
                calculated_days = days
            
            print(f"\n开始回测:")
            print(f"  币种: {', '.join(symbols)}")
            if start_date:
                print(f"  时间范围: {date_range_str}")
            else:
                print(f"  天数: {days if days else calculated_days}")
            print(f"  K线周期: {interval}")
            print(f"  初始资金: {self.initial_balance:.2f} USDT")
            
            # 1. 下载数据
            print(f"\n下载历史数据...")
            all_data = {}
            for i, symbol in enumerate(symbols):
                try:
                    # 使用日期范围下载
                    if start_date and end_date:
                        df = self.downloader.download_history_data_by_date_range(
                            symbol, interval, start_date, end_date
                        )
                    else:
                        # 使用降级间隔尝试（5m → 3m → 1m → 1h，带聚合功能）
                        print(f"  [{i+1}/{len(symbols)}] {symbol}: 尝试下载 {interval} 数据...")
                        df, actual_interval = self.downloader.try_download_multiple_intervals(
                            symbol=symbol,
                            days=calculated_days,
                            preferred_intervals=[interval, '3m', '1m', '1h'],
                            force_download=False
                        )
                        if actual_interval and actual_interval != interval and df is not None:
                            print(f"  [{i+1}/{len(symbols)}] {symbol}: 使用 {actual_interval} (已聚合为 {interval})")

                    if df is not None and len(df) > 0:
                        # 检查需要聚合
                        df_interval = self.downloader.detect_interval(df)
                        if df_interval != interval and df_interval != '1m':  # 避免重复聚合
                            print(f"  [{i+1}/{len(symbols)}] {symbol}: 需要聚合 {df_interval} → {interval}")
                            df = self.downloader.aggregate_klines(df, interval)

                        # 如果指定了日期范围，过滤数据
                        if start_date and end_date:
                            start_ts = int(date_start_dt.timestamp() * 1000)
                            end_ts = int(date_end_dt.timestamp() * 1000 + 24 * 3600 * 1000)  # 包含结束当天
                            df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)].copy()

                        if len(df) > 0:
                            all_data[symbol] = df
                            print(f"  [OK] ({i+1}/{len(symbols)}) {symbol}: {len(df)} 根K线")
                        else:
                            print(f"  [FAIL] ({i+1}/{len(symbols)}) {symbol}: 数据已被日期范围过滤")
                    else:
                        print(f"  [FAIL] ({i+1}/{len(symbols)}) {symbol}: 数据下载失败 (df=None)")
                except Exception as e:
                    print(f"  [ERROR] ({i+1}/{len(symbols)}) {symbol}: {str(e)[:100]}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not all_data:
                print("\n没有可用数据，回测结束")
                return
            
            # 2. 回测主循环
            print(f"\n开始回测...")
            
            # 获取所有时间点
            all_timestamps = set()
            for df in all_data.values():
                all_timestamps.update(df['timestamp'].tolist())
            
            sorted_timestamps = sorted(all_timestamps)
            print(f"  时间点数量: {len(sorted_timestamps)}")
            
            if sorted_timestamps:
                start_ts_str = datetime.fromtimestamp(sorted_timestamps[0] / 1000).strftime('%Y-%m-%d %H:%M')
                end_ts_str = datetime.fromtimestamp(sorted_timestamps[-1] / 1000).strftime('%Y-%m-%d %H:%M')
                print(f"  实际时间范围: {start_ts_str} ~ {end_ts_str}")
            
            # 按时间顺序处理
            for i, timestamp in enumerate(sorted_timestamps):
                self.current_timestamp = timestamp

                # 更新回测执行器的上下文
                current_prices = {}
                for symbol, df in all_data.items():
                    kline_data = df[df['timestamp'] == timestamp]
                    if not kline_data.empty:
                        row = kline_data.iloc[0]
                        current_prices[symbol] = float(row['close'])

                        # 设置执行器上下文
                        self.executor.set_context(timestamp, row.to_dict())

                        # 检查警报条件（每个币种都检查）
                        alert_data = self._check_alert_conditions(symbol, row, df)
                        if alert_data:
                            self._handle_alert(alert_data, timestamp)

                # 更新持仓并返回需要处理的行动
                actions = self.strategy.update_positions(current_prices, timestamp)

                # 记录资金历史
                total_value = self.balance + self.get_total_position_value()
                self.results['balance_history'].append(total_value)
                self.results['timestamp_history'].append(timestamp)

                # 进度显示
                if i % 1000 == 0:
                    dt_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
                    print(f"  进度: {i + 1}/{len(sorted_timestamps)} - {dt_str}")
            
            # 3. 结束回测并返回结果
            result = self._finalize_backtest()

            return result

        except Exception as e:
            logger.error(f"回测失败: {str(e)}", exc_info=True)
            print(f"\n[错误] 回测失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 返回错误结果
            return None

    def _check_alert_conditions(self, symbol, current_kline, df):
        """
        检查警报条件（回测版本）

        基于价格涨跌幅和成交量判断是否触发警报
        使用AND逻辑：价格和成交量都必须达到阈值

        Args:
            symbol: 币种
            current_kline: 当前K线数据
            df: 完整的DataFrame

        Returns:
            警报数据字典，或None
        """
        try:
            current_price = float(current_kline['close'])
            current_timestamp = int(current_kline['timestamp'])

            # 找到当前K线在DataFrame中的索引
            current_idx = df[df['timestamp'] == current_timestamp].index
            if len(current_idx) == 0:
                return None
            current_idx = current_idx[0]

            # 计算监控周期内的K线数量
            # 假设1根K线代表的时间（根据interval判断）
            # 暂时使用5分钟作为监控周期
            monitor_klines_count = 5  # 假设5根K线为一个监控周期
            start_idx = max(0, current_idx - monitor_klines_count)
            historical_klines = df.iloc[start_idx:current_idx]

            if historical_klines.empty or len(historical_klines) < 2:
                return None

            # 计算价格涨跌幅
            first_price = float(historical_klines.iloc[0]['close'])
            price_change = ((current_price - first_price) / first_price) * 100

            # 计算成交量倍数
            current_volume = float(current_kline['quote_volume'])
            avg_volume = historical_klines['quote_volume'].mean()
            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 0

            # 限制最大倍数，防止异常值
            MAX_VOLUME_RATIO = 100.0
            if volume_ratio > MAX_VOLUME_RATIO:
                volume_ratio = MAX_VOLUME_RATIO

            # 检查价格和成交量阈值（AND逻辑）
            if abs(price_change) < settings.PRICE_CHANGE_THRESHOLD or volume_ratio < settings.VOLUME_THRESHOLD:
                return None

            # 检查冷却时间
            if not hasattr(self, 'alert_cooldown'):
                self.alert_cooldown = {}
            if not hasattr(self, 'alert_count'):
                self.alert_count = {}

            if symbol in self.alert_cooldown:
                last_alert = self.alert_cooldown[symbol]
                # 冷却时间：避免连续K线触发
                if current_idx - last_alert < monitor_klines_count:
                    return None

            # 判断方向
            alert_direction = 'LONG' if price_change > 0 else 'SHORT'

            # 警报计数
            if symbol not in self.alert_count:
                self.alert_count[symbol] = 0
            self.alert_count[symbol] += 1

            # 更新冷却时间
            self.alert_cooldown[symbol] = current_idx

            alert_data = {
                'symbol': symbol,
                'current_price': current_price,
                'price_change': price_change,
                'volume_usdt': current_volume / 1000,
                'volume_ratio': volume_ratio,
                'direction': alert_direction,
                'alert_count': self.alert_count[symbol],
                'timestamp': current_timestamp
            }

            logger.info(f"[回测警报] {symbol} | 价格: {current_price:.4f} | 涨跌: {price_change:+.2f}% | 成交量: {volume_ratio:.2f}x | 方向: {alert_direction}")

            return alert_data

        except Exception as e:
            logger.error(f"检查警报条件失败 {symbol}: {str(e)}")
            return None

    def _handle_alert(self, alert_data, timestamp):
        """
        处理警报并开仓

        Args:
            alert_data: 警报数据
            timestamp: 时间戳
        """
        try:
            symbol = alert_data['symbol']
            current_price = alert_data['current_price']
            direction = alert_data['direction']

            # 检查是否已有持仓
            if symbol in self.strategy.positions:
                logger.debug(f"{symbol} 已有持仓，跳过警报")
                return

            # 检查是否达到最大持仓数
            active_positions = sum(1 for p in self.strategy.positions.values()
                                  if p.get('status') == 'active')
            if active_positions >= settings.MAX_POSITIONS:
                logger.debug(f"已达到最大持仓数 {settings.MAX_POSITIONS}，跳过 {symbol}")
                return

            # 调用策略的handle_alert开仓
            success = self.strategy.handle_alert(alert_data, timestamp)

            if success:
                logger.info(f"✓ {symbol} 开仓成功: {direction} @ {current_price:.6f}")

        except Exception as e:
            logger.error(f"处理警报失败 {alert_data.get('symbol')}: {str(e)}")
    
    def simulate_alert(self, symbol, current_price, direction):
        """
        模拟警报触发（用于手动测试）
        
        Args:
            symbol: 币种
            current_price: 当前价格
            direction: 方向 'LONG' or 'SHORT'
        """
        alert_data = {
            'symbol': symbol,
            'current_price': current_price,
            'direction': direction,
            'timestamp': self.current_timestamp or int(time.time() * 1000)
        }
        
        success = self.strategy.handle_alert(alert_data, self.current_timestamp)
        
        if success:
            print(f"  ✓ {symbol} 开仓成功: {direction} @ {current_price:.6f}")
        else:
            print(f"  ✗ {symbol} 开仓失败")
        
        return success
    
    def _finalize_backtest(self):
        """结束回测，生成报告"""
        try:
            print(f"\n回测完成！")
            print(f"  最终资金: {self.balance + self.get_total_position_value():.2f} USDT")
            print(f"  初始资金: {self.initial_balance:.2f} USDT")

            total_profit = (self.balance + self.get_total_position_value()) - self.initial_balance
            profit_pct = (total_profit / self.initial_balance) * 100 if self.initial_balance > 0 else 0

            print(f"  总盈亏: {total_profit:+.2f} USDT ({profit_pct:+.2f}%)")

            print(f"\n持仓统计:")
            print(f"  交易次数: {len(self.executor.trade_history)}")
            print(f"  当前持仓: {len([p for p in self.strategy.positions.values() if p.get('status') == 'active'])}")

            # 显示最终持仓
            active_positions = []
            for symbol, pos in self.strategy.positions.items():
                if pos.get('status') == 'active':
                    print(f"  - {symbol}: {pos['direction']} 盈亏={pos['profit_pct']:+.2f}%")
                    active_positions.append(symbol)

            # 返回结果（不等待用户输入）
            return {
                'final_balance': self.balance + self.get_total_position_value(),
                'initial_balance': self.initial_balance,
                'total_profit': total_profit,
                'profit_pct': profit_pct,
                'trade_history': self.executor.trade_history,
                'active_positions': active_positions,
                'balance_history': self.results['balance_history'],
                'timestamp_history': self.results['timestamp_history']
            }

        except Exception as e:
            logger.error(f"结束回测失败: {str(e)}", exc_info=True)
            return None
    
    def get_results(self):
        """获取回测结果"""
        return self.results


def run_interactive_backtest():
    """交互式回测"""
    print("\n" + "=" * 70)
    print("统一回测系统（使用与实盘相同的交易逻辑）")
    print("=" * 70)
    
    # 创建回测引擎
    backtest = UnifiedBacktester()
    
    # 获取用户输入
    while True:
        try:
            symbols_input = input("\n请输入币种（多个用空格分隔，如 BTCUSDT ETHUSDT）: ").strip().upper()
            if symbols_input:
                symbols = symbols_input.split()
                break
            print("请输入至少一个币种！")
        except (EOFError, KeyboardInterrupt):
            print("\n用户取消")
            return
    
    # 选择日期范围模式
    use_date_range = False
    start_date = None
    end_date = None
    days = 180
    
    while True:
        try:
            choice = input("\n选择日期输入方式:\n  [1] 按天数（如: 30天）\n  [2] 按日期范围（如: 2025-01-01 ~ 2025-01-31）\n  请选择 [1/2]（默认1）: ").strip()
            if not choice:
                choice = '1'
            
            if choice == '1':
                # 按天数
                while True:
                    try:
                        days_input = input("回测天数（默认180天）: ").strip()
                        if not days_input:
                            days = 180
                            break
                        days = int(days_input)
                        if days > 0:
                            break
                        print("天数必须大于0！")
                    except ValueError:
                        print("请输入有效的数字！")
                break
            elif choice == '2':
                # 按日期范围
                use_date_range = True
                
                # 开始日期
                while True:
                    try:
                        start_input = input("开始日期（YYYY-MM-DD，如 2025-01-01）: ").strip()
                        if start_input:
                            datetime.strptime(start_input, '%Y-%m-%d')
                            start_date = start_input
                            break
                        print("请输入有效的日期格式！")
                    except ValueError:
                        print("日期格式错误，请使用 YYYY-MM-DD 格式（如 2025-01-01）")
                
                # 结束日期
                while True:
                    try:
                        end_input = input("结束日期（YYYY-MM-DD，如 2025-01-31）: ").strip()
                        if end_input:
                            datetime.strptime(end_input, '%Y-%m-%d')
                            
                            # 验证结束日期必须晚于开始日期
                            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                            end_dt = datetime.strptime(end_input, '%Y-%m-%d')
                            if end_dt <= start_dt:
                                print("结束日期必须晚于开始日期！")
                                continue
                            
                            end_date = end_input
                            break
                        print("请输入有效的日期格式！")
                    except ValueError:
                        print("日期格式错误，请使用 YYYY-MM-DD 格式（如 2025-01-31）")
                break
            else:
                print("无效选择，默认按天数")
                choice = '1'
        
        except (EOFError, KeyboardInterrupt):
            print("\n用户取消")
            return
    
    # 选择K线周期
    monitor_period = settings.MONITOR_INTERVAL
    
    intervals_map = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h'
    }
    
    if monitor_period <= 3:
        default_interval = '1m'
    elif monitor_period <= 5:
        default_interval = '1m'
    elif monitor_period <= 15:
        default_interval = '3m'
    elif monitor_period <= 30:
        default_interval = '5m'
    else:
        default_interval = '15m'
    
    print(f"\n根据监控周期 {monitor_period} 分钟，建议使用: {default_interval}")
    
    while True:
        interval_input = input(f"K线周期（默认{default_interval}）: ").strip().lower()
        if not interval_input:
            interval = default_interval
            break
        if interval_input in intervals_map.values():
            interval = interval_input
            break
        print(f"无效周期，可选: {', '.join(intervals_map.values())}")
    
    # 确认
    print(f"\n回测配置:")
    print(f"  币种: {', '.join(symbols)}")
    if use_date_range:
        print(f"  时间范围: {start_date} ~ {end_date}")
    else:
        print(f"  天数: {days}")
    print(f"  K线周期: {interval}")
    
    try:
        confirm = input("\n是否开始？(y/n): ").strip().lower()
        if confirm != 'y':
            print("回测已取消")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n用户取消")
        return
    
    # 运行回测
    try:
        if use_date_range:
            backtest.run_backtest(symbols, None, interval, start_date, end_date)
        else:
            backtest.run_backtest(symbols, days, interval)
    except Exception as e:
        print(f"\n[错误] 回测执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"回测执行失败: {str(e)}", exc_info=True)
    finally:
        # 确保不会闪退
        try:
            input("\n按回车键退出...")
        except:
            pass


def run_simple_backtest():
    """快速回测示例"""
    print("\n快速回测示例（使用与实盘相同的逻辑）")
    
    backtest = UnifiedBacktester()
    
    # 简单测试
    symbols = ['BTCUSDT']
    days = 30
    
    backtest.run_backtest(symbols, days, '1m')


if __name__ == '__main__':
    try:
        # 交互式回测
        run_interactive_backtest()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n[严重错误] {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"程序异常: {str(e)}", exc_info=True)
    finally:
        # 防止闪退
        import sys
        try:
            input("\n按回车键退出...")
        except:
            pass
        sys.exit(0)
