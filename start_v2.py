#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BNFRich v2.0 - 模块化交易框架
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== 全局交易记录缓存 ==========
_trades_cache = None
_trades_file = None
_trades_cache_time = None  # 缓存时间戳
_trades_cache_ttl = 300  # 缓存过期时间（秒），默认5分钟

def _load_trades():
    """加载交易记录（带缓存过期机制）"""
    global _trades_cache, _trades_file, _trades_cache_time

    current_time = time.time()

    # 检查缓存是否过期
    if _trades_cache is not None and _trades_cache_time is not None:
        if current_time - _trades_cache_time < _trades_cache_ttl:
            # 缓存未过期，直接返回
            return _trades_cache
        else:
            # 缓存过期，清除缓存
            print(f" [缓存过期] 交易记录缓存已过期（{_trades_cache_ttl}秒），重新加载")
            _trades_cache = None
            _trades_file = None
            _trades_cache_time = None

    import glob
    results_dir = Path(__file__).parent / 'results'
    csv_files = sorted(results_dir.glob('*_detailed_*.csv'), key=os.path.getmtime, reverse=True)

    if not csv_files:
        print(" [警告] 未找到交易记录文件")
        _trades_cache = []
        _trades_cache_time = current_time
        return _trades_cache

    # 使用最新的文件
    _trades_file = csv_files[0]
    import pandas as pd
    try:
        _trades_cache = pd.read_csv(_trades_file, encoding='utf-8-sig')
        _trades_cache_time = current_time
        print(f" 已加载: {_trades_file.name} ({len(_trades_cache)} 笔交易)")
    except Exception as e:
        print(f" 加载失败: {e}")
        _trades_cache = []
        _trades_cache_time = current_time

    return _trades_cache

def _show_trades():
    """显示交易记录表格"""
    global _trades_cache, _trades_file
    trades = _load_trades()
    
    if trades is None or len(trades) == 0:
        print(" 无交易记录")
        return
    
    print(f"\n{'='*100}")
    print(f"{'序号':<5} {'币种':<15} {'方向':<6} {'入场价':<12} {'出场价':<12} {'杠杆':<5} {'盈亏%':<10} {'持仓':<6} {'出场原因':<10} {'信号类型'}")
    print(f"{'-'*100}")
    
    for idx, row in trades.head(50).iterrows():  # 显示前50笔
        print(f"{idx+1:<5} {row['symbol']:<15} {row['side']:<6} {row['entry_price']:<12.6f} {row['exit_price']:<12.6f} {row['leverage']:<5.0f} {row['pnl_pct']:<10.2f} {row['hold_bars_5m']:<6} {row['exit_reason']:<10} {row['signal_type']}")
    
    print(f"{'='*100}")
    print(f"共 {len(trades)} 笔交易 | 文件: {_trades_file.name if _trades_file else 'N/A'}")
    print(f"查看指定交易图表: 图表 <序号>")
    print()

def _show_chart(idx: int):
    """生成指定交易的K线图表"""
    global _trades_cache, _trades_file
    trades = _load_trades()
    
    if trades is None or len(trades) == 0:
        print(" 无交易记录")
        return
    
    if idx < 1 or idx > len(trades):
        print(f" 序号 {idx} 超出范围 (1-{len(trades)})")
        return
    
    trade = trades.iloc[idx - 1]
    symbol = trade['symbol']
    
    # 尝试找到对应的K线文件
    import glob
    data_dir = Path(__file__).parent / 'data' / 'historical'
    kline_files = sorted(data_dir.glob(f'{symbol}_5m_*.csv'), key=os.path.getmtime, reverse=True)
    
    if not kline_files:
        kline_files = sorted(data_dir.glob(f'{symbol}_15m_*.csv'), key=os.path.getmtime, reverse=True)
    
    if not kline_files:
        print(f" 未找到 {symbol} 的K线数据文件")
        return
    
    kline_file = kline_files[0]
    
    # 生成图表
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    df = pd.read_csv(kline_file)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    entry_idx = int(trade['entry_idx_5m'] if 'entry_idx_5m' in trade else trade.get('entry_idx', 0))
    exit_idx = int(trade['exit_idx_5m'] if 'exit_idx_5m' in trade else trade.get('exit_idx', 0))
    
    # 调整范围
    start_idx = max(0, entry_idx - 20)
    end_idx = min(len(df), exit_idx + 10)
    plot_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    
    entry_rel = entry_idx - start_idx
    exit_rel = exit_idx - start_idx
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制K线
    for i in range(len(plot_df)):
        o = plot_df['open'].iloc[i]
        h = plot_df['high'].iloc[i]
        l = plot_df['low'].iloc[i]
        c = plot_df['close'].iloc[i]
        
        is_bullish = c >= o
        color = '#26a69a' if is_bullish else '#ef5350'
        
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < (h - l) * 0.1:
            body_height = (h - l) * 0.2
            body_bottom = (h + l) / 2 - body_height / 2
        
        rect = Rectangle((i - 0.4, body_bottom), 0.8, body_height, facecolor=color, edgecolor=color, alpha=0.9)
        ax.add_patch(rect)
        ax.plot([i, i], [l, min(o, c)], color=color, linewidth=1)
        ax.plot([i, i], [max(o, c), h], color=color, linewidth=1)
    
    # 入场/出场标记
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    
    ax.scatter(entry_rel, entry_price, color='blue', s=200, marker='^', zorder=10, label='ENTRY')
    ax.scatter(exit_rel, exit_price, color='green' if trade['pnl_pct'] > 0 else 'red', s=200, marker='v', zorder=10, label='EXIT')
    ax.axhline(y=entry_price, color='blue', linestyle='--', alpha=0.6)
    ax.axhline(y=exit_price, color='green' if trade['pnl_pct'] > 0 else 'red', linestyle='--', alpha=0.6)
    ax.axvline(x=entry_rel, color='blue', linestyle='-', alpha=0.3, linewidth=5)
    ax.axvline(x=exit_rel, color='red', linestyle='-', alpha=0.3, linewidth=5)
    ax.axvspan(0, entry_rel, alpha=0.1, color='yellow')
    
    ax.set_title(f"{symbol} | {trade['side']} | Entry@{entry_price:.6f} -> Exit@{exit_price:.6f} | PnL: {trade['pnl_pct']:.2f}% | {trade['signal_type']}", fontsize=12)
    ax.set_ylabel('Price')
    ax.set_xlabel('Bar Index')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path(__file__).parent / 'results' / 'charts'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"trade_{idx}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" 图表已保存: {output_file}")


def check_config():
    print("\n" + "=" * 60)
    print("配置检查 - 新框架配置系统")
    print("=" * 60)
    try:
        from framework.core.config import get_config_manager
        
        # 先初始化配置管理器
        config_manager = get_config_manager()
        
        # 加载主配置
        if not config_manager.load_main_config():
            print("[ERROR] 主配置加载失败")
            return False
        
# 加载策略配置
        config_manager.load_strategy_configs()
        main_config = config_manager.main_config
        
        # 获取策略配置
        v23_config = config_manager.get_strategy_config('15mTupo')
        hf_config = config_manager.get_strategy_config('hf')
        
        print(f"\n当前配置（新系统）:")
        print(f"  杠杆倍数: {main_config.leverage}x")
        print(f"  最大持仓: {main_config.max_positions}")
        print(f"  保证金模式: {main_config.margin_mode}")
        print(f"  日亏损限制: {main_config.emergency_daily_loss_percent}%")
        print(f"  连续亏损限制: {main_config.emergency_continuous_loss}")
        print(f"\n策略状态:")
        print(f"  V23策略: {'启用' if v23_config and v23_config.enabled else '禁用'}")
        print(f"  HF策略: {'启用' if hf_config and hf_config.enabled else '禁用'}")
        print("\n[OK] 配置验证通过（新系统）")
        return True
    except Exception as e:
        print(f"[ERROR] 配置检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    print("\n" + "=" * 60)
    print("依赖检查")
    print("=" * 60)
    required = {'pandas': 'pandas', 'numpy': 'numpy', 'requests': 'requests', 'python-dotenv': 'dotenv'}
    print(f"\nPython: {sys.version.split()[0]}")
    missing = []
    for pkg, mod in required.items():
        try:
            m = __import__(mod)
            ver = getattr(m, '__version__', 'unknown')
            print(f"  [OK] {pkg} ({ver})")
        except ImportError:
            print(f"  [X] {pkg} - 未安装")
            missing.append(pkg)
    if missing:
        print(f"\n缺失: {', '.join(missing)}")
    else:
        print("\n[OK] 所有依赖已安装")


def clean_files():
    print("\n" + "=" * 60)
    print("清理日志和缓存")
    print("=" * 60)
    import shutil
    
    total_deleted = 0
    
    # 1. 清理日志文件
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            print(f"\n[1/3] 删除 {len(log_files)} 个日志文件...")
            for f in log_files:
                try:
                    f.unlink()
                    print(f"  [OK] {f.name}")
                    total_deleted += 1
                except Exception as e:
                    print(f"  [SKIP] {f.name}: {e}")
    
    # 2. 清理Python缓存（直接删除__pycache__目录，不需要单独删除.pyc文件）
    print(f"\n[2/3] 清理Python缓存...")
    cache_dirs = []
    
    # 查找所有__pycache__目录
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_path = Path(root) / "__pycache__"
            cache_dirs.append(cache_path)
    
    # 删除__pycache__目录（会自动删除其中的.pyc文件）
    if cache_dirs:
        print(f"  删除 {len(cache_dirs)} 个__pycache__目录...")
        for cache_dir in cache_dirs:
            try:
                shutil.rmtree(cache_dir)
                print(f"  [OK] {cache_dir}")
                total_deleted += 1
            except Exception as e:
                print(f"  [SKIP] {cache_dir}: {e}")
    
    # 3. 清理其他缓存目录
    print(f"\n[3/3] 清理其他缓存...")
    other_cache_dirs = [
        Path(".pytest_cache"),
        Path(".mypy_cache"),
        Path(".ruff_cache"),
        Path("__pycache__"),
    ]
    
    for cache_dir in other_cache_dirs:
        if cache_dir.exists():
            try:
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    print(f"  [OK] {cache_dir}")
                    total_deleted += 1
                else:
                    cache_dir.unlink()
                    print(f"  [OK] {cache_dir}")
                    total_deleted += 1
            except Exception as e:
                print(f"  [SKIP] {cache_dir}: {e}")
    
    print(f"\n[OK] 清理完成，共删除 {total_deleted} 个文件/目录")


def show_help():
    print("\n" + "=" * 60)
    print("帮助")
    print("=" * 60)
    print("""
命令行:
    python start_v2.py                    # 交互模式
    python start_v2.py --mode testnet     # 测试网
    python start_v2.py --mode live        # 实盘
    python start_v2.py --check            # 检查配置

交互命令 (支持中英文):
    状态/status    - 账户状态
    持仓/positions - 当前持仓
    待处理/pending - 等待订单
    价格 X         - 查询价格
    平仓 X         - 平指定仓位
    全平/closeall  - 平所有仓位
    同步/sync      - 同步持仓
    交易/trade     - 开启交易（默认只监控）
    关交易/closetrade - 关闭交易（监控继续）
    停止/stop      - 停止监控
    退出/exit      - 退出程序
""")


class TradingFramework:
    def __init__(self, mode: str = 'testnet'):
        self.mode = mode
        self.testnet = (mode == 'testnet')
        self.client = None
        self.trader = None
        self._running = False
        self._monitor_running = False
        self._trading_active = False  # 交易开关，需手动开启
        
    def initialize(self, load_v23=True, load_hf=False) -> bool:
        try:
            from framework.core.logger import init_logging, get_logger, LogConfig
            from framework.core.config import ConfigManager, get_config_manager
            from framework.core.events import EventBus, get_event_bus
            from framework.business.kline_manager import KlineManager
            from framework.business.order_engine import OrderEngine
            from framework.business.position_monitor import PositionMonitor
            from framework.business.risk_center import RiskCenter
            from framework.strategy.scheduler import StrategyScheduler
            from utils.input_validator import safe_float, safe_int, normalize_position_side, normalize_order_side
            from framework.strategy.plugin_loader import PluginLoader
            
            init_logging(LogConfig(name='BNFRich', level='INFO', log_dir='logs'))

            # 启用配置热重载
            from config.settings import Settings
            _reload_interval = int(os.environ.get('RELOAD_INTERVAL', '30'))
            if _reload_interval > 0:
                Settings.enable_hot_reload(interval=_reload_interval)

            print("\n" + "=" * 60)
            print(f"BNFRich v2.0 - {'测试网' if self.testnet else '实盘'} 模式")
            print("=" * 60)
            print("[提示] 测试网模式，使用测试资金" if self.testnet else "[警告] 实盘模式，真实资金！")
            
            print("\n[1/6] 加载配置...")
            self.config_manager = get_config_manager()
            if not self.config_manager.load_main_config():
                print("  [错误] 配置加载失败")
                return False
            errors = self.config_manager.validate()
            if errors:
                for err in errors:
                    print(f"  [错误] {err}")
                return False
            print("  [OK] 配置已加载")
            
            print("[2/6] 连接交易所...")
            from trading.binance_client import BinanceClient
            
            self.client = BinanceClient(mode=self.mode)
            balance = self.client.get_account_balance()
            total = balance.get('total_balance', balance.get('balance', 0))
            available = balance.get('available_balance', balance.get('availableBalance', 0))
            print(f" 总余额: {total:.2f} USDT")
            print(f" 可用余额: {available:.2f} USDT")
            
            # 初始化 Telegram 机器人（信号→alert bot，交易→trade bot）
            try:
                from alert.telegram_bot import TelegramBot
                self.telegram_bot = TelegramBot(bot_type='alert')
                self.telegram_trade_bot = TelegramBot(bot_type='trade')
                print(" [OK] Telegram 机器人已连接 (alert+trade)")
            except ImportError:
                print(" [警告] Telegram 模块未安装，跳过")
                self.telegram_bot = None
                self.telegram_trade_bot = None
            
            print("[3/6] 初始化组件...")
            self.event_bus = get_event_bus()
            self.event_bus.start()
            # 构造函数注入，消除单例 + 双重依赖
            self.kline_manager = KlineManager(self.client)
            self.order_engine = OrderEngine(self.client)
            self.position_monitor = PositionMonitor(self.client)
            self.position_monitor.set_order_engine(self.order_engine)
            self.risk_center = RiskCenter()
            self.risk_center.initialize(total)
            
            print("[4/6] 初始化调度器...")
            self.scheduler = StrategyScheduler()
            self.scheduler.set_dependencies(self.kline_manager, self.order_engine, self.position_monitor, self.risk_center, self.event_bus)
            self.position_monitor.set_scheduler(self.scheduler)
            
            self.position_monitor.start_monitor()
            print("  [OK] 组件就绪")

            # 设置Telegram交易按钮回调
            if self.telegram_bot:
                self.telegram_bot.set_trade_executor(self.scheduler.execute_telegram_signal)

            # 添加 Telegram 消息发送的事件监听
            def on_signal_generated(event):
                """信号生成时发送 Telegram 消息（仅对不自发通知的策略有效）"""
                try:
                    data = event.data
                    symbol = data.get('symbol', '')
                    signal_type = data.get('signal_type', '')
                    price = data.get('price', 0)
                    reason = data.get('reason', '')
                    strategy = data.get('strategy', '')

                    # HF 和 15mTupo 策略自己发送带图表和交易按钮的 Telegram 消息，
                    # 避免在此处重复发送纯文本消息
                    if strategy in ('hf', '15mTupo'):
                        return

                    message = f"""<b>📊 信号生成</b>

<b>策略</b>: {strategy}
<b>币种</b>: {symbol}
<b>方向</b>: {signal_type}
<b>价格</b>: {price:.6f}
<b>原因</b>: {reason}

时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
                    # 使用线程池异步发送，避免阻塞事件处理
                    import concurrent.futures
                    executor = getattr(self, '_telegram_executor', None)
                    if executor is None:
                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='telegram')
                        self._telegram_executor = executor
                    executor.submit(self.telegram_bot.send_message, message)
                except Exception as e:
                    print(f" [警告] Telegram 发送失败: {e}")

            def on_order_filled(event):
                """订单成交时发送 Telegram 消息并记录持仓"""
                try:
                    data = event.data
                    # symbol/side/strategy 是 OrderEvent 的顶层属性，不在 data dict 中
                    symbol = getattr(event, 'symbol', '') or data.get('symbol', '')
                    side = getattr(event, 'side', '') or data.get('side', '')
                    quantity = data.get('quantity', 0) or data.get('filled_quantity', 0)
                    price = data.get('price', 0)
                    strategy = getattr(event, 'strategy', '') or data.get('strategy', '')

                    side_emoji = "🚀" if side == "BUY" else "⚡"
                    message = f"""<b>{side_emoji} 订单成交</b>

<b>币种</b>: {symbol}
<b>方向</b>: {side}
<b>数量</b>: {quantity}
<b>价格</b>: {price:.6f}

时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
                    # 使用线程池异步发送，避免阻塞事件处理
                    import concurrent.futures
                    executor = getattr(self, '_telegram_executor', None)
                    if executor is None:
                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='telegram')
                        self._telegram_executor = executor
                    tb = getattr(self, 'telegram_trade_bot', self.telegram_bot)
                    executor.submit(tb.send_message, message)

                    # 记录持仓到 position_monitor
                    if self.position_monitor and strategy and symbol and quantity > 0:
                        try:
                            from framework.business.position_monitor import PositionInfo, PositionSide
                            from framework.core.config import get_config_manager

                            # 获取策略配置
                            config_manager = get_config_manager()
                            strategy_config = config_manager.get_strategy_config(strategy)

                            # 确定持仓方向
                            position_side = PositionSide.LONG if side == "BUY" else PositionSide.SHORT

                            # 计算保证金
                            leverage = strategy_config.leverage if strategy_config else 20
                            margin = quantity * price / leverage if leverage > 0 else 0

                            # 获取止损价格
                            stop_loss_price = data.get('stop_loss_price')
                            take_profit_price = data.get('take_profit_price')

                            # 创建持仓信息
                            position = PositionInfo(
                                symbol=symbol,
                                strategy=strategy,
                                side=position_side,
                                quantity=quantity,
                                entry_price=price,
                                current_price=price,
                                leverage=leverage,
                                margin=margin,
                                stop_loss_price=stop_loss_price,
                                take_profit_price=take_profit_price,
                            )

                            # 记录持仓
                            if self.position_monitor.open_position(position):
                                print(f" [持仓记录] {symbol} 已记录到 position_monitor")

                                # 自动挂限价止损单
                                if stop_loss_price:
                                    try:
                                        self.position_monitor._place_stop_loss_order(symbol, position)
                                        print(f" [止损单] {symbol} 已挂限价止损单 @ {stop_loss_price:.6f}")
                                    except Exception as e:
                                        print(f" [止损单失败] {symbol}: {e}")

                        except Exception as e:
                            print(f" [持仓记录失败] {symbol}: {e}")

                except Exception as e:
                    print(f" [警告] 订单成交处理失败: {e}")
            
            from framework.core.events import EventType
            self.event_bus.subscribe(EventType.SIGNAL_GENERATED, on_signal_generated)
            self.event_bus.subscribe(EventType.ORDER_FILLED, on_order_filled)
            print(" [OK] 调度器就绪")
            
            print("[5/6] 加载策略...")
            loader = PluginLoader()
            discovered = loader.discover()
            loaded = []
            strategies = []
            if load_v23 and '15mTupo' in discovered: strategies.append('15mTupo')
            if load_hf and 'hf' in discovered: strategies.append('hf')
            for name in strategies:
                cls = discovered.get(name)
                if cls and self.scheduler.load_strategy(name, cls):
                    loaded.append(name)
                    print(f"  [OK] {name} 已加载")
            if not loaded:
                print("  [警告] 未加载策略")
            
            print("[6/6] 设置监控币种...")
            # 获取初步的监控币种列表（按最大需求获取）
            max_symbols = 0
            for name in loaded:
                if name == 'hf':
                    cnt = int(os.getenv('HF_MONITOR_SYMBOLS', '50'))
                elif name == '15mTupo':
                    cnt = int(os.getenv('15MTUPO_MONITOR_SYMBOLS',
                               os.getenv('MONITOR_SYMBOLS', '30')))
                else:
                    cnt = 30
                if cnt > max_symbols:
                    max_symbols = cnt

            raw_symbols = self._get_top_symbols(max_symbols * 2)
            
            # 获取已有持仓的币种（这些不需要再检测信号）
            existing_symbols = set()
            if self.position_monitor:
                try:
                    positions = self.position_monitor.get_all_positions()
                    for pos in positions:
                        if pos.symbol:
                            existing_symbols.add(pos.symbol)
                except Exception as e:
                    import logging
                    logging.debug(f"获取持仓列表失败: {e}")
            
            # 为每个策略设置独立的币种列表
            for name in loaded:
                if name == 'hf':
                    cnt = int(os.getenv('HF_MONITOR_SYMBOLS', '50'))
                elif name == '15mTupo':
                    cnt = int(os.getenv('15MTUPO_MONITOR_SYMBOLS',
                               os.getenv('MONITOR_SYMBOLS', '30')))
                else:
                    cnt = 30
                symbols = [s for s in raw_symbols if s not in existing_symbols]
                symbols = symbols[:cnt]
                self.scheduler.set_strategy_symbols(name, symbols)
                print(f"  {name}: {len(symbols)} 个币种")
                
            print("\n" + "=" * 60)
            print("系统就绪")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n[错误] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _get_top_symbols(self, count=50):
        import requests
        base = "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
        try:
            resp = requests.get(f"{base}/fapi/v1/ticker/24hr", timeout=30)
            data = resp.json()
            # 过滤USDT合约并按成交量排序
            usdt = [i for i in data if i['symbol'].endswith('USDT') and float(i.get('quoteVolume', 0)) > 0]
            if len(usdt) < count:
                print(f"[警告] 仅获取到 {len(usdt)} 个USDT交易对，少于请求的 {count}")
            sorted_symbols = [i['symbol'] for i in sorted(usdt, key=lambda x: float(x['quoteVolume']), reverse=True)]
            return sorted_symbols[:count]
        except Exception as e:
            import logging
            logging.warning(f"获取币种列表失败: {e}, 使用默认列表")
            # 返回更多默认币种
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT',
                    'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT', 'XLMUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT']
            
    def load_klines(self, symbols):
        if not self.kline_manager or not self.scheduler: return
        intervals = set()
        for info in self.scheduler._strategies.values():
            intervals.update(info.instance.get_intervals())
        print(f"\n加载K线: {len(symbols)} 个币种")
        loaded = 0
        for s in symbols:
            for iv in intervals:
                if self.kline_manager.load_history(s, iv): loaded += 1
                time.sleep(0.05)
        print(f"  [OK] 已加载: {loaded}")
        
    def start_monitor(self):
        if self._monitor_running: print("  已在运行"); return
        self._monitor_running = True
        if self.scheduler: self.scheduler.start()
        print("  监控已启动")
        
    def stop_monitor(self):
        if not self._monitor_running: print("  未运行"); return
        self._monitor_running = False
        if self.scheduler: self.scheduler.stop()
        print("  监控已停止")
        
    def get_status(self):
        s = {'positions': 0, 'daily_pnl': 0, 'balance': 0}
        if self.trader and hasattr(self.trader, 'get_status'):
            try: 
                s.update(self.trader.get_status())
            except Exception as e:
                import logging
                logging.debug(f"获取交易状态失败: {e}")
        if self.client:
            try: 
                s['balance'] = self.client.get_account_balance().get('total_balance', 0)
            except Exception as e:
                import logging
                logging.debug(f"获取账户余额失败: {e}")
        return s
        
    def get_positions(self):
        """获取所有持仓（从PositionMonitor获取）"""
        if self.position_monitor:
            return self.position_monitor.get_all_positions()
        return []

    def sync_positions(self):
        """同步持仓（从API拉取最新持仓到PositionMonitor）"""
        if self.position_monitor:
            if self.position_monitor.sync_from_api():
                print(" 已同步")
            else:
                print(" 同步失败")
        else:
            print(" [错误] PositionMonitor未初始化")

    def close_position(self, symbol):
        """平仓指定币种（含方向校验与反向持仓检查）"""
        # 先确保symbol格式正确（添加USDT后缀）
        if not symbol.upper().endswith('USDT'):
            symbol = symbol.upper() + 'USDT'

        if self.client:
            try:
                # 获取当前持仓
                positions = self.client.get_positions()

                # 反向持仓检查：同币种是否存在反向持仓
                opposite_symbols = set()
                for pos in positions:
                    sym = pos.get('symbol', '')
                    if sym and sym != symbol:
                        amt = safe_float(pos.get('position_amt', 0), default=0.0)
                        if abs(amt) > 1e-8:
                            opposite_symbols.add(sym)

                for pos in positions:
                    if pos.get('symbol') == symbol:
                        # 使用安全转换
                        pos_amt = safe_float(pos.get('position_amt') or pos.get('positionAmt', 0),
                                            default=0.0, context=f'{symbol} close_position pos_amt')
                        if abs(pos_amt) < 1e-8:
                            continue

                        # 方向标准化 + 平仓方向推断
                        pos_side = normalize_position_side(pos.get('side'), pos_amt, default='LONG')
                        order_side = normalize_order_side(None, pos_side, pos_amt, default='BUY')

                        # 校验平仓方向与持仓一致性
                        expected_side = 'SELL' if pos_side == 'LONG' else 'BUY'
                        if order_side != expected_side:
                            print(f"[方向校验] {symbol} 平仓方向修正: {order_side} -> {expected_side}")
                            order_side = expected_side

                        qty = abs(pos_amt)

                        # 调整精度
                        qty = self.client.adjust_quantity_precision(symbol, qty)

                        # 发送市价平仓单
                        result = self.client.create_order(symbol, order_side, 'MARKET', qty, reduce_only=True)
                        if result:
                            print(f" 已平: {symbol} {order_side} {qty}")
                            # 从本地监控移除
                            if self.position_monitor:
                                self.position_monitor.close_position(symbol, "手动平仓")
                        else:
                            print(f" 平仓失败: {symbol}")
                        return
                print(f" 无持仓: {symbol}")
            except Exception as e:
                print(f" 平仓错误: {e}")
        else:
            print(" [错误] Client未初始化")
            
    def close_all(self):
        """全平：一次性获取持仓列表，逐个市价平仓（避免重复查询API触发限流）"""
        positions = self.get_positions()
        if not positions:
            print(" 无持仓")
            return
        count = 0
        failed = 0
        for p in positions:
            sym = p.symbol if hasattr(p, 'symbol') else ''
            if not sym:
                continue
            try:
                pos_side = p.side.value if hasattr(p.side, 'value') else str(p.side)
                order_side = 'SELL' if pos_side == 'LONG' else 'BUY'
                qty = self.client.adjust_quantity_precision(sym, abs(p.quantity))
                if qty <= 0:
                    print(f" {sym} 数量无效，跳过")
                    continue
                result = self.client.create_order(sym, order_side, 'MARKET', qty, reduce_only=True)
                if result:
                    print(f" 已平: {sym} {order_side} {qty}")
                    self.position_monitor.close_position(sym, "手动平仓")
                    count += 1
                else:
                    print(f" 平仓失败: {sym}")
                    failed += 1
            except Exception as e:
                print(f" 平仓错误: {sym} {e}")
                failed += 1
            time.sleep(0.5)  # 避免触发限流
        print(f"全平完成: 成功{count}个, 失败{failed}个")
            
    def _handle_bot_command(self, text):
        """处理Telegram bot命令，返回回复文本"""
        text = text.strip().lower()
        parts = text.split(maxsplit=1)
        if not parts:
            return "命令不能为空"
        base_cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ''

        cmd_map = {
            '状态': 'status', 'status': 'status',
            '持仓': 'positions', 'positions': 'positions',
            '平仓': 'close', 'close': 'close',
            '全平': 'closeall', 'closeall': 'closeall',
            '同步': 'sync', 'sync': 'sync',
            '交易': 'trade', 'trade': 'trade',
            '关闭交易': 'tradestop', '关交易': 'tradestop',
            '自働': 'autotrade', '自动交易': 'autotrade', 'autotrade': 'autotrade',
            '停止': 'stop', 'stop': 'stop',
            '价格': 'price', 'price': 'price',
            '帮助': 'help', 'help': 'help',
        }
        mapped = cmd_map.get(base_cmd, base_cmd)

        if mapped == 'status':
            s = self.get_status()
            lines = [
                f"交易: {'开启' if self._trading_active else '关闭'}",
                f"持仓: {s['positions']}",
                f"日盈亏: {s['daily_pnl']:.2f} USDT",
                f"余额: {s['balance']:.2f} USDT",
            ]
            # 自働交易状态（各策略）
            if self.scheduler:
                for sname, sinfo in self.scheduler._strategies.items():
                    at = getattr(sinfo.instance, '_auto_trade', None)
                    if at is not None:
                        lines.append(f"{sname}自働: {'开' if at else '关'}")
            return '\n'.join(lines)

        elif mapped == 'positions':
            if not self.position_monitor:
                return "PositionMonitor未初始化"
            try:
                self.position_monitor.sync_from_api()
            except Exception:
                import logging
                logging.debug("持仓同步失败")
            ps = self.position_monitor.get_all_positions()
            if not ps:
                return "无持仓"
            lines = []
            for p in ps:
                side_str = p.side.value if hasattr(p.side, 'value') else str(p.side)
                pnl = getattr(p, 'unrealized_pnl', 0)
                pnl_pct = getattr(p, 'pnl_pct', 0)
                lines.append(f"{p.symbol} {side_str} {abs(p.quantity):.4f} @{p.entry_price:.4f} PnL:{pnl:.2f}({pnl_pct:.1f}%)")
            return '\n'.join(lines)

        elif mapped == 'close' and arg:
            symbol = arg.upper()
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            if not self.client:
                return "Client未初始化"
            try:
                positions = self.client.get_positions()
                for pos in positions:
                    if pos.get('symbol') == symbol:
                        from utils.input_validator import safe_float, normalize_position_side, normalize_order_side
                        pos_amt = safe_float(pos.get('position_amt') or pos.get('positionAmt', 0), default=0.0)
                        if abs(pos_amt) < 1e-8:
                            continue
                        pos_side = normalize_position_side(pos.get('side'), pos_amt, default='LONG')
                        order_side = normalize_order_side(None, pos_side, pos_amt, default='BUY')
                        expected_side = 'SELL' if pos_side == 'LONG' else 'BUY'
                        if order_side != expected_side:
                            order_side = expected_side
                        qty = abs(pos_amt)
                        qty = self.client.adjust_quantity_precision(symbol, qty)
                        result = self.client.create_order(symbol, order_side, 'MARKET', qty, reduce_only=True)
                        if result:
                            if self.position_monitor:
                                self.position_monitor.close_position(symbol, "Bot平仓")
                            return f"已平仓: {symbol}"
                        return f"平仓失败: {symbol}"
                return f"无持仓: {symbol}"
            except Exception as e:
                return f"平仓错误: {e}"

        elif mapped == 'closeall':
            if not self.position_monitor:
                return "无持仓"
            ps = self.position_monitor.get_all_positions()
            count = 0
            for p in ps:
                sym = p.symbol if hasattr(p, 'symbol') else ''
                if sym:
                    self.close_position(sym)
                    count += 1
            return f"已平 {count} 个持仓"

        elif mapped == 'sync':
            if self.position_monitor:
                try:
                    self.position_monitor.sync_from_api()
                    return "同步完成"
                except Exception as e:
                    return f"同步失败: {e}"
            return "PositionMonitor未初始化"

        elif mapped == 'trade':
            self._trading_active = True
            if self.scheduler:
                self.scheduler.enable_trading()
            return "交易已开启"

        elif mapped == 'autotrade':
            enabled = None
            al = arg.lower() if arg else ''
            if al in ('on', '1', 'true', 'yes', '开启', '开'):
                enabled = True
            elif al in ('off', '0', 'false', 'no', '关闭', '关'):
                enabled = False
            count = 0
            if self.scheduler:
                for sname, sinfo in self.scheduler._strategies.items():
                    if hasattr(sinfo.instance, '_auto_trade'):
                        if enabled is None:
                            enabled = not sinfo.instance._auto_trade
                        sinfo.instance._auto_trade = enabled
                        count += 1
            if enabled is None:
                return "自働交易状态查询失败（无策略实例）"
            status = '开启' if enabled else '关闭'
            return f"自働交易已{status}（{count}个策略）"

        elif mapped == 'tradestop':
            self._trading_active = False
            if self.scheduler:
                self.scheduler.disable_trading()
            return "交易已关闭，监控继续"

        elif mapped == 'stop':
            self._trading_active = False
            if self.scheduler:
                self.scheduler.disable_trading()
            self.stop_monitor()
            return "监控已停止"

        elif mapped == 'price' and arg:
            symbol = arg.upper()
            if not self.client:
                return "Client未初始化"
            try:
                price = self.client.get_price(symbol)
                if price:
                    return f"{symbol}: {price:.8f}"
                return f"获取价格失败: {symbol}"
            except Exception as e:
                return f"价格查询错误: {e}"

        elif mapped == 'help':
            return ("支持命令: 状态/持仓/平仓 X/全平/同步/交易/关闭交易/停止/价格 X")

        else:
            return f"未知命令: {text} (输入 help 查看)"

    def stop(self):
        self._running = False
        self.stop_monitor()
        # 清理Telegram线程池
        executor = getattr(self, '_telegram_executor', None)
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
            self._telegram_executor = None
        # 关闭TelegramBot的事件循环
        for _name in ('telegram_bot', 'telegram_trade_bot'):
            _bot = getattr(self, _name, None)
            if _bot:
                _bot.close()
        # 关闭BinanceClient连接
        _client = getattr(self, 'client', None)
        if _client and hasattr(_client, 'close'):
            _client.close()
        # 停止事件总线
        event_bus = getattr(self, 'event_bus', None)
        if event_bus:
            event_bus.stop()


def run_mode(testnet=True, load_v23=None, load_hf=None):
    """
    运行交易模式

    Args:
        testnet: 是否使用测试网（默认True）
        load_v23: 是否加载15mTupo策略（None表示根据环境变量决定）
        load_hf: 是否加载HF策略（None表示根据环境变量决定）
    """
    import os  # 确保模块级别也可用
    # 如果未指定，根据环境变量开关决定
    if load_v23 is None:
        load_v23 = os.getenv('15MTUPO_ENABLED', 'true').lower() == 'true'
    if load_hf is None:
        load_hf = os.getenv('HF_ENABLED', 'false').lower() == 'true'

    fw = TradingFramework(mode='testnet' if testnet else 'live')
    if not fw.initialize(load_v23, load_hf): return

    # 从 scheduler 获取已注册的监控币种（与 initialize 中设置的一致）
    # 避免因获取时机不同导致 K线加载和策略监控的币种不一致
    all_symbols = set()
    if fw.scheduler:
        for info in fw.scheduler._strategies.values():
            all_symbols.update(info.symbols)
    symbol_count = int(os.getenv('15MTUPO_MONITOR_SYMBOLS',
                       os.getenv('MONITOR_SYMBOLS', '30')))
    symbols = sorted(all_symbols) if all_symbols else fw._get_top_symbols(symbol_count)

    # 如果 scheduler 中没有 symbols（新启动），回退到正常获取
    if not symbols:
        symbols = fw._get_top_symbols(symbol_count)

    kline_count = int(os.getenv('15MTUPO_LOAD_KLINES_COUNT', '0'))

    # 加载历史K线（kline_count<=0 时加载全部）
    symbols_to_load = symbols if kline_count <= 0 else symbols[:min(len(symbols), kline_count)]
    fw.load_klines(symbols_to_load)
    fw.start_monitor()  # 启动监控（不启动交易）
    print(f"\n=== 监控启动 ===")
    print(f"  监控币种: {len(symbols)} 个 (请求: {symbol_count} 个)")
    if len(symbols) < symbol_count:
        print(f"  [注意] 实际获取 {len(symbols)} 个，少于请求的 {symbol_count}")
    print(f"  预加载K线: {kline_count} 个")
    print(f"  交易状态: 关闭 (输入'交易'开启)")
    print(f"================")

    # Telegram 命令接收
    if fw.telegram_bot:
        def bot_cmd_handler(chat_id, text):
            return fw._handle_bot_command(text)

        fw.telegram_bot.start_command_polling(bot_cmd_handler)
        print(" [OK] Telegram 命令接收已启动")
    else:
        print(" [警告] Telegram 未连接，命令接收不可用")

    print("\n" + "=" * 60)
    print("命令: 状态/持仓/待处理/价格 X/平仓 X/全平/同步/交易/关交易/停止/退出")
    print("=" * 60 + "\n")

    fw._running = True
    while fw._running:
        try:
            cmd = input(">> ").strip().lower()
            
            # 中文命令映射
            cmd_map = {
                '退出': 'exit', 'quit': 'exit', 'q': 'exit',
                '状态': 'status', 'status': 'status',
                '持仓': 'positions', 'positions': 'positions',
                '交易记录': 'trades', 'trades': 'trades',
                '图表': 'chart', 'chart': 'chart',
                '待处理': 'pending', 'pending': 'pending',
                '价格': 'price', 'price': 'price',
                '平仓': 'close', 'close': 'close',
                '全平': 'closeall', 'closeall': 'closeall',
                '同步': 'sync', 'sync': 'sync',
                '交易': 'trade', 'trade': 'trade',
                '关交易': 'tradestop', '关闭交易': 'tradestop',
                '停止': 'stop', 'stop': 'stop',
                '帮助': 'help', 'help': 'help',
            }
            
            # 提取命令和参数
            parts = cmd.split(maxsplit=1)
            if not parts:
                continue  # 空输入，跳过
            base_cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ''
            mapped = cmd_map.get(base_cmd, base_cmd)

            if mapped == 'exit':
                print("退出...")
                fw.stop()
                break
            elif mapped == 'help':
                print("状态/持仓/交易记录/图表 X/待处理/价格 X/平仓 X/全平/同步/交易/关交易/停止/退出")
                print("  交易记录 - 查看所有历史交易")
                print("  图表 X - 生成指定序号交易的K线图 (需先查看交易记录)")
                print("  关交易   - 关闭交易（监控继续运行）")
            elif mapped == 'status':
                s = fw.get_status()
                print(f" 交易状态: {'开启' if fw._trading_active else '关闭(输入交易开启)'}")
                # 获取连接状态和时间偏移
                try:
                    client = fw.client
                    session = getattr(client, 'session', None)
                    base_url = getattr(client, 'base_url', None)
                    if session and base_url:
                        session.get(f"{base_url}/fapi/v1/ping", timeout=3)
                    conn_status = "正常"
                except Exception:
                    import logging
                    logging.debug("连接检查失败")
                    conn_status = "断开"

                time_offset = getattr(fw.client, '_time_offset', 0) or 0
                sync_time = getattr(fw.client, '_last_time_sync', 0) or 0
                import time
                sync_ago = int(time.time() - sync_time) if sync_time > 0 else 0
                
                print(f"\n=== 系统状态 ===")
                print(f"  连接状态: {conn_status}")
                print(f"  时间偏移: {time_offset}ms (上次同步: {sync_ago}秒前)")
                print(f"  交易状态: {'开启' if fw._trading_active else '关闭(仅监控)'}")
                print(f"  持仓数量: {s['positions']}")
                print(f"  日盈亏: {s['daily_pnl']} USDT")
                print(f"  账户余额: {s['balance']} USDT")
                print(f"  监控币种: {s.get('symbols', 'N/A')}")
                print(f"================\n")
            elif mapped == 'positions':
                fw.sync_positions()
                ps = fw.get_positions()
                if not ps:
                    print(" 无持仓")
                else:
                    for p in ps:
                        side_str = p.side.value if hasattr(p.side, 'value') else str(p.side)
                        pnl = getattr(p, 'unrealized_pnl', 0)
                        pnl_pct = getattr(p, 'pnl_pct', 0)
                        print(f" {p.symbol}: {side_str} {abs(p.quantity):.4f} @{p.entry_price:.4f} PnL: {pnl:.2f} ({pnl_pct:.1f}%)")
            elif mapped == 'trades':
                _show_trades()
            elif mapped == 'chart' and arg:
                try:
                    _show_chart(int(arg.strip()))
                except ValueError:
                    print(" 图表序号无效，请输入数字")
            elif mapped == 'chart':
                print(" 请指定交易序号，例: 图表 1")
                print(" 可用交易序号请先执行: 交易记录")
            elif mapped == 'pending':
                print(" 无" if not (fw.trader and hasattr(fw.trader, 'pending_adds')) else " 有")
            elif mapped == 'price' and arg:
                try:
                    price = getattr(fw.client, 'get_price', lambda x: None)(arg.upper())
                    print(f" {arg.upper()}: {price if price else 'N/A'}")
                except Exception as e: print(f" 错误: {e}")
            elif mapped == 'close' and arg:
                fw.close_position(arg.upper())
            elif mapped == 'closeall': fw.close_all()
            elif mapped == 'sync': fw.sync_positions()
            elif mapped == 'trade':
                fw._trading_active = True
                if fw.scheduler:
                    fw.scheduler.enable_trading()
                print(" [交易] 已开启，策略将执行交易")
            elif mapped == 'tradestop':
                fw._trading_active = False
                if fw.scheduler:
                    fw.scheduler.disable_trading()
                print(" [交易] 已关闭，监控继续运行")
            elif mapped == 'stop':
                fw._trading_active = False
                if fw.scheduler:
                    fw.scheduler.disable_trading()
                fw.stop_monitor()
            else:
                print(" 未知命令，输入'帮助'查看")
        except Exception as e:
            print(f" 命令执行错误: {e}")


def run_backtest():
    """运行回测 - 调用 run_final.py（权威版）"""
    import subprocess
    import os

    print("\n" + "=" * 60)
    print("15mTupo 回测")
    print("=" * 60)
    print(" [1] 全39币 + 复利模拟（推荐参数）")
    print(" [2] 验证集（21币）")
    print(" [3] 自定义币种")
    print(" [0] 返回")
    print("=" * 60)
    c = input("请选择: ").strip()

    env = os.environ.copy()
    if c == '1':
        env['BASE_MARGIN_PCT'] = '0.20'
        env['DD_T1'] = '0.35'
        env['DD_T2'] = '0.55'
    elif c == '2':
        env['YZDIR'] = '1'
    elif c == '3':
        syms = input("输入币种（逗号分隔，如 BTCUSDT,ETHUSDT）: ").strip()
        if syms:
            env['SAVE_SAMPLE'] = syms
        else:
            print("未输入币种，返回")
            return
    else:
        return

    print("\n开始回测...")
    result = subprocess.run(
        [sys.executable, 'framework/backtest/run_final.py'],
        env=env, cwd=os.path.dirname(__file__) or '.')
    if result.returncode != 0:
        print(f"\n回测异常退出 (code={result.returncode})")
    else:
        print("\n回测完成，结果保存到 logs/all_trades.csv")
    input("\n按 Enter 返回菜单...")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['live', 'testnet', 'backtest'])
    p.add_argument('--strategy', choices=['15mTupo', 'v23', 'hf', 'both'])
    p.add_argument('--check', action='store_true')
    p.add_argument('--deps', action='store_true')
    p.add_argument('--clean', action='store_true')
    args = p.parse_args()
    
    if args.check: check_config(); return
    if args.deps: check_dependencies(); return
    if args.clean: clean_files(); return
    
    if args.mode:
        if args.mode == 'backtest':
            run_backtest()
        else:
            run_mode(testnet=(args.mode=='testnet'), load_v23=args.strategy in ['15mTupo','v23','both'] or args.strategy is None, load_hf=args.strategy in ['hf','both'] or args.strategy is None)
        return
    
    # 菜单
    while True:
        print("\n" + "=" * 60)
        print("BNFRich v2.0 - 模块化交易框架")
        print("=" * 60)
        print(" [1] 实盘模式")
        print(" [2] 测试网模式")
        print(" [3] 回测模式")
        print(" [4] 检查配置")
        print(" [5] 检查依赖")
        print(" [6] 清理缓存")
        print(" [7] 帮助")
        print(" [0] 退出")
        print("=" * 60)
        c = input("\n请选择: ").strip()
        try:
            if c == '0': break
            elif c == '1': run_mode(False, True, True); break
            elif c == '2': run_mode(True, True, True); break
            elif c == '3': run_backtest()
            elif c == '4': check_config()
            elif c == '5': check_dependencies()
            elif c == '6': clean_files()
            elif c == '7': show_help()
            else: print("无效")
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()
            input("\n按回车键退出...")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[致命错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保窗口不立即关闭
        input("\n按回车键退出...")