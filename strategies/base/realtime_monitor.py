# -*- coding: utf-8 -*-
"""
策略实盘监控器基类

每个策略可以有自己的实盘监控器，独立管理：
- 历史K线缓存
- 实时价格累计
- 信号检测

与回测共用同一个策略核心（StrategyCore），确保逻辑一致。
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from utils.logger import Logger


@dataclass
class KlineBar:
    """K线数据"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class RealtimeMonitorBase(ABC):
    """
    策略实盘监控器基类
    
    功能：
    1. 管理历史K线缓存（启动时加载，周期结束时更新）
    2. 实时tick累计成当前K线
    3. 调用策略核心进行信号检测
    4. 信号回调处理
    
    使用流程：
    1. 初始化：创建策略核心实例
    2. load_history(): 加载历史K线
    3. start(): 启动WebSocket订阅
    4. on_tick(): 实时价格更新，累计到当前K线
    5. K线周期结束：保存当前K线到历史，创建新K线
    6. check_signal(): 检测信号（可随时调用）
    """
    
    def __init__(
        self,
        client: Any,
        strategy_core: Any,
        config: Any,
        interval: str = '15m',
        history_count: int = 100
    ):
        """
        初始化监控器
        
        Args:
            client: BinanceClient实例
            strategy_core: 策略核心实例（如V23StrategyCore）
            config: 策略配置
            interval: K线周期
            history_count: 历史K线数量
        """
        self.client = client
        self.strategy_core = strategy_core
        self.config = config
        self.interval = interval
        self.history_count = history_count
        
        self.logger = Logger.get_logger(self.__class__.__name__)
        
        # 监控币种
        self.symbols: List[str] = []
        
        # K线缓存: {symbol: {'history': [...], 'current': KlineBar}}
        self._kline_cache: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'history': [],
            'current': None
        })
        
        # 锁
        self._cache_lock = threading.RLock()
        
        # 运行状态
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 信号回调
        self._signal_callbacks: List[Callable] = []

        # 冷却机制（避免重复信号）
        self._signal_cooldown: Dict[str, float] = {}
        try:
            from config.settings import Settings as _Settings
            self._cooldown_seconds = getattr(_Settings, 'V23_SIGNAL_COOLDOWN', 300)
        except Exception:
            self._cooldown_seconds = 300
        self._check_interval = getattr(config, 'SIGNAL_CHECK_INTERVAL', 2) if config else 2

        # K线周期（毫秒）
        self._interval_ms = self._get_interval_ms(interval)
        
    def _get_interval_ms(self, interval: str) -> int:
        """获取周期对应的毫秒数"""
        mapping = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return mapping.get(interval, 15 * 60 * 1000)
    
    def set_symbols(self, symbols: List[str]):
        """设置监控币种"""
        self.symbols = symbols
        self.logger.info(f"设置监控币种: {len(symbols)}个")
    
    def add_signal_callback(self, callback: Callable):
        """添加信号回调函数"""
        self._signal_callbacks.append(callback)
    
    # ==================== 抽象方法 ====================
    
    @abstractmethod
    def load_history(self) -> bool:
        """
        加载历史K线数据
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def check_signal(self, symbol: str) -> Optional[Any]:
        """
        检测交易信号
        
        Args:
            symbol: 币种
            
        Returns:
            Signal 或 None
        """
        pass
    
    # ==================== K线管理 ====================
    
    def _process_raw_klines(self, raw_klines: List) -> List[Dict]:
        """处理原始K线数据"""
        processed = []
        for k in raw_klines:
            processed.append({
                'timestamp': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
            })
        return processed
    
    def _get_current_period_start(self, timestamp: int) -> int:
        """获取当前周期的开始时间"""
        return (timestamp // self._interval_ms) * self._interval_ms
    
    def _should_create_new_kline(self, symbol: str, timestamp: int) -> bool:
        """检查是否需要创建新K线"""
        with self._cache_lock:
            cache = self._kline_cache[symbol]
            current = cache.get('current')
            
            if current is None:
                return True
            
            current_period = current['timestamp'] // self._interval_ms
            new_period = timestamp // self._interval_ms
            
            return new_period > current_period
    
    def _save_current_to_history(self, symbol: str):
        """保存当前K线到历史"""
        with self._cache_lock:
            cache = self._kline_cache[symbol]
            current = cache.get('current')
            
            if current and current.get('tick_count', 0) > 0:
                # 添加到历史
                cache['history'].append(current)
                
                # 保持历史长度
                if len(cache['history']) > self.history_count:
                    cache['history'] = cache['history'][-self.history_count:]
                
                self.logger.debug(f"{symbol}: K线周期结束，保存到历史，当前历史长度: {len(cache['history'])}")
    
    def _create_new_kline(self, symbol: str, timestamp: int, price: float):
        """创建新的当前K线"""
        period_start = self._get_current_period_start(timestamp)
        
        with self._cache_lock:
            self._kline_cache[symbol]['current'] = {
                'timestamp': period_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0.0,
                'tick_count': 0,
            }
    
    # ==================== 实时数据处理 ====================
    
    def on_tick(self, symbol: str, price: float, timestamp: int = None, volume: float = 0):
        """
        处理实时价格更新
        
        Args:
            symbol: 币种
            price: 价格
            timestamp: 时间戳（毫秒），默认当前时间
            volume: 成交量
        """
        if not self._running:
            return
        
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        with self._cache_lock:
            cache = self._kline_cache[symbol]
            
            # 检查是否需要新建K线（周期切换）
            if self._should_create_new_kline(symbol, timestamp):
                # 保存旧K线到历史
                self._save_current_to_history(symbol)
                # 创建新K线
                self._create_new_kline(symbol, timestamp, price)
                cache = self._kline_cache[symbol]
            
            # 更新当前K线
            current = cache.get('current')
            if current is None:
                self._create_new_kline(symbol, timestamp, price)
                current = cache['current']
            
            # 更新价格
            current['high'] = max(current['high'], price)
            current['low'] = min(current['low'], price)
            current['close'] = price
            current['volume'] += volume
            current['tick_count'] = current.get('tick_count', 0) + 1
    
    # ==================== 信号处理 ====================
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """检查是否在冷却期"""
        with self._cache_lock:
            last_signal_time = self._signal_cooldown.get(symbol, 0)
        return (time.time() - last_signal_time) < self._cooldown_seconds
    
    def _update_cooldown(self, symbol: str):
        """更新冷却时间"""
        with self._cache_lock:
            self._signal_cooldown[symbol] = time.time()
    
    def _emit_signal(self, symbol: str, signal: Any):
        """发出信号"""
        if self._is_in_cooldown(symbol):
            self.logger.debug(f"{symbol}: 信号冷却中，跳过")
            return
        
        self._update_cooldown(symbol)
        
        self.logger.info(f"[信号] {symbol}: {signal.signal_type.value if hasattr(signal, 'value') else signal}")
        
        # 调用回调
        for callback in self._signal_callbacks:
            try:
                callback(symbol, signal)
            except Exception as e:
                self.logger.error(f"信号回调执行失败: {e}")
    
    # ==================== 构建K线数据 ====================
    
    def _build_klines_for_analyze(self, symbol: str) -> List[Dict]:
        """
        构建用于策略分析的K线数据
        
        返回: 历史 + 当前（如果存在）
        """
        with self._cache_lock:
            cache = self._kline_cache[symbol]
            history = cache.get('history', []).copy()
            current = cache.get('current')
            
            if current and current.get('tick_count', 0) > 0:
                # 包含正在形成的K线
                history.append({
                    'timestamp': current['timestamp'],
                    'open': current['open'],
                    'high': current['high'],
                    'low': current['low'],
                    'close': current['close'],
                    'volume': current['volume'],
                })
            
            return history
    
    # ==================== 生命周期 ====================
    
    def start(self):
        """启动监控"""
        if self._running:
            self.logger.warning("监控器已在运行")
            return
        
        self._running = True
        
        # 启动监控循环
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info(f"监控器已启动: {len(self.symbols)}个币种, 周期={self.interval}")
    
    def stop(self):
        """停止监控"""
        self._running = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("监控器已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        self.logger.info("监控循环启动")
        
        while self._running:
            try:
                for symbol in self.symbols:
                    if self._is_in_cooldown(symbol):
                        continue
                    
                    signal = self.check_signal(symbol)
                    if signal:
                        self._emit_signal(symbol, signal)
                
                time.sleep(self._check_interval)  # 使用配置的检查间隔
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(5)
        
        self.logger.info("监控循环结束")
    
    def get_status(self) -> Dict:
        """获取监控状态"""
        with self._cache_lock:
            status = {
                'running': self._running,
                'symbols': len(self.symbols),
                'interval': self.interval,
                'cache_info': {}
            }
            
            for symbol in self.symbols:
                cache = self._kline_cache.get(symbol, {})
                status['cache_info'][symbol] = {
                    'history_count': len(cache.get('history', [])),
                    'has_current': cache.get('current') is not None,
                }
            
            return status
