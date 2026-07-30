# -*- coding: utf-8 -*-
"""
15mTupo策略实盘监控器

功能：
1. 加载历史15m/5m K线数据
2. 实时价格累计到当前15m K线
3. 使用TupoStrategy进行信号检测（与回测一致）
4. 支持信号回调连接到交易执行

使用方式：
    from strategies.15mTupo.realtime_monitor import RealtimeMonitor
    from trading.binance_client import BinanceClient
    
    client = BinanceClient(mode='testnet')
    monitor = RealtimeMonitor(client)
    monitor.set_symbols(['BTCUSDT', 'ETHUSDT'])
    
    # 添加信号回调（连接到交易执行）
    def on_signal(symbol, signal):
        print(f"信号: {symbol} {signal}")
    monitor.add_signal_callback(on_signal)
    
    monitor.load_history()
    monitor.start()
"""

import os
import time
import threading
from typing import Dict, List, Optional, Any, Callable

from strategies.base.realtime_monitor import RealtimeMonitorBase
import importlib
TupoStrategy = importlib.import_module("strategies.15mTupo.strategy").TupoStrategy
from framework.strategy.base import Signal, SignalType
from framework.business.kline_manager import KlineBar
from utils.logger import Logger


class TupoRealtimeMonitor(RealtimeMonitorBase):
    """
    15mTupo策略实盘监控器
    
    _env_lock: 保护 os.environ 的并发访问，防止 check_signal 多线程竞争
    
    特点：
    - 使用15分钟K线级别判断趋势和信号
    - 实时tick累计到当前15m K线
    - 与回测共用TupoStrategy，确保信号逻辑一致
    - 支持WebSocket实时价格推送
    """

    def __init__(
        self,
        client: Any,
        config: Any = None,
        interval: str = '15m',
        history_count: int = 100,
        testnet: bool = True
    ):
        """
        初始化15mTupo监控器
        
        Args:
            client: BinanceClient实例
            config: 策略配置（Settings15mTupo实例）
            interval: K线周期，默认15m
            history_count: 历史K线数量
            testnet: 是否使用测试网WebSocket
        """
        # 创建TupoStrategy实例（与回测共用同一个策略逻辑）
        self.strategy = TupoStrategy()
        
        # 调用父类初始化
        super().__init__(
            client=client,
            strategy_core=None,
            config=config,
            interval=interval,
            history_count=history_count
        )
        
        self.logger = Logger.get_logger('TupoRealtimeMonitor')
        self._testnet = testnet
        self._env_lock = threading.Lock()  # 保护 os.environ 并发访问
        
        # WebSocket实例
        self._ws = None
        self._ws_running = False
        
    def load_history(self) -> bool:
        """
        加载历史K线数据
        
        为每个监控币种加载历史15m K线，用于：
        1. 计算ADX、DI等指标
        2. 判断趋势和支撑阻力位
        
        Returns:
            是否成功
        """
        self.logger.info(f"加载历史K线数据 ({self.interval})...")
        
        success_count = 0
        failed_symbols = []
        
        for symbol in self.symbols:
            try:
                # 从API获取历史K线
                raw_klines = self.client.get_klines(symbol, self.interval, self.history_count)
                
                if not raw_klines:
                    self.logger.warning(f"{symbol}: 无法获取历史K线")
                    failed_symbols.append(symbol)
                    continue
                
                # 处理K线数据
                processed = self._process_raw_klines(raw_klines)
                
                # 存入缓存
                with self._cache_lock:
                    self._kline_cache[symbol]['history'] = processed
                    self._kline_cache[symbol]['current'] = None
                
                self.logger.info(f"{symbol}: 加载{len(processed)}根{self.interval}K线完成")
                
                # 加载5m历史K线（TupoStrategy需要）
                raw_klines_5m = self.client.get_klines(symbol, '5m', self.history_count)
                if raw_klines_5m:
                    processed_5m = self._process_raw_klines(raw_klines_5m)
                    with self._cache_lock:
                        self._kline_cache[symbol]['history_5m'] = processed_5m
                    self.logger.info(f"{symbol}: 加载{len(processed_5m)}根5m K线完成")
                else:
                    self.logger.warning(f"{symbol}: 无法获取5m历史K线")
                
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"{symbol}: 加载K线失败: {e}")
                failed_symbols.append(symbol)
        
        self.logger.info(f"历史K线加载完成: 成功{success_count}个, 失败{len(failed_symbols)}个")
        if failed_symbols:
            self.logger.warning(f"失败币种: {', '.join(failed_symbols)}")
        return success_count > 0
    
    def check_signal(self, symbol: str) -> Optional[Dict]:
        """
        检测交易信号
        
        使用TupoStrategy.analyze()进行信号检测，与回测逻辑一致。
        
        Args:
            symbol: 币种
            
        Returns:
            信号字典或None
        """
        with self._cache_lock:
            cache = self._kline_cache.get(symbol)
            if not cache:
                return None
            
            history = cache.get('history', [])
            current = cache.get('current')
            
            # 需要至少50根K线
            total_bars = len(history) + (1 if current and current.get('tick_count', 0) > 0 else 0)
            if total_bars < 50:
                self.logger.debug(f"{symbol}: K线数据不足 ({total_bars}/50)")
                return None

        # 构建K线字典用于TupoStrategy分析
        klines = self._build_klines_for_analyze(symbol)

        bars_15m = klines.get('15m', [])
        if len(bars_15m) < 50:
            self.logger.debug(f"{symbol}: 15m分析数据不足 ({len(bars_15m)}/50)")
            return None

        # 确保AUTO_TRADE已设置，使TupoStrategy返回信号
        with self._env_lock:
            old_auto_trade = os.environ.get('AUTO_TRADE', '')
            os.environ['AUTO_TRADE'] = '1'
        try:
            signal = self.strategy.analyze(symbol, klines)

            if signal and signal.signal_type != SignalType.NONE:
                direction = 'long' if signal.signal_type == SignalType.LONG else 'short'
                self.logger.info(f"【信号检测】{symbol}: {signal.signal_type.value} {direction} "
                               f"开仓价:{signal.price:.4f} 止损:{signal.stop_loss:.4f} "
                               f"置信度:{signal.confidence:.0%} 原因:{signal.reason[:50]}...")
                return {
                    'symbol': symbol,
                    'signal_type': signal.signal_type.value,
                    'direction': direction,
                    'entry_price': signal.price,
                    'stop_loss': signal.stop_loss,
                    'is_batch_entry': False,
                    'confidence': signal.confidence,
                    'reason': signal.reason,
                    'leverage': signal.leverage,
                }
            else:
                self.logger.debug(f"{symbol}: 无信号 (类型:{signal.signal_type if signal else 'None'})")

        except Exception as e:
            self.logger.error(f"{symbol}: 信号检测失败: {e}")
        finally:
            with self._env_lock:
                if old_auto_trade:
                    os.environ['AUTO_TRADE'] = old_auto_trade
                else:
                    os.environ.pop('AUTO_TRADE', None)

        return None
    
    def _build_klines_for_analyze(self, symbol: str) -> Dict[str, List[KlineBar]]:
        """
        构建用于TupoStrategy分析的K线数据
        
        覆盖父类方法，返回 {interval: [KlineBar, ...]} 格式。
        """
        result = {}
        with self._cache_lock:
            cache = self._kline_cache.get(symbol)
            if not cache:
                return result
            
            # 构建15m K线列表
            bars_15m = []
            for d in cache.get('history', []):
                bars_15m.append(KlineBar(
                    timestamp=int(d['timestamp']),
                    open=float(d['open']),
                    high=float(d['high']),
                    low=float(d['low']),
                    close=float(d['close']),
                    volume=float(d.get('volume', 0)),
                ))
            current = cache.get('current')
            if current and current.get('tick_count', 0) > 0:
                bars_15m.append(KlineBar(
                    timestamp=int(current['timestamp']),
                    open=float(current['open']),
                    high=float(current['high']),
                    low=float(current['low']),
                    close=float(current['close']),
                    volume=float(current.get('volume', 0)),
                ))
            result['15m'] = bars_15m
            
            # 构建5m K线列表
            bars_5m = []
            for d in cache.get('history_5m', []):
                bars_5m.append(KlineBar(
                    timestamp=int(d['timestamp']),
                    open=float(d['open']),
                    high=float(d['high']),
                    low=float(d['low']),
                    close=float(d['close']),
                    volume=float(d.get('volume', 0)),
                ))
            result['5m'] = bars_5m
        
        return result
    
    def on_price_update(self, symbol: str, price: float, timestamp: int = None):
        """
        价格更新回调
        
        可从WebSocket价格推送调用此方法。
        
        Args:
            symbol: 币种
            price: 最新价格
            timestamp: 时间戳（毫秒）
        """
        self.on_tick(symbol, price, timestamp)
    
    def start_websocket(self):
        """
        启动WebSocket实时价格推送
        
        使用共用的 WebSocketManager，避免多策略重复连接
        """
        if self._ws_running:
            self.logger.warning("WebSocket已在运行")
            return
        
        try:
            from framework.shared import get_websocket_manager
            
            # 获取共用的 WebSocket 管理器
            self._ws = get_websocket_manager(testnet=self._testnet)
            
            # 订阅所有监控币种的价格
            self._ws.subscribe(self.symbols, self._on_ws_price)
            
            # 如果 WebSocket 未启动，启动它
            if not self._ws.is_running():
                self._ws.start()
            
            self._ws_running = True
            self.logger.info(f"已订阅 {len(self.symbols)} 个币种的实时价格")
            
        except Exception as e:
            self.logger.error(f"WebSocket订阅失败: {e}")
    
    def stop_websocket(self):
        """停止WebSocket订阅（不停止共用连接）"""
        if self._ws:
            self._ws.unsubscribe(self.symbols, self._on_ws_price)
            self._ws_running = False
            self.logger.info("已取消WebSocket订阅")
    
    def _on_ws_price(self, symbol: str, price: float):
        """
        WebSocket价格回调
        
        Args:
            symbol: 币种
            price: 最新价格
        """
        self.on_price_update(symbol, price)

    def get_kline_status(self, symbol: str) -> Dict:
        """
        获取K线状态
        
        Returns:
            包含历史长度、当前K线进度等信息的字典
        """
        with self._cache_lock:
            cache = self._kline_cache.get(symbol, {})
            history = cache.get('history', [])
            current = cache.get('current')
            
            status = {
                'symbol': symbol,
                'interval': self.interval,
                'history_count': len(history),
                'has_current': current is not None,
                'tick_count': current.get('tick_count', 0) if current else 0,
            }
            
            if current:
                # 计算K线进度（基于tick数量估算）
                expected_ticks = 100  # 假设每根K线约100个tick
                status['progress'] = min(current.get('tick_count', 0) / expected_ticks, 1.0)
            
            return status
    
    def _emit_signal(self, symbol: str, signal: Dict):
        """
        发出信号（重写父类方法以添加15mTupo特定处理）
        """
        # 添加额外信息
        signal['strategy'] = '15mTupo'
        signal['interval'] = self.interval
        signal['timestamp'] = int(time.time() * 1000)
        
        super()._emit_signal(symbol, signal)
