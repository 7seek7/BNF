# -*- coding: utf-8 -*-
"""
K线管理器

统一管理K线数据的获取、缓存和分发

特性：
1. 多周期K线缓存
2. 增量更新
3. WebSocket实时推送
4. 按需加载（策略注册周期后自动订阅）
5. 技术指标预计算
"""

import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from framework.core.logger import get_logger, track_performance

logger = get_logger('kline_manager')


@dataclass
class KlineBar:
    """单根K线数据"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_volume: float = 0.0
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000)
    
    @property
    def body(self) -> float:
        """K线实体"""
        return abs(self.close - self.open)
    
    @property
    def range(self) -> float:
        """K线振幅"""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """是否阳线"""
        return self.close > self.open
    
    @property
    def close_position(self) -> float:
        """收盘位置（0-1，越接近1越靠近最高价）"""
        if self.high == self.low:
            return 0.5
        return (self.close - self.low) / (self.high - self.low)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }


@dataclass
class KlineCache:
    """K线缓存"""
    symbol: str
    interval: str
    bars: List[KlineBar] = field(default_factory=list)
    last_update: float = 0.0
    is_complete: bool = False  # 当前K线是否完结
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    @property
    def current_bar(self) -> Optional[KlineBar]:
        """获取当前（最新）K线"""
        with self._lock:
            return self.bars[-1] if self.bars else None
    
    @property
    def prev_bar(self) -> Optional[KlineBar]:
        """获取上一根完整K线"""
        with self._lock:
            return self.bars[-2] if len(self.bars) >= 2 else None
    
    def get_bars(self, count: int) -> List[KlineBar]:
        """获取最近N根K线（线程安全）"""
        with self._lock:
            return self.bars[-count:].copy() if len(self.bars) >= count else self.bars.copy()
    
    def get_bars_since(self, since: int) -> List[KlineBar]:
        """获取指定时间以来的K线"""
        return [b for b in self.bars if b.timestamp >= since]
    
    def update(self, bar: KlineBar):
        """更新K线"""
        if not self.bars:
            self.bars.append(bar)
            self.is_complete = False
            return
            
        last = self.bars[-1]
        
        # 同一根K线更新
        if bar.timestamp == last.timestamp:
            self.bars[-1] = bar
            self.is_complete = False
        else:
            # 新K线：前一根已完结，当前根未完结
            self.is_complete = False
            self.bars.append(bar)
            # 限制缓存大小（保留足够历史用于长期指标）
            if len(self.bars) > 5000:
                self.bars = self.bars[-4000:]
                
        self.last_update = time.time()


class KlineManager:
    """
    K线管理器
    
    统一管理所有币种的K线数据
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, client=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, client=None):
        if self._initialized:
            return
            
        self._client = client
        self._caches: Dict[str, Dict[str, KlineCache]] = defaultdict(dict)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # 策略订阅的周期
        self._strategy_intervals: Dict[str, List[str]] = defaultdict(list)
        
        # WebSocket连接状态
        self._ws_connected = False
        self._ws_streams: Dict[str, Any] = {}
        
        self._initialized = True
        logger.info("K线管理器已初始化")
        
    def set_client(self, client):
        """设置API客户端"""
        self._client = client
        
    def register_strategy(self, strategy_name: str, intervals: List[str]):
        """
        策略注册需要的K线周期
        
        Args:
            strategy_name: 策略名称
            intervals: K线周期列表，如 ['5m', '1h', '4h']
        """
        self._strategy_intervals[strategy_name] = intervals
        logger.info(f"策略注册K线周期: {strategy_name} -> {intervals}")
        
    def unregister_strategy(self, strategy_name: str):
        """取消策略注册"""
        if strategy_name in self._strategy_intervals:
            del self._strategy_intervals[strategy_name]
            logger.info(f"策略取消注册: {strategy_name}")
            
    @track_performance('load_history')
    def load_history(self, symbol: str, interval: str, limit: int = 500) -> bool:
        """
        加载历史K线数据
        
        Args:
            symbol: 币种
            interval: 周期
            limit: 数量
            
        Returns:
            是否成功
        """
        if not self._client:
            logger.error("未设置API客户端")
            return False
            
        try:
            # 从API获取历史数据
            klines = self._client.get_klines(symbol, interval, limit=limit)
            
            if not klines:
                logger.warning(f"获取K线失败: {symbol} {interval}")
                return False
                
            # 转换为KlineBar
            bars = []
            for k in klines:
                bar = KlineBar(
                    timestamp=k[0],
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    quote_volume=float(k[7]) if len(k) > 7 else 0,
                    trades=int(k[8]) if len(k) > 8 else 0,
                    taker_buy_volume=float(k[9]) if len(k) > 9 else 0,
                )
                bars.append(bar)
                
            # 存入缓存
            cache = KlineCache(symbol=symbol, interval=interval, bars=bars)
            with self._lock:
                self._caches[symbol][interval] = cache
                
            logger.info(
                f"K线历史加载完成: {symbol} {interval} 共{len(bars)}根",
                extra={'context': {'symbol': symbol, 'interval': interval, 'count': len(bars)}}
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"加载K线历史失败: {symbol} {interval}, {e}")
            return False
            
    def load_all_for_strategy(self, strategy_name: str, symbols: List[str]) -> Dict[str, bool]:
        """
        为策略加载所有需要的K线
        
        Args:
            strategy_name: 策略名称
            symbols: 币种列表
            
        Returns:
            加载结果 {symbol: success}
        """
        intervals = self._strategy_intervals.get(strategy_name, [])
        if not intervals:
            logger.warning(f"策略未注册K线周期: {strategy_name}")
            return {}
            
        results = {}
        for symbol in symbols:
            success = True
            for interval in intervals:
                if not self.load_history(symbol, interval):
                    success = False
                time.sleep(0.05)  # 避免限流
            results[symbol] = success
            
        return results
        
    def get_cache(self, symbol: str, interval: str) -> Optional[KlineCache]:
        """获取K线缓存"""
        with self._lock:
            return self._caches.get(symbol, {}).get(interval)
            
    def get_bars(self, symbol: str, interval: str, count: int = 100) -> List[KlineBar]:
        """
        获取K线数据
        
        Args:
            symbol: 币种
            interval: 周期
            count: 数量
            
        Returns:
            K线列表
        """
        cache = self.get_cache(symbol, interval)
        if cache:
            return cache.get_bars(count)
        return []
        
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 优先从5m或1m获取
        for interval in ['1m', '5m', '15m']:
            cache = self.get_cache(symbol, interval)
            if cache and cache.current_bar:
                return cache.current_bar.close
        return None
        
    def update_bar(self, symbol: str, interval: str, bar: KlineBar):
        """
        更新K线（从WebSocket调用）

        Args:
            symbol: 币种
            interval: 周期
            bar: K线数据
        """
        with self._lock:
            # 确保缓存存在（避免重复创建）
            if symbol not in self._caches:
                self._caches[symbol] = {}

            if interval not in self._caches[symbol]:
                self._caches[symbol][interval] = KlineCache(symbol=symbol, interval=interval)

            cache = self._caches[symbol][interval]
            if cache:
                cache.update(bar)
            
            # 在锁内快照订阅者，避免竞态
            subscribers = list(self._subscribers.get(symbol, []))

        # 通知订阅者（锁外执行，避免死锁）
        for callback in subscribers:
            try:
                callback(symbol, interval, bar)
            except Exception as e:
                logger.exception(f"K线订阅回调失败: {callback.__name__}")
        
    def subscribe(self, symbol: str, callback: Callable[[str, str, KlineBar], None]):
        """
        订阅K线更新
        
        Args:
            symbol: 币种
            callback: 回调函数 callback(symbol, interval, bar)
        """
        with self._lock:
            if symbol not in self._subscribers:
                self._subscribers[symbol] = []
            self._subscribers[symbol].append(callback)
            
    def unsubscribe(self, symbol: str, callback: Callable):
        """取消订阅"""
        with self._lock:
            if symbol in self._subscribers:
                try:
                    self._subscribers[symbol].remove(callback)
                except ValueError:
                    logger.debug(f"订阅者不在列表中，跳过取消订阅")
                    
    def get_all_symbols(self) -> List[str]:
        """获取所有已缓存的币种"""
        with self._lock:
            return list(self._caches.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = {
                'symbols': len(self._caches),
                'total_caches': sum(len(v) for v in self._caches.values()),
                'strategy_subscriptions': dict(self._strategy_intervals),
                'subscriber_count': sum(len(v) for v in self._subscribers.values()),
            }
        return stats
        
    def poll_latest_bars(self, symbol: str, interval: str, limit: int = 5):
        """轮询最新K线并更新缓存（无WebSocket时使用）"""
        if not self._client:
            return
        try:
            klines = self._client.get_klines(symbol, interval, limit=limit)
            if not klines:
                return
            for k in klines:
                bar = KlineBar(
                    timestamp=k[0],
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    quote_volume=float(k[7]) if len(k) > 7 else 0,
                    trades=int(k[8]) if len(k) > 8 else 0,
                    taker_buy_volume=float(k[9]) if len(k) > 9 else 0,
                )
                self.update_bar(symbol, interval, bar)
        except Exception as e:
            logger.debug(f"轮询K线失败: {symbol} {interval}, {e}")

    def clear(self, symbol: str = None):
        """清除缓存"""
        with self._lock:
            if symbol:
                self._caches.pop(symbol, None)
            else:
                self._caches.clear()
        logger.info(f"K线缓存已清除: {symbol or '全部'}")


# 便捷函数
_kline_manager: Optional[KlineManager] = None


def get_kline_manager(client=None) -> KlineManager:
    """获取K线管理器单例"""
    global _kline_manager
    if _kline_manager is None:
        _kline_manager = KlineManager(client)
    elif client:
        _kline_manager.set_client(client)
    return _kline_manager
