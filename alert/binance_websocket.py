# -*- coding: utf-8 -*-
"""
币安WebSocket客户端 - 实时价格推送
支持：自动重连、心跳、多币种订阅
"""
import json
import time
import threading
import websocket
from typing import Dict, List, Callable, Optional
from datetime import datetime
from collections import defaultdict


class BinanceWebSocket:
    """
    币安WebSocket客户端
    
    特性：
    - 自动重连机制
    - 心跳保活
    - 多币种订阅
    - 回调函数支持
    """
    
    WS_URL_LIVE = "wss://fstream.binance.com/ws"
    WS_URL_TESTNET = "wss://stream.binancefuture.com/ws"
    
    def __init__(self, testnet: bool = False, on_message: Callable = None):
        self.testnet = testnet
        self.ws_url = self.WS_URL_TESTNET if testnet else self.WS_URL_LIVE
        self.on_message = on_message
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.running = False
        self._ws_lock = threading.Lock()  # 连接/重连锁
        
        self.subscriptions: List[str] = []
        self._ticker_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._mark_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        self._last_prices: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}
        self._price_lock = threading.Lock()  # 价格缓存锁
        
        self._reconnect_count = 0
        self._max_reconnect = 10
        
        # 缓存清理相关
        self._cleanup_interval = 300  # 5分钟清理一次
        self._last_cleanup = time.time()
        self._stale_timeout = 3600  # 1小时未更新的视为过期
        
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [{level}] WebSocket: {msg}")
    
    def subscribe_ticker(self, symbols: List[str], callback: Callable = None):
        """
        订阅实时价格
        
        Args:
            symbols: 币种列表 ['BTCUSDT', 'ETHUSDT']
            callback: 回调函数 callback(symbol, price)
        """
        for symbol in symbols:
            stream = f"{symbol.lower()}@ticker"
            if stream not in self.subscriptions:
                self.subscriptions.append(stream)
            
            if callback:
                self._ticker_callbacks[symbol].append(callback)
        
        self.log(f"订阅 {len(symbols)} 个币种的实时价格")
        
        if self.ws and self.running:
            self._send_subscribe()
    
    def subscribe_mark_price(self, symbols: List[str], callback: Callable = None):
        """订阅标记价格"""
        for symbol in symbols:
            stream = f"{symbol.lower()}@markPrice"
            if stream not in self.subscriptions:
                self.subscriptions.append(stream)
            
            if callback:
                self._mark_callbacks[symbol].append(callback)
        
        if self.ws and self.running:
            self._send_subscribe()
    
    def _send_subscribe(self):
        """发送订阅请求"""
        if not self.ws or not self.subscriptions:
            return
        
        msg = {
            "method": "SUBSCRIBE",
            "params": self.subscriptions,
            "id": int(time.time() * 1000)
        }
        
        try:
            self.ws.send(json.dumps(msg))
        except Exception as e:
            self.log(f"发送订阅失败: {e}", "ERROR")
    
    def _on_message(self, ws, message):
        """消息处理"""
        try:
            data = json.loads(message)
            
            if 'e' in data:
                event_type = data['e']
                
                if event_type == '24hrTicker':
                    symbol = data.get('s')
                    price = float(data.get('c', 0))

                    with self._price_lock:
                        self._last_prices[symbol] = price
                        self._last_update[symbol] = time.time()

                    for callback in self._ticker_callbacks.get(symbol, []):
                        try:
                            callback(symbol, price)
                        except Exception as e:
                            self.log(f"回调执行失败: {e}", "ERROR")

                elif event_type == 'markPriceUpdate':
                    symbol = data.get('s')
                    mark_price = float(data.get('p', 0))

                    with self._price_lock:
                        self._last_prices[f"{symbol}_mark"] = mark_price
                        self._last_update[f"{symbol}_mark"] = time.time()
                    
                    for callback in self._mark_callbacks.get(symbol, []):
                        try:
                            callback(symbol, mark_price)
                        except Exception as e:
                            self.log(f"回调执行失败: {e}", "ERROR")
            
            if self.on_message:
                self.on_message(data)
                
        except json.JSONDecodeError:
            self._json_error_count = getattr(self, '_json_error_count', 0) + 1
            if self._json_error_count <= 5 or self._json_error_count % 100 == 0:
                self.log(f"JSON解析失败 (累计{self._json_error_count}次): {str(message)[:80]}", "WARN")
        except Exception as e:
            self.log(f"消息处理异常: {e}", "ERROR")
    
    def _on_error(self, ws, error):
        """错误处理"""
        self.log(f"WebSocket错误: {error}", "ERROR")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """关闭处理（主动断开后不重连）"""
        self.log(f"连接关闭: {close_status_code} - {close_msg}")
        with self._ws_lock:
            if not self.running:
                self.log("主动断开，不重连")
                return
            self.running = False

            if self._reconnect_count < self._max_reconnect:
                self._reconnect_count += 1
                wait_time = min(5 * self._reconnect_count, 60)
                self.log(f"{wait_time}秒后尝试重连 ({self._reconnect_count}/{self._max_reconnect})")
                threading.Thread(target=self._delayed_reconnect, args=(wait_time,), daemon=True).start()
            else:
                self.log("达到最大重连次数，停止重连", "ERROR")

    def _delayed_reconnect(self, wait_time: float):
        """在独立线程中延迟重连，避免阻塞WebSocket回调线程"""
        time.sleep(wait_time)
        self.connect()
    
    def _on_open(self, ws):
        """连接打开"""
        self.log("连接成功")
        self._reconnect_count = 0
        self.running = True
        
        if self.subscriptions:
            self._send_subscribe()
    
    def connect(self):
        """建立连接（线程安全）"""
        with self._ws_lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    self.log("关闭websocket连接异常", "DEBUG")
                self.ws = None

            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
                self.ws_thread.start()
                
                self.log(f"正在连接: {self.ws_url}")
                
            except Exception as e:
                self.log(f"连接失败: {e}", "ERROR")
                raise e from e
    
    def disconnect(self):
        """断开连接（线程安全）"""
        with self._ws_lock:
            self.running = False
            if self.ws:
                self.ws.close()
            self.ws = None
        self.log("已断开连接")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        with self._price_lock:
            return self._last_prices.get(symbol)
    
    def get_mark_price(self, symbol: str) -> Optional[float]:
        """获取最新标记价格"""
        return self._last_prices.get(f"{symbol}_mark")
    
    def get_all_prices(self) -> Dict[str, float]:
        """获取所有价格"""
        with self._price_lock:
            return self._last_prices.copy()
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.running and self.ws is not None
    
    def get_age(self, symbol: str) -> float:
        """获取价格数据的年龄（秒）"""
        last_update = self._last_update.get(symbol, 0)
        if last_update == 0:
            return float('inf')
        return time.time() - last_update


class PriceCache:
    """
    价格缓存 - 整合WebSocket和HTTP轮询
    
    WebSocket主用，HTTP轮询备用
    """
    
    def __init__(self, client, symbols: List[str], testnet: bool = False):
        """
        Args:
            client: BinanceClient实例（用于HTTP轮询）
            symbols: 监控的币种列表
            testnet: 是否测试网
        """
        self.client = client
        self.symbols = symbols
        self.testnet = testnet
        
        self._prices: Dict[str, float] = {}
        self._mark_prices: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}
        self._price_lock = threading.Lock()
        self._cleanup_interval = 300
        self._stale_timeout = 3600

        self._ws: Optional[BinanceWebSocket] = None
        self._http_thread: Optional[threading.Thread] = None
        self._running = False

        self._max_age_seconds = 5
        self._http_interval = 2
        self._last_cleanup = time.time()

        self._http_loop_running = False
    
    def start(self):
        """启动价格缓存"""
        self._running = True
        
        self._ws = BinanceWebSocket(testnet=self.testnet)
        self._ws.subscribe_ticker(self.symbols, self._on_ticker_update)
        self._ws.subscribe_mark_price(self.symbols, self._on_mark_update)
        self._ws.connect()
        
        self._http_thread = threading.Thread(target=self._http_loop, daemon=True)
        self._http_thread.start()
        
        self.log("价格缓存已启动（WebSocket + HTTP轮询）")
    
    def stop(self):
        """停止"""
        self._running = False
        if self._ws:
            self._ws.disconnect()
        # 等待HTTP线程结束
        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join(timeout=5)
    
    def log(self, msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [PriceCache] {msg}")
    
    def _on_ticker_update(self, symbol: str, price: float):
        """实时价格更新回调"""
        with self._price_lock:
            self._prices[symbol] = price
            self._last_update[symbol] = time.time()

    def _on_mark_update(self, symbol: str, price: float):
        """标记价格更新回调"""
        with self._price_lock:
            self._mark_prices[symbol] = price
            self._last_update[symbol] = time.time()
    
    def _http_loop(self):
        """HTTP轮询备用"""
        while self._running:
            try:
                # 定期清理过期缓存
                current_time = time.time()
                if current_time - self._last_cleanup > self._cleanup_interval:
                    self._cleanup_stale_cache()
                    self._last_cleanup = current_time
                
                for symbol in self.symbols:
                    age = self._get_price_age(symbol)
                    
                    if age > self._max_age_seconds:
                        price = self.client.get_price(symbol)
                        with self._price_lock:
                            self._prices[symbol] = price
                            self._last_update[symbol] = time.time()
                        self.log(f"[HTTP] {symbol}: {price} (WebSocket数据过期)")
                
                time.sleep(self._http_interval)
                
            except Exception as e:
                self.log(f"HTTP轮询异常: {e}")
                time.sleep(5)
    
    def _cleanup_stale_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        with self._price_lock:
            stale_keys = [
                symbol for symbol, last_update in self._last_update.items()
                if current_time - last_update > self._stale_timeout
            ]
            for symbol in stale_keys:
                self._prices.pop(symbol, None)
                # 修复：移除不存在的_last_prices（BUG-M006）
                self._last_update.pop(symbol, None)
        if stale_keys:
            self.log(f"清理了 {len(stale_keys)} 个过期缓存")
    
    def _get_price_age(self, symbol: str) -> float:
        """获取价格数据年龄"""
        last = self._last_update.get(symbol, 0)
        if last == 0:
            return float('inf')
        return time.time() - last
    
    def get_price(self, symbol: str) -> float:
        """获取价格（优先WebSocket，必要时HTTP）"""
        with self._price_lock:
            price = self._prices.get(symbol)
            age = self._get_price_age(symbol)
        
        if price is None or age > self._max_age_seconds:
            try:
                price = self.client.get_price(symbol)
                with self._price_lock:
                    self._prices[symbol] = price
                    self._last_update[symbol] = time.time()
            except Exception as e:
                if price is None:
                    raise e from e
        return price
    
    def get_mark_price(self, symbol: str) -> Optional[float]:
        """获取标记价格（返回None表示无数据，避免除零）"""
        with self._price_lock:
            return self._mark_prices.get(symbol) or self._prices.get(symbol)
    
    def is_fresh(self, symbol: str) -> bool:
        """检查价格是否新鲜"""
        with self._price_lock:
            return self._get_price_age(symbol) <= self._max_age_seconds


if __name__ == "__main__":
    ws = BinanceWebSocket(testnet=True)
    ws.subscribe_ticker(['BTCUSDT', 'ETHUSDT'], lambda s, p: print(f"{s}: {p}"))
    ws.connect()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ws.disconnect()
