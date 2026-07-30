# -*- coding: utf-8 -*-
"""
生产级币安客户端 - 完全参考trading_GPT实现
支持：实盘、测试网、自动重连、时间同步、API限流
"""
from decimal import Decimal
from framework.core.config import get_main_config
from binance.exceptions import BinanceAPIException
import time
import json
import threading
import requests
import hmac
import hashlib
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
from datetime import datetime

# 使用统一的日志系统
from framework.core.logger import get_logger
# 修复：统一使用utils.retry中的重试装饰器
from utils.retry import retry_on_failure

logger = get_logger('binance_client')

# 程序订单唯一标识前缀，用于区分程序开仓和手动开仓
BNFF_ORDER_PREFIX = 'BNFF_'


class BinanceClient:
    """
    生产级币安合约客户端
    
    特性：
    - 自动时间同步（解决-1021错误）
    - API请求限流
    - 自动重连机制
    - 支持HEDGING/ONE-WAY模式
    - 完善的错误处理
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 mode: str = 'live', testnet: bool = False):
        """
        初始化客户端
        
        Args:
            api_key: API密钥
            api_secret: API密钥
            mode: 运行模式 (live/testnet/backtest)
            testnet: 是否使用测试网
        """
        from framework.core.config import get_config_manager
        
        self.mode = mode
        self.testnet = testnet or (mode == 'testnet')
        
        # 加载配置
        config_manager = get_config_manager()
        if not config_manager.load_main_config():
            logger.error("主配置加载失败")
        
        main_config = config_manager.main_config
        
        # API密钥 - 优先使用传入参数，否则使用配置
        if self.testnet:
            self.api_key = api_key or main_config.testnet_api_key
            self.api_secret = api_secret or main_config.testnet_api_secret
        else:
            self.api_key = api_key or main_config.binance_api_key
            self.api_secret = api_secret or main_config.binance_api_secret
        
        # API地址
        self.base_url = "https://demo-fapi.binance.com" if self.testnet else "https://fapi.binance.com"
        
        # 会话
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
        # 时间同步
        self._time_offset = 0
        self._last_time_sync = 0
        self._time_sync_interval = 300  # 5分钟同步一次
        self._auto_sync_enabled = True  # 自动同步开关
        
        # 持仓模式缓存
        self._position_mode = None
        self._position_mode_cache_time = 0
        
        # 交易对信息缓存
        self._exchange_info_cache = None
        self._symbol_info_cache: Dict[str, Dict] = {}
        self._leverage_brackets_cache: Dict[str, List] = {}
        
        # API限流
        self._request_timestamps: List[float] = []
        self._request_lock = threading.Lock()
        self._max_requests_per_minute = 1200
        self._max_orders_per_second = 50
        
        # 连接状态
        self._connected = False
        
        # 初始化
        if mode != 'backtest':
            self._initialize()
    
    def close(self):
        """关闭HTTP会话"""
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False

    def __del__(self):
        """析构时关闭会话"""
        self.close()
    
    def _initialize(self):
        """初始化连接"""
        try:
            # 测试连接
            self._test_connection()
            
            # 同步时间
            self._sync_time()
            
            # 获取交易对信息
            self._load_exchange_info()
            
            self._connected = True
            logger.info(f"币安客户端初始化成功 - 模式: {self.mode}")
            
        except Exception as e:
            logger.error(f"币安客户端初始化失败: {e}")
            raise e from e
    
    def _test_connection(self):
        """测试连接"""
        try:
            response = self.session.get(f"{self.base_url}/fapi/v1/ping", timeout=5)
            response.raise_for_status()
            logger.info(f"API连接成功 - 模式:{self.testnet if hasattr(self, 'testnet') else 'live'}")
        except Exception as e:
            logger.error(f"API连接失败: {e}")
            raise e from e
    
    def _sync_time(self):
        """
        同步服务器时间并自动校准
        
        解决APIError -1021 (Timestamp for this request is outside of the recvWindow)
        """
        try:
            response = self.session.get(f"{self.base_url}/fapi/v1/time", timeout=5, allow_redirects=False)
            if response.status_code == 301:
                logger.warning(f"时间同步: 301重定向到 {response.headers.get('Location', '未知')}")
                return
            text = response.text.strip()
            if text == 'ok':
                logger.info("时间同步: 测试网返回ok，跳过")
                self._last_time_sync = time.time()
                return
            data = response.json()
            
            server_time = data.get('serverTime')
            if server_time is None:
                logger.warning("时间同步失败: serverTime为空")
                return
            server_time = int(server_time)
            local_time = int(time.time() * 1000)
            
            old_offset = self._time_offset
            self._time_offset = server_time - local_time
            self._last_time_sync = time.time()
            
            # 根据偏移值判断同步状态并自动校准
            abs_offset = abs(self._time_offset)
            
            if abs_offset > 1000:
                # 偏移超过1秒，启用自动时间补偿
                if self._auto_sync_enabled:
                    logger.warning(f"时间偏移较大: {old_offset}ms → {self._time_offset}ms (启用自动补偿)")
                else:
                    logger.warning(f"时间偏移较大: {self._time_offset}ms (建议校准本地时间)")
            elif abs_offset > 100:
                logger.info(f"时间同步完成: 偏移={self._time_offset}ms (警告)")
            else:
                logger.info(f"时间同步完成: 偏移={self._time_offset}ms (正常)")
            
        except Exception as e:
            if 'Expecting value' in str(e):
                logger.warning(f"时间同步失败: 返回非JSON (status={getattr(response, 'status_code', '?')}, url={getattr(response, 'url', '?')}, body={getattr(response, 'text', '')[:100]})")
            else:
                logger.warning(f"时间同步失败: {e}")
    
    def _check_time_sync(self):
        """检查是否需要重新同步时间"""
        current_time = time.time()
        if current_time - self._last_time_sync > self._time_sync_interval:
            self._sync_time()
    
    def get_timestamp(self) -> int:
        """获取带时间偏移的时间戳"""
        return int(time.time() * 1000) + self._time_offset
    
    def _get_timestamp(self) -> int:
        """获取时间戳（已同步）"""
        self._check_time_sync()
        return int(time.time() * 1000) + self._time_offset
    
    def _sign(self, params: Dict) -> str:
        """生成签名"""
        query_string = urlencode(params)
        signature = hmac.HMAC(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _rate_limit(self):
        """API请求限流（优化版：使用双端队列，避免频繁创建新列表）"""
        with self._request_lock:
            current_time = time.time()

            # 清理1分钟前的请求记录（优化：使用while循环原地删除）
            # 从列表头部删除过期记录，避免创建新列表
            cutoff_time = current_time - 60
            while self._request_timestamps and self._request_timestamps[0] < cutoff_time:
                self._request_timestamps.pop(0)

            # 检查是否超过限制
            if len(self._request_timestamps) >= self._max_requests_per_minute - 10:
                if self._request_timestamps:
                    sleep_time = 60 - (current_time - self._request_timestamps[0])
                    if sleep_time > 0:
                        logger.warning(f"API限流，等待{sleep_time:.1f}秒")
                        # 在锁内等待，防止其他线程绕过限流
                        time.sleep(sleep_time)
                        # 重获锁后重新清理
                        current_time = time.time()
                        cutoff_time = current_time - 60
                        while self._request_timestamps and self._request_timestamps[0] < cutoff_time:
                            self._request_timestamps.pop(0)

            # 记录本次请求
            self._request_timestamps.append(current_time)
    
    def _request(self, method: str, endpoint: str,
                 params: Dict = None, signed: bool = True,
                 max_retries: int = 3, leverage: int = None) -> Any:
        """
        发送API请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            params: 参数
            signed: 是否需要签名
            max_retries: 最大重试次数
        
        Returns:
            API响应
        """
        url = self.base_url + endpoint
        params = (params or {}).copy()
        
        last_error = None
        for attempt in range(max_retries):
            try:
                # 限流
                self._rate_limit()
                
                # 签名
                if signed:
                    params['timestamp'] = self._get_timestamp()
                    params['recvWindow'] = 10000  # 10秒窗口
                    params['signature'] = self._sign(params)
                
                # 发送请求
                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=30)
                elif method == 'POST':
                    response = self.session.post(url, params=params, timeout=30)
                elif method == 'DELETE':
                    response = self.session.delete(url, params=params, timeout=30)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")
                
                # 检查响应
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.HTTPError as e:
                last_error = e
                
                # 429 限流错误
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"API限流(429)，等待{retry_after}秒")
                    time.sleep(retry_after)
                    continue
                
                # 400错误：解析Binance返回的具体错误信息
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_code = error_data.get('code', 'N/A')
                        error_msg = error_data.get('msg', 'Unknown error')
                        logger.warning(f"Binance API 400错误: code={error_code}, msg={error_msg}")
                        # 不记录params以防泄露签名

                        # -2027 错误：超过当前杠杆档位的持仓限制
                        # 自动循环降低杠杆重试
                        if str(error_code) == '-2027' and leverage is not None:
                            symbol = params.get('symbol', '')
                            order_type = params.get('type', 'MARKET')
                            original_leverage = leverage

                            for try_leverage in range(leverage, 0, -1):
                                try:
                                    logger.warning(f"检测到-2027错误，尝试降低杠杆到 {try_leverage}x...")

                                    # 重新设置杠杆
                                    self.set_leverage(symbol, try_leverage)
                                    logger.info(f"✅ {symbol} 杠杆已设置为 {try_leverage}x")

                                    # 用新杠杆重新调整数量（考虑新杠杆档位的限额）
                                    new_quantity = self.adjust_quantity_to_symbol(
                                        symbol, params.get('quantity', 0), order_type,
                                        leverage=try_leverage, reduce_only=params.get('reduceOnly') == 'true'
                                    )

                                    if new_quantity is None or new_quantity <= 0:
                                        logger.warning(f"调整后数量无效({new_quantity})，跳过")
                                        continue

                                    # 更新params中的数量
                                    params['quantity'] = new_quantity
                                    # 清除旧签名（确保干净的参数）
                                    params.pop('signature', None)

                                    # 重新签名（数量变了，签名必须重新计算）
                                    params['timestamp'] = self._get_timestamp()
                                    params['recvWindow'] = 10000
                                    params['signature'] = self._sign(params)

                                    logger.info(f"调整后: 杠杆={try_leverage}x, 数量={new_quantity}, 签名={params['signature'][:8]}...")

                                    # 重试请求
                                    retry_response = self.session.post(url, params=params, timeout=30)
                                    retry_status = retry_response.status_code

                                    if retry_status == 200:
                                        logger.info(f"✅ -2027错误已解决: 杠杆={try_leverage}x, 数量={new_quantity}")
                                        return retry_response.json()
                                    else:
                                        # 解析重试响应
                                        try:
                                            retry_error_data = retry_response.json()
                                            retry_error_code = retry_error_data.get('code', 'N/A')
                                            retry_error_msg = retry_error_data.get('msg', '')

                                            # -1022 签名错误：说明参数有问题，不是-2027
                                            if str(retry_error_code) == '-1022':
                                                logger.error(f"降低杠杆重试时返回-1022签名错误: {retry_error_msg}")
                                                logger.error(f"URL: {url}")
                                                # 重新签名验证
                                                test_params = params.copy()
                                                test_params.pop('signature', None)
                                                test_params['timestamp'] = self._get_timestamp()
                                                test_params['recvWindow'] = 10000
                                                new_sig = self._sign(test_params)
                                                test_params['signature'] = new_sig
                                                logger.error(f"重新签名: timestamp={test_params['timestamp']}, signature={new_sig[:16]}...")
                                                logger.error(f"参数签名串: {urlencode(test_params)}")
                                                # 不再降杠杆
                                                break

                                            if str(retry_error_code) != '-2027':
                                                # 不是-2027错误，不再降杠杆
                                                logger.warning(f"降低杠杆后返回错误 code={retry_error_code}, msg={retry_error_msg}")
                                                break
                                            # 仍是-2027，继续降杠杆
                                            logger.warning(f"降低杠杆到{try_leverage}x后仍返回-2027，继续降杠杆...")
                                        except Exception:
                                            logger.debug(f"降杠杆{try_leverage}x重试时异常", exc_info=True)

                                        # 如果已经降到1x仍然失败，直接退出
                                        if try_leverage <= 1:
                                            logger.error(f"杠杆已降至1x仍无法下单，放弃")
                                            break

                                except Exception as retry_e:
                                    logger.debug(f"尝试杠杆{try_leverage}x时异常: {retry_e}")
                                    continue

                            # -2027重试循环结束后，如果还没成功，抛出原始错误
                            raise last_error

                        # 以下400错误码可以重试（临时性问题）
                        # -1000: 未知错误
                        # -1001: 断开连接
                        # -1003: 服务器繁忙
                        # -1006: 未知错误
                        # -1021: 时间戳错误（已在下面处理）
                        # -2013: 没有新的订单可以取消
                        # -2014: 认证失败
                        retryable_400_codes = ['-1000', '-1001', '-1003', '-1006']
                        if error_code in retryable_400_codes:
                            logger.warning(f"400错误码 {error_code} 可重试，等待后重试...")
                            time.sleep(1)
                            continue

                    except json.JSONDecodeError as je:
                        logger.warning(f"400错误，响应体JSON解析失败: {je}")
                        # JSON解析失败可能是临时问题，重试
                        logger.warning("JSON解析失败，等待后重试...")
                        time.sleep(1)
                        continue
                    # 其他400错误不重试，直接失败
                    raise
                
                # -1021 时间戳错误
                try:
                    error_data = response.json()
                    if error_data.get('code') == -1021:
                        logger.warning("时间戳错误，重新同步时间")
                        self._sync_time()
                        time.sleep(1)
                        continue
                except json.JSONDecodeError as je:
                        logger.debug(f"非JSON响应或解析失败: {je}")
                
            except requests.exceptions.Timeout:
                last_error = Exception("请求超时")
                logger.warning(f"请求超时，第{attempt+1}次重试")
                time.sleep(1)
                continue
            
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(f"连接错误，第{attempt+1}次重试")
                time.sleep(2)
                continue
            
            except Exception as e:
                last_error = e
                logger.error(f"请求异常: {e}")
        
        raise last_error or Exception("请求失败")
    
    # ==================== 市场数据 ====================
    

    def get_exchange_info(self) -> Dict:
        """获取交易所信息"""
        if self._exchange_info_cache is None:
            self._load_exchange_info()
        return self._exchange_info_cache
    
    def _load_exchange_info(self):
        """加载交易所信息"""
        data = self._request('GET', '/fapi/v1/exchangeInfo', signed=False)
        self._exchange_info_cache = data
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """获取交易对信息"""
        if symbol not in self._symbol_info_cache:
            info = self.get_exchange_info()
            for s in info.get('symbols', []):
                if s['symbol'] == symbol:
                    self._symbol_info_cache[symbol] = s
                    break
        return self._symbol_info_cache.get(symbol, {})
    

    def get_price(self, symbol: str) -> float:
        """获取当前价格"""
        data = self._request('GET', '/fapi/v1/ticker/price', 
                            {'symbol': symbol}, signed=False)
        price = data.get('price') if data else None
        if price is None:
            raise ValueError(f"获取价格失败: {symbol}")
        return float(price)

    def get_ticker_price(self, symbol: str) -> float:
        """获取当前价格（别名）"""
        return self.get_price(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        """批量获取所有币种当前价格（weight=5，比逐个取价更省）"""
        data = self._request('GET', '/fapi/v1/ticker/price', params={}, signed=False)
        if not data or not isinstance(data, list):
            return {}
        return {item['symbol']: float(item['price']) for item in data if 'symbol' in item and 'price' in item}
    

    def get_mark_price(self, symbol: str) -> float:
        """获取标记价格"""
        data = self._request('GET', '/fapi/v1/premiumIndex',
                            {'symbol': symbol}, signed=False)
        price = data.get('markPrice') if data else None
        if price is None:
            raise ValueError(f"获取标记价格失败: {symbol}")
        return float(price)
    

    def get_klines(self, symbol: str, interval: str = '15m', 
                   limit: int = 500) -> List:
        """获取K线数据"""
        return self._request('GET', '/fapi/v1/klines',
                            {'symbol': symbol, 'interval': interval, 'limit': limit},
                            signed=False)
    

    def get_ticker_24h(self, symbol: str) -> Dict:
        """获取24小时行情"""
        return self._request('GET', '/fapi/v1/ticker/24hr',
            {'symbol': symbol}, signed=False)
    

    def get_top_symbols(self, count: int = 30) -> List[str]:
        """
        获取成交量最大的币种列表
        
        Args:
            count: 返回币种数量
            
        Returns:
            币种列表，按成交量降序排列
        """
        data = self._request('GET', '/fapi/v1/ticker/24hr', signed=False)
        
        # 过滤USDT交易对
        usdt_pairs = [
            item for item in data 
            if item.get('symbol', '').endswith('USDT') 
            and float(item.get('quoteVolume', 0)) > 0
        ]
        
        # 按成交量排序
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
        
        # 返回前N个
        return [item['symbol'] for item in sorted_pairs[:count]]

    # ==================== 账户信息 ====================
    

    def get_account_balance(self) -> Dict:
        """获取账户余额 - 统一使用/fapi/v2/balance端点

        返回统一结构：
        - total_balance: 总余额
        - available_balance: 可用余额
        - cross_wallet_balance: 交叉钱包余额
        - cross_unpnl: 未实现盈亏
        """
        # 使用/fapi/v2/balance端点（更简洁，返回资产列表）
        try:
            data = self._request('GET', '/fapi/v2/balance', signed=True)
            # 找到USDT资产
            for item in data:
                if item.get('asset') == 'USDT':
                    total_balance = float(item.get('balance', 0))
                    unrealized_pnl = float(item.get('crossUnPnl', 0))
                    available_balance = float(item.get('availableBalance', item.get('withdrawAvailable', 0)))

                    logger.info(f"[余额] v2/balance - balance={total_balance}, crossUnPnl={unrealized_pnl}, available={available_balance}")

                    return {
                        'total_balance': total_balance,
                        'available_balance': available_balance,
                        'cross_wallet_balance': float(item.get('crossWalletBalance', 0)),
                        'cross_unpnl': unrealized_pnl,
                    }
            # 如果没有USDT，返回第一个
            if data and len(data) > 0:
                item = data[0]
                total_balance = float(item.get('balance', 0))
                unrealized_pnl = float(item.get('crossUnPnl', 0))
                available_balance = float(item.get('availableBalance', item.get('withdrawAvailable', 0)))

                return {
                    'total_balance': total_balance,
                    'available_balance': available_balance,
                    'cross_wallet_balance': float(item.get('crossWalletBalance', 0)),
                    'cross_unpnl': unrealized_pnl,
                }
        except Exception as e:
            logger.warning(f"[余额] /fapi/v2/balance 失败: {e}")

        # 备用：使用/fapi/v2/account端点，转换为统一结构
        try:
            data = self._request('GET', '/fapi/v2/account')

            total_balance = float(data.get('totalWalletBalance', 0))
            total_margin = float(data.get('totalInitialMargin', 0))
            unrealized_pnl = float(data.get('totalUnrealizedProfit', 0))
            available_balance = float(data.get('availableBalance', 0))

            logger.info(f"[余额] v2/account - totalWalletBalance={total_balance}, totalInitialMargin={total_margin}, totalUnrealizedProfit={unrealized_pnl}, availableBalance={available_balance}")

            # 如果availableBalance为0，使用totalWalletBalance - totalInitialMargin计算
            if available_balance == 0 and total_balance > 0:
                available_balance = total_balance - total_margin
                logger.info(f"[余额] 重新计算可用余额: {available_balance} = {total_balance} - {total_margin}")

            # 转换为统一结构（与v2/balance一致）
            return {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'cross_wallet_balance': total_balance,  # v2/account没有crossWalletBalance，使用total_balance
                'cross_unpnl': unrealized_pnl,
            }
        except Exception as e:
            logger.error(f"[余额] /fapi/v2/account 也失败: {e}")
            raise
    

    def get_positions(self) -> List[Dict]:
        """获取所有持仓"""
        data = self._request('GET', '/fapi/v2/positionRisk')
        
        positions = []
        for pos in data:
            position_amt = float(pos.get('positionAmt', 0))
            if abs(position_amt) > 0.000001:
                positions.append({
                    'symbol': pos['symbol'],
                    'position_amt': position_amt,
                    'entry_price': float(pos.get('entryPrice', 0)),
                    'unrealized_pnl': float(pos.get('unRealizedProfit', 0)),
                    'leverage': int(pos.get('leverage', 1)),
                    'side': 'LONG' if position_amt > 0 else 'SHORT',
                    'liquidation_price': float(pos.get('liquidationPrice', 0)),
                    'mark_price': float(pos.get('markPrice', 0)),
                    'position_side': pos.get('positionSide', 'BOTH'),
                    'isolated_wallet': float(pos.get('isolatedWallet', 0)),
                    'margin_type': pos.get('marginType', 'cross'),
                })
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """获取单个持仓"""
        positions = self.get_positions()
        for pos in positions:
            if pos['symbol'] == symbol:
                return pos
        return None
    
    def get_position_mode(self) -> str:
        """
        获取持仓模式
        
        Returns:
            'HEDGING' 或 'ONE-WAY'
        """
        # 强制每次重新检测，不使用缓存
        try:
            data = self._request('GET', '/fapi/v1/positionSide/dual')
            mode = 'HEDGING' if data.get('dualSidePosition', False) else 'ONE-WAY'
            logger.info(f"检测到持仓模式: {mode}")
            return mode
        except Exception as e:
            logger.warning(f"获取持仓模式失败: {e}，尝试设置为ONE-WAY模式")
            # 尝试设置为 ONE-WAY 模式
            try:
                self._request('POST', '/fapi/v1/positionSide/dual', {'dualSidePosition': 'false'})
                return 'ONE-WAY'
            except (Exception, ValueError):
                return 'ONE-WAY'
    
    # ==================== 交易接口 ====================
    

    def get_leverage_bracket(self, symbol: str) -> int:
        """获取币种允许的最大杠杆倍数
        
        币安期货每个币种有不同的杠杆档位限制
        通过 /fapi/v1/leverageBracket 查询
        """
        try:
            data = self._request('GET', '/fapi/v1/leverageBracket')
            for item in data:
                if item.get('symbol') == symbol:
                    brackets = item.get('brackets', [])
                    if brackets:
                        # 返回最大杠杆
                        return max(b.get('initialLeverage', 20) for b in brackets)
            return 20  # 默认值
        except Exception as e:
            logger.warning(f"获取杠杆档位失败 {symbol}: {e}，使用默认值20")
            return 20
    

    def get_symbol_position_limit(self, symbol: str, leverage: int) -> float:
        """获取币种在指定杠杆下的最大持仓限制（USDT名义价值）
        
        通过 /fapi/v1/leverageBracket 查询
        注意：测试网每个币种有独立限额（如FXSUSDT 20x只有5000 USDT），
        比账户余额限制更严，必须查询真实值。
        """
        try:
            data = self._request('GET', '/fapi/v1/leverageBracket')
            for item in data:
                if item.get('symbol') == symbol:
                    brackets = item.get('brackets', [])
                    logger.info(f"[仓位限额] {symbol} 杠杆档位: {brackets}")
                    # 精确匹配杠杆档位
                    for b in brackets:
                        if int(b.get('initialLeverage', 0)) == leverage:
                            # notionalCap 是该杠杆档位的最大名义价值
                            limit = float(b.get('notionalCap', float('inf')))
                            logger.info(f"[仓位限额] {symbol} 杠杆{leverage}x 限额: {limit} USDT")
                            return limit
                    # 没找到精确杠杆，使用比实际杠杆更小的最大杠杆档位
                    if brackets:
                        # 找到所有比实际杠杆更小的档位
                        lower_brackets = [b for b in brackets if int(b.get('initialLeverage', 0)) <= leverage]
                        if lower_brackets:
                            # 使用最大的那个（最接近实际杠杆）
                            lower_brackets.sort(key=lambda x: int(x.get('initialLeverage', 0)), reverse=True)
                            limit = float(lower_brackets[0].get('notionalCap', float('inf')))
                            logger.info(f"[仓位限额] {symbol} 未找到杠杆{leverage}x，使用档位{lower_brackets[0].get('initialLeverage')}x: {limit} USDT")
                            return limit
                        else:
                            # 如果没有比实际杠杆更小的档位，使用最小档位（最高杠杆）
                            limit = float(brackets[0].get('notionalCap', float('inf')))
                            logger.info(f"[仓位限额] {symbol} 未找到杠杆{leverage}x，使用最小档位{brackets[0].get('initialLeverage')}x: {limit} USDT")
                            return limit
            return float('inf')
        except Exception as e:
            logger.warning(f"获取仓位限制失败 {symbol}: {e}，使用无限制")
            return float('inf')
    

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """设置杠杆（自动限制在币种允许范围内，支持降级）"""
        # 先获取该币种允许的最大杠杆
        max_leverage = self.get_leverage_bracket(symbol)
        
        # 如果请求的杠杆超过限制，使用最大杠杆
        if leverage > max_leverage:
            leverage = max_leverage
            logger.warning(f"{symbol} 请求杠杆{leverage}x超过限制{max_leverage}x，已调整为{max_leverage}x")
        
        # 如果杠杆<1，使用1
        if leverage < 1:
            leverage = 1
        
        # 尝试设置杠杆
        try:
            return self._request('POST', '/fapi/v1/leverage',
                                {'symbol': symbol, 'leverage': leverage})
        except Exception as e:
            # 如果设置杠杆失败（如测试网某些币种不支持），尝试降低杠杆
            if 'Bad Request' in str(e) or '400' in str(e):
                # 从max_leverage向下遍历，而不是硬编码序列
                for trial_leverage in range(max_leverage, 0, -1):
                    if trial_leverage >= 1:
                        try:
                            result = self._request('POST', '/fapi/v1/leverage',
                                {'symbol': symbol, 'leverage': trial_leverage})
                            logger.warning(f"{symbol} 设置杠杆失败，已降级到{trial_leverage}x")
                            return result
                        except (BinanceAPIException, ValueError, KeyError):
                            continue

                # 如果所有杠杆都失败，返回成功（避免阻塞交易）
                logger.warning(f"{symbol} 无法设置杠杆，跳过（可能测试网不支持）")
                return {'msg': 'leverage skipped'}
            raise
    

    def set_margin_type(self, symbol: str, margin_type: str = 'ISOLATED') -> Dict:
        """设置保证金模式"""
        # 测试网可能有持仓限制，直接返回成功
        if self.testnet:
            return {'msg': 'testnet mode, skip margin type'}
        try:
            return self._request('POST', '/fapi/v1/marginType',
                                {'symbol': symbol, 'marginType': margin_type})
        except Exception as e:
            if 'No need to change' in str(e):
                return {'msg': 'margin type unchanged'}
            raise e from e
    
    def get_max_notional_for_leverage(self, symbol: str, leverage: int) -> float:
        """根据杠杆计算最大名义价值
        
        基于账户可用余额和杠杆计算最大可开仓位
        """
        try:
            balance_data = self.get_account_balance()
            available = float(balance_data.get('available_balance', 0))
            return available * leverage
        except Exception as e:
            logger.warning(f"计算最大名义价值失败 {symbol}: {e}")
            return 0.0
    
    def load_all_symbol_filters(self) -> Dict:
        """加载所有币种的交易规则"""
        try:
            data = self._request('GET', '/fapi/v1/exchangeInfo')
            all_filters = {}
            for s in data.get('symbols', []):
                if s.get('status') != 'TRADING':
                    continue
                symbol = s.get('symbol', '')
                filters = {
                    'quantityPrecision': int(s.get('quantityPrecision', 0)),
                    'pricePrecision': int(s.get('pricePrecision', 2)),
                }
                for f in s.get('filters', []):
                    ft = f.get('filterType')
                    if ft == 'LOT_SIZE':
                        filters['minQty'] = self._parse_number(f.get('minQty', 1))
                        filters['maxQty'] = self._parse_number(f.get('maxQty', 1000000))
                        filters['stepSize'] = self._parse_number(f.get('stepSize', 1))
                    elif ft == 'MARKET_LOT_SIZE':
                        filters['marketMinQty'] = self._parse_number(f.get('minQty', 1))
                        filters['marketMaxQty'] = self._parse_number(f.get('maxQty', 1000000))
                    elif ft == 'PRICE_FILTER':
                        filters['tickSize'] = float(f.get('tickSize', 0.01))
                    elif ft == 'MIN_NOTIONAL':
                        filters['minNotional'] = float(f.get('notional', 5))
                all_filters[symbol] = filters
            logger.info(f"已加载 {len(all_filters)} 个币种交易规则")
            return all_filters
        except Exception as e:
            logger.warning(f"加载交易规则失败: {e}")
            return {}
    
    def get_symbol_filters(self, symbol: str) -> Dict:
        """获取币种交易规则（从缓存或API）"""
        # 优先从缓存获取
        if hasattr(self, '_symbol_filters') and self._symbol_filters and symbol in self._symbol_filters:
            return self._symbol_filters[symbol]
        
        # 否则从API获取并缓存
        try:
            data = self._request('GET', '/fapi/v1/exchangeInfo')
            for s in data.get('symbols', []):
                if s.get('symbol') == symbol:
                    filters = {
                        'quantityPrecision': self._parse_number(s.get('quantityPrecision', 0)),
                        'pricePrecision': self._parse_number(s.get('pricePrecision', 2)),
                    }
                    for f in s.get('filters', []):
                        ft = f.get('filterType')
                        if ft == 'LOT_SIZE':
                            filters['minQty'] = self._parse_number(f.get('minQty', '1'))
                            filters['maxQty'] = self._parse_number(f.get('maxQty', '1000000'))
                            filters['stepSize'] = self._parse_number(f.get('stepSize', '1'))
                        elif ft == 'MARKET_LOT_SIZE':
                            filters['marketMinQty'] = self._parse_number(f.get('minQty', '1'))
                            filters['marketMaxQty'] = self._parse_number(f.get('maxQty', '1000000'))
                        elif ft == 'MIN_NOTIONAL':
                            filters['minNotional'] = self._parse_number(f.get('notional', '5'))
                        elif ft == 'PRICE_FILTER':
                            filters['tickSize'] = float(f.get('tickSize', 0.01))
                    return filters
            return {}
        except Exception as e:
            logger.warning(f"获取{symbol}交易规则失败: {e}")
            return {}
    
    def _parse_number(self, value) -> float:
        """安全转换数字，支持字符串和小数"""
        if value is None:
            return 0
        try:
            return float(value) if isinstance(value, str) else float(value)
        except (ValueError, TypeError):
            return 0
    
    def _get_symbol_leverage(self, symbol: str) -> int:
        """获取当前设置的杠杆倍数（从账户信息查询）"""
        try:
            # 用 v2/positionRisk 兼容测试网（v1/account 在测试网返回404）
            data = self._request('GET', '/fapi/v2/positionRisk')
            for pos in data:
                if pos.get('symbol') == symbol and abs(float(pos.get('positionAmt', 0))) > 0.001:
                    return int(pos.get('leverage', 20))
            return 20  # 无持仓时默认20x
        except Exception as e:
            logger.warning(f"获取{symbol}杠杆失败: {e}，使用默认值20")
            return 20
    
    def _get_symbol_existing_notional(self, symbol: str) -> float:
        """获取币种已有持仓的名义价值（用于从限额中扣减）"""
        try:
            data = self._request('GET', '/fapi/v2/positionRisk')
            
            # 打印前几个持仓的完整信息用于调试
            if data and len(data) > 0:
                logger.info(f"[仓位限额] {symbol} 查询到 {len(data)} 个持仓")
                # 打印所有有持仓的符号
                active_positions = [(p.get('symbol'), p.get('positionAmt'), p.get('entryPrice')) for p in data if float(p.get('positionAmt', 0)) != 0]
                logger.info(f"[仓位限额] 有持仓的符号: {active_positions}")
            
            for pos in data:
                pos_symbol = pos.get('symbol', '')
                # 直接比较，看是否匹配
                if pos_symbol == symbol:
                    # 使用驼峰命名（和 get_positions 保持一致）
                    position_amt = float(pos.get('positionAmt', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    
                    logger.info(f"[仓位限额] {symbol} raw: positionAmt={position_amt}, entryPrice={entry_price}")
                    
                    # 计算名义价值
                    if position_amt != 0 and entry_price > 0:
                        notional = abs(position_amt * entry_price)
                        logger.info(f"[仓位限额] {symbol} 已有持仓: amt={position_amt}, price={entry_price}, notional={notional}")
                        return notional
                    
                    # 备用：尝试 notional 字段
                    notional = float(pos.get('notional', 0))
                    if notional > 0:
                        logger.info(f"[仓位限额] {symbol} 已有持仓名义价值: {notional}")
                        return abs(notional)
            
            # 没找到持仓，记录一下
            logger.info(f"[仓位限额] {symbol} 未找到持仓")
            return 0.0
        except Exception as e:
            logger.warning(f"获取已有持仓失败 {symbol}: {e}")
            return 0.0
    
    def adjust_quantity_to_symbol(self, symbol: str, quantity: float, order_type: str = 'MARKET', leverage: int = None, reduce_only: bool = False) -> float:
        """根据币安规则调整数量

        限制优先级（由强到弱）：
        1. Binance 币种档位 notionalCap（leverageBracket，测试网每个币种独立限额）
        2. 账户可用余额 × 杠杆
        3. 最小数量 minQty（LOT_SIZE / MARKET_LOT_SIZE）
        4. 最大数量 maxQty（LOT_SIZE / MARKET_LOT_SIZE）
        5. minNotional 名义价值下限

        Args:
            symbol: 交易对
            quantity: 原始数量
            order_type: 订单类型 LIMIT/MARKET
            leverage: 杠杆倍数（优先使用，未传入则从账户查询）
            reduce_only: 是否只减仓（平仓订单跳过持仓限额检查）
        """
        from decimal import Decimal, ROUND_DOWN

        # 获取基础信息
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return int(quantity)

        step = Decimal(str(filters.get('stepSize', 1)))
        price = self.get_price(symbol)
        leverage = leverage if leverage is not None else self._get_symbol_leverage(symbol)

        # 初始化数量
        qty = Decimal(str(quantity))

        # 限制1：Binance 币种档位 notionalCap（平仓订单跳过）
        if not reduce_only:
            qty = self._apply_notional_limit(symbol, qty, price, leverage)

        # 限制2：账户余额 × 杠杆
        qty = self._apply_balance_limit(symbol, qty, price, leverage)

        # 限制3：应用步长
        if step > 0:
            qty = (qty / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step

        # 限制4：minQty / maxQty
        min_qty = Decimal(str(filters.get('minQty', 1)))
        max_qty = Decimal(str(filters.get('marketMaxQty' if order_type == 'MARKET' else 'maxQty', 1000000)))
        qty = max(min(qty, max_qty), min_qty)

        # 限制5：minNotional 名义价值下限
        if price and price > 0 and qty > 0:
            qty = self._apply_min_notional(qty, price, filters, step, max_qty)

        return int(qty) if step == 1 else float(qty)

    def _apply_notional_limit(self, symbol: str, qty: Decimal, price: float, leverage: int) -> Decimal:
        """应用Binance档位notional限制"""
        from decimal import Decimal

        notional_limit = self.get_symbol_position_limit(symbol, leverage)
        existing_notional = self._get_existing_notional(symbol)
        available_notional = notional_limit - existing_notional

        if available_notional <= 0:
            logger.warning(f"{symbol} 已有持仓已占用全部限额({notional_limit} USDT, 已有持仓占{existing_notional:.0f} USDT)，无法下单")
            return Decimal('0')

        if price and price > 0:
            max_qty_by_notional = Decimal(str(available_notional)) / Decimal(str(price))
            if max_qty_by_notional < qty:
                logger.warning(f"{symbol} 数量 {qty} 超过Binance档位限额({notional_limit} USDT, 已有持仓占{existing_notional:.0f} USDT, 剩余{available_notional:.0f} USDT)，调整为 {max_qty_by_notional}")
                return max_qty_by_notional

        return qty

    def _apply_balance_limit(self, symbol: str, qty: Decimal, price: float, leverage: int) -> Decimal:
        """应用账户余额限制"""
        from decimal import Decimal

        try:
            balance_data = self.get_account_balance()
            available = float(balance_data.get('available_balance', 0))
            max_notional = available * leverage

            if max_notional <= 0:
                logger.error(f"{symbol} 账户余额不足（available={available} USDT），无法下单")
                return Decimal('0')

            if price and price > 0:
                max_qty_by_balance = Decimal(str(max_notional)) / Decimal(str(price))
                if max_qty_by_balance < qty:
                    logger.warning(f"{symbol} 数量 {qty} 超过余额限额 {available}x{leverage}={max_notional} USDT，调整为 {max_qty_by_balance}")
                    return max_qty_by_balance
        except Exception as e:
            logger.warning(f"{symbol} 获取余额失败，跳过余额限制: {e}")

        return qty

    def _apply_min_notional(self, qty: Decimal, price: float, filters: Dict, step: Decimal, max_qty: Decimal) -> Decimal:
        """应用最小名义价值限制"""
        from decimal import Decimal, ROUND_DOWN

        min_notional = Decimal(str(filters.get('minNotional', 5)))
        notional = qty * Decimal(str(price))

        if notional < min_notional and qty < max_qty:
            min_qty_for_notional = (min_notional / Decimal(str(price)))
            min_qty_for_notional = (min_qty_for_notional / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step
            return max(qty, min_qty_for_notional)

        return qty

    def _get_existing_notional(self, symbol: str) -> float:
        """获取已有持仓的名义价值"""
        try:
            positions = self.get_positions()
            for pos in positions:
                if pos.get('symbol') == symbol:
                    position_amt = float(pos.get('position_amt', 0))
                    entry_price = float(pos.get('entry_price', 0))
                    if position_amt != 0 and entry_price > 0:
                        notional = abs(position_amt * entry_price)
                        logger.info(f"[仓位限额] {symbol} 已有持仓(get_positions): amt={position_amt}, price={entry_price}, notional={notional}")
                        return notional
        except Exception as e:
            logger.warning(f"[仓位限额] {symbol} get_positions查询失败: {e}")
            return self._get_symbol_existing_notional(symbol)

        return 0.0
    
    # ==================== 交易接口 ====================
    

    def create_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: float = None,
                    stop_price: float = None, reduce_only: bool = False,
                    position_side: str = None, leverage: int = None, **kwargs) -> Dict:
        """
        创建订单
        
        Args:
            symbol: 交易对
            side: 方向 BUY/SELL
            order_type: 订单类型 MARKET/LIMIT/STOP_MARKET
            quantity: 数量
            price: 限价单价格
            stop_price: 止损触发价
            reduce_only: 是否只减仓
            position_side: 持仓方向 LONG/SHORT（HEDGING模式需要）
            leverage: 杠杆倍数（优先使用，未传入则从账户查询）
        """
        # ========== 参数验证 ==========
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol必须是非空字符串")
        
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"side必须是BUY或SELL，当前值: {side}")
        
        if order_type not in ['MARKET', 'LIMIT', 'STOP_MARKET', 'STOP', 'TAKE_PROFIT_MARKET', 'TAKE_PROFIT']:
            raise ValueError(f"order_type无效，当前值: {order_type}")
        
        if quantity is None or quantity <= 0:
            raise ValueError(f"quantity必须大于0，当前值: {quantity}")
        
        if order_type == 'LIMIT' and (price is None or price <= 0):
            raise ValueError(f"LIMIT订单必须指定有效的price，当前值: {price}")
        
        if order_type in ['STOP_MARKET', 'STOP', 'TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] and (stop_price is None or stop_price <= 0):
            raise ValueError(f"{order_type}订单必须指定有效的stop_price，当前值: {stop_price}")
        
        # ========== 第一步：读取币安的持仓模式 ==========
        position_mode = self.get_position_mode()
        logger.info(f"[create_order] {symbol} 持仓模式: {position_mode}")
        
        # ========== 第二步：读取杠杆 ==========
        if leverage is None:
            leverage = self._get_symbol_leverage(symbol)
        logger.info(f"[create_order] {symbol} 杠杆: {leverage}x")
        
        # ========== 第三步：读取下单数量限制并调整数量 ==========
        quantity = self.adjust_quantity_to_symbol(symbol, quantity, order_type, leverage=leverage, reduce_only=reduce_only)
        logger.info(f"[create_order] {symbol} 调整后数量: {quantity}")
        
        # 检查调整后的数量是否有效
        if quantity is None or quantity <= 0:
            raise ValueError(f"调整后数量无效（quantity={quantity}），可能是账户余额不足或交易限制")
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'newClientOrderId': kwargs.pop('newClientOrderId', None) or f"{BNFF_ORDER_PREFIX}{int(time.time()*1000)}"
        }
        
        # ========== 第四步：根据持仓模式设置参数 ==========
        if position_mode == 'HEDGING':
            # HEDGING模式：必须指定positionSide，不能用reduceOnly
            if position_side:
                params['positionSide'] = position_side
            else:
                # 自动判断positionSide：
                # 1. 如果reduceOnly=True，说明是平仓订单，positionSide必须与现有持仓相反
                # 2. 如果reduceOnly=False，说明是开仓订单，positionSide与side一致
                if reduce_only:
                    # 平仓订单：读取现有持仓确定positionSide
                    try:
                        positions = self.get_positions()
                        for pos in positions:
                            if pos.get('symbol') == symbol and float(pos.get('position_amt', 0)) != 0:
                                # 现有持仓方向
                                existing_side = 'LONG' if float(pos['position_amt']) > 0 else 'SHORT'
                                # 平仓时positionSide应该与现有持仓一致
                                params['positionSide'] = existing_side
                                logger.debug(f"{symbol} 平仓订单，自动设置positionSide={existing_side} (现有持仓)")
                                break
                        else:
                            # 无现有持仓，仍基于side判断
                            params['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
                    except Exception:
                        logger.debug(f"读取{symbol}持仓失败，基于side判断positionSide")
                        params['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
                else:
                    # 开仓订单：positionSide与side一致
                    params['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
        else:
            # ONE-WAY模式：可以使用reduceOnly
            if reduce_only:
                params['reduceOnly'] = 'true'
        
        # ========== 第五步：设置订单类型参数 ==========
        # 限价单
        if order_type == 'LIMIT':
            if price is None:
                raise ValueError("限价单必须指定价格")
            params['price'] = self.adjust_price_precision(symbol, price)
            params['timeInForce'] = kwargs.get('timeInForce', 'GTC')
        
        # 止损单
        if order_type in ['STOP_MARKET', 'STOP', 'TAKE_PROFIT_MARKET']:
            if stop_price is None:
                raise ValueError(f"{order_type}必须指定stop_price")
            params['stopPrice'] = self.adjust_price_precision(symbol, stop_price)
        
        if order_type == 'STOP':
            if price is None:
                raise ValueError("STOP订单必须指定价格")
            params['price'] = self.adjust_price_precision(symbol, price)
            params['timeInForce'] = kwargs.get('timeInForce', 'GTC')
        
        return self._request('POST', '/fapi/v1/order', params, leverage=leverage)
    
    # ==================== 条件止损单 ====================
    

    def create_algo_order(self, symbol: str, side: str, order_type: str,
                         quantity: float, stop_price: float,
                         position_side: str = None, **kwargs) -> Dict:
        """
        创建条件订单（止损单/止盈单）
        
        Args:
            symbol: 交易对
            side: 方向 BUY/SELL
            order_type: 订单类型 STOP_MARKET / TAKE_PROFIT_MARKET
            quantity: 数量
            stop_price: 触发价格
            position_side: 持仓方向 LONG/SHORT（HEDGING模式需要）
        """
        # 调整数量和价格精度
        quantity = self.adjust_quantity_precision(symbol, quantity)
        stop_price = self.adjust_price_precision(symbol, stop_price)
        
        # ✅ 修复：按官方文档和 trading_GPT 使用 triggerPrice
        # HEDGING 模式不能使用 reduceOnly，必须用 positionSide
        params = {
            'symbol': symbol,
            'side': side,
            'algoType': 'CONDITIONAL',  # 必填
            'type': order_type,          # 必填: STOP_MARKET/STOP/TAKE_PROFIT_MARKET/TAKE_PROFIT
            'triggerPrice': stop_price,
            'quantity': quantity,
            'clientOrderId': kwargs.pop('client_order_id', None) or f"{BNFF_ORDER_PREFIX}SL_{int(time.time()*1000)}",
        }
        
        # STOP 和 TAKE_PROFIT 需要 price 参数（限价）
        if order_type in ['STOP', 'TAKE_PROFIT']:
            price = self.adjust_price_precision(symbol, kwargs.get('price', stop_price))
            params['price'] = price
            params['timeInForce'] = kwargs.get('timeInForce', 'GTC')
        
        # workingType: 所有条件单类型都需要
        if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP', 'TAKE_PROFIT']:
            params['workingType'] = kwargs.get('workingType', 'MARK_PRICE')
        
        # 持仓模式处理 - HEDGING 模式必须传 positionSide
        position_mode = self.get_position_mode()
        
        if position_mode == 'HEDGING':
            if position_side:
                params['positionSide'] = position_side
            else:
                params['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
        else:
            params['reduceOnly'] = kwargs.get('reduceOnly', 'false')
        
        # 调试日志
        logger.info(f"[API] create_algo_order 请求: {params}")
        
        result = self._request('POST', '/fapi/v1/algoOrder', params)
        logger.info(f"[API] create_algo_order 响应: {result}")
        return result
    

    def cancel_algo_order(self, symbol: str, algo_id: int) -> Dict:
        """取消条件订单"""
        return self._request('DELETE', '/fapi/v1/algoOrder',
                            {'symbol': symbol, 'algoId': algo_id})
    

    def get_algo_open_orders(self, symbol: str = None) -> List[Dict]:
        """获取未成交条件订单"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._request('GET', '/fapi/v1/openAlgoOrders', params)
    

    def cancel_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """取消订单
        
        Args:
            symbol: 交易对
            order_id: 订单ID（优先使用）
            orig_client_order_id: 客户端订单ID（order_id为空时使用）
        """
        params = {'symbol': symbol}
        if order_id is not None:
            params['orderId'] = order_id
        elif orig_client_order_id is not None:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("cancel_order需要 order_id 或 orig_client_order_id 至少一个")
        return self._request('DELETE', '/fapi/v1/order', params)
    

    def cancel_all_orders(self, symbol: str) -> Dict:
        """取消所有订单"""
        return self._request('DELETE', '/fapi/v1/allOpenOrders',
                            {'symbol': symbol})
    

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """获取未成交订单"""
        params = {'symbol': symbol} if symbol else {}
        return self._request('GET', '/fapi/v1/openOrders', params)
    

    def get_all_orders(self, symbol: str, limit: int = 100) -> List[Dict]:
        """获取用户订单历史（已成交/已取消/全部）

        Args:
            symbol: 交易对
            limit: 返回数量（默认100，最大1000）

        Returns:
            List[Dict]: 订单列表
        """
        params = {'symbol': symbol, 'limit': limit}
        return self._request('GET', '/fapi/v1/allOrders', params)


    def get_account_trades(self, symbol: str = None, limit: int = 100, start_time: int = None, end_time: int = None) -> List[Dict]:
        """获取账户成交历史

        Args:
            symbol: 交易对（如不传则获取全部）
            limit: 返回数量（默认100，最大1000）
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）

        Returns:
            List[Dict]: 成交记录列表
        """
        # 测试网不支持 /fapi/v1/userTrades 接口，直接返回空列表避免大量404错误
        if self.testnet:
            logger.debug("get_account_trades: 测试网不支持userTrades接口，返回空列表")
            return []

        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._request('GET', '/fapi/v1/userTrades', params, signed=True)


    def get_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """查询订单"""
        params = {'symbol': symbol}
        if order_id is not None:
            params['orderId'] = order_id
        if orig_client_order_id is not None:
            params['origClientOrderId'] = orig_client_order_id
        return self._request('GET', '/fapi/v1/order', params)
    
    def verify_order_execution(self, symbol: str, order_id: int, max_wait_seconds: int = 10) -> Dict:
        """
        验证订单执行状态
        
        Args:
            symbol: 币种
            order_id: 订单ID
            max_wait_seconds: 最大等待时间（秒）
            
Returns:
            订单状态信息
        """
        for i in range(max_wait_seconds):
            try:
                order = self.get_order(symbol, order_id)
                status = order.get('status', '')
                
                if status == 'FILLED':
                    logger.info(f"✅ {symbol} 订单已成交: ID={order_id}")
                    return {'success': True, 'status': status, 'order': order}
                elif status in ['CANCELED', 'EXPIRED', 'REJECTED']:
                    logger.error(f"❌ {symbol} 订单失败: ID={order_id}, 状态={status}")
                    return {'success': False, 'status': status, 'order': order}
                
                time.sleep(1)
            except Exception as e:
                logger.warning(f"{symbol} 查询订单状态失败: {e}")
                time.sleep(1)
        
        logger.warning(f"⚠️ {symbol} 订单未在{max_wait_seconds}秒内成交: ID={order_id}")
        return {'success': False, 'status': 'TIMEOUT', 'order': None}
    
    def create_order_with_verify(self, symbol: str, side: str, order_type: str,
                                 quantity: float, verify: bool = True, **kwargs) -> Dict:
        """
        创建订单并验证执行状态
        
        Args:
            symbol: 币种
            side: 方向 BUY/SELL
            order_type: 订单类型
            quantity: 数量
            verify: 是否验证执行状态
            **kwargs: 其他参数
            
        Returns:
            订单信息（包含验证结果）
        """
        # 创建订单
        order = self.create_order(symbol, side, order_type, quantity, **kwargs)
        order_id = order.get('orderId')
        
        if not order_id:
            logger.error(f"{symbol} 订单创建失败：无orderId")
            return {'success': False, 'order': order}
        
        if not verify:
            return {'success': True, 'order': order}
        
        # 验证执行状态
        result = self.verify_order_execution(symbol, order_id)
        result['order'] = order
        return result
    
    # ==================== 精度调整 ====================
    
    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """调整数量精度 - 使用Decimal避免浮点精度问题，确保符合币安stepSize要求"""
        import math
        from decimal import Decimal, ROUND_DOWN
        
        info = self.get_symbol_info(symbol)
        for f in info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                step_size = Decimal(str(f.get('step_size', f.get('stepSize', 0.001))))
                
                # 使用Decimal精确计算：向下取整确保不超过最大允许值
                quantity_dec = Decimal(str(quantity))
                adjusted = (quantity_dec / step_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_size
                
                # 确保不小于最小值
                min_qty = Decimal(str(f.get('min_qty', f.get('minQty', 0))))
                if adjusted < min_qty:
                    adjusted = min_qty
                
                # 获取最大数量 - 优先使用MARKET_LOT_SIZE
                max_qty = f.get('max_qty', f.get('maxQty', 1000000))
                max_qty_dec = Decimal(str(max_qty))
                if adjusted > max_qty_dec:
                    adjusted = max_qty_dec
                
                # stepSize=1（精度为0）时返回整数
                if step_size == Decimal('1'):
                    return int(adjusted)
                
                return float(adjusted)
        
        return quantity
    
    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """调整价格精度 - 使用Decimal避免浮点精度问题，确保符合币安tick_size要求"""
        from decimal import Decimal, ROUND_DOWN
        
        info = self.get_symbol_info(symbol)
        for f in info.get('filters', []):
            if f['filterType'] == 'PRICE_FILTER':
                tick_size_str = f.get('tick_size', f.get('tickSize', '0.01'))
                tick_size = Decimal(tick_size_str)
                
                # 使用Decimal精确计算：先除后乘，然后四舍五入
                price_dec = Decimal(str(price))
                adjusted = (price_dec / tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_size
                
                # 关键：转换为字符串以去除浮点精度问题，再转回float
                # 例如：Decimal('0.04434') -> '0.04434' -> float -> 0.04434
                result_str = str(adjusted)
                return float(result_str)
        
        return price
    
    def get_min_quantity(self, symbol: str) -> float:
        """获取最小下单量"""
        info = self.get_symbol_info(symbol)
        for f in info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                return float(f['minQty'])
        return 0.001
    
    def get_max_leverage(self, symbol: str) -> int:
        """获取最大杠杆"""
        # 从缓存获取
        if symbol in self._leverage_brackets_cache:
            brackets = self._leverage_brackets_cache[symbol]
            if brackets:
                return max(b['initialLeverage'] for b in brackets)
        
        # 查询API
        try:
            data = self._request('GET', '/fapi/v1/leverageBracket', {'symbol': symbol})
            if data:
                brackets = data[0].get('brackets', [])
                self._leverage_brackets_cache[symbol] = brackets
                if brackets:
                    return max(b['initialLeverage'] for b in brackets)
        except Exception as e:
            logger.warning(f"获取杠杆档位失败: {e}")
        
        return 20  # 默认返回20x
    
    # ==================== 辅助方法 ====================
    
    def calculate_profit_rate(self, entry_price: float, current_price: float, 
                             side: str, leverage: int = 1) -> float:
        """计算收益率（含杠杆）"""
        if entry_price <= 0:
            return 0
        
        if side == 'LONG' or side == 'BUY':
            return (current_price - entry_price) / entry_price * 100 * leverage
        else:
            return (entry_price - current_price) / entry_price * 100 * leverage
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        try:
            self._request('GET', '/fapi/v1/ping', signed=False)
            return True
        except Exception:
            logger.debug("ping失败")
            return False
    
    def reconnect(self) -> bool:
        """重新连接"""
        logger.info("尝试重新连接...")
        try:
            self._test_connection()
            self._sync_time()
            self._connected = True
            logger.info("重新连接成功")
            return True
        except Exception as e:
            logger.error(f"重新连接失败: {e}")
            return False

    # ==================== WebSocket 用户数据流 ====================
    
    def create_listen_key(self) -> Optional[str]:
        """创建 listenKey 用于用户数据流"""
        try:
            data = self._request('POST', '/fapi/v1/listenKey')
            listen_key = data.get('listenKey')
            if listen_key:
                logger.info(f"创建 listenKey 成功")
                return listen_key
        except Exception as e:
            logger.error(f"创建 listenKey 失败: {e}")
        return None
    
    def keepalive_listen_key(self) -> bool:
        """延长 listenKey 有效期（建议每30分钟调用一次）"""
        try:
            self._request('PUT', '/fapi/v1/listenKey')
            logger.debug("listenKey 保活成功")
            return True
        except Exception as e:
            logger.warning(f"listenKey 保活失败: {e}")
            return False
    
    def close_listen_key(self) -> bool:
        """关闭 listenKey"""
        try:
            self._request('DELETE', '/fapi/v1/listenKey')
            logger.info("listenKey 已关闭")
            return True
        except Exception as e:
            logger.warning(f"关闭 listenKey 失败: {e}")
            return False

    def get_user_stream_url(self, listen_key: str = None) -> str:
        """获取用户数据流 WebSocket URL"""
        if not listen_key:
            listen_key = self.create_listen_key()
        if not listen_key:
            return None

        if self.testnet:
            # 测试网：主域名可能被 DNS 污染，但保留原地址供 VPN 用户使用
            return f"wss://fstream.binancefuture.com/ws/{listen_key}"
        else:
            # 实网：使用稳定的域名
            return f"wss://fstream.binance.com/ws/{listen_key}"

    def get_user_stream_url_backup(self, listen_key: str = None) -> str:
        """获取用户数据流 WebSocket 备用 URL"""
        if not listen_key:
            listen_key = self.create_listen_key()
        if not listen_key:
            return None

        if self.testnet:
            # 测试网备用：尝试 API 域名（可能更稳定）
            return f"wss://fstream.binancefuture.com/ws/{listen_key}"
        else:
            # 实网备用
            return f"wss://ws-fstream.binance.com/ws/{listen_key}"