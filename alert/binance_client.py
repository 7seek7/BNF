from binance.client import Client
from binance.exceptions import BinanceAPIException
from config.settings import settings
from utils.logger import Logger
from utils.helpers import retry_on_failure
import time

logger = Logger.get_logger('binance_client')

class BinanceClient:
    """币安客户端类 - 修复版"""
    
    def __init__(self, mode='live'):
        """
        初始化
        :param mode: 运行模式 'live', 'testnet', 'backtest'
        """
        self.mode = mode

        # 初始化 positionSide 模式缓存
        self._position_side_mode = None
        self._position_side_cache_time = 0

        # ✅ 测试连接标志
        self._connection_tested = False

        # ✅ 杠杆档位缓存（避免 API 限流）
        self._leverage_brackets_cache = {}
        self._leverage_brackets_cache_time = {}

        if mode == 'testnet':
            api_key = settings.TESTNET_API_KEY
            api_secret = settings.TESTNET_API_SECRET
            # ✅ 显式指定期货测试网 API URL，避免自动选择错误
            try:
                self.client = Client(
                    api_key,
                    api_secret,
                    testnet=True,
                    requests_params={'timeout': 30}  # 增加到30秒
                )
                # ✅ 立即测试期货连接，而不是依赖库的默认 ping
                self._test_futures_connection()
                logger.info(f"币安客户端初始化完成 - 模式: {mode} (期货测试网)")
            except Exception as e:
                logger.error(f"币安客户端初始化失败: {str(e)}")
                logger.error(f"  可能原因:")
                logger.error(f"  1. DNS解析失败: 检查网络连接")
                logger.error(f"  2. API密钥错误: 检查.env文件")
                logger.error(f"  3. 网络超时: 检查防火墙/代理设置")
                logger.error(f"  4. IP被封禁: 尝试更换网络或使用代理")
                logger.error(f"  5. 测试网API限制: 测试网可能有访问限制")
                # 抛出异常，让系统可以继续运行（而不是崩溃）
                raise
        elif mode == 'live':
            api_key = settings.BINANCE_API_KEY
            api_secret = settings.BINANCE_API_SECRET
            try:
                # ✅ 显式指定期货实盘 API
                self.client = Client(
                    api_key,
                    api_secret,
                    requests_params={'timeout': 30}  # 增加到30秒
                )
                # ✅ 立即测试期货连接，而不是依赖库的默认 ping
                self._test_futures_connection()
                logger.info(f"币安客户端初始化完成 - 模式: {mode} (期货实盘)")
            except Exception as e:
                logger.error(f"币安客户端初始化失败: {str(e)}")
                logger.error(f"  可能原因:")
                logger.error(f"  1. DNS解析失败: 检查网络连接")
                logger.error(f"  2. API密钥错误: 检查.env文件")
                logger.error(f"  3. 网络超时: 检查防火墙/代理设置")
                logger.error(f"  4. VPN未开启或代理设置错误")
                logger.error(f"  5. IP被封禁: 检查币安IP限制")
                raise
        else:
            # 对于其他模式（如backtest），使用默认配置
            try:
                if mode == 'backtest':
                    # 回测模式可能不需要实际连接
                    api_key = settings.BINANCE_API_KEY or "dummy"
                    api_secret = settings.BINANCE_API_SECRET or "dummy"
                    self.client = None  # 回测模式不需要实际客户端
                    logger.info(f"币安客户端初始化完成 - 模式: {mode} (模拟模式)")
                else:
                    # 未知模式，使用live配置
                    api_key = settings.BINANCE_API_KEY
                    api_secret = settings.BINANCE_API_SECRET
                    self.client = Client(api_key, api_secret)
                    logger.info(f"币安客户端初始化完成 - 模式: {mode} (默认live)")
            except Exception as e:
                logger.error(f"币安客户端初始化失败: {str(e)}")
                # 在backtest模式下不抛出异常
                if mode != 'backtest':
                    raise

    def _test_futures_connection(self):
        """
        测试期货 API 连接
        避免使用库的默认 ping（可能连接现货 API），直接调用期货接口测试
        """
        try:
            # ✅ 直接调用期货 API 测试连接，而不是默认的 ping
            server_time = self.client.futures_ping()
            logger.info("期货 API 连接测试成功")
            self._connection_tested = True
        except Exception as e:
            logger.error(f"期货 API 连接测试失败: {str(e)}")
            logger.error(f"  可能原因:")
            logger.error(f"  1. 测试网API不可用或维护中")
            logger.error(f"  2. 网络连接问题（代理/VPN/防火墙）")
            logger.error(f"  3. API密钥错误或权限不足")
            logger.error(f"  4. IP地址被封禁或限制")
            # 重新抛出异常
            raise

    def is_connected(self):
        """
        测试连接是否正常
        
        Returns:
            bool: 连接正常返回 True，否则返回 False
        """
        if self.client is None:
            logger.debug("连接测试失败: client 为 None（回测模式）")
            return False
        
        try:
            self.client.futures_ping()
            return True
        except Exception as e:
            logger.warning(f"连接测试失败: {str(e)}")
            return False

    def reconnect(self):
        """
        重新连接币安 API
        
        Returns:
            bool: 重连成功返回 True，否则返回 False
        """
        logger.info("尝试重新连接币安API...")
        try:
            if self.mode == 'testnet':
                api_key = settings.TESTNET_API_KEY
                api_secret = settings.TESTNET_API_SECRET
                self.client = Client(
                    api_key,
                    api_secret,
                    testnet=True,
                    requests_params={'timeout': 30}
                )
            elif self.mode == 'live':
                api_key = settings.BINANCE_API_KEY
                api_secret = settings.BINANCE_API_SECRET
                self.client = Client(
                    api_key,
                    api_secret,
                    requests_params={'timeout': 30}
                )
            else:
                logger.error(f"无法重连：未知模式 {self.mode}")
                return False
            
            # 测试新连接
            self._test_futures_connection()
            logger.info("重新连接成功")
            return True
        except Exception as e:
            logger.error(f"重新连接失败: {str(e)}")
            return False

    def _sync_time(self):
        """同步服务器时间"""
        try:
            # 获取服务器时间
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)

            # 计算时间差（毫秒）
            time_offset = server_time['serverTime'] - local_time

            logger.info(f"时间同步: 服务器={server_time['serverTime']}, "
                       f"本地={local_time}, 偏移={time_offset}ms")

            # 设置时间偏移（Binance Python Client 库支持）
            self.client.timestamp_offset = time_offset
                        # 增大recvWindow到10000ms（10秒），处理网络波动
            # 原因：VPN或网络不稳定时，延迟峰值可达3-5秒
            # 值: 10000ms 可处理大部分网络波动
            self.client.options = {
                'defaultType': 'future',
                'recvWindow': 10000  # 从3秒增加到10秒，处理网络波动
            }
            logger.debug(f"时间偏移已设置: {time_offset}ms, recvWindow=10000ms (处理网络波动)")
        except Exception as e:
            logger.warning(f"时间同步失败: {str(e)}")

    def _validate_client(self, method_name):
        """
        验证 client 是否可用
        :param method_name: 调用的方法名称（用于日志）
        :raises: RuntimeError 如果 client 为 None
        """
        if self.client is None:
            error_msg = f"{method_name}: 回测模式下不支持此方法（self.client=None）"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _is_valid_symbol(self, symbol):
        """
        检查symbol是否有效
        :param symbol: 币种符号
        :return: 是否有效
        """
        try:
            # 必须以USDT结尾
            if not symbol.endswith('USDT'):
                return False

            # 基础货币部分（去掉USDT）
            base_currency = symbol[:-4]

            # 检查长度（合理的加密货币名称长度）
            if len(base_currency) < 2 or len(base_currency) > 10:
                return False

            # 只允许字母、数字和一些特殊字符
            allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not all(c in allowed_chars for c in base_currency.upper()):
                return False

            # 不允许全数字
            if base_currency.isdigit():
                return False

            return True

        except Exception as e:
            logger.warning(f"验证symbol失败 {symbol}: {str(e)}")
            return False

    @retry_on_failure()
    def get_top_symbols(self, count=30):
        """获取交易量前N的币种"""
        self._validate_client("get_top_symbols")
        try:
            tickers = self.client.futures_ticker()

            # 过滤USDT合约
            valid_tickers = []
            for t in tickers:
                symbol = t['symbol']
                # 检查是否以USDT结尾
                if not symbol.endswith('USDT'):
                    continue
                # 检查成交量是否有效
                quote_volume = float(t.get('quoteVolume', 0))
                if quote_volume <= 0:
                    continue
                # 过滤掉包含非ASCII字符或奇怪字符的symbol
                if not self._is_valid_symbol(symbol):
                    logger.debug(f"过滤掉无效symbol: {symbol}")
                    continue
                valid_tickers.append(t)

            # 按成交量排序
            sorted_tickers = sorted(
                valid_tickers,
                key=lambda x: float(x.get('quoteVolume', 0)),
                reverse=True
            )

            # 取前N个
            top_symbols = [t['symbol'] for t in sorted_tickers[:count]]

            logger.info(f"获取到 {len(top_symbols)} 个交易量最高的币种")
            return top_symbols

        except BinanceAPIException as e:
            logger.error(f"获取币种列表失败 [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"获取币种列表失败: {str(e)}")
            raise
    
    @retry_on_failure()
    def get_klines(self, symbol, interval, limit=100):
        """
        获取K线数据
        :param symbol: 币种
        :param interval: 周期 '1m', '5m', '15m', '1h', '4h'
        :param limit: 数量
        """
        self._validate_client("get_klines")
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
            
        except BinanceAPIException as e:
            logger.error(f"获取K线失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"获取K线失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def get_ticker_price(self, symbol):
        """获取当前价格"""
        self._validate_client("get_ticker_price")
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"获取价格失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"获取价格失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def get_ticker_24h(self, symbol):
        """获取24小时行情"""
        self._validate_client("get_ticker_24h")
        try:
            ticker = self.client.futures_ticker(symbol=symbol)
            return ticker
        except Exception as e:
            logger.error(f"获取24小时行情失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def get_open_interest(self, symbol):
        """获取持仓量"""
        self._validate_client("get_open_interest")
        try:
            data = self.client.futures_open_interest(symbol=symbol)
            # 币安 API 返回的是字符串格式的数字，如 "1234567890"
            # 需要直接转换为 float，而不是从字典中获取
            if isinstance(data, str):
                return float(data)
            elif isinstance(data, (int, float)):
                return float(data)
            elif isinstance(data, dict) and 'openInterest' in data:
                return float(data['openInterest'])
            else:
                logger.warning(f"{symbol} 持仓量数据格式异常: {type(data)} = {data}")
                return None
        except Exception as e:
            logger.warning(f"获取持仓量失败 {symbol}: {str(e)}")
            return None
    
    @retry_on_failure()
    def get_funding_rate(self, symbol):
        """获取资金费率"""
        self._validate_client("get_funding_rate")
        try:
            data = self.client.futures_funding_rate(symbol=symbol, limit=1)
            if data:
                return float(data[0]['fundingRate']) * 100
            return None
        except Exception as e:
            logger.warning(f"获取资金费率失败 {symbol}: {str(e)}")
            return None
    
    @retry_on_failure()
    def get_account_balance(self):
        """获取账户余额"""
        self._validate_client("get_account_balance")
        try:
            account = self.client.futures_account()

            total_balance = float(account['totalWalletBalance'])
            available_balance = float(account['availableBalance'])
            total_margin = float(account['totalInitialMargin'])
            unrealized_pnl = float(account['totalUnrealizedProfit'])
            # 获取账户的 positionSide 模式
            position_side = account.get('positionSide', 'ONE-WAY')

            return {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'total_margin': total_margin,
                'unrealized_pnl': unrealized_pnl,
                'position_side': position_side  # ONE-WAY 或 HEDGING
            }
        except BinanceAPIException as e:
            logger.error(f"获取账户余额失败 [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"获取账户余额失败: {str(e)}")
            raise
    
    @retry_on_failure()
    def get_leverage_bracket(self, symbol):
        """获取杠杆档位信息

        返回该币种支持的最大杠杆倍数
        """
        try:
            # 获取该币种的杠杆档位信息
            brackets = self.get_leverage_brackets(symbol)
            if brackets and len(brackets) > 0:
                # 返回最大杠杆倍数
                max_leverage = max(b['initialLeverage'] for b in brackets)
                logger.debug(f"获取杠杆档位 {symbol}: 最大杠杆{max_leverage}x")
                return max_leverage
            else:
                # 如果获取失败，使用保守的默认值
                logger.warning(f"{symbol} 无法获取杠杆档位信息，使用默认值20x")
                return 20
        except Exception as e:
            logger.warning(f"获取杠杆档位失败 {symbol}: {str(e)}，使用默认值20x")
            return 20

    @retry_on_failure()
    def get_max_notional_for_leverage(self, symbol, leverage):
        """获取指定杠杆的最大名义价值

        注意：当前实现返回保守的默认值，后续可根据账户VIP等级调整
        """
        # 币安名义价值限制：
        # - 普通账户：根据杠杆和币种不同而异
        # - 通常1x杠杆对应更大的名义价值
        # 这里返回一个保守的默认值：100万美元名义价值
        max_notional = 1000000  # 100万美元
        logger.debug(f"获取名义价值限制 {symbol} {leverage}x: 使用默认值{max_notional}")
        return max_notional

    @retry_on_failure()
    def get_position(self, symbol):
        """获取持仓信息"""
        self._validate_client("get_position")
        try:
            positions = self.client.futures_position_information(symbol=symbol)

            for pos in positions:
                if pos['symbol'] == symbol:
                    position_amt = float(pos['positionAmt'])
                    if position_amt != 0:
                        # 使用币安API提供的数据计算回报率
                        unrealized_pnl = float(pos['unRealizedProfit'])

                        # 修复：使用get方法处理缺失的initialMargin字段
                        # 测试网API可能不返回此字段，需要从其他字段计算或使用默认值
                        initial_margin = float(pos.get('initialMargin', 0))

                        # 如果initialMargin不存在，尝试从其他字段计算
                        if initial_margin == 0:
                            # 使用isolatedMargin或maxNotionalValue作为替代
                            isolated_margin = float(pos.get('isolatedMargin', 0))
                            notional_value = abs(position_amt) * float(pos['entryPrice'])
                            max_notional = float(pos.get('maxNotionalValue', 0))

                            # 优先使用isolatedMargin，否则根据notional和leverage估算
                            if isolated_margin > 0:
                                initial_margin = isolated_margin
                            elif max_notional > 0:
                                initial_margin = notional_value / float(pos['leverage'])
                            else:
                                # 兜底：根据notional和leverage计算
                                initial_margin = notional_value / float(pos['leverage'])

                        # 计算回报率百分比（基于保证金）
                        if initial_margin > 0:
                            profit_rate = (unrealized_pnl / initial_margin) * 100
                        else:
                            profit_rate = 0

                        return {
                            'symbol': symbol,
                            'position_amt': position_amt,
                            'entry_price': float(pos['entryPrice']),
                            'unrealized_pnl': unrealized_pnl,
                            'profit_rate': profit_rate,
                            'leverage': int(pos['leverage']),
                            'side': 'LONG' if position_amt > 0 else 'SHORT',
                            'margin': initial_margin,
                            'notional': float(pos.get('notional', 0)),
                            'isolated_wallet_balance': float(pos.get('isolatedWalletBalance', 0))
                        }

            return None
        except BinanceAPIException as e:
            logger.error(f"获取持仓信息失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"获取持仓信息失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def create_order(self, symbol, side, order_type, quantity, price=None, stop_price=None, **kwargs):
        """
        创建订单
        :param symbol: 币种
        :param side: BUY/SELL
        :param order_type: MARKET/LIMIT/STOP_MARKET/STOP
        :param quantity: 数量
        :param price: 限价单价格（LIMIT和STOP订单需要）
        :param stop_price: 止损价格（STOP_MARKET和STOP订单需要）
        :param **kwargs: 额外参数（如 timeInForce）
        """
        self._validate_client("create_order")
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }

            if order_type == 'LIMIT':
                if price is None:
                    raise ValueError("限价单必须指定价格")
                params['price'] = price
                # 默认 timeInForce='GTC'，但允许通过 kwargs 覆盖
                params['timeInForce'] = kwargs.get('timeInForce', 'GTC')
            elif order_type == 'STOP_MARKET':
                if stop_price is None:
                    raise ValueError("止损市价单必须指定止损价格")
                params['stopPrice'] = stop_price
            elif order_type == 'STOP':
                if price is None or stop_price is None:
                    raise ValueError("止损限价单必须指定价格和止损价格")
                params['price'] = price
                params['stopPrice'] = stop_price
                # 默认 timeInForce='GTC'，但允许通过 kwargs 覆盖
                params['timeInForce'] = kwargs.get('timeInForce', 'GTC')
            else:
                # MARKET 类型不需要 timeInForce 参数，不要添加
                pass
             
            # 添加其他额外参数
            for key, value in kwargs.items():
                if key not in params:  # 避免覆盖已设置的参数
                    params[key] = value
            
            # 获取账户的 positionSide 模式（每次都获取，避免缓存错误）
            try:
                account = self.client.futures_account()
                position_side_mode = account.get('positionSide', 'ONE-WAY')
                logger.debug(f"创建订单时账户 positionSide 模式: {position_side_mode}")
            except Exception as e:
                logger.warning(f"获取账户 positionSide 失败: {str(e)}")
                position_side_mode = 'ONE-WAY'  # 默认使用ONE-WAY
            
            # 只在 HEDGING 模式下才添加 positionSide 参数（从kwargs中获取）
            try:
                account = self.client.futures_account()
                position_side_mode = account.get('positionSide', 'ONE-WAY')
                logger.debug(f"创建订单时账户 positionSide 模式: {position_side_mode}")
            except Exception as e:
                logger.warning(f"获取账户 positionSide 失败: {str(e)}")
                position_side_mode = 'ONE-WAY'  # 默认使用ONE-WAY

            # 记录完整参数用于调试
            logger.info(f"[DEBUG] create_order 参数: symbol={symbol}, side={side}, type={order_type}, quantity={quantity}")
            logger.info(f"[DEBUG] position_side_mode={position_side_mode}, kwargs中的positionSide={kwargs.get('positionSide', 'NOT_PROVIDED')}")

            # 只在 HEDGING 模式下才添加 positionSide 参数
            if position_side_mode == 'HEDGING':
                if 'positionSide' in kwargs:
                    params['positionSide'] = kwargs['positionSide']
                    logger.debug(f"HEDGING模式: 添加positionSide={kwargs['positionSide']}")
                else:
                    # HEDGING模式必须提供positionSide
                    logger.warning(f"HEDGING模式但未提供positionSide，使用默认LONG")
                    params['positionSide'] = 'LONG'
            else:
                # ONE-WAY 模式：绝对不发送positionSide参数
                # 币安API在ONE-WAY模式下会自动根据side(BUY/SELL)判断持仓方向
                # 发送positionSide会导致 -4061 错误：Order's position side does not match user's setting
                logger.debug(f"ONE-WAY模式: 忽略kwargs中的positionSide（如有），不发送该参数")
                # 不添加positionSide参数到params中

            # 记录最终参数
            logger.info(f"[DEBUG] 最终API参数: {params}")

            order = self.client.futures_create_order(**params)
            logger.info(f"订单创建成功: {symbol} {side} {order_type} {quantity}")
            return order

        except BinanceAPIException as e:
            logger.error(f"创建订单失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"创建订单失败 {symbol}: {str(e)}")
            raise

    @retry_on_failure()
    def create_algo_order(self, symbol, side, trigger_price, quantity=None,
                          order_type='STOP_MARKET', price=None, reduce_only='false',
                          working_type='CONTRACT_PRICE', position_side=None,
                          client_algo_id=None, **kwargs):
        """
        创建 Algo 订单（条件订单）- 使用币安新的 /fapi/v1/algoOrder 端点

        币安从2025-12-09起要求条件订单必须使用此端点
        支持的订单类型: STOP_MARKET, TAKE_PROFIT_MARKET, STOP, TAKE_PROFIT, TRAILING_STOP_MARKET

        Args:
            symbol: 币种，如 'BTCUSDT'
            side: 'BUY' 或 'SELL'
            trigger_price: 触发价格
            quantity: 订单数量（可选，closePosition=True时不需要）
            order_type: 订单类型 (STOP_MARKET/TAKE_PROFIT_MARKET/STOP/TAKE_PROFIT/TRAILING_STOP_MARKET)
            price: 限价单价格（STOP/TAKE_PROFIT 订单需要）
            reduce_only: 'true' 或 'false'，是否为减仓订单
            working_type: 触发价格类型 MARK_PRICE or CONTRACT_PRICE
            position_side: 持仓方向（HEDGING模式需要）
            client_algo_id: 客户端算法订单ID（可选）
            **kwargs: 额外参数（如 timeInForce, priceProtect 等）

        Returns:
            dict: 创建成功返回 algoId, clientAlgoId 等信息
        """
        self._validate_client("create_algo_order")
        try:
            # 基础参数（严格按照币安官方文档命名）
            params = {
                'symbol': symbol,
                'side': side,
                'algoType': 'CONDITIONAL',  # 目前只支持 CONDITIONAL
                'type': order_type,
                'triggerPrice': trigger_price,
                'workingType': working_type,
                'reduceOnly': reduce_only
            }

            # 添加可选参数
            if quantity is not None:
                params['quantity'] = quantity

            if price is not None:
                params['price'] = price

            if client_algo_id:
                params['clientAlgoId'] = client_algo_id

            # 处理 positionSide（HEDGING 模式需要）
            try:
                account = self.client.futures_account()
                position_side_mode = account.get('positionSide', 'ONE-WAY')
                if position_side_mode == 'HEDGING':
                    if position_side:
                        params['positionSide'] = position_side
                    else:
                        params['positionSide'] = 'LONG'  # HEDGING 模式默认
                # ONE-WAY 模式不添加 positionSide
            except Exception as e:
                logger.debug(f"获取 positionSide 失败: {str(e)}")

            # 添加其他额外参数（包括 timeInForce, priceProtect 等）
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value

            logger.info(f"创建 Algo 订单: symbol={symbol}, side={side}, type={order_type}, trigger={trigger_price:.6f}, qty={quantity}")

            # 使用 python-binance 的 _request_futures_api 方法
            # algoOrder 是新的端点，可能没封装，但可以尝试
            response = self.client._request_futures_api('post', 'algoOrder', True, data=params)

            logger.info(f"Algo 订单创建成功: algoId={response.get('algoId')}, symbol={symbol}")
            return response

        except BinanceAPIException as e:
            logger.error(f"创建 Algo 订单失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"创建 Algo 订单失败 {symbol}: {str(e)}")
            raise

    @retry_on_failure()
    def cancel_algo_order(self, symbol, algo_id=None, client_algo_id=None):
        """
        取消 Algo 订单

        Args:
            symbol: 币种
            algo_id: 算法订单ID（与 client_algo_id 二选一）
            client_algo_id: 客户端算法订单ID（与 algo_id 二选一）

        Returns:
            dict: 取消结果
        """
        self._validate_client("cancel_algo_order")
        try:
            params = {'symbol': symbol}

            # algoId 与 clientAlgoId 必须至少发送一个
            if algo_id:
                params['algoId'] = algo_id
            elif client_algo_id:
                params['clientAlgoId'] = client_algo_id
            else:
                raise ValueError("必须提供 algoId 或 clientAlgoId 之一")

            response = self.client._request_futures_api('delete', 'algoOrder', True, data=params)

            logger.info(f"Algo 订单已取消: symbol={symbol}, algoId={algo_id}")
            return response

        except BinanceAPIException as e:
            logger.error(f"取消 Algo 订单失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"取消 Algo 订单失败 {symbol}: {str(e)}")
            raise

    @retry_on_failure()
    def query_algo_order(self, symbol, algo_id=None, client_algo_id=None):
        """
        查询 Algo 订单

        Args:
            symbol: 币种
            algo_id: 算法订单ID（与 client_algo_id 二选一）
            client_algo_id: 客户端算法订单ID（与 algo_id 二选一）

        Returns:
            list: 订单列表（API返回数组）
        """
        self._validate_client("query_algo_order")
        try:
            params = {'symbol': symbol}

            # 至少需要发送 algoId 与 clientAlgoId 中的一个
            if algo_id:
                params['algoId'] = algo_id
            elif client_algo_id:
                params['clientAlgoId'] = client_algo_id
            else:
                raise ValueError("必须提供 algoId 或 clientAlgoId 之一")

            response = self.client._request_futures_api('get', 'algoOrder', True, data=params)

            logger.debug(f"查询 Algo 订单: symbol={symbol}, count={len(response)}")
            return response

        except BinanceAPIException as e:
            logger.error(f"查询 Algo 订单失败 {symbol} [API]: {e.message}")
            return []
        except Exception as e:
            logger.error(f"查询 Algo 订单失败 {symbol}: {str(e)}")
            return []

    @retry_on_failure()
    def get_algo_open_orders(self, symbol=None):
        """
        获取当前开放的 Algo 订单

        Args:
            symbol: 币种（可选，不指定则获取所有）

        Returns:
            list: 开放的算法订单列表
        """
        self._validate_client("get_algo_open_orders")
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol

            # GET /fapi/v1/openAlgoOrders
            response = self.client._request_futures_api('get', 'openAlgoOrders', True, data=params)

            logger.debug(f"获取到 {len(response)} 个开放的 Algo 订单")
            return response

        except BinanceAPIException as e:
            logger.error(f"获取开放 Algo 订单失败 [API]: {e.message}")
            return []
        except Exception as e:
            logger.error(f"获取开放 Algo 订单失败: {str(e)}")
            return []

    @retry_on_failure()
    def cancel_order(self, symbol, order_id):
        """取消订单"""
        self._validate_client("cancel_order")
        try:
            result = self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"订单已取消: {symbol} {order_id}")
            return result
        except BinanceAPIException as e:
            logger.error(f"取消订单失败 {symbol} {order_id} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"取消订单失败 {symbol} {order_id}: {str(e)}")
            raise
    
    @retry_on_failure()
    def cancel_all_orders(self, symbol):
        """取消所有订单"""
        self._validate_client("cancel_all_orders")
        try:
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"已取消所有订单: {symbol}")
            return result
        except BinanceAPIException as e:
            logger.warning(f"取消所有订单失败 {symbol} [API]: {e.message}")
            return None
        except Exception as e:
            logger.warning(f"取消所有订单失败 {symbol}: {str(e)}")
            return None
    
    @retry_on_failure()
    def get_open_orders(self, symbol=None):
        """获取未成交订单"""
        self._validate_client("get_open_orders")
        try:
            if symbol:
                orders = self.client.futures_get_open_orders(symbol=symbol)
            else:
                orders = self.client.futures_get_open_orders()
            return orders
        except Exception as e:
            logger.error(f"获取未成交订单失败: {str(e)}")
            return []
    
    @retry_on_failure()
    def get_all_orders(self, symbol, limit=100):
        """
        获取某个币种的所有订单历史
        
        Args:
            symbol: 币种符号
            limit: 返回数量限制（最多1000）
        
        Returns:
            List[Dict]: 订单列表
        """
        try:
            orders = self.client.futures_get_all_orders(symbol=symbol, limit=limit)
            return orders
        except BinanceAPIException as e:
            logger.error(f"获取订单历史失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"获取订单历史失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def get_user_trades(self, symbol=None, limit=100):
        """
        获取成交历史
        
        Args:
            symbol: 币种符号（为None则获取所有）
            limit: 返回数量限制（最多1000）
        
        Returns:
            List[Dict]: 成交记录列表
        """
        try:
            if symbol:
                trades = self.client.futures_account_trades(symbol=symbol, limit=limit)
            else:
                trades = self.client.futures_account_trades(limit=limit)
            return trades
        except BinanceAPIException as e:
            logger.error(f"获取成交历史失败 {symbol} [API]: {e.message}")
            raise
        except Exception as e:
            logger.error(f"获取成交历史失败 {symbol}: {str(e)}")
            raise
    
    @retry_on_failure()
    def get_order(self, symbol, order_id):
        """查询订单状态"""
        try:
            order = self.client.futures_get_order(symbol=symbol, orderId=order_id)
            return order
        except BinanceAPIException as e:
            # 特殊处理订单不存在的错误
            if e.code == -2013:  # Order does not exist
                logger.warning(f"订单不存在 {symbol} {order_id}: {e.message}")
                return None
            else:
                logger.error(f"查询订单失败 {symbol} {order_id} [API]: {e.message}")
                raise
        except Exception as e:
            logger.error(f"查询订单失败 {symbol} {order_id}: {str(e)}")
            return None
    
    def wait_order_filled(self, symbol, order_id, timeout=30, check_interval=1):
        """
        等待订单成交
        
        Args:
            symbol: 币种
            order_id: 订单ID
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）
        
        Returns:
            (is_filled, order_info): 是否成交, 订单信息
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                order = self.get_order(symbol, order_id)
                
                if not order:
                    logger.warning(f"{symbol} 订单 {order_id} 查询失败")
                    return False, None
                
                status = order.get('status', '')
                
                if status == 'FILLED':
                    logger.info(f"{symbol} 订单 {order_id} 已完全成交")
                    return True, order
                elif status == 'PARTIALLY_FILLED':
                    logger.info(f"{symbol} 订单 {order_id} 部分成交")
                elif status in ['CANCELED', 'EXPIRED', 'REJECTED']:
                    logger.warning(f"{symbol} 订单 {order_id} 状态: {status}")
                    return False, order
                
                time.sleep(check_interval)
            
            # 超时
            logger.warning(f"{symbol} 订单 {order_id} 超时（{timeout}秒）未成交")
            return False, None
            
        except Exception as e:
            logger.error(f"{symbol} 等待订单成交失败 {order_id}: {str(e)}")
            return False, None
    
    @retry_on_failure()
    def change_leverage(self, symbol, leverage):
        """修改杠杆"""
        self._validate_client("change_leverage")
        try:
            result = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"{symbol} 杠杆已设置为 {leverage}x")
            return result
        except BinanceAPIException as e:
            # 可能已经是该杠杆，不算错误
            if 'No need to change leverage' in str(e):
                logger.debug(f"{symbol} 杠杆无需修改")
                return None
            logger.warning(f"设置杠杆失败 {symbol} [API]: {e.message}")
            return None
        except Exception as e:
            logger.warning(f"设置杠杆失败 {symbol}: {str(e)}")
            return None
    
    @retry_on_failure()
    def change_margin_type(self, symbol, margin_type):
        """修改保证金模式"""
        self._validate_client("change_margin_type")
        try:
            result = self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            logger.info(f"{symbol} 保证金模式已设置为 {margin_type}")
            return result
        except BinanceAPIException as e:
            # 可能已经是该模式，不算错误
            if 'No need to change margin type' in str(e):
                logger.debug(f"{symbol} 保证金模式无需修改")
                return None
            logger.debug(f"设置保证金模式 {symbol}: {e.message}")
            return None
        except Exception as e:
            logger.debug(f"设置保证金模式 {symbol}: {str(e)}")
            return None
    
    @retry_on_failure()
    def get_exchange_info(self):
        """获取交易规则"""
        try:
            info = self.client.futures_exchange_info()
            return info
        except Exception as e:
            logger.error(f"获取交易规则失败: {str(e)}")
            return None
    
    def get_symbol_info(self, symbol):
        """获取单个币种交易规则"""
        try:
            # 使用缓存避免频繁调用
            if not hasattr(self, '_exchange_info_cache'):
                self._exchange_info_cache = {}
                self._cache_time = {}

            # 缓存5分钟
            current_time = time.time()
            if symbol in self._exchange_info_cache:
                if current_time - self._cache_time.get(symbol, 0) < 300:
                    return self._exchange_info_cache[symbol]

            # 获取交易规则
            info = self.get_exchange_info()
            if not info:
                return None

            for s in info['symbols']:
                if s['symbol'] == symbol:
                    self._exchange_info_cache[symbol] = s
                    self._cache_time[symbol] = current_time
                    return s

            return None
        except Exception as e:
            logger.error(f"获取币种信息失败 {symbol}: {str(e)}")
            return None

    def get_tick_size_and_precision(self, symbol):
        """
        获取币种的tick_size和精度信息

        Args:
            symbol: 币种

        Returns:
            dict: {'tick_size': float, 'price_precision': int, 'qty_precision': int}
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"{symbol} 无法获取交易规则，使用默认值")
                return {'tick_size': 0.0001, 'price_precision': 4, 'qty_precision': 3}

            # 遍历filters找到相关信息
            tick_size = 0.0001
            price_precision = 4
            qty_precision = 3

            for f in symbol_info.get('filters', []):
                filter_type = f.get('filterType')

                if filter_type == 'PRICE_FILTER':
                    tick_size = float(f.get('tickSize', 0.0001))
                    # 从tick_size推算价格精度
                    if tick_size == 1:
                        price_precision = 0
                    elif tick_size == 0.1:
                        price_precision = 1
                    elif tick_size == 0.01:
                        price_precision = 2
                    elif tick_size == 0.001:
                        price_precision = 3
                    elif tick_size == 0.0001:
                        price_precision = 4
                    elif tick_size == 0.00001:
                        price_precision = 5
                    elif tick_size == 0.000001:
                        price_precision = 6
                    else:
                        # 从tickSize的小数位数推算精度
                        tick_str = str(tick_size)
                        if '.' in tick_str:
                            price_precision = len(tick_str.split('.')[1])
                        else:
                            price_precision = 0

                elif filter_type == 'LOT_SIZE':
                    step_size = float(f.get('stepSize', 0.001))
                    # 从stepSize计算数量精度（正确方法）
                    # stepSize=1 → 精度=0, stepSize=0.1 → 精度=1, stepSize=0.001 → 精度=3
                    step_str = str(step_size)
                    if '.' in step_str:
                        # 移除末尾的0
                        step_str = step_str.rstrip('0')
                        qty_precision = len(step_str.split('.')[-1])
                    else:
                        qty_precision = 0

            logger.debug(f"{symbol} tick_size={tick_size}, price_precision={price_precision}, qty_precision={qty_precision}")
            return {
                'tick_size': tick_size,
                'price_precision': price_precision,
                'qty_precision': qty_precision
            }
        except Exception as e:
            logger.error(f"{symbol} 获取tick_size和精度失败: {str(e)}")
            return {'tick_size': 0.0001, 'price_precision': 4, 'qty_precision': 3}

    def get_min_order_info(self, symbol, leverage: int):
        """
        获取币种的最小订单信息（基于固定的minQty计算最小保证金）
        
        根据币安API文档，LOT_SIZE和NOTIONAL filter的值是固定的，
        我们可以基于minQty和任意价格计算出最小的实际风险。
        
        Args:
            symbol: 币种
            leverage: 杠杆倍数
            
        Returns:
            dict: {
                'min_qty': float,          # 最小下单数量（LOT_SIZE minQty）
                'step_size': float,        # 数量精度步进（LOT_SIZE stepSize）
                'max_qty': float,          # 最大下单数量（LOT_SIZE maxQty）
                'min_notional': float,     # 最小名义价值（NOTIONAL minNotional）
                'min_margin_standard': float,  # 基于minQty的标准单位保证金（数量×价格/杠杆）
                'min_margin_for_notional': float  # 满足minNotional所需的最小保证金
            }
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"{symbol} 无法获取交易规则，使用默认值")
                return {
                    'min_qty': 0.001,
                    'step_size': 0.001,
                    'max_qty': float('inf'),
                    'min_notional': 5.0,
                    'min_margin_standard': 0.25,
                    'min_margin_for_notional': 0.25
                }

            min_qty = 0.001
            step_size = 0.001
            max_qty = float('inf')
            min_notional = 5.0

            for f in symbol_info.get('filters', []):
                filter_type = f.get('filterType')

                if filter_type == 'LOT_SIZE':
                    min_qty = float(f.get('minQty', 0.001))
                    step_size = float(f.get('stepSize', 0.001))
                    max_qty = float(f.get('maxQty', float('inf')))
                elif filter_type == 'NOTIONAL':
                    min_notional = float(f.get('minNotional', 5.0))

            # ✅ 基于minQty计算：这是每次下单的最小"币种单位"
            # 不管价格多少，至少要买这么多币种单位
            # 实际保证金 = min_qty × 当前价格 / 杠杆
            current_price = self.get_ticker_price(symbol)
            min_margin_standard = (min_qty * current_price) / leverage

            # ✅ 基于minNotional计算：满足最小名义价值所需的最小保证金
            min_margin_for_notional = min_notional / leverage

            logger.debug(f"{symbol} 最小订单信息: min_qty={min_qty}, step_size={step_size}, "
                        f"min_notional={min_notional} USDT, minMarginStandard={min_margin_standard:.2f} "
                        f"(价格={current_price}), minMarginForNotional={min_margin_for_notional:.2f}")

            return {
                'min_qty': min_qty,
                'step_size': step_size,
                'max_qty': max_qty,
                'min_notional': min_notional,
                'min_margin_standard': min_margin_standard,
                'min_margin_for_notional': min_margin_for_notional
            }

        except Exception as e:
            logger.error(f"{symbol} 获取最小订单信息失败: {str(e)}")
            return {
                'min_qty': 0.001,
                'step_size': 0.001,
                'max_qty': float('inf'),
                'min_notional': 5.0,
                'min_margin_standard': 0.25,
                'min_margin_for_notional': 0.25
            }

    def round_price(self, symbol, price):
        """
        按照币安规则对价格进行四舍五入

        Args:
            symbol: 币种
            price: 价格

        Returns:
            float: 调整后的价格
        """
        try:
            precision_info = self.get_tick_size_and_precision(symbol)
            tick_size = precision_info['tick_size']

            # 按照tick_size对齐
            rounded = round(price / tick_size) * tick_size

            logger.debug(f"{symbol} 价格调整: {price} -> {rounded} (tick_size={tick_size})")
            return rounded
        except Exception as e:
            logger.warning(f"{symbol} 价格调整失败: {str(e)}")
            return price

    def round_quantity(self, symbol, quantity):
        """
        按照币安规则对数量进行四舍五入

        Args:
            symbol: 币种
            quantity: 数量

        Returns:
            float: 调整后的数量
        """
        try:
            precision_info = self.get_tick_size_and_precision(symbol)
            qty_precision = precision_info['qty_precision']

            # 按照精度四舍五入
            rounded = round(quantity, qty_precision)

            logger.debug(f"{symbol} 数量调整: {quantity} -> {rounded} (precision={qty_precision})")
            return rounded
        except Exception as e:
            logger.warning(f"{symbol} 数量调整失败: {str(e)}")
            return quantity
    @retry_on_failure()
    def get_leverage_bracket(self, symbol=None):
        """
        获取币种最大杠杆

        :param symbol: 币种
        :return: 最大杠杆倍数（int）
        """
        try:
            # 获取交易规则
            info = self.get_exchange_info()
            if not info:
                logger.debug(f"{symbol} 无法获取杠杆档位信息，使用默认值")
                return 125

            # 查找该币种
            symbols = info.get('symbols', [])
            for s in symbols:
                if s.get('symbol') == symbol:
                    # 返回默认杠杆作为最大杠杆
                    max_leverage = int(s.get('defaultLeverage', 125))
                    logger.debug(f"{symbol} 最大杠杆: {max_leverage}x")
                    return max_leverage

            logger.debug(f"{symbol} 未找到交易规则，使用默认杠杆 125x")
            return 125

        except Exception as e:
            logger.error(f"获取杠杆信息失败 {symbol}: {str(e)}")
            return 125

    @retry_on_failure()
    def get_leverage_brackets(self, symbol=None):
        """
        获取杠杆档位详细信息（包括持仓价值限制）- 带缓存

        Args:
            symbol: 币种，如果为None则返回所有币种

        Returns:
            symbol=None: 返回所有币种的杠杆档位信息
            symbol!=None: 返回该币种的杠杆档位列表
        """
        try:
            # ✅ 检查缓存（5分钟有效期）
            cache_key = symbol or 'all'
            current_time = time.time()

            if cache_key in self._leverage_brackets_cache:
                if current_time - self._leverage_brackets_cache_time.get(cache_key, 0) < 300:
                    logger.debug(f"从缓存获取杠杆档位: {cache_key}")
                    return self._leverage_brackets_cache[cache_key]

            # Binance Python库只有单数形式的futures_leverage_bracket方法
            if symbol:
                # 获取单个币种的杠杆档位
                try:
                    brackets = self.client.futures_leverage_bracket(symbol=symbol)
                    # API返回的是list格式: [{'symbol': 'BTCUSDT', 'brackets': [...]}]
                    if isinstance(brackets, list) and len(brackets) > 0:
                        bracket_data = brackets[0]  # 获取第一个元素
                        if 'brackets' in bracket_data and len(bracket_data['brackets']) > 0:
                            result = []
                            for b in bracket_data['brackets']:
                                result.append({
                                    'bracket': b.get('bracket'),
                                    'initialLeverage': b.get('initialLeverage'),
                                    'notionalCap': b.get('notionalCap'),  # 最大持仓价值
                                    'notionalFloor': b.get('notionalFloor')
                                })
                            # ✅ 缓存结果
                            self._leverage_brackets_cache[cache_key] = result
                            self._leverage_brackets_cache_time[cache_key] = current_time
                            return result
                    return None
                except Exception as e:
                    logger.warning(f"获取单个币种杠杆档位失败 {symbol}: {str(e)}")
                    return None
            else:
                # 获取所有币种的杠杆档位（这里暂时不支持，API效率太低）
                logger.warning("获取所有币种杠杆档位暂不支持，请指定具体币种")
                return None

        except Exception as e:
            logger.error(f"获取杠杆档位详情失败 {symbol}: {str(e)}")
            return None

    def get_max_notional_for_leverage(self, symbol, leverage: int) -> float:
        """
        获取指定杠杆下的最大持仓价值

        Args:
            symbol: 币种
            leverage: 杠杆倍数

        Returns:
            float: 最大持仓价值（USDT），如果获取失败返回0
        """
        try:
            brackets = self.get_leverage_brackets(symbol)
            if not brackets:
                logger.debug(f"{symbol} 无法获取杠杆档位信息，使用默认值")
                return 0

            # 找到指定杠杆对应的档位
            for bracket in brackets:
                if bracket.get('initialLeverage') == leverage:
                    return float(bracket.get('notionalCap', 0))

            # 如果没有找到精确匹配，找到第一个杠杆大于等于目标杠杆的档位
            for bracket in brackets:
                if bracket.get('initialLeverage', 0) >= leverage:
                    return float(bracket.get('notionalCap', 0))

            logger.debug(f"{symbol} 杠杆{leverage}x无对应档位，使用最大值")
            return float(brackets[-1].get('notionalCap', 0)) if brackets else 0

        except Exception as e:
            logger.debug(f"{symbol} 获取杠杆{leverage}x最大持仓价值失败: {str(e)}")
            return 0

    def adjust_quantity_precision(self, symbol, quantity):
        """调整数量精度"""
        try:
            # 获取交易对信息
            symbol_info = self.get_symbol_info(symbol)
            # 调用helpers中的函数
            from utils.helpers import adjust_quantity_precision as adjust_qty
            return adjust_qty(symbol_info, quantity)
        except Exception as e:
            logger.warning(f"调整数量精度失败 {symbol}: {str(e)}")
            return quantity

    def adjust_price_precision(self, symbol, price):
        """调整价格精度"""
        try:
            # 获取交易对信息
            symbol_info = self.get_symbol_info(symbol)
            # 调用helpers中的函数
            from utils.helpers import adjust_price_precision as adjust_price
            return adjust_price(symbol_info, price)
        except Exception as e:
            logger.warning(f"调整价格精度失败 {symbol}: {str(e)}")
            return price