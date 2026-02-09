import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from alert.binance_client import BinanceClient
from utils.logger import Logger

logger = Logger.get_logger('data_downloader')

POPULAR_SYMBOLS = [
    'BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','SOLUSDT','ADAUSDT','DOGEUSDT','MATICUSDT','DOTUSDT','AVAXUSDT',
    'LTCUSDT','LINKUSDT','UNIUSDT','ATOMUSDT','ICPUSDT','ETCUSDT','FLOWUSDT','NEARUSDT','FTMUSDT','ETCUSDT',
    'COMPUSDT','AAVEUSDT','XLMUSDT','CHZUSDT','VETUSDT','XTZUSDT','OKBUSDT','FILUSDT','EOSUSDT','THETUSDT'
]

class DataDownloader:
    """K线数据下载器 - 支持离线模式"""

    def __init__(self, mode='live', use_testnet=False, offline=False):
        """
        初始化下载器

        :param mode: 模式 ('live', 'testnet')
        :param use_testnet: 是否使用测试网（在中国大陆建议开启）
        :param offline: 是否使用离线模式（只使用缓存，不连接网络）
        """
        import os

        self.offline = offline
        self.client = None  # 默认不创建客户端

        # 离线模式：只使用缓存，不连接API
        if offline:
            print("[离线模式] 只使用本地缓存数据，不连接网络")
            print("      - 适合已有缓存数据的情况")
            print("      - 不需要网络连接")
            print("      - 加速优化过程")
        else:
            # 检查环境变量，是否强制使用测试网
            if use_testnet or os.getenv('USE_TESTNET') == '1':
                self.mode = 'testnet'
                print("[提示] 使用测试网数据源 (testnet.binance.vision)")
                print("   - 适合中国大陆用户")
                print("   - 数据与主网基本相同")
                print("   - 不需要API密钥")
            else:
                self.mode = mode

            try:
                self.client = BinanceClient(self.mode)
            except Exception as e:
                if "Connection" in str(type(e).__name__) or "Timeout" in str(type(e).__name__):
                    print("\n" + "="*70)
                    print("❌ 无法连接到币安服务器")
                    print("="*70)
                    print("\n可能的原因和解决方案：")
                    print("\n1️⃣ 你在中国大陆，币安主网API无法直接访问")
                    print("   解决方案A: 设置环境变量使用测试网")
                    print("   ```")
                    print("   # Windows CMD")
                    print("   set USE_TESTNET=1")
                    print("   python optimizer/optimizer.py --quick --symbols BTCUSDT --days 30")
                    print("")
                    print("   # Windows PowerShell")
                    print("   $env:USE_TESTNET=1")
                    print("   python optimizer/optimizer.py --quick --symbols BTCUSDT --days 30")
                    print("   ```")
                    print("")
                    print("   解决方案B: 使用离线模式（推荐，如果有缓存数据）")
                    print("   ```")
                    print("   python optimizer/optimizer.py --offline --quick --symbols BTCUSDT --days 30")
                    print("   ```")
                    print("\n2️⃣ 需要开启VPN或配置代理")
                    print("   - 开启VPN后重试")
                    print("   - 或配置系统代理")
                    print("\n3️⃣ 防火墙阻止了网络连接")
                    print("   - 检查防火墙设置")
                    print("   - 允许Python网络访问")
                    print("\n" + "="*70)
                    print()
                    raise ConnectionError("无法连接币安API，请使用测试网/离线模式或VPN")
                else:
                    raise

        # 使用绝对路径，兼容不同部署环境
        import os
        # 尝试多个可能的数据目录位置
        possible_paths = [
            Path('data/historical'),  # 当前目录
            Path(__file__).parent.parent / 'data' / 'historical',  # 相对于本文件
            Path('/app/data/historical'),  # Hugging Face Spaces绝对路径
        ]
        
        self.data_dir = None
        # 首先查找已存在且有数据的路径（用于离线模式）
        for path in possible_paths:
            if path.exists() and any(path.glob('*.csv')):
                self.data_dir = path
                print(f"[数据目录] 使用: {path}")
                break
        
        # 如果没有找到已有数据的路径，使用第一个可写的路径
        if self.data_dir is None:
            for path in possible_paths:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self.data_dir = path
                    print(f"[数据目录] 创建并使用: {path}")
                    break
                except:
                    continue
        
        if self.data_dir is None:
            # 默认使用相对于本文件的路径
            self.data_dir = Path(__file__).parent.parent / 'data' / 'historical'
            self.data_dir.mkdir(parents=True, exist_ok=True)
            print(f"[数据目录] 默认使用: {self.data_dir}")
        
        # 验证数据目录
        if self.offline:
            csv_files = list(self.data_dir.glob('*.csv'))
            print(f"[离线模式] 数据目录: {self.data_dir}")
            print(f"[离线模式] 找到 {len(csv_files)} 个CSV文件")
            if len(csv_files) == 0:
                print(f"[警告] 数据目录为空，离线模式将无法运行")
    
    def download_symbol_data(self, symbol, interval, days, force_download=False):
        """
        下载单个币种的历史数据
        :param symbol: 币种
        :param interval: K线周期 '1m', '5m', '15m', '1h', '4h' 等
        :param days: 回测天数
        :param force_download: 是否强制重新下载
        :return: DataFrame
        """
        try:
            # 检查是否已有缓存（精确匹配）
            cache_file = self.get_cache_filename(symbol, interval, days)

            if cache_file.exists():
                if not force_download:
                    logger.info(f"{symbol} 数据已缓存，从文件加载")
                    return self.load_from_cache(cache_file)
                else:
                    logger.info(f"强制重新下载 {symbol} 数据")

            # 离线模式：尝试查找任何匹配的缓存文件（忽略日期）
            if self.offline:
                # 尝试查找任何可用的缓存
                cached_file, cached_days, cached_date_str = self.find_any_cached_file(symbol, interval)
                if cached_file and cached_file.exists():
                    print(f"\n[降级] 找到历史缓存（缓存于 {cached_date_str}）")
                    print(f"[降级] 文件: {cached_file.name}")
                    print(f"[降级] 数据天数: {cached_days} 天 (请求: {days} 天)")
                    print(f"[降级] 使用历史缓存继续优化...")
                    return self.load_from_cache(cached_file)

                # 真正找不到任何缓存
                print(f"\n[错误] 离线模式下找不到任何 {symbol} {interval} 缓存数据")
                print(f"\n可用文件列表:")
                if self.data_dir.exists():
                    for file in sorted(self.data_dir.glob("*.csv"))[:10]:
                        print(f"  - {file.name}")
                print(f"\n解决方法:")
                print(f"  1. 先运行在线模式下载数据:")
                print(f"     python optimizer/optimizer.py --testnet --quick --symbols {symbol} --days {days}")
                print(f"  2. 或使用 --force 参数强制重新下载")
                print(f"  3. 或退出离线模式进行数据下载")
                raise FileNotFoundError(f"离线模式下缺少缓存: {cache_file}")

            logger.info(f"开始下载 {symbol} {interval} 数据，共 {days} 天")

            interval_minutes = self.get_interval_minutes(interval)
            total_klines_needed = int((days * 24 * 60) / interval_minutes)

            all_klines = self.download_in_batches(symbol, interval, total_klines_needed)

            if not all_klines:
                logger.error(f"{symbol} 数据下载失败")
                return None

            df = self.klines_to_dataframe(all_klines)

            self.save_to_cache(df, cache_file)

            logger.info(f"{symbol} 数据下载完成: {len(df)} 根K线")

            return df

        except Exception as e:
            logger.error(f"下载 {symbol} 数据失败: {str(e)}")
            return None
    
    def download_history_data_by_date_range(self, symbol, interval, start_date, end_date):
        """
        按日期范围下载历史数据
        
        Args:
            symbol: 币种
            interval: K线周期
            start_date: 开始日期 (YYYY-MM-DD 格式)
            end_date: 结束日期 (YYYY-MM-DD 格式)
        
        Returns:
            DataFrame
        """
        try:
            from datetime import timedelta
            
            # 转换为时间戳
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # 包含结束当天
            
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)
            
            # 计算天数
            days = (end_dt - start_dt).days
            
            logger.info(f"下载 {symbol} 数据: {start_date} ~ {end_date} ({days}天)")
            
            # 计算需要的K线数量
            interval_minutes = self.get_interval_minutes(interval)
            total_klines_needed = int((days * 24 * 60) / interval_minutes)
            
            # 下载
            all_klines = self.download_in_batches_by_time_range(symbol, interval, start_ts, end_ts, total_klines_needed)
            
            if not all_klines:
                logger.error(f"{symbol} 数据下载失败")
                return None
            
            df = self.klines_to_dataframe(all_klines)
            
            # 过滤时间范围
            df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)].copy()
            
            if df.empty:
                logger.error(f"{symbol} 数据为空（时间范围：{start_date} ~ {end_date}）")
                return None
            
            logger.info(f"{symbol} 数据下载完成: {len(df)} 根K线")
            
            return df
            
        except Exception as e:
            logger.error(f"按日期范围下载 {symbol} 数据失败: {str(e)}")
            return None
    
    def download_in_batches_by_time_range(self, symbol, interval, start_ts, end_ts, total_needed):
        """
        按时间范围分批下载K线数据
        
        Args:
            symbol: 币种
            interval: K线周期
            start_ts: 开始时间戳（毫秒）
            end_ts: 结束时间戳（毫秒）
            total_needed: 需要的总K线数
        
        Returns:
            K线列表
        """
        all_klines = []
        batch_size = 1500  # 币安API单次最大限制
        num_batches = (total_needed + batch_size - 1) // batch_size
        
        logger.info(f"需要 {total_needed} 根K线，分 {num_batches} 批下载")
        
        current_end_ts = end_ts
        
        for batch in range(num_batches):
            try:
                remaining = total_needed - len(all_klines)
                limit = min(batch_size, remaining)
                
                # 计算这批的开始时间
                interval_minutes = self.get_interval_minutes(interval)
                current_start_ts = current_end_ts - (limit - 1) * interval_minutes * 60 * 1000
                
                # 确保不超过开始时间
                if current_start_ts < start_ts:
                    current_start_ts = start_ts
                
                # 下载数据
                klines = self.client.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=current_start_ts,
                    endTime=current_end_ts
                )
                
                if not klines:
                    logger.warning(f"{symbol} 第 {batch+1} 批数据为空，可能已到最早可用数据")
                    break
                
                all_klines = klines + all_klines
                current_end_ts = int(klines[0][0])
                
                logger.info(f"已下载 {len(all_klines)}/{total_needed} 根K线 (第 {batch+1}/{num_batches} 批)")
                
                if len(all_klines) >= total_needed:
                    break
                    
                # API限流等待
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"第 {batch+1} 批下载失败: {str(e)}")
                if len(all_klines) == 0:
                    raise
                break
        
        return all_klines
    
    def download_in_batches(self, symbol, interval, total_needed):
        """
        分批下载K线数据
        :param symbol: 币种
        :param interval: 周期
        :param total_needed: 需要的总K线数
        :return: K线列表
        """
        all_klines = []
        batch_size = 1500  # 币安API单次最大限制
        num_batches = (total_needed + batch_size - 1) // batch_size
        
        logger.info(f"需要 {total_needed} 根K线，分 {num_batches} 批下载")
        
        end_time = None  # 最新的时间
        
        for batch in range(num_batches):
            try:
                remaining = total_needed - len(all_klines)
                limit = min(batch_size, remaining)
                
                if end_time is None:
                    klines = self.client.get_klines(symbol, interval, limit)
                else:
                    klines = self.client.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        endTime=end_time - 1
                    )
                if not klines:
                    logger.warning(f"{symbol} 第 {batch+1} 批数据为空，可能已到最早可用数据")
                    break
                end_time = int(klines[0][0])
                all_klines = klines + all_klines
                logger.info(f"已下载 {len(all_klines)}/{total_needed} 根K线 (第 {batch+1}/{num_batches} 批)")
                if len(all_klines) >= total_needed:
                    break
                if len(klines) < limit:
                    logger.warning(f"{symbol} 已到最早可用数据，实际获取 {len(all_klines)} 根")
                    break
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"第 {batch+1} 批下载失败: {str(e)}")
                if len(all_klines) == 0:
                    raise
                break
        return all_klines

    def aggregate_klines(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        聚合低时间间隔K线到高时间间隔

        Args:
            df: 原始K线数据（低时间间隔）
            target_interval: 目标时间间隔（如 '5m', '15m', '1h'）

        Returns:
            聚合后的DataFrame
        """
        if len(df) == 0:
            return df

        # 判断原始间隔
        original_interval = self.detect_interval(df)
        target_minutes = self.get_interval_minutes(target_interval)
        original_minutes = self.get_interval_minutes(original_interval)

        # 计算需要聚合的K线数量
        agg_ratio = target_minutes // original_minutes

        if agg_ratio <= 1:
            return df  # 不需要聚合

        print(f"[聚合] 将 {original_interval} 聚合为 {target_interval} (1:{agg_ratio})")

        # 按时间戳分组聚合
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 创建聚合组编号
        df['group'] = df.index // agg_ratio

        # 聚合函数
        def agg_fn(group):
            # Open: 第一根K线的开盘价
            open_price = group.iloc[0]['open']
            # High: 所有K线的最高价
            high_price = group['high'].max()
            # Low: 所有K线的最低价
            low_price = group['low'].min()
            # Close: 最后一根K线的收盘价
            close_price = group.iloc[-1]['close']
            # Volume: 所有K线的成交量之和
            volume = group['volume'].sum()
            # Quote_volume: 所有K线的成交额之和
            quote_volume = group['quote_volume'].sum()
            # Timestamp: 最后一根K线的时间戳（或第一根，根据需求）
            timestamp = group.iloc[-1]['timestamp']

            return pd.Series({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'quote_volume': quote_volume,
                'datetime': pd.to_datetime(timestamp, unit='ms')
            })

        # 分组聚合
        agg_df = df.groupby('group').apply(agg_fn).reset_index(drop=True)

        print(f"[聚合] 完成: {len(df)} 根 {original_interval} → {len(agg_df)} 根 {target_interval}")

        return agg_df

    def detect_interval(self, df: pd.DataFrame) -> str:
        """
        检测DataFrame中的K线间隔

        Args:
            df: K线数据

        Returns:
            间隔字符串（如 '1m', '5m', '1h'）
        """
        if len(df) < 2:
            return '1m'  # 默认

        # 计算相邻K线的时间差
        time_diffs = df['timestamp'].diff().dropna()

        # 使用最常见的间隔
        for diff in time_diffs.value_counts().index:
            minutes = int(diff / (1000 * 60))
            if minutes == 1:
                return '1m'
            elif minutes == 3:
                return '3m'
            elif minutes == 5:
                return '5m'
            elif minutes == 15:
                return '15m'
            elif minutes == 30:
                return '30m'
            elif minutes == 60:
                return '1h'
            elif minutes == 240:
                return '4h'
            elif minutes == 1440:
                return '1d'

        return '1m'  # 默认

    def get_interval_minutes(self, interval):
        """获取周期对应的分钟数"""
        mapping = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
            '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080
        }
        return mapping.get(interval, 5)
    
    def klines_to_dataframe(self, klines):
        """将K线数据转换为DataFrame"""
        data = []
        for k in klines:
            data.append({
                'timestamp': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'close_time': int(k[6]),
                'quote_volume': float(k[7]),
                'trades': int(k[8]),
                'taker_buy_base': float(k[9]),
                'taker_buy_quote': float(k[10])
            })
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def get_cache_filename(self, symbol, interval, days):
        """生成缓存文件名"""
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{symbol}_{interval}_{days}days_{date_str}.csv"
        return self.data_dir / filename

    def find_any_cached_file(self, symbol, interval):
        """
        查找任何匹配的缓存文件（忽略日期）

        Args:
            symbol: 币种
            interval: K线周期

        Returns:
            (Path, days, cached_days_ago) - 缓存文件路径, 数据天数, 缓存天数前的日期
            如果没有找到返回 (None, 0, 0)
        """
        if not self.data_dir.exists():
            return None, 0, 0

        # 查找所有匹配的文件
        pattern = f"{symbol}_{interval}_"
        matching_files = []

        try:
            for file in self.data_dir.glob(f"{pattern}*.csv"):
                filename = file.stem  # SOLUSDT_1m_180days_20260114
                parts = filename.split('_')
                if len(parts) >= 4:
                    try:
                        days_str = parts[2].replace('days', '')
                        days = int(days_str)
                        date_str = parts[3]
                        matching_files.append((file, days, date_str))
                    except (ValueError, IndexError):
                        continue
        except Exception:
            pass

        if not matching_files:
            return None, 0, 0

        # 按天数排序，优先使用天数最多的（数据最长的）
        matching_files.sort(key=lambda x: x[1], reverse=True)
        return matching_files[0]

    def load_from_cache(self, cache_file):
        """从缓存加载数据"""
        try:
            df = pd.read_csv(cache_file)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.info(f"从缓存加载: {len(df)} 根K线")
            return df
        except Exception as e:
            logger.error(f"加载缓存失败: {str(e)}")
            return None
    
    def download_multiple_symbols(self, symbols, interval='5m', days=90, force_download=False, fallback_intervals=None):
        """
        下载多个币种的数据（带多周期降级支持）

        Args:
            symbols: 币种列表
            interval: K线周期（首选，默认5m）
            days: 回测天数
            force_download: 是否强制重新下载
            fallback_intervals: 降级周期列表（默认: ['5m', '3m', '1m', '1h']）

        Returns:
            {symbol: (DataFrame, actual_interval_used)}
        """
        if fallback_intervals is None:
            fallback_intervals = ['5m', '3m', '1m', '1h']

        result = {}
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"下载进度: {i+1}/{len(symbols)} - {symbol}")

                # 使用多周期降级尝试
                df, actual_interval = self.try_download_multiple_intervals(
                    symbol=symbol,
                    days=days,
                    preferred_intervals=[interval] + [intv for intv in fallback_intervals if intv != interval],
                    force_download=force_download
                )

                if df is not None:
                    result[symbol] = (df, actual_interval)
                    if actual_interval != interval:
                        logger.info(f"[降级] {symbol} 使用 {actual_interval} 代替 {interval}")
                else:
                    logger.error(f"[失败] {symbol} 所有周期都不可用")

                if not force_download and i < len(symbols) - 1:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"下载 {symbol} 失败: {str(e)}")
                continue

        successful = len(result)
        logger.info(f"数据下载完成: {successful}/{len(symbols)} 个币种")
        return result

    def try_download_multiple_intervals(self, symbol, days, preferred_intervals=None, force_download=False):
        """
        尝试使用多个时间间隔下载数据（带自动降级和聚合）

        优先级顺序：5m → 3m → 1m → 1h
        如果首选间隔找不到缓存，会尝试：
        1. 直接查找次选间隔
        2. 如果找到低间隔（如1m）但需要高间隔（如5m），则聚合成目标间隔

        Args:
            symbol: 币种
            days: 天数
            preferred_intervals: 偏好的间隔列表（默认：['5m', '3m', '1m', '1h']）
            force_download: 是否强制重新下载

        Returns:
            (DataFrame, str) - (数据, 实际使用的间隔)
        """
        if preferred_intervals is None:
            preferred_intervals = ['5m', '3m', '1m', '1h']

        last_error = None
        df_loaded = None
        loaded_interval = None

        # 定义聚合映射：哪些间隔可以 聚合成 哪些间隔
        # key: 目标间隔, value: 提供数据的可能低间隔
        agg_mapping = {
            '5m': ['1m'],
            '15m': ['1m', '3m', '5m'],
            '30m': ['1m', '3m', '5m', '15m'],
            '1h': ['1m', '3m', '5m', '15m', '30m'],
        }

        for i, target_interval in enumerate(preferred_intervals):
            try:
                # 尝试目标间隔
                cache_file = self.get_cache_filename(symbol, target_interval, days)

                # 缓存存在且不强制重新下载 - 直接返回
                if cache_file.exists() and not force_download:
                    logger.info(f"[降级] 找到精确缓存: {symbol}_{target_interval}_{days}days")
                    df = self.load_from_cache(cache_file)
                    if df is not None:
                        return df, target_interval

                # 离线模式：尝试查找或聚合低间隔数据
                if self.offline:
                    logger.info(f"[降级] 离线模式下无 {target_interval} 缓存")
                    
                    # 尝试查找可聚合的低间隔数据
                    if target_interval in agg_mapping:
                        for lower_interval in agg_mapping[target_interval]:
                            lower_cache_file, lower_days, _ = self.find_any_cached_file(symbol, lower_interval)
                            if lower_cache_file and lower_cache_file.exists():
                                print(f"  [聚合] 找到可聚合的 {symbol}_{lower_interval} 数据")
                                df_lower = self.load_from_cache(lower_cache_file)
                                if df_lower is not None and len(df_lower) > 0:
                                    df = self.aggregate_klines(df_lower, target_interval)
                                    print(f"  [聚合] 成功生成 {target_interval} 数据 ({len(df)} 根K线)")
                                    return df, target_interval

                    logger.info(f"[降级] 离线模式下无 {target_interval} 及可聚合的低间隔缓存，继续尝试下一个间隔...")
                    continue

                # 在线模式：尝试下载
                df = self.download_symbol_data(symbol, target_interval, days, force_download)
                if df is not None:
                    logger.info(f"[降级] 成功下载: {symbol}_{target_interval}_{days}days")
                    return df, target_interval

            except Exception as e:
                last_error = e
                logger.warning(f"[降级] {symbol} {target_interval} 尝试失败: {str(e)[:100]}")
                continue

        # 所有间隔都失败
        logger.error(f"[降级] {symbol} 所有间隔 ({', '.join(preferred_intervals)}) 均失败")
        if last_error:
            raise last_error
        return None, None
    
    def clear_cache(self, older_than_days=7):
        """
        清理旧缓存
        :param older_than_days: 清理多少天前的缓存
        """
        try:
            cutoff_time = time.time() - (older_than_days * 24 * 3600)
            deleted_count = 0
            for cache_file in self.data_dir.glob('*.csv'):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    deleted_count += 1
            if deleted_count > 0:
                logger.info(f"已清理 {deleted_count} 个旧缓存文件")
        except Exception as e:
            logger.error(f"清理缓存失败: {str(e)}")

    def get_top_symbols(self, n=10):
        """根据历史缓存或默认列表，返回前 n 名成交量最高的币种列表。
        优先从本地缓存数据统计，如无缓存则回落使用默认常用币对。
        """
        try:
            symbol_vols = {}
            if self.data_dir.exists():
                for file in self.data_dir.glob('*.csv'):
                    stem = file.stem  # e.g. BTCUSDT_5m_180days_20260115
                    parts = stem.split('_')
                    if not parts:
                        continue
                    symbol = parts[0]
                    try:
                        df = pd.read_csv(file)
                        if 'quote_volume' in df.columns:
                            vol = float(df['quote_volume'].astype(float).sum())
                            symbol_vols[symbol] = vol
                    except Exception:
                        continue
            # 排序并取前 n 名
            sorted_symbols = sorted(symbol_vols.items(), key=lambda x: x[1], reverse=True)
            top = [s for s, _ in sorted_symbols[:n]]
            if len(top) < n:
                for s in POPULAR_SYMBOLS:
                    if s not in top:
                        top.append(s)
                        if len(top) >= n:
                            break
            return top[:n]
        except Exception:
            return POPULAR_SYMBOLS[:n]

POPULAR_SYMBOLS = [
    'BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','SOLUSDT','ADAUSDT','DOGEUSDT','MATICUSDT','DOTUSDT','AVAXUSDT',
    'LTCUSDT','LINKUSDT','UNIUSDT','ATOMUSDT','ICPUSDT','ETCUSDT','FLOWUSDT','NEARUSDT','FTMUSDT','ETCUSDT',
    'COMPUSDT','AAVEUSDT','XLMUSDT','CHZUSDT','VETUSDT','XTZUSDT','OKBUSDT','FILUSDT','EOSUSDT','REQUSDT',
    'KCSUSDT','ZENUSDT','SUSHIUSDT','SNXUSDT','YFIIUSDT','CRVUSDT','RUNEUSDT','KLAYUSDT','BCHUSDT','ZILUSDT'
]


def main():
    downloader = DataDownloader()
    symbol = 'BTCUSDT'
    interval = '5m'
    days = 180
    df = downloader.download_symbol_data(symbol, interval, days)
    if df is not None:
        print(f"Downloaded {len(df)} rows for {symbol}")
    else:
        print("Download failed")

if __name__ == '__main__':
    main()
