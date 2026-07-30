"""
资金分配模块 (Fund Allocator Module)

公共模块，通用资金计算逻辑

职责：
- 计算可分配资金
- 计算订单数量和杠杆
- 根据币安限制调整参数
"""

import time
from typing import Dict, Optional, Tuple
from framework.core.config import get_main_config
from config.settings import Settings
from utils.logger import Logger
from utils.helpers import adjust_price_precision
from utils.input_validator import safe_float, safe_int
from binance.exceptions import BinanceAPIException

# 延迟初始化Settings，避免模块导入时污染环境变量
_settings_instance = None

def _get_settings():
    """获取Settings实例（延迟初始化）"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

# 延迟初始化main_config
_main_config = None

def _get_main_config():
    """获取main_config实例（延迟初始化）"""
    global _main_config
    if _main_config is None:
        _main_config = get_main_config()
    return _main_config

logger = Logger.get_logger('fund_allocator')


class FundAllocator:
    """资金分配管理器（支持依赖注入）
    
    职责：
    - 计算可分配资金
    - 计算订单数量和杠杆
    - 根据币安限制调整参数
    
    使用方式（依赖注入，推荐）：
        allocator = FundAllocator(client=binance_client)
        available = allocator.get_available_funds()
    
    使用方式（全局单例，兼容）：
        from framework.business.fund_allocator import get_fund_allocator
        allocator = get_fund_allocator(client=binance_client)
        available = allocator.get_available_funds()
    """

    def __init__(self, client=None):
        self.client = client
        self._balance_cache = None
        self._balance_cache_time = 0
        logger.info("资金分配管理器已初始化")

    def set_client(self, client):
        """设置币安客户端"""
        self.client = client

    def get_available_funds(self, client=None) -> float:
        """获取可用资金"""
        actual_client = client or self.client
        if not actual_client:
            logger.error("没有可用的币安客户端")
            return 0
            
        try:
            current_time = time.time()
            if self._balance_cache and (current_time - self._balance_cache_time) < 5:
                return self._balance_cache

            account_info = actual_client.get_account_balance()
            # 安全获取账户信息各字段，防止 API 返回格式变化导致崩溃
            total_balance = safe_float(
                account_info.get('total_balance', account_info.get('balance', 0)),
                default=0.0, context='total_balance')
            used_margin = safe_float(
                account_info.get('total_margin', account_info.get('usedMargin', 0)),
                default=0.0, context='used_margin')
            available_funds = total_balance - used_margin

            self._balance_cache = available_funds
            self._balance_cache_time = current_time

            logger.debug(f"可用资金计算: 总余额={total_balance:.2f}, "
                       f"已用保证金={used_margin:.2f}, "
                       f"可用={available_funds:.2f}")

            return available_funds

        except Exception as e:
            logger.error(f"获取可用资金失败: {str(e)}")
            return 0

    def calculate_allocatable_funds(self, positions: Dict[str, Dict], client=None) -> float:
        """计算可分配资金"""
        available = self.get_available_funds(client)
        
        if not positions:
            return available
            
        allocated_for_pending = 0
        for symbol, pos in positions.items():
            pending_orders = pos.get('pending_orders', [])
            for order in pending_orders:
                allocated_for_pending += order.get('order_value', 0)
        
        allocatable = available - allocated_for_pending
        
        logger.debug(f"可分配资金: 可用={available:.2f}, "
                    f"已分配={allocated_for_pending:.2f}, "
                    f"可分配={allocatable:.2f}")
        
        return max(0, allocatable)

    def calculate_position_allocation(self, symbol: str, positions: Dict[str, Dict],
                                    total_allocatable: float, client=None) -> float:
        """
        计算单个币种的分配金额
        
        逻辑：
        1. 有持仓 -> 检查是否加仓（有配置按比例，没有按次数平均，未开启加仓=不分配）
        2. 无持仓 -> 平均分配
        """
        try:
            existing_position = positions.get(symbol)
            if existing_position:
                # 有持仓，检查是否需要加仓
                position_amt = safe_float(
                    existing_position.get('position_amt', 0),
                    default=0.0, context=f'{symbol} position_amt')
                if abs(position_amt) > 0.00001:
                    pnl_rate = safe_float(
                        existing_position.get('pnl_rate', 0),
                        default=0.0, context=f'{symbol} pnl_rate')
                    add_amount = 0

                    # 检查加仓次数
                    add_count = safe_int(
                        existing_position.get('add_count', 0),
                        default=0, context=f'{symbol} add_count')
                    
                    # 盈利加仓
                    if pnl_rate > 0:
                        # 检查是否配置了加仓比例
                        step = getattr(_get_settings(), 'PROFIT_STEP1', 0)
                        add = getattr(_get_settings(), 'PROFIT_ADD1', 0)
                        if step > 0 and add > 0:
                            # 有配置，按比例分配
                            if pnl_rate >= step:
                                add_amount = add
                        elif add > 0:
                            # 无配置比例但有配置金额，按次数平均分配
                            total_steps = 3  # 默认3次加仓机会
                            remaining_steps = max(1, total_steps - add_count)  # 防止除零
                            add_amount = add / remaining_steps
                        # 否则视为不分配（add_amount=0）

                    # 亏损加仓
                    if pnl_rate < 0:
                        step = getattr(_get_settings(), 'LOSS_STEP1', 0)
                        add = getattr(_get_settings(), 'LOSS_ADD1', 0)
                        if step < 0 and add > 0:
                            if pnl_rate <= step:
                                add_amount = add
                        elif add > 0:
                            total_steps = 3
                            remaining_steps = max(1, total_steps - add_count)  # 防止除零
                            add_amount = add / remaining_steps
                        # 否则视为不分配
                    
                    return add_amount
            
            # 新开仓：平均分配
            max_positions = _get_main_config().max_positions
            
            # 修复：防止max_positions为0或None导致除零错误
            if max_positions is None or max_positions <= 0:
                logger.warning(f"max_positions配置无效({max_positions})，使用默认值1")
                max_positions = 1
            
            if _get_settings().POSITION_ALLOCATION_MODE == 'EQUAL':
                allocation = total_allocatable / max_positions
            else:
                allocation = total_allocatable / max_positions
            
            allocation = min(allocation, _get_settings().SINGLE_SYMBOL_MAX_INVESTMENT)

            return allocation
            
        except Exception as e:
            logger.error(f"计算{symbol}分配金额失败: {str(e)}")
            return 0

    def get_max_order_quantity(self, symbol: str, price: float, client=None) -> float:
        """获取币种的最大订单数量限制"""
        actual_client = client or self.client
        if not actual_client:
            return float('inf')
            
        try:
            symbol_info = actual_client.get_symbol_info(symbol)
            if not symbol_info or 'filters' not in symbol_info:
                return float('inf')

            max_qty = float('inf')
            for f in symbol_info['filters']:
                if f.get('filterType') == 'LOT_SIZE':
                    max_qty = float(f.get('maxQty', float('inf')))

            return max_qty

        except Exception as e:
            logger.warning(f"{symbol} 获取最大订单数量失败: {str(e)}")
            return float('inf')

    def calculate_safe_order_params(self, symbol: str, target_margin: float, 
                                 target_leverage: int, current_price: float,
                                 client=None) -> Tuple[float, int]:
        """计算安全的下单参数"""
        actual_client = client or self.client
        if not actual_client:
            return target_margin, target_leverage
            
        try:
            leverage_brackets = actual_client.get_leverage_brackets(symbol)
            
            if leverage_brackets:
                for bracket in leverage_brackets:
                    min_leverage = int(bracket.get('minLeverage', 1))
                    max_leverage = int(bracket.get('maxLeverage', 125))
                    
                    if min_leverage <= target_leverage <= max_leverage:
                        break
                else:
                    target_leverage = min(max(target_leverage, min_leverage), max_leverage)
                    logger.warning(f"{symbol} 杠杆{target_leverage}x不在允许范围，已调整")
            
            min_notional = 5.0
            min_margin = min_notional / target_leverage
            
            if target_margin < min_margin:
                logger.warning(f"{symbol} 目标保证金{target_margin}低于最小值{min_margin}，已调整")
                target_margin = min_margin
            
            max_margin = getattr(_settings, 'MAX_MARGIN_PER_SYMBOL', 10000.0)
            if target_margin > max_margin:
                logger.warning(f"{symbol} 目标保证金{target_margin}超过最大值{max_margin}，已调整")
                target_margin = max_margin
            
            return target_margin, target_leverage
            
        except Exception as e:
            logger.warning(f"{symbol} 计算安全参数失败: {str(e)}")
            return target_margin, target_leverage

    def calculate_position_size(self, symbol: str, allocation: float, 
                                leverage: int, current_price: float,
                                client=None) -> float:
        """计算开仓数量"""
        try:
            raw_quantity = allocation * leverage / current_price
            quantity = (client or self.client).adjust_quantity_precision(symbol, raw_quantity)
            
            if quantity <= 0:
                logger.error(f"{symbol} 计算出的数量无效: {quantity}")
                return 0
            
            logger.debug(f"{symbol} 开仓数量计算: 分配={allocation}, "
                        f"杠杆={leverage}, 价格={current_price}, "
                        f"数量={quantity}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"{symbol} 计算开仓数量失败: {str(e)}")
            return 0

    def calculate_leverage(self, symbol: str, client=None) -> int:
        """计算杠杆倍数"""
        leverage = _get_main_config().leverage
        
        if getattr(_settings, 'VOLATILITY_LEVERAGE_ENABLED', False):
            try:
                actual_client = client or self.client
                volatility = actual_client.calculate_volatility(symbol) if actual_client else 0
                avg_threshold = getattr(_settings, 'VOLATILITY_AVG_THRESHOLD', 4.0)
                reduction = getattr(_settings, 'VOLATILITY_LEVERAGE_REDUCTION', 0.5)
                
                if volatility > avg_threshold:
                    original_leverage = leverage
                    leverage = int(leverage * reduction)
                    logger.info(f"{symbol} 高波动率({volatility:.2f}%)，杠杆从{original_leverage}x降至{leverage}x")
                    
            except Exception as e:
                logger.warning(f"{symbol} 波动率计算失败，使用默认杠杆: {str(e)}")
        
        return leverage


# ========== 依赖注入模式 ==========
# 推荐使用方式：通过构造函数注入
_fund_allocator_instance = None  # 保留兼容旧代码

def get_fund_allocator(client=None) -> FundAllocator:
    """获取资金分配管理器（兼容旧API，推荐使用 DI 方式）
    
    新代码推荐：
        allocator = FundAllocator(client=client)
    
    这样可以在测试时 mock client，实现依赖注入。
    """
    global _fund_allocator_instance
    if _fund_allocator_instance is None:
        _fund_allocator_instance = FundAllocator(client)
    elif client and _fund_allocator_instance.client is None:
        _fund_allocator_instance.set_client(client)
    return _fund_allocator_instance