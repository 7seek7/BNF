# -*- coding: utf-8 -*-
"""
交易模块 - 保留向后兼容

⚠️ 注意: 大部分类已移至 framework/business/ 或 strategies/*/private/
此模块保留是为了向后兼容，新代码应使用新模块。

废弃模块 (不再使用):
- fund_allocator.py
- order_persistence.py  
- duplicate_detector.py
- risk_manager.py
- circuit_breaker.py
- trade_recorder.py
- position_builder.py (使用 strategies/v23/private/)
- take_profit_manager.py (使用 strategies/v23/private/)
- stop_loss_manager.py (使用 strategies/v23/private/)
- position_monitor.py (使用 strategies/v23/private/)
- position_manager.py (使用 framework/business/)
"""

from .binance_client import BinanceClient

__all__ = [
    'BinanceClient',
]