# -*- coding: utf-8 -*-
"""
框架异常定义

统一异常体系，便于错误处理和追踪
"""

import time
from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(Enum):
    """错误码"""
    # 配置错误 (1000-1099)
    CONFIG_LOAD_FAILED = 1000
    CONFIG_VALIDATION_FAILED = 1001
    CONFIG_FILE_NOT_FOUND = 1002
    CONFIG_INVALID_VALUE = 1003
    
    # API错误 (2000-2099)
    API_CONNECTION_FAILED = 2000
    API_TIMEOUT = 2001
    API_RATE_LIMIT = 2002
    API_AUTH_FAILED = 2003
    API_INVALID_RESPONSE = 2004
    
    # 订单错误 (3000-3099)
    ORDER_CREATE_FAILED = 3000
    ORDER_CANCEL_FAILED = 3001
    ORDER_FILL_FAILED = 3002
    ORDER_INSUFFICIENT_BALANCE = 3003
    ORDER_INVALID_PARAMS = 3004
    ORDER_REJECTED = 3005
    
    # 持仓错误 (4000-4099)
    POSITION_NOT_FOUND = 4000
    POSITION_ALREADY_EXISTS = 4001
    POSITION_INSUFFICIENT = 4002
    
    # 风控错误 (5000-5099)
    RISK_LIMIT_EXCEEDED = 5000
    RISK_EMERGENCY_STOP = 5001
    RISK_BLACK_SWAN = 5002
    
    # 策略错误 (6000-6099)
    STRATEGY_NOT_FOUND = 6000
    STRATEGY_LOAD_FAILED = 6001
    STRATEGY_EXECUTE_FAILED = 6002
    STRATEGY_INVALID_SIGNAL = 6003
    
    # 数据错误 (7000-7099)
    DATA_INVALID_FORMAT = 7000
    DATA_MISSING_FIELD = 7001
    DATA_OUT_OF_RANGE = 7002
    
    # 系统错误 (9000-9099)
    SYSTEM_UNEXPECTED_ERROR = 9000
    SYSTEM_THREAD_ERROR = 9001
    SYSTEM_RESOURCE_EXHAUSTED = 9002


class FrameworkException(Exception):
    """
    框架异常基类
    
    所有框架异常都继承此类，便于统一处理
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.timestamp = time.time()
        
        # 构建完整消息
        full_message = f"[{code.name}] {message}"
        if details:
            full_message += f" | Details: {details}"
        if cause:
            full_message += f" | Cause: {str(cause)}"
            
        super().__init__(full_message)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于日志或API响应）"""
        return {
            'code': self.code.value,
            'code_name': self.code.name,
            'message': self.message,
            'details': self.details,
            'cause': str(self.cause) if self.cause else None,
        }


class ConfigException(FrameworkException):
    """配置异常"""
    pass


class APIException(FrameworkException):
    """API异常"""
    pass


class OrderException(FrameworkException):
    """订单异常"""
    pass


class PositionException(FrameworkException):
    """持仓异常"""
    pass


class RiskException(FrameworkException):
    """风控异常"""
    pass


class StrategyException(FrameworkException):
    """策略异常"""
    pass


class DataException(FrameworkException):
    """数据异常"""
    pass


# 便捷创建异常的函数
def raise_config_error(message: str, code: ErrorCode = ErrorCode.CONFIG_LOAD_FAILED, details: Dict = None, cause: Exception = None):
    """抛出配置异常"""
    raise ConfigException(code, message, details, cause)


def raise_api_error(code: ErrorCode, message: str, details: Dict = None, cause: Exception = None):
    """抛出API异常"""
    raise APIException(code, message, details, cause)


def raise_order_error(code: ErrorCode, message: str, details: Dict = None, cause: Exception = None):
    """抛出订单异常"""
    raise OrderException(code, message, details, cause)


def raise_risk_error(code: ErrorCode, message: str, details: Dict = None, cause: Exception = None):
    """抛出风控异常"""
    raise RiskException(code, message, details, cause)


def raise_strategy_error(code: ErrorCode, message: str, details: Dict = None, cause: Exception = None):
    """抛出策略异常"""
    raise StrategyException(code, message, details, cause)


def raise_data_error(message: str, code: ErrorCode = ErrorCode.DATA_INVALID_FORMAT, details: Dict = None, cause: Exception = None):
    """抛出数据异常"""
    raise DataException(code, message, details, cause)
