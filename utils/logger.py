# -*- coding: utf-8 -*-
"""
统一日志模块 - 重定向到框架日志系统

已废弃，请使用 framework.core.logger
"""
# 重定向到统一日志系统
from framework.core.logger import get_logger as _get_logger
from framework.core.logger import LoggerManager, ContextualLogger

# 兼容旧接口
class Logger:
    """统一日志器（已废弃，请使用 get_logger）"""
    
    @staticmethod
    def get_logger(name: str = "bnftrading"):
        return _get_logger(name)

def get_logger(name: str = "bnftrading"):
    return _get_logger(name)
