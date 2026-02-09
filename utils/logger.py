import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from config.settings import settings

class Logger:
    """日志管理类"""

    _loggers = {}

    @classmethod
    def get_logger(cls, name):
        """获取日志器"""
        if name in cls._loggers:
            return cls._loggers[name]

        # 创建日志目录
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # 创建日志器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))

        # 避免重复添加处理器
        if logger.handlers:
            return logger

        # 文件处理器 - 使用WARNING级别减少日志量
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')

        # 文件只记录WARNING及以上级别，减少日志文件大小
        # 特殊模块可以记录INFO级别用于交易记录和优化过程
        if name in ['trader', 'unified_backtest', 'strategy', 'backtest_executor',
                    'global_optimizer', 'parallel_evaluator', 'state_manager',
                    'tpe_sampler', 'cma_es_optimizer', 'differential_evolution']:
            file_handler.setLevel(logging.INFO)
        else:
            file_handler.setLevel(logging.WARNING)

        # 控制台处理器（修复 Windows 控制台编码问题）
        if sys.platform == 'win32':
            # Windows 下设置环境变量强制使用 UTF-8 编码
            os.environ['PYTHONIOENCODING'] = 'utf-8'

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        cls._loggers[name] = logger
        return logger