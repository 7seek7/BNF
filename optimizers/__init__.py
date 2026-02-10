# 在模块导入前设置 sys.path，解决 Streamlit Cloud 多线程导入问题
import sys
from pathlib import Path

# 获取 platform_deployment 目录路径（optimizers 的父目录）
current_file = Path(__file__)  # .../platform_deployment/optimizers/__init__.py
project_root = current_file.parent.parent  # .../platform_deployment

# 设置 Python 路径
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'optimizers'))
sys.path.insert(0, str(project_root / 'utils'))
sys.path.insert(0, str(project_root / 'config'))
sys.path.insert(0, str(project_root / 'alert'))
sys.path.insert(0, str(project_root / 'backtest'))
sys.path.insert(0, str(project_root / 'trading'))

# Original exports
"""
Optimization Module

包含多阶段优化算法和评估工具。
"""
from .global_optimizer import GlobalOptimizer
from .parallel_evaluator import ParallelEvaluator, EvaluationResult
from .state_manager import StateManager, OptimizationState
from .tpe_sampler import TPE_Optimizer
from .cma_es_optimizer import MultiStartCMAES
from .differential_evolution import MultiStartDE

__all__ = [
    'GlobalOptimizer',
    'ParallelEvaluator',
    'EvaluationResult',
    'StateManager',
    'OptimizationState',
    'TPE_Optimizer',
    'MultiStartCMAES',
    'MultiStartDE',
]
