#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行评估引擎 - 用于多参数回测的并行执行
支持ThreadPool进行并发回测
"""

import sys
import os
import json
import time
import logging
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 设置日志
sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import Logger
logger = Logger.get_logger('parallel_evaluator')


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    params: Dict[str, Any]  # 参数组合
    fitness: float  # 适应度（收益率）
    details: Dict[str, Any]  # 详细信息
    evaluation_id: int  # 评估ID
    evaluation_time: float  # 评估耗时（秒）
    success: bool  # 是否成功
    error: Optional[str] = None  # 错误信息


class ParallelEvaluator:
    """
    并行评估引擎
    
    功能：
    1. 并发执行多个参数配置的回测
    2. 管理工作线程池
    3. 跟踪评估进度
    4. 错误处理和重试
    5. 结果缓存（避免重复评估）
    """

    def __init__(self, max_workers: int = 10, 
                 evaluation_function: Optional[Callable] = None,
                 cache_enabled: bool = True):
        """
        初始化并行评估器
        
        Args:
            max_workers: 最大工作线程数
            evaluation_function: 评估函数（回测函数）
            cache_enabled: 是否启用结果缓存
        """
        self.max_workers = max_workers
        self.evaluation_function = evaluation_function
        self.cache_enabled = cache_enabled
        
        # 评估缓存（参数hash → 结果）
        self._cache: Dict[str, EvaluationResult] = {}
        
        # 统计信息
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.cache_hits = 0
        
        # 进度回调
        self.progress_callback: Optional[Callable] = None
        
        logger.info(f"[ParallelEvaluator] 初始化完成，max_workers={max_workers}")

    def set_evaluation_function(self, func: Callable):
        """设置评估函数"""
        self.evaluation_function = func
        logger.info("[ParallelEvaluator] 评估函数已设置")

    def _hash_params(self, params: Dict[str, Any]) -> str:
        """计算参数的哈希值（用于缓存）"""
        # 对参数进行排序后生成字符串
        sorted_params = json.dumps(params, sort_keys=True)
        import hashlib
        return hashlib.md5(sorted_params.encode()).hexdigest()

    def _evaluate_single(self, params: Dict[str, Any], eval_id: int,
                        backtest_days: int = 60) -> EvaluationResult:
        """
        执行单次评估（内部方法）
        
        Args:
            params: 参数组合
            eval_id: 评估ID
            backtest_days: 回测天数
            
        Returns:
            EvaluationResult对象
        """
        start_time = time.time()
        
        try:
            if self.evaluation_function is None:
                raise ValueError("评估函数未设置，请先调用set_evaluation_function()")
            
            # 调用评估函数（回测）
            result = self.evaluation_function(params, backtest_days)
            
            evaluation_time = time.time() - start_time
            
            # 解析结果
            # 优先使用返回的fitness，如果没有则从balance计算
            if 'fitness' in result and result['fitness'] is not None:
                fitness = result['fitness']
            else:
                fitness = result.get('final_balance', 0) / result.get('initial_balance', 1) - 1
                fitness *= 100  # 转换为百分比
            
            eval_result = EvaluationResult(
                params=params,
                fitness=fitness,
                details=result,
                evaluation_id=eval_id,
                evaluation_time=evaluation_time,
                success=True
            )
            
            self.successful_evaluations += 1
            
            logger.debug(f"[评估{eval_id}] 成功: 参数={list(params.keys())[:3]}...fitness={fitness:.2f}%, "
                         f"time={evaluation_time:.1f}s")
            
            return eval_result
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            
            eval_result = EvaluationResult(
                params=params,
                fitness=-float('inf'),  # 失败赋予极低适应度
                details={'error': str(e)},
                evaluation_id=eval_id,
                evaluation_time=evaluation_time,
                success=False,
                error=str(e)
            )
            
            self.failed_evaluations += 1
            
            logger.error(f"[评估{eval_id}] 失败: {str(e)[:100]}")
            
            return eval_result

    def evaluate_batch(self, parameters_list: List[Dict[str, Any]], 
                      backtest_days: int = 60,
                      show_progress: bool = True) -> List[EvaluationResult]:
        """
        批量评估参数组合
        
        Args:
            parameters_list: 参数组合列表
            backtest_days: 回测天数
            show_progress: 是否显示进度条
            
        Returns:
            评估结果列表
        """
        logger.info(f"[ParallelEvaluator] 开始批量评估，数量={len(parameters_list)}, "
                   f"workers={self.max_workers}, backtest_days={backtest_days}")
        
        # 重置批次统计信息
        batch_successful = 0
        batch_failed = 0
        
        total = len(parameters_list)
        results = []
        completed = 0
        
        start_time_batch = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers, 
                                   thread_name_prefix='Evaluator') as executor:
                # 提交所有评估任务
                future_to_params = {}
                for i, params in enumerate(parameters_list, start=1):
                    # 检查缓存
                    param_hash = self._hash_params(params)
                    if self.cache_enabled and param_hash in self._cache:
                        eval_result = self._cache[param_hash]
                        results.append(eval_result)
                        self.cache_hits += 1
                        completed += 1
                        
                        if show_progress and completed % 10 == 0:
                            print(f"  进度: {completed}/{total} ({completed*100//total}%) [缓存]", end="\r", flush=True)
                        
                        continue
                    
                    # 提交新任务
                    future = executor.submit(self._evaluate_single, params, i, backtest_days)
                    future_to_params[future] = params
                
                # 收集结果
                for future in as_completed(future_to_params):
                    try:
                        result = future.result()
                        
                        # 添加到结果列表
                        results.append(result)
                        
                        # 更新统计
                        if result.success:
                            batch_successful += 1
                            self.successful_evaluations += 1
                        else:
                            batch_failed += 1
                            self.failed_evaluations += 1
                        
                        # 缓存成功的结果
                        if result.success and self.cache_enabled:
                            param_hash = self._hash_params(result.params)
                            self._cache[param_hash] = result
                        
                        completed += 1
                        self.total_evaluations += 1
                        
                        # 显示进度
                        if show_progress and completed % 10 == 0:
                            elapsed = time.time() - start_time_batch
                            avg_time = elapsed / completed
                            eta = avg_time * (total - completed) / 60  # 分钟
                            print(f"  进度: {completed}/{total} ({completed*100//total}%) "
                                  f"成功={batch_successful} 失败={batch_failed} "
                                  f"ETA={eta:.1f}min", end="\r", flush=True)
                        
                        # 进度回调
                        if self.progress_callback:
                            self.progress_callback(completed, total, result)
                            
                    except Exception as e:
                        logger.error(f"[ParallelEvaluator] 任务异常: {e}")
                        batch_failed += 1
                        self.failed_evaluations += 1
                        completed += 1
                
        except Exception as e:
            logger.error(f"[ParallelEvaluator] 批量评估失败: {e}")
        
        elapsed = time.time() - start_time_batch
        
        # 统计信息
        avg_fitness = sum(r.fitness for r in results) / len(results) if results else 0
        best_result = max(results, key=lambda x: x.fitness) if results else None
        
        logger.info(f"[ParallelEvaluator] 批量评估完成:")
        logger.info(f"  - 总评估: {len(results)}")
        logger.info(f"  - 成功: {batch_successful}, 失败: {batch_failed}")
        logger.info(f"  - 缓存命中: {self.cache_hits}")
        logger.info(f"  - 平均fitness: {avg_fitness:.2f}%")
        if best_result:
            logger.info(f"  - 最优fitness: {best_result.fitness:.2f}%")
        logger.info(f"  - 总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        logger.info(f"  - 平均每评估: {elapsed/len(results):.1f}秒")
        
        return results

    def evaluate_parallel_de_generation(self, population: List[Dict[str, Any]],
                                      backtest_days: int = 60) -> List[float]:
        """
        为DE优化器评估整个种群的适应度
        
        Args:
            population: 种群（参数组合列表）
            backtest_days: 回测天数
            
        Returns:
            适应度列表（对应population每个个体）
        """
        logger.debug(f"[ParallelEvaluator] 评估DE种群，size={len(population)}")
        
        # 使用批量评估
        results = self.evaluate_batch(population, backtest_days, show_progress=False)
        
        # 返回适应度列表
        fitness_list = []
        result_dict = {self._hash_params(r.params): r.fitness for r in results}
        
        for params in population:
            param_hash = self._hash_params(params)
            if param_hash in result_dict:
                fitness_list.append(result_dict[param_hash])
            else:
                # 理论上不应该发生
                fitness_list.append(-float('inf'))
        
        return fitness_list

    def get_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        return {
            'total_evaluations': self.total_evaluations,
            'successful_evaluations': self.successful_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'cache_hits': self.cache_hits,
            'success_rate': (self.successful_evaluations / self.total_evaluations 
                            if self.total_evaluations > 0 else 0)
        }

    def clear_cache(self):
        """清空评估缓存"""
        self._cache.clear()
        logger.info("[ParallelEvaluator] 评估缓存已清空")

    def save_cache(self, filepath: Path):
        """保存评估缓存到文件"""
        cache_data = {
            'cache': {k: {
                'params': v.params.dataclass.asdict(),
                'fitness': v.fitness,
                'details': v.details,
                'evaluation_time': v.evaluation_time,
                'success': v.success
            } for k, v in self._cache.items()},
            'statistics': self.get_statistics(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[ParallelEvaluator] 评估缓存已保存: {filepath}")

    def load_cache(self, filepath: Path):
        """从文件加载评估缓存"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 重建缓存
            self._cache = {}
            for k, v in cache_data['cache'].items():
                self._cache[k] = EvaluationResult(
                    params=v['params'],
                    fitness=v['fitness'],
                    details=v['details'],
                    evaluation_id=0,
                    evaluation_time=v['evaluation_time'],
                    success=v['success']
                )
            
            logger.info(f"[ParallelEvaluator] 评估缓存已加载: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[ParallelEvaluator] 加载缓存失败: {e}")
            return False


# 向后兼容的函数包装器
def create_parallel_evaluation_wrapper(backtest_func):
    """
    创建一个包装函数，将现有的回测函数包装成ParallelEvaluator可用的格式
    
    Args:
        backtest_func: 现有的回测函数，签名为 (params, backtest_days) -> result
        
    Returns:
        包装后的函数
    """
    def wrapper(params: Dict[str, Any], backtest_days: int = 60) -> Dict[str, Any]:
        result = backtest_func(params, backtest_days)
        return result
    
    return wrapper
