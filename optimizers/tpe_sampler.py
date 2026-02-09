#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPE (Tree-structured Parzen Estimator) 采样器
基于贝叶斯优化的样本效率优化方法

TPE优势：
1. 相比随机搜索，样本效率提升3-5倍
2. 利用历史评估结果构建概率模型
3. 智能采样：在低价值区域采样更多，在高价值区域采样更少
4. 适合黑盒函数优化
"""

import sys
import os
import logging
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from scipy.stats import gamma, norm

# 设置日志
sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import Logger
logger = Logger.get_logger('tpe_sampler')


class TPE_Sampler:
    """
    Tree-structured Parzen Estimator采样器
    
    基于论文: "Making a Science of Model Search" (Bergstra et al., 2011)
    """
    
    def __init__(self, param_bounds: Dict[str, Dict[str, float]],
                 n_observations: int = 20):
        """
        初始化TPE采样器
        
        Args:
            param_bounds: 参数边界 {param_name: {'min': x, 'max': y}}
            n_observations: 构建模型所需的最小观察数
        """
        self.param_bounds = param_bounds
        self.n_observations = n_observations
        self.params_names = list(param_bounds.keys())
        self.dim = len(self.params_names)
        
        # 历史观察
        self.observed_params = []
        self.observed_fitness = []
        
        # gamma: 用于划分观察的分位数
        self.gamma = 0.25  # 前25%作为"好"的观察
        
        logger.info(f"[TPE_Sampler] 初始化，dim={self.dim}, n_obs={n_observations}")

    def add_observation(self, params: Dict[str, Any], fitness: float):
        """
        添加观察
        
        Args:
            params: 参数组合
            fitness: 适应度（fitness）
        """
        self.observed_params.append(params)
        self.observed_fitness.append(fitness)
        
        logger.debug(f"[TPE_Sampler] 添加观察: fitness={fitness:.4f}, total={len(self.observed_fitness)}")

    def sample(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """
        采样新参数
        
        Args:
            n_samples: 采样数量
            
        Returns:
            新参数列表
        """
        if len(self.observed_params) < self.n_observations:
            # 观察不足，使用随机采样
            return self._random_sample(n_samples)
        
        # 计算gamma分位数
        threshold = np.percentile(self.observed_fitness, self.gamma * 100)
        
        # 划分"好"和"坏"的观察
        good_indices = [i for i, f in enumerate(self.observed_fitness) if f >= threshold]
        bad_indices = [i for i, f in enumerate(self.observed_fitness) if f < threshold]
        
        if len(good_indices) == 0 or len(bad_indices) == 0:
            return self._random_sample(n_samples)
        
        good_params = [self.observed_params[i] for i in good_indices]
        bad_params = [self.observed_params[i] for i in bad_indices]
        
        # 为每个参数采样
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param in self.params_names:
                sample[param] = self._sample_single_param(
                    param, good_params, bad_params
                )
            samples.append(sample)
        
        return samples

    def _sample_single_param(self, param: str,
                            good_params: List[Dict[str, Any]],
                            bad_params: List[Dict[str, Any]]) -> float:
        """
        采样单个参数
        
        Args:
            param: 参数名
            good_params: "好"的观察列表
            bad_params: "坏"的观察列表
            
        Returns:
            采样值
        """
        good_values = [p[param] for p in good_params]
        bad_values = [p[param] for p in bad_params]
        
        if len(good_values) == 0 or len(bad_values) == 0:
            # 兜底：使用随机采样
            low = self.param_bounds[param]['min']
            high = self.param_bounds[param]['max']
            return random.uniform(low, high)
        
        # 估计"好"和"坏"分布的参数
        l_good, s_good = self._fit_gamma(good_values)
        l_bad, s_bad = self._fit_gamma(bad_values)
        
        # 从"好"分布采样一次
        while True:
            x = self._sample_gamma(l_good, s_good)
            # 检查边界
            if self.param_bounds[param]['min'] <= x <= self.param_bounds[param]['max']:
                # 计算该值在"坏"分布中的概率
                prob_bad = self._pdf_gamma(x, l_bad, s_bad)
                if prob_bad > 0:
                    # 根据采样策略决定是否接受
                    if random.random() < 0.5 or prob_bad < 0.01:
                        return x
        # 如果循环很久，返回随机值
        return random.uniform(self.param_bounds[param]['min'], 
                              self.param_bounds[param]['max'])

    def _fit_gamma(self, values: List[float]) -> Tuple[float, float]:
        """
        使用最大似然估计拟合Gamma分布
        
        Args:
            values: 数值列表
            
        Returns:
            (shape, scale) 参数
        """
        # 标准化到正值
        values = np.array(values) - min(values)
        min_val = values.min()
        if min_val <= 0:
            values += (abs(min_val) + 1e-8)
        
        # 使用scipy的gamma fit
        try:
            shape, loc, scale = gamma.fit(values)
            return shape, scale
        except:
            # 如果拟合失败，返回默认值
            return 2.0, 1.0

    def _sample_gamma(self, shape: float, scale: float) -> float:
        """从Gamma分布采样"""
        return gamma.rvs(shape, scale=scale)

    def _pdf_gamma(self, x: float, shape: float, scale: float) -> float:
        """计算Gamma分布的PDF"""
        return gamma.pdf(x, shape, scale=scale)

    def _random_sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """随机采样（兜底方法）"""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, bounds in self.param_bounds.items():
                sample[param] = random.uniform(bounds['min'], bounds['max'])
            samples.append(sample)
        return samples

    def update_gamma(self, gamma: float):
        """更新gamma参数"""
        self.gamma = gamma
        logger.debug(f"[TPE_Sampler] gamma更新为 {gamma}")


class TPE_Optimizer:
    """
    基于TPE的优化器
    
    结合TPE采样和并行评估进行高效优化
    """
    
    def __init__(self, param_bounds: Dict[str, Dict[str, float]],
                 max_evaluations: int = 1000,
                 n_initial: int = 100,
                 parallel_evaluator=None):
        """
        初始化TPE优化器
        
        Args:
            param_bounds: 参数边界
            max_evaluations: 最大评估次数
            n_initial: 初始随机评估次数
            parallel_evaluator: 并行评估器
        """
        self.param_bounds = param_bounds
        self.max_evaluations = max_evaluations
        self.n_initial = n_initial
        self.evaluator = parallel_evaluator
        
        self.tpe = TPE_Sampler(param_bounds, n_observations=20)
        
        self.evaluation_history = []
        self.best_solution = None
        self.best_fitness = -float('inf')
        
        logger.info(f"[TPE_Optimizer] 初始化，max_evals={max_evaluations}, "
                   f"n_initial={n_initial}")

    def optimize(self) -> Dict[str, Any]:
        """
        执行优化
        
        Returns:
            最优解字典
        """
        logger.info(f"[TPE_Optimizer] 开始TPE优化")
        
        # Phase 1: 随机初始化
        logger.info("[TPE_Optimizer] Phase 1: 随机初始化")
        initial_samples = [self._random_sample() for _ in range(self.n_initial)]
        
        if self.evaluator:
            initial_results = self.evaluator.evaluate_batch(initial_samples)
        else:
            initial_results = [{'params': s, 'fitness': self._evaluate_single(s)} 
                             for s in initial_samples]
        
        # 记录观察
        for result in initial_results:
            # 处理 EvaluationResult 对象
            if hasattr(result, 'params'):
                params = result.params
                fitness = result.fitness
            else:
                # 字典格式（兼容旧代码）
                params = result['params']
                fitness = result['fitness']

            self.tpe.add_observation(params, fitness)
            # 转换为字典格式保存到历史记录
            self.evaluation_history.append({'params': params, 'fitness': fitness})
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = params
        
        # Phase 2: TPE迭代优化
        logger.info("[TPE_Optimizer] Phase 2: TPE迭代优化")
        n_iterations = (self.max_evaluations - self.n_initial)
        batch_size = min(100, n_iterations)  # 每批100个
        
        for iteration in range(0, n_iterations, batch_size):
            current_batch_size = min(batch_size, n_iterations - iteration)
            
            # TPE采样
            if iteration > 0 and iteration % 50 == 0:
                # 每50次评估，动态调整gamma
                self.tpe.update_gamma(0.3)
            
            samples = self.tpe.sample(n_samples=current_batch_size)
            
            # 评估
            if self.evaluator:
                results = self.evaluator.evaluate_batch(samples)
            else:
                results = [{'params': s, 'fitness': self._evaluate_single(s)} 
                         for s in samples]
            
            # 记录和更新
            for result in results:
                # 处理 EvaluationResult 对象
                if hasattr(result, 'params'):
                    params = result.params
                    fitness = result.fitness
                else:
                    # 字典格式（兼容旧代码）
                    params = result['params']
                    fitness = result['fitness']

                self.tpe.add_observation(params, fitness)
                # 转换为字典格式保存到历史记录
                self.evaluation_history.append({'params': params, 'fitness': fitness})
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = params
            
            # 每批显示进度
            total = iteration + current_batch_size + self.n_initial
            logger.info(f"[TPE_Optimizer] 进度: {total}/{self.max_evaluations} "
                       f"({total*100//self.max_evaluations}%), "
                       f"best_fitness={self.best_fitness:.2f}%")
        
        logger.info(f"[TPE_Optimizer] 优化完成，总评估={len(self.evaluation_history)}, "
                   f"best_fitness={self.best_fitness:.2f}%")
        
        return {
            'params': self.best_solution,
            'fitness': self.best_fitness,
            'n_evaluations': len(self.evaluation_history),
            'history': self.evaluation_history
        }

    def _random_sample(self) -> Dict[str, Any]:
        """随机采样一个参数组合"""
        sample = {}
        for param, bounds in self.param_bounds.items():
            sample[param] = random.uniform(bounds['min'], bounds['max'])
        return sample

    def _evaluate_single(self, params: Dict[str, Any]) -> float:
        """单个评估（如果没有提供并行 evaluator）"""
        # 这里只返回占位值，实际评估需要用户提供评估函数
        logger.warning("[TPE_Optimizer] 未设置评估函数，返回随机值")
        return random.uniform(0, 100)

    def set_evaluator(self, evaluator):
        """设置并行评估器"""
        self.evaluator = evaluator
        logger.info("[TPE_Optimizer] 并行评估器已设置")


def create_adaptive_gamma_strategy():
    """
    创建自适应gamma策略
    
    思路：根据优化进度动态调整gamma
    - 早期：gamma较低(0.1-0.2)，保持探索
    - 中期：gamma适中(0.25-0.3)，平衡探索和利用
    - 后期：gamma较高(0.4-0.5)，聚焦利用
    """
    class AdaptiveGamma:
        def __init__(self):
            self.iterations = 0
        
        def get_gamma(self, iteration: int, max_iterations: int) -> float:
            """
            根据迭代进度返回适当的gamma值
            
            Args:
                iteration: 当前迭代
                max_iterations: 最大迭代数
                
            Returns:
                gamma值
            """
            progress = iteration / max_iterations
            
            if progress < 0.2:
                # 早期：低gamma，保持探索
                gamma = 0.15 + 0.05 * progress / 0.2
            elif progress < 0.5:
                # 中期：中等gamma
                gamma = 0.25 + 0.05 * (progress - 0.2) / 0.3
            elif progress < 0.8:
                # 中后期：较高gamma
                gamma = 0.35 + 0.1 * (progress - 0.5) / 0.3
            else:
                # 后期：高gamma，深度利用
                gamma = 0.45
            
            return gamma
    
    return AdaptiveGamma()
