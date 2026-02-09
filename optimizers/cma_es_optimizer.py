#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 优化器
用于高维参数空间的局部精调优化

CMA-ES优势：
1. 自适应协方差矩阵，利用目标函数的局部几何结构
2. 对高维黑盒优化非常有效（20-200维度）
3. 收敛速度快，样本效率高
4. 基于进化策略原理，适用于不可导函数

基于论文: Hansen et al., "The CMA Evolution Strategy: A Comparing Review"
"""

import sys
import os
import logging
import random
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from datetime import datetime
from pathlib import Path

# 设置日志
sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import Logger
logger = Logger.get_logger('cma_es_optimizer')


class CMAES:
    """
    CMA-ES优化器（简化实现）
    
    参考资料:
    - Hansen, Nikolaus and Auger, Anne (2015)
    - "Evolution Strategies"
    """

    def __init__(self, param_bounds: Dict[str, Dict[str, float]],
                 population_size: Optional[int] = None,
                 max_generations: int = 100,
                 target_fitness: Optional[float] = None,
                 parallel_evaluator=None):
        """
        初始化CMA-ES优化器
        
        Args:
            param_bounds: 参数边界 {param_name: {'min': x, 'max': y}}
            population_size: 种群大小（如果不指定则根据维度计算）
            max_generations: 最大代数
            target_fitness: 目标适应度（达到后停止）
            parallel_evaluator: 并行评估器
        """
        self.dim = len(param_bounds)
        self.bounds = param_bounds
        self.params_names = list(param_bounds.keys())
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.evaluator = parallel_evaluator
        
        # 种群大小：4 + 3*ln(dim) (CMA-ES推荐值)
        if population_size is None:
            self.population_size = int(4 + 3 * np.log(self.dim))
        else:
            self.population_size = max(population_size, self.dim + 1)
        
        # CMA-ES参数
        self.mu_eff = self.population_size / 2  # 选择压力参考值（需要先计算）
        self.parents_size = int(self.population_size / 2)

        self.sigma = 0.2  # 初始步长（相对于搜索空间）
        self.cc = 4 / (self.dim + 4)  # 累积步长调整率
        self.cs = 0.83 / (self.dim + 4)  # 累积协方差矩阵调整率
        self.c1 = 2 / ((self.dim + 1.3)**2 + 1)  # 秩位向量进化的学习率
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2) / ((self.dim + 2) * (self.mu_eff - 2) + self.mu_eff))
        
        # 初始化种群
        self.population = self._initialize_population()
        
        # 协方差矩阵
        self.C = np.eye(self.dim)  # 初始为单位矩阵
        self.mean = np.mean(np.array([list(p.values()) for p in self.population]), axis=0)
        
        # 步长
        self.sigma = 0.3  # 搜索空间的30%
        
        # 秩位向量
        self.p_sigma = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        # 进化路径
        self.evolution_path = np.zeros((self.dim, 2))  # 存储最近的进化路径
        
        logger.info(f"[CMA-ES] 初始化: dim={self.dim}, pop_size={self.population_size}, "
                   f"max_gen={max_generations}")

    def _initialize_population(self) -> List[Dict[str, float]]:
        """
        初始化种群（均匀分布）
        
        Returns:
            参数组合列表
        """
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, bounds in self.bounds.items():
                # 在边界内均匀随机采样
                individual[param] = random.uniform(bounds['min'], bounds['max'])
            population.append(individual)
        
        return population

    def _dict_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """将字典转换为数组"""
        return np.array([params[p] for p in self.params_names])

    def _array_to_dict(self, arr: np.ndarray) -> Dict[str, float]:
        """将数组转换为字典"""
        return {self.params_names[i]: arr[i] for i in range(len(self.params_names))}

    def optimize(self) -> Dict[str, Any]:
        """
        执行CMA-ES优化
        
        Returns:
            最优解字典
        """
        logger.info(f"[CMA-ES] 开始优化，max_generations={self.max_generations}")
        
        # 评估初始种群
        fitness_values = self._evaluate_population()
        
        best_idx = np.argmax(fitness_values)
        self.best_fitness = fitness_values[best_idx]
        self.best_solution = self.population[best_idx]
        self.best_generation = 0
        
        logger.info(f"[CMA-ES] 初始最优: fitness={self.best_fitness:.4f}")
        
        # 进化循环
        for generation in range(1, self.max_generations + 1):
            logger.info(f"[CMA-ES] Generation {generation}/{self.max_generations}")
            
            # 选择父代（截断选择）
            parents_idx = self._select_parents(fitness_values)
            parents = [self.population[i] for i in parents_idx]
            
            # 计算均值和协方差矩阵
            self._update_mean_and_covariance(parents)
            
            # 生成新种群（采样）
            new_population = self._sample_population()
            
            # 评估新种群
            new_fitness = self._evaluate_population(new_population)
            
            # 选择策略（(μ, λ)-ES）
            combined_pop = self.population + new_population
            combined_fit = np.concatenate([fitness_values, new_fitness])
            
            # 选择最优的population_size个个体
            selected_idx = np.argsort(combined_fit)[-self.population_size:]
            self.population = [combined_pop[i] for i in selected_idx]
            self.fitness_values = combined_fit[selected_idx]
            
            # 更新进化路径
            self._update_evolution_path()
            
            # 检查收敛
            if self._should_stop(generation):
                logger.info(f"[CMA-ES] 在Generation {generation} 检测到收敛，停止优化")
                break
            
            # 更新最优解
            current_best_idx = np.argmax(self.fitness_values)
            if self.fitness_values[current_best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_values[current_best_idx]
                self.best_solution = self.population[current_best_idx]
                self.best_generation = generation
                logger.info(f"[CMA-ES] 更新最优: gen={generation}, fitness={self.best_fitness:.4f}")
        
        logger.info(f"[CMA-ES] 优化完成: best_fitness={self.best_fitness:.4f}, "
                   f"best_generation={self.best_generation}")
        
        return {
            'params': self.best_solution,
            'fitness': self.best_fitness,
            'generation': self.best_generation,
            'final_population': self.population,
            'final_fitness': self.fitness_values,
            'n_evaluations': len(self.final_population) * self.max_generations
        }

    def _select_parents(self, fitness: np.ndarray) -> np.ndarray:
        """
        截断选择父代
        选取适应度最高的 parents_size 个个体作为父代
        
        Returns:
        父代索引数组
        """
        idx = np.argsort(fitness)[-self.parents_size:]
        return idx

    def _update_mean_and_covariance(self, parents: List[Dict[str, float]]):
        """
        更新均值向量和协方差矩阵
        
        Args:
            parents: 父代个体列表
        """
        # 转换为矩阵
        parent_matrix = np.array([list(p.values()) for p in parents])
        
        # 更新均值
        old_mean = self.mean.copy()
        self.mean = np.mean(parent_matrix, axis=0)
        
        # 更新协方差矩阵
        # C_covariance = (1/n) * Σ (xi - mean)(xi - mean)^T
        deviations = parent_matrix - self.mean
        covariance = np.cov(deviations, rowvar=False)
        
        # 混合旧和新协方差矩阵
        self.C = (1 - self.cc) * covariance + self.cc * self.C
        
        # 确保正定性
        self.C = (self.C + self.C.T) / 2  # 对称化
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 0)  # 确保非负
        self.C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # 更新步长（根据进化路径）
        self._update_step_size()

    def _update_step_size(self):
        """
        根据进化路径更新步长
        """
        if np.linalg.norm(self.evolution_path[:, 1]) > 0:
            # 进化路径长度
            path_length = np.linalg.norm(self.evolution_path[:, 1])
            
            # 步长调整
            if path_length > 0.5:  # 进化快，增大步长
                self.sigma *= 1.2
            elif path_length < 0.2:  # 进化慢，减小步长
                self.sigma /= 1.1
            
            # 限制步长范围
            self.sigma = max(0.01, min(1.0, self.sigma))

    def _update_evolution_path(self):
        """
        更新进化路径
        """
        # 移动路径：丢弃最老的，加入新的
        self.evolution_path[:, 0] = self.evolution_path[:, 1]
        self.evolution_path[:, 1] = self.mean

    def _sample_population(self) -> List[Dict[str, float]]:
        """
        从多元正态分布采样新种群
        
        Returns:
        新种群
        """
        # 考虑边界处理
        new_population = []
        
        for _ in range(self.population_size):
            while True:
                # 生成样本
                z = np.random.multivariate_normal(self.mean, self.sigma ** 2 * self.C)
                
                # 转换为字典并检查边界
                sample_dict = self._array_to_dict(z)
                
                # 检查是否在搜索空间内
                in_bounds = all(self.bounds[p]['min'] <= sample_dict[p] <= self.bounds[p]['max'] 
                              for p in self.params_names)
                
                if in_bounds:
                    new_population.append(sample_dict)
                    break
                
                # 如果超出边界，重新采样（简化版本，更复杂的版本会使用反射处理）
                # 这是一种简化处理，真正的CMA-ES有更复杂的边界处理
                pass
        
        return new_population

    def _evaluate_population(self, population: Optional[List[Dict[str, float]]] = None
                     ) -> np.ndarray:
        """
        评估种群适应度
        
        Args:
            population: 要评估的种群（如果None则评估self.population）
            
        Returns:
            适应度数组
        """
        if population is None:
            population = self.population
        
        if self.evaluator:
            results = self.evaluator.evaluate_batch(population, show_progress=False)
            fitness = np.array([r.fitness for r in results])
        else:
            # 兜底：返回随机适应度
            fitness = np.random.randn(len(population))
        
        return fitness

    def _should_stop(self, generation: int) -> bool:
        """
        判断是否应该停止优化
        
        Args:
            generation: 当前代数
            
        Returns:
            是否应该停止
        """
        # 收敛检测：10代内改进小于阈值
        if generation >= 11 and len(self.history_fitness) >= 10:  # 需要至少11代才能比较第1代和第11代
            improvement = self.best_fitness - self.history_fitness[-10]
            if improvement < 0.001:  # 阈值
                logger.info(f"[CMA-ES] 收敛：{improvement:.6f}")
                return True
        
        # 步长太小
        if self.sigma < 1e-6:
            logger.info(f"[CMA-ES] 步长太小：{self.sigma:.6f}")
            return True
        
        return False

    @property
    def final_population(self) -> List[Dict[str, float]]:
        """返回最终种群"""
        return self.population

    @property
    def final_fitness(self) -> np.ndarray:
        """返回最终适应度"""
        return getattr(self, 'fitness_values', np.array([]))
    
    @property  
    def history_fitness(self) -> List[float]:
        """返回历史最佳适应度"""
        if not hasattr(self, '_history_fitness'):
            self._history_fitness = []
        return self._history_fitness

    def set_evaluator(self, evaluator):
        """设置评估器"""
        self.evaluator = evaluator
        logger.info("[CMA-ES] 评估器已设置")


class MultiStartCMAES:
    """
    Multi-start CMA-ES
    
    从多个随机起点开始，每个运行CMA-ES，最后选择最优结果
    增强全局探索能力
    """
    
    def __init__(self, param_bounds: Dict[str, Dict[str, float]],
                 num_starts: int = 5,
                 cma_params: Optional[Dict[str, Any]] = None):
        """
        初始化Multi-start CMA-ES
        
        Args:
            param_bounds: 参数边界
            num_starts: 独立启动次数
            cma_params: CMA-ES参数配置
        """
        self.param_bounds = param_bounds
        self.num_starts = num_starts
        
        if cma_params is None:
            cma_params = {}
        
        self.cma_params = cma_params
        self.all_results = []
        self.best_overall = None
        
        logger.info(f"[MultiStartCMAES] 初始化: num_starts={num_starts}")

    def optimize(self, parallel_evaluator=None) -> Dict[str, Any]:
        """
        执行Multi-start优化
        
        Args:
            parallel_evaluator: 并行评估器
            
        Returns:
            最优解字典
        """
        logger.info(f"[MultiStartCMAES] 开始Multi-start优化，{self.num_starts}个起点")
        
        for start_id in range(1, self.num_starts + 1):
            logger.info(f"[MultiStartCMAES] ===== 起点 {start_id}/{self.num_starts} =====")
            
            # 随机初始化（扰动参数范围）
            perturbed_bounds = self._perturb_bounds()
            
            # 创建CMA-ES实例
            cma = CMAES(
                perturbed_bounds,
                parallel_evaluator=parallel_evaluator,
                **self.cma_params
            )
            
            # 运行优化
            result = cma.optimize()
            result['start_id'] = start_id
            result['perturbed_bounds'] = perturbed_bounds
            
            self.all_results.append(result)
            
            # 更新全局最优
            if self.best_overall is None or result['fitness'] > self.best_overall['fitness']:
                self.best_overall = result
                logger.info(f"[MultiStartCMAES] 新的全局最优: start={start_id}, "
                           f"fitness={result['fitness']:.4f}")
        
        logger.info(f"[MultiStartCMAES] 所有start完成")
        logger.info(f"  总评估数: {sum(r['n_evaluations'] for r in self.all_results)}")
        
        # 按fitness排序
        sorted_results = sorted(self.all_results, key=lambda x: x['fitness'], reverse=True)
        
        logger.info(f"Top 5结果:")
        for i, r in enumerate(sorted_results[:5], 1):
            logger.info(f"  {i}. fitness={r['fitness']:.4f}, start={r['start_id']}")

        # 确保始终返回有效的结果字典
        if self.best_overall is None:
            logger.warning("[MultiStartCMAES] 未找到有效结果，返回默认值")
            return {
                'params': {},
                'fitness': -float('inf'),
                'generation': 0,
                'start_id': 0,
                'n_evaluations': 0
            }
        return self.best_overall

    def _perturb_bounds(self, perturb_factor: float = 0.3) -> Dict[str, Dict[str, float]]:
        """
        扰动参数边界（用于多start的多样性）
        
        Args:
            perturb_factor: 扰动因子（0.3 = ±30%）
            
        Returns:
            扰动后的参数边界
        """
        perturbed = {}
        
        for param, bounds in self.param_bounds.items():
            mid = (bounds['min'] + bounds['max']) / 2
            width = bounds['max'] - bounds['min']
            
            # 重新计算边界（保持宽度，移动中心）
            shift = (random.random() - 0.5) * 2 * perturb_factor * width
            new_mid = mid + shift
            perturbed[param] = {
                'min': max(bounds['min'], new_mid - width / 2),
                'max': min(bounds['max'], new_mid + width / 2)
            }
        
        return perturbed
