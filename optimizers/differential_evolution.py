#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differential Evolution (DE) 算法实现
全局优化的经典进化算法，特别适合高维非凸优化

DE优势：
1. 专为全局优化设计，能有效避免局部最优
2. 参数少，易于调优
3. 收敛速度快，样本效率高
4. 对参数缩放不敏感

基于论文: Storn and Price (1997) "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces"
"""

import sys
import os
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from pathlib import Path

# 设置日志
sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import Logger
logger = Logger.get_logger('differential_evolution')


class DifferentialEvolution:
    """
    差分进化算法（DE）优化器
    
    使用策略: DE/rand/1/bin (经典DE策略)
    """

    def __init__(self, param_bounds: Dict[str, Dict[str, float]],
                 population_size: int = 50,
                 max_generations: int = 500,
                 F: float = 0.8,  # 差分权重 [0, 2]
                 CR: float = 0.9,  # 交叉概率 [0, 1]
                 target_fitness: Optional[float] = None,
                 parallel_evaluator=None):
        """
        初始化DE优化器
        
        Args:
            param_bounds: 参数边界 {param_name: {'min': x, 'max': y}}
            population_size: 种群大小
            max_generations: 最大代数
            F: 差分权重
            CR: 交叉概率
            target_fitness: 目标适应度
            parallel_evaluator: 并行评估器
        """
        self.dim = len(param_bounds)
        self.bounds = param_bounds
        self.params_names = list(param_bounds.keys())
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.target_fitness = target_fitness
        self.evaluator = parallel_evaluator
        
        logger.info(f"[DE] 初始化: dim={self.dim}, pop_size={population_size}, "
                   f"max_gen={max_generations}, F={F}, CR={CR}")
        
        # 初始化种群
        self.population = self._initialize_population()
        self.fitness = np.full(population_size, -float('inf'))
        
        # 最佳解
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.best_generation = 0
        
        # 历史记录
        self.history_best_fitness = []  # 每代的最优适应度
        self.history_mean_fitness = []   # 每代的平均适应度
        
        return

    def _initialize_population(self) -> List[Dict[str, float]]:
        """初始化种群（均匀分布）"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, bounds in self.bounds.items():
                individual[param] = random.uniform(bounds['min'], bounds['max'])
            population.append(individual)
        return population

    def _dict_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """字典转数组"""
        return np.array([params[p] for p in self.params_names])

    def _array_to_dict(self, arr: np.ndarray) -> Dict[str, float]:
        """数组转字典"""
        return {self.params_names[i]: float(arr[i]) for i in range(len(self.params_names))}

    def mutate(self, target_idx: int, population: List[np.ndarray]) -> np.ndarray:
        """
        变异操作 - DE/rand/1/bin
        V = X_r1 + F * (X_r2 - X_r3)
        
        Args:
            target_idx: 目标个体索引
            population: 当前种群（数组形式）
            
        Returns:
            变异向量
        """
        # 随机选择3个不同的个体（必须不同于目标个体）
        candidates = [i for i in range(len(population)) if i != target_idx]
        r1, r2, r3 = random.sample(candidates, 3)
        
        # 变异公式: V = X_r1 + F * (X_r2 - X_r3)
        return population[r1] + self.F * (population[r2] - population[r3])

    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        交叉操作 - 二项式交叉
        
        Args:
            target: 目标个体
            mutant: 变异向量
            
        Randomly选参数
        Returns:
            试验向量
        """
        trial = target.copy()
        
        # 对每个维度，以CR概率选择变异个体的值，否则保留原值
        for i in range(self.dim):
            if random.random() < self.CR:
                trial[i] = mutant[i]
        
        return trial

    def ensure_bounds(self, trial: np.ndarray) -> np.ndarray:
        """
        确保试验向量在边界内（反射策略）
        
        Args:
            trial: 试验向量
            
        Returns:
        边界内的向量
        """
        for i in range(self.dim):
            min_val = self.bounds[self.params_names[i]]['min']
            max_val = self.bounds[self.params_names[i]]['max']
            
            if trial[i] < min_val:
                trial[i] = min_val + random.uniform(0, (max_val - min_val) * 0.1)
            elif trial[i] > max_val:
                trial[i] = max_val - random.uniform(0, (max_val - min_val) * 0.1)
        
        return trial

    def optimize(self) -> Dict[str, Any]:
        """
        执行DE优化
        
        Returns:
            最优解字典
        """
        logger.info(f"[DE] 开始优化，generations={self.max_generations}")
        
        # 评估初始种群
        self.fitness = self._evaluate_population()
        
        # 找到初始最优
        best_idx = np.argmax(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.population[best_idx]
        self.best_generation = 0
        
        self.history_best_fitness.append(self.best_fitness)
        self.history_mean_fitness.append(np.mean(self.fitness))
        
        logger.info(f"[DE] 初始最优: fitness={self.best_fitness:.4f}")
        
        # 进化循环
        for generation in range(1, self.max_generations + 1):
            logger.debug(f"[DE] Generation {generation}/{self.max_generations}")
            
            new_population = []
            new_fitness = []
            
            for i in range(self.population_size):
                target_array = self._dict_to_array(self.population[i])
                
                # 变异
                mutant = self.mutate(i, self._to_array_population())
                
                # 交叉
                trial_array = self.crossover(target_array, mutant)
                
                # 边界检查
                trial_array = self.ensure_bounds(trial_array)
                trial_dict = self._array_to_dict(trial_array)
                
                # 评估
                fitness = self._evaluate_single_trial(trial_dict)
                new_population.append(trial_dict)
                new_fitness.append(fitness)
                
                # 选择策略：贪婪选择（如果新个体更好，替换旧个体）
                if fitness > self.fitness[i]:
                    self.population[i] = trial_dict
                    self.fitness[i] = fitness
            
            # 每代结束后的处理
            current_best_idx = np.argmax(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            
            # 更新全局最优
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[current_best_idx]
                self.best_generation = generation
                logger.debug(f"[DE] Generation {generation}: 更新最优 fitness={self.best_fitness:.4f}")
            
            # 记录历史
            self.history_best_fitness.append(self.best_fitness)
            self.history_mean_fitness.append(np.mean(self.fitness))
            
            # 每10代显示一次进度
            if generation % 10 == 0:
                logger.info(f"[DE] Generation {generation}: best={self.best_fitness:.4f}, "
                       f"mean={np.mean(self.fitness):.4f}, std={np.std(self.fitness):.4f}")
            
            # 收敛检测
            if self._should_stop(generation):
                logger.info(f"[DE] 在Generation {generation} 检测到收敛，停止优化")
                break
            
            # 检查是否达到目标
            if self.target_fitness and self.best_fitness >= self.target_fitness:
                logger.info(f"[DE] 达到目标适应度 {self.target_fitness}")
                break
        
        logger.info(f"[DE] 优化完成: best_fitness={self.best_fitness:.4f}, "
                   f"best_generation={self.best_generation}")
        
        return {
            'params': self.best_solution,
            'fitness': self.best_fitness,
            'generation': self.best_generation,
            'history_best': self.history_best_fitness,
            'history_mean': self.history_mean_fitness,
            'final_population': self.population,
            'final_fitness': self.fitness,
            'n_evaluations': self.population_size * self.best_generation
        }

    def _to_array_population(self) -> List[np.ndarray]:
        """将种群转换为数组列表"""
        return [self._dict_to_array(ind) for ind in self.population]

    def _evaluate_population(self) -> np.ndarray:
        """评估整个种群（使用并行评估器如果可用）"""
        if self.evaluator:
            results = self.evaluator.evaluate_batch(self.population, show_progress=False)
            return np.array([r.fitness for r in results])
        else:
            # 当没有并行评估器时，逐个评估
            fitness_values = []
            for ind in self.population:
                arr = self._dict_to_array(ind)
                fitness = self._evaluate_single(arr)
                fitness_values.append(fitness)
            return np.array(fitness_values)

    def _evaluate_single_trial(self, trial_dict: Dict[str, float]) -> float:
        """单个评估（没有并行评估器时的回退）"""
        # 这里应该调用实际的回测函数
        # 返回适应度（收益率）
        logger.debug(f"[DE] 评估参数: {list(trial_dict.keys())[:3]}...")
        return random.uniform(0, 100)

    def _evaluate_single(self, params: Union[Dict, np.ndarray, List[Dict]],
                       trial_array: Optional[np.ndarray] = None) -> float:
        """评估包装器（单个参数评估）"""
        # 处理不同输入类型，但此函数只评估单个参数
        if isinstance(params, list) and len(params) == 1:
            params = params[0]
        elif isinstance(params, list):
            # 多个参数的情况，在 _evaluate_population 中处理
            raise ValueError("_evaluate_single 只用于单个参数评估，请使用 _evaluate_population")

        # 兜底：返回随机值（实际使用时会设置真实的评估函数）
        logger.debug("[DE] 未设置评估函数，返回随机值")
        return random.uniform(0, 100)

    def _should_stop(self, generation: int) -> bool:
        """停止条件判断"""
        # 要求至少运行50代
        if generation < 50:
            return False
        
        # 检查最近20代是否有改进
        if generation > 20:
            recent_improvement = max(0, self.history_best_fitness[-1] - self.history_best_fitness[-20])
            if recent_improvement < 0.01:  # 小于0.01%的改进
                logger.debug(f"[DE] 20代内无显著改进: {recent_improvement:.6f}")
                return True
        
        # 检查种群多样性（标准差）
        if generation > 30:
            std_dev = np.std(self.fitness)
            if std_dev < 0.01:  # 种群收敛到极小范围
                logger.debug(f"[DE] 种群多样性缺失: std={std_dev:.6f}")
                return True
        
        return False

    def set_evaluator(self, evaluator):
        """设置评估器"""
        self.evaluator = evaluator
        logger.info("[DE] 评估器已设置")


class MultiStartDE:
    """
    Multi-Start Differential Evolution
    
    从多个随机起点开始独立运行DE，最后选择最优结果
    """
    
    def __init__(self, param_bounds: Dict[str, Dict[str, float]],
                 num_starts: int = 10,
                 population_size: int = 40,
                 generations: int = 300,
                 parallel_evaluator=None):
        """
        初始化Multi-Start DE
        
        Args:
            param_bounds: 参数边界
            num_starts: 启动次数
            population_size: 每个start的种群大小
            generations: 每个start的代数
            parallel_evaluator: 并行评估器
        """
        self.param_bounds = param_bounds
        self.num_starts = num_starts
        self.population_size = population_size
        self.generations = generations
        self.evaluator = parallel_evaluator
        
        self.all_results = []
        self.best_overall = None
        
        logger.info(f"[MultiStartDE] 初始化: num_starts={num_starts}, "
                   f"pop_size={population_size}, generations={generations}")

    def optimize(self) -> Dict[str, Any]:
        """
        执行Multi-Start DE优化
        
        Returns:
            最优解字典
        """
        logger.info(f"[MultiStartDE] 开始Multi-Start优化，{self.num_starts}个独立start")
        
        all_start_results = []
        
        for start_id in range(1, self.num_starts + 1):
            logger.info(f"[MultiStartDE] ===== Start {start_id}/{self.num_starts} =====")
            
            # 随机初始化（扰动边界以增加多样性）
            perturbed_bounds = self._perturb_bounds()
            
            # 创建DE实例
            de = DifferentialEvolution(
                perturbed_bounds,
                population_size=self.population_size,
                max_generations=self.generations,
                parallel_evaluator=self.evaluator
            )
            
            # 运行优化
            start_time = datetime.now()
            result = de.optimize()
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            
            result['start_id'] = start_id
            result['elapsed_minutes'] = elapsed
            result['perturbed_bounds'] = perturbed_bounds
            
            all_start_results.append(result)
            
            # 更新全局最优
            if self.best_overall is None or result['fitness'] > self.best_overall['fitness']:
                self.best_overall = result
                logger.info(f"[MultiStartDE] 新的全局最优: start={start_id}, "
                           f"fitness={result['fitness']:.4f}, gen={result['generation']}")
            
            logger.info(f"[MultiStartDE] Start {start_id}完成: fitness={result['fitness']:.4f}, "
                       f"耗时={elapsed:.1f}min")
        
        # 按fitness排序所有结果
        sorted_results = sorted(all_start_results, key=lambda x: x['fitness'], reverse=True)
        
        self.all_results = sorted_results
        
        logger.info(f"\n[MultiStartDE] {self.num_starts}个start全部完成")
        logger.info(f"  总评估次数: {sum(r.get('n_evaluations', 0) for r in self.all_results):,}")
        logger.info(f"  总耗时: {sum(r.get('elapsed_minutes', 0) for r in self.all_results):.1f}分钟")
        
        logger.info(f"\n[MultiStartDE] Top 5结果:")
        for i, r in enumerate(sorted_results[:5], 1):
            logger.info(f"  {i}. fitness={r['fitness']:.4f}, start={r['start_id']}, "
                       f"gen={r.get('generation', 0)}, time={r.get('elapsed_minutes', 0):.1f}min")
        
        # 统计有多少个start找到了相似的优良解（全局最优5%以内）
        top_fitness = sorted_results[0]['fitness']
        similar_count = sum(1 for r in sorted_results if r['fitness'] >= top_fitness * 0.95)
        logger.info(f"\n[MultiStartDE] 统计:")
        logger.info(f"  - 最优fitness: {top_fitness:.4f}")
        logger.info(f"  - 接近最优的start数(±5%): {similar_count}/{self.num_starts}")
        
        # 确保始终返回有效的结果字典
        if self.best_overall is None:
            logger.warning("[MultiStartDE] 未找到有效结果，返回默认值")
            return {
                'params': {},
                'fitness': -float('inf'),
                'generation': 0,
                'start_id': 0,
                'n_evaluations': 0,
                'elapsed_minutes': 0
            }
        return self.best_overall

    def _perturb_bounds(self, perturb_factor: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        扰动参数边界
        
        Args:
            perturb_factor: 扰动因子
            
        Returns:
            扰动后的边界
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
