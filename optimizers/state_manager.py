#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态管理器 - 用于多日优化的状态保存和恢复
支持检查点、进度追踪、恢复执行
"""

import sys
import os
import json
import pickle
import hashlib
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# 设置日志
sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import Logger
logger = Logger.get_logger('state_manager')


@dataclass
class OptimizationState:
    """优化状态数据类"""
    metadata: Dict[str, Any]
    phase: str  # 当前阶段
    progress: int  # 当前进度（已评估次数）
    best_solution: Dict[str, Any]
    evaluation_history: List[Any]
    phase_results: Dict[str, Any]
    config: Dict[str, Any]
    timestamp: str
    
    # DE特定状态
    population: Optional[List[Dict[str, Any]]] = None
    generation: Optional[int] = None
    
    # CMA-ES特定状态
    mean: Optional[List[float]] = None
    sigma: Optional[float] = None
    covariance: Optional[List[List[float]]] = None


class StateManager:
    """
    状态管理器
    
    功能：
    1. 保存优化状态到文件
    2. 从文件恢复优化状态
    3. 检查点管理
    4. 进度追踪
    5. 配置验证
    """

    def __init__(self, optimizer_dir: Path = None):
        """
        初始化状态管理器
        
        Args:
            optimizer_dir: 优化器目录（默认为当前目录的optimizer子目录）
        """
        if optimizer_dir is None:
            optimizer_dir = Path(__file__).parent / "optimizer_state"
        
        self.optimizer_dir = Path(optimizer_dir)
        self.optimizer_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.optimizer_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.optimizer_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.optimizer_dir / "state.json"
        self.history_file = self.optimizer_dir / "history.csv"
        self.best_file = self.optimizer_dir / "best_result.json"
        
        logger.info(f"[StateManager] 初始化完成，目录: {self.optimizer_dir}")

    def _get_state_hash(self) -> str:
        """生成配置哈希，用于验证配置兼容性"""
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

    def init_state(self, config: Dict[str, Any]) -> OptimizationState:
        """
        初始化新状态
        
        Args:
            config: 优化配置
            
        Returns:
            OptimizationState对象
        """
        state = OptimizationState(
            metadata={
                'version': '1.0',
                'algorithm': 'hybrid_multi_method',
                'created_at': datetime.now().isoformat(),
                'state_hash': self._get_state_hash()
            },
            phase='initialization',
            progress=0,
            best_solution={'fitness': -float('inf'), 'params': {}},
            evaluation_history=[],
            phase_results={},
            config=config,
            timestamp=datetime.now().isoformat()
        )
        
        self.save_state(state)
        logger.info("[StateManager] 新状态已初始化")
        return state

    def save_state(self, state: OptimizationState):
        """
        保存优化状态（主文件）
        
        Args:
            state: OptimizationState对象
        """
        state.timestamp = datetime.now().isoformat()
        
        # 转换为可序列化的字典
        state_dict = {
            'metadata': state.metadata,
            'phase': state.phase,
            'progress': state.progress,
            'best_solution': state.best_solution,
            'evaluation_history': [{'params': h['params'], 'fitness': h['fitness']} 
                                  for h in state.evaluation_history],
            'phase_results': state.phase_results,
            'config': state.config,
            'timestamp': state.timestamp
        }
        
        # 如果有特定算法的状态，也保存
        if state.population is not None:
            state_dict['population'] = state.population
        if state.generation is not None:
            state_dict['generation'] = state.generation
        if state.mean is not None:
            state_dict['cma_mean'] = state.mean
        if state.sigma is not None:
            state_dict['cma_sigma'] = state.sigma
        
        # 保存主状态文件
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"[StateManager] 状态已保存: phase={state.phase}, progress={state.progress}")

    def load_state(self) -> Optional[OptimizationState]:
        """
        加载优化状态
        
        Returns:
            OptimizationState对象，如果文件不存在则返回None
        """
        if not self.state_file.exists():
            logger.warning("[StateManager] 状态文件不存在")
            return None
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            
            # 重建EvaluationHistory（添加evaluation_id）
            eval_history = []
            for i, h in enumerate(state_dict.get('evaluation_history', []), 1):
                eval_history.append({
                    'params': h['params'],
                    'fitness': h['fitness'],
                    'evaluation_id': i
                })
            
            # 恢复CMA-ES特定状态
            mean = state_dict.get('cma_mean')
            sigma = state_dict.get('cma_sigma')
            
            state = OptimizationState(
                metadata=state_dict['metadata'],
                phase=state_dict['phase'],
                progress=state_dict['progress'],
                best_solution=state_dict['best_solution'],
                evaluation_history=eval_history,
                phase_results=state_dict.get('phase_results', {}),
                config=state_dict['config'],
                timestamp=state_dict['timestamp'],
                population=state_dict.get('population'),
                generation=state_dict.get('generation'),
                mean=mean,
                sigma=sigma,
                covariance=state_dict.get('cma_covariance')
            )
            
            logger.info(f"[StateManager] 状态已加载: phase={state.phase}, "
                       f"progress={state.progress}, best_fitness={state.best_solution.get('fitness', -999):.2f}")
            return state
            
        except Exception as e:
            logger.error(f"[StateManager] 加载状态失败: {e}")
            return None

    def save_checkpoint(self, phase: str, generation: int, 
                       population: Optional[List[Dict[str, Any]]] = None,
                       best_solution: Dict[str, Any] = None):
        """
        保存检查点
        
        Args:
            phase: 当前阶段
            generation: 当前代数/进度
            population: 种群（可选）
            best_solution: 当前最佳解（可选）
        """
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{phase}_gen{generation}.json"
        
        checkpoint_data = {
            'phase': phase,
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'best_solution': best_solution
        }
        
        if population is not None:
            checkpoint_data['population'] = population
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"[StateManager] 检查点已保存: {checkpoint_file.name}")

    def load_latest_checkpoint(self, phase: str) -> Optional[Dict[str, Any]]:
        """
        加载指定阶段的最新检查点
        
        Args:
            phase: 阶段名称
            
        Returns:
            检查点数据，如果不存在则返回None
        """
        # 查找该阶段的所有检查点
        checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_{phase}_*.json"))
        
        if not checkpoints:
            logger.warning(f"[StateManager] 阶段 {phase} 没有检查点")
            return None
        
        # 按代数排序，取最新的
        checkpoints.sort()
        latest_checkpoint = checkpoints[-1]
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"[StateManager] 已加载最新检查点: {latest_checkpoint.name}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"[StateManager] 加载检查点失败: {e}")
            return None

    def save_result(self, params: Dict[str, Any], fitness: float,
                    phase: str, details: Dict[str, Any] = None):
        """
        保存单个评估结果到历史记录
        
        Args:
            params: 参数组合
            fitness: 适应度
            phase: 当前阶段
            details: 详细信息
        """
        result = {
            'params': params,
            'fitness': fitness,
            'phase': phase,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            result['details'] = details
        
        # 添加到历史
        state = self.load_state()
        if state:
            # 添加evaluation_id
            result['evaluation_id'] = len(state.evaluation_history) + 1
            state.evaluation_history.append(result)
            state.progress = len(state.evaluation_history)
            
            # 更新最佳解
            if fitness > state.best_solution.get('fitness', -float('inf')):
                state.best_solution = {
                    'fitness': fitness,
                    'params': params,
                    'phase': phase,
                    'evaluation_id': result['evaluation_id'],
                    'timestamp': result['timestamp']
                }
                
                # 保存最佳解单独文件
                self._save_best_solution(state.best_solution)
            
            self.save_state(state)

    def _save_best_solution(self, best_solution: Dict[str, Any]):
        """保存最佳解到单独文件"""
        with open(self.best_file, 'w', encoding='utf-8') as f:
            json.dump(best_solution, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"[StateManager] 最佳解已更新: fitness={best_solution['fitness']:.4f}")

    def save_phase_results(self, phase: str, results: List[Dict[str, Any]]):
        """
        保存阶段结果
        
        Args:
            phase: 阶段名称
            results: 阶段结果列表
        """
        state = self.load_state()
        if state:
            phase_stats = {
                'phase': phase,
                'num_evaluations': len(results),
                'best_fitness': max([r['fitness'] for r in results]),
                'avg_fitness': sum([r['fitness'] for r in results]) / len(results),
                'timestamp': datetime.now().isoformat()
            }
            
            state.phase_results[phase] = phase_stats
            self.save_state(state)
            
            # 保存详细结果（序列化numpy array）
            import numpy as np

            def serialize(obj):
                """JSON序列化辅助函数，处理numpy array"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize(item) for item in obj]
                else:
                    return obj

            # 序列化结果
            serializable_results = serialize(results)

            phase_file = self.optimizer_dir / f"phase_{phase}_results.json"
            with open(phase_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[StateManager] 阶段 {phase} 结果已保存，数量={len(results)}")

    def clear_old_checkpoints(self, keep_latest: int = 5):
        """
        清理旧检查点，只保留最新的几个
        
        Args:
            keep_latest: 保留最新检查点的数量
        """
        all_checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        
        if len(all_checkpoints) <= keep_latest:
            logger.debug("[StateManager] 不需要清理检查点")
            return
        
        # 按文件修改时间排序，删除旧的
        all_checkpoints.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        to_delete = all_checkpoints[keep_latest:]
        
        for checkpoint_file in to_delete:
            try:
                checkpoint_file.unlink()
                logger.debug(f"[StateManager] 已删除旧检查点: {checkpoint_file.name}")
            except Exception as e:
                logger.warning(f"[StateManager] 删除检查点失败 {checkpoint_file.name}: {e}")

    def get_progress_summary(self) -> Dict[str, Any]:
        """
        获取优化进度摘要
        
        Returns:
            进度摘要字典
        """
        state = self.load_state()
        if not state:
            return {'status': 'no_state'}
        
        summary = {
            'status': 'in_progress' if state.phase != 'completed' else 'completed',
            'phase': state.phase,
            'progress': state.progress,
            'best_fitness': state.best_solution.get('fitness', None),
            'total_phases': len(state.phase_results),
            'completed_phases': len(state.phase_results),
            'timestamp': state.timestamp
        }
        
        return summary

    def export_history_csv(self):
        """导出历史记录为CSV"""
        state = self.load_state()
        if not state:
            logger.warning("[StateManager] 无状态可导出")
            return
        
        try:
            import pandas as pd
            
            # 转换为DataFrame
            if state.evaluation_history:
                df = pd.DataFrame(state.evaluation_history)
                df.to_csv(self.history_file, index=False)
                logger.info(f"[StateManager] 历史记录已导出: {self.history_file}, 记录数={len(df)}")
            else:
                logger.warning("[StateManager] 没有历史记录可导出")
                
        except Exception as e:
            logger.error(f"[StateManager] 导出CSV失败: {e}")

    def validate_config(self, new_config: Dict[str, Any]) -> bool:
        """
        验证新配置与保存状态是否兼容
        
        Args:
            new_config: 新的配置字典
            
        Returns:
            True if compatible, False otherwise
        """
        state = self.load_state()
        if not state:
            logger.info("[StateManager] 无保存状态，新配置可以接受")
            return True
        
        old_config = state.config
        
        # 关键参数检查
        critical_keys = ['dimensionality', 'total_evaluations', 'backtest_days']
        
        for key in critical_keys:
            if key in old_config and key in new_config:
                if old_config[key] != new_config[key]:
                    logger.warning(f"[StateManager] 配置不兼容: {key} "
                                   f"(old={old_config[key]}, new={new_config[key]})")
                    return False
        
        logger.info("[StateManager] 配置验证通过")
        return True

    def cleanup(self):
        """
        清理优化状态
        - 删除检查点
        - 保留最终结果
        """
        logger.info("[StateManager] 开始清理...")
        
        # 清理检查点
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                checkpoint_file.unlink()
            except Exception as e:
                logger.warning(f"[StateManager] 删除检查点失败: {e}")
        
        # 删除检查点目录
        try:
            self.checkpoint_dir.rmdir()
        except:
            pass
        
        logger.info("[StateManager] 清理完成，保留了state.json和best_result.json")
