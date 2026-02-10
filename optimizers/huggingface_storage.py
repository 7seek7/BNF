#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 持久化存储模块

用于 Streamlit Cloud 等临时容器环境，将优化状态持久化到 HuggingFace。
容器重启后可以从 HuggingFace 恢复优化状态。

依赖: pip install huggingface_hub
"""

import sys
import os
import json
import pickle
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import Logger
logger = Logger.get_logger('huggingface_storage')


try:
    from huggingface_hub import (
        HfApi,
        login,
        create_repo,
        upload_file,
        hf_hub_download,
        Repository,
        snapshot_download
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("[HuggingFaceStorage] huggingface_hub 未安装")


class HuggingFaceStorage:
    """
    HuggingFace 持久化存储

    功能：
    1. 上传优化状态到 HuggingFace Datasets
    2. 从 HuggingFace 下载优化状态
    3. 自动同步本地文件到 HuggingFace
    4. 支持恢复模式

    使用方法：
    1. 创建 HuggingFace Token: https://huggingface.co/settings/tokens
    2. 在 settings.py 中配置 HUGGINGFACE_TOKEN
    3. 调用 upload_state() 上传状态
    4. 调用 download_state() 下载状态
    """

    def __init__(self,
                 repo_id: Optional[str] = None,
                 token: Optional[str] = None,
                 local_dir: Optional[Path] = None):
        """
        初始化 HuggingFace 存储

        Args:
            repo_id: HuggingFace 仓库 ID，格式为 "username/repo-name"
            token: HuggingFace access token
            local_dir: 本地目录（用于缓存）
        """
        self.token = token
        self.repo_id = repo_id
        self.local_dir = local_dir or Path(__file__).parent / "optimizer_state"
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.api = None
        self.initialized = False

        if self._init_huggingface():
            logger.info(f"[HuggingFaceStorage] 初始化成功")
            logger.info(f"  仓库: {self.repo_id}")
            logger.info(f"  本地目录: {self.local_dir}")
        else:
            logger.warning("[HuggingFaceStorage] 初始化失败，将使用本地存储")

    def _init_huggingface(self) -> bool:
        """初始化 HuggingFace 连接"""
        if not HF_AVAILABLE:
            logger.warning("[HuggingFaceStorage] huggingface_hub 未安装")
            return False

        try:
            # 获取 token
            if not self.token:
                # 尝试从环境变量获取
                self.token = os.environ.get('HUGGINGFACE_TOKEN')
                if not self.token:
                    # 尝试从 settings 获取
                    try:
                        from config.settings import settings
                        self.token = getattr(settings, 'HUGGINGFACE_TOKEN', None)
                    except Exception:
                        pass

                    if not self.token:
                        logger.warning("[HuggingFaceStorage] 未配置 HUGGINGFACE_TOKEN")
                        return False

            # 登录
            login(token=self.token)
            self.api = HfApi(token=self.token)

            # 检查或创建仓库
            if self.repo_id:
                try:
                    self.api.repo_info(repo_id=self.repo_id, repo_type="dataset")
                    logger.info(f"[HuggingFaceStorage] 仓库已存在: {self.repo_id}")
                except Exception as e:
                    # 仓库不存在，创建新仓库
                    try:
                        create_repo(repo_id=self.repo_id, repo_type="dataset", private=True)
                        logger.info(f"[HuggingFaceStorage] 创建仓库: {self.repo_id}")
                    except Exception as create_error:
                        logger.error(f"[HuggingFaceStorage] 创建仓库失败: {create_error}")
                        return False

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"[HuggingFaceStorage] 初始化失败: {e}")
            return False

    def upload_file(self, file_path: str, remote_path: Optional[str] = None) -> bool:
        """
        上传单个文件到 HuggingFace

        Args:
            file_path: 本地文件路径
            remote_path: 远程文件路径（默认使用文件名）

        Returns:
            是否上传成功
        """
        if not self.initialized:
            logger.warning("[HuggingFaceStorage] 未初始化，跳过上传")
            return False

        if not self.repo_id:
            logger.warning("[HuggingFaceStorage] 未配置仓库，跳过上传")
            return False

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"[HuggingFaceStorage] 文件不存在: {file_path}")
                return False

            remote_path = remote_path or file_path.name

            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )

            logger.debug(f"[HuggingFaceStorage] 上传成功: {file_path.name} -> {remote_path}")
            return True

        except Exception as e:
            logger.error(f"[HuggingFaceStorage] 上传文件失败 {file_path}: {e}")
            return False

    def download_file(self, remote_path: str, local_path: Optional[str] = None) -> bool:
        """
        从 HuggingFace 下载单个文件

        Args:
            remote_path: 远程文件路径
            local_path: 本地文件路径（默认使用文件名保存在本地目录）

        Returns:
            是否下载成功
        """
        if not self.initialized:
            logger.warning("[HuggingFaceStorage] 未初始化，跳过下载")
            return False

        if not self.repo_id:
            logger.warning("[HuggingFaceStorage] 未配置仓库，跳过下载")
            return False

        try:
            if local_path is None:
                local_path = self.local_dir / Path(remote_path).name
            else:
                local_path = Path(local_path)

            # 确保父目录存在
            local_path.parent.mkdir(parents=True, exist_ok=True)

            hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type="dataset",
                local_dir=str(self.local_dir),
                token=self.token
            )

            # 移动文件到指定位置
            downloaded_file = self.local_dir / remote_path
            if downloaded_file.exists() and str(downloaded_file) != str(local_path):
                local_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_file.rename(local_path)

            logger.debug(f"[HuggingFaceStorage] 下载成功: {remote_path} -> {local_path}")
            return True

        except Exception as e:
            logger.error(f"[HuggingFaceStorage] 下载文件失败 {remote_path}: {e}")
            return False

    def upload_optimizer_state(self, state_file: str = "state.json", 
                               phase_results: Optional[List[str]] = None) -> bool:
        """
        上传优化器状态（包括主状态文件和阶段结果）

        Args:
            state_file: 主状态文件路径（相对本地目录）
            phase_results: 阶段结果文件列表

        Returns:
            是否上传成功
        """
        logger.info(f"[HuggingFaceStorage] 开始上传优化状态...")

        success = True
        uploaded_files = []

        # 上传主状态文件
        state_path = self.local_dir / state_file
        if state_path.exists():
            if self.upload_file(str(state_path), f"optimizer_state/{state_file}"):
                uploaded_files.append(state_file)
        else:
            logger.warning(f"[HuggingFaceStorage] 状态文件不存在: {state_path}")
            success = False

        # 上传阶段结果
        if phase_results:
            for phase_file in phase_results:
                phase_path = self.local_dir / phase_file
                if phase_path.exists():
                    if self.upload_file(str(phase_path), f"optimizer_state/{phase_file}"):
                        uploaded_files.append(phase_file)
                else:
                    logger.warning(f"[HuggingFaceStorage] 阶段文件不存在: {phase_path}")

        # 上传 final_report 和 display_report
        for report_file in ["final_report.json", "display_report.json"]:
            report_path = self.local_dir / report_file
            if report_path.exists():
                if self.upload_file(str(report_path), f"optimizer_state/{report_file}"):
                    uploaded_files.append(report_file)
                    success = True

        if uploaded_files:
            logger.info(f"[HuggingFaceStorage] 上传完成，共 {len(uploaded_files)} 个文件: {uploaded_files}")
        else:
            logger.warning("[HuggingFaceStorage] 没有文件被上传")

        return success

    def download_optimizer_state(self, state_file: str = "state.json",
                                 download_all: bool = True) -> bool:
        """
        下载优化器状态

        Args:
            state_file: 主状态文件名
            download_all: 是否下载所有相关文件

        Returns:
            是否下载成功
        """
        logger.info(f"[HuggingFaceStorage] 开始下载优化状态...")

        success = False

        # 下载主状态文件
        if self.download_file(f"optimizer_state/{state_file}", str(self.local_dir / state_file)):
            success = True
            logger.info(f"[HuggingFaceStorage] 下载主状态: {state_file}")

        if download_all:
            # 尝试下载所有阶段结果
            for i in range(1, 6):
                for phase_name in ["phase1_random", "phase2_tpe", "phase3_cmaes", "phase4_de", "phase5_validation"]:
                    phase_file = f"phase_{phase_name}_results.json"
                    self.download_file(f"optimizer_state/{phase_file}")

            # 下载报告文件
            for report_file in ["final_report.json", "display_report.json"]:
                self.download_file(f"optimizer_state/{report_file}")

        if success:
            logger.info("[HuggingFaceStorage] 下载完成")
        else:
            logger.warning("[HuggingFaceStorage] 没有文件被下载")

        return success

    def check_has_saved_state(self) -> bool:
        """
        检查 HuggingFace 上是否有保存的优化状态

        Returns:
            是否有保存的状态
        """
        if not self.initialized or not self.repo_id:
            return False

        try:
            try:
                repo_info = self.api.repo_info(repo_id=self.repo_id, repo_type="dataset")
                # 检查文件列表中是否有 state.json
                files = [f.path for f in repo_info.siblings if hasattr(repo_info, 'siblings')]
                if any("state.json" in f for f in files):
                    return True
            except Exception:
                # 仓库可能不存在或没有文件
                pass
            return False
        except Exception as e:
            logger.error(f"[HuggingFaceStorage] 检查状态失败: {e}")
            return False

    def get_repo_url(self) -> Optional[str]:
        """
        获取 HuggingFace 仓库 URL

        Returns:
            仓库 URL 字符串
        """
        if not self.repo_id:
            return None
        return f"https://huggingface.co/datasets/{self.repo_id}"

    def upload_final_results(self, results_dir: Path) -> bool:
        """
        上传最终优化结果（用于用户下载）

        Args:
            results_dir: 结果目录路径

        Returns:
            是否上传成功
        """
        logger.info(f"[HuggingFaceStorage] 上传最终结果...")

        success = True
        uploaded_count = 0

        # 上传所有 .json 和 .csv 文件
        for pattern in ["*.json", "*.csv"]:
            for file_path in results_dir.glob(pattern):
                if self.upload_file(str(file_path), f"final_results/{file_path.name}"):
                    uploaded_count += 1

        if uploaded_count > 0:
            logger.info(f"[HuggingFaceStorage] 上传最终结果完成，共 {uploaded_count} 个文件")
        else:
            logger.warning("[HuggingFaceStorage] 没有结果文件被上传")

        return success


def get_huggingface_token() -> Optional[str]:
    """
    获取 HuggingFace Token

    优先级：
    1. environment variable HUGGINGFACE_TOKEN
    2. settings.HUGGINGFACE_TOKEN
    3. .env file

    Returns:
        Token 字符串，如果没有配置则返回 None
    """
    # 1. 环境变量
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        return token

    # 2. settings
    try:
        from config.settings import settings
        token = getattr(settings, 'HUGGINGFACE_TOKEN', None)
        if token:
            return token
    except Exception:
        pass

    # 3. .env 文件
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            token = os.environ.get('HUGGINGFACE_TOKEN', None)
            if token:
                return token
    except Exception:
        pass

    return None


def is_huggingface_configured() -> bool:
    """检查是否已配置 HuggingFace Token"""
    return get_huggingface_token() is not None
