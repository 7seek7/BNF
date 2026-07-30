# -*- coding: utf-8 -*-
"""
配置适配器 - 新框架兼容旧配置系统

用于逐步迁移，允许新框架访问旧配置系统的参数
"""

from framework.core.config import get_main_config, get_strategy_config, get_config_manager
import os
from typing import Any, Dict, Optional
from framework.core.config import ConfigManager, MainConfig, StrategyConfig


class SettingsAdapter:
    """旧配置系统适配器"""
    
    _main_config = None
    
    @classmethod
    def _get_main_config(cls):
        if cls._main_config is None:
            cls._main_config = get_main_config()
        return cls._main_config
    
    @staticmethod
    def migrate_to_new_config() -> bool:
        """
        将旧配置迁移到新框架
        
        现在直接使用新配置系统
        """
        try:
            # 直接使用新配置系统
            config_manager = get_config_manager()
            
            # 加载主配置
            if not config_manager.load_main_config():
                print("主配置加载失败，请检查.env文件")
                return False
            
            # 加载策略配置
            config_manager.load_strategy_configs()
            
            return True
            
        except Exception as e:
            print(f"配置迁移失败: {e}")
            return False
    
    @staticmethod
    def _populate_main_config(config_manager: ConfigManager):
        """填充主配置（空实现，现在直接从.env加载）"""
        # 不再需要从旧配置系统填充
        # 配置现在直接从.env文件加载
        pass
    
    @staticmethod
    def _migrate_strategy_configs(config_manager: ConfigManager):
        """迁移策略配置（空实现，现在直接从strategies目录加载）"""
        # 策略配置现在直接从strategies/{策略名}/.env加载
        config_manager.load_strategy_configs()
    
    @staticmethod
    def get_legacy_setting(setting_name: str, default: Any = None) -> Any:
        """
        获取旧配置系统参数（兼容性方法）
        
        用于模块逐步迁移时暂时访问旧配置
        注意：旧配置系统已移除，此方法现在返回默认值
        """
        return default
    
    @staticmethod
    def is_legacy_setting_available(setting_name: str) -> bool:
        """检查旧配置参数是否可用（旧配置系统已移除）"""
        return False


def get_settings_adapter() -> SettingsAdapter:
    """获取设置适配器实例"""
    return SettingsAdapter()