# 在模块导入前设置 sys.path，解决 Streamlit Cloud 多线程导入问题
import sys
from pathlib import Path

# 获取 platform_deployment 目录路径（alert 的父目录的父目录）
current_file = Path(__file__)  # .../platform_deployment/alert/__init__.py
project_root = current_file.parent.parent  # .../platform_deployment

# 设置 Python 路径
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'optimizers'))
sys.path.insert(0, str(project_root / 'utils'))
sys.path.insert(0, str(project_root / 'config'))
sys.path.insert(0, str(project_root / 'alert'))
sys.path.insert(0, str(project_root / 'backtest'))
sys.path.insert(0, str(project_root / 'trading'))

