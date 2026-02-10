---
title: AI参数优化系统
emoji: 🤖
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: "1.28.0"
python_version: "3.10"
app_file: streamlit_app.py
pinned: false
---

# 🤖 AI全自动化参数优化系统 v2.0

基于AI的币安期货交易机器人参数优化系统，支持智能规划、自动下载、回测分析。

## ✨ 核心特性

### 🎯 AI全自主模式（推荐）
- **智能规划**: AI自动选择币种、参数、优化策略
- **智能Step值**: 用户只需提供start/stop，AI自动确定最优step值
- **平衡速度与精度**: AI智能控制组合数量，平衡测试时间和准确性
- **全自动下载**: AI自动从币安下载高波动币种历史数据
- **多AI支持**: NVIDIA NIM、OpenAI、通义千问、DeepSeek、自定义API

### ⚙️ 手动配置模式
- **完全控制**: 自己配置所有参数的start/stop/step
- **全面测试**: 支持测试所有参数组合
- **适合专家**: 满足有经验用户的细粒度需求

## 📋 支持的参数（30+）

### 警报参数
- `PRICE_CHANGE_THRESHOLD` - 价格变化阈值(%)
- `VOLUME_THRESHOLD` - 成交量阈值(倍)
- `MONITOR_INTERVAL` - 监控周期(分钟)
- `ALERT_COOLDOWN` - 警报冷却时间(分钟)
- `VOLUME_COMPARE_PERIODS` - 成交量比较周期数

### 交易参数
- `LEVERAGE` - 杠杆倍数
- `INITIAL_POSITION` - 初始仓位比例(%)
- `DELAY_RATIO` - 下单延迟系数

### 亏损加仓（6参数）
- `LOSS_STEP1/2/3` - 亏损加仓阈值
- `LOSS_ADD1/2/3` - 亏损加仓额度

### 盈利加仓（6参数）
- `PROFIT_STEP1/2/3` - 盈利加仓阈值
- `PROFIT_ADD1/2/3` - 盈利加仓额度

### 高利润止盈（5参数）
- `HIGH_PROFIT_THRESHOLD` - 高利润触发止盈(%)
- `HIGH_PROFIT_DRAWBACK1/2` - 回撤阈值
- `HIGH_PROFIT_CLOSE1/2` - 止盈比例

### 低利润止盈（4参数）
- `LOW_PROFIT_THRESHOLD` - 低利润触发止盈(%)
- `LOW_PROFIT_DRAWBACK1` - 回撤阈值
- `LOW_PROFIT_CLOSE1` - 止盈比例
- `BREAKEVEN_THRESHOLD` - 保本阈值

### 止损参数（5参数）
- `STOPLOSS_TRIGGER1/2/3` - 止损触发点
- `STOPLOSS_CLOSE1/2` - 平仓比例

## 🚀 快速开始

### 方法1: 本地运行Streamlit应用

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的AI API密钥

# 3. 运行应用
streamlit run streamlit_app.py
```

### 方法2: Hugging Face Spaces部署

应用已部署在Hugging Face: https://huggingface.co/spaces/noonese7en/trading-bot-optimizer

## 🔑 配置AI API

### NVIDIA NIM (推荐)
- API Base: `https://integrate.api.nvidia.com/v1`
- 推荐模型:
  - `nvidia/nemotron-4-340b-instruct` (默认)
  - `z-ai/glm4-7` (中文优化)
  - `minimaxai/minimax-m2` (中文优化)
- 获取API Key: https://build.nvidia.com/

### OpenAI
- API Base: `https://api.openai.com/v1`
- 推荐模型: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`

### 通义千问
- API Base: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 推荐模型: `qwen-max`, `qwen-plus`, `qwen-turbo`

### DeepSeek
- API Base: `https://api.deepseek.com/v1`
- 推荐模型: `deepseek-chat`, `deepseek-reasoner`

## 📖 使用指南

### AI全自主模式

1. **提供参数范围**: 为想要优化的参数设置最小值(Start)和最大值(Stop)
2. **AI智能规划**: AI自动决定:
   - 测试哪些币种
   - 优化哪些参数（如果参数过多）
   - 最优step值以控制组合数
   - 最佳优化策略
3. **查看AI计划**: 查看AI选择的币种、参数、预估时间
4. **开始优化**: 点击启动，AI自动执行
5. **分析结果**: 查看最佳参数配置，下载JSON/CSV数据

### 手动配置模式

1. **配置所有参数**: 为每个参数设置start/stop/step
2. **评估组合数**: 注意控制组合数，避免太多
3. **开始优化**: 运行完整的参数网格搜索
4. **分析结果**: 查看回测结果，识别最佳配置

## 📁 项目结构

```
platform_deployment/
├── streamlit_app.py           # Streamlit Web应用
├── auto_ai_optimizer.py       # AI自主优化核心
├── optimizer.py               # 参数优化引擎
├── parameter_grid.py          # 参数组合生成器
├── result_analyzer.py         # 结果分析器
├── analyze_failures.py        # 失败案例分析
├── ai_analyze.py              # AI分析模块
├── ai_loop.py                 # AI迭代逻辑
├── backtest/
│   ├── data_downloader.py     # 币安数据下载器
│   └── unified_backtest.py    # 统一回测引擎
├── config/
│   └── settings.py            # 参数配置（所有参数默认值）
├── utils/
│   ├── helpers.py             # 辅助工具
│   └── logger.py              # 日志模块
├── data/
│   ├── historical/            # 历史数据存储
│   └── logs/                  # 日志文件
├── optimizer_results/         # 优化结果输出
├── requirements.txt           # Python依赖
├── .env.example               # 环境变量模板
└── README.md                  # 本文档
```

## 🎨 AI优化流程

```
用户输入参数范围 (start/stop)
         ↓
1. AI规划优化策略
   - 选择币种（从币安获取高波动币种）
   - 选择参数（如果参数过多）
   - 确定优化方式
         ↓
2. AI智能确定Step值
   - 分析参数重要性
   - 计算最优step以控制组合数
   - 平衡重要参数的精度
         ↓
3. 下载数据
   - 自动从币安下载历史数据
   - 支持多个币种并行下载
         ↓
4. 执行回测优化
   - 生成参数组合
   - 批量回测
   - 保存结果
         ↓
5. 结果分析
   - 识别最佳参数组合
   - 生成统计报告
   - 导出JSON/CSV
```

## 📊 输出文件

优化完成后，以下文件会生成在 `optimizer_results/` 目录：

- `auto_results_YYYYMMDD_HHMMSS.json` - 完整优化结果
- `auto_results_YYYYMMDD_HHMMSS.csv` - 回测数据CSV
- `auto_results_YYYYMMDD_HHMMSS_report.json` - AI优化报告
- `best_config.json` - 最佳参数配置

## 🔧 常见问题

### Q: AI模式 vs 手动模式的区别？

**A**:
- **AI模式**: 用户只提供start/stop，AI自动确定step值、选择币种、选择参数子集。适合大多数用户，自动平衡速度和精度。
- **手动模式**: 用户完全控制start/stop/step，测试所有参数组合。适合有经验的用户，但可能产生极多组合（需要谨慎）。

### Q: 如何控制优化时间？

**A**:
1. **减少参数数量**: 只优化最重要的参数（如：PRICE_CHANGE_THRESHOLD, STOPLOSS_TRIGGER1, LEVERAGE等）
2. **调整最大组合数限制**: 在界面上设置"最大组合数限制"，AI会调整step值来满足
3. **使用AI模式**: AI会智能规划，自动平衡速度和精度

### Q: 需要币安实盘API吗？

**A**: 不需要。系统只需要币安API来下载历史数据用于回测。建议使用测试网API，不影响真实资金。

### Q: 数据会自动下载吗？

**A**: 是的。AI模式会自动检查并下载必要的数据。如果本地没有数据或数据过期，AI会自动从币安下载高波动币种的历史数据。

### Q: 支持哪些AI模型？

**A**: 支持任何OpenAI兼容的API：
- NVIDIA NIM (推荐) - 免费，性能强大
- OpenAI - GPT-4, GPT-3.5-turbo
- 通义千问 - qwen-max, qwen-plus
- DeepSeek - deepseek-chat
- 其他OpenAI兼容的API

## 🛠️ 开发说明

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env填入API密钥

# 运行应用
streamlit run streamlit_app.py

# 或运行命令行版本
python auto_ai_optimizer.py \
    --apikey YOUR_API_KEY \
    --base https://api.openai.com/v1 \
    --model gpt-4
```

### 添加新的参数

1. 在 `config/settings.py` 中配置默认值
2. 在 `auto_ai_optimizer.py` 的 `PARAMETER_GROUPS` 中添加参数定义
3. 在 `streamlit_app.py` 的 `PARAMETER_GROUPS` 中添加UI配置

## 📝 更新日志

### v2.0 (2026-02-02)
- ✨ 新增AI全自主模式
- ✨ AI智能确定step值
- ✨ AI自动规划优化策略（币种选择、参数选择）
- ✨ AI自动下载币安数据
- ✨ 支持多AI服务商（NVIDIA, OpenAI, 通义千问, DeepSeek）
- ✨ 新增结果导出（JSON + CSV）
- ✨ 优化UI体验，更直观的配置界面

### v1.0
- 基础参数优化功能
- 手动配置模式
- 回测分析
- 基本UI

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系

如有问题，请提交Issue或联系项目维护者。
