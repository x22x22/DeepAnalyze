# DeepAnalyze 训练方法深度分析

## 概述

DeepAnalyze是一个开创性的数据科学自主代理系统，采用了**课程式智能体训练（Curriculum-based Agentic Training）**方法。本文档深入分析该项目在训练集构建和训练代理方面的创新方法。

## 一、训练数据集

### 1.1 数据集规模与来源
- **数据集名称**: DataScience-Instruct-500K
- **数据量**: 约50万条训练样本
- **开源地址**: [HuggingFace Dataset](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)

### 1.2 数据集构成

训练数据分为两个主要阶段：

#### **阶段一：单能力训练数据（Reasoning）**
包含约42万条样本，涵盖：
- SKGInstruct: 199,989 条 - 结构化知识图谱指令
- TableQA_distillation: 39,301 条 - 表格问答蒸馏数据
- TableQA_refinement: 39,301 条 - 表格问答精炼数据
- TableGPT: 29,448 条 - 表格生成与处理
- File类任务:
  - Database: 3,833 条
  - CSV: 3,007 条
  - XLSX: 3,663 条
  - 其他格式: 2,520 条
- 通用能力增强:
  - Math: 20,000 条 - 数学推理
  - Code: 20,000 条 - 代码生成
  - Science: 20,000 条 - 科学推理
  - Instruction following: 20,000 条 - 指令遵循
  - Other: 19,998 条 - 其他任务

#### **阶段二：多能力迭代训练数据（Iteration）**
包含约2.6万条样本，专注于端到端任务：
- Data Pipeline: 3,601 条 - 完整数据处理流程
- Data Preparation: 3,311 条 - 数据准备
- Data Cleaning: 1,616 条 - 数据清洗
- Data Analysis: 3,936 条 - 数据分析
- Data Insight: 1,062 条 - 洞察生成
- Research任务（开放式研究）:
  - Database Research: 818 条
  - XLSX Research: 848 条
  - Other Research: 3,505 条
  - Data Preparation: 488 条
  - Data Analysis: 1,339 条
  - Data Insight: 1,351 条
  - Report Generation: 4,327 条

#### **阶段三：强化学习数据（RL）**
包含三类任务的轨迹数据：
- QA: 问答任务
- DataTask: 数据任务
- Research: 开放式研究任务

### 1.3 数据来源
- [Reasoning-Table](https://github.com/MJinXiang/Reasoning-Table) - 表格推理
- [Spider](https://yale-lily.github.io/spider) - SQL数据集
- [BIRD](https://bird-bench.github.io/) - SQL数据集
- [DABStep](https://huggingface.co/blog/dabstep) - 数据分析基准

## 二、训练方法论

### 2.1 三阶段课程式训练

DeepAnalyze采用渐进式的三阶段训练策略：

#### **第一阶段：单能力精炼（Single-ability Fine-tuning）**

**训练脚本**: `scripts/single.sh`

**目标**: 建立强大的基础能力
- 基础模型: DeepSeek-R1-0528-Qwen3-8B（添加特殊词汇后）
- 训练方式: 全参数微调（Full Fine-tuning）
- 关键参数:
  ```bash
  - 训练轮数: 3 epochs
  - 批次大小: 8 per device
  - 学习率: 5e-5
  - 最大长度: 8,192 tokens
  - 使用数据打包（packing）
  - DeepSpeed ZeRO-3 优化
  - Flash Attention
  - Liger Kernel优化
  ```
- 数据: 使用全部42万条推理数据
- 特点: 强化表格理解、SQL生成、代码执行等基础能力

#### **第二阶段：多能力冷启动（Multi-ability Cold Start）**

**训练脚本**: `scripts/multi_coldstart.sh`

**目标**: 培养端到端的智能体能力
- 基础模型: 第一阶段训练的模型
- 训练方式: 全参数微调
- 关键参数:
  ```bash
  - 训练轮数: 3 epochs
  - 批次大小: 1 per device（更长的上下文）
  - 学习率: 5e-6（降低学习率，更稳定）
  - 最大长度: 32,768 tokens（4倍增长）
  - 梯度累积: 32 steps
  ```
- 数据: 使用2.6万条迭代数据
- 特点: 
  - 学习多轮交互
  - 工具使用能力
  - 长上下文处理
  - 端到端任务完成

#### **第三阶段：强化学习优化（Reinforcement Learning）**

**训练脚本**: `scripts/multi_rl.sh`

**目标**: 通过环境反馈优化策略
- 基础模型: 第二阶段训练的模型
- 训练框架: **SkyRL**（新一代RL训练框架）
- RL算法: **GRPO (Group Relative Policy Optimization)**
- 关键配置:
  ```bash
  - 训练轮数: 1 epoch
  - 训练批次: 256
  - 学习率: 5e-7（进一步降低）
  - 最大输入长度: 32,768 tokens
  - 最大生成长度: 32,768 tokens
  - 每个提示采样: 5 个样本
  - 最大交互轮次: 30 turns
  - 策略: FSDP2（完全分片数据并行）
  ```

### 2.2 强化学习详细分析

#### **环境设计（DeepAnalyzeEnv）**

DeepAnalyze实现了专门的RL环境类，关键特性：

1. **状态空间**:
   - 当前对话历史
   - 工具执行结果
   - 任务指令和数据信息

2. **动作空间**:
   - `<Code>...</Code>`: 执行Python代码
   - `<Analyze>...</Analyze>`: 分析推理
   - `<Answer>...</Answer>`: 最终答案

3. **奖励函数设计**（核心创新点）:

   **a) QA任务奖励**:
   ```python
   - accuracy_reward: 答案准确性（完全匹配）
   - analysis_reward: 分析过程质量（LLM评判，1-5分）
   - 最终奖励 = (准确性 + 分析质量) / 2
   ```

   **b) DataTask任务奖励**:
   ```python
   - code_execution_reward: 代码执行成功率
   - llm_judge_reward: LLM对结果质量的评判
   - 最终奖励 = 各维度平均分
   ```

   **c) OpenResearch任务奖励**（最复杂）:
   ```python
   - usefulness: 洞察有用性（1-5分）
   - richness: 分析丰富度（1-5分）
   - soundness: 逻辑合理性（1-5分）
   - interpretability: 可解释性（1-5分）
   - readability: 可读性（1-5分）
   - code_execution_reward: 代码执行成功率
   - turns_reward: 交互轮次奖励（鼓励充分探索）
   - 最终奖励 = 所有维度平均分
   ```

4. **终止条件**:
   - 达到最大交互轮次（30轮）
   - 输出最终答案标签 `<Answer>...</Answer>`

#### **SkyRL框架特点**

DeepAnalyze使用了**SkyRL**，这是一个专为复杂推理任务设计的RL框架：

1. **优势**:
   - 支持超长上下文（32K tokens）
   - 高效的分布式训练（FSDP2）
   - 异步推理引擎
   - 支持多样本采样（每个提示5个样本）

2. **算法选择 - GRPO**:
   - Group Relative Policy Optimization
   - 相对于传统PPO更稳定
   - 使用相对奖励而非绝对奖励
   - 不使用KL散度损失（`use_kl_loss=false`）

3. **推理后端**:
   - 支持vLLM后端（高效推理）
   - 8个GPU并行推理
   - GPU内存利用率50%（为训练预留空间）

## 三、核心亮点与创新

### 3.1 课程式智能体训练

**创新点**: 三阶段渐进式训练策略
- 阶段1: 夯实基础（单能力）
- 阶段2: 能力整合（多能力）
- 阶段3: 策略优化（强化学习）

**优势**:
1. 避免直接端到端训练的不稳定性
2. 每个阶段有明确的学习目标
3. 逐步增加任务复杂度和上下文长度

### 3.2 混合奖励函数

**创新点**: 结合规则奖励和LLM评判
- 规则奖励: 准确性、代码执行成功率
- LLM评判: 分析质量、可读性、丰富度等主观维度

**优势**:
1. 客观指标确保正确性
2. 主观评判提升报告质量
3. 多维度评估更全面

### 3.3 长上下文处理

**创新点**: 渐进式增加上下文长度
- 阶段1: 8K tokens
- 阶段2-3: 32K tokens

**优势**:
1. 支持复杂的多轮交互
2. 处理大规模数据分析任务
3. 生成长篇研究报告

### 3.4 工具增强学习

**创新点**: 将代码执行作为环境反馈
- Python代码执行器
- SQL执行器
- 真实环境交互

**优势**:
1. 从真实执行结果学习
2. 减少幻觉问题
3. 提升代码质量

### 3.5 开源生态

**创新点**: 完全开源
- 模型权重（DeepAnalyze-8B）
- 训练数据（DataScience-Instruct-500K）
- 训练代码和框架（ms-swift + SkyRL）

## 四、SQL任务训练思路

### 4.1 SQL训练数据

DeepAnalyze在训练中整合了多个SQL数据集：
- **Spider**: 基础Text-to-SQL数据集
- **BIRD**: 复杂数据库推理
- **SKGInstruct**: 包含大量SQL相关任务

### 4.2 SQL任务特殊处理

#### **环境设计**
在`DeepAnalyzeEnv`中实现了SQL执行器：
```python
def execute_sql_single(db_file, sql):
    - 事务管理（BEGIN/ROLLBACK）
    - 超时控制
    - 错误捕获
    - 结果集冻结（frozenset）
```

#### **奖励机制**
1. **执行奖励**: SQL能否成功执行
2. **结果准确性**: 执行结果是否匹配ground truth
3. **推理质量**: 生成SQL的推理过程

### 4.3 SQL训练策略

#### **阶段1: 基础SQL能力**
- 数据: Spider、BIRD等标准数据集
- 目标: 学习正确的SQL语法和语义
- 方法: 监督学习 + 思维链（CoT）

#### **阶段2: 复杂查询与推理**
- 数据: TableQA + SKGInstruct
- 目标: 理解复杂表格关系和多步推理
- 方法: 长上下文训练 + 多轮交互

#### **阶段3: 执行反馈优化**
- 数据: RL阶段的QA任务
- 目标: 从执行结果学习，减少SQL错误
- 方法: 强化学习 + 环境反馈

### 4.4 SQL任务创新点

1. **多模态输入**: 支持数据库、CSV、Excel等多种数据源
2. **可解释性**: 使用`<Analyze>`标签解释SQL生成过程
3. **错误修正**: 通过多轮交互修正SQL错误
4. **环境验证**: 在真实数据库环境中验证SQL

### 4.5 与传统Text-to-SQL的区别

| 维度 | 传统方法 | DeepAnalyze |
|------|---------|------------|
| 训练方式 | 监督学习 | 监督 + RL |
| 评估标准 | 准确率 | 准确率 + 质量 |
| 交互能力 | 单轮 | 多轮迭代 |
| 错误处理 | 无 | 自动修正 |
| 可解释性 | 弱 | 强（推理过程） |

## 五、训练框架

### 5.1 ms-swift
- **用途**: 监督微调（SFT）阶段
- **特点**:
  - 支持多种模型架构
  - 集成DeepSpeed
  - 支持LoRA和全参数训练
  - 高效的数据处理

### 5.2 SkyRL
- **用途**: 强化学习（RL）阶段
- **特点**:
  - 专为长上下文设计
  - 支持复杂的环境交互
  - 高效的分布式训练
  - 灵活的奖励函数设计

## 六、复现指南

### 6.1 环境准备
```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装训练框架
cd ./deepanalyze/ms-swift/ && pip install -e .
cd ./deepanalyze/SkyRL/ && pip install -e .
```

### 6.2 数据准备
```bash
# 下载训练数据
# 从 HuggingFace 下载 DataScience-Instruct-500K
# 解压 RL 数据: unzip DataScience-Instruct-500K/RL/data.zip
```

### 6.3 训练流程
```bash
# 第一阶段: 单能力训练
cd ./deepanalyze/ms-swift/
bash ../../scripts/single.sh

# 第二阶段: 多能力冷启动
bash ../../scripts/multi_coldstart.sh

# 第三阶段: 强化学习
cd ../SkyRL/skyrl-train/
bash ../../scripts/multi_rl.sh
```

## 七、总结

DeepAnalyze在数据科学智能体训练方面的创新包括：

1. **✅ 使用强化学习**: 是的，第三阶段使用GRPO算法
2. **课程式训练**: 三阶段渐进式策略
3. **混合奖励**: 规则 + LLM评判
4. **长上下文**: 支持32K tokens
5. **工具增强**: 真实环境执行反馈
6. **完全开源**: 模型、数据、代码全开源

对于SQL类型的任务，DeepAnalyze采用了**三阶段训练 + 环境反馈 + 多轮迭代**的策略，相比传统Text-to-SQL方法，具有更强的推理能力、错误修正能力和可解释性。

## 八、参考资料

- 论文: [DeepAnalyze: Agentic Large Language Models for Autonomous Data Science](https://arxiv.org/abs/2510.16872)
- 模型: [DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
- 数据集: [DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
- 代码: [GitHub Repository](https://github.com/ruc-datalab/DeepAnalyze)
