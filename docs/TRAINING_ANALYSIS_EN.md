# DeepAnalyze Training Methodology: In-Depth Analysis

## Overview

DeepAnalyze is a pioneering autonomous data science agent system that employs **Curriculum-based Agentic Training** methodology. This document provides an in-depth analysis of the project's innovative approaches to training dataset construction and agent training.

## 1. Training Dataset

### 1.1 Dataset Scale and Sources
- **Dataset Name**: DataScience-Instruct-500K
- **Data Volume**: Approximately 500,000 training samples
- **Open Source**: [HuggingFace Dataset](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)

### 1.2 Dataset Composition

The training data is divided into two main stages:

#### **Stage 1: Single-Ability Training Data (Reasoning)**
Contains approximately 420,000 samples covering:
- SKGInstruct: 199,989 samples - Structured knowledge graph instructions
- TableQA_distillation: 39,301 samples - Table QA distillation data
- TableQA_refinement: 39,301 samples - Table QA refinement data
- TableGPT: 29,448 samples - Table generation and processing
- File-based tasks:
  - Database: 3,833 samples
  - CSV: 3,007 samples
  - XLSX: 3,663 samples
  - Other formats: 2,520 samples
- General capability enhancement:
  - Math: 20,000 samples - Mathematical reasoning
  - Code: 20,000 samples - Code generation
  - Science: 20,000 samples - Scientific reasoning
  - Instruction following: 20,000 samples
  - Other: 19,998 samples

#### **Stage 2: Multi-Ability Iterative Training Data (Iteration)**
Contains approximately 26,000 samples focused on end-to-end tasks:
- Data Pipeline: 3,601 samples - Complete data processing workflows
- Data Preparation: 3,311 samples
- Data Cleaning: 1,616 samples
- Data Analysis: 3,936 samples
- Data Insight: 1,062 samples
- Research tasks (open-ended research):
  - Database Research: 818 samples
  - XLSX Research: 848 samples
  - Other Research: 3,505 samples
  - Data Preparation: 488 samples
  - Data Analysis: 1,339 samples
  - Data Insight: 1,351 samples
  - Report Generation: 4,327 samples

#### **Stage 3: Reinforcement Learning Data (RL)**
Contains trajectory data for three types of tasks:
- QA: Question-answering tasks
- DataTask: Data processing tasks
- Research: Open-ended research tasks

### 1.3 Data Sources
- [Reasoning-Table](https://github.com/MJinXiang/Reasoning-Table) - Table reasoning
- [Spider](https://yale-lily.github.io/spider) - SQL dataset
- [BIRD](https://bird-bench.github.io/) - SQL dataset
- [DABStep](https://huggingface.co/blog/dabstep) - Data analysis benchmark

## 2. Training Methodology

### 2.1 Three-Stage Curriculum Training

DeepAnalyze employs a progressive three-stage training strategy:

#### **Stage 1: Single-Ability Fine-tuning**

**Training Script**: `scripts/single.sh`

**Objective**: Build strong foundational capabilities
- Base Model: DeepSeek-R1-0528-Qwen3-8B (with added special tokens)
- Training Type: Full parameter fine-tuning
- Key Parameters:
  ```bash
  - Training epochs: 3
  - Batch size per device: 8
  - Learning rate: 5e-5
  - Max length: 8,192 tokens
  - Uses packing
  - DeepSpeed ZeRO-3 optimization
  - Flash Attention
  - Liger Kernel optimization
  ```
- Data: All 420K reasoning data samples
- Features: Strengthens table understanding, SQL generation, code execution, and other fundamental capabilities

#### **Stage 2: Multi-Ability Cold Start**

**Training Script**: `scripts/multi_coldstart.sh`

**Objective**: Develop end-to-end agent capabilities
- Base Model: Model from Stage 1
- Training Type: Full parameter fine-tuning
- Key Parameters:
  ```bash
  - Training epochs: 3
  - Batch size per device: 1 (longer context)
  - Learning rate: 5e-6 (reduced for stability)
  - Max length: 32,768 tokens (4x increase)
  - Gradient accumulation: 32 steps
  ```
- Data: 26K iteration data samples
- Features:
  - Multi-turn interaction learning
  - Tool usage capabilities
  - Long context processing
  - End-to-end task completion

#### **Stage 3: Reinforcement Learning Optimization**

**Training Script**: `scripts/multi_rl.sh`

**Objective**: Optimize policy through environmental feedback
- Base Model: Model from Stage 2
- Training Framework: **SkyRL** (next-generation RL training framework)
- RL Algorithm: **GRPO (Group Relative Policy Optimization)**
- Key Configuration:
  ```bash
  - Training epochs: 1
  - Training batch size: 256
  - Learning rate: 5e-7 (further reduced)
  - Max input length: 32,768 tokens
  - Max generation length: 32,768 tokens
  - Samples per prompt: 5
  - Max interaction turns: 30
  - Strategy: FSDP2 (Fully Sharded Data Parallel v2)
  ```

### 2.2 Reinforcement Learning Deep Dive

#### **Environment Design (DeepAnalyzeEnv)**

DeepAnalyze implements a specialized RL environment class with key features:

1. **State Space**:
   - Current conversation history
   - Tool execution results
   - Task instructions and data information

2. **Action Space**:
   - `<Code>...</Code>`: Execute Python code
   - `<Analyze>...</Analyze>`: Analysis and reasoning
   - `<Answer>...</Answer>`: Final answer

3. **Reward Function Design** (Core Innovation):

   **a) QA Task Rewards**:
   ```python
   - accuracy_reward: Answer accuracy (exact match)
   - analysis_reward: Analysis process quality (LLM-judged, 1-5 scale)
   - final_reward = (accuracy + analysis_quality) / 2
   ```

   **b) DataTask Rewards**:
   ```python
   - code_execution_reward: Code execution success rate
   - llm_judge_reward: LLM judgment of result quality
   - final_reward = average of all dimensions
   ```

   **c) OpenResearch Rewards** (Most Complex):
   ```python
   - usefulness: Insight usefulness (1-5 scale)
   - richness: Analysis richness (1-5 scale)
   - soundness: Logical soundness (1-5 scale)
   - interpretability: Interpretability (1-5 scale)
   - readability: Readability (1-5 scale)
   - code_execution_reward: Code execution success rate
   - turns_reward: Interaction turns reward (encourages exploration)
   - final_reward = average of all dimensions
   ```

4. **Termination Conditions**:
   - Reaches maximum interaction turns (30 turns)
   - Outputs final answer tag `<Answer>...</Answer>`

#### **SkyRL Framework Features**

DeepAnalyze uses **SkyRL**, an RL framework designed for complex reasoning tasks:

1. **Advantages**:
   - Supports ultra-long context (32K tokens)
   - Efficient distributed training (FSDP2)
   - Asynchronous inference engine
   - Supports multi-sample sampling (5 samples per prompt)

2. **Algorithm Choice - GRPO**:
   - Group Relative Policy Optimization
   - More stable than traditional PPO
   - Uses relative rewards instead of absolute rewards
   - No KL divergence loss (`use_kl_loss=false`)

3. **Inference Backend**:
   - Supports vLLM backend (efficient inference)
   - 8 GPUs for parallel inference
   - GPU memory utilization 50% (reserves space for training)

## 3. Core Highlights and Innovations

### 3.1 Curriculum-based Agentic Training

**Innovation**: Three-stage progressive training strategy
- Stage 1: Solidify foundations (single abilities)
- Stage 2: Capability integration (multi-abilities)
- Stage 3: Policy optimization (reinforcement learning)

**Advantages**:
1. Avoids instability of direct end-to-end training
2. Each stage has clear learning objectives
3. Gradually increases task complexity and context length

### 3.2 Hybrid Reward Function

**Innovation**: Combines rule-based rewards and LLM judgment
- Rule-based rewards: Accuracy, code execution success rate
- LLM judgment: Analysis quality, readability, richness (subjective dimensions)

**Advantages**:
1. Objective metrics ensure correctness
2. Subjective judgment improves report quality
3. Multi-dimensional assessment is more comprehensive

### 3.3 Long Context Processing

**Innovation**: Progressive increase in context length
- Stage 1: 8K tokens
- Stages 2-3: 32K tokens

**Advantages**:
1. Supports complex multi-turn interactions
2. Handles large-scale data analysis tasks
3. Generates long-form research reports

### 3.4 Tool-Augmented Learning

**Innovation**: Uses code execution as environmental feedback
- Python code executor
- SQL executor
- Real environment interaction

**Advantages**:
1. Learns from real execution results
2. Reduces hallucination problems
3. Improves code quality

### 3.5 Open Source Ecosystem

**Innovation**: Fully open source
- Model weights (DeepAnalyze-8B)
- Training data (DataScience-Instruct-500K)
- Training code and frameworks (ms-swift + SkyRL)

## 4. SQL Task Training Approach

### 4.1 SQL Training Data

DeepAnalyze integrates multiple SQL datasets in training:
- **Spider**: Basic Text-to-SQL dataset
- **BIRD**: Complex database reasoning
- **SKGInstruct**: Contains numerous SQL-related tasks

### 4.2 SQL Task Special Processing

#### **Environment Design**
SQL executor implemented in `DeepAnalyzeEnv`:
```python
def execute_sql_single(db_file, sql):
    - Transaction management (BEGIN/ROLLBACK)
    - Timeout control
    - Error capture
    - Result set freezing (frozenset)
```

#### **Reward Mechanism**
1. **Execution Reward**: Whether SQL executes successfully
2. **Result Accuracy**: Whether execution results match ground truth
3. **Reasoning Quality**: Reasoning process of generated SQL

### 4.3 SQL Training Strategy

#### **Stage 1: Basic SQL Capabilities**
- Data: Spider, BIRD, and other standard datasets
- Objective: Learn correct SQL syntax and semantics
- Method: Supervised learning + Chain-of-Thought (CoT)

#### **Stage 2: Complex Queries and Reasoning**
- Data: TableQA + SKGInstruct
- Objective: Understand complex table relationships and multi-step reasoning
- Method: Long context training + multi-turn interaction

#### **Stage 3: Execution Feedback Optimization**
- Data: QA tasks in RL stage
- Objective: Learn from execution results, reduce SQL errors
- Method: Reinforcement learning + environmental feedback

### 4.4 SQL Task Innovations

1. **Multi-modal Input**: Supports multiple data sources (databases, CSV, Excel, etc.)
2. **Explainability**: Uses `<Analyze>` tags to explain SQL generation process
3. **Error Correction**: Corrects SQL errors through multi-turn interaction
4. **Environment Validation**: Validates SQL in real database environments

### 4.5 Differences from Traditional Text-to-SQL

| Dimension | Traditional Approach | DeepAnalyze |
|-----------|---------------------|-------------|
| Training Method | Supervised Learning | Supervised + RL |
| Evaluation Criteria | Accuracy | Accuracy + Quality |
| Interaction Capability | Single-turn | Multi-turn iterative |
| Error Handling | None | Automatic correction |
| Explainability | Weak | Strong (reasoning process) |

## 5. Training Frameworks

### 5.1 ms-swift
- **Purpose**: Supervised Fine-Tuning (SFT) stage
- **Features**:
  - Supports multiple model architectures
  - Integrates DeepSpeed
  - Supports LoRA and full parameter training
  - Efficient data processing

### 5.2 SkyRL
- **Purpose**: Reinforcement Learning (RL) stage
- **Features**:
  - Designed for long context
  - Supports complex environment interactions
  - Efficient distributed training
  - Flexible reward function design

## 6. Reproduction Guide

### 6.1 Environment Setup
```bash
# Install basic dependencies
pip install -r requirements.txt

# Install training frameworks
cd ./deepanalyze/ms-swift/ && pip install -e .
cd ./deepanalyze/SkyRL/ && pip install -e .
```

### 6.2 Data Preparation
```bash
# Download training data
# Download DataScience-Instruct-500K from HuggingFace
# Unzip RL data: unzip DataScience-Instruct-500K/RL/data.zip
```

### 6.3 Training Process
```bash
# Stage 1: Single-ability training
cd ./deepanalyze/ms-swift/
bash ../../scripts/single.sh

# Stage 2: Multi-ability cold start
bash ../../scripts/multi_coldstart.sh

# Stage 3: Reinforcement learning
cd ../SkyRL/skyrl-train/
bash ../../scripts/multi_rl.sh
```

## 7. Summary

DeepAnalyze's innovations in data science agent training include:

1. **✅ Uses Reinforcement Learning**: Yes, Stage 3 uses GRPO algorithm
2. **Curriculum Training**: Three-stage progressive strategy
3. **Hybrid Rewards**: Rule-based + LLM judgment
4. **Long Context**: Supports 32K tokens
5. **Tool Augmentation**: Real environment execution feedback
6. **Fully Open Source**: Model, data, and code all open source

For SQL-type tasks, DeepAnalyze employs a **three-stage training + environmental feedback + multi-turn iteration** strategy. Compared to traditional Text-to-SQL methods, it has stronger reasoning capabilities, error correction abilities, and explainability.

## 8. References

- Paper: [DeepAnalyze: Agentic Large Language Models for Autonomous Data Science](https://arxiv.org/abs/2510.16872)
- Model: [DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
- Dataset: [DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
- Code: [GitHub Repository](https://github.com/ruc-datalab/DeepAnalyze)
