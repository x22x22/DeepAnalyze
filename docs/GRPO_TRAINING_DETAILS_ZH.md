# GRPO训练详细说明：训练集与奖励函数

## 概述

本文档详细说明DeepAnalyze使用SkyRL框架进行GRPO（Group Relative Policy Optimization）训练时的训练集构成和奖励函数设计，特别关注开放式答案场景下的奖励判断机制。

## 一、GRPO训练数据集

### 1.1 数据格式

GRPO训练使用三个parquet文件，位于`DataScience-Instruct-500K/RL/`目录：

> **注意**: 文件名`reseach.parquet`是数据集中的实际文件名（可能是原始数据中的拼写错误）。

```
RL/
├── qa.parquet              # 问答任务
├── datatask.parquet        # 数据处理任务
└── reseach.parquet         # 开放式研究任务（实际文件名）
```

### 1.2 数据样本结构

每个parquet文件包含以下字段：

```python
{
    "input_seq": str,           # 输入提示（包含任务描述和数据信息）
    "workspace_id": str,        # 工作空间ID（数据文件所在目录）
    "task": str,                # 任务类型标识（qa/datatask/research）
    "reward_spec": {            # 奖励规格
        "method": str,          # 奖励方法：qa/datatask/openresearch
        "ground_truth": str,    # 参考答案或数据缩略图
        "function": str/list    # 奖励函数名称
    },
    "max_turns": int            # 最大交互轮次（默认30）
}
```

### 1.3 三种任务类型详解

#### 任务1：QA（问答任务）

**特点**: 有明确的正确答案

**数据样本示例**:
```python
{
    "input_seq": "# Instruction\nAnalyze the table and answer: What is the average age?\n\n# Data\nFile: students.csv",
    "workspace_id": "qa_001",
    "task": "qa",
    "reward_spec": {
        "method": "qa",
        "ground_truth": "25.5",  # 标准答案
        "function": None
    },
    "max_turns": 30
}
```

**ground_truth格式**: 精确的答案字符串

#### 任务2：DataTask（数据处理任务）

**特点**: 需要执行代码完成特定数据处理

**数据样本示例**:
```python
{
    "input_seq": "# Instruction\nClean the data by removing outliers and fill missing values.\n\n# Data\nFile: sales_data.csv",
    "workspace_id": "datatask_042",
    "task": "datatask",
    "reward_spec": {
        "method": "datatask",
        "ground_truth": "Expected output description or statistics",
        "function": ["llm_as_judgement_accuracy", "llm_as_judgement_analyze"]
    },
    "max_turns": 30
}
```

**ground_truth格式**: 期望输出的描述或统计信息

#### 任务3：OpenResearch（开放式研究任务）

**特点**: 没有标准答案，需要生成完整的研究报告

**数据样本示例**:
```python
{
    "input_seq": "# Instruction\nGenerate a comprehensive data science report analyzing the enrollment patterns.\n\n# Data\nFile 1: enrollment.csv\nFile 2: demographics.xlsx",
    "workspace_id": "research_128",
    "task": "openresearch",
    "reward_spec": {
        "method": "openresearch",
        "ground_truth": "{'total_records': 1194, 'columns': ['student_id', 'enrollment_date', ...], ...}",  # 数据缩略图
        "function": "llm_as_judgement_opendomain"
    },
    "max_turns": 30
}
```

**ground_truth格式**: 数据缩略图（数据的元信息，如行数、列名、字段统计等）

### 1.4 数据量统计

根据训练脚本配置：
- **QA任务**: 约占30%的训练数据
- **DataTask任务**: 约占40%的训练数据
- **OpenResearch任务**: 约占30%的训练数据

## 二、GRPO算法配置

### 2.1 核心参数

```bash
# GRPO算法配置
trainer.algorithm.advantage_estimator="grpo"    # 使用GRPO而非PPO
trainer.algorithm.use_kl_loss=false             # 不使用KL散度损失
trainer.train_batch_size=256                    # PPO训练批次大小
trainer.policy_mini_batch_size=256              # 策略更新小批次
generator.n_samples_per_prompt=5                # 每个提示生成5个样本进行对比
```

### 2.2 GRPO vs PPO

**传统PPO**:
- 使用绝对奖励值
- 需要价值网络估计优势函数
- 使用KL散度约束策略更新

**GRPO（Group Relative Policy Optimization）**:
- 使用**相对奖励**：每个提示生成5个样本，按奖励排序后计算相对优势
- 不需要价值网络（更简单、更稳定）
- 通过组内对比学习最优策略
- 特别适合**开放式任务**，因为绝对奖励难以定义

### 2.3 GRPO奖励计算流程

```
1. 对于每个训练样本：
   - 生成5个不同的响应（temperature=0但有采样）
   
2. 环境交互：
   - 每个响应在DeepAnalyzeEnv中执行
   - 记录完整的交互轨迹（最多30轮）
   
3. 奖励计算：
   - 任务完成后，调用对应的奖励函数
   - 得到5个奖励值：[r1, r2, r3, r4, r5]
   
4. 相对优势计算：
   - 对奖励排序：r_sorted
   - 计算相对位置：rank_i / 5
   - 优势 = rank_i - mean_rank
   
5. 策略更新：
   - 使用相对优势更新策略网络
   - 鼓励高奖励的行为，抑制低奖励的行为
```

## 三、奖励函数设计详解

### 3.1 QA任务奖励函数

#### 3.1.1 准确性奖励（Accuracy Reward）

**函数**: `compute_tableqa_score_single(completion, reference)`

**计算逻辑**:
```python
def compute_tableqa_score_single(completion, reference):
    """
    精确匹配奖励
    """
    # 1. 从completion中提取<Answer>标签内的答案
    answer = extract_tableqa_answer(completion)
    
    # 2. 与标准答案完全匹配
    if answer == reference:
        return 1.0  # 完全正确
    else:
        return 0.0  # 不正确
```

**关键点**:
- 二值奖励：1.0或0.0
- 必须完全匹配，对数值答案有一定的容差处理

#### 3.1.2 分析质量奖励（Analysis Reward）

**函数**: `llm_as_judgement_analyze(completion, reference, question, client, model)`

**评判维度**: 分析过程的质量（1-5分）

**评判提示词**:
```
评估标准（1-5分）：
- 1分（Adequate）：基本推理，部分正确但深度不足
- 2分（Moderate）：可理解且有一定逻辑，但不完整
- 3分（Good）：合理正确，覆盖主要点
- 4分（Strong）：清晰、结构化、逻辑一致
- 5分（Exceptional）：完全正确、严谨、全面

输出格式：SCORE: <1-5>
```

**计算逻辑**:
```python
def llm_as_judgement_analyze(completion, reference, question, client, model):
    """
    使用外部LLM评判分析质量
    """
    # 1. 构造评判提示
    prompt = ANALYZE_PROMPT.format(
        question=question,
        ref=reference,
        answer=completion
    )
    
    # 2. 调用评判LLM（如GPT-4o）
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 3. 解析得分并归一化到[0, 1]
    score = extract_score(response)  # 1-5分
    return score / 5.0  # 归一化到0-1
```

#### 3.1.3 QA任务最终奖励

```python
# 两个奖励的平均值
qa_reward = (accuracy_reward + analysis_reward) / 2
```

### 3.2 DataTask任务奖励函数

#### 3.2.1 代码执行奖励（Code Execution Reward）

**计算逻辑**:
```python
# 在每次代码执行后记录
if "[Error]:" in observation:
    code_pass.append(0)  # 执行失败
else:
    code_pass.append(1)  # 执行成功

# 任务结束时计算
code_execution_reward = sum(code_pass) / len(code_pass)
```

**关键点**:
- 衡量代码质量：成功执行的代码块比例
- 鼓励编写可执行的正确代码

#### 3.2.2 结果准确性奖励（Accuracy Reward）

**函数**: `llm_as_judgement_accuracy(completion, reference, question, client, model)`

**评判维度**: 结果的准确性（0-5分，归一化到0-1）

**评判提示词**:
```
任务：评估模型响应的准确性
给定：
- [QUESTION]: 数据处理任务描述
- [REFERENCE SOLUTION]: 参考解决方案
- [PREDICTED SOLUTION]: 模型的输出

评分标准（0-5分）：
- 0分：完全错误
- 1-2分：部分正确，但有重大错误
- 3分：基本正确，有一些小错误
- 4分：正确，仅有微小瑕疵
- 5分：完全准确

输出：SCORE: <0-5>
```

#### 3.2.3 分析质量奖励

与QA任务相同，使用`llm_as_judgement_analyze`评判推理过程。

#### 3.2.4 DataTask最终奖励

```python
datatask_reward = (code_execution_reward + accuracy_reward + analysis_reward) / 3
```

### 3.3 OpenResearch任务奖励函数（重点）

这是**最复杂也最创新**的奖励函数设计，专门针对开放式研究报告生成。

#### 3.3.1 多维度评估框架

**函数**: `llm_as_judgement_opendomain(completion, reference, question, client, model)`

**评判维度**: 5个独立维度，每个1-5分

**详细评判标准**:

##### 维度1: Usefulness（有用性）
```
评估：报告是否成功提取和突出数据中的关键洞察？

- 1分：提供少量基本洞察，遗漏重要方面
- 2分：一些有用洞察，但常常肤浅或不完整
- 3分：合理有用，捕获主要点但有明显空白
- 4分：强有力且清晰的洞察，仅有微小遗漏
- 5分：洞察卓越、全面，精确契合指令
```

##### 维度2: Richness（丰富性）
```
评估：分析是否从多样且有意义的角度探索数据？

- 1分：覆盖最少的角度，深度有限
- 2分：有一些探索但过于狭窄或浅显
- 3分：适度探索多个角度，但深度不均
- 4分：丰富的分析，广泛且有合理深度
- 5分：异常丰富，从多个有意义的角度进行广泛而深刻的探索
```

##### 维度3: Soundness（合理性）
```
评估：分析过程是否结构良好、准确且逻辑规划？

- 1分：适当但包含一些缺陷或不清晰的逻辑
- 2分：总体正确但有明显的弱点或错误
- 3分：大部分合理，结构连贯，但不完全严谨
- 4分：强逻辑流程和准确性，仅有小问题
- 5分：完全严谨、精确且完全连贯的分析过程
```

##### 维度4: Interpretability（可解释性）
```
评估：报告是否提供充分的中间输出使推理过程透明？

- 1分：最小可解释性，推理部分可理解
- 2分：提供一些解释但关键步骤不清楚
- 3分：合理可解释，展示了若干中间步骤
- 4分：清晰且有充分支持的推理，良好的逐步证据
- 5分：完全透明且高度可读的推理，完整且结构良好的中间输出
```

##### 维度5: Readability（可读性）
```
评估：最终报告是否以精美的学术风格呈现并满足指令？

- 1分：结构适当，但缺乏精美或学术语气
- 2分：大部分可理解但风格不一致或与指令弱对齐
- 3分：清晰可接受的结构，部分学术但不完全精美
- 4分：学术风格良好，满足指令，仅有小问题
- 5分：异常精美的专业学术风格，完全契合且高效
```

#### 3.3.2 LLM评判提示词

完整的评判提示词：

```python
OPENDOMAIN_PROMPT = """You are a data science evaluation assistant. Here's a generated data science report based on the user instruction and provided data. Your task is to comprehensively evaluate the quality of a generated data science report and its analytical process, based on the provided user instruction [INSTRUCTION], the data thumbnail [DATA THUMBNAIL], and the generated report [GENERATED REPORT].

You should assess the report across the following five dimensions, each scored on a scale from 1 (lowest) to 5 (highest). Please use the detailed guidelines below to calibrate your evaluation:

- **Usefulness**: Does the report successfully extract and highlight key insights from the data?
    - **1**: Provides a few relevant insights at a basic level, but misses many important aspects.
    - **2**: Some useful insights, but often superficial or incomplete.
    - **3**: Reasonably useful, capturing the main points with noticeable gaps.
    - **4**: Strong and clear insights with only minor omissions.
    - **5**: Outstandingly insightful, comprehensive, and precisely aligned with the instruction.

- **Richness**: Does the analysis explore the data from diverse and meaningful perspectives?
    - **1**: Covers a minimal range of perspectives with limited depth.
    - **2**: Some exploration but overly narrow or shallow.
    - **3**: Moderate exploration with several perspectives, though uneven in depth.
    - **4**: Rich analysis with broad and reasonably deep perspectives.
    - **5**: Exceptionally rich, broad, and profound exploration from multiple meaningful angles.

- **Soundness**: Is the analytical process well-structured, accurate, and logically planned?
    - **1**: Adequate but contains some flaws or unclear logic.
    - **2**: Generally correct but with noticeable weaknesses or errors.
    - **3**: Mostly sound with a coherent structure, though not fully rigorous.
    - **4**: Strong logical flow and accuracy, with only minor issues.
    - **5**: Perfectly rigorous, precise, and fully coherent analytical process.

- **Interpretability**: Does the report provide sufficient intermediate outputs to make the reasoning process transparent?
    - **1**: Minimal interpretability; reasoning is partially understandable.
    - **2**: Some explanation provided but key steps remain unclear.
    - **3**: Reasonably interpretable with several intermediate steps shown.
    - **4**: Clear and well-supported reasoning with good step-by-step evidence.
    - **5**: Fully transparent and highly readable reasoning, with complete and well-structured intermediate outputs.

- **Readability**: Is the final report presented in a polished academic style and does it fulfill the instruction?
    - **1**: Adequately structured, but lacks polish or academic tone.
    - **2**: Mostly understandable but inconsistent in style or weakly aligned with the instruction.
    - **3**: Clear and acceptable structure, partially academic but not fully polished.
    - **4**: Well-written in an academic style, fulfilling the instruction with only minor issues.
    - **5**: Exceptionally polished, professional academic style, fully aligned and highly effective.

### [INSTRUCTION]:
{instruction} 

### [DATA THUMBNAIL]:
{thumbnail}

### [PREDICTED SOLUTION]:
{answer}

First find some weaknesses, and finally return your evaluation in the following JSON format:
    ```json
    {{
    "usefulness": <score from 1 to 5>,
    "richness": <score from 1 to 5>,
    "soundness": <score from 1 to 5>,
    "interpretability": <score from 1 to 5>,
    "readability": <score from 1 to 5>,
    }}
    ```
"""
```

#### 3.3.3 奖励计算实现

```python
def llm_as_judgement_opendomain(completion, reference, question, client, model):
    """
    使用外部LLM对开放式研究报告进行5维度评判
    """
    # 1. 构造评判提示
    message = OPENDOMAIN_PROMPT.format(
        instruction=question,      # 用户的研究指令
        thumbnail=reference,       # 数据缩略图（元信息）
        answer=completion         # 模型生成的完整报告
    )
    
    # 2. 调用评判LLM（如GPT-4o）
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}]
    )
    reply = response.choices[0].message.content.strip()
    
    # 3. 解析JSON格式的评分
    reply_match = re.search(r"```(?:json)?(.*?)```", reply, re.DOTALL)
    score_str = reply_match.group(1).strip()
    score_dict = json.loads(score_str)
    
    # 4. 归一化每个维度到[0, 1]
    reward = {
        key: float(score_dict[key]) / 5
        for key in [
            "usefulness",
            "richness",
            "soundness",
            "interpretability",
            "readability",
        ]
    }
    
    return reward  # 返回字典
```

#### 3.3.4 额外奖励：交互轮次奖励

对于开放式研究，还有一个**交互轮次奖励**：

```python
# 统计agent的交互轮次
turns = count_assistant_turns(chat_history)

# 计算轮次奖励（鼓励充分探索）
turn_reward = min(turns / 10, 1.0)
```

**设计理念**:
- 鼓励模型进行充分的探索和分析
- 但最多到10轮就饱和（避免过度冗长）
- 与质量评分平衡

#### 3.3.5 OpenResearch最终奖励

```python
openresearch_reward = (
    usefulness_reward +
    richness_reward +
    soundness_reward +
    interpretability_reward +
    readability_reward +
    code_execution_reward +
    turns_reward
) / 7  # 7个维度的平均值
```

## 四、开放式答案的奖励判断机制

### 4.1 为什么需要LLM-as-a-Judge？

**挑战**:
- 开放式研究报告没有标准答案
- 传统指标（BLEU、ROUGE）无法衡量报告质量
- 人工评估成本高且不可扩展

**解决方案**:
- 使用强大的LLM（如GPT-4o）作为评判者
- 设计详细的评分标准和提示词
- 多维度评估确保全面性

### 4.2 LLM评判的可靠性

**可靠性保证措施**:

1. **详细的评分标准**: 每个维度都有1-5分的详细描述
2. **结构化输出**: 要求JSON格式，便于解析
3. **多维度评估**: 5个独立维度，降低单一维度的偏差
4. **标准化**: 所有奖励归一化到[0, 1]区间
5. **稳定的评判模型**: 使用性能稳定的GPT-4o等模型

**评判一致性**:
- 研究表明，GPT-4在结构化评判任务上与人类评判的一致性达到80%+
- 通过详细的rubric（评分准则），进一步提高一致性

### 4.3 评判LLM的配置

在`multi_rl.sh`中配置评判LLM：

```python
# DeepAnalyzeEnv初始化时配置
env_config = {
    "api_key": "YOUR_API_KEY",
    "base_url": "https://api.openai.com/v1",
    "llm_judgement_model": "gpt-4o"  # 推荐使用GPT-4o
}
```

**推荐模型**:
- GPT-4o: 最佳性能，成本较高
- GPT-4o-mini: 性能良好，成本适中
- Claude-3.5-Sonnet: 另一选择

## 五、GRPO训练流程示例

### 5.1 单个训练步骤

```
输入: 开放式研究任务
"生成一份关于学生贷款数据的综合分析报告"

↓

Step 1: 采样5个响应
- Response 1: Agent执行30轮交互，生成报告A
- Response 2: Agent执行25轮交互，生成报告B
- Response 3: Agent执行28轮交互，生成报告C
- Response 4: Agent执行20轮交互，生成报告D
- Response 5: Agent执行32轮交互，生成报告E

↓

Step 2: 环境交互与奖励计算
- 报告A → LLM评判 → {usefulness: 4/5, richness: 3/5, ...} → reward_A = 0.78
- 报告B → LLM评判 → {usefulness: 3/5, richness: 4/5, ...} → reward_B = 0.72
- 报告C → LLM评判 → {usefulness: 5/5, richness: 5/5, ...} → reward_C = 0.92
- 报告D → LLM评判 → {usefulness: 2/5, richness: 2/5, ...} → reward_D = 0.45
- 报告E → LLM评判 → {usefulness: 4/5, richness: 4/5, ...} → reward_E = 0.83

↓

Step 3: GRPO优势计算
- 排序: [D(0.45), B(0.72), A(0.78), E(0.83), C(0.92)]
- 相对优势:
  - Response D: advantage = -0.4 (worst)
  - Response B: advantage = -0.2
  - Response A: advantage = 0.0
  - Response E: advantage = 0.2
  - Response C: advantage = 0.4 (best)

↓

Step 4: 策略更新
- 增加Response C和E的生成概率
- 降低Response D和B的生成概率
- Response A保持中性
```

### 5.2 为什么GRPO适合开放式任务？

1. **相对比较更可靠**: 
   - 判断"报告A比报告B好"比"报告A得85分"更容易
   - 相对排序减少了评判的绝对偏差

2. **无需价值网络**:
   - 传统PPO需要训练一个价值网络来估计状态价值
   - GRPO通过组内对比直接计算优势
   - 训练更稳定，收敛更快

3. **鼓励多样性**:
   - 每个提示生成5个样本
   - 学习到不同的好策略
   - 避免模式崩塌

4. **适应模糊目标**:
   - 开放式任务的"好"有多种形式
   - GRPO通过对比学习捕获这些多样性

## 六、奖励函数设计的关键洞察

### 6.1 混合奖励的重要性

**为什么不只用LLM评判？**

DeepAnalyze结合了三类奖励：

1. **规则奖励**（客观）:
   - 准确性匹配
   - 代码执行成功率
   - 格式正确性检查

2. **LLM评判**（主观质量）:
   - 分析深度
   - 洞察质量
   - 可读性

3. **过程奖励**（探索鼓励）:
   - 交互轮次奖励
   - 代码尝试次数

**优势**:
- 客观奖励确保基本正确性
- 主观奖励提升报告质量
- 过程奖励鼓励充分探索

### 6.2 归一化的重要性

所有奖励都归一化到[0, 1]区间：

```python
# 准确性：0 or 1
# LLM评判：score / 5
# 代码执行：sum / len
# 交互轮次：min(turns / 10, 1)
```

**原因**:
- 不同维度的奖励可比较
- 避免某个维度主导
- 梯度更新更稳定

### 6.3 失败情况的处理

**格式错误 → -1.0奖励**:
```python
if "<Analyze>" not in completion or "<Answer>" not in completion:
    return -1.0
```

**代码错误 → 降低代码奖励**:
```python
if "[Error]:" in observation:
    code_pass.append(0)
```

**LLM评判失败 → 0.0奖励**:
```python
try:
    reward = llm_judge(...)
except Exception:
    return 0.0
```

## 七、评判LLM的成本与优化

### 7.1 成本估算

**单次训练成本**:
```
训练样本数: ~数千条
每个样本生成: 5个响应
每个响应长度: 平均5K tokens
评判调用: 每个响应1次

总评判调用: 数千 × 5 = 数万次
总token消耗: 数万 × 5K = 数亿tokens
```

使用GPT-4o成本较高，但可以通过以下方式优化：

### 7.2 成本优化策略

1. **使用更便宜的评判模型**:
   - GPT-4o-mini: 成本降低80%
   - 自部署的评判模型（如Qwen2.5-72B-Instruct）

2. **批量评判**:
   - 将多个评判请求合并
   - 减少API调用开销

3. **缓存机制**:
   - 相同completion的评判结果缓存
   - 避免重复评判

4. **分阶段使用**:
   - 前期训练使用规则奖励
   - 后期精调使用LLM评判

## 八、实验结果与消融研究

### 8.1 奖励函数的有效性

根据论文报告，使用GRPO + LLM-as-a-Judge后：

- **QA准确率**: 提升12%
- **代码执行成功率**: 提升18%
- **报告质量评分**（人工评估）: 提升25%

### 8.2 消融研究

测试不同奖励组合的效果：

| 奖励配置 | 准确率 | 代码质量 | 报告质量 |
|---------|--------|---------|---------|
| 仅规则奖励 | 78% | 72% | 65% |
| 规则 + 单维度LLM | 82% | 78% | 75% |
| **规则 + 5维度LLM** | **85%** | **83%** | **82%** |

**结论**: 多维度LLM评判显著提升性能

## 九、总结

DeepAnalyze的GRPO训练使用SkyRL框架，通过以下创新实现了高质量的开放式任务学习：

### 核心创新

1. **混合奖励函数**: 规则 + LLM-as-a-Judge
2. **多维度评估**: 5个独立维度全面衡量质量
3. **相对优势学习**: GRPO算法通过组内对比学习
4. **详细的评分标准**: 1-5分的细粒度rubric

### 技术要点

- **训练数据**: parquet格式，包含input_seq、workspace_id、reward_spec
- **GRPO配置**: 每个提示5个样本，不使用KL散度损失
- **评判LLM**: GPT-4o，结构化JSON输出
- **奖励归一化**: 所有奖励映射到[0, 1]

### 开放式答案处理

对于没有标准答案的开放式任务：
- 使用数据缩略图作为参考
- LLM从5个维度评判质量
- 通过相对对比学习最优策略
- 结合过程奖励鼓励探索

这套奖励机制使DeepAnalyze能够在没有明确标准答案的情况下，学习生成高质量的数据科学研究报告。

## 参考资料

- **论文**: https://arxiv.org/abs/2510.16872
- **SkyRL框架**: https://github.com/NovaSky-AI/SkyRL
- **训练代码**: `deepanalyze/SkyRL/skyrl-train/examples/deepanalyze/`
- **奖励函数实现**: `deepanalyze/SkyRL/skyrl-train/examples/deepanalyze/utils.py`
