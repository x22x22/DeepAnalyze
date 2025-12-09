# DeepAnalyze Training Quick Reference

## 📊 Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepAnalyze Training Pipeline                     │
└─────────────────────────────────────────────────────────────────────┘

Stage 1: Single-Ability Fine-tuning (SFT)
┌────────────────────────────────────────────────┐
│ Base Model: DeepSeek-R1-Qwen3-8B (+ vocab)   │
│ Data: 420K reasoning samples                   │
│ Context: 8K tokens                             │
│ Focus: Foundation (Table, SQL, Code, Math)    │
│ Framework: ms-swift                            │
└────────────────────────────────────────────────┘
                        ↓
Stage 2: Multi-Ability Cold Start (SFT)
┌────────────────────────────────────────────────┐
│ Base Model: Stage 1 checkpoint                │
│ Data: 26K iteration samples                    │
│ Context: 32K tokens (4x increase)              │
│ Focus: End-to-end agentic tasks               │
│ Framework: ms-swift                            │
└────────────────────────────────────────────────┘
                        ↓
Stage 3: Reinforcement Learning (RL)
┌────────────────────────────────────────────────┐
│ Base Model: Stage 2 checkpoint                │
│ Data: RL trajectories (QA, Task, Research)    │
│ Context: 32K tokens                            │
│ Algorithm: GRPO (Group Relative Policy Opt)   │
│ Framework: SkyRL                               │
│ Reward: Hybrid (Rule + LLM Judge)             │
└────────────────────────────────────────────────┘
                        ↓
          ┌────────────────────────┐
          │   DeepAnalyze-8B       │
          │   Final Model          │
          └────────────────────────┘
```

## 🎯 Key Questions Answered

### Q1: Does DeepAnalyze use Reinforcement Learning?
**✅ YES** - Stage 3 uses GRPO algorithm with SkyRL framework.

### Q2: What are the highlights?
1. **Curriculum Training**: Progressive 3-stage approach
2. **Hybrid Rewards**: Rule-based + LLM-as-a-Judge
3. **Long Context**: Supports 32K tokens
4. **Tool Augmentation**: Real code/SQL execution feedback
5. **Fully Open**: Model, data, and code all open-sourced

### Q3: How does it train SQL tasks?
- **Stage 1**: Learn SQL syntax from Spider/BIRD datasets
- **Stage 2**: Complex multi-table reasoning and long queries
- **Stage 3**: Optimize from real execution results via RL

**Unique Features**:
- Multi-modal input (DB, CSV, Excel)
- Explainable SQL generation (`<Analyze>` tags)
- Error correction through multi-turn interaction
- Real database validation

### Q4: What's special about the reward function?
DeepAnalyze uses **multi-dimensional rewards**:

For **QA tasks**:
- Accuracy (exact match)
- Analysis quality (LLM-judged)

For **DataTask**:
- Code execution success rate
- Result quality (LLM-judged)

For **Research tasks**:
- Usefulness (1-5)
- Richness (1-5)
- Soundness (1-5)
- Interpretability (1-5)
- Readability (1-5)
- Code execution rate
- Interaction turns

## 🔧 Training Configuration Summary

| Stage | Model | Data Size | Context | Batch Size | LR | Epochs | Framework |
|-------|-------|-----------|---------|------------|-----|--------|-----------|
| 1-SFT | Base+vocab | 420K | 8K | 8 | 5e-5 | 3 | ms-swift |
| 2-SFT | Stage-1 | 26K | 32K | 1 | 5e-6 | 3 | ms-swift |
| 3-RL | Stage-2 | RL data | 32K | 256* | 5e-7 | 1 | SkyRL |

*RL batch size is for PPO training, not per-device batch

## 📦 Data Structure

```
DataScience-Instruct-500K/
├── reasoning/                    # Stage 1 Data (~420K)
│   ├── SKGInstruct_199989.json
│   ├── TableQA_distillation_39301.json
│   ├── TableQA_refinement_39301.json
│   ├── TableGPT_29448.json
│   ├── file_database_3833.json
│   ├── file_csv_3007.json
│   ├── file_xlsx_3663.json
│   ├── file_any_2520.json
│   ├── math_20000.json
│   ├── code_20000.json
│   ├── science_20000.json
│   ├── instruction_following_20000.json
│   └── other_19998.json
│
├── interation/                   # Stage 2 Data (~26K)
│   ├── data_pipeline_3601.json
│   ├── data_preparation_3311.json
│   ├── data_cleaning_1616.json
│   ├── data_analysis_3936.json
│   ├── data_insight_1062.json
│   ├── research_database_818.json
│   ├── research_xlsx_848.json
│   ├── research_other_3505.json
│   ├── research_data_preparation_488.json
│   ├── research_data_analysis_1339.json
│   ├── research_data_insight_1351.json
│   └── research_report_generation_4327.json
│
└── RL/                          # Stage 3 Data
    ├── qa.parquet
    ├── datatask.parquet
    └── reseach.parquet
```

## 🚀 Quick Start Training

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd ./deepanalyze/ms-swift/ && pip install -e .
cd ../SkyRL/ && pip install -e .

# 2. Download data and model
# Download DataScience-Instruct-500K from HuggingFace
# Download DeepSeek-R1-0528-Qwen3-8B

# 3. Add special tokens
python deepanalyze/add_vocab.py \
  --model_path <path-to-base-model> \
  --save_path <save-path> \
  --add_tags

# 4. Run training stages
cd deepanalyze/ms-swift/
bash ../../scripts/single.sh          # Stage 1
bash ../../scripts/multi_coldstart.sh # Stage 2
cd ../SkyRL/skyrl-train/
bash ../../scripts/multi_rl.sh        # Stage 3
```

## 📚 Detailed Documentation

- **Full Analysis (English)**: [TRAINING_ANALYSIS_EN.md](./TRAINING_ANALYSIS_EN.md)
- **完整分析（中文）**: [TRAINING_ANALYSIS_ZH.md](./TRAINING_ANALYSIS_ZH.md)

## 🔗 Resources

- **Paper**: https://arxiv.org/abs/2510.16872
- **Model**: https://huggingface.co/RUC-DataLab/DeepAnalyze-8B
- **Dataset**: https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K
- **GitHub**: https://github.com/ruc-datalab/DeepAnalyze
