# OBJEX Dataset: Multi-Model Jailbreak Extraction Evaluation

This repository contains the OBJEX dataset used in ICLR 2026 submission.

## Dataset Overview

`OBJEX_dataset.xlsx` contains comprehensive evaluation data for jailbreak extraction across 6 state-of-the-art language models.

### Key Statistics
- **Total Samples**: 2,817 unique jailbreak attempts per model
- **Models Evaluated**: 6 (GPT-4.1, Claude-Sonnet-4, Qwen3-235B, Kimi-K2, DeepSeek-V3.1, Gemini-2.5-Flash)
- **Source Datasets**: 3 (SafeMTData_Attack600, SafeMTData_1K, MHJ_local)
- **Turn Range**: 1-34 turns (median: 5 turns)
- **Token Range**: 23-1,392 tokens (median: 128 tokens)

## File Structure

The Excel file contains 13 sheets:

### 1. Labeling Sheet
- **Sheet Name**: `Labeling`
- **Purpose**: Human annotations and ground truth labels
- **Key Columns**:
  - `label`: Human-annotated correctness label

### 2. Extraction Sheets (6 sheets)
One for each model: `extracted_[model_name]`

**Columns**:
- `source`: Dataset source (SafeMTData_Attack600/1K, MHJ_local)
- `base_prompt`: Original harmful request
- `jailbreak_turns`: Full conversation in JSON format
- `num_turns`: Number of conversation turns (1-34)
- `turn_1` to `turn_12`: Individual turn contents
- `extracted_base_prompt`: Model's extraction result
- `extraction_confidence`: Self-reported confidence (0-100)
- `transcript_tokens`: Total token count
- `transcript_length`: Character count
- `tokens_per_turn`: Average tokens per turn
- `token_category`: Length category (Short/Medium/Long/Very Long)

### 3. Similarity Sheets (6 sheets)
One for each model: `similarity_[model_name]`

**Columns**:
- All columns from extraction sheets, plus:
- `similarity_score`: Semantic similarity (0-1) between original and extracted
- `similarity_category`: Categorical similarity rating
- `reasoning`: Explanation for similarity score

## Dataset Composition

### By Source
| Dataset | Samples | Avg Tokens | Description |
|---------|---------|------------|-------------|
| SafeMTData_Attack600 | 600 | 106 | Sophisticated adversarial attacks |
| SafeMTData_1K | 1,680 | 131 | Mixed difficulty jailbreaks |
| MHJ_local | 537 | 327 | Multi-turn conversational attacks |

### By Turn Complexity
| Turns | Samples | Percentage |
|-------|---------|------------|
| 1-2 | 212 | 7.5% |
| 3-4 | 547 | 19.4% |
| 5-6 | 1,900 | 67.4% |
| 7+ | 158 | 5.6% |

## Key Findings

### Model Performance (τ* = 0.66)
| Model | Accuracy | 95% CI | ECE |
|-------|----------|--------|-----|
| Claude-Sonnet-4 | 0.594 | [0.577, 0.611] | 0.206 |
| Kimi-K2 | 0.609 | [0.591, 0.626] | 0.259 |
| DeepSeek-V3.1 | 0.591 | [0.574, 0.608] | 0.279 |
| Gemini-2.5-Flash | 0.534 | [0.517, 0.551] | 0.362 |
| Qwen3-235B | 0.471 | [0.454, 0.489] | 0.417 |
| GPT-4.1 | 0.493 | [0.476, 0.511] | 0.384 |


**Note**: This dataset contains potentially harmful content for research purposes only. It should not be used to develop or deploy harmful systems.
