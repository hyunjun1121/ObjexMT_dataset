# OBJEX Dataset Analysis Report - ICLR Quality
**Generated**: 2025-09-17 16:04:55
**Data Hash**: 3c62c9f3aa78e80920c5f4cc715fe910
**Random Seed**: 42

## Executive Summary
Comprehensive analysis of 6 models on the OBJEX dataset with 300 labeled samples.

### Key Findings
- **Best Performing Model**: claude-sonnet-4 (F1: 0.658)
- **Optimal Threshold**: 0.300 (F1: 0.485)
- **Significant Differences**: 7/10 pairwise comparisons

## 1. Human Alignment Analysis
### Threshold Optimization
- Best Threshold: 0.300
- Cross-validated F1: 0.485 ± 0.035

### Final Performance Metrics
- Accuracy: 0.448
- Precision: 0.418
- Recall: 0.578
- F1 Score: 0.485

### Inter-Model Agreement
- Mean Cohen's κ: 0.261


## 2. Model Performance Comparison
| Model | Accuracy | F1 Score | AUC-ROC | ECE | Brier Score |
|-------|----------|----------|---------|-----|-------------|
| claude-sonnet-4 | 0.551 [0.522, 0.580] | 0.658 [0.629, 0.686] | 0.637 | 0.074 | 0.155 |
| Qwen3-235B | 0.579 [0.549, 0.608] | 0.650 [0.621, 0.680] | 0.658 | 0.147 | 0.219 |
| kimi-k2 | 0.531 [0.502, 0.560] | 0.644 [0.614, 0.672] | 0.633 | 0.051 | 0.136 |
| deepseek-v3.1 | 0.544 [0.515, 0.573] | 0.654 [0.624, 0.681] | 0.641 | 0.049 | 0.131 |
| gemini-2.5-flash | 0.558 [0.528, 0.588] | 0.655 [0.626, 0.682] | 0.661 | 0.126 | 0.183 |

## 3. Statistical Analysis
### Normality Tests (Shapiro-Wilk)
- claude-sonnet-4: p=0.0000 (non-normal)
- Qwen3-235B: p=0.0000 (non-normal)
- kimi-k2: p=0.0000 (non-normal)
- deepseek-v3.1: p=0.0000 (non-normal)
- gemini-2.5-flash: p=0.0000 (non-normal)

### Significant Pairwise Comparisons (Bonferroni corrected)
- claude-sonnet-4_vs_Qwen3-235B: p<0.0000, Cohen's d=0.151 (negligible)
- claude-sonnet-4_vs_gemini-2.5-flash: p<0.0000, Cohen's d=0.087 (negligible)
- Qwen3-235B_vs_kimi-k2: p<0.0000, Cohen's d=-0.117 (negligible)
- Qwen3-235B_vs_deepseek-v3.1: p<0.0000, Cohen's d=-0.119 (negligible)
- Qwen3-235B_vs_gemini-2.5-flash: p<0.0029, Cohen's d=-0.065 (negligible)
- kimi-k2_vs_gemini-2.5-flash: p<0.0000, Cohen's d=0.053 (negligible)
- deepseek-v3.1_vs_gemini-2.5-flash: p<0.0000, Cohen's d=0.054 (negligible)

## 4. Error Analysis
### Error Distribution
| Model | False Positives | False Negatives | Total Errors |
|-------|----------------|----------------|-------------|
| gpt-4.1 | 0 | 0 | 0 |
| claude-sonnet-4 | 112 | 50 | 162 |
| Qwen3-235B | 90 | 70 | 160 |
| kimi-k2 | 114 | 55 | 169 |
| deepseek-v3.1 | 119 | 50 | 169 |
| gemini-2.5-flash | 108 | 60 | 168 |

### Most Difficult Examples
Found 20 examples that ≥3 models failed on.


## 5. Cross-Dataset Generalization
### Generalization Metrics
| Model | Mean Acc | Std Acc | CV | Best Dataset | Worst Dataset |
|-------|----------|---------|-----|--------------|---------------|
| claude-sonnet-4 | 0.458 | 0.000 | 0.000 | SafeMTData_Attack600 | SafeMTData_Attack600 |
| Qwen3-235B | 0.467 | 0.000 | 0.000 | SafeMTData_Attack600 | SafeMTData_Attack600 |
| kimi-k2 | 0.437 | 0.000 | 0.000 | SafeMTData_Attack600 | SafeMTData_Attack600 |
| deepseek-v3.1 | 0.437 | 0.000 | 0.000 | SafeMTData_Attack600 | SafeMTData_Attack600 |
| gemini-2.5-flash | 0.440 | 0.000 | 0.000 | SafeMTData_Attack600 | SafeMTData_Attack600 |

## 6. Reproducibility Information
- Random Seed: 42
- Bootstrap Iterations: 10000
- Cross-validation Folds: 5
- Multiple Testing Correction: bonferroni
- Significance Level: 0.05
