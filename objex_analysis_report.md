# OBJEX Dataset Comprehensive Analysis Report
## Executive Summary
Analysis conducted on 6 models: gpt-4.1, claude-sonnet-4, Qwen3-235B, kimi-k2, deepseek-v3.1, gemini-2.5-flash

## Main Results

### Objective Extraction Accuracy
| Model | Accuracy | 95% CI |
|-------|----------|--------|
| claude-sonnet-4 | 0.589 | [0.574, 0.603] |
| Qwen3-235B | 0.522 | [0.507, 0.537] |
| kimi-k2 | 0.576 | [0.562, 0.591] |
| deepseek-v3.1 | 0.573 | [0.557, 0.588] |
| gemini-2.5-flash | 0.554 | [0.539, 0.569] |

### Metacognition Analysis
| Model | ECE | Brier Score | Wrong@0.8 | Wrong@0.9 | Wrong@0.95 | AURC |
|-------|-----|-------------|-----------|-----------|------------|------|
| gpt-4.1 | 0.121 | 0.024 | 0.000 | 0.000 | 0.000 | 0.000 |
| claude-sonnet-4 | 0.238 | 0.298 | 0.390 | 0.397 | 0.297 | 0.393 |
| Qwen3-235B | 0.362 | 0.386 | 0.477 | 0.481 | 0.454 | 0.473 |
| kimi-k2 | 0.286 | 0.320 | 0.398 | 0.361 | 0.313 | 0.350 |
| deepseek-v3.1 | 0.290 | 0.326 | 0.409 | 0.370 | 0.295 | 0.368 |
| gemini-2.5-flash | 0.261 | 0.370 | 0.426 | 0.421 | 0.364 | 0.365 |

### Pairwise Model Comparisons
| Comparison | Mean Difference | 95% CI | Significant |
|------------|-----------------|--------|-------------|
| claude-sonnet-4_vs_Qwen3-235B | 0.050 | [0.038, 0.062] | Yes |
| claude-sonnet-4_vs_kimi-k2 | 0.011 | [-0.000, 0.022] | No |
| claude-sonnet-4_vs_deepseek-v3.1 | 0.011 | [-0.001, 0.023] | No |
| claude-sonnet-4_vs_gemini-2.5-flash | 0.029 | [0.016, 0.040] | Yes |
| Qwen3-235B_vs_kimi-k2 | -0.039 | [-0.052, -0.027] | Yes |
| Qwen3-235B_vs_deepseek-v3.1 | -0.040 | [-0.051, -0.028] | Yes |
| Qwen3-235B_vs_gemini-2.5-flash | -0.022 | [-0.033, -0.010] | Yes |
| kimi-k2_vs_deepseek-v3.1 | -0.001 | [-0.007, 0.006] | No |
| kimi-k2_vs_gemini-2.5-flash | 0.017 | [0.011, 0.024] | Yes |
| deepseek-v3.1_vs_gemini-2.5-flash | 0.018 | [0.011, 0.025] | Yes |

## Quality Control Metrics
| Model | Judge Coverage | Mean Similarity | Mean Confidence |
|-------|----------------|-----------------|----------------|
| gpt-4.1 | 0.000 | 0.000 | 0.877 |
| claude-sonnet-4 | 0.999 | 0.592 | 0.807 |
| Qwen3-235B | 0.999 | 0.542 | 0.887 |
| kimi-k2 | 1.000 | 0.580 | 0.866 |
| deepseek-v3.1 | 1.000 | 0.581 | 0.871 |
| gemini-2.5-flash | 1.000 | 0.563 | 0.894 |
