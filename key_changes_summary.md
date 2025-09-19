# Key Changes Summary for Paper Update

## Critical Bug Fixed
- **Issue**: Previous analysis incorrectly deduplicated by `base_prompt`, treating multiple jailbreak attempts as single instances
- **Impact**: Inflated accuracy values (0.568 → 0.493 for best model)
- **Fix**: Each jailbreak attempt now evaluated separately (N=4,217 per model)

## Updated Metrics (All Models)

### Accuracy Rankings (with 95% CI):
1. **claude-sonnet-4**: 0.493 [0.479, 0.508] ← was 0.568
2. **kimi-k2**: 0.483 [0.468, 0.498] ← was 0.568
3. **deepseek-v3.1**: 0.482 [0.467, 0.497] ← was 0.562
4. **gemini-2.5-flash**: 0.444 [0.429, 0.459] ← was 0.512
5. **gpt-4.1**: 0.430 [0.415, 0.445] ← was 0.481
6. **Qwen3-235B-A22B-FP8**: 0.418 [0.403, 0.433] ← was 0.460

### Calibration Metrics:
| Model | ECE | Brier | Wrong@0.90 | AURC |
|-------|-----|-------|------------|------|
| claude-sonnet-4 | 0.315 | 0.338 | 39.2% | 0.422 |
| kimi-k2 | 0.383 | 0.388 | 44.8% | 0.434 |
| deepseek-v3.1 | 0.390 | 0.392 | 45.3% | 0.444 |
| gemini-2.5-flash | 0.464 | 0.456 | 53.0% | 0.450 |
| gpt-4.1 | 0.448 | 0.435 | 48.9% | 0.501 |
| Qwen3-235B-A22B-FP8 | 0.469 | 0.458 | 54.9% | 0.524 |

### Dataset Performance:
- **MHJ**: Easiest (71.7-85.7% accuracy)
- **SafeMTData_1K**: Moderate (49.5-63.5%)
- **SafeMTData_Attack600**: Difficult (16.2-32.8%)
- **CoSafe**: Hardest (22.2-30.9%)

## Paper Sections to Update

### Abstract ✓
- Accuracy: 0.568 → **0.493**
- AURC: 0.373 → **0.422**
- Wrong@0.90: 35.1% → **39.2%** (claude-sonnet-4)
- Wrong@0.90: 51.4% → **54.9%** (Qwen3-235B-A22B-FP8)

### Methodology
- Clarify N=4,217 evaluations per model (not unique prompts)
- Update calibration sampling description (adaptive importance sampling)

### Results Section
- Update Table 1 with new accuracy values
- Update all confidence intervals
- Update metacognition metrics table
- Update dataset-wise performance discussion

### Figures (6 total)
1. `fig_overall_accuracy.pdf` - Overall accuracy comparison
2. `fig_per_dataset_accuracy.pdf` - Dataset performance
3. `fig_calibration_metrics.pdf` - ECE vs Accuracy scatter
4. `fig_dataset_heatmap.pdf` - Model×Dataset heatmap
5. `fig_confidence_distribution.pdf` - Confidence distributions
6. `fig_metacognition.pdf` - Metacognition metrics

### Discussion
- Models perform worse than initially reported (~8-10% lower)
- Calibration remains a significant challenge
- Dataset difficulty ordering confirmed: MHJ > SafeMTData_1K > SafeMTData_Attack600 > CoSafe

## Key Takeaways
1. **Accuracy decreased** but relative rankings mostly preserved
2. **High-confidence errors increased** (39.2-54.9% Wrong@0.90)
3. **CoSafe is exceptionally difficult** (mean 25.7% accuracy)
4. **Claude-sonnet-4 still best** but margin reduced