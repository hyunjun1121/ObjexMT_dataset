# OBJEX Analysis Results for New Models

**Analysis Date:** 2025-09-15
**Dataset Size:** 4,217 instances (100% coverage for all models)
**Threshold (τ*):** 0.61 (fixed, human-aligned)

---

## Executive Summary

새로운 3개 모델 (kimi-k2, deepseek-v3.1, gemini-2.5-flash)에 대한 완전한 OBJEXMT 벤치마크 분석을 수행했습니다. 모든 모델은 4,217개 인스턴스에 대해 100% coverage를 달성했으며, extraction error rate은 모두 0%였습니다.

---

## 1. 기본 메트릭 (Basic Metrics)

| Model | Overall Accuracy | 95% CI | Mean Similarity Score | Mean Self-Confidence | Extraction Error Rate |
|-------|------------------|--------|----------------------|---------------------|---------------------|
| **kimi-k2** | 0.491 | (0.476, 0.506) | 0.580 | 0.866 | 0.0% |
| **deepseek-v3.1** | 0.493 | (0.479, 0.507) | 0.581 | 0.871 | 0.0% |
| **gemini-2.5-flash** | 0.453 | (0.437, 0.468) | 0.563 | 0.894 | 0.0% |

### 주요 발견:
- **kimi-k2**와 **deepseek-v3.1**이 비슷한 성능 (49.1% vs 49.3%)
- **gemini-2.5-flash**가 상대적으로 낮은 성능 (45.3%)
- **gemini-2.5-flash**가 가장 높은 self-confidence (0.894)를 보임 → overconfidence 가능성

---

## 2. Per-source 분석

### 2.1 Dataset별 Accuracy

| Model | SafeMTData_Attack600 | SafeMTData_1K | MHJ_local | CoSafe | **Spread** |
|-------|---------------------|---------------|-----------|---------|-----------|
| **kimi-k2** | 0.347 (600) | 0.641 (1,680) | **0.873** (537) | 0.227 (1,400) | **0.646** |
| **deepseek-v3.1** | 0.337 (600) | 0.638 (1,680) | **0.834** (537) | 0.254 (1,400) | **0.580** |
| **gemini-2.5-flash** | **0.228** (600) | 0.582 (1,680) | **0.829** (537) | 0.251 (1,400) | **0.600** |

### 2.2 Dataset별 성능 패턴:
- **MHJ_local**에서 모든 모델이 최고 성능 (82.9%~87.3%)
- **CoSafe**에서 모든 모델이 최저 성능 (22.7%~25.4%)
- **SafeMTData_Attack600**에서 가장 큰 모델간 차이 (22.8%~34.7%)

---

## 3. Metacognition 메트릭

| Model | ECE (10 bins) | Brier Score | Wrong@0.80 | Wrong@0.90 | Wrong@0.95 | AURC |
|-------|---------------|-------------|------------|------------|------------|------|
| **kimi-k2** | 0.375 | 0.382 | 0.482 | 0.439 | 0.381 | 0.424 |
| **deepseek-v3.1** | 0.379 | 0.384 | 0.488 | 0.441 | 0.356 | 0.435 |
| **gemini-2.5-flash** | **0.451** | **0.448** | **0.527** | **0.521** | **0.454** | 0.442 |

### 주요 발견:
- **gemini-2.5-flash**가 모든 calibration 메트릭에서 가장 나쁜 성능
- 높은 ECE (0.451)와 Wrong@High-Conf 값들은 심각한 overconfidence를 시사
- **kimi-k2**와 **deepseek-v3.1**은 비슷한 calibration 특성

---

## 4. Robustness Check

### 4.1 Direct Categorical Mapping (Exact/High→1, else→0)

| Model | Overall Categorical Accuracy |
|-------|----------------------------|
| **kimi-k2** | 0.450 |
| **deepseek-v3.1** | 0.449 |
| **gemini-2.5-flash** | 0.415 |

### 4.2 Per-source Robustness

| Model | SafeMTData_Attack600 | SafeMTData_1K | MHJ_local | CoSafe |
|-------|---------------------|---------------|-----------|---------|
| **kimi-k2** | 0.287 | 0.592 | **0.831** | 0.203 |
| **deepseek-v3.1** | 0.272 | 0.591 | **0.808** | 0.217 |
| **gemini-2.5-flash** | **0.195** | 0.533 | **0.791** | 0.224 |

---

## 5. 종합 평가 및 권장사항

### 5.1 모델 순위 (종합 성능):
1. **deepseek-v3.1** - 균형잡힌 성능과 reasonable calibration
2. **kimi-k2** - 높은 정확도, 약간 나은 calibration
3. **gemini-2.5-flash** - 낮은 정확도, 심각한 overconfidence 문제

### 5.2 핵심 발견:
- **모든 모델이 MHJ_local에서 강점, CoSafe에서 약점**을 보임
- **gemini-2.5-flash의 overconfidence 문제**가 가장 심각
- **SafeMTData_Attack600**이 모델간 차별화가 가장 잘 되는 데이터셋

### 5.3 추가 분석 권장사항:
- 기존 3개 모델과의 pairwise comparison 분석
- Turn 수에 따른 성능 변화 분석
- Error case 질적 분석

---

## 데이터 검증

- ✅ **완전한 커버리지**: 모든 모델 4,217/4,217 (100%)
- ✅ **Zero extraction errors**: 모든 모델 0.0% error rate
- ✅ **Bootstrap confidence intervals**: 1,000 iterations
- ✅ **Robust metrics**: ECE (10 bins), AURC, multi-threshold analysis
- ✅ **Per-source validation**: 모든 4개 데이터셋에 대해 완전 분석

---

**분석 완료일**: 2025-09-15
**분석 툴**: Custom Python analysis with pandas, numpy, scipy
**결과 파일**: `new_models_analysis_results.json`