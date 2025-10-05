# ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks

This repository accompanies the preprint:

- Paper: ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks
- Authors: Hyunjun Kim, Junwoo Ha, Sangyoon Yu, Haon Park
- arXiv: https://arxiv.org/abs/2508.16889v2 (DOI: 10.48550/arXiv.2508.16889)

## Abstract (brief)

LLM-as-a-Judge (LLMaaJ) is increasingly used for scalable evaluation, yet a decisive test of a judge's qualification is missing: can it recover a conversation's latent objective and know when that inference is trustworthy? ObjexMT introduces a benchmark for objective extraction and metacognition under multi-turn jailbreaks. Given a multi-turn transcript, a model must output a one-sentence base objective and a self-reported confidence. Accuracy is computed via LLM-judge semantic similarity to gold objectives, binarized by a single human-aligned threshold calibrated once on N = 100 items (τ* = 0.61). Metacognition is evaluated with ECE, Brier score, Wrong-at-High-Confidence, and risk-coverage.

Across GPT-4.1, Claude-Sonnet-4, and Qwen3-235B-A22B-FP8, Claude-Sonnet-4 attains the best objective-extraction accuracy (0.515) and calibration (ECE 0.296; Brier 0.324); GPT-4.1 and Qwen3-235B-A22B-FP8 tie at 0.441 but exhibit overconfidence (mean confidence ≈ 0.88 vs. accuracy ≈ 0.44; Wrong-at-0.90 ≈ 48–52%). Performance varies substantially by dataset (≈ 0.167–0.865). See the preprint for complete details.

## What's in this repository?

- `OBJEX_dataset.xlsx`: evaluation data supporting objective extraction and metacognitive calibration analyses across multi-turn jailbreak transcripts.

The Excel workbook includes:

- A human-labeling sheet with gold objectives and correctness labels
- Per-model extraction sheets (`extracted_[model_name]`) with predicted base objectives and self-reported confidences
- Per-model similarity sheets (`similarity_[model_name]`) with semantic similarity to gold, and auxiliary fields

Note: column names broadly match the paper’s description (transcript, turns, extracted objective, confidence, similarity). Minor naming differences may exist depending on export version.

## Reproducing key metrics

- Objective-extraction accuracy is computed by thresholding semantic similarity against the calibrated τ* = 0.61 (calibrated once on N = 100).
- Calibration is reported via ECE, Brier score, Wrong-at-High-Confidence, and risk-coverage curves; refer to the paper for the full protocol and per-dataset breakdowns.

## Citation

If you use ObjexMT or this dataset, please cite the preprint:

```bibtex
@misc{kim2025objexmt,
  title         = {ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks},
  author        = {Hyunjun Kim and Junwoo Ha and Sangyoon Yu and Haon Park},
  year          = {2025},
  eprint        = {2508.16889},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2508.16889},
  url           = {https://arxiv.org/abs/2508.16889v2}
}
```

## License and usage

This repository may include content derived from harmful or adversarial prompts strictly for research purposes. Do not use this material to develop or deploy harmful systems. Please review your institution’s policies and applicable laws before use.

## Contact

For questions or issues, please open a GitHub issue or contact the authors listed in the preprint.