#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust recalibration with 300 human-labeled samples for OBJEXMT benchmark.
For NeurIPS 2025 submission - production-ready code with full error handling.

Author: OBJEXMT Team
Date: 2025-09-17
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    accuracy_score
)
import json
import warnings
import sys
import traceback
from typing import Dict, Tuple, List, Optional, Any

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class RobustRecalibration:
    """Robust recalibration class for OBJEXMT benchmark."""

    def __init__(self, excel_path: str):
        """Initialize with Excel file path."""
        self.excel_path = excel_path
        self.xls = None
        self.labeling_df = None
        self.threshold = None
        self.calibration_results = None

    def load_data(self) -> bool:
        """Safely load Excel file and verify structure."""
        try:
            print("Loading Excel file...")
            self.xls = pd.ExcelFile(self.excel_path)

            # Verify required sheets exist
            required_sheets = ['Labeling']
            missing = [s for s in required_sheets if s not in self.xls.sheet_names]
            if missing:
                raise ValueError(f"Missing required sheets: {missing}")

            # Load labeling data
            self.labeling_df = pd.read_excel(self.xls, 'Labeling')

            # Verify required columns
            required_cols = ['source', 'base_prompt', 'extracted_base_prompt',
                           'similarity_score', 'similarity_category', 'human_label']
            missing_cols = [c for c in required_cols if c not in self.labeling_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            print(f"✓ Successfully loaded {len(self.labeling_df)} labeling samples")
            return True

        except Exception as e:
            print(f"✗ Error loading data: {e}")
            traceback.print_exc()
            return False

    def calculate_threshold(self, n_bootstrap: int = 1000) -> Tuple[float, Dict]:
        """
        Calculate optimal threshold with bootstrap CI.

        Args:
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Tuple of (threshold, results_dict)
        """
        if self.labeling_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*60)
        print("THRESHOLD CALIBRATION WITH 300 SAMPLES")
        print("="*60)

        # Validate sample size
        n_samples = len(self.labeling_df)
        print(f"\nTotal calibration samples: {n_samples}")
        if n_samples != 300:
            print(f"WARNING: Expected 300 samples, got {n_samples}")

        # Source distribution
        source_counts = self.labeling_df['source'].value_counts().to_dict()
        print("\nSource distribution:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} ({count/n_samples*100:.1f}%)")

        # Human label distribution
        human_counts = self.labeling_df['human_label'].value_counts().to_dict()
        print("\nHuman label distribution:")
        for label, count in human_counts.items():
            print(f"  {label}: {count} ({count/n_samples*100:.1f}%)")

        # Convert to binary labels with validation
        def safe_label_to_binary(label):
            """Safely convert human label to binary."""
            if pd.isna(label):
                raise ValueError("Found NaN in human labels")
            if label in ['Exact match', 'High similarity']:
                return 1
            elif label in ['Moderate similarity', 'Low similarity']:
                return 0
            else:
                raise ValueError(f"Unknown label: {label}")

        try:
            self.labeling_df['human_binary'] = self.labeling_df['human_label'].apply(safe_label_to_binary)
        except Exception as e:
            print(f"Error converting labels: {e}")
            raise

        # Class balance
        pos_count = int(self.labeling_df['human_binary'].sum())
        neg_count = n_samples - pos_count
        pos_ratio = pos_count / n_samples
        neg_ratio = neg_count / n_samples

        print(f"\nClass balance after mapping (Exact/High→1, Moderate/Low→0):")
        print(f"  Positive: {pos_count} ({pos_ratio*100:.1f}%)")
        print(f"  Negative: {neg_count} ({neg_ratio*100:.1f}%)")

        # Validate similarity scores
        if self.labeling_df['similarity_score'].isna().any():
            print("WARNING: Found NaN similarity scores")
            self.labeling_df = self.labeling_df.dropna(subset=['similarity_score'])
            print(f"After removing NaNs: {len(self.labeling_df)} samples remain")

        # Find optimal threshold with grid search
        print("\nSearching for optimal threshold...")
        thresholds = np.arange(0.00, 1.01, 0.01)
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}

        for tau in thresholds:
            predictions = (self.labeling_df['similarity_score'] >= tau).astype(int)

            # Calculate metrics safely
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    self.labeling_df['human_binary'],
                    predictions,
                    average='binary',
                    zero_division=0
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = tau
                    best_metrics = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'support': int(support) if support is not None else 0
                    }
            except Exception as e:
                continue

        if best_threshold == 0:
            raise ValueError("Failed to find optimal threshold")

        print(f"\n[OPTIMAL THRESHOLD FOUND]")
        print(f"  τ* = {best_threshold:.2f}")
        print(f"  F1 = {best_metrics['f1']:.3f}")
        print(f"  Precision = {best_metrics['precision']:.3f}")
        print(f"  Recall = {best_metrics['recall']:.3f}")

        # Bootstrap confidence interval
        print(f"\nCalculating bootstrap CI (B={n_bootstrap})...")
        bootstrap_thresholds = []

        for i in range(n_bootstrap):
            if i % 200 == 0:
                print(f"  Bootstrap iteration {i}/{n_bootstrap}...")

            # Resample with replacement
            indices = np.random.choice(len(self.labeling_df), size=len(self.labeling_df), replace=True)
            boot_df = self.labeling_df.iloc[indices].copy()

            # Find optimal threshold for bootstrap sample
            boot_best_f1 = 0
            boot_best_tau = best_threshold  # Default to main threshold

            for tau in thresholds:
                predictions = (boot_df['similarity_score'] >= tau).astype(int)

                try:
                    f1 = f1_score(boot_df['human_binary'], predictions, zero_division=0)
                    if f1 > boot_best_f1:
                        boot_best_f1 = f1
                        boot_best_tau = tau
                except:
                    continue

            bootstrap_thresholds.append(boot_best_tau)

        # Calculate CI
        ci_lower = float(np.percentile(bootstrap_thresholds, 2.5))
        ci_upper = float(np.percentile(bootstrap_thresholds, 97.5))
        print(f"  95% CI for τ*: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Confusion matrix
        final_predictions = (self.labeling_df['similarity_score'] >= best_threshold).astype(int)
        cm = confusion_matrix(self.labeling_df['human_binary'], final_predictions)

        print("\nConfusion Matrix:")
        print("              Predicted")
        print("              Neg   Pos")
        print(f"Actual Neg   {cm[0,0]:3d}   {cm[0,1]:3d}")
        print(f"Actual Pos   {cm[1,0]:3d}   {cm[1,1]:3d}")

        # Store results
        self.threshold = best_threshold
        self.calibration_results = {
            'n_total': n_samples,
            'n_positive': pos_count,
            'n_negative': neg_count,
            'pos_ratio': pos_ratio,
            'neg_ratio': neg_ratio,
            'source_distribution': {str(k): int(v) for k, v in source_counts.items()},
            'human_label_distribution': {str(k): int(v) for k, v in human_counts.items()},
            'threshold': float(best_threshold),
            'f1': float(best_metrics['f1']),
            'precision': float(best_metrics['precision']),
            'recall': float(best_metrics['recall']),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confusion_matrix': cm.tolist(),
            'random_seed': RANDOM_SEED
        }

        return best_threshold, self.calibration_results

    def evaluate_model(self, model_name: str, sim_sheet: str,
                      ext_sheet: str, threshold: float) -> Dict:
        """
        Evaluate a single model with given threshold.

        Args:
            model_name: Model identifier
            sim_sheet: Similarity sheet name
            ext_sheet: Extraction sheet name
            threshold: Decision threshold

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Load data
            sim_df = pd.read_excel(self.xls, sim_sheet)
            ext_df = pd.read_excel(self.xls, ext_sheet)

            # Basic validation
            if len(sim_df) != len(ext_df):
                print(f"  WARNING: Size mismatch for {model_name}: sim={len(sim_df)}, ext={len(ext_df)}")

            # Calculate accuracy
            valid_mask = sim_df['similarity_score'].notna()
            valid_df = sim_df[valid_mask]
            n_total = len(sim_df)
            n_scored = len(valid_df)

            if n_scored == 0:
                raise ValueError(f"No valid similarity scores for {model_name}")

            predictions = (valid_df['similarity_score'] >= threshold).astype(int)
            accuracy = float(np.mean(predictions))

            # Bootstrap CI for accuracy
            n_bootstrap = 1000
            bootstrap_accs = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
                boot_predictions = predictions.iloc[indices] if hasattr(predictions, 'iloc') else predictions[indices]
                bootstrap_accs.append(np.mean(boot_predictions))

            ci_lower = float(np.percentile(bootstrap_accs, 2.5))
            ci_upper = float(np.percentile(bootstrap_accs, 97.5))

            # Coverage and mean scores
            coverage = float(n_scored / n_total)
            mean_similarity = float(valid_df['similarity_score'].mean())
            mean_confidence = float(ext_df['extraction_confidence'].mean())

            # Per-source accuracy (if available)
            per_source = {}
            if 'source' in sim_df.columns:
                for source in sim_df['source'].dropna().unique():
                    source_df = sim_df[sim_df['source'] == source]
                    source_valid = source_df[source_df['similarity_score'].notna()]
                    if len(source_valid) > 0:
                        source_preds = (source_valid['similarity_score'] >= threshold).astype(int)
                        per_source[str(source)] = {
                            'accuracy': float(np.mean(source_preds)),
                            'n_scored': int(len(source_valid)),
                            'n_total': int(len(source_df))
                        }

            # Calculate spread if per-source exists
            spread = None
            if per_source:
                accuracies = [v['accuracy'] for v in per_source.values()]
                if accuracies:
                    spread = float(max(accuracies) - min(accuracies))

            return {
                'model': model_name,
                'accuracy': accuracy,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'coverage': coverage,
                'n_scored': n_scored,
                'n_total': n_total,
                'mean_similarity': mean_similarity,
                'mean_confidence': mean_confidence,
                'per_source': per_source,
                'spread': spread
            }

        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
            return {
                'model': model_name,
                'error': str(e)
            }

    def evaluate_all_models(self, threshold: Optional[float] = None) -> Dict:
        """Evaluate all 6 models."""
        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            raise ValueError("No threshold available. Run calculate_threshold() first.")

        print("\n" + "="*60)
        print(f"EVALUATING ALL MODELS WITH THRESHOLD τ* = {threshold:.2f}")
        print("="*60)

        models = [
            {'name': 'gpt-4.1', 'sim': 'similarity_gpt-4.1', 'ext': 'extracted_gpt_4.1'},
            {'name': 'claude-sonnet-4', 'sim': 'similarity_claude-sonnet-4-2025', 'ext': 'extracted_claude-sonnet-4'},
            {'name': 'Qwen3-235B-A22B-FP8', 'sim': 'similarity_Qwen3-235B-A22B-fp8-', 'ext': 'extracted_Qwen3-235B-A22B-fp8-t'},
            {'name': 'kimi-k2', 'sim': 'similarity_moonshotaiKimi-K2-In', 'ext': 'extracted_moonshotaiKimi-K2-Ins'},
            {'name': 'deepseek-v3.1', 'sim': 'similarity_deepseek-aiDeepSeek-', 'ext': 'extracted_deepseek-aiDeepSeek-V'},
            {'name': 'gemini-2.5-flash', 'sim': 'similarity_gemini-2.5-flash', 'ext': 'extracted_gemini-2.5-flash'}
        ]

        results = {}
        for model in models:
            print(f"\nEvaluating {model['name']}...")
            result = self.evaluate_model(model['name'], model['sim'], model['ext'], threshold)

            if 'error' not in result:
                print(f"  ✓ Accuracy: {result['accuracy']:.3f} (CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}])")
                print(f"    Coverage: {result['coverage']:.3f} ({result['n_scored']}/{result['n_total']})")

                if result.get('per_source'):
                    print(f"    Per-source accuracy:")
                    for src, metrics in result['per_source'].items():
                        print(f"      {src}: {metrics['accuracy']:.3f} (n={metrics['n_scored']})")
            else:
                print(f"  ✗ Failed: {result['error']}")

            results[model['name']] = result

        return results


def generate_latex_updates(threshold: float, calibration: Dict, models: Dict) -> None:
    """Generate LaTeX code updates for the paper."""
    print("\n" + "="*60)
    print("LATEX UPDATES FOR PAPER")
    print("="*60)

    print("\n% === Abstract ===")
    print(f"% N=100 → N=300")
    print(f"calibrated once on \\textbf{{N=300}} items ($\\tau^\\star\\!=\\!{threshold:.2f}$)")

    print("\n% === Section 4.1 - Judge calibration ===")
    print(f"% Update Table 1")
    print(f"Pos./Neg. & {calibration['n_positive']} / {calibration['n_negative']} & "
          f"{threshold:.2f} & {calibration['f1']:.3f} & "
          f"{calibration['precision']:.3f} & {calibration['recall']:.3f} \\\\")

    print(f"\n% Class balance")
    print(f"class balance after the fixed mapping is "
          f"${calibration['pos_ratio']*100:.0f}\\%/{calibration['neg_ratio']*100:.0f}\\%$ (pos/neg)")

    print(f"\n% Source distribution")
    src_dist = calibration['source_distribution']
    print(f"The realized source mix is ", end="")
    parts = [f"\\emph{{{src.replace('_', '\\_')}: {cnt}}}" for src, cnt in src_dist.items()]
    print(", ".join(parts) + ".")

    print(f"\n% Threshold and metrics")
    print(f"$F_1$ peaks at $\\mathbf{{\\tau^\\star{{=}}{threshold:.2f}}}$ with "
          f"\\textbf{{$F_1$={calibration['f1']:.3f}}}, "
          f"\\textbf{{Precision={calibration['precision']:.3f}}}, "
          f"\\textbf{{Recall={calibration['recall']:.3f}}}.")

    print(f"\n% Bootstrap CI")
    print(f"A nonparametric bootstrap (B=1000) yields a \\textbf{{95\\% CI}} of "
          f"\\textbf{{[{calibration['ci_lower']:.3f}, {calibration['ci_upper']:.3f}]}} for $\\tau^\\star$.")

    print("\n% === Model accuracies (if changed) ===")
    for model_name, result in models.items():
        if 'accuracy' in result:
            print(f"% {model_name}: {result['accuracy']:.3f} "
                  f"(CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}])")


def main():
    """Main execution function."""
    print("="*60)
    print("OBJEXMT RECALIBRATION WITH 300 SAMPLES")
    print("Production-ready code for NeurIPS 2025")
    print("="*60)

    # Initialize
    excel_path = "E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx"
    recalibrator = RobustRecalibration(excel_path)

    # Load data
    if not recalibrator.load_data():
        print("Failed to load data. Exiting.")
        sys.exit(1)

    try:
        # Calculate new threshold
        threshold, calibration_results = recalibrator.calculate_threshold(n_bootstrap=1000)

        # Evaluate all models
        model_results = recalibrator.evaluate_all_models(threshold)

        # Generate LaTeX updates
        generate_latex_updates(threshold, calibration_results, model_results)

        # Save results
        final_results = {
            'calibration': calibration_results,
            'models': model_results,
            'timestamp': pd.Timestamp.now().isoformat(),
            'file_version': '1.0.0'
        }

        output_file = 'recalibration_300_final.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to {output_file}")
        print("\n" + "="*60)
        print("RECALIBRATION COMPLETE")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)