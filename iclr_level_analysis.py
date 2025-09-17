"""
OBJEX Dataset Analysis - ICLR Main Track Quality
Complete analysis pipeline with all necessary components for publication-quality research
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    shapiro, levene, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    spearmanr, pearsonr, chi2_contingency, fisher_exact
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, log_loss
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
    train_test_split, GridSearchCV
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import pickle
from datetime import datetime
import hashlib

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

@dataclass
class ExperimentConfig:
    """Configuration for experiment reproducibility"""
    random_seed: int = 42
    n_bootstrap: int = 10000  # Production value for server
    n_permutation: int = 10000  # Production value for server
    cv_folds: int = 5
    test_size: float = 0.2
    confidence_level: float = 0.95
    significance_alpha: float = 0.05
    correction_method: str = 'bonferroni'  # or 'fdr_bh'

class ICLRLevelAnalyzer:
    """Advanced analyzer for ICLR-quality OBJEX dataset analysis"""

    def __init__(self, excel_path: str, config: ExperimentConfig = None):
        """Initialize analyzer with configuration"""
        self.excel_path = excel_path
        self.config = config or ExperimentConfig()
        self.xls = pd.ExcelFile(excel_path)

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Data containers
        self.models = []
        self.data = {}
        self.results = {}
        self.visualizations = {}

        # Analysis metadata
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'data_hash': self._compute_data_hash()
        }

        # Load all data
        self._load_data()

    def _compute_data_hash(self) -> str:
        """Compute hash of data file for reproducibility tracking"""
        with open(self.excel_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_data(self):
        """Load and validate all data from Excel file"""
        print("Loading and validating data...")

        # Load primary sheets
        self.prompt_df = pd.read_excel(self.xls, 'Prompt')
        self.labeling_df = pd.read_excel(self.xls, 'Labeling')

        # Validate human labels exist
        if 'human_label' not in self.labeling_df.columns:
            raise ValueError("human_label column not found in Labeling sheet")

        # Model name mapping
        model_mapping = {
            'gpt_4.1': 'gpt-4.1',
            'claude-sonnet-4': 'claude-sonnet-4',
            'Qwen3-235B-A22B-fp8-t': 'Qwen3-235B',
            'moonshotaiKimi-K2-Ins': 'kimi-k2',
            'deepseek-aiDeepSeek-V': 'deepseek-v3.1',
            'gemini-2.5-flash': 'gemini-2.5-flash'
        }

        # Load model data
        for sheet in self.xls.sheet_names:
            if sheet.startswith('extracted_'):
                model_name = sheet.replace('extracted_', '')
                short_name = model_mapping.get(model_name, model_name)

                if short_name not in self.models:
                    self.models.append(short_name)
                    self.data[short_name] = {}

                # Load extracted data
                self.data[short_name]['extracted'] = pd.read_excel(self.xls, sheet)

            elif sheet.startswith('similarity_'):
                # Match similarity sheet to model
                for orig_name, short_name in model_mapping.items():
                    if orig_name[:15] in sheet or sheet.endswith(orig_name[:15]):
                        if short_name in self.data:
                            self.data[short_name]['similarity'] = pd.read_excel(self.xls, sheet)
                        break

        # Validate data integrity
        self._validate_data_integrity()

        print(f"Loaded data for {len(self.models)} models: {', '.join(self.models)}")

    def _validate_data_integrity(self):
        """Validate data consistency and completeness"""
        issues = []

        for model in self.models:
            if 'extracted' not in self.data[model]:
                issues.append(f"Missing extracted data for {model}")
            if 'similarity' not in self.data[model]:
                issues.append(f"Missing similarity data for {model}")

            # Check for required columns
            if 'extracted' in self.data[model]:
                df = self.data[model]['extracted']
                if 'extraction_confidence' not in df.columns:
                    issues.append(f"Missing extraction_confidence for {model}")

            if 'similarity' in self.data[model]:
                df = self.data[model]['similarity']
                if 'similarity_score' not in df.columns:
                    issues.append(f"Missing similarity_score for {model}")
                if 'similarity_category' not in df.columns:
                    issues.append(f"Missing similarity_category for {model}")

        if issues:
            print("Data validation warnings:")
            for issue in issues:
                print(f"  - {issue}")

    def analyze_human_alignment_advanced(self) -> Dict:
        """Advanced human alignment analysis with cross-validation"""
        print("\n=== Advanced Human Alignment Analysis ===")

        results = {
            'threshold_optimization': {},
            'cross_validation': {},
            'inter_annotator': {}
        }

        # Get human labels and convert categories to binary
        human_label_categories = self.labeling_df['human_label'].values

        # Convert categorical labels to binary (1 for correct, 0 for incorrect)
        label_mapping = {
            'Exact match': 1,
            'High similarity': 1,
            'Moderate similarity': 0,
            'Low similarity': 0
        }

        human_labels = np.array([label_mapping.get(label, 0) for label in human_label_categories])

        # Prepare combined data for all models
        all_scores = []
        all_labels = []
        model_indices = []

        for model in self.models:
            if 'similarity' in self.data[model]:
                df = self.data[model]['similarity']
                if 'similarity_score' in df.columns:
                    scores = pd.to_numeric(df['similarity_score'], errors='coerce')
                    valid_idx = ~scores.isna()
                    scores = scores[valid_idx].values

                    # Ensure alignment with human labels
                    min_len = min(len(scores), len(human_labels))
                    all_scores.extend(scores[:min_len])
                    all_labels.extend(human_labels[:min_len])
                    model_indices.extend([model] * min_len)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 1. Grid Search for Optimal Threshold with Cross-Validation
        print("Performing threshold optimization with cross-validation...")

        thresholds = np.linspace(0.3, 0.8, 51)
        cv_results = []

        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True,
                              random_state=self.config.random_seed)

        for threshold in thresholds:
            fold_scores = []

            for train_idx, val_idx in skf.split(all_scores, all_labels):
                train_scores, val_scores = all_scores[train_idx], all_scores[val_idx]
                train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

                # Predict on validation set
                val_preds = (val_scores >= threshold).astype(int)

                # Calculate metrics
                f1 = f1_score(val_labels, val_preds)
                fold_scores.append(f1)

            cv_results.append({
                'threshold': threshold,
                'mean_f1': np.mean(fold_scores),
                'std_f1': np.std(fold_scores),
                'ci_lower': np.percentile(fold_scores, 2.5),
                'ci_upper': np.percentile(fold_scores, 97.5)
            })

        # Find best threshold
        best_result = max(cv_results, key=lambda x: x['mean_f1'])
        results['threshold_optimization'] = {
            'best_threshold': best_result['threshold'],
            'best_f1': best_result['mean_f1'],
            'f1_std': best_result['std_f1'],
            'all_results': cv_results
        }

        print(f"Best threshold: {best_result['threshold']:.3f} (F1: {best_result['mean_f1']:.3f} ± {best_result['std_f1']:.3f})")

        # 2. Final evaluation with best threshold
        best_threshold = best_result['threshold']
        final_preds = (all_scores >= best_threshold).astype(int)

        results['final_metrics'] = {
            'accuracy': accuracy_score(all_labels, final_preds),
            'precision': precision_score(all_labels, final_preds),
            'recall': recall_score(all_labels, final_preds),
            'f1': f1_score(all_labels, final_preds),
            'confusion_matrix': confusion_matrix(all_labels, final_preds).tolist()
        }

        # 3. Inter-annotator agreement (if multiple annotators available)
        # For now, calculate agreement between model predictions
        if len(self.models) > 1:
            print("Calculating inter-model agreement...")

            # Reshape predictions for each model
            model_preds = {}
            for model in self.models:
                if 'similarity' in self.data[model]:
                    df = self.data[model]['similarity']
                    if 'similarity_score' in df.columns:
                        scores = pd.to_numeric(df['similarity_score'], errors='coerce')
                        valid_idx = ~scores.isna()
                        scores = scores[valid_idx].values
                        min_len = min(len(scores), len(human_labels))
                        preds = (scores[:min_len] >= best_threshold).astype(int)
                        model_preds[model] = preds

            # Calculate pairwise Cohen's kappa
            kappa_matrix = np.zeros((len(self.models), len(self.models)))
            for i, model1 in enumerate(self.models):
                for j, model2 in enumerate(self.models):
                    if model1 in model_preds and model2 in model_preds:
                        min_len = min(len(model_preds[model1]), len(model_preds[model2]))
                        kappa = cohen_kappa_score(
                            model_preds[model1][:min_len],
                            model_preds[model2][:min_len]
                        )
                        kappa_matrix[i, j] = kappa

            results['inter_annotator']['cohen_kappa_matrix'] = kappa_matrix.tolist()
            results['inter_annotator']['mean_kappa'] = np.mean(kappa_matrix[np.triu_indices_from(kappa_matrix, k=1)])

            print(f"Mean inter-model agreement (Cohen's κ): {results['inter_annotator']['mean_kappa']:.3f}")

        self.results['human_alignment'] = results
        return results

    def analyze_model_performance_comprehensive(self) -> Dict:
        """Comprehensive model performance analysis with multiple metrics"""
        print("\n=== Comprehensive Model Performance Analysis ===")

        results = {}
        best_threshold = self.results.get('human_alignment', {}).get('threshold_optimization', {}).get('best_threshold', 0.5)

        for model in self.models:
            print(f"\nAnalyzing {model}...")
            model_results = {}

            if 'similarity' in self.data[model]:
                df = self.data[model]['similarity']

                if 'similarity_score' in df.columns and 'base_prompt' in df.columns:
                    # Merge similarity data with human labels based on base_prompt
                    label_mapping = {
                        'Exact match': 1,
                        'High similarity': 1,
                        'Moderate similarity': 0,
                        'Low similarity': 0
                    }

                    # Create labeled dataframe
                    labeling_with_binary = self.labeling_df.copy()
                    labeling_with_binary['human_label_binary'] = labeling_with_binary['human_label'].map(label_mapping)

                    # Merge with model's similarity scores based on base_prompt
                    merged = pd.merge(
                        df[['base_prompt', 'similarity_score']],
                        labeling_with_binary[['base_prompt', 'human_label_binary']],
                        on='base_prompt',
                        how='inner'
                    )

                    # Get aligned scores and labels
                    scores = pd.to_numeric(merged['similarity_score'], errors='coerce')
                    human_labels = merged['human_label_binary'].values

                    # Remove NaN values
                    valid_mask = ~scores.isna()
                    scores = scores[valid_mask]
                    human_labels = human_labels[valid_mask]

                    # Binary predictions
                    predictions = (scores >= best_threshold).astype(int)

                    # 1. Basic metrics
                    model_results['basic_metrics'] = {
                        'n_samples': len(scores),
                        'accuracy': accuracy_score(human_labels, predictions),
                        'precision': precision_score(human_labels, predictions),
                        'recall': recall_score(human_labels, predictions),
                        'f1': f1_score(human_labels, predictions),
                        'mcc': self._calculate_mcc(human_labels, predictions)
                    }

                    # 2. Bootstrap confidence intervals
                    boot_metrics = self._bootstrap_metrics(scores.values, human_labels, best_threshold)
                    model_results['bootstrap_ci'] = boot_metrics

                    # 3. ROC and PR curves
                    fpr, tpr, roc_thresholds = roc_curve(human_labels, scores)
                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(human_labels, scores)

                    model_results['roc_analysis'] = {
                        'auc': auc(fpr, tpr),
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': roc_thresholds.tolist()
                    }

                    model_results['pr_analysis'] = {
                        'avg_precision': average_precision_score(human_labels, scores),
                        'precision': precision_curve.tolist(),
                        'recall': recall_curve.tolist(),
                        'thresholds': pr_thresholds.tolist()
                    }

                    # 4. Calibration analysis
                    if 'extraction_confidence' in self.data[model]['extracted'].columns:
                        confidences = pd.to_numeric(
                            self.data[model]['extracted']['extraction_confidence'],
                            errors='coerce'
                        ).dropna()

                        min_len = min(len(confidences), len(scores))
                        confidences = confidences[:min_len]
                        actuals = (scores[:min_len] >= best_threshold).astype(int)

                        # ECE and MCE
                        ece, mce = self._calculate_calibration_errors(confidences.values, actuals.values)

                        # Brier Score
                        brier = mean_squared_error(actuals, confidences)

                        model_results['calibration'] = {
                            'ece': ece,
                            'mce': mce,
                            'brier_score': brier,
                            'log_loss': log_loss(actuals, confidences)
                        }

                        # Reliability diagram data
                        fraction_pos, mean_pred = calibration_curve(actuals, confidences, n_bins=10)
                        model_results['calibration']['reliability_diagram'] = {
                            'fraction_positive': fraction_pos.tolist(),
                            'mean_predicted': mean_pred.tolist()
                        }

                    # 5. Per-category performance
                    if 'similarity_category' in df.columns:
                        categories = df['similarity_category'].values[:len(scores)]
                        category_mapping = {
                            'Exact match': 1,
                            'High similarity': 1,
                            'Moderate similarity': 0,
                            'Low similarity': 0
                        }

                        category_preds = np.array([category_mapping.get(c, 0) for c in categories])

                        model_results['category_performance'] = {
                            'accuracy': accuracy_score(human_labels, category_preds),
                            'category_distribution': Counter(categories)
                        }

            results[model] = model_results

            # Print summary
            if 'basic_metrics' in model_results:
                metrics = model_results['basic_metrics']
                ci = model_results.get('bootstrap_ci', {})
                print(f"  Accuracy: {metrics['accuracy']:.3f} ({ci.get('accuracy_ci', [0,0])[0]:.3f}, {ci.get('accuracy_ci', [1,1])[1]:.3f})")
                print(f"  F1 Score: {metrics['f1']:.3f} ({ci.get('f1_ci', [0,0])[0]:.3f}, {ci.get('f1_ci', [1,1])[1]:.3f})")
                print(f"  AUC-ROC: {model_results.get('roc_analysis', {}).get('auc', 0):.3f}")

        self.results['model_performance'] = results
        return results

    def _calculate_mcc(self, y_true, y_pred):
        """Calculate Matthews Correlation Coefficient"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator != 0 else 0

    def _bootstrap_metrics(self, scores, labels, threshold, n_boot=None):
        """Calculate bootstrap confidence intervals for metrics"""
        n_boot = n_boot or self.config.n_bootstrap
        n_samples = len(scores)

        boot_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for _ in range(n_boot):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            boot_scores = scores[idx]
            boot_labels = labels[idx]
            boot_preds = (boot_scores >= threshold).astype(int)

            boot_results['accuracy'].append(accuracy_score(boot_labels, boot_preds))
            boot_results['precision'].append(precision_score(boot_labels, boot_preds, zero_division=0))
            boot_results['recall'].append(recall_score(boot_labels, boot_preds, zero_division=0))
            boot_results['f1'].append(f1_score(boot_labels, boot_preds, zero_division=0))

        ci_results = {}
        for metric, values in boot_results.items():
            ci_results[f'{metric}_mean'] = np.mean(values)
            ci_results[f'{metric}_std'] = np.std(values)
            ci_results[f'{metric}_ci'] = [
                np.percentile(values, 2.5),
                np.percentile(values, 97.5)
            ]

        return ci_results

    def _calculate_calibration_errors(self, confidences, actuals, n_bins=10):
        """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        mce = 0

        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(actuals[mask])
                bin_conf = np.mean(confidences[mask])
                bin_weight = np.sum(mask) / len(confidences)

                calibration_error = np.abs(bin_acc - bin_conf)
                ece += bin_weight * calibration_error
                mce = max(mce, calibration_error)

        return ece, mce

    def analyze_statistical_significance_advanced(self) -> Dict:
        """Advanced statistical significance testing with multiple corrections"""
        print("\n=== Advanced Statistical Significance Analysis ===")

        results = {
            'normality_tests': {},
            'variance_tests': {},
            'pairwise_tests': {},
            'effect_sizes': {},
            'omnibus_tests': {}
        }

        # Prepare data for each model
        model_scores = {}
        for model in self.models:
            if 'similarity' in self.data[model]:
                df = self.data[model]['similarity']
                if 'similarity_score' in df.columns:
                    scores = pd.to_numeric(df['similarity_score'], errors='coerce').dropna().values
                    model_scores[model] = scores

        # 1. Test for normality
        print("Testing normality assumptions...")
        for model, scores in model_scores.items():
            stat, p = shapiro(scores[:min(5000, len(scores))])  # Shapiro-Wilk test
            results['normality_tests'][model] = {
                'statistic': stat,
                'p_value': p,
                'is_normal': p > 0.05
            }

        # 2. Test for homogeneity of variance
        print("Testing homogeneity of variance...")
        if len(model_scores) > 1:
            scores_list = list(model_scores.values())
            min_len = min(len(s) for s in scores_list)
            truncated_scores = [s[:min_len] for s in scores_list]

            stat, p = levene(*truncated_scores)
            results['variance_tests']['levene'] = {
                'statistic': stat,
                'p_value': p,
                'equal_variance': p > 0.05
            }

        # 3. Omnibus tests
        print("Performing omnibus tests...")
        if len(model_scores) > 2:
            # Kruskal-Wallis H-test (non-parametric)
            stat, p = kruskal(*truncated_scores)
            results['omnibus_tests']['kruskal_wallis'] = {
                'statistic': stat,
                'p_value': p,
                'significant': p < 0.05
            }

            # Friedman test (if we have paired data)
            try:
                stat, p = friedmanchisquare(*truncated_scores)
                results['omnibus_tests']['friedman'] = {
                    'statistic': stat,
                    'p_value': p,
                    'significant': p < 0.05
                }
            except:
                pass

        # 4. Pairwise comparisons with multiple testing correction
        print("Performing pairwise comparisons...")
        pairwise_results = []
        effect_sizes = {}

        for i, (model1, scores1) in enumerate(model_scores.items()):
            for j, (model2, scores2) in enumerate(model_scores.items()):
                if i < j:
                    min_len = min(len(scores1), len(scores2))
                    s1, s2 = scores1[:min_len], scores2[:min_len]

                    # Paired t-test
                    t_stat, t_p = stats.ttest_rel(s1, s2)

                    # Wilcoxon signed-rank test (non-parametric)
                    w_stat, w_p = wilcoxon(s1, s2)

                    # Mann-Whitney U test (independent samples)
                    u_stat, u_p = mannwhitneyu(s1, s2, alternative='two-sided')

                    # Effect size calculations
                    cohen_d = self._calculate_cohen_d(s1, s2)
                    cliff_delta = self._calculate_cliff_delta(s1, s2)

                    comparison_key = f"{model1}_vs_{model2}"

                    pairwise_results.append({
                        'comparison': comparison_key,
                        't_test_p': t_p,
                        'wilcoxon_p': w_p,
                        'mann_whitney_p': u_p
                    })

                    effect_sizes[comparison_key] = {
                        'mean_diff': np.mean(s1) - np.mean(s2),
                        'cohen_d': cohen_d,
                        'cliff_delta': cliff_delta,
                        'interpretation': self._interpret_effect_size(cohen_d)
                    }

        # Apply multiple testing correction
        if pairwise_results:
            # Extract p-values for correction
            t_pvals = [r['t_test_p'] for r in pairwise_results]
            w_pvals = [r['wilcoxon_p'] for r in pairwise_results]
            u_pvals = [r['mann_whitney_p'] for r in pairwise_results]

            # Apply Bonferroni correction
            t_corrected = multipletests(t_pvals, method='bonferroni')
            w_corrected = multipletests(w_pvals, method='bonferroni')
            u_corrected = multipletests(u_pvals, method='bonferroni')

            # Apply FDR correction
            t_fdr = multipletests(t_pvals, method='fdr_bh')
            w_fdr = multipletests(w_pvals, method='fdr_bh')
            u_fdr = multipletests(u_pvals, method='fdr_bh')

            for i, result in enumerate(pairwise_results):
                result['t_test_bonferroni'] = t_corrected[1][i]
                result['t_test_fdr'] = t_fdr[1][i]
                result['wilcoxon_bonferroni'] = w_corrected[1][i]
                result['wilcoxon_fdr'] = w_fdr[1][i]
                result['mann_whitney_bonferroni'] = u_corrected[1][i]
                result['mann_whitney_fdr'] = u_fdr[1][i]

                # Determine significance
                result['significant_bonferroni'] = t_corrected[0][i] or w_corrected[0][i]
                result['significant_fdr'] = t_fdr[0][i] or w_fdr[0][i]

        results['pairwise_tests'] = pairwise_results
        results['effect_sizes'] = effect_sizes

        # Print summary
        print("\nSignificant comparisons (Bonferroni corrected):")
        for result in pairwise_results:
            if result['significant_bonferroni']:
                comparison = result['comparison']
                effect = effect_sizes[comparison]
                print(f"  {comparison}: p<{result['t_test_bonferroni']:.4f}, Cohen's d={effect['cohen_d']:.3f} ({effect['interpretation']})")

        self.results['statistical_significance'] = results
        return results

    def _calculate_cohen_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0

    def _calculate_cliff_delta(self, group1, group2):
        """Calculate Cliff's Delta (non-parametric effect size)"""
        n1, n2 = len(group1), len(group2)
        dominance = 0

        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1

        return dominance / (n1 * n2)

    def _interpret_effect_size(self, cohen_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def analyze_errors_detailed(self) -> Dict:
        """Detailed error analysis and categorization"""
        print("\n=== Detailed Error Analysis ===")

        results = {
            'error_categories': {},
            'error_patterns': {},
            'feature_correlations': {},
            'difficult_examples': []
        }

        best_threshold = self.results.get('human_alignment', {}).get('threshold_optimization', {}).get('best_threshold', 0.5)

        for model in self.models:
            print(f"\nAnalyzing errors for {model}...")
            model_errors = {
                'false_positives': [],
                'false_negatives': [],
                'error_by_dataset': {},
                'error_by_length': {},
                'error_by_confidence': {}
            }

            if 'similarity' in self.data[model] and 'extracted' in self.data[model]:
                sim_df = self.data[model]['similarity']
                ext_df = self.data[model]['extracted']

                if 'similarity_score' in sim_df.columns:
                    scores = pd.to_numeric(sim_df['similarity_score'], errors='coerce')

                    # Convert human labels to binary
                    human_label_categories = self.labeling_df['human_label'].values
                    label_mapping = {
                        'Exact match': 1,
                        'High similarity': 1,
                        'Moderate similarity': 0,
                        'Low similarity': 0
                    }
                    human_labels_binary = np.array([label_mapping.get(label, 0) for label in human_label_categories])

                    # Match lengths
                    min_len = min(len(scores), len(human_labels_binary))
                    scores = scores[:min_len]
                    human_labels = human_labels_binary[:min_len]

                    # Get predictions
                    predictions = (scores >= best_threshold).astype(int)

                    # Identify errors
                    for i in range(min(len(predictions), len(human_labels))):
                        if not pd.isna(scores.iloc[i]):
                            pred, true = predictions[i], human_labels[i]

                            if pred == 1 and true == 0:  # False positive
                                error_info = {
                                    'index': i,
                                    'score': scores.iloc[i],
                                    'prediction': pred,
                                    'truth': true,
                                    'type': 'false_positive'
                                }

                                # Add additional features if available
                                if 'source' in ext_df.columns and i < len(ext_df):
                                    error_info['dataset'] = ext_df.iloc[i]['source']
                                if 'extraction_confidence' in ext_df.columns and i < len(ext_df):
                                    error_info['confidence'] = ext_df.iloc[i]['extraction_confidence']
                                if 'extracted_objective' in ext_df.columns and i < len(ext_df):
                                    obj = ext_df.iloc[i]['extracted_objective']
                                    if pd.notna(obj):
                                        error_info['objective_length'] = len(str(obj))

                                model_errors['false_positives'].append(error_info)

                            elif pred == 0 and true == 1:  # False negative
                                error_info = {
                                    'index': i,
                                    'score': scores.iloc[i],
                                    'prediction': pred,
                                    'truth': true,
                                    'type': 'false_negative'
                                }

                                # Add additional features if available
                                if 'source' in ext_df.columns and i < len(ext_df):
                                    error_info['dataset'] = ext_df.iloc[i]['source']
                                if 'extraction_confidence' in ext_df.columns and i < len(ext_df):
                                    error_info['confidence'] = ext_df.iloc[i]['extraction_confidence']
                                if 'extracted_objective' in ext_df.columns and i < len(ext_df):
                                    obj = ext_df.iloc[i]['extracted_objective']
                                    if pd.notna(obj):
                                        error_info['objective_length'] = len(str(obj))

                                model_errors['false_negatives'].append(error_info)

                    # Analyze error patterns

                    # 1. Errors by dataset
                    if 'source' in ext_df.columns:
                        sources = ext_df['source'].values[:len(predictions)]
                        for source in np.unique(sources):
                            source_mask = sources == source
                            # Ensure mask doesn't exceed human_labels length
                            mask_len = min(len(source_mask), len(human_labels))
                            source_mask = source_mask[:mask_len]
                            source_preds = predictions[:mask_len][source_mask]
                            source_labels = human_labels[:mask_len][source_mask[:len(human_labels)]]

                            if len(source_preds) > 0 and len(source_labels) > 0:
                                min_compare_len = min(len(source_preds), len(source_labels))
                                source_errors = np.sum(source_preds[:min_compare_len] != source_labels[:min_compare_len])
                                model_errors['error_by_dataset'][source] = {
                                    'error_rate': source_errors / min_compare_len if min_compare_len > 0 else 0,
                                    'n_errors': int(source_errors),
                                    'n_total': int(min_compare_len)
                                }

                    # 2. Errors by confidence level
                    if 'extraction_confidence' in ext_df.columns:
                        confidences = pd.to_numeric(ext_df['extraction_confidence'], errors='coerce')
                        conf_bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]

                        for low, high in conf_bins:
                            mask = (confidences >= low) & (confidences < high) & (~confidences.isna())
                            if np.sum(mask) > 0:
                                masked_preds = predictions[mask[:len(predictions)]]
                                masked_labels = human_labels[mask[:len(human_labels)]]
                                min_len = min(len(masked_preds), len(masked_labels))

                                errors = np.sum(masked_preds[:min_len] != masked_labels[:min_len])
                                model_errors['error_by_confidence'][f"{low:.1f}-{high:.1f}"] = {
                                    'error_rate': errors / min_len if min_len > 0 else 0,
                                    'n_errors': int(errors),
                                    'n_total': int(min_len)
                                }

                    # 3. Feature importance for errors
                    if len(model_errors['false_positives']) + len(model_errors['false_negatives']) > 0:
                        # Calculate correlation between features and errors
                        error_mask = predictions != human_labels

                        correlations = {}

                        # Correlation with score
                        correlations['score'] = pearsonr(scores.fillna(0), error_mask[:len(scores)])[0]

                        # Correlation with confidence
                        if 'extraction_confidence' in ext_df.columns:
                            confidences = pd.to_numeric(ext_df['extraction_confidence'], errors='coerce').fillna(0)
                            min_len = min(len(confidences), len(error_mask))
                            correlations['confidence'] = pearsonr(confidences[:min_len], error_mask[:min_len])[0]

                        # Correlation with objective length
                        if 'extracted_objective' in ext_df.columns:
                            lengths = ext_df['extracted_objective'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                            min_len = min(len(lengths), len(error_mask))
                            correlations['objective_length'] = pearsonr(lengths[:min_len], error_mask[:min_len])[0]

                        model_errors['feature_correlations'] = correlations

            results['error_categories'][model] = model_errors

            # Print summary
            print(f"  False positives: {len(model_errors['false_positives'])}")
            print(f"  False negatives: {len(model_errors['false_negatives'])}")
            if model_errors['error_by_dataset']:
                print("  Error rates by dataset:")
                for dataset, info in model_errors['error_by_dataset'].items():
                    print(f"    {dataset}: {info['error_rate']:.3f} ({info['n_errors']}/{info['n_total']})")

        # Identify consistently difficult examples across models
        all_errors = defaultdict(list)
        for model, errors in results['error_categories'].items():
            for fp in errors['false_positives']:
                all_errors[fp['index']].append((model, 'FP'))
            for fn in errors['false_negatives']:
                all_errors[fn['index']].append((model, 'FN'))

        # Find examples that multiple models get wrong
        difficult_examples = []
        for idx, model_errors in all_errors.items():
            if len(model_errors) >= len(self.models) // 2:  # At least half the models fail
                difficult_examples.append({
                    'index': idx,
                    'models_failed': model_errors,
                    'difficulty_score': len(model_errors) / len(self.models)
                })

        results['difficult_examples'] = sorted(difficult_examples,
                                              key=lambda x: x['difficulty_score'],
                                              reverse=True)[:20]  # Top 20 most difficult

        print(f"\nFound {len(results['difficult_examples'])} consistently difficult examples")

        self.results['error_analysis'] = results
        return results

    def analyze_cross_dataset_generalization(self) -> Dict:
        """Analyze model generalization across different datasets"""
        print("\n=== Cross-Dataset Generalization Analysis ===")

        results = {
            'dataset_performance': {},
            'generalization_matrix': {},
            'domain_shift_analysis': {}
        }

        best_threshold = self.results.get('human_alignment', {}).get('threshold_optimization', {}).get('best_threshold', 0.5)

        # Get unique datasets
        all_datasets = set()
        for model in self.models:
            if 'extracted' in self.data[model]:
                df = self.data[model]['extracted']
                if 'source' in df.columns:
                    all_datasets.update(df['source'].unique())

        all_datasets = sorted(list(all_datasets))
        print(f"Found {len(all_datasets)} datasets: {', '.join(all_datasets)}")

        # Analyze performance per dataset for each model
        for model in self.models:
            model_perf = {}

            if 'similarity' in self.data[model] and 'extracted' in self.data[model]:
                sim_df = self.data[model]['similarity']
                ext_df = self.data[model]['extracted']

                if 'similarity_score' in sim_df.columns and 'source' in ext_df.columns:
                    scores = pd.to_numeric(sim_df['similarity_score'], errors='coerce')
                    sources = ext_df['source'].values[:len(scores)]

                    # Convert human labels to binary
                    human_label_categories = self.labeling_df['human_label'].values
                    label_mapping = {
                        'Exact match': 1,
                        'High similarity': 1,
                        'Moderate similarity': 0,
                        'Low similarity': 0
                    }
                    human_labels_binary = np.array([label_mapping.get(label, 0) for label in human_label_categories])

                    # Match lengths
                    min_len = min(len(scores), len(human_labels_binary), len(sources))
                    scores = scores[:min_len]
                    sources = sources[:min_len]
                    human_labels = human_labels_binary[:min_len]

                    for dataset in all_datasets:
                        mask = sources == dataset
                        if np.sum(mask) > 0:
                            dataset_scores = scores[mask]
                            dataset_labels = human_labels[mask[:len(human_labels)]]

                            # Remove NaN values
                            valid_mask = ~dataset_scores.isna()
                            dataset_scores = dataset_scores[valid_mask]
                            dataset_labels = dataset_labels[valid_mask[:len(dataset_labels)]]

                            if len(dataset_scores) > 0:
                                predictions = (dataset_scores >= best_threshold).astype(int)

                                model_perf[dataset] = {
                                    'n_samples': len(dataset_scores),
                                    'accuracy': accuracy_score(dataset_labels, predictions),
                                    'f1': f1_score(dataset_labels, predictions),
                                    'mean_score': np.mean(dataset_scores)
                                }

            results['dataset_performance'][model] = model_perf

        # Calculate generalization matrix (performance variance across datasets)
        for model, perf in results['dataset_performance'].items():
            if perf:
                accuracies = [p['accuracy'] for p in perf.values()]
                results['generalization_matrix'][model] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'cv': np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0,
                    'range': max(accuracies) - min(accuracies) if accuracies else 0,
                    'worst_dataset': min(perf.items(), key=lambda x: x[1]['accuracy'])[0] if perf else None,
                    'best_dataset': max(perf.items(), key=lambda x: x[1]['accuracy'])[0] if perf else None
                }

        # Analyze domain shift
        if len(all_datasets) > 1:
            print("\nDomain shift analysis...")

            for model in self.models:
                if model in results['dataset_performance']:
                    perf = results['dataset_performance'][model]

                    # Calculate pairwise dataset similarity
                    dataset_similarity = {}
                    for ds1 in all_datasets:
                        for ds2 in all_datasets:
                            if ds1 < ds2 and ds1 in perf and ds2 in perf:
                                # Use performance difference as proxy for domain shift
                                perf_diff = abs(perf[ds1]['accuracy'] - perf[ds2]['accuracy'])
                                dataset_similarity[f"{ds1}_vs_{ds2}"] = 1 - perf_diff

                    results['domain_shift_analysis'][model] = dataset_similarity

        # Print summary
        print("\nGeneralization Summary:")
        for model, gen in results['generalization_matrix'].items():
            print(f"\n{model}:")
            print(f"  Mean accuracy: {gen['mean_accuracy']:.3f} ± {gen['std_accuracy']:.3f}")
            print(f"  Coefficient of variation: {gen['cv']:.3f}")
            print(f"  Best dataset: {gen['best_dataset']}")
            print(f"  Worst dataset: {gen['worst_dataset']}")

        self.results['generalization'] = results
        return results

    def create_visualizations(self) -> None:
        """Create all publication-quality visualizations"""
        print("\n=== Creating Visualizations ===")

        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

        # Create output directory
        vis_dir = Path('visualizations')
        vis_dir.mkdir(exist_ok=True)

        # 1. Threshold Optimization Curve
        self._plot_threshold_optimization(vis_dir)

        # 2. Model Performance Comparison
        self._plot_model_comparison(vis_dir)

        # 3. ROC Curves
        self._plot_roc_curves(vis_dir)

        # 4. Calibration Plots
        self._plot_calibration(vis_dir)

        # 5. Error Analysis Heatmap
        self._plot_error_heatmap(vis_dir)

        # 6. Cross-Dataset Performance
        self._plot_cross_dataset(vis_dir)

        # 7. Statistical Significance Matrix
        self._plot_significance_matrix(vis_dir)

        # 8. Effect Size Visualization
        self._plot_effect_sizes(vis_dir)

        print(f"Visualizations saved to {vis_dir}/")

    def _plot_threshold_optimization(self, output_dir):
        """Plot threshold optimization results"""
        if 'human_alignment' not in self.results:
            return

        results = self.results['human_alignment']['threshold_optimization']['all_results']

        fig, ax = plt.subplots(figsize=(10, 6))

        thresholds = [r['threshold'] for r in results]
        mean_f1s = [r['mean_f1'] for r in results]
        ci_lower = [r['ci_lower'] for r in results]
        ci_upper = [r['ci_upper'] for r in results]

        ax.plot(thresholds, mean_f1s, 'b-', linewidth=2, label='Mean F1')
        ax.fill_between(thresholds, ci_lower, ci_upper, alpha=0.3, label='95% CI')

        best_threshold = self.results['human_alignment']['threshold_optimization']['best_threshold']
        best_f1 = self.results['human_alignment']['threshold_optimization']['best_f1']
        ax.axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.3f}')
        ax.plot(best_threshold, best_f1, 'ro', markersize=10)

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Threshold Optimization via Cross-Validation', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_optimization.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_comparison(self, output_dir):
        """Plot model performance comparison"""
        if 'model_performance' not in self.results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = []
        accuracies = []
        f1_scores = []
        aucs = []
        ci_lower_acc = []
        ci_upper_acc = []
        ci_lower_f1 = []
        ci_upper_f1 = []

        for model, perf in self.results['model_performance'].items():
            if 'basic_metrics' in perf:
                models.append(model)
                accuracies.append(perf['basic_metrics']['accuracy'])
                f1_scores.append(perf['basic_metrics']['f1'])

                if 'bootstrap_ci' in perf:
                    ci_lower_acc.append(perf['bootstrap_ci']['accuracy_ci'][0])
                    ci_upper_acc.append(perf['bootstrap_ci']['accuracy_ci'][1])
                    ci_lower_f1.append(perf['bootstrap_ci']['f1_ci'][0])
                    ci_upper_f1.append(perf['bootstrap_ci']['f1_ci'][1])
                else:
                    ci_lower_acc.append(perf['basic_metrics']['accuracy'])
                    ci_upper_acc.append(perf['basic_metrics']['accuracy'])
                    ci_lower_f1.append(perf['basic_metrics']['f1'])
                    ci_upper_f1.append(perf['basic_metrics']['f1'])

                if 'roc_analysis' in perf:
                    aucs.append(perf['roc_analysis']['auc'])
                else:
                    aucs.append(0)

        # Accuracy plot
        x_pos = np.arange(len(models))
        axes[0].bar(x_pos, accuracies, yerr=[np.array(accuracies) - np.array(ci_lower_acc),
                                              np.array(ci_upper_acc) - np.array(accuracies)],
                   capsize=5, color='skyblue', edgecolor='navy', linewidth=2)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Accuracy Comparison', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')

        # F1 Score plot
        axes[1].bar(x_pos, f1_scores, yerr=[np.array(f1_scores) - np.array(ci_lower_f1),
                                            np.array(ci_upper_f1) - np.array(f1_scores)],
                   capsize=5, color='lightcoral', edgecolor='darkred', linewidth=2)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('F1 Score Comparison', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')

        # AUC plot
        axes[2].bar(x_pos, aucs, color='lightgreen', edgecolor='darkgreen', linewidth=2)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].set_ylabel('AUC-ROC', fontsize=12)
        axes[2].set_title('AUC-ROC Comparison', fontsize=14)
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self, output_dir):
        """Plot ROC curves for all models"""
        if 'model_performance' not in self.results:
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        for model, perf in self.results['model_performance'].items():
            if 'roc_analysis' in perf:
                fpr = perf['roc_analysis']['fpr']
                tpr = perf['roc_analysis']['tpr']
                auc_score = perf['roc_analysis']['auc']

                ax.plot(fpr, tpr, linewidth=2, label=f'{model} (AUC={auc_score:.3f})')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curves.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_calibration(self, output_dir):
        """Plot calibration plots"""
        if 'model_performance' not in self.results:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (model, perf) in enumerate(self.results['model_performance'].items()):
            if idx >= 6:
                break

            ax = axes[idx]

            if 'calibration' in perf and 'reliability_diagram' in perf['calibration']:
                fraction_pos = perf['calibration']['reliability_diagram']['fraction_positive']
                mean_pred = perf['calibration']['reliability_diagram']['mean_predicted']

                # Plot reliability diagram
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
                ax.plot(mean_pred, fraction_pos, 'bo-', linewidth=2, markersize=8, label='Model')

                # Add ECE and MCE info
                ece = perf['calibration']['ece']
                brier = perf['calibration']['brier_score']

                ax.text(0.05, 0.95, f'ECE: {ece:.3f}\nBrier: {brier:.3f}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_xlabel('Mean Predicted Probability', fontsize=10)
                ax.set_ylabel('Fraction of Positives', fontsize=10)
                ax.set_title(f'{model}', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=8)
                ax.set_aspect('equal')

        # Remove empty subplots
        for idx in range(len(self.results['model_performance']), 6):
            fig.delaxes(axes[idx])

        plt.suptitle('Calibration Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plots.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_heatmap(self, output_dir):
        """Plot error analysis heatmap"""
        if 'error_analysis' not in self.results:
            return

        # Create error rate matrix
        datasets = set()
        for model_errors in self.results['error_analysis']['error_categories'].values():
            datasets.update(model_errors.get('error_by_dataset', {}).keys())

        datasets = sorted(list(datasets))

        if not datasets:
            return

        error_matrix = []
        model_names = []

        for model in self.models:
            if model in self.results['error_analysis']['error_categories']:
                model_errors = self.results['error_analysis']['error_categories'][model]
                row = []
                for dataset in datasets:
                    if dataset in model_errors.get('error_by_dataset', {}):
                        row.append(model_errors['error_by_dataset'][dataset]['error_rate'])
                    else:
                        row.append(0)
                error_matrix.append(row)
                model_names.append(model)

        if error_matrix:
            fig, ax = plt.subplots(figsize=(10, 6))

            im = ax.imshow(error_matrix, cmap='YlOrRd', aspect='auto')

            ax.set_xticks(np.arange(len(datasets)))
            ax.set_yticks(np.arange(len(model_names)))
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.set_yticklabels(model_names)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Error Rate', rotation=270, labelpad=15)

            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(datasets)):
                    text = ax.text(j, i, f'{error_matrix[i][j]:.2f}',
                                  ha='center', va='center', color='black' if error_matrix[i][j] < 0.5 else 'white')

            ax.set_title('Error Rates by Model and Dataset', fontsize=14)
            ax.set_xlabel('Dataset', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)

            plt.tight_layout()
            plt.savefig(output_dir / 'error_heatmap.pdf', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_cross_dataset(self, output_dir):
        """Plot cross-dataset generalization"""
        if 'generalization' not in self.results:
            return

        gen_matrix = self.results['generalization']['generalization_matrix']

        if not gen_matrix:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Mean accuracy vs std
        models = []
        means = []
        stds = []

        for model, metrics in gen_matrix.items():
            models.append(model)
            means.append(metrics['mean_accuracy'])
            stds.append(metrics['std_accuracy'])

        ax1.scatter(means, stds, s=100, alpha=0.6)
        for i, model in enumerate(models):
            ax1.annotate(model, (means[i], stds[i]), fontsize=8)

        ax1.set_xlabel('Mean Accuracy', fontsize=12)
        ax1.set_ylabel('Std Accuracy', fontsize=12)
        ax1.set_title('Generalization: Mean vs Variability', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Performance by dataset
        dataset_perf = self.results['generalization']['dataset_performance']

        # Reorganize data for plotting
        all_datasets = set()
        for model_perf in dataset_perf.values():
            all_datasets.update(model_perf.keys())
        all_datasets = sorted(list(all_datasets))

        if all_datasets:
            x_pos = np.arange(len(all_datasets))
            width = 0.8 / len(self.models)

            for i, model in enumerate(self.models):
                if model in dataset_perf:
                    accuracies = []
                    for dataset in all_datasets:
                        if dataset in dataset_perf[model]:
                            accuracies.append(dataset_perf[model][dataset]['accuracy'])
                        else:
                            accuracies.append(0)

                    ax2.bar(x_pos + i * width, accuracies, width, label=model, alpha=0.8)

            ax2.set_xticks(x_pos + width * (len(self.models) - 1) / 2)
            ax2.set_xticklabels(all_datasets, rotation=45, ha='right')
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('Performance Across Datasets', fontsize=14)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Cross-Dataset Generalization Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'cross_dataset.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_significance_matrix(self, output_dir):
        """Plot statistical significance matrix"""
        if 'statistical_significance' not in self.results:
            return

        pairwise_tests = self.results['statistical_significance']['pairwise_tests']

        if not pairwise_tests:
            return

        # Create significance matrix
        n_models = len(self.models)
        sig_matrix = np.ones((n_models, n_models))

        for test in pairwise_tests:
            comparison = test['comparison']
            model1, model2 = comparison.split('_vs_')

            if model1 in self.models and model2 in self.models:
                i, j = self.models.index(model1), self.models.index(model2)
                # Use Bonferroni-corrected p-value
                p_val = test['t_test_bonferroni']
                sig_matrix[i, j] = p_val
                sig_matrix[j, i] = p_val

        fig, ax = plt.subplots(figsize=(8, 8))

        # Create custom colormap
        im = ax.imshow(sig_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')

        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(self.models, rotation=45, ha='right')
        ax.set_yticklabels(self.models)

        # Add significance indicators
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    p_val = sig_matrix[i, j]
                    if p_val < 0.001:
                        text = '***'
                    elif p_val < 0.01:
                        text = '**'
                    elif p_val < 0.05:
                        text = '*'
                    else:
                        text = 'ns'

                    ax.text(j, i, text, ha='center', va='center',
                           color='white' if p_val < 0.05 else 'black', fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value (Bonferroni corrected)', rotation=270, labelpad=15)

        ax.set_title('Statistical Significance Matrix', fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / 'significance_matrix.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_effect_sizes(self, output_dir):
        """Plot effect sizes for pairwise comparisons"""
        if 'statistical_significance' not in self.results:
            return

        effect_sizes = self.results['statistical_significance']['effect_sizes']

        if not effect_sizes:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        comparisons = []
        cohen_ds = []
        cliff_deltas = []

        for comp, effects in effect_sizes.items():
            comparisons.append(comp.replace('_vs_', ' vs\n'))
            cohen_ds.append(effects['cohen_d'])
            cliff_deltas.append(effects['cliff_delta'])

        x_pos = np.arange(len(comparisons))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, cohen_ds, width, label="Cohen's d", color='steelblue')
        bars2 = ax.bar(x_pos + width/2, cliff_deltas, width, label="Cliff's δ", color='coral')

        # Add interpretation colors
        for bar, d in zip(bars1, cohen_ds):
            if abs(d) < 0.2:
                bar.set_alpha(0.3)
            elif abs(d) < 0.5:
                bar.set_alpha(0.5)
            elif abs(d) < 0.8:
                bar.set_alpha(0.7)
            else:
                bar.set_alpha(1.0)

        ax.set_xlabel('Comparison', fontsize=12)
        ax.set_ylabel('Effect Size', fontsize=12)
        ax.set_title('Effect Sizes for Pairwise Model Comparisons', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparisons, rotation=0, ha='center', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Add effect size interpretation guide
        ax.text(0.02, 0.98, 'Effect Size:\nLarge (>0.8)\nMedium (0.5-0.8)\nSmall (0.2-0.5)\nNegligible (<0.2)',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / 'effect_sizes.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self, output_file='iclr_analysis_report.md'):
        """Generate comprehensive ICLR-quality analysis report"""
        print("\n=== Generating Comprehensive Report ===")

        report = []
        report.append("# OBJEX Dataset Analysis Report - ICLR Quality\n")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Data Hash**: {self.metadata['data_hash']}\n")
        report.append(f"**Random Seed**: {self.config.random_seed}\n\n")

        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(f"Comprehensive analysis of {len(self.models)} models on the OBJEX dataset ")
        report.append(f"with {len(self.labeling_df)} labeled samples.\n\n")

        # Key Findings
        report.append("### Key Findings\n")

        if 'model_performance' in self.results:
            # Find best model
            best_model = max(self.results['model_performance'].items(),
                           key=lambda x: x[1].get('basic_metrics', {}).get('f1', 0))
            report.append(f"- **Best Performing Model**: {best_model[0]} ")
            report.append(f"(F1: {best_model[1]['basic_metrics']['f1']:.3f})\n")

        if 'human_alignment' in self.results:
            best_threshold = self.results['human_alignment']['threshold_optimization']['best_threshold']
            best_f1 = self.results['human_alignment']['threshold_optimization']['best_f1']
            report.append(f"- **Optimal Threshold**: {best_threshold:.3f} (F1: {best_f1:.3f})\n")

        if 'statistical_significance' in self.results:
            sig_pairs = sum(1 for r in self.results['statistical_significance']['pairwise_tests']
                          if r['significant_bonferroni'])
            total_pairs = len(self.results['statistical_significance']['pairwise_tests'])
            report.append(f"- **Significant Differences**: {sig_pairs}/{total_pairs} pairwise comparisons\n")

        # Detailed Results Sections
        report.append("\n## 1. Human Alignment Analysis\n")
        if 'human_alignment' in self.results:
            ha = self.results['human_alignment']

            report.append("### Threshold Optimization\n")
            report.append(f"- Best Threshold: {ha['threshold_optimization']['best_threshold']:.3f}\n")
            report.append(f"- Cross-validated F1: {ha['threshold_optimization']['best_f1']:.3f} ± ")
            report.append(f"{ha['threshold_optimization']['f1_std']:.3f}\n\n")

            if 'final_metrics' in ha:
                report.append("### Final Performance Metrics\n")
                fm = ha['final_metrics']
                report.append(f"- Accuracy: {fm['accuracy']:.3f}\n")
                report.append(f"- Precision: {fm['precision']:.3f}\n")
                report.append(f"- Recall: {fm['recall']:.3f}\n")
                report.append(f"- F1 Score: {fm['f1']:.3f}\n\n")

            if 'inter_annotator' in ha and 'mean_kappa' in ha['inter_annotator']:
                report.append("### Inter-Model Agreement\n")
                report.append(f"- Mean Cohen's κ: {ha['inter_annotator']['mean_kappa']:.3f}\n\n")

        report.append("\n## 2. Model Performance Comparison\n")
        if 'model_performance' in self.results:
            report.append("| Model | Accuracy | F1 Score | AUC-ROC | ECE | Brier Score |\n")
            report.append("|-------|----------|----------|---------|-----|-------------|\n")

            for model, perf in self.results['model_performance'].items():
                if 'basic_metrics' in perf:
                    acc = perf['basic_metrics']['accuracy']
                    f1 = perf['basic_metrics']['f1']
                    auc_val = perf.get('roc_analysis', {}).get('auc', 0)
                    ece = perf.get('calibration', {}).get('ece', 0)
                    brier = perf.get('calibration', {}).get('brier_score', 0)

                    # Add confidence intervals if available
                    if 'bootstrap_ci' in perf:
                        acc_ci = perf['bootstrap_ci']['accuracy_ci']
                        f1_ci = perf['bootstrap_ci']['f1_ci']
                        acc_str = f"{acc:.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]"
                        f1_str = f"{f1:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
                    else:
                        acc_str = f"{acc:.3f}"
                        f1_str = f"{f1:.3f}"

                    report.append(f"| {model} | {acc_str} | {f1_str} | {auc_val:.3f} | {ece:.3f} | {brier:.3f} |\n")

        report.append("\n## 3. Statistical Analysis\n")
        if 'statistical_significance' in self.results:
            ss = self.results['statistical_significance']

            # Normality tests
            if 'normality_tests' in ss:
                report.append("### Normality Tests (Shapiro-Wilk)\n")
                for model, test in ss['normality_tests'].items():
                    report.append(f"- {model}: p={test['p_value']:.4f} ")
                    report.append(f"({'normal' if test['is_normal'] else 'non-normal'})\n")
                report.append("\n")

            # Significant comparisons
            report.append("### Significant Pairwise Comparisons (Bonferroni corrected)\n")
            for test in ss['pairwise_tests']:
                if test['significant_bonferroni']:
                    comparison = test['comparison']
                    p_val = test['t_test_bonferroni']

                    # Get effect size
                    if comparison in ss['effect_sizes']:
                        effect = ss['effect_sizes'][comparison]
                        report.append(f"- {comparison}: p<{p_val:.4f}, ")
                        report.append(f"Cohen's d={effect['cohen_d']:.3f} ({effect['interpretation']})\n")

        report.append("\n## 4. Error Analysis\n")
        if 'error_analysis' in self.results:
            ea = self.results['error_analysis']

            report.append("### Error Distribution\n")
            report.append("| Model | False Positives | False Negatives | Total Errors |\n")
            report.append("|-------|----------------|----------------|-------------|\n")

            for model, errors in ea['error_categories'].items():
                fp = len(errors['false_positives'])
                fn = len(errors['false_negatives'])
                total = fp + fn
                report.append(f"| {model} | {fp} | {fn} | {total} |\n")

            # Difficult examples
            if ea['difficult_examples']:
                report.append(f"\n### Most Difficult Examples\n")
                report.append(f"Found {len(ea['difficult_examples'])} examples that ")
                report.append(f"≥{len(self.models)//2} models failed on.\n\n")

        report.append("\n## 5. Cross-Dataset Generalization\n")
        if 'generalization' in self.results:
            gen = self.results['generalization']

            if 'generalization_matrix' in gen:
                report.append("### Generalization Metrics\n")
                report.append("| Model | Mean Acc | Std Acc | CV | Best Dataset | Worst Dataset |\n")
                report.append("|-------|----------|---------|-----|--------------|---------------|\n")

                for model, metrics in gen['generalization_matrix'].items():
                    report.append(f"| {model} | {metrics['mean_accuracy']:.3f} | ")
                    report.append(f"{metrics['std_accuracy']:.3f} | {metrics['cv']:.3f} | ")
                    report.append(f"{metrics['best_dataset']} | {metrics['worst_dataset']} |\n")

        # Reproducibility
        report.append("\n## 6. Reproducibility Information\n")
        report.append(f"- Random Seed: {self.config.random_seed}\n")
        report.append(f"- Bootstrap Iterations: {self.config.n_bootstrap}\n")
        report.append(f"- Cross-validation Folds: {self.config.cv_folds}\n")
        report.append(f"- Multiple Testing Correction: {self.config.correction_method}\n")
        report.append(f"- Significance Level: {self.config.significance_alpha}\n")

        # Write report
        report_text = ''.join(report)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"Report saved to {output_file}")

        # Also save as LaTeX table snippets
        self._generate_latex_tables()

        return report_text

    def _generate_latex_tables(self):
        """Generate LaTeX tables for paper inclusion"""
        latex_dir = Path('latex_tables')
        latex_dir.mkdir(exist_ok=True)

        # Main results table
        if 'model_performance' in self.results:
            with open(latex_dir / 'main_results.tex', 'w') as f:
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Model Performance on OBJEX Dataset}\n")
                f.write("\\label{tab:main_results}\n")
                f.write("\\begin{tabular}{lccccc}\n")
                f.write("\\toprule\n")
                f.write("Model & Accuracy & F1 Score & AUC-ROC & ECE & Brier Score \\\\\n")
                f.write("\\midrule\n")

                for model, perf in self.results['model_performance'].items():
                    if 'basic_metrics' in perf:
                        acc = perf['basic_metrics']['accuracy']
                        f1 = perf['basic_metrics']['f1']
                        auc_val = perf.get('roc_analysis', {}).get('auc', 0)
                        ece = perf.get('calibration', {}).get('ece', 0)
                        brier = perf.get('calibration', {}).get('brier_score', 0)

                        f.write(f"{model} & {acc:.3f} & {f1:.3f} & {auc_val:.3f} & {ece:.3f} & {brier:.3f} \\\\\n")

                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")

        print("LaTeX tables saved to latex_tables/")

    def save_results(self, output_file='iclr_analysis_results.pkl'):
        """Save all results for reproducibility"""
        with open(output_file, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'metadata': self.metadata,
                'config': self.config.__dict__
            }, f)
        print(f"Results saved to {output_file}")

    def run_complete_analysis(self):
        """Run the complete ICLR-quality analysis pipeline"""
        print("\n" + "="*60)
        print("Starting ICLR-Quality OBJEX Analysis Pipeline")
        print("="*60)

        try:
            # 1. Human alignment analysis
            self.analyze_human_alignment_advanced()

            # 2. Comprehensive model performance
            self.analyze_model_performance_comprehensive()

            # 3. Statistical significance testing
            self.analyze_statistical_significance_advanced()

            # 4. Detailed error analysis
            self.analyze_errors_detailed()

            # 5. Cross-dataset generalization
            self.analyze_cross_dataset_generalization()

            # 6. Create visualizations
            self.create_visualizations()

            # 7. Generate comprehensive report
            self.generate_comprehensive_report()

            # 8. Save results
            self.save_results()

            print("\n" + "="*60)
            print("Analysis Pipeline Completed Successfully!")
            print("="*60)

            # Print summary statistics
            self._print_summary()

        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()

        return self.results

    def _print_summary(self):
        """Print analysis summary"""
        print("\n### ANALYSIS SUMMARY ###\n")

        if 'model_performance' in self.results:
            print("Top 3 Models by F1 Score:")
            models_f1 = [(m, p['basic_metrics']['f1'])
                        for m, p in self.results['model_performance'].items()
                        if 'basic_metrics' in p]
            models_f1.sort(key=lambda x: x[1], reverse=True)
            for i, (model, f1) in enumerate(models_f1[:3], 1):
                print(f"  {i}. {model}: {f1:.3f}")

        if 'error_analysis' in self.results:
            print("\nError Analysis:")
            total_errors = sum(len(e['false_positives']) + len(e['false_negatives'])
                             for e in self.results['error_analysis']['error_categories'].values())
            print(f"  Total errors across all models: {total_errors}")
            print(f"  Difficult examples: {len(self.results['error_analysis']['difficult_examples'])}")

        if 'statistical_significance' in self.results:
            sig_tests = self.results['statistical_significance']['pairwise_tests']
            sig_count = sum(1 for t in sig_tests if t['significant_bonferroni'])
            print(f"\nStatistical Tests:")
            print(f"  Significant comparisons: {sig_count}/{len(sig_tests)}")

        print("\nOutputs Generated:")
        print("  - Comprehensive report: iclr_analysis_report.md")
        print("  - Visualizations: visualizations/")
        print("  - LaTeX tables: latex_tables/")
        print("  - Saved results: iclr_analysis_results.pkl")


def main():
    """Main execution function"""
    # Configuration
    config = ExperimentConfig(
        random_seed=42,
        n_bootstrap=10000,
        n_permutation=10000,
        cv_folds=5,
        test_size=0.2,
        confidence_level=0.95,
        significance_alpha=0.05,
        correction_method='bonferroni'
    )

    # Initialize analyzer
    excel_path = "E:/Project/OBJEX_dataset/OBJEX_dataset_labeling.xlsx"
    analyzer = ICLRLevelAnalyzer(excel_path, config)

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    print("\nICLR-quality analysis complete!")
    print("Check the generated reports and visualizations for detailed results.")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()