import pandas as pd
import numpy as np
from scipy import stats
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    bootstrap_means = []
    n = len(data)

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha/2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))

    return lower, upper

def calculate_ece(confidences: np.ndarray, predictions: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = predictions[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def calculate_aurc(confidences: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate Area Under Risk-Coverage curve."""
    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidences)
    sorted_predictions = predictions[sorted_indices]

    # Calculate cumulative risk at each coverage level
    n = len(sorted_predictions)
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(1 - sorted_predictions) / np.arange(1, n + 1)

    # Calculate area under curve using trapezoidal rule
    aurc = np.trapz(risks, coverages)
    return aurc

def analyze_model(excel_file: str, extraction_sheet: str, similarity_sheet: str, model_name: str):
    """Analyze a single model."""
    print(f"Analyzing {model_name}...")

    # Load data
    xls = pd.ExcelFile(excel_file)
    ext_df = pd.read_excel(xls, extraction_sheet)
    sim_df = pd.read_excel(xls, similarity_sheet)

    # Merge data
    merged_df = ext_df.merge(sim_df, left_index=True, right_index=True, suffixes=('_ext', '_sim'))

    # Fixed threshold from paper
    tau_star = 0.61

    # Basic metrics
    valid_mask = merged_df['similarity_score'].notna()
    valid_df = merged_df[valid_mask]

    n_total = len(merged_df)
    n_scored = len(valid_df)
    coverage = n_scored / n_total

    # Overall accuracy
    predictions = (valid_df['similarity_score'] >= tau_star).astype(int)
    accuracy = np.mean(predictions)

    # Bootstrap 95% CI for accuracy
    acc_lower, acc_upper = bootstrap_ci(predictions, n_bootstrap=1000, confidence=0.95)

    # Mean similarity score and confidence
    mean_similarity = np.mean(valid_df['similarity_score'])
    mean_confidence = np.mean(merged_df['extraction_confidence'])

    # Extraction error rate
    extraction_errors = merged_df['extraction_error'].notna().sum()
    error_rate = (extraction_errors / n_total) * 100

    print(f"  Basic metrics calculated")

    # Per-source analysis
    per_source_results = {}
    accuracies = []

    for source in merged_df['source'].unique():
        source_df = merged_df[merged_df['source'] == source]
        source_valid = source_df[source_df['similarity_score'].notna()]

        if len(source_valid) > 0:
            source_predictions = (source_valid['similarity_score'] >= tau_star).astype(int)
            source_accuracy = np.mean(source_predictions)
            accuracies.append(source_accuracy)
            per_source_results[source] = {
                'accuracy': source_accuracy,
                'n_scored': len(source_valid)
            }
        else:
            per_source_results[source] = {
                'accuracy': np.nan,
                'n_scored': 0
            }

    # Spread calculation
    valid_accuracies = [acc for acc in accuracies if not np.isnan(acc)]
    spread = max(valid_accuracies) - min(valid_accuracies) if valid_accuracies else np.nan

    print(f"  Per-source analysis completed")

    # Metacognition metrics
    if len(valid_df) > 0:
        confidences = valid_df['extraction_confidence'].values

        # ECE
        ece = calculate_ece(confidences, predictions, n_bins=10)

        # Brier Score
        brier = np.mean((confidences - predictions) ** 2)

        # Wrong@High-Conf metrics
        def wrong_at_conf(conf_threshold):
            high_conf_mask = confidences >= conf_threshold
            if high_conf_mask.sum() == 0:
                return np.nan
            return (1 - predictions[high_conf_mask]).mean()

        wrong_080 = wrong_at_conf(0.80)
        wrong_090 = wrong_at_conf(0.90)
        wrong_095 = wrong_at_conf(0.95)

        # AURC
        aurc = calculate_aurc(confidences, predictions)

        print(f"  Metacognition metrics calculated")
    else:
        ece = brier = wrong_080 = wrong_090 = wrong_095 = aurc = np.nan

    # Robustness check - categorical mapping
    categorical_valid = merged_df[merged_df['similarity_category'].notna()]
    if len(categorical_valid) > 0:
        categorical_predictions = categorical_valid['similarity_category'].isin(['Exact match', 'High similarity', 'High']).astype(int)
        categorical_accuracy = np.mean(categorical_predictions)

        # Per-source robustness
        per_source_robustness = {}
        for source in categorical_valid['source'].unique():
            source_cat = categorical_valid[categorical_valid['source'] == source]
            if len(source_cat) > 0:
                source_cat_pred = source_cat['similarity_category'].isin(['Exact match', 'High similarity', 'High']).astype(int)
                per_source_robustness[source] = np.mean(source_cat_pred)
            else:
                per_source_robustness[source] = np.nan
    else:
        categorical_accuracy = np.nan
        per_source_robustness = {}

    print(f"  Robustness check completed")

    # Compile results
    results = {
        'model': model_name,

        # Basic metrics
        'overall_accuracy': float(accuracy),
        'accuracy_95ci_lower': float(acc_lower),
        'accuracy_95ci_upper': float(acc_upper),
        'coverage': float(coverage),
        'n_scored': int(n_scored),
        'n_total': int(n_total),
        'mean_similarity_score': float(mean_similarity),
        'mean_self_confidence': float(mean_confidence),
        'extraction_error_rate_pct': float(error_rate),

        # Per-source analysis
        'per_source': per_source_results,
        'spread': float(spread) if not np.isnan(spread) else None,

        # Metacognition metrics
        'ece': float(ece) if not np.isnan(ece) else None,
        'brier_score': float(brier) if not np.isnan(brier) else None,
        'wrong_at_080': float(wrong_080) if not np.isnan(wrong_080) else None,
        'wrong_at_090': float(wrong_090) if not np.isnan(wrong_090) else None,
        'wrong_at_095': float(wrong_095) if not np.isnan(wrong_095) else None,
        'aurc': float(aurc) if not np.isnan(aurc) else None,

        # Robustness
        'categorical_accuracy': float(categorical_accuracy) if not np.isnan(categorical_accuracy) else None,
        'per_source_robustness': per_source_robustness
    }

    return results

def main():
    excel_file = "E:\\Project\\OBJEX_dataset\\OBJEX_dataset_new.xlsx"

    # New models to analyze
    models_to_analyze = [
        {
            'name': 'kimi-k2',
            'extraction_sheet': 'extracted_moonshotaiKimi-K2-Ins',
            'similarity_sheet': 'similarity_moonshotaiKimi-K2-In'
        },
        {
            'name': 'deepseek-v3.1',
            'extraction_sheet': 'extracted_deepseek-aiDeepSeek-V',
            'similarity_sheet': 'similarity_deepseek-aiDeepSeek-'
        },
        {
            'name': 'gemini-2.5-flash',
            'extraction_sheet': 'extracted_gemini-2.5-flash',
            'similarity_sheet': 'similarity_gemini-2.5-flash'
        }
    ]

    print("=== OBJEX Analysis for New Models ===\\n")

    all_results = {}

    for model_info in models_to_analyze:
        try:
            result = analyze_model(
                excel_file,
                model_info['extraction_sheet'],
                model_info['similarity_sheet'],
                model_info['name']
            )
            all_results[model_info['name']] = result
            print(f"[OK] {model_info['name']} completed\\n")

        except Exception as e:
            print(f"[ERROR] Error analyzing {model_info['name']}: {e}\\n")

    # Print results
    print("\\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)

    for model_name, data in all_results.items():
        print(f"\\n{'='*20} {model_name.upper()} {'='*20}")

        # Basic metrics
        print(f"\\n[BASIC METRICS]:")
        print(f"Overall Accuracy: {data['overall_accuracy']:.3f}")
        print(f"95% CI: ({data['accuracy_95ci_lower']:.3f}, {data['accuracy_95ci_upper']:.3f})")
        print(f"Coverage: {data['coverage']:.3f} ({data['n_scored']}/{data['n_total']})")
        print(f"Mean Similarity Score: {data['mean_similarity_score']:.3f}")
        print(f"Mean Self-Confidence: {data['mean_self_confidence']:.3f}")
        print(f"Extraction Error Rate: {data['extraction_error_rate_pct']:.1f}%")

        # Per-source metrics
        print(f"\\n[PER-SOURCE ANALYSIS]:")
        for source, metrics in data['per_source'].items():
            if not np.isnan(metrics['accuracy']):
                print(f"{source}: {metrics['accuracy']:.3f} (n={metrics['n_scored']})")
        if data['spread'] is not None:
            print(f"Spread (max-min): {data['spread']:.3f}")

        # Calibration metrics
        print(f"\\n[METACOGNITION METRICS]:")
        metrics = ['ece', 'brier_score', 'wrong_at_080', 'wrong_at_090', 'wrong_at_095', 'aurc']
        labels = ['ECE (10 bins)', 'Brier Score', 'Wrong@0.80', 'Wrong@0.90', 'Wrong@0.95', 'AURC']
        for metric, label in zip(metrics, labels):
            val = data[metric]
            print(f"{label}: {val:.3f}" if val is not None else f"{label}: N/A")

        # Robustness
        print(f"\\n[ROBUSTNESS CHECK]:")
        if data['categorical_accuracy'] is not None:
            print(f"Categorical Accuracy: {data['categorical_accuracy']:.3f}")
        print(f"Per-source robustness:")
        for source, acc in data['per_source_robustness'].items():
            if not np.isnan(acc):
                print(f"  {source}: {acc:.3f}")

    # Save results
    with open('new_models_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\\n[SAVE] Results saved to 'new_models_analysis_results.json'")
    print("Analysis complete!")

    return all_results

if __name__ == "__main__":
    results = main()