"""
Statistical significance testing for OBJEX dataset analysis
Addresses reviewer concern about limited statistical testing
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the prediction data for all models"""
    excel_file = 'OBJEX_dataset_labeling.xlsx'
    xl = pd.ExcelFile(excel_file)

    # Model name mapping
    model_sheet_map = {
        'gpt-4.1': ('similarity_gpt-4.1', 'extracted_gpt_4.1'),
        'claude-sonnet-4': ('similarity_claude-sonnet-4-2025', 'extracted_claude-sonnet-4'),
        'Qwen3-235B-A22B-FP8': ('similarity_Qwen3-235B-A22B-fp8-', 'extracted_Qwen3-235B-A22B-fp8-t'),
        'kimi-k2': ('similarity_moonshotaiKimi-K2-I', 'extracted_moonshotaiKimi-K2-Ins'),
        'deepseek-v3.1': ('similarity_deepseek-aiDeepSeek', 'extracted_deepseek-aiDeepSeek-V'),
        'gemini-2.5-flash': ('similarity_gemini-2.5-flash', 'extracted_gemini-2.5-flash')
    }

    # Load labeling data
    df_labeling = pd.read_excel(xl, 'Labeling')

    data = {}

    for model, (sim_sheet, ext_sheet) in model_sheet_map.items():
        df_sim = pd.read_excel(xl, sim_sheet)
        df_ext = pd.read_excel(xl, ext_sheet)

        # Merge similarity and extraction data
        df = pd.merge(df_sim[['base_prompt', 'similarity_score']].rename(columns={'similarity_score': 'similarity'}),
                      df_ext[['base_prompt', 'source', 'extraction_confidence']].rename(columns={'extraction_confidence': 'model_confidence'}),
                      on='base_prompt', how='inner')

        # Merge with human labels
        df = pd.merge(df, df_labeling[['base_prompt', 'human_label']].rename(columns={'human_label': 'human_labeled'}),
                      on='base_prompt', how='inner')

        # Apply threshold
        df['predicted'] = (df['similarity'] >= 0.66).astype(int)
        df['actual'] = df['human_labeled']

        data[model] = df

    return data

def mcnemar_test_pairwise(data):
    """
    Perform McNemar's test for pairwise model comparisons
    This tests if two models have significantly different error rates on the same data
    """
    models = list(data.keys())
    n_models = len(models)

    # Create results matrix
    p_values = np.ones((n_models, n_models))
    odds_ratios = np.ones((n_models, n_models))

    print("\n=== McNemar's Test for Pairwise Model Comparisons ===")
    print("H0: Two models have the same error rate")
    print("H1: Two models have different error rates\n")

    results = []

    for i in range(n_models):
        for j in range(i+1, n_models):
            model1 = models[i]
            model2 = models[j]

            # Align predictions on same samples
            df1 = data[model1]
            df2 = data[model2]

            # Merge on base_prompt to ensure same samples
            merged = pd.merge(
                df1[['base_prompt', 'predicted', 'actual']],
                df2[['base_prompt', 'predicted']],
                on='base_prompt',
                suffixes=('_1', '_2')
            )

            # Create contingency table for McNemar's test
            # Both correct
            n00 = ((merged['predicted_1'] == merged['actual']) &
                   (merged['predicted_2'] == merged['actual'])).sum()
            # Model1 correct, Model2 wrong
            n01 = ((merged['predicted_1'] == merged['actual']) &
                   (merged['predicted_2'] != merged['actual'])).sum()
            # Model1 wrong, Model2 correct
            n10 = ((merged['predicted_1'] != merged['actual']) &
                   (merged['predicted_2'] == merged['actual'])).sum()
            # Both wrong
            n11 = ((merged['predicted_1'] != merged['actual']) &
                   (merged['predicted_2'] != merged['actual'])).sum()

            # McNemar's test focuses on discordant pairs (n01 and n10)
            contingency_table = [[n00, n01], [n10, n11]]

            # Perform McNemar's test
            result = mcnemar(contingency_table, exact=True if (n01 + n10) < 25 else False)

            p_values[i, j] = result.pvalue
            p_values[j, i] = result.pvalue

            # Calculate odds ratio
            if n10 > 0:
                or_value = n01 / n10
            else:
                or_value = np.inf if n01 > 0 else 1.0

            odds_ratios[i, j] = or_value
            odds_ratios[j, i] = 1/or_value if or_value != np.inf else 0

            results.append({
                'model1': model1,
                'model2': model2,
                'n01': n01,
                'n10': n10,
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'odds_ratio': or_value
            })

    return results, p_values

def cochrans_q_test(data):
    """
    Cochran's Q test for comparing multiple models on same binary outcomes
    This is an extension of McNemar's test for k>2 models
    """
    models = list(data.keys())

    # Create aligned dataset
    base_prompts = data[models[0]]['base_prompt'].values

    # Create matrix of predictions
    predictions = []
    actuals = None

    for model in models:
        df = data[model]
        df_sorted = df.set_index('base_prompt').loc[base_prompts]
        predictions.append((df_sorted['predicted'] == df_sorted['actual']).astype(int).values)
        if actuals is None:
            actuals = df_sorted['actual'].values

    predictions = np.array(predictions).T  # Shape: (n_samples, n_models)

    # Cochran's Q statistic
    k = predictions.shape[1]  # number of models
    n = predictions.shape[0]  # number of samples

    # Row totals
    row_totals = predictions.sum(axis=1)

    # Column totals
    col_totals = predictions.sum(axis=0)

    # Grand total
    grand_total = predictions.sum()

    # Cochran's Q statistic
    numerator = k * (k - 1) * np.sum((col_totals - grand_total/k)**2)
    denominator = k * grand_total - np.sum(row_totals**2)

    if denominator == 0:
        Q = 0
    else:
        Q = numerator / denominator

    # Q follows chi-square distribution with k-1 degrees of freedom
    df = k - 1
    p_value = 1 - stats.chi2.cdf(Q, df)

    print("\n=== Cochran's Q Test ===")
    print(f"Testing if all {k} models have equal accuracy")
    print(f"Q statistic: {Q:.4f}")
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Significant difference among models (p < 0.05)")
    else:
        print("Result: No significant difference among models (p >= 0.05)")

    return Q, p_value

def friedman_test_with_confidence(data):
    """
    Friedman test for comparing models across different datasets
    Non-parametric alternative to repeated measures ANOVA
    """
    models = list(data.keys())
    datasets = ['SafeMTData_1K', 'MHJ', 'SafeMTData_Attack600', 'CoSafe']

    # Calculate accuracy for each model-dataset combination
    accuracy_matrix = []

    for dataset in datasets:
        dataset_accuracies = []
        for model in models:
            df = data[model]
            df_dataset = df[df['source'] == dataset]
            if len(df_dataset) > 0:
                acc = (df_dataset['predicted'] == df_dataset['actual']).mean()
            else:
                acc = 0
            dataset_accuracies.append(acc)
        accuracy_matrix.append(dataset_accuracies)

    accuracy_matrix = np.array(accuracy_matrix)

    # Perform Friedman test
    statistic, p_value = stats.friedmanchisquare(*accuracy_matrix.T)

    print("\n=== Friedman Test ===")
    print("Testing if model rankings are consistent across datasets")
    print(f"Chi-square statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Significant difference in model rankings (p < 0.05)")
        print("Post-hoc analysis recommended (e.g., Nemenyi test)")
    else:
        print("Result: No significant difference in model rankings (p >= 0.05)")

    # Calculate average ranks
    ranks = np.array([stats.rankdata(-accuracy_matrix[i]) for i in range(len(datasets))])
    mean_ranks = ranks.mean(axis=0)

    print("\nMean ranks across datasets (lower is better):")
    for model, rank in zip(models, mean_ranks):
        print(f"  {model}: {rank:.2f}")

    return statistic, p_value, mean_ranks

def bonferroni_correction(p_values_list):
    """
    Apply Bonferroni correction for multiple comparisons
    """
    p_values = [r['p_value'] for r in p_values_list]

    # Bonferroni correction
    rejected_bonf, p_adjusted_bonf = multipletests(p_values, method='bonferroni')[:2]

    # Holm-Bonferroni (less conservative)
    rejected_holm, p_adjusted_holm = multipletests(p_values, method='holm')[:2]

    # FDR correction (Benjamini-Hochberg)
    rejected_fdr, p_adjusted_fdr = multipletests(p_values, method='fdr_bh')[:2]

    print("\n=== Multiple Comparison Corrections ===")
    print(f"Number of comparisons: {len(p_values)}")
    print(f"Alpha level: 0.05")

    for i, result in enumerate(p_values_list):
        result['p_bonferroni'] = p_adjusted_bonf[i]
        result['p_holm'] = p_adjusted_holm[i]
        result['p_fdr'] = p_adjusted_fdr[i]
        result['sig_bonferroni'] = rejected_bonf[i]
        result['sig_holm'] = rejected_holm[i]
        result['sig_fdr'] = rejected_fdr[i]

    return p_values_list

def bootstrap_confidence_intervals_difference(data, n_bootstrap=10000):
    """
    Bootstrap confidence intervals for pairwise accuracy differences
    """
    models = list(data.keys())

    print("\n=== Bootstrap CIs for Pairwise Accuracy Differences ===")
    print(f"Number of bootstrap samples: {n_bootstrap}")

    results = []

    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]

            df1 = data[model1]
            df2 = data[model2]

            # Merge to ensure same samples
            merged = pd.merge(
                df1[['base_prompt', 'predicted', 'actual']],
                df2[['base_prompt', 'predicted', 'actual']],
                on='base_prompt',
                suffixes=('_1', '_2')
            )

            # Bootstrap
            differences = []
            n_samples = len(merged)

            for _ in range(n_bootstrap):
                idx = np.random.choice(n_samples, n_samples, replace=True)
                sample = merged.iloc[idx]

                acc1 = (sample['predicted_1'] == sample['actual_1']).mean()
                acc2 = (sample['predicted_2'] == sample['actual_2']).mean()
                differences.append(acc1 - acc2)

            differences = np.array(differences)
            ci_lower = np.percentile(differences, 2.5)
            ci_upper = np.percentile(differences, 97.5)
            mean_diff = differences.mean()

            # Check if CI includes 0
            significant = (ci_lower > 0) or (ci_upper < 0)

            results.append({
                'model1': model1,
                'model2': model2,
                'mean_difference': mean_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': significant
            })

    return results

def main():
    print("Loading data...")
    data = load_data()

    # 1. Cochran's Q test (omnibus test)
    Q, p_cochran = cochrans_q_test(data)

    # 2. Pairwise McNemar tests
    mcnemar_results, p_matrix = mcnemar_test_pairwise(data)

    # 3. Multiple comparison corrections
    mcnemar_corrected = bonferroni_correction(mcnemar_results)

    # 4. Friedman test across datasets
    friedman_stat, p_friedman, mean_ranks = friedman_test_with_confidence(data)

    # 5. Bootstrap CIs for differences
    bootstrap_results = bootstrap_confidence_intervals_difference(data, n_bootstrap=10000)

    # Save results
    results = {
        'cochrans_q': {
            'statistic': Q,
            'p_value': p_cochran,
            'significant': p_cochran < 0.05
        },
        'friedman': {
            'statistic': friedman_stat,
            'p_value': p_friedman,
            'mean_ranks': {m: float(r) for m, r in zip(data.keys(), mean_ranks)}
        },
        'pairwise_comparisons': []
    }

    # Combine McNemar and bootstrap results
    for mc, bs in zip(mcnemar_corrected, bootstrap_results):
        results['pairwise_comparisons'].append({
            'models': f"{mc['model1']} vs {mc['model2']}",
            'mcnemar_p': mc['p_value'],
            'mcnemar_p_bonferroni': mc['p_bonferroni'],
            'mcnemar_p_holm': mc['p_holm'],
            'mcnemar_p_fdr': mc['p_fdr'],
            'significant_bonferroni': mc['sig_bonferroni'],
            'significant_holm': mc['sig_holm'],
            'significant_fdr': mc['sig_fdr'],
            'accuracy_difference': bs['mean_difference'],
            'difference_ci': [bs['ci_lower'], bs['ci_upper']],
            'difference_significant': bs['significant']
        })

    # Save to JSON
    with open('statistical_significance_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF STATISTICAL TESTS")
    print("="*60)

    print(f"\n1. Cochran's Q Test: p = {p_cochran:.6f}")
    print(f"   Conclusion: {'Models differ significantly' if p_cochran < 0.05 else 'No significant difference'}")

    print(f"\n2. Friedman Test: p = {p_friedman:.6f}")
    print(f"   Conclusion: {'Rankings differ across datasets' if p_friedman < 0.05 else 'Consistent rankings'}")

    print("\n3. Significant Pairwise Differences (Bonferroni-corrected):")
    sig_pairs = [r for r in results['pairwise_comparisons'] if r['significant_bonferroni']]
    if sig_pairs:
        for pair in sig_pairs:
            print(f"   {pair['models']}: Δ = {pair['accuracy_difference']:.3f}, "
                  f"95% CI [{pair['difference_ci'][0]:.3f}, {pair['difference_ci'][1]:.3f}], "
                  f"p_adj = {pair['mcnemar_p_bonferroni']:.4f}")
    else:
        print("   No significant differences after Bonferroni correction")

    print("\n4. Significant Pairwise Differences (FDR-corrected):")
    sig_pairs_fdr = [r for r in results['pairwise_comparisons'] if r['significant_fdr']]
    if sig_pairs_fdr:
        for pair in sig_pairs_fdr:
            print(f"   {pair['models']}: Δ = {pair['accuracy_difference']:.3f}, "
                  f"p_fdr = {pair['mcnemar_p_fdr']:.4f}")
    else:
        print("   No significant differences after FDR correction")

    print(f"\nResults saved to: statistical_significance_results.json")
    print("\nThese tests address the reviewer's concern by providing:")
    print("- Omnibus test (Cochran's Q) for overall model differences")
    print("- Pairwise comparisons with McNemar's test")
    print("- Multiple comparison corrections (Bonferroni, Holm, FDR)")
    print("- Bootstrap CIs for accuracy differences")
    print("- Friedman test for consistency across datasets")

if __name__ == "__main__":
    main()