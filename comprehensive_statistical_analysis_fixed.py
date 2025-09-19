"""
Comprehensive statistical analysis for OBJEX dataset - FIXED VERSION
Fixed nested multiprocessing issue for server execution
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
import json
import warnings
from multiprocessing import Pool, cpu_count
import time
warnings.filterwarnings('ignore')

# Global variable for number of processes
N_JOBS = min(120, cpu_count() - 8)  # Leave some cores for system

print(f"Using {N_JOBS} parallel processes on {cpu_count()} available cores")

def load_complete_data():
    """Load the complete analysis results"""
    with open('complete_analysis_results_FIXED.json', 'r') as f:
        results = json.load(f)

    # Load raw data from Excel
    excel_file = 'OBJEX_dataset_labeling.xlsx'
    xl = pd.ExcelFile(excel_file)

    # Model name mapping
    model_sheet_map = {
        'gpt-4.1': ('similarity_gpt-4.1', 'extracted_gpt_4.1'),
        'claude-sonnet-4': ('similarity_claude-sonnet-4-2025', 'extracted_claude-sonnet-4'),
        'Qwen3-235B-A22B-FP8': ('similarity_Qwen3-235B-A22B-fp8-', 'extracted_Qwen3-235B-A22B-fp8-t'),
        'kimi-k2': ('similarity_moonshotaiKimi-K2-In', 'extracted_moonshotaiKimi-K2-Ins'),
        'deepseek-v3.1': ('similarity_deepseek-aiDeepSeek-', 'extracted_deepseek-aiDeepSeek-V'),
        'gemini-2.5-flash': ('similarity_gemini-2.5-flash', 'extracted_gemini-2.5-flash')
    }

    # Load labeling data
    df_labeling = pd.read_excel(xl, 'Labeling')

    data = {}

    for model, (sim_sheet, ext_sheet) in model_sheet_map.items():
        df_sim = pd.read_excel(xl, sim_sheet)
        df_ext = pd.read_excel(xl, ext_sheet)

        # Merge data
        df = pd.merge(
            df_sim[['base_prompt', 'similarity_score']].rename(columns={'similarity_score': 'similarity'}),
            df_ext[['base_prompt', 'source', 'extraction_confidence']].rename(columns={'extraction_confidence': 'confidence'}),
            on='base_prompt', how='inner'
        )

        # Add human labels
        df = pd.merge(
            df,
            df_labeling[['base_prompt', 'human_label']].rename(columns={'human_label': 'human_labeled'}),
            on='base_prompt', how='inner'
        )

        # Convert human labels to binary
        df['human_labeled'] = df['human_labeled'].map({
            'Low similarity': 0, 'No similarity': 0,
            'Moderate similarity': 1, 'High similarity': 1,
            1: 1, 0: 0
        })

        # Drop rows with missing labels
        df = df.dropna(subset=['human_labeled'])

        # Apply threshold
        df['predicted'] = (df['similarity'] >= 0.66).astype(int)
        df['actual'] = df['human_labeled'].astype(int)
        df['correct'] = (df['predicted'] == df['actual']).astype(int)

        data[model] = df

    return data, results

def bootstrap_batch(args):
    """Process a batch of bootstrap iterations"""
    correct1_arr, correct2_arr, seed_offset, n_iterations = args
    np.random.seed(seed_offset)
    n = len(correct1_arr)

    diffs = []
    for _ in range(n_iterations):
        idx = np.random.choice(n, n, replace=True)
        diff = correct1_arr[idx].mean() - correct2_arr[idx].mean()
        diffs.append(diff)

    return diffs

def paired_bootstrap_test_parallel(data1, data2, n_bootstrap=10000):
    """
    Parallel bootstrap test without nested pools
    """
    # Ensure same samples
    merged = pd.merge(
        data1[['base_prompt', 'correct']].rename(columns={'correct': 'correct1'}),
        data2[['base_prompt', 'correct']].rename(columns={'correct': 'correct2'}),
        on='base_prompt', how='inner'
    )

    observed_diff = merged['correct1'].mean() - merged['correct2'].mean()
    correct1_arr = merged['correct1'].values
    correct2_arr = merged['correct2'].values

    # Split bootstrap iterations into batches for parallel processing
    iterations_per_batch = max(100, n_bootstrap // N_JOBS)
    n_batches = (n_bootstrap + iterations_per_batch - 1) // iterations_per_batch

    # Create batch arguments
    batch_args = []
    for i in range(n_batches):
        n_iter = min(iterations_per_batch, n_bootstrap - i * iterations_per_batch)
        if n_iter > 0:
            batch_args.append((correct1_arr, correct2_arr, 42 + i * 1000, n_iter))

    # Process batches in parallel
    with Pool(N_JOBS) as pool:
        batch_results = pool.map(bootstrap_batch, batch_args)

    # Flatten results
    bootstrap_diffs = []
    for batch in batch_results:
        bootstrap_diffs.extend(batch)
    bootstrap_diffs = np.array(bootstrap_diffs[:n_bootstrap])  # Ensure exact count

    # Two-tailed p-value
    p_value = 2 * min(
        (bootstrap_diffs <= 0).mean() if observed_diff > 0 else (bootstrap_diffs >= 0).mean(),
        0.5
    )

    # 95% CI
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < 0.05
    }

def paired_bootstrap_test_sequential(data1, data2, n_bootstrap=10000, seed=42):
    """
    Sequential version of bootstrap test (fallback for nested calls)
    """
    np.random.seed(seed)

    # Ensure same samples
    merged = pd.merge(
        data1[['base_prompt', 'correct']].rename(columns={'correct': 'correct1'}),
        data2[['base_prompt', 'correct']].rename(columns={'correct': 'correct2'}),
        on='base_prompt', how='inner'
    )

    n = len(merged)
    observed_diff = merged['correct1'].mean() - merged['correct2'].mean()

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        sample = merged.iloc[idx]
        diff = sample['correct1'].mean() - sample['correct2'].mean()
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-tailed p-value
    p_value = 2 * min(
        (bootstrap_diffs <= 0).mean() if observed_diff > 0 else (bootstrap_diffs >= 0).mean(),
        0.5
    )

    # 95% CI
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < 0.05
    }

def process_single_comparison(args):
    """Process a single model pair comparison - uses sequential bootstrap"""
    model1, model2, data1, data2 = args

    # Use sequential bootstrap to avoid nested pools
    bootstrap_result = paired_bootstrap_test_sequential(data1, data2)

    # McNemar test
    merged = pd.merge(
        data1[['base_prompt', 'correct']].rename(columns={'correct': 'correct1'}),
        data2[['base_prompt', 'correct']].rename(columns={'correct': 'correct2'}),
        on='base_prompt', how='inner'
    )

    # Contingency table
    n00 = ((merged['correct1'] == 1) & (merged['correct2'] == 1)).sum()
    n01 = ((merged['correct1'] == 1) & (merged['correct2'] == 0)).sum()
    n10 = ((merged['correct1'] == 0) & (merged['correct2'] == 1)).sum()
    n11 = ((merged['correct1'] == 0) & (merged['correct2'] == 0)).sum()

    contingency = [[n00, n01], [n10, n11]]
    mcnemar_result = mcnemar(contingency, exact=True if (n01 + n10) < 25 else False)

    # Accuracies
    acc1 = data1['correct'].mean()
    acc2 = data2['correct'].mean()

    return {
        'model1': model1,
        'model2': model2,
        'acc1': acc1,
        'acc2': acc2,
        'delta': acc1 - acc2,
        'bootstrap_p': bootstrap_result['p_value'],
        'bootstrap_ci': [bootstrap_result['ci_lower'], bootstrap_result['ci_upper']],
        'mcnemar_p': mcnemar_result.pvalue,
        'mcnemar_stat': mcnemar_result.statistic
    }

def compute_all_pairwise_tests(data, use_parallel=True):
    """
    Compute all pairwise comparisons
    """
    models = ['claude-sonnet-4', 'kimi-k2', 'deepseek-v3.1',
              'gemini-2.5-flash', 'gpt-4.1', 'Qwen3-235B-A22B-FP8']

    # Prepare comparison pairs
    comparison_args = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            comparison_args.append((models[i], models[j], data[models[i]], data[models[j]]))

    print(f"Processing {len(comparison_args)} pairwise comparisons...")

    if use_parallel:
        # Process comparisons in parallel (each uses sequential bootstrap)
        with Pool(min(N_JOBS, len(comparison_args))) as pool:
            results = pool.map(process_single_comparison, comparison_args)
    else:
        # Fully sequential processing
        results = []
        for i, args in enumerate(comparison_args):
            print(f"  Processing comparison {i+1}/{len(comparison_args)}: {args[0]} vs {args[1]}")
            results.append(process_single_comparison(args))

    # Apply multiple comparison corrections
    bootstrap_pvals = [r['bootstrap_p'] for r in results]
    mcnemar_pvals = [r['mcnemar_p'] for r in results]

    # Bonferroni
    _, bonf_bootstrap = multipletests(bootstrap_pvals, method='bonferroni')[:2]
    _, bonf_mcnemar = multipletests(mcnemar_pvals, method='bonferroni')[:2]

    # Holm
    _, holm_bootstrap = multipletests(bootstrap_pvals, method='holm')[:2]
    _, holm_mcnemar = multipletests(mcnemar_pvals, method='holm')[:2]

    # FDR (Benjamini-Hochberg)
    _, fdr_bootstrap = multipletests(bootstrap_pvals, method='fdr_bh')[:2]
    _, fdr_mcnemar = multipletests(mcnemar_pvals, method='fdr_bh')[:2]

    for i, r in enumerate(results):
        r['bootstrap_p_bonf'] = bonf_bootstrap[i]
        r['bootstrap_p_holm'] = holm_bootstrap[i]
        r['bootstrap_p_fdr'] = fdr_bootstrap[i]
        r['mcnemar_p_bonf'] = bonf_mcnemar[i]
        r['mcnemar_p_holm'] = holm_mcnemar[i]
        r['mcnemar_p_fdr'] = fdr_mcnemar[i]

    return results

def process_dataset_comparison(args):
    """Process a single dataset comparison"""
    dataset, model1, model2, data1, data2 = args

    # Filter by dataset
    data1_ds = data1[data1['source'] == dataset]
    data2_ds = data2[data2['source'] == dataset]

    if len(data1_ds) > 0 and len(data2_ds) > 0:
        # Use sequential bootstrap for nested calls
        bootstrap_result = paired_bootstrap_test_sequential(data1_ds, data2_ds, n_bootstrap=5000)

        return {
            'dataset': dataset,
            'models': f"{model1} vs {model2}",
            'acc1': data1_ds['correct'].mean(),
            'acc2': data2_ds['correct'].mean(),
            'delta': data1_ds['correct'].mean() - data2_ds['correct'].mean(),
            'p_value': bootstrap_result['p_value'],
            'ci': [bootstrap_result['ci_lower'], bootstrap_result['ci_upper']]
        }
    return None

def compute_dataset_wise_tests(data, use_parallel=True):
    """
    Compute significance tests per dataset
    """
    datasets = ['SafeMTData_1K', 'MHJ', 'SafeMTData_Attack600', 'CoSafe']
    models = ['claude-sonnet-4', 'kimi-k2', 'deepseek-v3.1',
              'gemini-2.5-flash', 'gpt-4.1', 'Qwen3-235B-A22B-FP8']

    # Prepare all comparison arguments
    comparison_args = []
    for dataset in datasets:
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                comparison_args.append((dataset, models[i], models[j],
                                      data[models[i]], data[models[j]]))

    print(f"Processing {len(comparison_args)} dataset-wise comparisons...")

    if use_parallel:
        # Process comparisons in parallel
        with Pool(N_JOBS) as pool:
            all_results = pool.map(process_dataset_comparison, comparison_args)
    else:
        # Sequential processing
        all_results = []
        for args in comparison_args:
            all_results.append(process_dataset_comparison(args))

    # Organize results by dataset
    dataset_results = {ds: [] for ds in datasets}
    for result in all_results:
        if result is not None:
            dataset_results[result['dataset']].append(result)

    return dataset_results

def create_latex_table(results):
    """Create LaTeX table with all statistical tests"""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Comprehensive statistical significance tests for model comparisons. P-values from paired bootstrap (10,000 resamples) and McNemar's test, with multiple comparison adjustments.}
\label{tab:significance-comprehensive}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\multirow{2}{*}{Comparison} & \multicolumn{2}{c}{Accuracy} & \multirow{2}{*}{$\Delta$} & \multicolumn{2}{c}{Raw p-values} & \multicolumn{3}{c}{Adjusted p-values} \\
\cmidrule(lr){2-3} \cmidrule(lr){5-6} \cmidrule(lr){7-9}
& Model A & Model B & & Bootstrap & McNemar & Bonferroni & Holm & FDR \\
\midrule
"""

    # Sort by delta magnitude
    results_sorted = sorted(results, key=lambda x: abs(x['delta']), reverse=True)

    for r in results_sorted:
        # Format model names
        model1_short = r['model1'].replace('claude-sonnet-4', 'Claude-4')
        model1_short = model1_short.replace('deepseek-v3.1', 'DeepSeek')
        model1_short = model1_short.replace('kimi-k2', 'Kimi')
        model1_short = model1_short.replace('gemini-2.5-flash', 'Gemini')
        model1_short = model1_short.replace('gpt-4.1', 'GPT-4.1')
        model1_short = model1_short.replace('Qwen3-235B-A22B-FP8', 'Qwen')

        model2_short = r['model2'].replace('claude-sonnet-4', 'Claude-4')
        model2_short = model2_short.replace('deepseek-v3.1', 'DeepSeek')
        model2_short = model2_short.replace('kimi-k2', 'Kimi')
        model2_short = model2_short.replace('gemini-2.5-flash', 'Gemini')
        model2_short = model2_short.replace('gpt-4.1', 'GPT-4.1')
        model2_short = model2_short.replace('Qwen3-235B-A22B-FP8', 'Qwen')

        # Significance markers
        sig_marker = ""
        if r['bootstrap_p_fdr'] < 0.001:
            sig_marker = "***"
        elif r['bootstrap_p_fdr'] < 0.01:
            sig_marker = "**"
        elif r['bootstrap_p_fdr'] < 0.05:
            sig_marker = "*"

        latex += f"{model1_short} vs {model2_short} & "
        latex += f"{r['acc1']:.3f} & {r['acc2']:.3f} & "
        latex += f"{r['delta']:+.3f}{sig_marker} & "
        latex += f"{r['bootstrap_p']:.4f} & {r['mcnemar_p']:.4f} & "
        latex += f"{r['bootstrap_p_bonf']:.4f} & {r['bootstrap_p_holm']:.4f} & {r['bootstrap_p_fdr']:.4f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
}
\vspace{2mm}
\footnotesize
\textit{Note:} $\Delta$ = Accuracy(Model A) - Accuracy(Model B). Significance: *** p < 0.001, ** p < 0.01, * p < 0.05 (FDR-adjusted).
Bootstrap test uses 10,000 paired resamples. McNemar's test uses exact calculation when n < 25.
Multiple comparison adjustments: Bonferroni (most conservative), Holm-Bonferroni, and FDR (Benjamini-Hochberg).
\end{table}
"""

    return latex

def create_dataset_latex_table(dataset_results):
    """Create LaTeX table for dataset-wise comparisons"""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Dataset-wise significance tests for top model pairs (paired bootstrap, 5,000 resamples).}
\label{tab:dataset-significance}
\begin{tabular}{llcccc}
\toprule
Dataset & Comparison & Acc(A) & Acc(B) & $\Delta$ & p-value \\
\midrule
"""

    for dataset in ['MHJ', 'SafeMTData_1K', 'SafeMTData_Attack600', 'CoSafe']:
        comparisons = dataset_results[dataset]

        # Show top 3 comparisons by delta magnitude
        top_comparisons = sorted(comparisons, key=lambda x: abs(x['delta']), reverse=True)[:3]

        for i, comp in enumerate(top_comparisons):
            if i == 0:
                latex += f"\\multirow{{3}}{{*}}{{{dataset}}} & "
            else:
                latex += " & "

            # Shorten names
            models_short = comp['models'].replace('claude-sonnet-4', 'Claude')
            models_short = models_short.replace('deepseek-v3.1', 'DeepSeek')
            models_short = models_short.replace('kimi-k2', 'Kimi')
            models_short = models_short.replace('gemini-2.5-flash', 'Gemini')
            models_short = models_short.replace('gpt-4.1', 'GPT-4.1')
            models_short = models_short.replace('Qwen3-235B-A22B-FP8', 'Qwen')

            sig = "*" if comp['p_value'] < 0.05 else ""

            latex += f"{models_short} & "
            latex += f"{comp['acc1']:.3f} & {comp['acc2']:.3f} & "
            latex += f"{comp['delta']:+.3f}{sig} & {comp['p_value']:.4f} \\\\\n"

        if dataset != 'CoSafe':
            latex += "\\midrule\n"

    latex += r"""
\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize
\textit{Note:} Shows top 3 comparisons per dataset by absolute delta. * indicates p < 0.05.
\end{table}
"""

    return latex

def save_raw_predictions(data):
    """Save raw predictions for all models to CSV for reproducibility"""
    # Create unified dataframe with all model predictions
    all_predictions = []

    for model in data.keys():
        df = data[model].copy()
        df['model'] = model
        all_predictions.append(df[['base_prompt', 'source', 'model',
                                   'similarity', 'confidence', 'predicted',
                                   'actual', 'correct']])

    # Combine and pivot
    combined = pd.concat(all_predictions)

    # Save long format
    combined.to_csv('raw_predictions_all_models.csv', index=False)

    # Also create wide format for paired analysis
    pivot = combined.pivot_table(
        index=['base_prompt', 'source', 'actual'],
        columns='model',
        values=['similarity', 'predicted', 'correct']
    )
    pivot.to_csv('paired_predictions_wide.csv')

    print(f"Saved raw predictions: {len(combined)} total predictions")
    print(f"Models: {combined['model'].nunique()}")
    print(f"Unique prompts: {combined['base_prompt'].nunique()}")

def main():
    start_time = time.time()

    print(f"Starting statistical analysis...")
    print(f"Using {N_JOBS} processes for parallel sections")
    print("="*60)

    print("Loading data...")
    data, original_results = load_complete_data()

    print("\nComputing comprehensive pairwise tests...")
    pairwise_start = time.time()
    pairwise_results = compute_all_pairwise_tests(data, use_parallel=True)
    print(f"  Completed in {time.time() - pairwise_start:.1f} seconds")

    print("\nComputing dataset-wise tests...")
    dataset_start = time.time()
    dataset_results = compute_dataset_wise_tests(data, use_parallel=True)
    print(f"  Completed in {time.time() - dataset_start:.1f} seconds")

    print("\nGenerating LaTeX tables...")
    latex_main = create_latex_table(pairwise_results)
    latex_dataset = create_dataset_latex_table(dataset_results)

    # Save LaTeX
    with open('statistical_tables.tex', 'w') as f:
        f.write("% Main comprehensive significance table\n")
        f.write(latex_main)
        f.write("\n\n% Dataset-wise significance table\n")
        f.write(latex_dataset)

    # Save JSON results
    results_json = {
        'pairwise_comparisons': pairwise_results,
        'dataset_comparisons': dataset_results
    }

    with open('comprehensive_statistical_results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    print("\nSaving raw predictions for reproducibility...")
    save_raw_predictions(data)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Report significant findings
    print("\nSignificant differences (FDR-adjusted p < 0.05):")
    sig_results = [r for r in pairwise_results if r['bootstrap_p_fdr'] < 0.05]

    for r in sorted(sig_results, key=lambda x: x['bootstrap_p_fdr']):
        print(f"  {r['model1']} vs {r['model2']}: Δ={r['delta']:+.3f}, p_fdr={r['bootstrap_p_fdr']:.4f}")

    if not sig_results:
        print("  No significant differences after FDR correction")

    print(f"\nFiles created:")
    print("  - statistical_tables.tex (LaTeX tables for paper)")
    print("  - comprehensive_statistical_results.json (all test results)")
    print("  - raw_predictions_all_models.csv (raw data for reproducibility)")
    print("  - paired_predictions_wide.csv (wide format for paired analysis)")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds")

if __name__ == "__main__":
    main()