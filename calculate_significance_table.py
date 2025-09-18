import pandas as pd
import numpy as np

print("="*70)
print("Calculating Significance Tests for All 6 Models (τ=0.66)")
print("="*70)

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

threshold = 0.66

# Model configurations
models = [
    ('extracted_gpt_4.1', 'similarity_gpt-4.1', 'gpt-4.1'),
    ('extracted_claude-sonnet-4', 'similarity_claude-sonnet-4-2025', 'claude-sonnet-4'),
    ('extracted_Qwen3-235B-A22B-fp8-t', 'similarity_Qwen3-235B-A22B-fp8-', 'Qwen3-235B-A22B-FP8'),
    ('extracted_moonshotaiKimi-K2-Ins', 'similarity_moonshotaiKimi-K2-In', 'kimi-k2'),
    ('extracted_deepseek-aiDeepSeek-V', 'similarity_deepseek-aiDeepSeek-', 'deepseek-v3.1'),
    ('extracted_gemini-2.5-flash', 'similarity_gemini-2.5-flash', 'gemini-2.5-flash')
]

# Load all model data
model_data = {}
model_accs = {}

for ext_sheet, sim_sheet, model_name in models:
    df_e = pd.read_excel(xl, ext_sheet, usecols=['base_prompt'])
    df_s = pd.read_excel(xl, sim_sheet, usecols=['base_prompt', 'similarity_score'])
    merged = pd.merge(df_e, df_s, on='base_prompt', how='inner')

    model_data[model_name] = merged
    model_accs[model_name] = (merged['similarity_score'] >= threshold).mean()

# Calculate pairwise comparisons
print("\nModel Accuracies (τ=0.66):")
print("-"*40)
for model, acc in sorted(model_accs.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:25s}: {acc:.4f}")

print("\n" + "="*70)
print("Pairwise Comparisons with Bootstrap CI")
print("="*70)

comparisons = []
model_list = list(model_data.keys())

for i, model1 in enumerate(model_list):
    for model2 in model_list[i+1:]:
        # Align by base_prompt
        merged = pd.merge(
            model_data[model1][['base_prompt', 'similarity_score']],
            model_data[model2][['base_prompt', 'similarity_score']],
            on='base_prompt',
            suffixes=('_1', '_2')
        )

        pred1 = (merged['similarity_score_1'] >= threshold).astype(int)
        pred2 = (merged['similarity_score_2'] >= threshold).astype(int)

        diff = pred1.mean() - pred2.mean()

        # Bootstrap for CI
        n_bootstrap = 10000
        diffs = []
        np.random.seed(42)

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(pred1), len(pred1), replace=True)
            boot_diff = pred1.iloc[indices].mean() - pred2.iloc[indices].mean()
            diffs.append(boot_diff)

        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)

        # Check significance
        significant = (ci_lower > 0) or (ci_upper < 0)

        comparisons.append({
            'Model1': model1,
            'Model2': model2,
            'Difference': diff,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Significant': significant
        })

# Sort by absolute difference
comparisons.sort(key=lambda x: abs(x['Difference']), reverse=True)

# Print key comparisons
print("\nKey Comparisons (sorted by |Δ|):")
print("-"*70)

for comp in comparisons[:10]:  # Top 10 largest differences
    sig = "Yes" if comp['Significant'] else "No"
    print(f"{comp['Model1']:20s} - {comp['Model2']:20s}: "
          f"Δ={comp['Difference']:+.4f} [{comp['CI_Lower']:+.4f}, {comp['CI_Upper']:+.4f}] "
          f"Sig: {sig}")

# Generate LaTeX table for paper
print("\n" + "="*70)
print("LaTeX Table for Appendix")
print("="*70)

print("""
\\subsection{Significance tests (accuracy deltas)}
\\label{app:significance}

Full pairwise comparisons between all six models with bootstrap 95\\% confidence intervals (10,000 iterations):

\\begin{table}[htbp]
\\centering
\\caption{Model comparisons (accuracy deltas, $\\Delta=A-B$, frozen $\\tau^\\star=0.66$).}
\\label{tab:significance}
\\begin{tabular}{lccc}
\\toprule
Comparison & $\\Delta$ Acc & 95\\% CI & Significant? \\\\
\\midrule""")

# Select key comparisons for the table
key_comparisons = [
    # Comparisons with best model (claude/kimi)
    ('claude-sonnet-4', 'gpt-4.1'),
    ('claude-sonnet-4', 'Qwen3-235B-A22B-FP8'),
    ('kimi-k2', 'gpt-4.1'),
    ('kimi-k2', 'Qwen3-235B-A22B-FP8'),
    # Between top models
    ('claude-sonnet-4', 'kimi-k2'),
    ('claude-sonnet-4', 'deepseek-v3.1'),
    # Other important comparisons
    ('gpt-4.1', 'Qwen3-235B-A22B-FP8'),
    ('deepseek-v3.1', 'gemini-2.5-flash'),
]

for model1, model2 in key_comparisons:
    # Find the comparison
    comp = None
    for c in comparisons:
        if (c['Model1'] == model1 and c['Model2'] == model2) or \
           (c['Model1'] == model2 and c['Model2'] == model1):
            comp = c
            # Adjust sign if needed
            if c['Model1'] == model2:
                comp = {
                    'Model1': model1,
                    'Model2': model2,
                    'Difference': -c['Difference'],
                    'CI_Lower': -c['CI_Upper'],
                    'CI_Upper': -c['CI_Lower'],
                    'Significant': c['Significant']
                }
            break

    if comp:
        sig = "Yes" if comp['Significant'] else "No"
        print(f"\\texttt{{{comp['Model1']}}} $-$ \\texttt{{{comp['Model2']}}} & "
              f"{comp['Difference']:+.3f} & [{comp['CI_Lower']:+.3f}, {comp['CI_Upper']:+.3f}] & {sig} \\\\")

print("""\\bottomrule
\\end{tabular}
\\end{table}

\\emph{Note:} Positive values indicate the first model outperforms the second.
All confidence intervals computed via bootstrap with 10,000 iterations.
""")

# Summary statistics
print("\n" + "="*70)
print("Summary Statistics")
print("="*70)

sig_count = sum(1 for c in comparisons if c['Significant'])
total_count = len(comparisons)

print(f"Total comparisons: {total_count}")
print(f"Significant differences: {sig_count}/{total_count} ({sig_count/total_count:.1%})")
print(f"Largest difference: {max(comparisons, key=lambda x: abs(x['Difference']))['Difference']:.4f}")
print(f"Smallest significant difference: {min([abs(c['Difference']) for c in comparisons if c['Significant']]):.4f}")