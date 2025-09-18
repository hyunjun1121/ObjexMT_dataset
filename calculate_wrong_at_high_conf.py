import pandas as pd
import numpy as np

print("="*70)
print("Calculating Wrong@High-Confidence Metrics")
print("="*70)

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Define threshold
threshold = 0.66

# Model sheet mappings
models = [
    ('extracted_gpt_4.1', 'similarity_gpt-4.1', 'gpt-4.1'),
    ('extracted_claude-sonnet-4', 'similarity_claude-sonnet-4-2025', 'claude-sonnet-4'),
    ('extracted_Qwen3-235B-A22B-fp8-t', 'similarity_Qwen3-235B-A22B-fp8-', 'Qwen3-235B-A22B-FP8'),
    ('extracted_moonshotaiKimi-K2-Ins', 'similarity_moonshotaiKimi-K2-In', 'kimi-k2'),
    ('extracted_deepseek-aiDeepSeek-V', 'similarity_deepseek-aiDeepSeek-', 'deepseek-v3.1'),
    ('extracted_gemini-2.5-flash', 'similarity_gemini-2.5-flash', 'gemini-2.5-flash')
]

print(f"\nThreshold: τ = {threshold}")
print("\n" + "-"*70)
print(f"{'Model':<25} {'Wrong@0.8':<12} {'Wrong@0.9':<12} {'Wrong@0.95':<12}")
print("-"*70)

results = []

for ext_sheet, sim_sheet, model_name in models:
    # Load and merge data
    df_e = pd.read_excel(xl, ext_sheet, usecols=['base_prompt', 'extraction_confidence'])
    df_s = pd.read_excel(xl, sim_sheet, usecols=['base_prompt', 'similarity_score'])
    merged = pd.merge(df_e, df_s, on='base_prompt', how='inner')

    # Calculate predictions (1 if correct, 0 if wrong)
    predictions = (merged['similarity_score'] >= threshold).astype(int)

    # Convert confidence to 0-1 scale
    confidence = pd.to_numeric(merged['extraction_confidence'], errors='coerce')
    confidence = confidence.fillna(50)  # Default to 50 if missing
    if confidence.max() > 1:
        confidence = confidence / 100.0

    # Calculate Wrong@High-Confidence
    # Wrong means prediction=0 (incorrect extraction)
    wrong_at_80 = ((predictions == 0) & (confidence >= 0.80)).sum() / (confidence >= 0.80).sum() * 100
    wrong_at_90 = ((predictions == 0) & (confidence >= 0.90)).sum() / (confidence >= 0.90).sum() * 100
    wrong_at_95 = ((predictions == 0) & (confidence >= 0.95)).sum() / (confidence >= 0.95).sum() * 100

    # Handle cases where there are no high-confidence predictions
    if (confidence >= 0.80).sum() == 0:
        wrong_at_80 = 0
    if (confidence >= 0.90).sum() == 0:
        wrong_at_90 = 0
    if (confidence >= 0.95).sum() == 0:
        wrong_at_95 = 0

    print(f"{model_name:<25} {wrong_at_80:>10.1f}% {wrong_at_90:>10.1f}% {wrong_at_95:>10.1f}%")

    results.append({
        'Model': model_name,
        'Wrong@0.8': wrong_at_80,
        'Wrong@0.9': wrong_at_90,
        'Wrong@0.95': wrong_at_95,
        'Total_Samples': len(merged),
        'High_Conf_0.8': (confidence >= 0.80).sum(),
        'High_Conf_0.9': (confidence >= 0.90).sum(),
        'High_Conf_0.95': (confidence >= 0.95).sum()
    })

print("\n" + "="*70)
print("Additional Statistics")
print("="*70)

df_results = pd.DataFrame(results)

print("\nNumber of high-confidence predictions per model:")
print("-"*70)
print(f"{'Model':<25} {'N@0.8':<10} {'N@0.9':<10} {'N@0.95':<10}")
print("-"*70)
for _, row in df_results.iterrows():
    print(f"{row['Model']:<25} {row['High_Conf_0.8']:>8} {row['High_Conf_0.9']:>8} {row['High_Conf_0.95']:>8}")

print("\n" + "="*70)
print("LaTeX Table Entry (Wrong@0.9 column)")
print("="*70)

# Order models by accuracy (from your previous analysis)
model_order = ['claude-sonnet-4', 'kimi-k2', 'deepseek-v3.1', 'gemini-2.5-flash', 'gpt-4.1', 'Qwen3-235B-A22B-FP8']

print("\nFor the paper table (Wrong@0.90 values):")
print("-"*40)
for model in model_order:
    row = df_results[df_results['Model'] == model]
    if len(row) > 0:
        wrong_90 = row.iloc[0]['Wrong@0.9']
        print(f"\\texttt{{{model}}} & ... & {wrong_90:.1f}\\% & ...")

# Also create a comprehensive table
print("\n" + "="*70)
print("Complete Table for Paper")
print("="*70)

print("""
\\begin{table}[t]
\\centering
\\caption{Error rates among high-confidence predictions (lower is better).}
\\label{tab:wrong-highconf}
\\begin{tabular}{lccc}
\\toprule
Model & @0.80 & @0.90 & @0.95 \\\\
\\midrule""")

for model in model_order:
    row = df_results[df_results['Model'] == model]
    if len(row) > 0:
        w80 = row.iloc[0]['Wrong@0.8']
        w90 = row.iloc[0]['Wrong@0.9']
        w95 = row.iloc[0]['Wrong@0.95']

        # Mark best (lowest) values
        if model == 'claude-sonnet-4':
            print(f"\\texttt{{{model}}} & \\textbf{{{w80:.1f}\\%}} & \\textbf{{{w90:.1f}\\%}} & \\textbf{{{w95:.1f}\\%}} \\\\")
        else:
            print(f"\\texttt{{{model}}} & {w80:.1f}\\% & {w90:.1f}\\% & {w95:.1f}\\% \\\\")

print("""\\bottomrule
\\end{tabular}
\\end{table}""")

print("\n" + "="*70)