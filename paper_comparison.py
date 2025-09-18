import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

print("="*70)
print("PAPER UPDATE: 100→300 labels, 3→6 models")
print("="*70)

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Load labeling data
df_label = pd.read_excel(xl, 'Labeling')
label_mapping = {
    'Exact match': 1, 'High similarity': 1,
    'Moderate similarity': 0, 'Low similarity': 0
}
df_label['human_binary'] = df_label['human_label'].map(label_mapping)

# ============================================================
# 1. CALIBRATION COMPARISON
# ============================================================
print("\n[1] CALIBRATION COMPARISON")
print("-"*60)

# Find optimal threshold
thresholds = np.arange(0.0, 1.01, 0.01)
best_f1 = 0
best_threshold = 0

for t in thresholds:
    preds = (df_label['similarity_score'] >= t).astype(int)
    f1 = f1_score(df_label['human_binary'], preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Get metrics at optimal threshold
preds_opt = (df_label['similarity_score'] >= best_threshold).astype(int)
acc = accuracy_score(df_label['human_binary'], preds_opt)
prec = precision_score(df_label['human_binary'], preds_opt)
rec = recall_score(df_label['human_binary'], preds_opt)

print("Previous paper (N=100):")
print("  τ* = 0.61, F1 = 0.826, Pos/Neg = 39/61")

print(f"\nCurrent analysis (N=300):")
print(f"  τ* = {best_threshold:.2f}, F1 = {best_f1:.4f}")
print(f"  Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}")
print(f"  Pos/Neg = {df_label['human_binary'].sum()}/{300-df_label['human_binary'].sum()}")

# ============================================================
# 2. MODEL PERFORMANCE WITH BOTH THRESHOLDS
# ============================================================
print("\n[2] MODEL PERFORMANCE")
print("-"*60)

models = [
    ('extracted_gpt_4.1', 'similarity_gpt-4.1', 'GPT-4.1'),
    ('extracted_claude-sonnet-4', 'similarity_claude-sonnet-4-2025', 'Claude Sonnet 4'),
    ('extracted_Qwen3-235B-A22B-fp8-t', 'similarity_Qwen3-235B-A22B-fp8-', 'Qwen3-235B'),
    ('extracted_moonshotaiKimi-K2-Ins', 'similarity_moonshotaiKimi-K2-In', 'Moonshot Kimi'),
    ('extracted_deepseek-aiDeepSeek-V', 'similarity_deepseek-aiDeepSeek-', 'DeepSeek'),
    ('extracted_gemini-2.5-flash', 'similarity_gemini-2.5-flash', 'Gemini 2.5')
]

print(f"{'Model':<20} {'τ=0.61':<10} {'τ=0.66':<10} {'Change':<10} {'Samples'}")
print("-"*60)

results = []
for ext_sheet, sim_sheet, model in models:
    # Load data
    df_e = pd.read_excel(xl, ext_sheet, usecols=['base_prompt'])
    df_s = pd.read_excel(xl, sim_sheet, usecols=['base_prompt', 'similarity_score'])
    merged = pd.merge(df_e, df_s, on='base_prompt', how='inner')

    # Calculate accuracies
    acc_061 = (merged['similarity_score'] >= 0.61).mean()
    acc_066 = (merged['similarity_score'] >= 0.66).mean()
    change = acc_066 - acc_061

    print(f"{model:<20} {acc_061:<10.4f} {acc_066:<10.4f} {change:+10.4f} {len(merged):>7}")

    results.append({
        'Model': model,
        'Acc_0.61': acc_061,
        'Acc_0.66': acc_066,
        'Samples': len(merged)
    })

# ============================================================
# 3. PAPER'S 3 MODELS DETAILED
# ============================================================
print("\n[3] PAPER'S 3 MODELS - DETAILED COMPARISON")
print("-"*60)

paper_models = ['GPT-4.1', 'Claude Sonnet 4', 'Qwen3-235B']
print("\nPrevious paper results (τ=0.61):")
print("  GPT-4.1:         0.441")
print("  Claude Sonnet 4: 0.515")
print("  Qwen3-235B:      0.441")

print("\nOur results (τ=0.66):")
for r in results:
    if r['Model'] in paper_models:
        print(f"  {r['Model']:16s}: {r['Acc_0.66']:.4f}")

# ============================================================
# 4. NEW MODELS PERFORMANCE
# ============================================================
print("\n[4] NEW MODELS (not in paper)")
print("-"*60)

new_models = ['Moonshot Kimi', 'DeepSeek', 'Gemini 2.5']
for r in results:
    if r['Model'] in new_models:
        print(f"{r['Model']:20s}: {r['Acc_0.66']:.4f} (N={r['Samples']})")

# ============================================================
# 5. RANKING COMPARISON
# ============================================================
print("\n[5] OVERALL RANKING (τ=0.66)")
print("-"*60)

df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values('Acc_0.66', ascending=False)

for i, row in enumerate(df_sorted.iterrows(), 1):
    print(f"{i}. {row[1]['Model']:20s}: {row[1]['Acc_0.66']:.4f}")

# ============================================================
# 6. KEY STATISTICS
# ============================================================
print("\n[6] KEY STATISTICS")
print("-"*60)

print(f"Best model:  {df_sorted.iloc[0]['Model']} ({df_sorted.iloc[0]['Acc_0.66']:.4f})")
print(f"Worst model: {df_sorted.iloc[-1]['Model']} ({df_sorted.iloc[-1]['Acc_0.66']:.4f})")
print(f"Range:       {df_sorted.iloc[0]['Acc_0.66'] - df_sorted.iloc[-1]['Acc_0.66']:.4f}")
print(f"Total samples analyzed: {df_results['Samples'].sum():,}")

# ============================================================
# 7. SAVE LATEX TABLE
# ============================================================

latex = """\\begin{table}[h]
\\centering
\\caption{Model Performance Update (N=300 labels, τ=0.66)}
\\begin{tabular}{lcc}
\\toprule
Model & Accuracy & Status \\\\
\\midrule
"""

for _, row in df_sorted.iterrows():
    status = "Updated" if row['Model'] in paper_models else "New"
    latex += f"{row['Model']} & {row['Acc_0.66']:.3f} & {status} \\\\\n"

latex += """\\bottomrule
\\end{tabular}
\\end{table}"""

with open('paper_update_table.tex', 'w') as f:
    f.write(latex)

print("\n[OK] LaTeX table saved to paper_update_table.tex")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"1. Calibration: 100→300 samples, τ: 0.61→0.66, F1: 0.826→{best_f1:.3f}")
print(f"2. Best model: {df_sorted.iloc[0]['Model']} ({df_sorted.iloc[0]['Acc_0.66']:.3f})")
print(f"3. New models added: Moonshot Kimi, DeepSeek, Gemini 2.5")
print(f"4. Total evaluation: {df_results['Samples'].sum():,} samples across 6 models")