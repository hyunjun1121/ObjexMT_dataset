import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" "*15 + "OBJEX Dataset Complete Analysis - All Models")
print("="*70)

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl_file = pd.ExcelFile(excel_path)

# ============================================================
# 1. Human-aligned thresholding
# ============================================================
print("\n1. HUMAN-ALIGNED THRESHOLDING")
print("="*60)

df_label = pd.read_excel(xl_file, 'Labeling')

label_mapping = {
    'Exact match': 1,
    'High similarity': 1,
    'Moderate similarity': 0,
    'Low similarity': 0
}

df_label['human_binary'] = df_label['human_label'].map(label_mapping)

# Find optimal threshold
thresholds = np.arange(0.0, 1.01, 0.01)
best_f1 = 0
best_threshold = 0
best_metrics = {}

for threshold in thresholds:
    predictions = (df_label['similarity_score'] >= threshold).astype(int)
    f1 = f1_score(df_label['human_binary'], predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = {
            'f1': f1,
            'accuracy': accuracy_score(df_label['human_binary'], predictions),
            'precision': precision_score(df_label['human_binary'], predictions),
            'recall': recall_score(df_label['human_binary'], predictions)
        }

print(f"Optimal threshold: {best_threshold:.2f}")
print(f"F1 Score: {best_metrics['f1']:.4f}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")

# ============================================================
# 2. Model performance analysis
# ============================================================
print("\n2. MODEL PERFORMANCE ANALYSIS")
print("="*60)

# Define model mappings - exact sheet names
model_sheets = [
    ('extracted_gpt_4.1', 'similarity_gpt-4.1', 'GPT-4.1'),
    ('extracted_claude-sonnet-4', 'similarity_claude-sonnet-4-2025', 'Claude Sonnet 4'),
    ('extracted_Qwen3-235B-A22B-fp8-t', 'similarity_Qwen3-235B-A22B-fp8-', 'Qwen3-235B'),
    ('extracted_moonshotaiKimi-K2-Ins', 'similarity_moonshotaiKimi-K2-In', 'Moonshot Kimi'),
    ('extracted_deepseek-aiDeepSeek-V', 'similarity_deepseek-aiDeepSeek-', 'DeepSeek'),
    ('extracted_gemini-2.5-flash', 'similarity_gemini-2.5-flash', 'Gemini 2.5 Flash')
]

results = []
model_data = {}  # Store merged data for later use

for extract_sheet, sim_sheet, model_name in model_sheets:
    try:
        # Load data
        df_extract = pd.read_excel(xl_file, extract_sheet)
        df_sim = pd.read_excel(xl_file, sim_sheet)

        # Merge data
        merged = pd.merge(
            df_extract[['source', 'base_prompt', 'extraction_confidence']],
            df_sim[['base_prompt', 'similarity_score']],
            on='base_prompt',
            how='inner'
        )

        # Store for later use
        model_data[model_name] = merged

        # Calculate accuracy
        predictions = (merged['similarity_score'] >= best_threshold).astype(int)
        accuracy = predictions.mean()

        # Bootstrap for confidence interval
        n_bootstrap = 1000
        accuracies = []
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(predictions), len(predictions), replace=True)
            boot_acc = predictions.iloc[indices].mean()
            accuracies.append(boot_acc)

        ci_lower = np.percentile(accuracies, 2.5)
        ci_upper = np.percentile(accuracies, 97.5)

        # Calculate confidence metrics
        confidence = pd.to_numeric(merged['extraction_confidence'], errors='coerce')
        confidence = confidence.fillna(50) / 100.0

        # ECE calculation
        n_bins = 10
        ece = 0
        for i in range(n_bins):
            bin_min = i / n_bins
            bin_max = (i + 1) / n_bins
            mask = (confidence >= bin_min) & (confidence < bin_max)
            if i == n_bins - 1:
                mask = (confidence >= bin_min) & (confidence <= bin_max)

            if mask.sum() > 0:
                bin_conf = confidence[mask].mean()
                bin_acc = predictions[mask].mean()
                bin_weight = mask.sum() / len(predictions)
                ece += bin_weight * abs(bin_conf - bin_acc)

        # Brier score
        brier = np.mean((confidence - predictions)**2)

        # Wrong at high confidence
        wrong_80 = ((predictions == 0) & (confidence >= 0.8)).mean()
        wrong_90 = ((predictions == 0) & (confidence >= 0.9)).mean()
        wrong_95 = ((predictions == 0) & (confidence >= 0.95)).mean()

        # AURC
        sorted_indices = np.argsort(-confidence)
        sorted_preds = predictions.iloc[sorted_indices].values
        coverages = np.arange(1, len(predictions) + 1) / len(predictions)
        risks = 1 - np.cumsum(sorted_preds) / np.arange(1, len(predictions) + 1)
        aurc = np.trapz(risks, coverages)

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Samples': len(merged),
            'ECE': ece,
            'Brier': brier,
            'Wrong@0.8': wrong_80,
            'Wrong@0.9': wrong_90,
            'Wrong@0.95': wrong_95,
            'AURC': aurc
        })

        print(f"{model_name:20s}: Acc={accuracy:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], N={len(merged):5d}")

    except Exception as e:
        print(f"Error processing {model_name}: {e}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "-"*60)
print("TOP MODELS BY ACCURACY:")
print(results_df[['Model', 'Accuracy', 'Samples']].head())

print("\n" + "-"*60)
print("BEST CALIBRATED MODELS (lowest ECE):")
print(results_df.nsmallest(3, 'ECE')[['Model', 'ECE', 'Brier']])

# ============================================================
# 3. Dataset-wise performance
# ============================================================
print("\n3. DATASET-WISE PERFORMANCE")
print("="*60)

datasets = ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ', 'CoSafe']
dataset_results = []

for model_name, merged in model_data.items():
    for dataset in datasets:
        dataset_data = merged[merged['source'] == dataset]
        if len(dataset_data) > 0:
            acc = (dataset_data['similarity_score'] >= best_threshold).mean()
            dataset_results.append({
                'Model': model_name,
                'Dataset': dataset,
                'Accuracy': acc,
                'Samples': len(dataset_data)
            })

dataset_df = pd.DataFrame(dataset_results)

# Print performance spread
for model in results_df['Model']:
    model_perf = dataset_df[dataset_df['Model'] == model]
    if len(model_perf) > 0:
        spread = model_perf['Accuracy'].max() - model_perf['Accuracy'].min()
        print(f"{model:20s}: Spread = {spread:.4f} (Max={model_perf['Accuracy'].max():.4f}, Min={model_perf['Accuracy'].min():.4f})")

# ============================================================
# 4. Pairwise comparisons
# ============================================================
print("\n4. PAIRWISE MODEL COMPARISONS")
print("="*60)

pairwise_results = []
models_list = list(model_data.keys())

for i, model1 in enumerate(models_list):
    for model2 in models_list[i+1:]:
        data1 = model_data[model1]
        data2 = model_data[model2]

        # Align by base_prompt
        merged_comp = pd.merge(
            data1[['base_prompt', 'similarity_score']],
            data2[['base_prompt', 'similarity_score']],
            on='base_prompt',
            suffixes=('_1', '_2')
        )

        pred1 = (merged_comp['similarity_score_1'] >= best_threshold).astype(int)
        pred2 = (merged_comp['similarity_score_2'] >= best_threshold).astype(int)

        diff = pred1.mean() - pred2.mean()

        # Bootstrap test
        n_bootstrap = 1000
        diffs = []
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(pred1), len(pred1), replace=True)
            boot_diff = pred1.iloc[indices].mean() - pred2.iloc[indices].mean()
            diffs.append(boot_diff)

        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        significant = (ci_lower > 0) or (ci_upper < 0)

        if significant:
            pairwise_results.append({
                'Model1': model1,
                'Model2': model2,
                'Difference': diff,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper
            })

if pairwise_results:
    print("Significant differences (p < 0.05):")
    for result in pairwise_results:
        print(f"  {result['Model1']} vs {result['Model2']}: "
              f"Δ = {result['Difference']:.4f} [{result['CI_Lower']:.4f}, {result['CI_Upper']:.4f}]")
else:
    print("No significant differences found between models")

# ============================================================
# 5. Visualizations
# ============================================================
print("\n5. GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Model accuracy comparison
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df)))
bars = ax.bar(range(len(results_df)), results_df['Accuracy'], color=colors)
ax.errorbar(range(len(results_df)), results_df['Accuracy'],
           yerr=[results_df['Accuracy'] - results_df['CI_Lower'],
                 results_df['CI_Upper'] - results_df['Accuracy']],
           fmt='none', color='black', capsize=5)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance (95% CI)')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

# 2. Dataset heatmap
ax = axes[0, 1]
pivot = dataset_df.pivot(index='Model', columns='Dataset', values='Accuracy')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
ax.set_title('Performance by Dataset')

# 3. ECE comparison
ax = axes[0, 2]
results_df_sorted = results_df.sort_values('ECE')
ax.barh(range(len(results_df_sorted)), results_df_sorted['ECE'])
ax.set_yticks(range(len(results_df_sorted)))
ax.set_yticklabels(results_df_sorted['Model'])
ax.set_xlabel('Expected Calibration Error')
ax.set_title('Calibration Quality (lower is better)')
ax.grid(True, alpha=0.3)

# 4. Wrong at high confidence
ax = axes[1, 0]
x = np.arange(len(results_df))
width = 0.25
ax.bar(x - width, results_df['Wrong@0.8'], width, label='@0.8')
ax.bar(x, results_df['Wrong@0.9'], width, label='@0.9')
ax.bar(x + width, results_df['Wrong@0.95'], width, label='@0.95')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.set_ylabel('Error Rate')
ax.set_title('Errors at High Confidence')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Brier score
ax = axes[1, 1]
results_df_brier = results_df.sort_values('Brier')
ax.barh(range(len(results_df_brier)), results_df_brier['Brier'])
ax.set_yticks(range(len(results_df_brier)))
ax.set_yticklabels(results_df_brier['Model'])
ax.set_xlabel('Brier Score')
ax.set_title('Prediction Reliability (lower is better)')
ax.grid(True, alpha=0.3)

# 6. AURC comparison
ax = axes[1, 2]
results_df_aurc = results_df.sort_values('AURC')
ax.barh(range(len(results_df_aurc)), results_df_aurc['AURC'])
ax.set_yticks(range(len(results_df_aurc)))
ax.set_yticklabels(results_df_aurc['Model'])
ax.set_xlabel('AURC')
ax.set_title('Risk-Coverage Trade-off (lower is better)')
ax.grid(True, alpha=0.3)

plt.suptitle('OBJEX Dataset Analysis Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('complete_analysis_results.png', dpi=300, bbox_inches='tight')
print("[OK] Visualizations saved to complete_analysis_results.png")

# ============================================================
# 6. Save results
# ============================================================
print("\n6. SAVING RESULTS")
print("="*60)

# Save to Excel
with pd.ExcelWriter('complete_analysis_results.xlsx', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Model_Performance', index=False)
    dataset_df.to_excel(writer, sheet_name='Dataset_Performance', index=False)

    # Add threshold info
    threshold_df = pd.DataFrame([{
        'Optimal_Threshold': best_threshold,
        'F1_Score': best_f1,
        'Accuracy': best_metrics['accuracy'],
        'Precision': best_metrics['precision'],
        'Recall': best_metrics['recall'],
        'Calibration_Samples': len(df_label)
    }])
    threshold_df.to_excel(writer, sheet_name='Threshold', index=False)

print("[OK] Results saved to complete_analysis_results.xlsx")

# Generate LaTeX table
latex_lines = []
latex_lines.append("\\begin{table}[h]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Model Performance on OBJEX Dataset}")
latex_lines.append("\\begin{tabular}{lccc}")
latex_lines.append("\\toprule")
latex_lines.append("Model & Accuracy & ECE & Brier \\\\")
latex_lines.append("\\midrule")

for _, row in results_df.iterrows():
    latex_lines.append(f"{row['Model']} & {row['Accuracy']:.3f} & {row['ECE']:.3f} & {row['Brier']:.3f} \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

with open('complete_latex_table.tex', 'w') as f:
    f.write('\n'.join(latex_lines))

print("[OK] LaTeX table saved to complete_latex_table.tex")

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "="*70)
print(" "*25 + "ANALYSIS SUMMARY")
print("="*70)

print(f"\nKey Results:")
print(f"  * Optimal Threshold: {best_threshold:.2f}")
print(f"  * Best Model: {results_df.iloc[0]['Model']}")
print(f"  * Best Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"  * Total Models Analyzed: {len(results_df)}")
print(f"  * Total Samples: {results_df['Samples'].sum():,}")

print(f"\nTop 3 Models by Accuracy:")
for i, row in enumerate(results_df.head(3).itertuples()):
    print(f"  {i+1}. {row.Model}: {row.Accuracy:.4f}")

print(f"\nBest Calibrated Models (lowest ECE):")
for row in results_df.nsmallest(3, 'ECE').itertuples():
    print(f"  - {row.Model}: ECE = {row.ECE:.4f}")

print("\n" + "="*70)
print(" "*20 + "ANALYSIS COMPLETE!")
print("="*70)