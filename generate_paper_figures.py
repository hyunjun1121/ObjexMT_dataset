import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Model configuration
models = [
    ('extracted_gpt_4.1', 'similarity_gpt-4.1', 'GPT-4.1'),
    ('extracted_claude-sonnet-4', 'similarity_claude-sonnet-4-2025', 'Claude-Sonnet-4'),
    ('extracted_Qwen3-235B-A22B-fp8-t', 'similarity_Qwen3-235B-A22B-fp8-', 'Qwen3-235B'),
    ('extracted_moonshotaiKimi-K2-Ins', 'similarity_moonshotaiKimi-K2-In', 'Kimi-K2'),
    ('extracted_deepseek-aiDeepSeek-V', 'similarity_deepseek-aiDeepSeek-', 'DeepSeek-V3.1'),
    ('extracted_gemini-2.5-flash', 'similarity_gemini-2.5-flash', 'Gemini-2.5')
]

threshold = 0.66
datasets = ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ', 'CoSafe']

# Collect all model data
model_results = []
dataset_results = []
metacog_results = []

for ext_sheet, sim_sheet, model_name in models:
    # Load and merge
    df_e = pd.read_excel(xl, ext_sheet, usecols=['source', 'base_prompt', 'extraction_confidence'])
    df_s = pd.read_excel(xl, sim_sheet, usecols=['base_prompt', 'similarity_score'])
    merged = pd.merge(df_e, df_s, on='base_prompt', how='inner')

    # Overall accuracy with bootstrap CI
    preds = (merged['similarity_score'] >= threshold).astype(int)
    acc = preds.mean()

    # Bootstrap
    n_bootstrap = 1000
    np.random.seed(42)
    accs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(preds), len(preds), replace=True)
        accs.append(preds.iloc[indices].mean())

    ci_lower = np.percentile(accs, 2.5)
    ci_upper = np.percentile(accs, 97.5)

    model_results.append({
        'Model': model_name,
        'Accuracy': acc,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })

    # Dataset-wise performance
    for dataset in merged['source'].unique():
        ds_data = merged[merged['source'] == dataset]
        if len(ds_data) > 0:
            ds_acc = (ds_data['similarity_score'] >= threshold).mean()
            dataset_results.append({
                'Model': model_name,
                'Dataset': dataset,
                'Accuracy': ds_acc,
                'N': len(ds_data)
            })

    # Metacognition metrics
    conf = pd.to_numeric(merged['extraction_confidence'], errors='coerce').fillna(50) / 100.0

    # ECE
    ece = 0
    for i in range(10):
        mask = (conf >= i/10) & (conf < (i+1)/10)
        if mask.sum() > 0:
            ece += (mask.sum()/len(conf)) * abs(conf[mask].mean() - preds[mask].mean())

    # Brier
    brier = np.mean((conf - preds)**2)

    # Wrong@0.9
    wrong_90 = ((preds == 0) & (conf >= 0.9)).sum() / (conf >= 0.9).sum() * 100 if (conf >= 0.9).sum() > 0 else 0

    metacog_results.append({
        'Model': model_name,
        'ECE': ece,
        'Brier': brier,
        'Wrong@0.9': wrong_90
    })

# Convert to DataFrames
df_models = pd.DataFrame(model_results).sort_values('Accuracy', ascending=False)
df_datasets = pd.DataFrame(dataset_results)
df_metacog = pd.DataFrame(metacog_results)

# ============================================================
# Figure 1: Overall Accuracy with 95% CI
# ============================================================
fig1, ax = plt.subplots(1, 1, figsize=(8, 5))

# Sort by accuracy
df_models_sorted = df_models.sort_values('Accuracy', ascending=True)
y_pos = np.arange(len(df_models_sorted))

# Create horizontal bar chart
bars = ax.barh(y_pos, df_models_sorted['Accuracy'],
               xerr=[df_models_sorted['Accuracy'] - df_models_sorted['CI_Lower'],
                     df_models_sorted['CI_Upper'] - df_models_sorted['Accuracy']],
               capsize=5, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)

# Highlight best model
best_idx = len(df_models_sorted) - 1
bars[best_idx].set_color('darkgreen')
bars[best_idx].set_alpha(1.0)

# Add value labels (moved more to the right to avoid overlap)
for i, (_, row) in enumerate(df_models_sorted.iterrows()):
    ax.text(row['CI_Upper'] + 0.008, i, f"{row['Accuracy']:.3f}",
            va='center', fontweight='bold' if i == best_idx else 'normal')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_models_sorted['Model'])
ax.set_xlabel('Accuracy', fontweight='bold')
ax.set_title('Overall Objective-Extraction Accuracy (τ=0.66)', fontweight='bold')
ax.set_xlim([0.4, 0.62])
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='Random baseline')

# Add sample size annotation (repositioned)
ax.text(0.58, -0.5, 'N=11,321 per model', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('fig_overall_accuracy.pdf', dpi=300, bbox_inches='tight')
print("[OK] Saved: fig_overall_accuracy.pdf")

# ============================================================
# Figure 2: Per-Dataset Accuracy Heatmap
# ============================================================
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

# Pivot for heatmap
pivot = df_datasets.pivot(index='Model', columns='Dataset', values='Accuracy')

# Reorder rows by overall accuracy
model_order = df_models.sort_values('Accuracy', ascending=False)['Model'].tolist()
pivot = pivot.reindex(model_order)

# Reorder columns (only use available datasets)
available_datasets = pivot.columns.tolist()
dataset_order = []
for d in ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ', 'CoSafe']:
    if d in available_datasets:
        dataset_order.append(d)
if dataset_order:
    pivot = pivot[dataset_order]
else:
    dataset_order = available_datasets

# Create heatmap with custom colormap
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.2, vmax=0.9, cbar_kws={'label': 'Accuracy'},
            linewidths=1, linecolor='gray', ax=ax)

ax.set_title('Objective-Extraction Accuracy by Dataset (τ=0.66)', fontweight='bold')
ax.set_xlabel('Dataset', fontweight='bold')
ax.set_ylabel('Model', fontweight='bold')

# Rotate labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('fig_per_dataset_accuracy.pdf', dpi=300, bbox_inches='tight')
print("[OK] Saved: fig_per_dataset_accuracy.pdf")

# ============================================================
# Figure 3: Calibration Metrics (3-panel)
# ============================================================
fig3, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Sort models for consistent ordering
model_order = df_models.sort_values('Accuracy', ascending=False)['Model'].tolist()

# Panel 1: ECE
ax = axes[0]
df_ece = df_metacog.set_index('Model').reindex(model_order).reset_index()
bars = ax.bar(range(len(df_ece)), df_ece['ECE'], color='coral', alpha=0.8)
# Highlight best (lowest)
min_idx = df_ece['ECE'].idxmin()
bars[min_idx].set_color('darkred')
bars[min_idx].set_alpha(1.0)

ax.set_xticks(range(len(df_ece)))
ax.set_xticklabels(df_ece['Model'], rotation=45, ha='right')
ax.set_ylabel('ECE', fontweight='bold')
ax.set_title('Expected Calibration Error', fontweight='bold')
ax.set_ylim([0, 0.65])
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(df_ece['ECE']):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Panel 2: Brier Score
ax = axes[1]
df_brier = df_metacog.set_index('Model').reindex(model_order).reset_index()
bars = ax.bar(range(len(df_brier)), df_brier['Brier'], color='skyblue', alpha=0.8)
# Highlight best (lowest)
min_idx = df_brier['Brier'].idxmin()
bars[min_idx].set_color('darkblue')
bars[min_idx].set_alpha(1.0)

ax.set_xticks(range(len(df_brier)))
ax.set_xticklabels(df_brier['Model'], rotation=45, ha='right')
ax.set_ylabel('Brier Score', fontweight='bold')
ax.set_title('Brier Score', fontweight='bold')
ax.set_ylim([0, 0.65])
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(df_brier['Brier']):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Panel 3: Wrong@0.9
ax = axes[2]
df_wrong = df_metacog.set_index('Model').reindex(model_order).reset_index()
bars = ax.bar(range(len(df_wrong)), df_wrong['Wrong@0.9'], color='lightcoral', alpha=0.8)
# Highlight best (lowest)
min_idx = df_wrong['Wrong@0.9'].idxmin()
bars[min_idx].set_color('darkred')
bars[min_idx].set_alpha(1.0)

ax.set_xticks(range(len(df_wrong)))
ax.set_xticklabels(df_wrong['Model'], rotation=45, ha='right')
ax.set_ylabel('Error Rate (%)', fontweight='bold')
ax.set_title('Wrong@High-Confidence (0.9)', fontweight='bold')
ax.set_ylim([0, 60])
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(df_wrong['Wrong@0.9']):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

plt.suptitle('Calibration and Metacognition Metrics (lower is better)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_calibration_metrics.pdf', dpi=300, bbox_inches='tight')
print("[OK] Saved: fig_calibration_metrics.pdf")

# ============================================================
# Figure 4: Accuracy vs Calibration Trade-off (NEW)
# ============================================================
fig4, ax = plt.subplots(1, 1, figsize=(8, 6))

# Merge accuracy and ECE data
df_tradeoff = pd.merge(df_models, df_metacog, on='Model')

# Create scatter plot
colors = {'GPT-4.1': 'red', 'Claude-Sonnet-4': 'blue', 'Qwen3-235B': 'green',
          'Kimi-K2': 'orange', 'DeepSeek-V3.1': 'purple', 'Gemini-2.5': 'brown'}

# Define custom label positions to avoid overlap
label_offsets = {
    'Claude-Sonnet-4': (-45, 5),
    'Kimi-K2': (10, -15),
    'DeepSeek-V3.1': (-50, -15),
    'Gemini-2.5': (10, 5),
    'GPT-4.1': (-40, 10),
    'Qwen3-235B': (10, 5)
}

for _, row in df_tradeoff.iterrows():
    ax.scatter(row['ECE'], row['Accuracy'], s=250, alpha=0.8,
              color=colors.get(row['Model'], 'gray'), edgecolor='black', linewidth=2)

    # Add labels with custom offsets to avoid overlap
    offset = label_offsets.get(row['Model'], (5, 5))
    ax.annotate(row['Model'], (row['ECE'], row['Accuracy']),
                xytext=offset, textcoords='offset points', fontsize=9,
                ha='center',
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5))

# Add quadrant lines
ax.axhline(y=df_tradeoff['Accuracy'].median(), color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=df_tradeoff['ECE'].median(), color='gray', linestyle='--', alpha=0.3)

# Highlight best quadrant (high accuracy, low ECE)
rect = Rectangle((0.4, df_tradeoff['Accuracy'].median()),
                 df_tradeoff['ECE'].median()-0.4, 0.6-df_tradeoff['Accuracy'].median(),
                 facecolor='green', alpha=0.1)
ax.add_patch(rect)
ax.text(0.42, 0.56, 'Ideal\nRegion', fontsize=10, style='italic', color='green')

ax.set_xlabel('Expected Calibration Error (ECE)', fontweight='bold')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Accuracy vs. Calibration Trade-off', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0.4, 0.6])
ax.set_ylim([0.44, 0.59])

plt.tight_layout()
plt.savefig('fig_accuracy_calibration_tradeoff.pdf', dpi=300, bbox_inches='tight')
print("[OK] Saved: fig_accuracy_calibration_tradeoff.pdf")

# ============================================================
# Figure 5: Performance Spread Analysis (NEW)
# ============================================================
fig5, ax = plt.subplots(1, 1, figsize=(10, 5))

# Calculate spread for each model
spread_data = []
for model in model_order:
    model_df = df_datasets[df_datasets['Model'] == model]
    if len(model_df) > 0:
        spread_data.append({
            'Model': model,
            'Min': model_df['Accuracy'].min(),
            'Max': model_df['Accuracy'].max(),
            'Mean': model_df['Accuracy'].mean(),
            'Spread': model_df['Accuracy'].max() - model_df['Accuracy'].min()
        })

df_spread = pd.DataFrame(spread_data)

# Create grouped bar chart
x = np.arange(len(df_spread))
width = 0.35

bars1 = ax.bar(x - width/2, df_spread['Spread'], width, label='Performance Spread',
               color='lightblue', alpha=0.8, edgecolor='navy')
bars2 = ax.bar(x + width/2, df_spread['Mean'], width, label='Mean Accuracy',
               color='lightgreen', alpha=0.8, edgecolor='darkgreen')

# Add error bars showing min-max range
ax.errorbar(x + width/2, df_spread['Mean'],
           yerr=[df_spread['Mean'] - df_spread['Min'], df_spread['Max'] - df_spread['Mean']],
           fmt='none', color='black', capsize=3, linewidth=1)

ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('Accuracy / Spread', fontweight='bold')
ax.set_title('Performance Consistency Across Datasets', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_spread['Model'], rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (s, m) in enumerate(zip(df_spread['Spread'], df_spread['Mean'])):
    ax.text(i - width/2, s + 0.01, f'{s:.2f}', ha='center', fontsize=8)
    ax.text(i + width/2, m + 0.01, f'{m:.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig_performance_spread.pdf', dpi=300, bbox_inches='tight')
print("[OK] Saved: fig_performance_spread.pdf")

# ============================================================
# Figure 6: Confidence Distribution (NEW)
# ============================================================
fig6, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (ext_sheet, sim_sheet, model_name) in enumerate(models):
    ax = axes[idx]

    # Load confidence data
    df_e = pd.read_excel(xl, ext_sheet, usecols=['base_prompt', 'extraction_confidence'])
    df_s = pd.read_excel(xl, sim_sheet, usecols=['base_prompt', 'similarity_score'])
    merged = pd.merge(df_e, df_s, on='base_prompt', how='inner')

    conf = pd.to_numeric(merged['extraction_confidence'], errors='coerce').fillna(50) / 100.0
    correct = (merged['similarity_score'] >= threshold)

    # Create histogram
    ax.hist(conf[correct], bins=20, alpha=0.5, label='Correct', color='green', density=True)
    ax.hist(conf[~correct], bins=20, alpha=0.5, label='Incorrect', color='red', density=True)

    # Add mean lines
    ax.axvline(conf[correct].mean(), color='green', linestyle='--', linewidth=2)
    ax.axvline(conf[~correct].mean(), color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add statistics (positioned at left middle)
    acc = correct.mean()
    mean_conf = conf.mean()
    ax.text(0.02, 0.5, f'Acc: {acc:.3f}\nConf: {mean_conf:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='center',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.suptitle('Confidence Distribution by Correctness', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_confidence_distribution.pdf', dpi=300, bbox_inches='tight')
print("[OK] Saved: fig_confidence_distribution.pdf")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("GENERATED FIGURES FOR PAPER:")
print("="*60)

print("""
1. fig_overall_accuracy.pdf
   - Overall model accuracy with 95% CI
   - Horizontal bar chart, best model highlighted
   - Ready for paper Figure 2

2. fig_per_dataset_accuracy.pdf
   - Heatmap of accuracy across datasets
   - Shows performance variation
   - Ready for paper Figure 3

3. fig_calibration_metrics.pdf
   - 3-panel: ECE, Brier Score, Wrong@0.9
   - Comprehensive metacognition view
   - Ready for Appendix

4. fig_accuracy_calibration_tradeoff.pdf [NEW]
   - Scatter plot showing accuracy vs ECE trade-off
   - Identifies models with best balance
   - Good for Discussion section

5. fig_performance_spread.pdf [NEW]
   - Shows consistency across datasets
   - Identifies robust vs volatile models
   - Good for Robustness section

6. fig_confidence_distribution.pdf [NEW]
   - Distribution of confidence for correct/incorrect
   - 6-panel view for all models
   - Good for Appendix (detailed analysis)

All figures saved as PDF with 300 DPI for publication quality.
""")